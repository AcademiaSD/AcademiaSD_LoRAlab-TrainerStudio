# -*- coding: utf-8 -*-
"""
slider_prompts.py — Escribe los prompts del generador de sliders con Qwen3-VL-8B
Writes the slider generator prompts with Qwen3-VL-8B

El usuario describe el slider en cualquier idioma (efecto, extremo +5 y, opcional, extremo -5) y
Qwen3-VL-8B (el captioner compartido, ver 0_caption_klein9b.py) rellena con una plantilla fija, en
inglés, los campos detallados del generador: tema, posición 0, instrucciones de edición de -100,
-50, +50 y +100, y si conviene variar personas y pelo. Se guardan en settings/slider_gen_settings.json
y la GUI los muestra editables antes de generar.

The user describes the slider in any language and Qwen3-VL-8B fills in, in English, the detailed
fields of the generator, saved to settings/slider_gen_settings.json and editable in the GUI.
"""
import importlib.util
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "settings/slider_gen_settings.json"
CAPTIONER_DIR = "Captioner-Qwen3-VL-8B"
POSITIONS = ("-100", "-50", "50", "100")

TEMPLATE = """You write the prompts of a synthetic dataset for a "slider" LoRA: one LoRA whose strength is a dial \
along one visual concept. The dataset has base images (position 0) and, for each one, edited versions at \
positions -100, -50, +50 and +100, made by an image-editing model that receives one instruction per position.

The user describes the slider, possibly in another language:
EFFECT: {effect}
MAXIMUM (+5, position +100): {max_text}
MINIMUM (-5, position -100): {min_text}

Answer ONLY with this JSON, every text in English:
{{"subject": "...", "neutral": "...", "edits": {{"-100": "...", "-50": "...", "50": "...", "100": "..."}}, \
"vary_people": true, "vary_hair": true}}

Rules:
- "subject": what the base images show, as the start of an image prompt, WITHOUT the concept or its ends. \
Examples: "Head and shoulders photo portrait", "Close-up photo of a human hand, palm facing the camera, fingers spread".
- "neutral": the state at position 0, a short comma-separated description added after the subject. If both ends \
are given, it is the middle state, so there is room to change in both directions (for hair length: "medium-length \
hair"). If the MINIMUM is empty or describes the normal, ordinary state of the subject, position 0 IS that state: \
describe it in "neutral" and leave "-100" and "-50" empty ("").
- "edits": imperative instructions that turn the position-0 image into each position, at most 40 words each. \
"100" reaches the MAXIMUM and "-100" the MINIMUM. "50" and "-50" are exactly halfway: describe the halfway state \
with concrete, visible features (lengths, sizes, shapes, colors, amounts), never only with words like "slightly". \
Name only what changes; the rest of the image is kept by the editing model.
- "vary_people": true if the subject is a person or part of a person's body (the base images then show people of \
different age, sex and origin); false otherwise.
- "vary_hair": false if the effect is about hair or the subject does not show hair; true otherwise.

Example. EFFECT: facial expression. MAXIMUM: very happy. MINIMUM: very sad.
{{"subject": "Head and shoulders photo portrait", "neutral": "neutral relaxed expression, mouth closed, looking at \
the camera", "edits": {{"-100": "Make the person look very sad, close to crying: inner eyebrows raised, eyes watery \
and downcast, mouth corners pulled down", "-50": "Make the person look a little sad: mouth corners slightly lowered, \
inner eyebrows a bit raised, eyes looking down", "50": "Make the person smile with a soft closed-mouth smile and \
slightly narrowed eyes", "100": "Make the person laugh, with a broad smile showing the teeth and crinkled eyes"}}, \
"vary_people": true, "vary_hair": true}}"""


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_answer(text):
    """El JSON de la respuesta, validado; lanza ValueError si no sirve."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in the answer / no hay JSON en la respuesta")
    data = json.loads(m.group(0))
    edits = data.get("edits") or {}
    out = {
        "subject": str(data.get("subject", "")).strip(),
        "neutral": str(data.get("neutral", "")).strip(),
        # El modelo a veces escribe "+50" en lugar de "50".
        "edits": {p: str(edits.get(p) or edits.get("+" + p) or "").strip() for p in POSITIONS},
        "vary_people": bool(data.get("vary_people", True)),
        "vary_hair": bool(data.get("vary_hair", True)),
    }
    if not out["subject"] or not (out["edits"]["100"] or out["edits"]["-100"]):
        raise ValueError("incomplete answer / respuesta incompleta")
    return out


def main():
    cfg = read_json(CONFIG_PATH)
    effect, max_text, min_text = (str(cfg.get(k, "")).strip() for k in ("effect", "max_text", "min_text"))
    if not effect or not max_text:
        print("[!] Write the effect and the maximum (+5) / Escribe el efecto y el máximo (+5)")
        return 1
    print(f"  Effect / Efecto      : {effect}")
    print(f"  Maximum / Máximo (+5): {max_text}")
    print(f"  Minimum / Mínimo (-5): {min_text or '(none: one-sided slider / ninguno: slider de un solo lado)'}", flush=True)

    # El cargador del captioner (descarga incluida) es el del auto-caption.
    spec = importlib.util.spec_from_file_location("captioner", os.path.join(SCRIPTS_DIR, "0_caption_klein9b.py"))
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    import torch
    model, processor = cap.load_captioner(CAPTIONER_DIR)

    prompt = TEMPLATE.format(effect=effect, max_text=max_text, min_text=min_text or "(empty)")
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    result = None
    for attempt in range(2):
        print("Writing the prompts... / Escribiendo los prompts...", flush=True)
        with torch.inference_mode():
            # El segundo intento muestrea: si el primero no dio un JSON válido, repetirlo igual no serviría.
            out = model.generate(**inputs, max_new_tokens=700, do_sample=attempt > 0, temperature=0.7 if attempt else None)
        answer = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            result = parse_answer(answer)
            break
        except ValueError as e:
            print(f"[!] {e}\n{answer}", flush=True)
    if result is None:
        print("[!] The model did not give valid prompts: try again or write them by hand.")
        print("[!] El modelo no dio prompts válidos: vuelve a intentarlo o escríbelos a mano.")
        return 1

    cfg = read_json(CONFIG_PATH)
    cfg.update(result)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CONFIG_PATH)

    print(f"\n  Subject / Tema    : {result['subject']}")
    print(f"  Position 0        : {result['neutral']}")
    for p in POSITIONS:
        print(f"  {('+' if not p.startswith('-') else '') + p:>5}             : {result['edits'][p] or '-'}")
    print(f"  Vary people / hair: {result['vary_people']} / {result['vary_hair']}")
    print("\n✓ Prompts written: review them and press Generate Pairs / Prompts escritos: revísalos y pulsa Generar Pares")
    return 0


if __name__ == "__main__":
    sys.exit(main())
