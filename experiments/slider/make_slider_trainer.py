# Prototipo de slider: copia del entrenador de Klein.
#   nombre_pos -> LoRA a +PUSH, nombre_neg -> -PUSH, nombre_anc (anchor, antes = después) -> ±PUSH al azar.
#   SLIDER_PUSH (3 por defecto): fuerza a la que se entrena; en ComfyUI la fuerza PUSH da el ejemplo del dataset.
#   SLIDER_ULTRA=1: LoRA solo en los 8 bloques dobles (fusión texto-imagen) y los 8 primeros simples (composición).
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
src = ROOT / "scripts" / "2_train_lora_klein9b.py"
dst = Path(os.environ.get("SLIDER_WORK", ROOT / "experiments" / "slider" / "work")) / "train_slider_klein9b.py"
dst.parent.mkdir(parents=True, exist_ok=True)
t = open(src, encoding="utf-8").read().replace("\r\n", "\n")
pairs = [
("""    print(f"Target LoRA Layers / Capas LoRA objetivo: {len(target_modules)} ({LORA_TARGETS})")
""", """    if os.environ.get("SLIDER_ULTRA") == "1":
        target_modules = [n for n in target_modules if n.startswith("transformer_blocks.")
                          or (n.startswith("single_transformer_blocks.") and int(n.split(".")[1]) < 8)]
    print(f"Target LoRA Layers / Capas LoRA objetivo: {len(target_modules)} ({LORA_TARGETS})")
"""),
("""    model.print_trainable_parameters()
""", """    model.print_trainable_parameters()
    from peft.tuners.lora import LoraLayer
    slider_layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    slider_base = LORA_ALPHA / LORA_RANK
    slider_push = float(os.environ.get("SLIDER_PUSH", "3"))
    print(f"Slider: push {slider_push}, {len(slider_layers)} layers")

    def slider_sign(sign):
        for m in slider_layers:
            m.scaling["default"] = slider_base * sign
"""),
("""            names = [random.choice(buckets[size]) for _ in range(BATCH_SIZE)]
""", """            names = [random.choice(buckets[size]) for _ in range(BATCH_SIZE)]
            n0 = names[0]
            slider_sign(slider_push * (-1.0 if n0.endswith("_neg") else random.choice((-1.0, 1.0)) if n0.endswith("_anc") else 1.0))
"""),
("""            loss.backward()
""", """            loss.backward()
            slider_sign(slider_push)
"""),
]
for a, c in pairs:
    assert t.count(a) == 1, (a, t.count(a))
    t = t.replace(a, c)
open(dst, "w", encoding="utf-8").write(t)
print("ok", dst)
