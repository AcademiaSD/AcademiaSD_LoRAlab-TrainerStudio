# Barrido de fuerza de un LoRA de edición de Klein sobre retratos que no están en el dataset.
# uso: test_slider.py <carpeta resume_checkpoint> <salida.jpg>
import importlib.util, json, os, sys, torch
from pathlib import Path
from PIL import Image, ImageDraw
from safetensors.torch import load_file

ROOT = str(Path(__file__).resolve().parents[2])  # raíz del repo
WORK = os.environ.get("SLIDER_WORK", os.path.join(ROOT, "experiments", "slider", "work"))
CKPT, OUT = sys.argv[1], sys.argv[2]
os.chdir(ROOT)
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("kt", os.path.join(ROOT, "scripts", "2_train_lora_klein9b.py"))
kt = importlib.util.module_from_spec(spec); spec.loader.exec_module(kt)
kt.TURBO_LORA_STRENGTH = 1.0

from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from transformers import Qwen3Model, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.tuners.lora import LoraLayer

M = os.path.join(ROOT, "FLUX.2-Klein-9B_NF4")
tr = kt.load_nf4_transformer(M)
cfg = json.load(open(os.path.join(CKPT, "adapter_config.json")))
weights = load_file(os.path.join(CKPT, "adapter_model.safetensors"))
targets = sorted({k.split(".lora_")[0].replace("base_model.model.", "", 1) for k in weights})
model = get_peft_model(tr, LoraConfig(r=cfg["r"], lora_alpha=cfg["lora_alpha"], target_modules=targets))
set_peft_model_state_dict(model, {k: v.to(torch.bfloat16) for k, v in weights.items()})
model.to("cuda", dtype=None)
kt.apply_turbo_lora(model)
base_scale = cfg["lora_alpha"] / cfg["r"]
layers = [m for m in model.modules() if isinstance(m, LoraLayer)]

def set_strength(s):
    for m in layers:
        m.scaling["default"] = base_scale * s

te = Qwen3Model.from_pretrained(os.path.join(M, "text_encoder"), dtype=torch.bfloat16, device_map="cuda")
pipe = Flux2KleinPipeline(
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(M, subfolder="scheduler"),
    vae=AutoencoderKLFlux2.from_pretrained(M, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda"),
    text_encoder=te, tokenizer=AutoTokenizer.from_pretrained(M, subfolder="tokenizer"), transformer=model.base_model.model)

PEOPLE = [os.path.join(WORK, "pairs", "p%02d_neutral.png" % i) for i in (20, 21, 22, 23)]
PROMPTS = os.environ.get("PROMPTS", "Make the person happy.|Keep the photo exactly as it is.").split("|")
STRENGTHS = [float(x) for x in os.environ.get("STRENGTHS", "-1,0,0.5,1,1.5").split(",")]
T = 200
rows = []
for prompt in PROMPTS:
    for path in PEOPLE:
        src = Image.open(path).convert("RGB")
        row = []
        for s in STRENGTHS:
            set_strength(s)
            img = pipe(image=src, prompt=prompt, height=512, width=512, num_inference_steps=4, guidance_scale=1.0,
                       generator=torch.Generator("cuda").manual_seed(5)).images[0].resize((T, T))
            ImageDraw.Draw(img).text((4, 4), f"{s:+.1f}", fill="yellow")
            row.append(img)
        rows.append(row)
    print("done", prompt, flush=True)
sheet = Image.new("RGB", (T * len(STRENGTHS), T * len(rows)))
for y, row in enumerate(rows):
    for x, im in enumerate(row):
        sheet.paste(im, (x * T, y * T))
sheet.save(OUT, quality=88)
print("saved", OUT)
