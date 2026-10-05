# Anchors para el slider: imágenes sin personas, antes = después. Enseñan al LoRA a no tocar lo que no es una cara.
import importlib.util, os, sys, torch
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])  # raíz del repo
WORK = os.environ.get("SLIDER_WORK", os.path.join(ROOT, "experiments", "slider", "work"))
os.chdir(ROOT)
sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("kt", os.path.join(ROOT, "scripts", "2_train_lora_klein9b.py"))
kt = importlib.util.module_from_spec(spec); spec.loader.exec_module(kt)
kt.TURBO_LORA_STRENGTH = 1.0

from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from transformers import Qwen3Model, AutoTokenizer

M = os.path.join(ROOT, "FLUX.2-Klein-9B_NF4")
OUT = os.path.join(WORK, "anchors")
os.makedirs(OUT, exist_ok=True)
tr = kt.load_nf4_transformer(M).to("cuda")
wrap = torch.nn.Module(); wrap.base_model = torch.nn.Module(); wrap.base_model.model = tr
kt.apply_turbo_lora(wrap)
pipe = Flux2KleinPipeline(
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(M, subfolder="scheduler"),
    vae=AutoencoderKLFlux2.from_pretrained(M, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda"),
    text_encoder=Qwen3Model.from_pretrained(os.path.join(M, "text_encoder"), dtype=torch.bfloat16, device_map="cuda"),
    tokenizer=AutoTokenizer.from_pretrained(M, subfolder="tokenizer"), transformer=tr)
scenes = [
    "a quiet beach with gentle waves at midday", "a cozy living room with a sofa and bookshelves",
    "a modern office with desks and large windows", "a red bicycle leaning against a brick wall",
    "a kitchen counter with fruit and a coffee machine", "a park path with autumn trees",
    "a library aisle full of books", "a cafe table with a cup of coffee and a croissant",
    "a city street with parked cars on an overcast day", "a mountain lake at golden hour",
    "a desk with a laptop, a lamp and a plant", "a garden with flowers and a wooden fence",
]
for i, s in enumerate(scenes):
    path = os.path.join(OUT, f"a{i:02d}.png")
    if not os.path.exists(path):
        pipe(prompt=f"Realistic photo of {s}, no people", height=768, width=768, num_inference_steps=4,
             guidance_scale=1.0, generator=torch.Generator("cuda").manual_seed(3000 + i)).images[0].save(path)
    print("done", i, flush=True)
