# Dataset sintético para el slider triste <-> alegre con Klein 9B NF4 + LoRA turbo (4 pasos).
# Genera retratos neutros y, editando cada uno, su versión alegre y triste.
import importlib.util, os, sys, random, torch
from pathlib import Path
from PIL import Image

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
OUT = os.path.join(WORK, "pairs")
os.makedirs(OUT, exist_ok=True)

tr = kt.load_nf4_transformer(M).to("cuda")
# apply_turbo_lora busca las capas como en el modelo PEFT del entrenamiento (base_model.model.*).
wrap = torch.nn.Module(); wrap.base_model = torch.nn.Module(); wrap.base_model.model = tr
kt.apply_turbo_lora(wrap)
te = Qwen3Model.from_pretrained(os.path.join(M, "text_encoder"), dtype=torch.bfloat16, device_map="cuda")
pipe = Flux2KleinPipeline(
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(M, subfolder="scheduler"),
    vae=AutoencoderKLFlux2.from_pretrained(M, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda"),
    text_encoder=te, tokenizer=AutoTokenizer.from_pretrained(M, subfolder="tokenizer"), transformer=tr)

people = [
    "a young East Asian woman with long straight black hair", "an elderly white man with a grey beard",
    "a middle-aged Black woman with short curly hair", "a teenage boy with freckles and red hair",
    "a South Asian man in his thirties with a short beard", "an old Japanese woman with white hair in a bun",
    "a blonde woman in her twenties with wavy hair", "a Latino man in his forties with glasses",
    "a young Black man with short dreadlocks", "a middle-aged woman with a brown bob haircut",
    "a bald man in his sixties", "a young Middle Eastern woman wearing a headscarf",
    "a little girl with pigtails", "a man in his fifties with salt and pepper hair",
    "an Indigenous woman with long braided hair", "a young man with a buzz cut and earrings",
    "a Korean man in his twenties with dyed blue hair", "an elderly Black man with a white moustache",
    "a woman in her thirties with freckles and a red ponytail", "a chubby man in his forties with curly brown hair",
    "a young woman with an undercut and nose piercing", "an old woman with curly grey hair and glasses",
    "a boy around ten years old with messy brown hair", "a Scandinavian man with long blond hair",
]
places = ["in a park", "in a modern office", "on a city street", "in a kitchen", "against a plain grey wall",
          "in a library", "at the beach", "in a cafe"]
light = ["soft daylight", "warm golden hour light", "studio lighting", "overcast light"]
EDITS = {
    "happy": "Make the person smile broadly with a genuinely happy, joyful expression. Keep exactly the same person, face, hair, clothes, pose, background, lighting and framing.",
    "sad": "Make the person look sad, with a downturned mouth, sad eyes and slightly raised inner eyebrows. Keep exactly the same person, face, hair, clothes, pose, background, lighting and framing.",
}
rng = random.Random(7)
for i, who in enumerate(people):
    stem = os.path.join(OUT, f"p{i:02d}")
    if not os.path.exists(stem + "_neutral.png"):
        prompt = (f"Photo portrait of {who} {rng.choice(places)}, head and shoulders, looking at the camera, "
                  f"neutral relaxed expression, mouth closed, {rng.choice(light)}, realistic photography")
        img = pipe(prompt=prompt, height=768, width=768, num_inference_steps=4, guidance_scale=1.0,
                   generator=torch.Generator("cuda").manual_seed(1000 + i)).images[0]
        img.save(stem + "_neutral.png")
    src = Image.open(stem + "_neutral.png").convert("RGB")
    for name, instr in EDITS.items():
        if os.path.exists(f"{stem}_{name}.png"):
            continue
        out = pipe(image=src, prompt=instr, height=768, width=768, num_inference_steps=4, guidance_scale=1.0,
                   generator=torch.Generator("cuda").manual_seed(2000 + i)).images[0]
        out.save(f"{stem}_{name}.png")
    print("done", i, who, flush=True)
