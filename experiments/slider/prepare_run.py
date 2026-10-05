# Prepara una carpeta de entrenamiento del slider a partir del dataset de data/ (o de work/pairs y work/anchors).
#   python experiments/slider/prepare_run.py <nombre> [--push 3] [--ultra]
# Crea work/<nombre>/ con ds/ (pares _pos, _neg y anchors _anc + su caption), settings/ y el enlace al modelo,
# y deja en work/pairs las imágenes de las 4 personas reservadas para test_slider.py.
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("SLIDER_WORK", HERE / "work"))
CAPTION = "Keep the photo exactly as it is."
TRAIN_PEOPLE, TEST_PEOPLE = range(0, 20), range(20, 24)

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--push", type=float, default=3.0)
ap.add_argument("--ultra", action="store_true")
args = ap.parse_args()


def find(folder, stem):
    """Imagen generada en work/ si existe; si no, la del dataset del repo (data/)."""
    for base in (WORK / folder, HERE / "data" / folder):
        for ext in (".png", ".jpg"):
            if (base / (stem + ext)).exists():
                return base / (stem + ext)
    sys.exit(f"[!] Missing / Falta: {folder}/{stem}")


run = WORK / args.name
ds = run / "ds"
ds.mkdir(parents=True, exist_ok=True)
(run / "settings").mkdir(exist_ok=True)


def put(src, name):
    Image.open(src).convert("RGB").save(ds / f"{name}.png")


for i in TRAIN_PEOPLE:
    for kind, target in (("pos", "happy"), ("neg", "sad")):
        put(find("pairs", f"p{i:02d}_neutral"), f"p{i:02d}_{kind}_before")
        put(find("pairs", f"p{i:02d}_{target}"), f"p{i:02d}_{kind}_after")
        (ds / f"p{i:02d}_{kind}.txt").write_text(CAPTION, encoding="utf-8")
for i in range(12):
    src = find("anchors", f"a{i:02d}")
    put(src, f"a{i:02d}_anc_before")
    put(src, f"a{i:02d}_anc_after")
    (ds / f"a{i:02d}_anc.txt").write_text(CAPTION, encoding="utf-8")

# Personas reservadas: test_slider.py las lee de work/pairs.
(WORK / "pairs").mkdir(parents=True, exist_ok=True)
for i in TEST_PEOPLE:
    out = WORK / "pairs" / f"p{i:02d}_neutral.png"
    if not out.exists():
        Image.open(find("pairs", f"p{i:02d}_neutral")).convert("RGB").save(out)

json.dump({"dataset_path": "./ds", "project_name": "slider", "target_area": 512 * 512, "max_side": 1024, "multiple": 16,
           "lora_type": "edit", "preview_custom_prompt": CAPTION,
           "preview_edit_image": str(WORK / "pairs" / "p20_neutral.png")},
          open(run / "settings" / "pre_cache_settings_klein9b.json", "w"), indent=1)
json.dump({"project_name": "slider", "total_steps": 1200, "grad_accum_steps": 1, "lr": 2e-4, "warmup_steps": 50,
           "lora_rank": 4, "lora_alpha": 4, "save_every": 400, "preview_every": 0, "seed": 42},
          open(run / "settings" / "train_settings_klein9b.json", "w"), indent=1)

model = ROOT / "FLUX.2-Klein-9B_NF4"
link = run / "FLUX.2-Klein-9B_NF4"
if model.exists() and not link.exists():
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(model)], check=True, capture_output=True)
    else:
        link.symlink_to(model, target_is_directory=True)

env = f"SLIDER_PUSH={args.push:g}" + (" SLIDER_ULTRA=1" if args.ultra else "")
print(f"Ready / Listo: {run} ({len(list(ds.glob('*.png')))} images)")
print(f"  cd {run}")
print(f"  python {ROOT / 'scripts' / '1_pre_cache_klein9b.py'}")
print(f"  {env} python {WORK / 'train_slider_klein9b.py'}")
