# Copia el dataset de data/ con los nombres del tipo Slider del trainer (docs/SLIDERS.md):
#   pXX_sad -> pXX_-100, pXX_neutral -> pXX_0, pXX_happy -> pXX_100, aXX -> aXX_anc.
# Las personas 20-23 se quedan fuera, para probar el LoRA con caras que no ha visto.
#   python experiments/slider/make_dataset.py <carpeta de salida>
import shutil, sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
out = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset_slider_happy_sad")
out.mkdir(parents=True, exist_ok=True)
positions = {"sad": -100, "neutral": 0, "happy": 100}
for i in range(20):
    for name, pos in positions.items():
        shutil.copy(DATA / "pairs" / f"p{i:02d}_{name}.jpg", out / f"p{i:02d}_{pos}.jpg")
for src in sorted((DATA / "anchors").glob("a*.jpg")):
    shutil.copy(src, out / f"{src.stem}_anc.jpg")
print(f"Ready / Listo: {out.resolve()} ({len(list(out.glob('*.jpg')))} images / imágenes)")
print(f"Test faces / Caras de prueba: {DATA / 'pairs'} p20_neutral ... p23_neutral")
