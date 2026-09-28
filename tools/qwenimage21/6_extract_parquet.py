# -*- coding: utf-8 -*-
"""
6_extract_parquet.py — Extrae imágenes y captions de datasets .parquet (Hugging Face)
Extracts images and captions from .parquet datasets (Hugging Face)

Uso / Usage:
    venv\\Scripts\\python.exe 6_extract_parquet.py <fichero.parquet | carpeta> --info
    venv\\Scripts\\python.exe 6_extract_parquet.py <parquet> -o L:\\mi_dataset --before bw_image --after color_image --caption "Colorize this manga page" --limit 30 --max-per title

Con --before y --after escribe pares de edición:  nombre_before.ext, nombre_after.ext, nombre.txt
Con una sola columna de imagen (--image) escribe un dataset normal: nombre.ext, nombre.txt
Sin columnas indicadas se detectan solas: 1 columna de imagen -> dataset normal; 2 -> hay que elegir cuál es el antes.

Las imágenes se guardan con sus bytes originales (sin recomprimir) y la extensión de su formato real.

Para datasets de colorear: --exclude "monochrome,greyscale,cover,sketch,no humans,text focus,credits"
descarta portadas, créditos y bocetos por sus etiquetas, y --min-colorfulness 15 los pares cuyo
después apenas tiene color. Con varios .parquet en una carpeta y --limit/--max-per title se mezclan obras.
"""
import argparse
import io
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8", ".jpg"),
    (b"RIFF", ".webp"),
    (b"GIF8", ".gif"),
    (b"BM", ".bmp"),
)


def image_ext(data):
    for magic, ext in MAGIC:
        if data.startswith(magic):
            return ext
    return None


def image_bytes(value):
    # Formato de imagen de Hugging Face: {"bytes": ..., "path": ...}; algunos datasets guardan bytes sin más.
    if isinstance(value, dict):
        value = value.get("bytes")
    return value if isinstance(value, (bytes, bytearray)) else None


def colorfulness(data):
    # Métrica de Hasler y Süsstrunk: ~0 en escala de grises, 15 poco color, 40+ muy colorido.
    im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB").resize((256, 256)), dtype=np.float32)
    rg = np.abs(im[..., 0] - im[..., 1])
    yb = np.abs(0.5 * (im[..., 0] + im[..., 1]) - im[..., 2])
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def parquet_files(path):
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet"))
    return [path]


def describe(files):
    first = pq.ParquetFile(files[0])
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"Files / Ficheros: {len(files)} | Rows / Filas: {total}")

    sample = next(first.iter_batches(batch_size=5)).to_pylist()
    image_cols, text_cols = [], []
    for name in first.schema_arrow.names:
        values = [row[name] for row in sample]
        data = [image_bytes(v) for v in values]
        if any(d is not None and image_ext(d) for d in data):
            image_cols.append(name)
            print(f"  [image / imagen] {name}: {Counter(image_ext(d) for d in data if d)}")
        elif all(isinstance(v, str) for v in values if v is not None):
            text_cols.append(name)
            example = next((v for v in values if v), "")
            print(f"  [text / texto]   {name}: {example[:100]!r}")
        else:
            print(f"  [other / otro]   {name}: {first.schema_arrow.field(name).type}")
    return image_cols, text_cols


def main():
    ap = argparse.ArgumentParser(description="Extract images and captions from .parquet / Extrae imágenes y captions de .parquet")
    ap.add_argument("source", help=".parquet file or folder with .parquet files / fichero .parquet o carpeta con .parquet")
    ap.add_argument("-o", "--output", help="output folder / carpeta de salida")
    ap.add_argument("--info", action="store_true", help="only show the columns / solo muestra las columnas")
    ap.add_argument("--image", help="image column (normal dataset) / columna de imagen (dataset normal)")
    ap.add_argument("--before", help="'before' image column (edit pairs) / columna de la imagen de antes")
    ap.add_argument("--after", help="'after' image column (edit pairs) / columna de la imagen de después")
    ap.add_argument("--caption-column", help="text column used as caption / columna de texto usada como caption")
    ap.add_argument("--caption", default="", help="fixed caption for every sample / caption fijo para todas las muestras")
    ap.add_argument("--prefix", default="img", help="file name prefix / prefijo de los nombres (default: img)")
    ap.add_argument("--limit", type=int, default=0, help="max samples, spread over all rows / máximo de muestras, repartidas por todas las filas (0 = all)")
    ap.add_argument("--max-per", help="column to balance by, e.g. title / columna para equilibrar, p. ej. title")
    ap.add_argument("--max-per-count", type=int, default=0, help="max samples per value of --max-per / máximo por valor (0 = limit / nº de valores)")
    ap.add_argument("--exclude", default="", help="comma-separated words; rows whose text columns contain them are skipped / palabras separadas por comas; se saltan las filas que las contengan")
    ap.add_argument("--min-colorfulness", type=float, default=0, help="skip samples whose (after) image is nearly grey, e.g. 15 / salta muestras cuya imagen (de después) es casi gris, p. ej. 15")
    args = ap.parse_args()

    files = parquet_files(args.source)
    if not files:
        sys.exit(f"[!] No .parquet files in / No hay ficheros .parquet en: {args.source}")

    image_cols, text_cols = describe(files)
    if args.info:
        return

    pairs = bool(args.before or args.after)
    if pairs and not (args.before and args.after):
        sys.exit("[!] Edit pairs need both --before and --after / Los pares de edición necesitan --before y --after")
    if not pairs and not args.image:
        if len(image_cols) != 1:
            sys.exit(f"[!] Choose the columns: --image, or --before and --after. Image columns / Elige las columnas: {image_cols}")
        args.image = image_cols[0]
    if not args.output:
        sys.exit("[!] Missing output folder / Falta la carpeta de salida: -o")

    columns = [args.before, args.after] if pairs else [args.image]
    exclude = [w.strip().lower() for w in args.exclude.split(",") if w.strip()]
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)

    # Reparto uniforme: con --limit se toma una fila de cada tramo, no las primeras N seguidas.
    stride = max(1, total // args.limit) if args.limit else 1
    per_value = Counter()
    max_per = args.max_per_count or (max(1, args.limit // 2) if args.limit else 0)

    os.makedirs(args.output, exist_ok=True)
    written, row_index, wanted = 0, 0, False
    with open(os.path.join(args.output, "_sources.txt"), "a", encoding="utf-8") as sources:
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    index, row_index = row_index, row_index + 1
                    if args.limit:
                        # Si la fila del tramo se descarta por un filtro, vale la siguiente válida.
                        wanted = wanted or index % stride == 0
                        if written >= args.limit or not wanted:
                            continue
                    if exclude and any(w in " ".join(str(row[c]).lower() for c in text_cols) for w in exclude):
                        continue
                    if args.max_per and max_per and per_value[row.get(args.max_per)] >= max_per:
                        continue

                    data = [image_bytes(row[c]) for c in columns]
                    if any(d is None or image_ext(d) is None for d in data):
                        continue
                    if args.min_colorfulness and colorfulness(data[-1]) < args.min_colorfulness:
                        continue

                    written += 1
                    wanted = False
                    stem = f"{args.prefix}_{written:04d}"
                    suffixes = ("_before", "_after") if pairs else ("",)
                    for d, suffix in zip(data, suffixes):
                        with open(os.path.join(args.output, stem + suffix + image_ext(d)), "wb") as f:
                            f.write(d)

                    caption = str(row.get(args.caption_column) or "") if args.caption_column else args.caption
                    if caption:
                        with open(os.path.join(args.output, stem + ".txt"), "w", encoding="utf-8") as f:
                            f.write(caption.strip())

                    if args.max_per:
                        per_value[row.get(args.max_per)] += 1
                    sources.write(f"{stem}\t{os.path.basename(path)}\trow {index}\n")

                if args.limit and written >= args.limit:
                    break

    print(f"\n✓ {written} {'pairs / pares' if pairs else 'images / imágenes'} -> {os.path.abspath(args.output)}")
    if args.max_per:
        for value, count in per_value.most_common():
            print(f"  {count} x {value}")


if __name__ == "__main__":
    main()
