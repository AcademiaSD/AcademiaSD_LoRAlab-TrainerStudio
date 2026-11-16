"""
make_dora_norms.py — Normas por fila del peso original para exportar DoRA desde un modelo NF4
Row norms of the original weights, for exporting DoRA from an NF4 model

DoRA aprende la magnitud de cada fila de salida. El trainer la mide sobre el peso NF4, pero ComfyUI la divide
por la norma del peso que carga (bf16, fp8...). La norma NF4 se aparta de la original un 0,2 % por fila, más
que lo que aprende DoRA (0,02-0,05 %), y el LoRA pierde el parecido. Con estas normas la exportación usa la
del peso original (ver lora_options.dora_scales, base_norms).

Escribe <carpeta del transformer NF4>/dora_norms.safetensors: una entrada fp32 [out] por capa, con el nombre
de la capa en el modelo NF4. Los originales de Hugging Face se leen en streaming, sin guardarlos en disco.
Solo CPU.

Uso / usage:
    python tools/dora_norms/make_dora_norms.py              # todos los que falten
    python tools/dora_norms/make_dora_norms.py zimage wan22 # solo esos
    python tools/dora_norms/make_dora_norms.py --force ...  # rehacerlos
"""
import argparse, glob, json, os, struct, sys, time
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M = os.path.join(ROOT, "Models")
OUT_NAME = "dora_norms.safetensors"
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _cached(repo, sub):
    hits = glob.glob(os.path.join(HF_CACHE, "models--" + repo.replace("/", "--"), "snapshots", "*", sub))
    return hits[0] if hits else None


def _fused_qkv_names(key):
    """Z-Image (ComfyUI) e Ideogram 4 (fp8) fusionan q/k/v en attention.qkv y llaman attention.out / .o
    a to_out.0."""
    base, _, rest = key.rpartition(".attention.")
    if rest == "qkv.weight":
        return [(base + ".attention." + n, i) for i, n in enumerate(("to_q", "to_k", "to_v"))]
    if rest in ("out.weight", "o.weight"):
        return [(base + ".attention.to_out.0", None)]
    return None


# nombre: (carpeta NF4 relativa a Models, origen, sufijo de la subcarpeta del origen)
# origen: "hf:<repo>" (streaming), "local:<ruta>" o "cache:<repo>" (caché de Hugging Face).
MODELS = {
    "zimage":      ("Z-Image_NF4/transformer", r"local:F:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\z_image_bf16.safetensors", ""),
    "flux1":       ("FLUX.1-Dev_NF4/transformer", "hf:black-forest-labs/FLUX.1-dev", "transformer"),
    "klein9b":     ("FLUX.2-Klein-9B_NF4/transformer", "hf:black-forest-labs/FLUX.2-klein-base-9B", "transformer"),
    "flux3klein":  ("FLUX.3-Image-Klein_NF4/transformer", "hf:black-forest-labs/flux-3-image-klein-base", "transformer"),
    "krea2":       ("Krea-2-NF4", "hf:krea/Krea-2-Raw", "transformer"),
    "qwenimage21": ("Qwen-Image21-NF4/transformer", "hf:Qwen/Qwen-Image-2.1", "transformer"),
    "qwen2512":    ("Qwen-Image-2512-NF4/transformer", "hf:Qwen/Qwen-Image-2512", "transformer"),
    "qwen2512edit": ("Qwen-Image-2512-NF4/transformer_edit", "hf:Qwen/Qwen-Image-Edit-2511", "transformer"),
    "boogu":       ("Boogu-image-edit-NF4/transformer", "hf:Boogu/Boogu-Image-0.1-Edit", "transformer"),
    "mingimage":   ("Ming-Image_NF4/transformer", "hf:inclusionAI/Ming-Image-0.1-Design", "transformer"),
    "moss_base":   ("MOSS-Image-NF4/transformer", "hf:OpenMOSS-Team/MOSS-Image-Base", "transformer"),
    "moss_sft":    ("MOSS-Image-NF4/transformer_sft", "hf:OpenMOSS-Team/MOSS-Image-SFT", "transformer"),
    "ernie":       ("Ernie-NF4/transformer", "hf:baidu/ERNIE-Image", "transformer"),
    "fibo2":       ("Fibo2_Dev-NF4/transformer", "local:" + os.path.join(M, "Fibo2_Dev-Raw", "transformer"), ""),
    "ideogram4":   ("Ideogram4-NF4/transformer", "cache:ideogram-ai/ideogram-4-fp8", "transformer"),
    "ltx23":       ("LTX23-NF4", "hf:diffusers/LTX-2.3-Diffusers", "transformer"),
    "wan22_high":  ("Wan22-T2V-A14B-NF4/transformer_nf4", "hf:Wan-AI/Wan2.2-T2V-A14B-Diffusers", "transformer"),
    "wan22_low":   ("Wan22-T2V-A14B-NF4/transformer_2_nf4", "hf:Wan-AI/Wan2.2-T2V-A14B-Diffusers", "transformer_2"),
}
RENAME = {"zimage": _fused_qkv_names, "ideogram4": _fused_qkv_names}

DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
      "F8_E5M2": torch.float8_e5m2}


# ---------------------------------------------------------------- capas NF4 del modelo de destino
def nf4_layers(folder):
    """Nombres de las capas cuantizadas (los que ve el trainer) y una función que las descuantiza."""
    idx = os.path.join(folder, "index.json")
    if os.path.isfile(idx):
        q = json.load(open(idx, encoding="utf-8"))["quantized"]
        q = {n: v for n, v in q.items() if v.get("quant_type", "nf4") == "nf4"}
        wdir = os.path.join(folder, "weights") if os.path.isdir(os.path.join(folder, "weights")) else folder

        def get(name):
            with safe_open(os.path.join(wdir, q[name]["file"]), "pt") as f:
                keys = list(f.keys())
                if "weight" in keys:
                    w, pre = f.get_tensor("weight"), "quant_state."
                    st = {k[len(pre):]: f.get_tensor(k) for k in keys if k.startswith(pre)}
                else:                                  # Wan: <capa>.weight y <capa>.weight.qs.*
                    w, pre = f.get_tensor(name + ".weight"), name + ".weight.qs."
                    st = {k[len(pre):]: f.get_tensor(k) for k in keys if k.startswith(pre)}
                    if not st:
                        pre = name + ".weight."
                        st = {k[len(pre):]: f.get_tensor(k) for k in keys if k.startswith(pre)}
            return w, st
        return sorted(q), get
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    files = [f for f in files if os.path.basename(f) != OUT_NAME]
    where = {}
    for p in files:
        with safe_open(p, "pt") as f:
            for k in f.keys():
                if k.endswith(".weight.quant_state.bitsandbytes__nf4"):
                    where[k[:-len(".weight.quant_state.bitsandbytes__nf4")]] = p

    def get(name):
        with safe_open(where[name], "pt") as f:
            pre = name + ".weight."
            return f.get_tensor(name + ".weight"), {k[len(pre):]: f.get_tensor(k) for k in f.keys() if k.startswith(pre)}
    return sorted(where), get


def dequant(w, st):
    import bitsandbytes.functional as F
    from bitsandbytes.functional import QuantState
    return F.dequantize_4bit(w, QuantState.from_dict(st, device="cpu")).float()


# ---------------------------------------------------------------- lectura de safetensors (local o streaming)
def _files(src, sub):
    kind, _, where = src.partition(":")
    if kind == "local":
        if os.path.isfile(where):
            return [("local", where)]
        return [("local", p) for p in sorted(glob.glob(os.path.join(where, "*.safetensors")))]
    if kind == "cache":
        d = _cached(where, sub)
        if not d:
            raise RuntimeError(f"{where}/{sub} no está en la caché de Hugging Face")
        return [("local", p) for p in sorted(glob.glob(os.path.join(d, "*.safetensors")))]
    from huggingface_hub import HfApi
    api = HfApi()
    names = sorted(f.path for f in api.list_repo_tree(where, path_in_repo=sub or None, recursive=False)
                   if f.path.endswith(".safetensors"))
    return [("hf", (where, n)) for n in names]


class _Reader:
    """safetensors por rangos: local o por HTTP (HfFileSystem), sin bajar el archivo entero."""
    def __init__(self, kind, where):
        if kind == "local":
            self.f = open(where, "rb")
        else:
            from huggingface_hub import HfFileSystem
            repo, path = where
            self.f = HfFileSystem().open(f"{repo}/{path}", "rb", block_size=64 * 2 ** 20)
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n

    def tensor(self, key):
        h = self.header[key]
        a, b = h["data_offsets"]
        self.f.seek(self.base + a)
        buf = bytearray(self.f.read(b - a))
        return torch.frombuffer(buf, dtype=DT[h["dtype"]]).reshape(h["shape"])

    def close(self):
        self.f.close()


def _norms_of(kind, where, wanted):
    """Normas por fila de los .weight 2D (o más) de un archivo. wanted(key) decide si interesa."""
    r = _Reader(kind, where)
    out = {}
    try:
        keys = [k for k, h in r.header.items() if k != "__metadata__" and k.endswith(".weight")
                and len(h["shape"]) >= 2 and h["dtype"] in DT and wanted(k)]
        keys.sort(key=lambda k: r.header[k]["data_offsets"][0])       # lectura en orden: menos saltos
        for k in keys:
            w = r.tensor(k)
            n = w.float().reshape(w.shape[0], -1).norm(dim=1)
            sk = k + "_scale"                                         # fp8 con escala por fila (Ideogram)
            if sk in r.header:
                s = r.tensor(sk).float().reshape(-1)
                n = n * (s if s.numel() == n.numel() else s.expand(n.numel()))
            out[k] = n
    finally:
        r.close()
    return out


# ---------------------------------------------------------------- un modelo
def make(name, force=False, workers=4):
    folder, src, sub = MODELS[name]
    folder = os.path.join(M, folder)
    dst = os.path.join(folder, OUT_NAME)
    if not os.path.isdir(folder):
        print(f"[{name}] no está instalado ({folder}); se salta", flush=True)
        return
    if os.path.isfile(dst) and not force:
        print(f"[{name}] ya existe {dst}; se salta (--force para rehacerlo)", flush=True)
        return
    layers, get = nf4_layers(folder)
    want_set = set(layers)
    ren = RENAME.get(name)

    def wanted(k):
        if ren:
            return ren(k) is not None or k[:-len(".weight")] in want_set
        return k[:-len(".weight")] in want_set

    t0 = time.time()
    files = _files(src, sub)
    print(f"[{name}] {len(layers)} capas NF4 | {len(files)} archivo(s) de {src} {sub}", flush=True)
    raw = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(files))) as ex:
        for i, part in enumerate(ex.map(lambda fw: _norms_of(fw[0], fw[1], wanted), files)):
            raw.update(part)
            print(f"[{name}]   archivo {i + 1}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
    norms = {}
    for k, n in raw.items():
        mapped = ren(k) if ren else None
        if mapped:
            parts = len(mapped)
            for nn, i in mapped:
                norms[nn] = n.chunk(parts)[i] if i is not None else n
        else:
            norms[k[:-len(".weight")]] = n
    missing = [n for n in layers if n not in norms]
    if missing:
        raise RuntimeError(f"[{name}] faltan {len(missing)} capas en el original, p. ej. {missing[:5]}")
    norms = {n: norms[n].contiguous() for n in layers}
    # Comprobación: la norma NF4 debe quedar a ~0,2 % de la original (otra variante del modelo daría mucho más).
    step = max(1, len(layers) // 12)
    # Las filas que valen 0 en el original (p. ej. la mitad de algunas adaLN) no cuentan: su dora_scale sale 0
    # y la fila sigue en 0 en ComfyUI, como en el original.
    rs, zeros = [], int(sum((norms[n] == 0).sum().item() for n in layers))
    for n in layers[::step]:
        q = dequant(*get(n))
        live = norms[n] > 0
        rs.append(q.reshape(q.shape[0], -1).norm(dim=1)[live] / norms[n][live])
    r = torch.cat(rs)
    print(f"[{name}] |NF4|/|original| en {len(rs)} capas: media {r.mean():.5f} desv {r.std():.5f} "
          f"min {r.min():.4f} max {r.max():.4f}" + (f" | {zeros} filas a 0 en el original" if zeros else ""), flush=True)
    if abs(r.mean().item() - 1) > 0.01 or r.std().item() > 0.01:
        raise RuntimeError(f"[{name}] las normas no cuadran con el NF4: ¿otro modelo de origen?")
    meta = {"format": "dora_norms", "version": "1", "source": f"{src} {sub}".strip(),
            "layers": str(len(norms)), "nf4_ratio_std": f"{r.std().item():.5f}"}
    tmp = dst + ".tmp"
    save_file(norms, tmp, metadata=meta)
    os.replace(tmp, dst)
    mb = os.path.getsize(dst) / 2 ** 20
    print(f"[{name}] OK {dst} ({len(norms)} capas, {mb:.1f} MB, {time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help=", ".join(MODELS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    bad = []
    for name in a.models or list(MODELS):
        try:
            make(name, a.force, a.workers)
        except Exception as e:
            bad.append(name)
            print(f"[{name}] ERROR: {type(e).__name__}: {e}", flush=True)
    print("FIN" + (f" con errores en: {', '.join(bad)}" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
