# -*- coding: utf-8 -*-
"""Project storage layout: Models, Outputs and Cache_Data."""
import errno
import json
import os
import shutil
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
CACHE_DATA_DIR = PROJECT_ROOT / "Cache_Data"

REPORT_NAME = "layout_migration.json"

LEGACY_MODEL_NAMES = (
    "Anima-Base",
    "Krea-2-NF4",
    "Krea-2-Raw",
    "LTX23-NF4",
    "LTX23-Raw",
    "LTX23-NF4_weights",
    "MiniMax-H3",
    "MiniMax-H3-NF4",
    "Qwen-Image21-NF4",
    "Qwen-Image21-Raw",
    "Qwen-Image-2512-NF4",
    "Qwen-Image-2512",
    "Qwen-Image-Edit-2511",
    "Z-Image_NF4",
    "Z-Image-Raw",
    "Z-Image-Turbo_NF4",
    "Z-Image-Turbo-Raw",
    "FLUX.2-Klein-9B_NF4",
    "FLUX.2-Klein-Base-9B-Raw",
    "FLUX.3-Image-Klein_NF4",
    "flux-3-image-klein-base",
    "MOSS-Image-NF4",
    "MOSS-Image-Base",
    "MOSS-Image-SFT",
    "Fibo2_Dev-NF4",
    "Fibo2_Dev-Raw",
    "Ming-Image_NF4",
    "Ming-Image_Raw",
    "Ming-Image-Raw",
    "Ernie-NF4",
    "Ernie-Raw",
    "SDXL-Models",
    "Ideogram4-NF4",
    "Boogu-image-edit-NF4",
    "Boogu-image-edit-Raw",
    "Captioner-Qwen3-VL-8B",
    "Qwen3-VL-4B-Instruct",
    "MelBandRoFormer",
    # YuE2-3B (al final, para que la union con main no toque las demas lineas)
    # YuE2-3B (at the end, so the merge with main does not touch the other lines)
    "YuE2-3B_NF4",
    "YuE2-3B_Raw",
    "MERT-v2-FullSong",
    "YuE2-RealAudio-Head",
    "YuE2-Vae",
    "YuE2-Vae-legacy",
    "Qwen3-ASR",
    "SheetSage2",
)

CACHE_NAME_ALIASES = {
    "cached_data_LTX23": "cached_data_ltx23",
    "cached_data_MiniMaxH3": "cached_data_minimaxh3",
    "cached_data_minimaxh3_v3": "cached_data_minimaxh3",
}

OUTPUT_NAME_ALIASES = {
    "LTX23_lora_output": "ltx23_lora_output",
    "MiniMaxH3_lora_output": "minimaxh3_lora_output",
}

OUTPUT_NAME_PREFIXES = (
    "anima_lora_output",
    "boogu_lora_output",
    "ernie_lora_output",
    "fibo2_lora_output",
    "flux3klein_lora_output",
    "ideogram4_lora_output",
    "klein9b_lora_output",
    "krea2_lora_output",
    "ltx23_lora_output",
    "minimaxh3_lora_output",
    "mingimage_lora_output",
    "mossimage_lora_output",
    "qwen_image21_lora_output",
    "qwenimage2512_lora_output",
    "sdxl_lora_output",
    "yue2_lora_output",
    "zimage_lora_output",
)

_BUSY_CACHE = {}


def canonical_name(name, aliases):
    for old, new in aliases.items():
        if name == old:
            return new
        if name.startswith(old + "_"):
            return new + name[len(old):]
    return name


def canonical_cache_name(name):
    return canonical_name(name, CACHE_NAME_ALIASES)


def canonical_output_name(name):
    return canonical_name(name, OUTPUT_NAME_ALIASES)


def is_legacy_output_name(name):
    canonical = canonical_output_name(name)
    return any(canonical == prefix or canonical.startswith(prefix + "_") for prefix in OUTPUT_NAME_PREFIXES)


def training_in_progress(root=PROJECT_ROOT):
    root_path = Path(root).resolve()
    key = str(root_path)
    now = time.monotonic()
    cached = _BUSY_CACHE.get(key)
    if cached and now - cached[0] < 2.0:
        return cached[1]
    busy = False
    if psutil is not None:
        for proc in psutil.process_iter(["pid", "cmdline", "cwd"]):
            try:
                info = proc.info
                if info.get("pid") == os.getpid():
                    continue
                cwd = info.get("cwd")
                if not cwd or Path(cwd).resolve() != root_path:
                    continue
                arguments = info.get("cmdline") or []
                if "-m" in arguments:
                    continue
                for argument in arguments[1:]:
                    if argument.startswith("-"):
                        continue
                    filename = os.path.basename(argument)
                    if filename.endswith(".py") and filename.startswith(("2_train_lora_", "1_pre_cache_", "0_caption_")):
                        busy = True
                    break
                if busy:
                    break
            except (psutil.Error, OSError):
                continue
    _BUSY_CACHE[key] = (now, busy)
    return busy


def _layout_dirs(root):
    root = Path(root).resolve()
    return root, root / "Models", root / "Outputs", root / "Cache_Data"


def ensure_layout_dirs(root=PROJECT_ROOT):
    _, models, outputs, cache = _layout_dirs(root)
    models.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return models, outputs, cache


def _lexists(path):
    return os.path.lexists(str(path))


def _is_dir(path):
    try:
        return Path(path).is_dir()
    except OSError:
        return False


def _directory_size(path):
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def move_path(src, dst):
    src = Path(src)
    dst = Path(dst)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return "failed", str(exc)
    if _lexists(dst):
        return "collision", ""
    try:
        os.rename(src, dst)
        return "moved", ""
    except OSError as exc:
        if getattr(exc, "errno", None) != errno.EXDEV and getattr(exc, "winerror", None) != 17:
            return "failed", str(exc)
    try:
        if shutil.disk_usage(dst.parent).free < _directory_size(src):
            return "no_space", ""
        if src.is_symlink():
            os.symlink(os.readlink(src), dst)
        elif src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
    except OSError as exc:
        if _lexists(dst):
            try:
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            except OSError:
                pass
        return "failed", str(exc)
    try:
        if src.is_symlink() or not src.is_dir():
            src.unlink()
        else:
            shutil.rmtree(src)
    except OSError as exc:
        return "copied", "legacy folder not fully removed: " + str(exc)
    return "copied", ""


def _normalized(value):
    return os.path.normcase(os.path.normpath(value))


def _remap_string(value, mapping):
    if not isinstance(value, str) or not value:
        return value
    normalized = _normalized(value)
    for old, new in mapping:
        old_norm = _normalized(old)
        if normalized == old_norm or normalized.startswith(old_norm + os.sep):
            return new + value[len(old):]
    return value


def remap_value(value, mapping):
    if isinstance(value, str):
        return _remap_string(value, mapping)
    if isinstance(value, list):
        return [remap_value(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: remap_value(item, mapping) for key, item in value.items()}
    return value


def normalize_storage_field(key, value):
    if not isinstance(value, str) or not value.strip():
        return value
    if Path(value).is_absolute():
        return value
    name = Path(value).name
    if key == "cache_dir" and name.startswith("cached_data_"):
        return "./Cache_Data/" + canonical_cache_name(name)
    if key == "output_dir" and is_legacy_output_name(name):
        return "./Outputs/" + canonical_output_name(name)
    return value


def _write_json_atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return True
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        return False


def _rewrite_json(path, mapping, normalize=True):
    try:
        original = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    data = remap_value(original, mapping)
    if normalize and isinstance(data, dict):
        for key in ("cache_dir", "output_dir"):
            if key in data:
                data[key] = normalize_storage_field(key, data[key])
    if data == original:
        return False
    return _write_json_atomic(path, data)


def _restore_storage_value(kind, value, moves):
    if not isinstance(value, str) or Path(value).is_absolute():
        return value
    name = Path(value).name
    for item in moves:
        if item.get("kind") == kind and Path(item["old"]).name == name:
            return "./" + Path(item["new"]).name
    return value


def _restore_storage_json(path, moves):
    try:
        original = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    data = dict(original) if isinstance(original, dict) else original
    if isinstance(data, dict):
        if "cache_dir" in data:
            data["cache_dir"] = _restore_storage_value("cache", data["cache_dir"], moves)
        if "output_dir" in data:
            data["output_dir"] = _restore_storage_value("output", data["output_dir"], moves)
    if data == original:
        return False
    return _write_json_atomic(path, data)


def _restore_storage_fields(root, moves):
    changed = []
    settings = Path(root) / "settings"
    if settings.is_dir():
        for path in sorted(settings.glob("*.json")):
            if path.name == REPORT_NAME:
                continue
            if _restore_storage_json(path, moves):
                changed.append(str(path))
    for item in moves:
        target = Path(item["new"])
        if item.get("kind") in ("cache", "output") and target.is_dir():
            for path in sorted(target.glob("*.json")):
                if _restore_storage_json(path, moves):
                    changed.append(str(path))
    return changed


def _rewrite_custom_images(path, mapping):
    try:
        original = path.read_text(encoding="utf-8")
    except OSError:
        return False
    lines = original.splitlines()
    rewritten = [remap_value(line, mapping) for line in lines]
    if rewritten == lines:
        return False
    text = "\n".join(rewritten)
    if original.endswith(("\n", "\r")):
        text += "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        return True
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        return False


def rewrite_references(moves, root=PROJECT_ROOT, normalize=True):
    root = Path(root).resolve()
    changed = []
    mapping = [(str(Path(item["old"])), str(Path(item["new"]))) for item in moves]
    if not mapping:
        return changed
    settings = root / "settings"
    if settings.is_dir():
        for path in sorted(settings.glob("*.json")):
            if path.name == REPORT_NAME:
                continue
            if _rewrite_json(path, mapping, normalize):
                changed.append(str(path))
    for item in moves:
        target = Path(item["new"])
        if item.get("kind") == "cache" and target.is_dir():
            for path in sorted(target.glob("*.json")):
                if _rewrite_json(path, mapping, normalize):
                    changed.append(str(path))
            custom = target / "_custom_image.txt"
            if custom.is_file() and _rewrite_custom_images(custom, mapping):
                changed.append(str(custom))
        if item.get("kind") == "output" and target.is_dir():
            for path in sorted(target.glob("*.json")):
                if _rewrite_json(path, mapping, normalize):
                    changed.append(str(path))
    return changed


def _report_path(root):
    return Path(root).resolve() / "settings" / REPORT_NAME


def write_report(moves, rewritten, root=PROJECT_ROOT):
    if not moves:
        return None
    path = _report_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    existing = {"moves": [], "rewrites": []}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing["moves"] = loaded.get("moves") or []
                existing["rewrites"] = loaded.get("rewrites") or []
        except (OSError, ValueError):
            pass
    by_old = {str(item.get("old")): item for item in existing["moves"] if item.get("old")}
    for item in moves:
        by_old[str(item["old"])] = item
    rewrites = list(dict.fromkeys(list(existing["rewrites"]) + list(rewritten)))
    report = {
        "version": 1,
        "root": str(Path(root).resolve()),
        "moves": list(by_old.values()),
        "rewrites": rewrites,
    }
    return path if _write_json_atomic(path, report) else None


def _migration_entry(kind, src, dst):
    return {
        "kind": kind,
        "old": str(Path(src).resolve()),
        "new": str(Path(dst).resolve()),
    }


def _migrate_item(kind, src, dst, root, dry_run, moves, collisions):
    src = Path(src)
    dst = Path(dst)
    entry = _migration_entry(kind, src, dst)
    if not _lexists(src):
        return
    if _lexists(dst):
        entry["status"] = "collision"
        collisions.append(entry)
        return
    if dry_run:
        entry["status"] = "planned"
        moves.append(entry)
        return
    status, error = move_path(src, dst)
    entry["status"] = status
    if error:
        entry["error"] = error
    if status in ("moved", "copied"):
        moves.append(entry)
    elif status in ("collision", "no_space", "failed"):
        collisions.append(entry)


def ensure_layout(root=PROJECT_ROOT, dry_run=False):
    root = Path(root).resolve()
    models, outputs, cache = _layout_dirs(root)[1:]
    if not dry_run and training_in_progress(root):
        entry = {
            "kind": "layout",
            "old": str(root),
            "new": str(models),
            "status": "busy",
        }
        return {"root": str(root), "moves": [], "collisions": [entry], "rewrites": []}
    if not dry_run:
        try:
            models.mkdir(parents=True, exist_ok=True)
            outputs.mkdir(parents=True, exist_ok=True)
            cache.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            entry = {
                "kind": "layout",
                "old": str(root),
                "new": str(models),
                "status": "failed",
                "error": str(exc),
            }
            return {"root": str(root), "moves": [], "collisions": [entry], "rewrites": []}
    moves = []
    collisions = []
    for name in LEGACY_MODEL_NAMES:
        _migrate_item("model", root / name, models / name, root, dry_run, moves, collisions)
    if root.is_dir():
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if path.name.startswith("cached_data_") and _is_dir(path):
                target = cache / canonical_cache_name(path.name)
                _migrate_item("cache", path, target, root, dry_run, moves, collisions)
            if is_legacy_output_name(path.name) and _is_dir(path):
                target = outputs / canonical_output_name(path.name)
                _migrate_item("output", path, target, root, dry_run, moves, collisions)
    rewritten = []
    if not dry_run and moves:
        rewritten = rewrite_references(moves, root)
        write_report(moves, rewritten, root)
    return {"root": str(root), "moves": moves, "collisions": collisions, "rewrites": rewritten}


def ensure_layout_and_log(root=PROJECT_ROOT):
    result = ensure_layout(root)
    for item in result["moves"]:
        print("[layout] {} -> {}".format(item["old"], item["new"]))
    for item in result["collisions"]:
        line = "[layout] {}: {} -> {}".format(item["status"], item["old"], item["new"])
        if item.get("error"):
            line += " | " + item["error"]
        print(line)
    if result["moves"]:
        print("[layout] moved {} folders, rewrote {} files".format(
            len(result["moves"]), len(result["rewrites"])))
    return result


def _remap_absolute_model(path, root):
    path = Path(path)
    try:
        parent = path.parent.resolve()
    except OSError:
        return path
    if parent != root or path.name not in LEGACY_MODEL_NAMES:
        return path
    if _lexists(path) and training_in_progress(root):
        return path
    target = root / "Models" / path.name
    if target.exists():
        return target
    if _lexists(path):
        status, _ = move_path(path, target)
        if status in ("moved", "copied"):
            entry = _migration_entry("model", path, target)
            entry["status"] = status
            rewritten = rewrite_references([entry], root)
            write_report([entry], rewritten, root)
            return target
    return path


def resolve_model_dir(value=None, default=None, root=PROJECT_ROOT):
    root = Path(root).resolve()
    raw = str(value if value is not None and str(value).strip() else (default or "")).strip()
    if not raw:
        raise ValueError("Model path is empty")
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if path.is_absolute():
        return _remap_absolute_model(path, root)
    parts = path.parts
    if parts and parts[0].casefold() == "models":
        target = root / "Models" / Path(*parts[1:])
        source = root / parts[1] if len(parts) == 2 else None
        if source is not None and _lexists(source) and training_in_progress(root):
            return source
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            if source is not None and _lexists(source):
                return source
            return target
        if source is not None and not target.exists() and _lexists(source) and _is_dir(source):
            status, _ = move_path(source, target)
            if status in ("moved", "copied"):
                entry = _migration_entry("model", source, target)
                entry["status"] = status
                rewritten = rewrite_references([entry], root)
                write_report([entry], rewritten, root)
            elif status in ("collision", "no_space", "failed"):
                return source
        return target
    if len(parts) == 1:
        name = parts[0]
        target = root / "Models" / name
        source = root / name
        if _lexists(source) and training_in_progress(root):
            return source
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return source if _lexists(source) else target
        if not target.exists() and _lexists(source) and _is_dir(source):
            status, _ = move_path(source, target)
            if status in ("moved", "copied"):
                entry = _migration_entry("model", source, target)
                entry["status"] = status
                rewritten = rewrite_references([entry], root)
                write_report([entry], rewritten, root)
            elif status in ("collision", "no_space", "failed"):
                return source
        return target
    return (root / path).resolve()


def _storage_target(kind, name, root):
    root, models, outputs, cache = _layout_dirs(root)
    if kind == "cache":
        return cache / canonical_cache_name(name)
    return outputs / canonical_output_name(name)


def _resolve_storage(kind, value, default, root):
    root = Path(root).resolve()
    raw = str(value if value is not None and str(value).strip() else (default or "")).strip()
    if not raw:
        raise ValueError("Storage path is empty")
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    name = path.name
    match = name.startswith("cached_data_") if kind == "cache" else is_legacy_output_name(name)
    if path.is_absolute():
        if not match:
            return path
        parent = path.parent.resolve()
        if parent == root:
            if _lexists(path) and training_in_progress(root):
                return path
            target = _storage_target(kind, name, root)
            if not target.exists() and _lexists(path) and _is_dir(path):
                status, _ = move_path(path, target)
                if status in ("moved", "copied"):
                    entry = _migration_entry(kind, path, target)
                    entry["status"] = status
                    rewritten = rewrite_references([entry], root)
                    write_report([entry], rewritten, root)
                    return target
            return target if target.exists() else path
        return path
    if not match or len(path.parts) != 1:
        return (root / path).resolve()
    target = _storage_target(kind, name, root)
    source = root / name
    if _lexists(source) and training_in_progress(root):
        return source
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return source if _lexists(source) else target
    if not target.exists() and _lexists(source) and _is_dir(source):
        status, _ = move_path(source, target)
        if status in ("moved", "copied"):
            entry = _migration_entry(kind, source, target)
            entry["status"] = status
            rewritten = rewrite_references([entry], root)
            write_report([entry], rewritten, root)
        elif status in ("collision", "no_space", "failed"):
            return source
    return target


def resolve_cache_dir(value=None, default=None, root=PROJECT_ROOT):
    return _resolve_storage("cache", value, default, root)


def resolve_output_dir(value=None, default=None, root=PROJECT_ROOT):
    return _resolve_storage("output", value, default, root)


def resolve_project_path(value=None, default=None, base_dir=PROJECT_ROOT):
    base = Path(base_dir).resolve()
    raw = str(value if value is not None and str(value).strip() else (default or "")).strip()
    if not raw:
        return base
    name = Path(raw).name
    if name.startswith("cached_data_"):
        return resolve_cache_dir(raw, default, base)
    if is_legacy_output_name(name):
        return resolve_output_dir(raw, default, base)
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_report(root=PROJECT_ROOT):
    path = _report_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def rollback(root=PROJECT_ROOT):
    root = Path(root).resolve()
    report = load_report(root)
    if not report:
        return {"root": str(root), "moves": [], "collisions": [], "rewrites": []}
    reverse = []
    collisions = []
    for item in reversed(report.get("moves") or []):
        if item.get("status") not in ("moved", "copied"):
            continue
        old = Path(item["old"])
        new = Path(item["new"])
        entry = {"kind": item.get("kind"), "old": str(new), "new": str(old)}
        if old.exists():
            entry["status"] = "collision"
            collisions.append(entry)
            continue
        if not new.exists():
            entry["status"] = "missing"
            collisions.append(entry)
            continue
        status, error = move_path(new, old)
        entry["status"] = status
        if error:
            entry["error"] = error
        if status in ("moved", "copied"):
            reverse.append({"kind": item.get("kind"), "old": item["new"], "new": item["old"]})
        else:
            collisions.append(entry)
    rewritten = rewrite_references(reverse, root, normalize=False) if reverse else []
    if reverse:
        rewritten.extend(_restore_storage_fields(root, reverse))
    return {"root": str(root), "moves": reverse, "collisions": collisions, "rewrites": rewritten}
