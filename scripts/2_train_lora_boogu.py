# -*- coding: utf-8 -*-
"""Launch the official DiffSynth-Studio Boogu Image Edit LoRA trainer."""
import json
import hashlib
import os
import signal
import subprocess
import sys
from pathlib import Path

from boogu_dataset import CACHE_VERSION, build_metadata, cache_path, resolve_path
from boogu_model import NF4_TRANSFORMER
from i18n import t
from project_layout import resolve_cache_dir, resolve_output_dir
import lora_options

BASE_DIR = Path(__file__).resolve().parents[1]
SETTINGS_DIR = BASE_DIR / "settings"
OUTPUT_DEFAULT = "./Outputs/boogu_lora_output"
MODEL_ID = "Boogu/Boogu-Image-0.1-Edit"
TARGET_MODULES = "to_q,to_k,to_v,to_out.0,img_to_q,img_to_k,img_to_v,img_out,instruct_to_q,instruct_to_k,instruct_to_v,instruct_out"
ACTIVE_CHILD = None


def forward_stop(signum, _frame):
    # Stop de la GUI: en Windows el CTRL_BREAK llega a todo el grupo, también al motor, que guarda y sale;
    # aquí solo hay que seguir esperando. En Linux el SIGINT llega solo a este proceso: se le reenvía.
    child = ACTIVE_CHILD
    if os.name != "nt" and child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def engine_python():
    return Path(sys.executable)


def configure_console_encoding():
    """Keep forwarded DiffSynth Unicode output safe on Windows consoles and pipes."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main():
    configure_console_encoding()
    try:
        precache_cfg = read_json(SETTINGS_DIR / "pre_cache_settings_boogu.json")
        train_cfg = read_json(SETTINGS_DIR / "train_settings_boogu.json")
        dataset_dir = resolve_path(precache_cfg.get("dataset_path", train_cfg.get("dataset_path", "./dataset")), BASE_DIR)
        cache_dir = resolve_cache_dir(precache_cfg.get("cache_dir", train_cfg.get("cache_dir")), "./Cache_Data/cached_data_boogu", BASE_DIR)
        output_dir = resolve_output_dir(train_cfg.get("output_dir"), OUTPUT_DEFAULT, BASE_DIR)
        lora_type = str(train_cfg.get("lora_type", precache_cfg.get("lora_type", "edit"))).lower()
        if lora_type not in ("normal", "edit"):
            lora_type = "edit"
        if not NF4_TRANSFORMER.is_file():
            raise RuntimeError("Falta el transformer NF4 local de Boogu. Ejecuta primero la preparación del modelo.")

        train_script = BASE_DIR / "scripts" / "boogu_diffsynth_train.py"
        python = engine_python()
        if not python.is_file() or not train_script.is_file():
            raise RuntimeError(
                "Falta DiffSynth en el entorno normal del Studio. Comprueba la instalación estándar de dependencias."
            )

        metadata_path = cache_dir / "metadata.jsonl"
        rows = build_metadata(dataset_dir, metadata_path, lora_type)
        manifest_path = cache_dir / "cache_manifest.json"
        manifest = read_json(manifest_path)
        metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        trigger = str(train_cfg.get("trigger_word", precache_cfg.get("trigger_word", ""))).strip()
        expected_cache_files = [cache_path(cache_dir, row).name for row in rows]
        if (
            manifest.get("cache_version") != CACHE_VERSION
            or manifest.get("mode") != lora_type
            or manifest.get("complete") is not True
            or manifest.get("metadata_sha256") != metadata_hash
            or manifest.get("trigger_word", "") != trigger
            or manifest.get("samples") != expected_cache_files
            or any(not cache_path(cache_dir, row).is_file() for row in rows)
        ):
            print("\n[!] " + t("Please run Pre-Cache first!"), flush=True)
            return 2  # la GUI muestra este código como "falta la pre-caché"
        total_steps = max(1, int(train_cfg.get("total_steps", 500)))
        grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
        max_pixels = max(262144, min(1048576, int(precache_cfg.get("target_area", 589824))))
        size_multiple = max(16, int(precache_cfg.get("multiple", 32)))
        max_image_side = max(256, int(precache_cfg.get("max_side", 1280)))
        model_files = [NF4_TRANSFORMER]
        batch_size = max(1, int(train_cfg.get("batch_size", 1)))
        output_dir.mkdir(parents=True, exist_ok=True)

        args = [
            str(python), str(train_script),
            "--dataset_base_path", str(dataset_dir),
            "--dataset_metadata_path", str(metadata_path),
            "--boogu_cache_dir", str(cache_dir),
            "--data_file_keys", "image,edit_image" if lora_type == "edit" else "image",
            "--max_pixels", str(max_pixels),
            "--size_multiple", str(size_multiple),
            "--max_image_side", str(max_image_side),
            "--dataset_num_workers", str(max(0, int(train_cfg.get("dataset_num_workers", 0)))),
            "--model_paths", json.dumps([str(path) for path in model_files]),
            "--learning_rate", str(float(train_cfg.get("lr", 2e-4))),
            "--num_epochs", "1",
            "--remove_prefix_in_ckpt", "pipe.dit.",
            "--output_path", str(output_dir),
            "--lora_base_model", "dit",
            "--lora_target_modules", "all" if train_cfg.get("lora_targets") == "all" else TARGET_MODULES,
            "--lora_rank", str(max(1, int(train_cfg.get("lora_rank", 16)))),
            "--lora_alpha", str(max(1, int(train_cfg.get("lora_alpha", train_cfg.get("lora_rank", 16))))),
            "--lora_batch_size", str(batch_size),
            "--total_steps", str(total_steps),
            "--seed", str(int(train_cfg.get("seed", 42))),
            "--lr_scheduler", str(train_cfg.get("lr_scheduler", "cosine")),
            "--warmup_steps", str(max(0, int(train_cfg.get("warmup_steps", 0)))),
            "--lr_cycles", str(max(1, int(train_cfg.get("lr_cycles", 1)))),
            "--preview_every", str(max(0, int(train_cfg.get("preview_every", 0)))),
            "--preview_steps", str(max(1, int(train_cfg.get("preview_steps", 30)))),
            "--preview_cfg", str(float(train_cfg.get("preview_cfg", 4.0))),
            "--preview_size", str(max(0, int(train_cfg.get("preview_size", 0)))),
            "--preview_caption_mode", str(train_cfg.get("preview_caption_mode", "first")),
            "--preview_custom_prompt", str(train_cfg.get("preview_custom_prompt", "")),
            "--preview_edit_image", str(train_cfg.get("preview_edit_image", "")),
            "--lora_type", lora_type,
            "--trigger_word", str(train_cfg.get("trigger_word", precache_cfg.get("trigger_word", ""))),
            "--live_settings", str(SETTINGS_DIR / "train_settings_boogu.json"),
            "--gradient_accumulation_steps", str(grad_accum),
            "--use_gradient_checkpointing",
            "--find_unused_parameters",
        ]
        if lora_type == "edit":
            args += ["--extra_inputs", "edit_image"]
        if train_cfg.get("use_rslora", False) and not train_cfg.get("use_lokr", False):
            args.append("--use_rslora")
        if train_cfg.get("use_lokr", False):
            args += ["--use_lokr", "--lokr_factor", str(max(1, int(train_cfg.get("lokr_factor", 16))))]
        elif train_cfg.get("use_loraplus", False):
            args += ["--use_loraplus", "--loraplus_ratio", str(max(1.0, float(train_cfg.get("loraplus_ratio", 16))))]
            args += ["--customized_optimizer", "boogu_loraplus.LoRAPlusAdamW"]
        save_every = max(0, int(train_cfg.get("save_every", 50)))
        if save_every:
            args += ["--save_steps", str(save_every)]

        env = os.environ.copy()
        for key in (
            "HF_TOKEN_PATH", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
            "MODELSCOPE_CACHE", "XDG_CACHE_HOME", "PYTHONPATH", "PYTHONHOME",
        ):
            env.pop(key, None)
        env.update({
            "DIFFSYNTH_SKIP_DOWNLOAD": "True",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        })
        token = str(read_json(SETTINGS_DIR / "HF_token.json").get("token", "")).strip()
        if token:
            env["HF_TOKEN"] = token
        else:
            env.pop("HF_TOKEN", None)

        print(f"  {t('Model ID'):<22}: {MODEL_ID} ({'edit' if lora_type == 'edit' else 'normal'})")
        print(f"  {t('Trigger Word'):<22}: {trigger}")
        print(f"  {t('Cache Dir'):<22}: {cache_dir} ({len(rows)})")
        print(f"  {t('Output Dir'):<22}: {output_dir}")
        print(f"  {t('Total Steps'):<22}: {total_steps}")
        print(f"  {t('Learning Rate'):<22}: {train_cfg.get('lr', 2e-4)}")
        print(f"  {t('LR Scheduler'):<22}: {lora_options.lr_scheduler_label(train_cfg.get('lr_scheduler', 'cosine'), int(train_cfg.get('warmup_steps', 0)), int(train_cfg.get('lr_cycles', 1)))}")
        print(f"  {'LoRA Rank/Alpha':<22}: {args[args.index('--lora_rank') + 1]}/{args[args.index('--lora_alpha') + 1]}")
        print(f"  {'rsLoRA / LoRA+ / LoKr':<22}: {'on' if '--use_rslora' in args else 'off'} / {'on' if '--use_loraplus' in args else 'off'} / {'on' if '--use_lokr' in args else 'off'}")
        print(f"  {'Batch / Grad Accum':<22}: {batch_size}/{grad_accum}")
        print(f"  {t('Save Every'):<22}: {save_every}")
        print(f"  {t('Preview Every'):<22}: {train_cfg.get('preview_every', 0)}", flush=True)
        global ACTIVE_CHILD
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), forward_stop)
        process = subprocess.Popen(
            args, cwd=str(BASE_DIR), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        ACTIVE_CHILD = process
        # En binario: el modo texto convierte el "\r" de la barra de pasos en "\n" y cada paso salía en una línea.
        while chunk := process.stdout.read1(4096):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return process.wait()
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
