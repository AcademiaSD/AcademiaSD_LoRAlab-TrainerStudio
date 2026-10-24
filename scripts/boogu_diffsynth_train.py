# -*- coding: utf-8 -*-
"""Boogu Edit training entry point adapted from ModelScope DiffSynth-Studio.

The Boogu model pipeline and training implementation remain provided by the
DiffSynth-Studio package installed in LoRAlab's regular Python environment.
Upstream recipe: examples/boogu_image/model_training/train.py (Apache-2.0).
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import accelerate
import torch
from torch.utils.data import DataLoader, Dataset
from boogu_dataset import CACHE_VERSION, cache_path
from boogu_model import NF4_DIR, quiet
from i18n import t
import training_metrics
from diffsynth.core.data.operators import *  # noqa: F401,F403
from diffsynth.diffusion import *  # noqa: F401,F403
from diffsynth.pipelines.boogu_image import BooguImagePipeline, ModelConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# La GUI busca "<Paso> N/Total" en la consola para saber por qué paso va.
STEP_WORD = t("Step")


def configure_console_encoding():
    """Keep DiffSynth's Unicode startup banner safe on Windows cp1252 consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


class BooguImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_alpha=None,
        use_rslora=False,
        use_lokr=False,
        lokr_factor=16,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        quant_options=None,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            quant_options=quant_options,
            device=device,
        )
        processor_config = ModelConfig(processor_path) if processor_path is not None else None
        self.pipe = BooguImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            processor_config=processor_config,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        self.lora_alpha = int(lora_alpha if lora_alpha is not None else lora_rank)
        self.use_lokr = bool(use_lokr)
        self.use_rslora = bool(use_rslora and not self.use_lokr)
        self.lokr_factor = max(1, int(lokr_factor))
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            None if self.use_lokr else lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(
                pipe, **inputs_shared, **inputs_posi
            ),
        }

    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        # The shared DiffSynth hook does not expose alpha or rsLoRA.
        # Add those PEFT options while preserving its adapter injection behavior.
        # LoRA en FP32 (DiffSynth lo deja en BF16): como en los demás entrenadores, en 16 bits los updates
        # pequeños se redondean y se pierde el detalle fino. El LoRA exportado se guarda en BF16.
        upcast_dtype = torch.float32
        from peft import LoraConfig, inject_adapter_in_model

        if target_modules == ["all"] or target_modules == "all":
            target_modules = [name for name, module in model.named_modules()
                              if name and isinstance(module, torch.nn.Linear)]
            if not target_modules:
                raise RuntimeError("No se encontraron capas lineales para aplicar LoRA.")
        if self.use_lokr:
            from lora_options import LoKrModel
            if isinstance(target_modules, str):
                target_modules = target_modules.split(",")
            names = [name for name, _module in model.named_modules()
                     if name and any(name == target or name.endswith("." + target) for target in target_modules)]
            if not names:
                raise RuntimeError("No se encontraron las capas elegidas para LoKr.")
            return LoKrModel(model, names, lora_rank, self.lora_alpha, self.lokr_factor)
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        alpha = int(lora_alpha if lora_alpha is not None else self.lora_alpha)
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            use_rslora=self.use_rslora,
        )
        model = inject_adapter_in_model(config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        for module in model.modules():
            for adapter in getattr(module, "lora_A", {}).values():
                for param in adapter.parameters():
                    param._boogu_loraplus_side = "A"
            for adapter in getattr(module, "lora_B", {}).values():
                for param in adapter.parameters():
                    param._boogu_loraplus_side = "B"
        return model

    def get_pipeline_inputs(self, data):
        if "prompt_embeds" in data and "input_latents" in data:
            from diffsynth.pipelines.boogu_image import BooguImageUnit_FreqsCis

            height, width = int(data["height"]), int(data["width"])
            inputs_shared = {
                "height": height,
                "width": width,
                "input_latents": data["input_latents"],
                "instruction_hidden_states": data["prompt_embeds"],
                "instruction_attention_mask": data["prompt_mask"],
                "ref_image_hidden_states": (
                    [[data["reference_latents"]]] if "reference_latents" in data else None
                ),
                "freqs_cis": BooguImageUnit_FreqsCis().process(
                    self.pipe, height, width,
                )["freqs_cis"],
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            }
            return inputs_shared, {}, {}

        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Boogu's noise initializer encodes the target image as training latents.
            "input_image": data["image"],
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        if "prompt_embeds" not in data:
            for unit in self.pipe.units:
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.task_to_loss[self.task](self.pipe, *inputs)


class BooguCachedDataset(Dataset):
    """Reads precomputed Boogu text states and VAE latents; no raw-image encoders at train time."""
    def __init__(self, metadata_path, cache_dir, mode, repeat=1):
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        with open(metadata_path, "r", encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        if not self.rows:
            raise ValueError("Los metadatos de Boogu no contienen muestras.")
        self.repeat = max(1, int(repeat))
        for row in self.rows:
            path = cache_path(self.cache_dir, row)
            if not path.is_file():
                raise FileNotFoundError(f"Falta la precaché de {row['image']}: {path}. Ejecuta Pre-Caché.")
            try:
                cached = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise RuntimeError(f"No se pudo leer la precaché de {row['image']}: {exc}") from exc
            required = {"prompt_embeds", "prompt_mask", "negative_embeds", "negative_mask", "input_latents", "width", "height"}
            missing = required - cached.keys()
            if cached.get("cache_version") != CACHE_VERSION or cached.get("mode") != self.mode:
                raise RuntimeError(f"La precaché de {row['image']} no corresponde a la versión/modo actual; vuelve a ejecutar Pre-Caché.")
            if missing:
                raise RuntimeError(f"La precaché de {row['image']} está incompleta: {', '.join(sorted(missing))}.")
            if row.get("edit_image") and "reference_latents" not in cached:
                raise RuntimeError(f"Falta el latente de referencia para {row['image']}. Vuelve a ejecutar Pre-Caché.")

    def __len__(self):
        return len(self.rows) * self.repeat

    def __getitem__(self, index):
        row = self.rows[index % len(self.rows)]
        cached = torch.load(cache_path(self.cache_dir, row), map_location="cpu", weights_only=True)
        return {key: cached[key] for key in (
            "prompt_embeds", "prompt_mask", "negative_embeds", "negative_mask",
            "input_latents", "reference_latents", "width", "height",
        ) if key in cached}


def boogu_image_parser():
    parser = argparse.ArgumentParser(description="Boogu Image Edit LoRA training.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--boogu_cache_dir", type=str, default=None)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--use_rslora", action="store_true")
    parser.add_argument("--use_lokr", action="store_true")
    parser.add_argument("--lokr_factor", type=int, default=16)
    parser.add_argument("--lora_batch_size", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--size_multiple", type=int, default=32)
    parser.add_argument("--max_image_side", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_scheduler", choices=("constant", "constant_with_warmup", "cosine", "linear", "cosine_with_restarts"), default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_cycles", type=int, default=1)
    parser.add_argument("--preview_every", type=int, default=0)
    parser.add_argument("--preview_steps", type=int, default=30)
    parser.add_argument("--preview_cfg", type=float, default=4.0)
    parser.add_argument("--preview_caption_mode", choices=("first", "random", "rotate4", "custom"), default="first")
    parser.add_argument("--preview_custom_prompt", type=str, default="")
    parser.add_argument("--preview_edit_image", type=str, default="")
    parser.add_argument("--lora_type", choices=("normal", "edit"), default="edit")
    parser.add_argument("--use_loraplus", action="store_true")
    parser.add_argument("--loraplus_ratio", type=float, default=16.0)
    parser.add_argument("--trigger_word", type=str, default="")
    parser.add_argument("--live_settings", type=str, default="")
    parser.add_argument("--preview_size", type=int, default=0)
    return parser


# Ajustes en caliente: clave del json de la GUI -> (atributo de args, tipo). Rank, alpha, batch,
# resolución y carpetas se fijan al arrancar y siguen necesitando Stop -> Resume.
LIVE_KEYS = {
    "total_steps": ("total_steps", int),
    "save_every": ("save_steps", int),
    "lr": ("learning_rate", float),
    "preview_every": ("preview_every", int),
    "preview_steps": ("preview_steps", int),
    "preview_cfg": ("preview_cfg", float),
    "preview_caption_mode": ("preview_caption_mode", str),
    "preview_custom_prompt": ("preview_custom_prompt", lambda v: str(v).strip()),
    "preview_edit_image": ("preview_edit_image", lambda v: str(v).strip()),
    "preview_size": ("preview_size", int),
    "seed": ("seed", int),  # solo previews
}


def reload_live_settings(args, live):
    """Si la GUI ha reescrito el json (Save JSON con el entrenamiento en marcha), aplica LIVE_KEYS a args."""
    try:
        mtime = os.path.getmtime(args.live_settings)
    except OSError:
        return []
    if live.get("mtime") is None or mtime == live["mtime"]:
        # La primera llamada solo memoriza la fecha: el guardado previo al lanzamiento no es un cambio.
        live["mtime"] = live.get("mtime") or mtime
        return []
    try:
        with open(args.live_settings, "r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, ValueError):
        return []  # guardado a medias: se vuelve a intentar en el paso siguiente
    live["mtime"] = mtime
    changes = []
    for key, (attr, cast) in LIVE_KEYS.items():
        if key not in raw:
            continue
        try:
            value = cast(raw[key])
        except (TypeError, ValueError):
            continue
        if value != getattr(args, attr):
            changes.append((attr, getattr(args, attr), value))
            setattr(args, attr, value)
    return changes


def launch_boogu_training_task(accelerator, dataset, model, model_logger, args):
    # DiffSynth provides these helpers in ``diffusion.runner`` but does not
    # re-export them from ``diffusion.__init__`` in the Studio's installed build.
    from diffsynth.diffusion.runner import get_optimizer_class, exclude_quantized_params_from_ddp_sync
    from safetensors.torch import load_file, save_file
    from lora_options import lr_factor

    optimizer_class = get_optimizer_class(args.customized_optimizer)
    optimizer_kwargs = {"b_lr_ratio": args.loraplus_ratio} if args.use_loraplus else {}
    optimizer = optimizer_class(
        model.trainable_modules(), lr=args.learning_rate,
        weight_decay=args.weight_decay, **optimizer_kwargs,
    )
    warmup = min(max(0, args.warmup_steps), max(1, args.total_steps))
    cycles = max(1, args.lr_cycles)
    # args.total_steps se lee en cada paso: puede cambiar en caliente.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_factor(args.lr_scheduler, step, max(1, args.total_steps), warmup, 0.1, cycles),
    )
    dataloader = DataLoader(
        dataset, batch_size=max(1, args.lora_batch_size), shuffle=True,
        collate_fn=lambda samples: samples,
        num_workers=max(0, args.dataset_num_workers),
    )
    model.to(device=accelerator.device)
    exclude_quantized_params_from_ddp_sync(accelerator, model)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    unwrapped = accelerator.unwrap_model(model)
    trainable = {name: param for name, param in unwrapped.named_parameters() if param.requires_grad}

    # Reanudar es seguir EL MISMO entrenamiento: pesos entrenables, optimizador, scheduler y paso. Un
    # checkpoint con otras capas o formas (otro rank, LoKr...) no encaja: se avisa y se empieza de cero.
    resume_dir = os.path.join(args.output_path, "resume_checkpoint")
    weights_file = os.path.join(resume_dir, "trainable.safetensors")
    state_file = os.path.join(resume_dir, "training_state.pt")
    step_file = os.path.join(args.output_path, "current_step.txt")
    global_step = 0
    if os.path.isfile(weights_file) and os.path.isfile(state_file):
        saved = load_file(weights_file, device="cpu")
        if saved.keys() == trainable.keys() and all(saved[k].shape == p.shape for k, p in trainable.items()):
            with torch.no_grad():
                for name, param in trainable.items():
                    param.copy_(saved[name].to(param.dtype))
            state = torch.load(state_file, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = int(state["step"])
            print(t("Resuming training from step {n}...", n=global_step), flush=True)
        else:
            print("[!] " + t("The saved checkpoint does not match these LoRA settings (rank, LoKr...); starting a new training."), flush=True)

    def save_checkpoint(step):
        if not accelerator.is_main_process or step <= 0:
            return
        print("\n" + t("Saving checkpoint state at step {n}...", n=step), flush=True)
        os.makedirs(resume_dir, exist_ok=True)
        save_file({name: param.detach().cpu().contiguous() for name, param in trainable.items()}, weights_file)
        torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, state_file)
        with open(step_file, "w", encoding="utf-8") as stream:
            stream.write(str(step))
        model_logger._last_step = step
        name = f"Boogu_LoRA_step_{step}.safetensors"
        model_logger.save_model(accelerator, model, name)
        print("✓ " + t("Checkpoint saved at step {n}: {path}", n=step, path=os.path.join(args.output_path, name)), flush=True)

    # Stop desde la GUI: CTRL_BREAK (Windows) o SIGINT. Se termina el paso en curso, se guarda y se sale.
    stop = {"requested": False}

    def request_stop(sig, _frame):
        stop["requested"] = True
        print("\n[!] " + t("Stop signal received ({sig}).", sig=sig), flush=True)

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)

    model.train()
    start_step, running_loss, t_avg = global_step, 0.0, 0.0
    metrics, logged_loss = training_metrics.MetricsLog(args.output_path, start_step), 0.0
    t0 = time.time()
    live = {}
    reload_live_settings(args, live)
    while global_step < args.total_steps and not stop["requested"]:
        for samples in dataloader:
            changes = reload_live_settings(args, live)
            if changes:
                print("\n[LIVE] " + t("Settings reloaded without stopping:"), flush=True)
                for attr, old, new in changes:
                    print(f"[LIVE]   {attr}: {old} -> {new}", flush=True)
                    if attr == "learning_rate" and old > 0:
                        # LoRA+ tiene otro grupo con lr * ratio: se escalan todos por igual.
                        scheduler.base_lrs = [lr * new / old for lr in scheduler.base_lrs]
                if global_step >= args.total_steps:
                    break
            with accelerator.accumulate(model):
                losses = [model(sample) for sample in samples]
                loss = torch.stack(losses).mean()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
            if not accelerator.sync_gradients:
                continue
            scheduler.step()
            global_step += 1
            dt = time.time() - t0
            t0 = time.time()
            t_avg = dt if t_avg == 0 else 0.1 * dt + 0.9 * t_avg
            done = global_step - start_step
            total_steps = args.total_steps
            eta = int(max(0, total_steps - global_step) * t_avg)
            pct = min(1.0, global_step / total_steps)
            bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            avg = running_loss / (done * max(1, args.gradient_accumulation_steps))
            print(f"\r{STEP_WORD} {global_step:4d}/{total_steps} [{bar}] {pct * 100:5.1f}% | Loss {avg:.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {t_avg:.2f}s/it | ETA {eta // 3600:02d}:{eta % 3600 // 60:02d}:{eta % 60:02d}",
                  end="", flush=True)
            # El loss de este paso es la media de sus micro-pasos de acumulación.
            metrics.log(global_step, (running_loss - logged_loss) / max(1, args.gradient_accumulation_steps),
                        scheduler.get_last_lr()[0], None, dt, total_steps)
            logged_loss = running_loss
            if args.save_steps and global_step % args.save_steps == 0:
                save_checkpoint(global_step)
            if args.preview_every > 0 and global_step % args.preview_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print(flush=True)
                    run_boogu_preview(unwrapped, dataset, args, global_step)
                accelerator.wait_for_everyone()
                t0 = time.time()
            if global_step >= args.total_steps or stop["requested"]:
                break
    if stop["requested"]:
        if not (args.save_steps and global_step % args.save_steps == 0):
            save_checkpoint(global_step)
        return
    if not (args.save_steps and global_step % args.save_steps == 0):
        save_checkpoint(global_step)
    model_logger._last_step = global_step
    model_logger.save_model(accelerator, model, "Boogu_FINAL_LoRA.safetensors")
    print("\n\n" + t("Training completed!"), flush=True)
    print("✓ " + t("Final LoRA saved to: {path}", path=os.path.join(args.output_path, "Boogu_FINAL_LoRA.safetensors")), flush=True)


def run_boogu_preview(model, dataset, args, step):
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.boogu_image import BooguImagePipeline, BooguImageUnit_FreqsCis

    pipe = model.pipe
    rows = dataset.rows
    mode = args.preview_caption_mode
    if mode == "custom" and args.preview_custom_prompt.strip():
        sample_key = {"image": "_custom"}
        cached_path = cache_path(args.boogu_cache_dir, sample_key)
        manifest = json.loads((Path(args.boogu_cache_dir) / "cache_manifest.json").read_text(encoding="utf-8"))
        expected_prompt = args.preview_custom_prompt.strip()
        trigger = args.trigger_word.strip()
        if trigger and trigger.casefold() not in expected_prompt.casefold():
            expected_prompt = f"{trigger}, {expected_prompt}"
        default_edit_image = rows[0].get("edit_image", "") if args.lora_type == "edit" else ""
        expected_edit_image = args.preview_edit_image.strip() or default_edit_image
        if (manifest.get("preview_custom_prompt") != expected_prompt
                or manifest.get("preview_edit_image", "") != expected_edit_image or not cached_path.is_file()):
            # El servidor lo está codificando (Save JSON): mientras tanto, la primera muestra del dataset.
            print("[PREVIEW] " + t("The new prompt is being encoded on CPU (Save JSON); previews switch to it when ready."), flush=True)
            mode = "first"
    if mode != "custom" or not args.preview_custom_prompt.strip():
        if mode == "random":
            index = random.randrange(len(rows))
        elif mode == "rotate4":
            index = (step // max(1, args.preview_every) - 1) % min(4, len(rows))
        else:
            index = 0
        sample_key = rows[index]
        cached_path = cache_path(args.boogu_cache_dir, sample_key)
    sample = torch.load(cached_path, map_location="cpu", weights_only=True)
    height, width = int(sample["height"]), int(sample["width"])
    if args.preview_size > 0:
        # Misma proporción con el área elegida; múltiplo 16 (VAE /8 y parches 2x2).
        scale = args.preview_size / math.sqrt(height * width)
        height, width = max(16, round(height * scale / 16) * 16), max(16, round(width * scale / 16) * 16)
    device = pipe.device
    try:
        pipe.dit.eval()
        with torch.inference_mode():
            vae_files = sorted((NF4_DIR / "vae").glob("*.safetensors"))
            if not vae_files:
                raise FileNotFoundError("No se encuentran los pesos locales del VAE para las previews.")
            with quiet():
                preview_pipe = BooguImagePipeline.from_pretrained(
                    torch_dtype=pipe.torch_dtype,
                    device=device,
                    model_configs=[ModelConfig(path=[str(path) for path in vae_files])],
                )
            preview_pipe.dit = pipe.dit
            preview_pipe.dit.to(device)
            preview_pipe.vae_decoder.to(device)
            preview_pipe.scheduler.set_timesteps(max(1, args.preview_steps), training=False)

            latents = preview_pipe.generate_noise(
                (1, 16, height // 8, width // 8),
                seed=args.seed if args.seed > 0 else None,
                rand_device="cpu",
                device=device,
                torch_dtype=pipe.torch_dtype,
            )
            frequencies = BooguImageUnit_FreqsCis().process(preview_pipe, height, width)["freqs_cis"]
            ref_latents = sample.get("reference_latents")
            inputs_shared = {
                "latents": latents,
                "freqs_cis": frequencies,
                "ref_image_hidden_states": [[ref_latents.to(device)]] if ref_latents is not None else None,
            }
            inputs_posi = {
                "instruction_hidden_states": sample["prompt_embeds"].to(device),
                "instruction_attention_mask": sample["prompt_mask"].to(device),
            }
            inputs_nega = {
                "instruction_hidden_states": sample["negative_embeds"].to(device),
                "instruction_attention_mask": sample["negative_mask"].to(device),
            }
            for progress_id, timestep in enumerate(preview_pipe.scheduler.timesteps):
                timestep = timestep.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
                prediction = preview_pipe.cfg_guided_model_fn(
                    preview_pipe.model_fn,
                    args.preview_cfg,
                    inputs_shared,
                    inputs_posi,
                    inputs_nega,
                    dit=preview_pipe.dit,
                    timestep=timestep,
                    progress_id=progress_id,
                )
                latents = preview_pipe.step(
                    preview_pipe.scheduler,
                    latents,
                    progress_id,
                    prediction,
                )
                inputs_shared["latents"] = latents
            decoded = preview_pipe.vae_decoder(latents, device=device)
            result = preview_pipe.vae_output_to_image(decoded)
            del preview_pipe
        preview_dir = args.output_path
        os.makedirs(preview_dir, exist_ok=True)
        result.save(os.path.join(preview_dir, f"preview_step_{step}.png"))
        print("  ↳ " + t("Preview saved to: {path}", path=os.path.join(preview_dir, f"preview_step_{step}.png")), flush=True)
    except Exception as exc:
        print(f"[Boogu preview] Error en step {step}: {exc}", flush=True)
    finally:
        pipe.dit.train()


class BooguModelLogger(ModelLogger):
    def __init__(self, *args, use_lokr=False, lokr_alpha=16, lora_rank=32,
                 lokr_factor=16,
                 lora_alpha=32, use_rslora=False, use_loraplus=False,
                 loraplus_ratio=16, output_name="boogu_lora", trigger_word="",
                 train_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_lokr = use_lokr
        self.lokr_alpha = lokr_alpha
        self.lokr_factor = lokr_factor
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora
        self.use_loraplus = use_loraplus
        self.loraplus_ratio = loraplus_ratio
        self.output_name = output_name
        self.trigger_word = trigger_word
        self.train_args = train_args

    def _metadata(self, network_module, network_alpha):
        args = self.train_args
        metadata = {
            "format": "lycoris" if self.use_lokr else "peft",
            "base_model": "Boogu-Image-0.1-Edit",
            "ss_base_model_version": "Boogu-Image-0.1-Edit",
            "ss_network_module": network_module,
            "ss_network_dim": str(self.lora_rank),
            "ss_network_alpha": str(network_alpha),
            "ss_output_name": self.output_name,
        }
        if args is not None:
            metadata.update({
                "ss_learning_rate": str(args.learning_rate),
                "ss_lr_scheduler": str(args.lr_scheduler),
                "ss_lr_warmup_steps": str(args.warmup_steps),
                "ss_max_train_steps": str(args.total_steps),
                "ss_steps": str(getattr(self, "_last_step", args.total_steps)),
                "ss_batch_size_per_device": str(args.lora_batch_size),
                "ss_gradient_accumulation_steps": str(args.gradient_accumulation_steps),
                "ss_seed": str(args.seed),
                "ss_mixed_precision": "bf16",
            })
        if self.use_rslora and not self.use_lokr:
            metadata["rslora"] = "true"
            metadata["rslora_train_alpha"] = str(self.lora_alpha)
        if self.use_loraplus and not self.use_lokr:
            metadata["loraplus_ratio"] = str(self.loraplus_ratio)
        if self.use_lokr:
            metadata["lokr_factor"] = str(self.lokr_factor)
        if self.trigger_word:
            import json
            metadata["trigger_word"] = self.trigger_word
            metadata["ss_tag_frequency"] = json.dumps({"dataset": {self.trigger_word: 1}}, ensure_ascii=False)
        return metadata

    def save_model(self, accelerator, model, file_name):
        import lora_options
        from safetensors.torch import save_file
        from peft import get_peft_model_state_dict
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            alpha = lora_options.export_alpha(self.lora_alpha, self.lora_rank, self.use_rslora)
            if self.use_lokr:
                state = lora_options.lokr_export(
                    unwrapped.pipe.dit, lambda name: "diffusion_model." + name, self.lokr_alpha
                )
                metadata = self._metadata("lycoris.kohya", self.lokr_alpha)
            else:
                state = {}
                for key, value in get_peft_model_state_dict(unwrapped.pipe.dit).items():
                    key = "diffusion_model." + key.replace("base_model.model.", "")
                    state[key] = value.to(torch.bfloat16).cpu().contiguous()
                    if key.endswith(".lora_A.weight"):
                        state[key[:-len(".lora_A.weight")] + ".alpha"] = torch.tensor(float(alpha))
                metadata = self._metadata("peft.LoraModel", alpha)
            os.makedirs(self.output_path, exist_ok=True)
            save_file(state, os.path.join(self.output_path, file_name),
                      metadata=metadata)


def main():
    configure_console_encoding()
    args = boogu_image_parser().parse_args()
    accelerate.utils.set_seed(args.seed)
    if args.model_paths and "Boogu-image-edit-NF4" in args.model_paths:
        from boogu_model import register_nf4_models
        register_nf4_models()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )
    if not args.boogu_cache_dir:
        raise ValueError("Falta --boogu_cache_dir; ejecuta la precaché Boogu antes del entrenamiento.")
    dataset = BooguCachedDataset(
        args.dataset_metadata_path,
        args.boogu_cache_dir,
        mode=args.lora_type,
        repeat=args.dataset_repeat,
    )
    print(t("Loading {name}...", name="Boogu Image Edit Transformer (NF4)"), flush=True)
    with quiet():
        model = BooguImageTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            processor_path=args.processor_path,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            use_rslora=args.use_rslora and not args.use_lokr,
            use_lokr=args.use_lokr,
            lokr_factor=args.lokr_factor,
            lora_checkpoint=args.lora_checkpoint,
            preset_lora_path=args.preset_lora_path,
            preset_lora_model=args.preset_lora_model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            extra_inputs=args.extra_inputs,
            fp8_models=args.fp8_models,
            offload_models=args.offload_models,
            quant_options=args.quant_options,
            resume_from_checkpoint=args.resume_from_checkpoint,
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
            task=args.task,
            device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        )
    model_logger = BooguModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        use_lokr=args.use_lokr,
        lokr_alpha=args.lora_alpha or args.lora_rank,
        lokr_factor=args.lokr_factor,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha or args.lora_rank,
        use_rslora=args.use_rslora and not args.use_lokr,
        use_loraplus=args.use_loraplus,
        loraplus_ratio=args.loraplus_ratio,
        output_name=args.output_path.rstrip("/\\").split("/")[-1].split("\\")[-1],
        trigger_word=args.trigger_word,
        train_args=args,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_boogu_training_task,
        "sft:train": launch_boogu_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args)


if __name__ == "__main__":
    main()
