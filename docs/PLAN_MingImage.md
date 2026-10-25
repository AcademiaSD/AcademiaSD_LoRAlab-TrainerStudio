# Plan — Añadir entrenador Ming-Image a AcademiaSD LoRAlab Trainer Studio

Estado: **Fase 6 en curso** (validación y cierre) · Fases 0–5 completadas · id del entrenador: **`mingimage`**

> Nota técnica de la Fase 0 al final del documento.

## Decisiones tomadas

| # | Decisión | Valor |
| :-- | :--- | :--- |
| 1 | Orden de modelos | **Design (t2i) primero**, **Layer después** |
| 2 | Tipo de LoRA | **Design = LoRA normal (texto→imagen) y LoRA de edición** (pares antes→después): su perfil `generation_edit` acepta las tareas `text-to-image` e `image-edit` (corregido en el repaso del 2026-10-20; ver «Edición»). **Layer** (después) sería la descomposición en capas |
| 3 | Captioner | **El del proyecto, sin tocar nada** (Qwen3-VL-8B compartido) |
| 4 | Conversor NF4 | Ming-Image es arquitectura **custom** (no nativa de `transformers`/`diffusers`) ⇒ **hay que escribir el conversor** |
| 5 | Alpha / RGBA | **Resuelto en Fase 0**: el VAE es RGBA (`input_channels: 4`) ⇒ **se entrena con alpha**; las imágenes opacas entran con alpha = 1.0 |
| 6 | Nombres | `mingimage` → `server_mingimage.py`, `trainer_ui_mingimage.html`, `0_caption_mingimage.py`, `1_pre_cache_mingimage.py`, `2_train_lora_mingimage.py`, `mingimage_model.py`, `Run_LoRAlab-MingImage.bat/.sh`, `assets/mingimage/`, `cached_data_mingimage_*`, `mingimage_lora_output_*` |

## Hallazgo clave (verificado)

El DiT de Ming-Image es **la arquitectura de Z-Image** con dos flags extra:

| Parámetro | Ming-Image | Z-Image_NF4 (local) |
| :--- | :--- | :--- |
| `dim` / `n_layers` / `n_heads` / `n_kv_heads` | 3840 / 30 / 30 / 30 | 3840 / 30 / 30 / 30 |
| `in_channels` / `n_refiner_layers` / `cap_feat_dim` | 16 / 2 / 2560 | 16 / 2 / 2560 |
| `qk_norm` / `rope_theta` / `t_scale` | true / 256 / 1000 | true / 256 / 1000 |
| `all_patch_size` / `all_f_patch_size` / `axes_dims` | [2] / [1] / [32,48,48] | [2] / [1] / [32,48,48] |
| `axes_lens` | **[20480**, 512, 512] | [1536, 512, 512] |
| Extra | `multi_frame_output`, `alignment_padding_mode` | — |

⇒ **`2_train_lora_zimage.py` es la plantilla de entrenamiento.**

## Inventario: piezas de un entrenador y cuánto cambian

| # | Pieza | Fuente a copiar | Cambios reales |
| :-- | :--- | :--- | :--- |
| 1 | `scripts/server_mingimage.py` | `server_zimage.py` | ~25 líneas |
| 2 | `GUI/trainer_ui_mingimage.html` | `trainer_ui_zimage.html` | ~112 líneas |
| 3 | `scripts/0_caption_mingimage.py` | `0_caption_zimage.py` | prompt/formatos |
| 4 | `scripts/1_pre_cache_mingimage.py` | `1_pre_cache_qwen_image21.py` (MLLM) + estructura de Z-Image | text encoder + VAE |
| 5 | `scripts/2_train_lora_mingimage.py` | `2_train_lora_zimage.py` | núcleo |
| 6 | `scripts/mingimage_model.py` | patrón de `qwen_image2512_model.py` | clase DiT + carga NF4 |
| 7 | `code/Run_LoRAlab-MingImage.bat` + `.sh` | los de Z-Image | strings |
| 8 | `assets/mingimage/` | — | 4 imágenes nuevas |
| 9 | `GUI/launcher.json` | — | +1 entrada |
| 10 | `settings/*_settings_mingimage.json` | — | autogenerados (ya en `.gitignore`) |

Genéricos, **sin tocar**: `i18n`/`locales`, `remote_access`, `file_transfer`, `gpu_stats`, `console_stream`, `trainer_stream`, `lora_options`, `sync_pinned`.

## Particularidades a resolver

1. **Arquitectura custom**: clase `DiffusionTransformer` (repo `inclusionAI/Ming-Image`), no está en diffusers ⇒ vendorizar en `mingimage_model.py`.
2. **"Text encoder" = MLLM**: `BailingMoeV2ForCausalLM` (20 capas, hidden 2048, 256 expertos top-8, vocab 157 k) + ViT `Qwen2_5_VisionTransformer` (~34 GB BF16) + `connector` (Qwen2 28 capas, **FP32**, ~6,2 GB) + `mlp` (125 MB) → `cap_feat_dim 2560`. Referencia para el pre-caché: **Qwen-Image 2.1 / 2512** (MLLM + plantilla de chat), no Z-Image.
3. **RGBA**: `in_channels: 16`; el modelo genera alpha. Decisión pendiente (decisión 5).
4. **Design vs Design-Layer**: Layer entrega **FP32** (24,6 GB) y `multi_frame_output: true` + `alignment_padding_mode: learned`. Design es BF16 (~12,3 GB).
5. **Conversor NF4**: en el repo no hay `5_conversor_*.py`. Hay que producirlo. Formato de caché que consume `load_nf4_transformer` (visto en `Z-Image_NF4/transformer/`): `config.json`, `index.json`, `metadata.json`, `others.safetensors`, `weights/`.
6. **Formato de captions/prompts**: las prompts oficiales son **JSON** (p. ej. `assets/t2i_four_seasons_cabin_prompt.json`); hay *prompt enhancement* con `Ling-3.0-flash-VL`. Revisar antes de escribir el captioner.

## Fases

### Fase 0 — Inspección previa (sin escribir código del entrenador)
- Leer el repo oficial `inclusionAI/Ming-Image`: clase `DiffusionTransformer`, firma del forward, camino `cap_feats` (MLLM → connector → mlp), plantilla de chat, `scheduler_config`, formato de prompt JSON.
- Confirmar el VAE (canales, `shift_factor`, `scaling_factor`, tipo).
- Nota técnica con las firmas y el diff exacto contra Z-Image.

### Fase 1 — Esqueleto (aparece en el lanzador) ✅ HECHA

| Fichero | Origen | Estado |
| :--- | :--- | :--- |
| `scripts/server_mingimage.py` | `server_zimage.py` | ✅ 932 líneas, sólo strings/rutas/ids |
| `GUI/trainer_ui_mingimage.html` | `trainer_ui_zimage.html` | ✅ defaults de preview 12 pasos / CFG 1 |
| `code/Run_LoRAlab-MingImage.bat` / `.sh` | los de Z-Image | ✅ |
| `assets/mingimage/` | copia provisional de `assets/zimage/` | ⏳ el usuario pondrá las definitivas |
| `GUI/launcher.json` | — | ✅ entrada `mingimage` tras Z-Image |
| `.gitignore` | — | ✅ `Ming-Image_NF4/` y `Ming-Image-Raw/` |

**Pruebas pasadas**: el servidor arranca en el puerto 5000; `GET /` devuelve la UI (109 KB, título correcto); `/api/settings`, `/api/system-stats`, `/api/dataset-info`, `/api/previews`, `/api/checkpoint-info`, `/api/hf-token` y los assets responden 200; la consola del navegador queda **sin errores**; `GET`/`POST` de guardado escribe `settings/train_settings_mingimage.json` con `./cached_data_mingimage`, `./mingimage_lora_output` y `mingimage_lora.safetensors`; los 72 `getElementById` del JS apuntan a ids existentes.

**Decisión de la Fase 1**: se **retiró el bloque Turbo** de la UI (Ming-Image no tiene LoRA de destilación publicado y el bloque apuntaba a un fichero inexistente). Si aparece uno, se recupera pegando el bloque `TURBO LORA` de `trainer_ui_zimage.html` y su `toggleTurboMode`.

### Fase 2 — Captioner ✅ HECHA

`scripts/0_caption_mingimage.py`, con **el mismo captioner de los demás entrenadores** (Qwen3-VL-8B-Instruct NF4 de `AcademiaSD/Qwen-Image-2.1-NF4-for-LoRA-Training`, ~5,5 GB de VRAM, `Captioner-Qwen3-VL-8B/`). Ningún otro captioner se ha tocado.

- **Formato**: un JSON compacto por imagen, con el esquema oficial (`canvas_settings` + `layers[]`), en lugar de prosa.
- **Trigger**: se antepone a `canvas_settings.image_style` (el campo libre de estilo); no se duplica si ya está.
- **Transparencia**: detecta el canal alpha real y compone la copia del modelo sobre blanco (para que no vea manchas negras). El aviso va **por imagen**: si hay alpha, "el fondo es transparente"; si no, "el lienzo es opaco, no lo describas como transparente". La regla **no** está en el prompt base: en la primera pasada por los datasets de prueba, 6 de 30 imágenes opacas salieron como `"transparent background"` (20 %); con el aviso por imagen, 0 de 30.
- **Pares antes/después**: en `nombre_before` + `nombre_after` se captiona **solo el `_after`** (el `_before` es la entrada de una edición, no el objetivo de un modelo texto→imagen); un `_before` sin pareja se captiona como una imagen más.
- **Prompt**: el `t2i_rewriter_system_prompt.txt` oficial adaptado a *describir* una imagen (no a expandir una petición). 1.965 caracteres, sincronizado carácter a carácter con el `DEFAULT_CAPTION_PROMPT` del HTML. El cierre del literal se localiza con `";` + salto de línea, no con el primer `;` (el prompt lleva puntos y coma dentro).
- **Robustez**: parser tolerante (markdown, JSON cortado, varios objetos), `StopOnLoop` para capas repetidas, deduplicado de capas, coordenadas normalizadas al string oficial `cx: 0.500, cy: 0.500, w: 1.000, h: 1.000` y colores hex en mayúsculas sin repetidos.
- **`aspect_ratio`**: se calcula del **fichero original** (`Fraction.limit_denominator(64)`), no de la copia reducida.
- **"Apply to All" / "Clear Captions"**: `server_mingimage.py` ahora detecta los captions JSON y aplica trigger y texto común dentro de `canvas_settings.image_style` (antes habría antepuesto texto y roto el JSON).

Pruebas: 18/18 del parser, 14/14 del `batch-caption`, y 2 imágenes reales captionadas (una opaca y una con alpha) con el esquema validado. La UI carga el prompt nuevo sin errores de consola.

**Prueba sobre datasets reales** (44 imágenes, ~21 min):

| Dataset | Imágenes | Captions | Capas | Chars | Tiempo |
|---|---|---|---|---|---|
| `Dataset_mrrtsgrl_15` (retratos, trigger `mrrtsgrl`) | 14 | 14 | 53 (3,8 de media) | 1.110 | ~30 s/imagen |
| `Dataset_tosti_edit` (ilustración, trigger `TOSTIOK`) | 60 (30 pares) | 30 (`_after`) | 92 (3,1 de media) | 898 | ~27 s/imagen |

Validación automática de las 44: **0 errores de esquema, 0 de `aspect_ratio` (coincide con el tamaño real del fichero) y 0 discrepancias de transparencia**. Las capas de 7–10 elementos son las lentas (58–66 s); el resto, 14–30 s.

**Límite conocido**: un JSON inválido del modelo hace fallar esa imagen (no escribe `.txt`); se reintenta volviendo a lanzar. Se revisan en el Dataset Manager antes del pre-caché.

### Fase 3 — Pre-caché ✔ HECHA
`1_pre_cache_mingimage.py` con el código oficial vendorizado (`scripts/mingimage_vendor/`, 18 ficheros, MIT). El «text encoder» de Ming no es un encoder de texto: es el MLLM completo, así que se le pide la condición con `generate(image_gen=True, image_gen_only_extract_hidden_states=True)`, que devuelve exactamente lo que consume el pipeline oficial:

| Fichero | Forma | Origen |
| :-- | :-- | :-- |
| `<nombre>_embed.pt` | `(1, 256, 2560)` | `scale_embeds` (proj_in → connector → proj_out); es el `cap_feats` del DiT |
| `<nombre>_embed2.pt` | `(1, N, 3840)` | `directvlm` (hidden states del LLM → `proj_directvlm`); es `cap_feats_2`; N varía por prompt (551–2068 en tosti) |
| `<nombre>_latent.pt` | `(1, 16, h/8, w/8)` | VAE RGBA, `mode()` × 8,0064, sin shift |
| `_neg_embed.pt` / `_neg_embed2.pt` | ceros | el CFG oficial usa `condition_embeds * 0.0` |

- **MLLM en NF4** (bitsandbytes en carga): 10,3 B parámetros, **11,6 GB de VRAM**; el DiT no se carga (`load_image_gen_diffusion=False`) y el connector va en bf16. El VAE se carga después, nunca a la vez.
- **Sin `mlp` en el entrenamiento**: `mlp/config.json` trae `use_identity_mlp: true`, `text_encoder_norm: false`, `connector_norm: false` → el `cap_feats` cacheado entra tal cual al `cap_embedder` del DiT.
- **`_before` ignorado**: 30 pares → 30 muestras (misma regla que el captioner).
- **`--prompt-only <caché> <prompt> <cuda|cpu>`**: igual que Z-Image; el servidor ya mete el trigger antes de llamar. En CPU (durante el entrenamiento) cae a bf16 si el 4 bits no está disponible.
- **Reanudable**: los latentes ya cacheados con el mismo bucket y más nuevos que su imagen se saltan.
- **Validación**: 4 imágenes de prueba con round-trip del VAE → **PSNR 31,1–36,5 dB RGB / 45,5–56,7 dB alfa**; después las 30 de `Dataset_tosti_edit` (0 errores de forma, 0 NaN).

**Parches al código oficial para `transformers` 5.18** (el repo pide 4.57.1): `is_torch_fx_available` (ya no existe), `torch_dtype`→`dtype` en `load_image_gen_modules`, el setter de `add_bos_token` del tokenizer (ahora llama a `update_post_processor` antes de inicializar) y la entrada `"default"` de `ROPE_INIT_FUNCTIONS` (v5 la quitó; se repone la fórmula clásica). Todo comentado en el propio fichero vendorizado.

**Descarga**: 37,8 GB en `Ming-Image_Raw/` (mllm + connector + mlp + vae + scheduler + `transformer/config.json`); los pesos del DiT (11,46 GB) se añadieron en la Fase 5.

### Fase 4 — Entrenamiento — HECHA (validación pendiente en Fase 6)
`mingimage_model.py` (DiT vendorizado + `load_nf4_transformer`) y `2_train_lora_mingimage.py` desde la plantilla de Z-Image, cambiando solo: carga NF4, `predict()` (dos condiciones), `denoise()`/CFG de previews, VAE RGBA y layout de export/metadatos.
El DiT recibe **dos** condiciones (`cap_feats` 256×2560 y `cap_feats_2` N×3840, ambas cacheadas) y **no** lleva mlp (identidad); el negativo de CFG son ceros con la **misma longitud que el positivo** (`zeros_like`, porque N varía por prompt). `cap_feats_2` tiene longitud variable por muestra → el lote se pasa como lista, no como tensor apilado.
La preview sigue el pipeline oficial: `pred = pos + cfg·(pos−neg)` solo con cfg > 1, y el scheduler recibe `−pred` (la salida del DiT es x0−ruido). El VAE decodifica RGBA y la preview se guarda con alfa.
Se mantienen idénticos: resume exacto, live settings, `lora_options` (rsLoRA/LoRA+/LoKr), checkpoints, stop por señal, tiled VAE, barra de progreso y el formato `Step N/Total`. **No hay Turbo LoRA**: Ming no tiene destilado y el modelo ya se usa a 12 pasos/CFG 1.
Archivos: `MingImage_LoRA_step_N.safetensors` y `MingImage_FINAL_LoRA.safetensors` (el servidor ya los espera).

### Fase 5 — Modelo NF4 en HF — HECHA (paquete generado; publicación pendiente)
`scripts/convert_nf4_mingimage.py` (`--only transformer|mllm|extras`, reanudable, `--overwrite`). Genera `Ming-Image_NF4/` con `mllm/` prequantizado, `transformer/` NF4, `connector/` bf16, `mlp/`, `vae/`, `scheduler/` y `config.json` raíz, listo para publicar en `AcademiaSD/Ming-Image-NF4-for-LoRA-Training`.

**Resultados (verificados)**
- **DiT** (`Ming-Image_NF4/transformer/`): `config.json`, `index.json`, `metadata.json`, `others.safetensors` y `weights/` (276 ficheros, **3,38 GB**). **238 capas NF4 + 38 bf16 + 205 otros tensores** (mismos recuentos que Z-Image). Cuantización `quantize_4bit(blocksize=64, nf4, double quant)`; el estado se guarda con `qs.as_dict(packed=True)` (prefijo doble `quant_state.quant_state.bitsandbytes__nf4`, que el loader recorta). Frente al BF16 de referencia: **cos 0,974114**, RMSE 0,19082, diff máx 0,6875, VRAM 3,39 GB, salida (16,1,32,32) correcta.
- **MLLM** (`mllm/`): 15.283 pesos en 2 shards, **9,35 GB**, guardado con `save_pretrained(max_shard_size="5GB")`. Parche para transformers 5.18: el LM declara `_tied_weights_keys` en formato lista (4.x) y `_get_tied_weight_keys` espera un dict; el conversor normaliza los módulos a `{k: k}` antes de guardar.
- **Extras**: `connector/` a **bf16** (3,09 GB; en disco está fp32 y el loader siempre lo carga en bf16), `mlp/` 0,12 GB, `vae/` 0,25 GB, `scheduler/` y `config.json` raíz copiados.
- **Total: 16,2 GB** frente a ~49 GB del checkpoint BF16 (Raw).
- **Windows**: repetir `safe_open` del mismo shard ~205 veces provocaba `0xC0000005`; el conversor agrupa las claves por shard y abre cada uno una sola vez.
- **Carga del paquete**: el MLLM prequantizado produce embeddings **idénticos bit a bit** a los del Raw cuantizado al vuelo (`maxdiff 0.0`; embed 1×256×2560, embed2 1×80×3840). La configuración `quantization_config` del `config.json` hace que transformers lo cargue sin recuantizar.

### Fase 6 — Validación y cierre — EN CURSO
pre-caché → 30 pasos → preview → stop/resume exacto → export a `models/loras` → cargar en ComfyUI. Documentar en `README.md` (tabla + sección) y `docs/README_MingImage.md`.

**Hecho (repaso del 2026-10-20)**
- Entrenamiento real `morriv11_ming` (14 imágenes, 512², rank 8): 960/1000 pasos, reanudado desde el 500, ~1,3 s/it; previews a 1024 coherentes con la identidad desde el paso 500.
- **El pre-caché usaba aún `Ming-Image_Raw` + `inclusionAI/Ming-Image-0.1-Design`** (38 GB, y la carpeta ya no existe): ahora usa el paquete publicado `Ming-Image_NF4` / `AcademiaSD/Ming-Image-NF4-for-LoRA-Training`, igual que el trainer. Detecta el MLLM prequantizado y no lo recuantiza. En GPU, embeddings **idénticos bit a bit** a la caché hecha con el Raw (`maxdiff 0.0`).
- **Prompt de preview en caliente** (Save JSON con el entrenamiento en marcha → `--prompt-only … cpu`): funcionaba solo con el Raw; ahora, con el paquete NF4 en CPU y fp32: **16 s por prompt** (71 s en bf16), ~16 GB de RAM. No sale idéntico al de la GPU (cos 0,94 en `cap_feats`, 0,98 en `cap_feats_2`; el MoE elige otros expertos con cualquier cambio de precisión, tampoco el autocast bf16 en CPU lo arregla: 0,95 y 105 s). Solo afecta a las previews; relanzar el Pre-Cache lo recodifica en GPU.
- `with_trigger` del pre-caché mete el trigger en `canvas_settings.image_style` si el caption es JSON (antes lo anteponía al `{` y el MLLM recibía un JSON roto).
- Ajustes en caliente del trainer: mismas claves que Z-Image menos Turbo (`total_steps`, `save_every`, `lr`, `max_grad_norm`, `preview_*`, `seed`).
- Traducciones que faltaban (8 cadenas del pre-caché y del captioner) en los 7 idiomas; `Ming-Image_NF4_HF/` en `.gitignore`.

**Edición (image-edit) — HECHA**

Cómo la hace el código oficial (`infer.py` + `modeling_bailingmm2.py` + `diffusion/transformer.py`): el mensaje al MLLM es `[imagen antes, instrucción en texto libre]` (no el JSON), así que el ViT mete el antes en las dos condiciones; el DiT recibe además el latente VAE del antes como `ref_x`, que `patchify_and_embed` concatena al ruido **como segundo fotograma** (`torch.cat([image, image_ref], dim=1)`), y la salida útil es el primero. Bucket oficial: solo 1024; muestreo por defecto el mismo que texto→imagen (12 pasos, CFG 1).

- **Captioner** (`caption_mode: edit`): Qwen3-VL ve antes y después y escribe UNA instrucción (mismo prompt que Qwen-Image 2512) en `nombre.txt`; el JSON de texto→imagen sigue en `nombre_after.txt`, así que un dataset sirve para los dos tipos.
- **Pre-caché** (`lora_type: edit`): muestra = par; `nombre_latent.pt` (después) y `nombre_ctrl.pt` (antes) con el bucket del después; el MLLM codifica el antes recortado a ese bucket + la instrucción. La extracción declara `text-to-image` porque la validación de `image-edit` pide el tensor de referencia del DiT, que aquí va aparte. Preview manual con su imagen de antes (`_custom_ctrl.pt`), también en caliente en CPU.
- **Trainer**: `ref_x` = `_ctrl`, salida = primer fotograma; preview con la referencia y a su tamaño (ignora Preview Size); metadato `lora_type: edit`; error si la caché mezcla pares e imágenes sueltas.
- **Servidor / GUI**: selector LoRA Type, imagen de antes de la preview (Browse / Upload), modo del captioner, y el Dataset Manager enseña el caption del tipo elegido (`nombre.txt` o `nombre_after.txt`); en edición borrar una mitad borra el par.

**Verificado**
- Condiciones con imagen **idénticas bit a bit** a las del camino oficial completo (`image-edit` + `image_gen_ref_images`): `maxdiff 0.0`; la imagen añade ~580 tokens a `cap_feats_2`.
- DiT base sin LoRA: el pipeline oficial a 1024 y 512 y el `denoise()` del trainer a 512 editan bien (conservan cara, pelo, ropa y cambian el estilo).
- Que `ref_x` llega al DiT en eval, en train con gradient checkpointing y a través de PEFT.
- **Control (blanco y negro, 8 pares alineados, 512, LR 1e-4)**: aprende la edición en 50 pasos y la aplica a una imagen de fuera del dataset sin degradarse a 100 (pérdida 0,16).
- **TOSTIOK (8 pares de `Dataset_tosti_edit`, 512)**: se degrada (imágenes planas, caras borradas) a LR 4e-4 en 40 pasos y a 1e-4 hacia el paso 100, aunque la mecánica es la misma que en el control. Sospecha: pocos pares, estilización fuerte, pares no alineados píxel a píxel y 512 fuera de la distribución de edición. Pendiente de probar con los 30 pares a 1024.
- **1024**: 12 s/it y pico de **13,2 GB** de VRAM (RTX 5080, batch 1); a 512, 2,4 s/it.

**Pendiente**
- Entrenar el TOSTIOK de edición completo (30 pares) a 1024 con LR 1e-4 y ver si la degradación desaparece.
- Avisos del código vendorizado (2026-10-20): `torch.cuda.amp.autocast` → `torch.amp.autocast("cuda")`, `use_return_dict` → `return_dict` y `torch_dtype` → `dtype` en el connector, todos equivalentes: recodificando 3 pares de una caché de edición hecha con el código anterior, embeddings **idénticos bit a bit** (`maxdiff 0.0`) y la pre-caché ya no muestra ningún aviso.
- El LLM usa la API antigua de máscaras (`transformers.modeling_attn_mask_utils`); su aviso solo se silencia en la pre-caché. Si una versión de transformers la quita, la pre-caché fallará al arrancar: habrá que migrar a `masking_utils`.
- Validar en la GUI: Save JSON en caliente (LR / preview / prompt manual) y ver los `[LIVE]` en la consola.
- Export a `models/loras` y carga fuera del trainer (ComfyUI no tiene el DiT de Ming de serie).
- `README.md` (tabla + sección) y `docs/README_MingImage.md`.
- La ficha de HF dice que 512 y 768 están fuera de distribución, pero la GUI propone 768 por defecto y el run a 512 sale bien: decidir y alinear.
- `scripts/_test_*.py` (22 pruebas temporales de previews): borrar o mover a `experiments/` antes del commit.

---

# Fase 0 — Nota técnica (resultados)

Fuentes: repo `github.com/inclusionAI/Ming-Image` (MIT), y los repos HF `inclusionAI/Ming-Image-0.1-Design` / `-Layer` (MIT).

## 1. VAE — resuelve la duda del alpha (decisión 5)

`vae/config.json`:

```json
{ "_class_name": "AutoencoderKLQwenImage", "base_dim": 96, "dim_mult": [1,2,4,4],
  "input_channels": 4, "z_dim": 16, "scaling_factor": 8.0064, "shift_factor": 0.0,
  "latents_mean": null, "latents_std": null }
```

- **`input_channels: 4`** → R, G, B **y alpha**. El canal alfa es real, es la razón de ser del modelo (fondos transparentes).
- `z_dim: 16` ⇔ `in_channels: 16` del DiT. **Escala** `8.0064`, **shift** `0.0`, `latents_mean/std` = `null` ⇒ latente = `z * 8.0064`, sin más (más simple que Krea 2).
- Es **la misma clase que ya usa Krea 2** (`AutoencoderKLQwenImage`) ⇒ el código de VAE del pre-caché se reutiliza tal cual, sólo añadiendo el 4º canal.
- `vae_sample_mode: "argmax"` ⇒ `.latent_dist.mode()`, igual que en Z-Image.

## 2. DiT — es Z-Image con dos flags

`diffusion/transformer.py`:

```python
class DiffusionTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    def forward(self, x: List[Tensor], t, cap_feats: List[Tensor],
                patch_size=2, f_patch_size=1, ref_x=None, cap_feats_2=None, return_dict=True):
```

- **Hereda de `PeftAdapterMixin`** ⇒ LoRA por PEFT está previsto en la propia arquitectura.
- Módulos: `t_embedder`, `all_x_embedder["2-1"]`, `cap_embedder`, `noise_refiner`, `context_refiner`, `layers` (30), `all_final_layer["2-1"]`, `rope_embedder` (+ `x_pad_token` / `cap_pad_token` en modo *learned*).
- **Mismos nombres que `ZImageTransformer2DModel`** ⇒ el `is_target()` de Z-Image (`layers.`, `noise_refiner.`, `context_refiner.`, excluyendo la modulación `adaLN`) se reutiliza **tal cual**.
- `unpatchify` y `patchify_and_embed` implementan `multi_frame_output` y `alignment_padding_mode`.

## 3. Convención de tiempo y de pérdida — idénticas a Z-Image

| | Ming-Image (pipeline) | Z-Image (tu trainer) |
| :--- | :--- | :--- |
| Tiempo que recibe el DiT | `timestep = (1000 - t)/1000`, y dentro `t = t * t_scale` | `t = 1.0 - sigma` |
| Salida del modelo | `noise_pred = -model_out` y `scheduler.step(noise_pred, …)` | `scheduler.step(-pred, …)` |
| Target de entrenamiento | — | `target = latents - noise`, `MSE(model_out, target)` |

⇒ **La loss de `2_train_lora_zimage.py` se reutiliza directamente.** Queda como verificación empírica obligatoria (smoke test de 1 paso con un `_latent.pt` real).

**Scheduler**: `FlowMatchEulerDiscreteScheduler`, `shift: 6.0`, `use_dynamic_shifting: false`. Precedente de Z-Image: **entrenar con shift 3.0** e inferir con 6.0 ⇒ empezar con 3.0.

## 4. Text encoder (MLLM) y cap_feats

`ImageGenerationPipeline._encode_prompt`:

```python
prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
embeds = text_encoder(input_ids=..., attention_mask=..., output_hidden_states=True).hidden_states[-2]
prompt_embeds = [embeds[i][mask[i]] for i in range(len(embeds))]   # List[[n_tokens, 2560]]
```

- **`hidden_states[-2]`**, igual que Z-Image. Cada prompt → lista de tokens de 2560 dims.
- La cadena MLLM → `connector` → `mlp` vive dentro de su modelo custom ⇒ **en el pre-caché no se reimplementa: se instancia su pipeline y se guarda lo que devuelve `encode_prompt`** (patrón ya usado en Qwen-Image 2.1).
- `mlp/config.json`: `use_identity_mlp: true`, `diffusion_c_input_dim: 2560`, `selected_hidden_states_layers: [5, 12, 20]`.

**Carga del MLLM** (`infer.py`):
- `BailingMM2NativeForConditionalGeneration.from_pretrained(dir, torch_dtype=…, load_image_gen=True, device_map=plan)`
- ⚠️ **Solo `eager` o `flash_attention_2`: `sdpa` falla al cargar.** En Windows ⇒ `eager`.
- Reutilizar su planificador de offload `mllm_device_map.build_mllm_device_plan` / `allocate_mllm_layer_counts` (reserva la GPU 0 para los módulos fijos y reparte el MLLM entre GPU y RAM).
- Preprocesado de imagen: `min_pixels = max_pixels = 451584` (672²), patch 14, merge 2.

## 5. Perfil de checkpoint (archivo que hay que publicar en HF)

De `examples/profiles/*.json`. El `--task` y estos flags los lee `inference_profile.load_checkpoint_capabilities`:

| Perfil | Modelo | `alignment_padding_mode` | `multi_frame_output` | `vae_input_channels` | `vae_sample_mode` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `generation_edit` | **Design** | `zero_masked` | `false` | 4 | `argmax` |
| `layer_decompose` | **Layer** | `learned` | `true` | 4 | `argmax` |

⇒ El repo NF4 de Design debe incluir `generation_edit.json` (o equivalente).

## 6. Formato de los captions (prompt del captioner)

De `assets/t2i_rewriter_system_prompt.txt` — **JSON estilo Figma**, dos claves raíz:

```json
{ "canvas_settings": { "aspect_ratio": "...", "ambient_lighting": "...", "image_style": "..." },
  "layers": [ { "description": "...", "coordinates": "cx: 0.500, cy: 0.500, w: 1.000, h: 1.000",
                "hierarchy_and_relation": "...", "color_specs": ["#FFFFFF"] } ] }
```

`coordinates` es **un string**, no un objeto, con valores normalizados 0–1. `color_specs` es un array de hex. ⇒ Ese es el prompt de `0_caption_mingimage.py` (la infraestructura del captioner no se toca, decisión 3).

## 7. Código a vendorizar (MIT)

Todo el stack es custom; hay que copiarlo al repo en `scripts/mingimage_vendor/`:

`diffusion/{__init__,transformer,pipeline,padding,generator,autoencoder_kl_qwenimage}.py`,
`modeling_bailingmm2.py`, `configuration_bailingmm2.py`, `modeling_bailing_moe_v2.py`,
`configuration_bailing_moe_v2.py`, `qwen2_5_vit.py`, `tokenization_bailing.py`,
`image_processing_bailingmm2.py`, `processing_bailingmm2.py`, `bailingmm_utils.py`,
`modeling_utils.py`, `mllm_device_map.py`, `inference_profile.py`.

## 8. Riesgos detectados

| Riesgo | Impacto | Mitigación |
| :--- | :--- | :--- |
| **NF4 del MLLM** (MoE custom, 256 expertos, ~34 GB BF16) | Es el grueso de la descarga y del conversor | 1ª versión: MLLM con `load_in_4bit` + `device_map` de `transformers` (sin conversor propio); optimización posterior: NF4 publicado en HF |
| `connector` en **FP32** (6,2 GB) | Descarga innecesaria | Reconvertir a BF16 (3,1 GB) |
| `sdpa` no soportado por el LLM | Falla al cargar | Forzar `eager` (Windows) / `flash_attention_2` (Linux con Triton) |
| Verificar la pérdida | Un target incorrecto entrena mal sin dar error | Smoke test de 1 paso comparando contra la denoise del pipeline oficial |
| **34 GB de MLLM en BF16** vs 16–32 GB de RAM | El pre-caché no cabe | NF4 / INT8 (8,5 / 17 GB) para el MLLM |

