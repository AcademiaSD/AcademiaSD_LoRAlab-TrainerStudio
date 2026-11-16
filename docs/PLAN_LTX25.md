# Estudio — Entrenador LTX-2.5 para AcademiaSD LoRAlab Trainer Studio

Estado: **Fase 0 completada · Fase 1 casi completa** (2026-11-16) · id: **`ltx25`** · rama `ltx25`

> **Fase 1 (2026-11-16):** conversor NF4 (`tools/ltx25`), block swap (`scripts/ltx25_model.py`), pre-cache,
> trainer, previews e interfaz con **VRAM Profile**. Medido con perfil de 8 GB en una RTX 5080: pre-cache 6.0 GB,
> entrenamiento 6.65 GB (~6.8 s/it), preview 6.84 GiB. Corregido además el `sigma` en escala 0–1 heredado de
> LTX 2.3 (el pipeline oficial lo pasa en 0–1000) y añadido el muestreo de timesteps de `ltx-trainer`.
> Detalle técnico en [README_LTX25.md](README_LTX25.md). Pendiente: validar un LoRA real en ComfyUI y DoRA con swap.

> Objetivo: un entrenador nuevo en el launcher que reúna lo que hoy está repartido entre tres:
> personaje/estilo desde imágenes (LTX 2.3), vídeo y vídeo+audio (MiniMax-H3), previews en vídeo (Wan 2.2),
> y si es posible LoRAs de edición (IC-LoRA de LTX).
>
> **Requisito: que funcione en 8 GB de VRAM**, con la gestión de memoria de MiniMax-H3 como modelo
> (block swap con perfiles, `nf4_cpu_home`, park en RAM/disco, checkpoint reentrante).

---

## 1. Veredicto

**Viable, y más barato de lo previsto.** La Fase 0 confirma que LTX-2.5 es LTX-2.3 con el transformer
re-entrenado y un codificador de texto nuevo; **todo lo demás es idéntico bit a bit** (VAE de vídeo,
VAE de audio, vocoder, estadísticas de latentes, formato de claves LoRA). La mitad del trabajo de
LTX 2.3 se reutiliza tal cual.

**Corrección importante:** el trainer de LTX 2.3 **no entrena LTX-2.5**. Entrena sobre los pesos de 2.3
y sus LoRAs se cargan en 2.5 (mismas claves y formas), pero los pesos del transformer se han movido un
16–31 % y los connectors de texto son otros. La tarjeta "LTX 2.3 (2.5)" y las notas "also works with
LTX-2.5" deben decir "sus LoRAs suelen cargar en 2.5", no "entrena 2.5".

| Modo | Viabilidad | Base principal | Comentario |
| :--- | :--- | :--- | :--- |
| A. Personaje / estilo desde imágenes | **Alta** | LTX 2.3 | Mismo flujo; cambian pre-cache de texto y pesos |
| B. Vídeo (movimiento, efectos, cámara) | **Alta** | MiniMax-H3 + Wan 2.2 | Clips, preparación de fps/fotogramas, block swap |
| C. Vídeo + audio | **Media-alta** | MiniMax-H3 | VAE de audio ya conocido; abrir la rama de audio al LoRA |
| D. Edición (IC-LoRA) | **Media** | nuevo + trainers de edición | `diffusers` tiene `pipeline_ltx2_ic_lora.py`; portar el formato de `ltx-trainer` |
| E. Solo audio | **Baja, no recomendado** | — | En MiniMax-H3 degrada la rama de vídeo |

---

## 2. Fase 0 — qué se ha medido y cómo

El repo oficial `Lightricks/LTX-2.5-Diffusers` tiene gate (`auto`), así que no se pudo bajar su
`config.json`. En su lugar se leyeron **solo las cabeceras** de los `.safetensors` (peticiones HTTP con
`Range`, sin descargar pesos) del checkpoint original de 2.5 (espejo público `comfyicu/LTX-2.5`, mismos
ficheros y tamaños que `Lightricks/LTX-2.5`) y de `Lightricks/LTX-2.3` (sin gate). Las cabeceras traen
la configuración completa en `__metadata__.config`. Para comparar pesos se descargaron tensores
sueltos por rango de bytes.

Cabeceras y scripts guardados en `F:\Claude_temp\ltx25_cfg\` (`read_headers.py`, `read_tensor.py`, `headers\*.json`).

### 2.1 Transformer (22B)

`model_version 2.5.0`, `gemma_version gemma4-12b-ltx-v1`. Configuración comparada clave a clave con 2.3:

| Clave | 2.3 | 2.5 |
| :--- | :--- | :--- |
| `ff_bias` | (true) | **false** — desaparecen `ff.net.*.bias` y `ff.net.*.proj.bias` del vídeo |
| `use_keyframes_abs_pos_embedding` | — | **true** — un tensor nuevo `keyframes_abs_pos_embedding` (solo DFR/keyframes) |
| todo lo demás | igual | igual: 48 bloques, 32 cabezas × 128, cross-attn 4096, audio 32 × 64, `caption_channels` 3840, rope `split`, gated attention, cross-attention AdaLN |

- **Tensores:** 4 349 en 2.5 frente a 4 444 en 2.3 (la diferencia son los bias del FF). **Ninguna forma cambia.**
- **Pesos:** 2.5 es un re-entrenamiento de 2.3, no un modelo nuevo:

| Tensor | Diferencia relativa | Coseno |
| :--- | ---: | ---: |
| `transformer_blocks.0.attn1.to_q` | 18.7 % | 0.986 |
| `transformer_blocks.24.attn2.to_k` | 18.9 % | 0.985 |
| `transformer_blocks.47.ff.net.0.proj` | 31.4 % | 0.962 |
| `transformer_blocks.10.audio_attn1.to_v` | 15.9 % | 0.988 |
| `patchify_proj` | 11.0 % | 0.995 |
| `video_embeddings_connector...to_q` | 136 % | **0.000** — connectors re-entrenados desde cero |

- **Claves LoRA:** el LoRA de destilación oficial de 2.3 (rank 384) y el de 2.5 (rank 450) tienen las mismas
  124 familias de claves (`diffusion_model.transformer_blocks.N.{attn1,attn2,audio_attn1,audio_attn2,audio_to_video_attn,video_to_audio_attn}.{to_q,to_k,to_v,to_out.0,to_gate_logits}`, `ff.net.*`, `audio_ff.net.*`).
  **El exportador a ComfyUI de LTX 2.3 sirve sin cambios.**
- `diffusers` 0.42.0.dev0 instalado ya soporta `ff_bias=False` y `use_keyframes_abs_pos_embedding`.

### 2.2 VAE de vídeo, VAE de audio y vocoder — **idénticos a 2.3**

- La configuración del VAE convencional (`ltx-2.5-video-vae-conv`) y del encoder coincide con 2.3 (32× espacial, 8× temporal, 128 canales).
- Comparados tensor a tensor: `per_channel_statistics` (media y std), `encoder.conv_in`, `encoder.conv_out`,
  `decoder.conv_in`, `audio_vae.per_channel_statistics`, `audio_vae.encoder.conv_in`, `vocoder.conv_pre`:
  **diferencia máxima 0.0**.
- El VAE "por defecto" de 2.5 (`ltx-2.5-video-vae`) cambia **solo el decoder** por uno de difusión
  (`NADiffusionDecoder`, 310 tensores). El encoder es el mismo.

**Consecuencias:** los latentes de vídeo y audio son los mismos que en 2.3. Se pueden reutilizar las
carpetas `vae/`, `audio_vae/` y `vocoder/` de `Models/LTX23-NF4` (sin descarga nueva), el código de
pre-cache de latentes de LTX 2.3 y su decodificación para previews (decoder convencional, más barato).

### 2.3 Codificador de texto — **lo único realmente nuevo**

- `Gemma4UnifiedForConditionalGeneration` (`gemma4_unified`), 48 capas, hidden 3840, 16 cabezas
  (8 KV), sliding + full attention cada 6 capas, vocabulario 262 144. Incluye torres de visión y audio
  que **no hacen falta** para texto.
- El fichero `gemma4-12b-with-proj-ltx-2.5` trae también `text_embedding_projection.{video,audio}_aggregate_embed`
  con entrada **188 160 = 3840 × 49**: igual que en 2.3, se apilan los 49 estados ocultos (embeddings + 48 capas)
  con `per_token_rms_norm`. El mecanismo de `diffusers` (`_get_gemma_prompt_embeds` + `LTX2TextConnectors`) es el mismo.
- `transformers` 5.17.0 instalado ya tiene `gemma4_unified`.
- **Los cachés de texto de 2.3 no valen** (otro encoder y otros connectors).

### 2.4 Licencia

`LTX-2.x Community License Agreement` del 11-08-2026 (texto distinto del de 2.3, del 05-01-2026). Puntos relevantes:

- Uso libre para particulares y empresas de **menos de 10 M$** de ingresos anuales; por encima, licencia comercial de pago.
- **Redistribuir derivados está permitido** (sección 3) con condiciones: incluir copia completa del acuerdo,
  trasladar las restricciones de uso (sección 4 y Anexo A), marcar los ficheros modificados y conservar avisos.
  Un NF4 es un derivado: el repo `AcademiaSD/LTX25-NF4-for-LoRA-Training` debe llevar el `LICENSE` y una nota de modificación (como el de 2.3).
- Sección 3.5: transferir un derivado (LoRAs incluidos) no da derechos comerciales al receptor. Conviene decirlo en el README.
- El gate de Hugging Face exige aceptar los términos con la cuenta del token: **solo lo necesita quien convierte a NF4**
  (una vez); los usuarios bajan el NF4 de AcademiaSD.

*(Resumen informativo, no asesoramiento legal.)*

---

## 3. Presupuesto de 8 GB (gestión de memoria al estilo MiniMax-H3)

Medido sobre la cabecera del transformer de 2.5:

| Pieza | Parámetros | Tamaño |
| :--- | ---: | ---: |
| 1 bloque (Linear) | 0.386 B | **~0.21 GB en NF4** (0.77 GB en BF16) |
| 48 bloques | 18.5 B | ~9.8 GB NF4 |
| Resto del transformer (proj_in/out, AdaLN, tablas) | 0.43 B | ~0.85 GB BF16 |
| Connectors (+ aggregate embeds) | 2.0 B | ~4 GB BF16 — **solo en pre-cache**, no en entrenamiento |

Comparado con MiniMax-H3 (bloque NF4 de 0.333 GB, 50 bloques, suelo de 5.41 GB), LTX-2.5 lo tiene **más fácil**:
bloques un 38 % más pequeños y un modelo un 30 % menor.

**Suelo estimado con 0 bloques residentes** (a medir en la Fase 1):

```text
~0.9 GB  resto del transformer
~3.0 GB  contexto CUDA + workspaces de cuBLAS + búfer de swap (2 bloques)   ← en H3 fueron 3.49
~0.2 GB  LoRA rank 16 FP32 + estado de AdamW
──────────
~4.1 GB  + activaciones
```

Tokens: una imagen de 768×768 son 576 tokens; un clip de 512×512 y 121 fotogramas, 4 096 tokens de vídeo
y ~125 de audio. Con checkpoint reentrante por bloque, la entrada guardada de cada bloque es
`tokens × 4096 × 2 B` (32 MB a 4 096 tokens, ×48 = 1.6 GB), que el *activation offload* de Wan 2.2 manda a RAM.

| Tarjeta | Imágenes 768² | Clip 512² / 121 f | Previsión |
| :--- | :--- | :--- | :--- |
| 8 GB | ~10–14 bloques residentes | ~2–6 bloques + activation offload | entra; lento |
| 12 GB | ~30 bloques | ~20 bloques | cómodo |
| 16 GB | 48 (sin swap) | ~38 bloques | rápido |

**Coste del swap:** a ~2.8 GB/s por PCIe, cada bloque aparcado cuesta ~0.075 s por subida; con forward +
recomputación + backward, ~0.12–0.15 s/it por bloque (H3 midió 0.1175 con bloques mayores). En 8 GB con
~40 bloques aparcados son ~5–6 s/it extra: un LoRA de imágenes de 800 pasos, ~1.5–2 h.

**Pre-cache en 8 GB:** Gemma 4 12B en NF4 ≈ 6 GB de capas + 1 B de `embed_tokens` (2 GB BF16, se deja en
CPU como lookup). Las torres de visión y audio no se cargan. Los connectors (4 GB) corren después,
con Gemma ya descargado, o en NF4. El VAE encoder en tiles. Con eso el pre-cache cabe en 8 GB.

**Lo que se porta de MiniMax-H3:** perfiles 8/10/12/16/24/32 GB calculados a partir del caché,
`nf4_cpu_home`, park RAM/disco/auto, tope duro con `set_per_process_memory_fraction`, checkpoint
reentrante (bitsandbytes no libera su peso NF4 con el no-reentrante), RNG por paso y `train_log.txt`.
**De Wan 2.2:** checkpoint en dos mitades por bloque y activation offload, tope del allocator para que
Windows no derrame VRAM a RAM sin avisar, `expandable_segments`.

> El NF4 de LTX 2.3 guarda **un fichero por Linear** (1 660 ficheros en `weights/`), lo que encaja bien
> con el block swap. Para 2.5 se propone el mismo formato, agrupable por bloque.

---

## 4. Diseño propuesto: un trainer `ltx25` con selector de modo

Un solo trainer en el launcher, con **LoRA Type** en el panel de Pre-Cache:

- **Image** (personaje / estilo): imágenes + `.txt`, igual que LTX 2.3.
- **Video**: clips (y opcionalmente imágenes mezcladas). Audio en silencio, LoRA solo en vídeo.
- **Video + Audio**: clips con pista; LoRA también en `audio_attn*`, `audio_ff`, `audio_to_video_attn`, `video_to_audio_attn`; pérdida de audio ponderable.
- **Edit (IC-LoRA)**: pares `nombre_before.mp4|png` / `nombre_after.mp4|png` + `nombre.txt`. Fase posterior.

### Ficheros

```text
scripts/0_caption_LTX25.py        captioner Qwen3-VL-8B NF4 (compartido), prompts por tipo: imagen / vídeo / vídeo+audio
scripts/1_pre_cache_LTX25.py      Gemma 4 NF4 + connectors 2.5; VAE de vídeo y audio de 2.3 (idénticos)
scripts/2_train_lora_LTX25.py     bucle de LTX 2.3 + block swap de H3 + preview mp4 con audio
scripts/ltx25_model.py            piezas comunes: carga NF4 por bloque, swap, rejilla 8n+1, read/write video, mux de audio
scripts/server_ltx25.py           Flask (base: server_ltx23 + endpoints de clips/audio de server_minimaxh3)
GUI/trainer_ui_ltx25.html         base: trainer_ui_minimaxh3 (dataset con vídeo/audio, perfiles VRAM) + <video> de Wan
code/Run_LoRAlab-LTX25.bat / .sh
GUI/launcher.json                 tarjeta "LTX 2.5": "Image, video and audio LoRAs"
assets/ltx25/portada.jpg
GUI/locales/*.json                cadenas nuevas en los 8 idiomas
tools/Run_Conversor_LTX25.bat     conversor a NF4 → repo AcademiaSD/LTX25-NF4-for-LoRA-Training (con LICENSE)
docs/README_LTX25.md, README.md   guía, tabla de VRAM, changelog; corregir la nota "(2.5)" de LTX 2.3
```

### Qué se reutiliza de cada uno

| Pieza | De dónde |
| :--- | :--- |
| Carga NF4 vacía + relleno, targets LoRA, rsLoRA/LoRA+/DoRA, precisión FP16/BF16, flow matching con shift por longitud, **export a ComfyUI (mismas claves)**, pre-cache de latentes, VAE/audio VAE/vocoder | LTX 2.3 |
| Lectura de clips, **Prepare clips** (fps + rejilla sin recortar el final, `_originals/`), captioner por tipo, dataset con vídeo y audio | MiniMax-H3 |
| **Block swap con perfiles de VRAM**, `nf4_cpu_home`, park RAM/disco/auto, checkpoint reentrante, RNG por paso, `train_log.txt` | MiniMax-H3 |
| Preview **.mp4** (ffmpeg, H.264 CRF 18, fallback `.webp`), mínimo de pasos y tamaño para vídeo, VAE tras sacar el transformer, tope del allocator, checkpoint en dos mitades + activation offload | Wan 2.2 |
| Pares before/after y su UI | Qwen-Image / Boogu / MiniMax-H3 Image (edit) |

### Novedad: preview de vídeo **con sonido**

Ningún trainer lo hace hoy (H3 previsualiza 1 fotograma, Wan da mp4 mudo). En LTX es barato: el sampler
ya denoisa vídeo y audio juntos; se decodifica el audio con `audio_vae` + `vocoder` (los de 2.3) y se
mezcla en el mp4 con ffmpeg (`-c:a aac`). En 8 GB la preview de vídeo sale con el transformer fuera de la
GPU antes de decodificar, como en Wan.

---

## 5. Fases

| Fase | Contenido | Criterio de "hecho" |
| :--- | :--- | :--- |
| ~~0. Verificación~~ | Hecha: ver sección 2 | ✅ |
| **1. Modelo NF4 + modo Image en 8 GB** | Conversor NF4 (transformer dev + Gemma 4 solo texto + connectors), pre-cache de texto, trainer de imágenes con block swap y perfiles desde el principio, preview de 1 fotograma, export probado en ComfyUI con LTX-2.5 (dev y distilled) | LoRA de personaje con buen parecido; pico medido ≤ 7.3 GB en perfil 8 GB |
| **2. Modo Video** | Clips, Prepare clips (rejilla 8n+1, fps nativos), buckets por forma latente, activation offload, preview `.mp4` | LoRA de movimiento reproducible; tabla de VRAM medida 8/12/16 GB |
| **3. Modo Video + Audio** | Pre-cache del audio de los clips, LoRA en rama de audio y cross-modal, pérdida de audio ponderable, preview mp4 con sonido | Sonido coherente sin degradar la imagen |
| **4. Modo Edit (IC-LoRA)** | Formato de `ltx-trainer` (referencia como tokens de condición, pérdida solo en el objetivo, CRF 18), pares en la UI, preview con `pipeline_ltx2_ic_lora` | IC-LoRA de estilo vídeo→vídeo funcionando en ComfyUI |
| **5. Integración** | Launcher, locales, README + `docs/README_LTX25.md`, `.sh`, cola de trabajos, corrección de la tarjeta de LTX 2.3 | Igual que los demás trainers |

Las fases 2 y 3 pueden ir en la misma rama; la 4 en rama aparte.

---

## 6. Riesgos y preguntas abiertas

1. **Gemma 4 en NF4**: comprobar que cuantizar no degrada los 49 estados apilados (en H3 el sitio exacto de la capa importaba).
   Validar comparando embeddings NF4 contra BF16 en unos prompts (coseno por token).
2. **Prompt adaln**: la configuración no trae `use_prompt_adaln_single`; confirmar con el loader de `diffusers` cómo se convierte el checkpoint original de 2.5.
3. **Audio que arrastra al vídeo** (lección de H3): pérdida de audio con peso, LR o rank menores para módulos de audio, sin modo "solo audio".
4. **dev vs distilled**: entrenar sobre dev y verificar que el LoRA funciona sobre distilled (lo más usado en ComfyUI).
5. **Fps nativos y rejilla de fotogramas** de 2.5: la cabecera no los trae; leerlos de `pipeline_ltx2` / `duration_head` (por defecto 121 fotogramas en 2.3).
6. **¿Qué pasa con LTX 2.3?** Mantenerlo (hay usuarios y LoRAs de 2.3) y corregir "(2.5)" en su tarjeta y docs.

## Fuentes

- Cabeceras de `Lightricks/LTX-2.3` (`ltx-2.3-22b-dev`, `ltx-2.3-22b-distilled-lora-384-1.1`) y de `comfyicu/LTX-2.5` (espejo de `Lightricks/LTX-2.5`).
- `diffusers` 0.42.0.dev0 y `transformers` 5.17.0 instalados.
- [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) · [LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
- [LTX-2.5-22b-IC-LoRA-Refine-Details](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details) · [IC-LoRA-Ingredients](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients)
- [LTX LoRA training (ltx.io)](https://ltx.io/model/capabilities/lora-training)
