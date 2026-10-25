# Plan — Añadir entrenador YuE2-3B a AcademiaSD LoRAlab Trainer Studio

Estado: **Fase 1 completada** · Fases 0 completada · id del entrenador: **`yue2`**

> Primer entrenador de **audio** del Studio (los de vídeo ya tocan audio: MiniMax-H3, LTX-2.3). Música: letra + estilo → canción con voz y acompañamiento, 48 kHz estéreo.
> Precedente comunitario: `monsterovich/yue2-industrial-rock-lora` y `HaileyStorm/sv-billie-yue2-lora` demuestran que LoRA por etapa (AR y NAR) funciona con datasets de ~100–250 clips.

## Fuentes oficiales

| Fuente | Qué contiene |
| :--- | :--- |
| [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | Modelo: `modeling_yue2.py` (custom code), `model.safetensors` BF16 3.63 B, `qwen.tiktoken` (vocab 184 704), wheels `yue2_infer-0.1.3/0.1.5` |
| [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | VAE: `modeling_vae.py` (custom code), FP32, 132.6 M params, ~530 MB |
| [m-a-p/YuE2-Vae-legacy](https://huggingface.co/m-a-p/YuE2-Vae-legacy) | Variante usada en los benchmarks |
| [m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2) | Transcriptor lead-sheet (audio → ABC) para datasets de covers |
| [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) | Repo GitHub, foro de issues, technical report PDF |
| `monsterovich/yue2-industrial-rock-lora` | LoRAs entrenados por la comunidad + not-to-do documentado |

Licencia base: **CC BY-NC 4.0** (solo no comercial; igual que el resto de modelos no licencia-Apache del Studio ya manejados). Código vendorizado: MIT (stable-audio-tools / SnakeBeta / NVIDIA), ver `THIRD_PARTY_NOTICES.md` de los repos.

## Decisiones tomadas

| # | Decisión | Valor |
| :-- | :--- | :--- |
| 1 | Orden de etapas | **NAR (timbre) primero** — es lo que pide un LoRA de música (estilo/voz) y es la etapa con precedente comunitario. AR (plan) después |
| 2 | Un LoRA o dos | **Dos adaptadores separados** (uno por etapa), siguiendo el precedente industrial-rock: a la hora de inferir se cargan con strengths independientes (`ar_scale`, `nar_scale`). Export: dos `.safetensors` + modo "ambos" |
| 3 | Captioner | **No aplica el de imágenes** (Qwen3-VL no sirve). El "caption" de YuE2 son la **letra** (`.txt`, con etiquetas `[verse]`/`[chorus]`) + el **prompt de estilo**. Fase de dataset = trocear audio + letra; ASR opcional (Qwen3-ASR) para extraer letra de material real |
| 4 | Stems | Opcional: separar voz/instrumental con **Mel-Band RoFormer ya vendorizado** (`scripts/melband/`) si el material no viene limpio. Misma advertencia del `__init__.py`: separación destructiva, off por defecto |
| 5 | Conversor NF4 | Hay que escribirlo (arquitectura custom; ver Fase 0 §Convertidor) |
| 6 | Nombres | `yue2` → `server_yue2.py`, `trainer_ui_yue2.html`, `0_prepare_yue2.py` (audio), `1_pre_cache_yue2.py`, `2_train_lora_yue2.py`, `yue2_model.py`, `Run_LoRAlab-YuE2.bat/.sh`, `assets/yue2/`, `cached_data_yue2_*`, `yue2_lora_output_*` |

## Hallazgos clave (Fase 0, verificado desde el código publicado)

### 1. Un solo modelo con caminos duales AR + NAR en cada capa (MoT)

`YuE2ForCausalLM` (28 capas, hidden 2048, GQA 16/8, vocab 184 704, RoPE θ 1e6) contiene **toda** la arquitectura:

```
model.embed_tokens          # Embedding(184704, 2048) — 378 M params
model.layers.N.             # 28 × DecoderLayer, CADA capa con DOS caminos:
  self_attn                 #   camino AR:  q/k/v/o_proj + q_norm/k_norm (QK-RMSNorm)
  mlp                       #   camino AR:  gate/up/down_proj (inter 6144)
  nar_self_attn             #   camino NAR: q/k/v/o_proj propios
  nar_mlp                   #   camino NAR: gate/up/down_proj propios
  (input_layernorm, post_attention_layernorm, nar_*) norms duales
model.norm                  # RMSNorm final
lm_head                     # Linear(2048, 184704) — 378 M, NO tied
llm2vae                     # Linear(2048 → 64)   — solo NAR (cabeza de velocidad)
vae2llm                     # Linear(64 → 2048)   — solo NAR (inyección del estado x_t)
time_embedder               # sinusoidal → MLP(256→2048→2048)
latent_pos_embed            # buffer no entrenable, PE 1D hasta 24 576 frames
```

Conteo aproximado de params (cuadra con los 3.63 B de safetensors): cada camino ≈ 50.3 M/capa × 28 ≈ 1.41 B × 2 = 2.82 B, + embeds 760 M.

**⇒ LoRA por etapa es trivial por nombres de capa: el LoRA AR toca `self_attn.*` y `mlp.*`; el NAR toca `nar_self_attn.*` y `nar_mlp.*`. Son módulos INDEPENDIENTES — no se pisan.** (Monsterovich: AR r=8/α8, NAR r=32/α32 — cada ~35 MB y ~140 MB.)

### 2. AR = causal LM estándar

`forward` clásico con KV cache y loss CE con shift. Para el LoRA AR: entrenar con cross-entropy sobre la secuencia (estilo + letra + plan ABC…), igual que cualquier LM. Normal `S` de secuencia corta (texto + ABC).

### 3. NAR = flow matching con máscara híbrida (la pieza más delicada)

Firma exacta de `nar_velocity(tokens, ar_mask, nar_mask, nar_content_mask, x_t, t_value, nar_cond_end=0)`. Condiciones que revela el propio código oficial:

- El estado `x_t` ([T_lat, 64]) se inyecta en **TODAS** las posiciones NAR (incluidos `LATENT_START/END`, con `x=0` en los marcadores): `vae2llm(x_nar) + time_embedder(t) + latent_pos_embed(pos)`.
- Posiciones NAR via cumsum: START=0, contenido=[1..T_lat], END=T_lat+1.
- Máscara de atención híbrida: AR→AR causal, NAR→AR **completa**, NAR→NAR **bidireccional**, AR→NAR **bloqueado**.
- `nar_cond_end > 0` = *codec dropout* (el NAR solo ve el texto, no el audio semántico previo).
- Conversión de tiempo: `t_shifted = θ·σ(t)/(1+(θ−1)·σ(t))`, con **`timestep_shift = 1.0`** ⇒ efectivamente `t' = σ(t_raw)`; el t crudo vive en espacio logit.
- El modelo NAR es no-causal y **sin KV cache** (batch 1 con secuencia completa).

**⇒ Pérdida NAR = MSE(v_pred, ε − x0) sobre los frames de contenido, con `x_t = (1−t)·x0 + t·ε` y `t` pasado como logit — convención RESUELTA del wheel, ver Nota técnica 2 §2.**

### 4. VAE (Oobleck / stable-audio-tools, SnakeBeta)

| Dato | Valor | Consecuencia |
| :--- | :--- | :--- |
| Sample rate / canales | 48 000 / 2 | Requiere **resample + mezcla a estéreo** de TODO dataset |
| `downsampling_ratio` | 1 920 | **25 frames de latente por segundo** (frame = 40 ms) |
| Latent | `[B, 64, T]` | Canal-primero; el modelo espera `[T_lat, 64]` ⇒ transponer al cachar |
| `encode` | media posterior (o sample con `generator`) | `[B, 64, T]` FP32, sin scaling/shift factors explícitos |
| `decode_tiled` | decodificado teselado exacto, output a CPU | Infalible para previews largos sin VRAM |
| dtype | **FP32 estricto** (lo valida al cargar) | Solo corre en pre-caché y preview; nunca en el bucle de entrenamiento |
| Tamaño | 132.6 M FP32 ≈ 530 MB | Se puede mantener residente sin apenas coste |

Clips de 30 s ⇒ T = 750 frames ⇒ latente 750×64. `max_latent_frames = 24 576` ⇒ contexto máximo ≈ 16.4 min (en la práctica entrenamos con clips de 20–60 s).

### 5. Perfil de recursos (estimación a confirmar en Fase 4)

| Componente | BF16 | NF4 |
| :--- | ---: | ---: |
| Transformer 3.63 B | ~7.3 GB | ~2.0–2.3 GB |
| VAE (pre-caché / preview) | 0.53 GB FP32 | no aplica |
| AdamW 8-bit sobre LoRA r16 de 1 camino | ~0.1 GB | |
| Activaciones (S≈2–4 k tokens, grad ckpt) | ~1–2 GB | |

⇒ objetivo realista: **NAR r16–32 en 8 GB**; AR es aún más ligero (secuencias cortas, ~4–6 GB). Inferencia de referencia: 11.2 GiB pico en 4090 con el pipeline completo sin cuantizar.

### 6. Cómo carga el LoRA la comunidad (formato de export)

La comunidad carga sus adapters con un paquete propio (el README de `monsterovich/yue2-industrial-rock-lora` muestra `YuE2Model.load_lora("lora.safetensors")` con strengths independientes por etapa: `ar_scale=0.5, nar_scale=1.0`). El wheel oficial **no tiene loader de LoRA** (ver Nota técnica 2 §6). También hay nodos custom que usan el wheel en Spaces ⇒ compatibilidad objetivo: PEFT estándar + strength por etapa.

## Inventario: piezas de un entrenador y cuánto cambian

| # | Pieza | Fuente a copiar | Cambios reales |
| :-- | :--- | :--- | :--- |
| 1 | `scripts/server_yue2.py` | `server_anima.py` (más ligero, sin turbo) | estructura + sin previews de imagen; panel de clips de audio |
| 2 | `GUI/trainer_ui_yue2.html` | `trainer_ui_anima.html` | + selector de etapa AR/NAR/ambos; preview = re-canición de un clip corto |
| 3 | `scripts/0_prepare_yue2.py` | NUEVO (rol del captioner) | troceo de audio en clips, resample 48 k estéreo, letra vía ASR opcional, stems opcionales |
| 4 | `scripts/1_pre_cache_yue2.py` | `1_pre_cache_MiniMaxH3.py` (VAE de audio, encode real) | reemplaza TE imagen por: (a) latentes del VAE de audio, (b) tokenización de letra+estilo+ABC |
| 5 | `scripts/2_train_lora_yue2.py` | NUEVO, con la plantilla de `2_train_lora_anima.py` | loss doble: CE (AR) y MSE de velocidad (NAR); resume exacto, lora_options intactos |
| 6 | `scripts/yue2_model.py` | patrón `mingimage_model.py` | vendorizar `modeling_yue2.py` + `modeling_vae.py` + carga NF4 |
| 7 | `tools/yue2/5_conversor_YuE2_NF4.py` | patrón de los conversores existentes | cuantizar las dos rutas + heads NAR; mantener embeds en BF16 |
| 8 | `code/Run_LoRAlab-YuE2.bat` + `.sh` | los de Anima | strings |
| 9 | `assets/yue2/` | — | imágenes nuevas |
| 10 | `GUI/launcher.json` | — | +1 entrada |

Genéricos **sin tocar**: `i18n`/`locales` (los mensajes nuevos se traducen después), `remote_access`, `file_transfer`, `gpu_stats`, `console_stream`, `trainer_stream`, `lora_options`, `sync_pinned`, `melband` (vendido, se reutiliza tal cual).

## Particularidades a resolver

1. ~~Convención FM~~ **RESUELTA** (ver Nota técnica 2: `x_t = (1−t)·x0 + t·ε`, target `v = ε − x0`, `t` se pasa al modelo como logit→σ con shift 1.0 = identidad). Queda el smoke-test de siempre.
2. ~~Tokens/protocolo~~ **RESUELTOS** (protocolo completo en Nota técnica 2: instruccion + `[Tags]` + `[Lyrics]`, ABC entre `<abc>`/`</abc>`, luego `MUSIC_START` + códigos semánticos de vocab 32 768). **Pendiente de decisión**: cómo obtener los códigos semánticos para el dataset NAR (ver riesgo R2).
3. **Dataset NAR ≠ dataset AR**: NAR necesita el plan ABC del clip (vía SheetSage2 o ABC manual) porque el NAR condiciona en el score; el AR solo necesita letra + estilo + ABC (que es el propio target del AR —なく SheetSage2 también vale). Confirmar qué produce `SheetSage2` exactamente (melody sin acordes vs full).
4. **Checkpoint del NAR no debe desconectar el grafo**: torch.utils.checkpoint con `use_reentrant=False` es seguro solo si no se hace `detach` de las entradas entre capas (bug documentado por monsterovich: MSE estancada en ~1.38; corregida → ~0.59). Regla: gradiente completo a través de las 28 capas.
5. **Datos en disco grandes**: 1 min de audio = 1 500 frames × 64 × 4 B ≈ 0.38 MB en latente — trivial. El WAV crudo 48 k estéreo es lo pesado (10 MB/min) ⇒ guardar los latents y decidir si conservar WAV.
6. **Previews caras**: una canción entera tarda ~60 s+ incluso en 4090 ⇒ preview = **clip corto de ~20–25 s** cada N pasos rezando `timestep sampling` corta. Coste estimado ~10–20 s por preview.
7. **trust_remote_code**: `modeling_yue2.py`/`modeling_vae.py` son custom → se vendorizan (como Ming-Image) en `yue2_model.py`, nunca se ejecuta código remoto en el Studio.
8. **Windows**: nada de kernels CUDA custom (el modeling no importa extensiones; SDPA nativo) ✓. Tarjetas ≤ 8 GB sin BF16 (GTX 16xx) → cuantizar el acceso solo si hace falta; de momento se exige GPU con BF16 como en el resto del Studio.

## Fases

### Fase 0 — Inspección previa ✅
- [x] Leer `modeling_yue2.py` y `modeling_vae.py` (ver Hallazgos).
- [x] Configs de ambos repos (ver Hallazgos).
- [x] Extraer `yue2_infer-0.1.5` y leer: ODE solver, convención FM, protocolo de tokens, tokenizer (ver **Nota técnica 2** al final).
- [ ] Comparar `YuE2-Vae` vs `legacy` (¿diferencia de pesos? el card lo sugiere).
- [ ] Confirmar salida de SheetSage2 (formato ABC, con/sin acordes).

### Fase 1 — Esqueleto (aparece en el lanzador) ✅ HECHA

| Fichero | Origen | Estado |
| :--- | :--- | :--- |
| `scripts/server_yue2.py` | `server_anima.py` | ✅ renombres de rutas/ids/configs; `DATASET_EXTS` cambia a audio (`.wav .flac .mp3 .ogg .m4a .opus`); el script de caption apunta a `0_prepare_yue2.py` (llegará en Fase 2) |
| `GUI/trainer_ui_yue2.html` | `trainer_ui_anima.html` | ✅ renombres a secas (Anima→YuE2, 16 ocurrencias); el audio/por-etapas específico llega en Fases 2–4 |
| `code/Run_LoRAlab-YuE2.bat` / `.sh` | los de Anima | ✅ |
| `assets/yue2/` | copia provisional de `assets/anima/` | ⏳ el usuario pondrá portada/logo definitivos |
| `GUI/launcher.json` | — | ✅ entrada `yue2` tras Anima («Music LoRAs (timbre NAR and plan AR)») |
| `.gitignore` | — | ✅ `YuE2-3B_Raw/`, `YuE2-3B_NF4/`, `YuE2-Vae/`, `YuE2-Vae-legacy/`, `Qwen3-ASR/`, `SheetSage2/` |

**Pruebas pasadas** (servidor arrancado desde venv, `LORALAB_SKIP_BROWSER=1`, puerto 5000):
`GET /` devolvió la UI (110 576 b); `/api/settings`, `/api/system-stats`, `/api/dataset-info`, `/api/checkpoint-info`, `/api/hf-token`, `/api/status`, `/api/previews`, `/api/caption-settings` y los assets (`/assets/portada.jpg`, `/assets/logo_128.png`) responden 200. `/favicon.ico` 404 igual que en Anima (nadie tiene `logo.png`).
`POST /api/save-precache` y `/api/save-train` con proyecto `yue_smoke` escribieron `settings/pre_cache_settings_yue2.json`, `settings/train_settings_yue2.json` con `./cached_data_yue2_yue_smoke` y `./yue2_lora_output_yue_smoke`, más las copias dentro de las carpetas del proyecto; carpetas creadas y limpiadas tras la prueba.
Integridad JS: **74 `getElementById`, 0 ids faltantes**; `launcher.json` JSON válido. Consola del servidor sin errores (el proceso se paró a mano).

**Pendiente visual del skeleton**: la UI sigue mostrando textos/ajustes propios de imagen (captions de foto, previews de imagen). Se adapta a audio/letra/etapas AR-NAR en Fases 2–4; es el mismo enfoque Ming-Image (esqueleto funcional primero, refinado después).

### Fase 2 — Dataset de audio (rol "captioner") ✅ HECHA (script + test CPU)
`0_prepare_yue2.py`: normalizar audio (48 k estéreo), troceado por silencia/sec contiguos a 20–60 s, letra en `.txt` con la misma base (a mano, o ASR/Qwen3-ASR como ayuda), stems opcionales con `melband/`. Dataset mínimo viable: clips + letra + prompt de estilo (+ ABC si va a NAR).

**Estado real (2026-10-19)**: script completo y probado con audio sintético a 48 k (test en `Temp/opencode/test_prepare_yue2.py`, TODO OK). Decisiones que difieren del borrador original:
- Corta cada `clip_seconds` (30 s por defecto, min 10 s), llevando el límite al punto de menos energía en ±`snap_seconds` (1.5 s): cortes musicales, no metrónomo.
- Salida `<base>__cNN.wav` en PCM16 48 k estéreo (`peak_normalize` solo baja picos > −1 dBFS). Con stems: `__voz__cNN` / `__inst__cNN`.
- Idempotente: los ficheros con sufijo `__cNN` nunca se toman como fuente (regex `CLIP_NAME_RE`), una 2.ª pasada no re-trocea. Fuente sin clips usables se queda en el sitio y avisa en cada pasada.
- La letra NO se reparte sola entre clips (sin alineación sería inventar): la fuente y su `.txt` van a `dataset/_fuentes/` (subcarpeta fuera del escaneo de la pre-caché y respetada por Clear Dataset) y la letra se asigna por clip en el Dataset Manager, o con ASR en una fase futura.
- Décode/remuestreo por ffmpeg (wav/flac/mp3/m4a/ogg/opus...), sin torchaudio; `LORALAB_FFMPEG` configurable.
- Stems con Mel-Band RoFormer vendorizado (una pasada por FUENTE, antes del corte). 100 % CPU por defecto; GPU solo con `stems_device='cuda'` (pendiente de probar con descarga del modelo de separación).
- Lee `caption_settings_yue2.json` + `dataset_path` de `pre_cache_settings_yue2.json`, igual que los captioners; YA enlazado al botón de captions del server (corre vía `/api/run`).

### Fase 3 — Pre-caché ✅ HECHA (script + test GPU)
`1_pre_cache_yue2.py`: (a) VAE encode → `_latent.pt`; (b) tokenizar estilo+letra(+ABC) → `_tokens.pt`; (c) `_neg_prompt` fijo. Reutiliza la infraestructura de backfill/undo del pre-caché de MiniMax-H3.

**Estado real (2026-10-19)** — script completo y probado en GPU (RTX 5080, ~530 MB de VAE FP32; VRAM liberada al acabar). Modelo: plantilla ANIMA (289 líneas), no la de MiniMax. Decisiones:
- **Vendorización**: `scripts/yue2/` con `protocol.py` + `tokenization_yue2.py` copiados del wheel `yue2_infer-0.1.5` (Apache-2.0; ajustado el `from_pretrained`, sin resolución de rutas del wheel). El compositor de prefijos (`token_prefixes`/`negative_prefix`/`SongRequest.text()`) ES el del wheel: imposibilidad de divergencia de formato.
- **Por clip**: `<name>_latent.pt` = `[1, 64, T]` FP32 (VAE-native; el trainer transpone a `[T,64]`), latente SIN normalizar (el pipeline oficial lo consume tal cual; std medida ~1.05–1.10). `<name>_prompt.pt` = dict `{base_ids, abc_ids, neg_ids, lyrics_len}` — `base_ids` trae el prefijo completo con andamio horneado: para cot=off `[EOD]+texto+[ABC_START,ABC_END,MUSIC_START]` (verificado byte a byte contra el compositor del wheel); el trainer AR añade los códigos codec tras MUSIC_START.
- `_neg_prefix.pt` global solo con cot=off (instrucción sola + MUSIC_START, 12 ids). Con cot≠off el negativo (instrucción + mismo ABC) va por clip. `_meta.json` guarda versión de formato=1, cot, estilo, protocolo yue2-native-v1.
- Recortes a múltiplos de 1920 muestras (1 frame=1 código=40 ms); autotest de roundtrip encode/decode del primer clip sin cachear: error RMS 2.2 % (sano).
- Escaneo: TODO audio del primer nivel del dataset es muestra (los `__cNN` incluidos); `_fuentes/` fuera. Limpieza de .pt obsoletos estilo Anima; skip por mtime+geometría para no recodificar; los prompt se reescriben SIEMPRE (tokenizar es instantáneo y refresca cot/estilo).
- `--prompt-only <cache_dir> <prompt> <device>` (contrato server de Anima): tokeniza y escribe `_custom_prompt.pt` + `_custom_prompt.txt` (ids puros de texto; el trainer compone estilo/letra de preview en Fase 4). Probado ✓.
- Todos los .pt contienen tensores torch (nunca numpy pickled): `torch.load(weights_only=True)` funciona en toda la caché.
- settings usados (y borrados tras el test): dataset_path, project_name → `cached_data_yue2_{proyecto}`, `style_prompt` (NUEVA clave, aún sin UI), `cot=off`, `force_gpu=true`.
- **Pendientes de Fase 4**: decidir camino A (muestrear códigos codec del clip con el modelo base para condicionar NAR con señal acústica real) vs camino B (codec dropout vía `nar_cond_end`, texto solamente); el camino A necesita pre-cache de códigos codec por clip y aún no existe.

### Fase 4 — Entrenamiento ✅ HECHA (trainer + smoke GPU completo)
`2_train_lora_yue2.py` (≈1050 líneas, patrón Anima):
- Trainer ÚNICO con cfg `stage` = `nar` | `ar` | `both`. En `both` se entran las
  dos etapas de forma secuencial y CADA UNA recarga el modelo desde disco
  (PEFT envuelve los Linear in situ; partir de pesos frescos evita mezclar
  adapters). VRAM estable en 7,3 GB por etapa (los handlers de señal se
  liberan al cerrar cada etapa: el closure retenía el modelo anterior).
- NAR (Camino B): `nar_context()` hace el prefill AR del prefijo en
  `no_grad` (K/V constantes, el LoRA NAR no pasa por la capa AR) y
  `nar_velocity()` reconstruye el forward del CachedNAR con GRADIENTE en las
  ramas nar_* (pad START/END x=0, `_shift_t_value` con t=logit clampado,
  posiciones RoPE contiguas, sdpa bidireccional con GQA). x_t =
  (1-t)x0+t·eps, target = eps-x0 en float32. MSE inicial ≈ 2.2 (latente sin
  normalizar, std ~1.07) — el plan vaticinaba ~1.0 con latente unitario: la
  cifra real con este VAE parte de ~2.2 y baja hacia ~1.0.
- AR: códigos por clip muestreados UNA vez con el modelo BASE (UNA función
  `generate_codes` bit-exacta con `sampling.distribution` del release:
  allow-list codec+MUSIC_END, min_tokens, ventana de repetición, temp/top_k/
  top_p con cumsum-cancel), cacheados en `_codes_<clip>.pt` (seed por clip
  estable vía crc32). CE teacher-forcing: prefijo+códigos+MUSIC_END, sólo
  predicen los códigos y el cierre; logits recortados a la cola (memory:
  [K+1, V], ~570 MB en vez de la secuencia entera).
- Export: claves NATIVAS PEFT `model.layers.N.{nar_|}self_attn…/mlp.…lora_A/B`
  + `alpha` por capa + metadata ss_* + `yue2_stage`. Fichero por etapa:
  NAR→`YuE2_(FINAL_)LoRA*.safetensors`, AR→`YuE2_AR_(FINAL_)LoRA*.safetensors`
  (ambos pisan el regex `step_(\d+)\.safetensors` del server).
- Trampa encontrada en `is_target`: `nar_mlp.gate_proj` TERMINA igual que
  `mlp.gate_proj` (sufijo casual) — con endswith a secas el LoRA AR secuestraría
  la rama NAR (392 capas en vez de 196). Ancla `.` delante.
- Resume/checkpoints POR ETAPA (resume_checkpoint[_ar]/current_step[_ar].txt/
  optimizer[_ar].pt), mismas reglas que Anima (saved targets desde los PESOS).
- NF4: cuantización bnb en la carga (solo atención+MLP de las dos etapas;
  puentes del latente/normas/embeds en BF16) — 392 capas → 3,1 GB de VRAM.
- Preview de audio: generate_codes + ODE midpoint (steps, cfg) + decoder VAE
  → `preview_step_N.wav` 48k estéreo. En stage=ar la ruta NAR honesta queda
  base (es exactamente lo que exporta el fichero AR).
- Smoke GPU completo (dataset sintético 2 clips, seed 777): NAR 2.02→1.05,
  AR CE ~4.0→3.5, both secuencial, resume exacto ("Resuming from step 2"),
  NF4 backward por Linear4bit, preview WAV generado. ~0,8 s/it por paso NAR
  (clip 12 s, 750 frames) sin contención.
- Settings reales creados: `settings/train_settings_yue2.json` +
  `settings/pre_cache_settings_yue2.json`.

### Fase 5 — Modelo NF4 en HF ✅ HECHA (conversor + carga en trainer; subida a HF pendiente del usuario)
`tools/yue2/5_conversor_YuE2_NF4.py`:
- Caché NF4 en `YuE2-3B_NF4/`: `model/` (formato caché studio, Qwen-Image 2.1:
  `index.json` + `weights/` con peso uint8 + QuantState `as_dict(packed=True)` +
  `others.safetensors`), `vae/` copiado tal cual, `modeling_yue2.py` +
  config/tokenizador/`yue2_generation_config.json` para consumo offline, README.
- Mismas reglas de cuantización que el trainer (`is_quantized` espejo): 392 capas
  NF4 (atención+MLP de AR y NAR en los 28 bloques), 5 Linear BF16 (llm2vae,
  vae2llm, time_embedder.mlp.0/1, lm_head), 227 tensores en others (embeds,
  normas, RoPE, latent_pos_embed). Total 3,4 GB (vs 7,26 GB Raw).
- Verificación integrada: el QuantState reconstruido con `from_dict` reproduce
  la capa viva con error 0 y los logits AR del modelo reconstruido difieren del
  BF16 un 7,55 % relativo (acotado; NF4 es con pérdida).
- `2_train_lora_yue2.py` ahora DETECTA la caché (`YuE2-3B_NF4/model/index.json`)
  y carga de ella (`load_nf4_backbone`, mismo cargador): 5,1 s / 3,1 GB de VRAM
  en vez de ~23 s; smoke NAR OK con loss descendente igual.
- Subida a HF (`AcademiaSD/YuE2-3B-NF4-for-LoRA-Training`) pendiente: el token
  no está configurado; la carpeta queda lista para `huggingface-cli upload`. 

### Fase 6 — Validación y cierre — HECHA la parte automatizable (entorno sintético)

**Automatizado y verificado:**
- `tools/yue2/6_validar_lora_yue2.py`: carga un export en un modelo LIMPIO con
  PEFT estándar (objetivos derivados de las propias claves del fichero),
  inyecta con `set_peft_model_state_dict` (añadiendo el prefijo
  `base_model.model.` que PEFT espera: el export nativo va sin él, pensado
  para cargadores tipo ComfyUI), comprueba que lora_B no queda a cero y corre
  la rama de la etapa antes/después: NAR diff 2,29 % | AR diff 1,42 % con un
  adapter de 1-2 pasos de smoke (lo relevante es que aplica y es finito) y la
  OTRA rama sigue finita (el LoRA no la toca). Ambos exports →
  **VALIDATION PASSED**. Hallazgos ya corregidos: el remap de peft 0.21 opera
  con claves PREFIJADAS y el rank se lee de `lora_A.shape[0]`.
- Trainer estable en relanzamientos: resume por etapa y `both` secuencial con
  VRAM 7,3 GB constante (los handlers de señal liberan el modelo al cerrar
  cada etapa). Pre-caché idempotente.
- README.md: sección `YuE2-3B — song LoRAs` + fila de licencia CC BY-NC 4.0.

**Tabla de inicio recomendada (se mantiene):**

| Inicio NAR (timbre) | |
| :--- | :--- |
| Clips | 20–60 s, 48 k estéreo |
| Rank/Alpha | 16/16 (comunidad: 32/32 con dataset grande) |
| LR | 2e-4 a 4e-4 (FM+mse es estable con AdamW) |
| Pasos | ~1200 como referencia (comunidad) |
| Inicio AR (plan) | rank 8/8, LR 4e-4, ~600-1200 pasos |

MSE NAR inicial real ≈ 2.2 (latente sin normalizar, std ~1.07): la nota previa
de ~1.0 asumía latente unitario.

**Pendiente del usuario (canciones reales / oído / credenciales HF):**

### Fase 7 de facto — Dataset real (Agila, Extremoduro) ✅ primer run completo
Run real grabado tras el cierre del plan (todo NF4, RTX 5080 16 GB):
- Dataset: álbum completo (13 pistas mp3) → 0_prepare: **91 clips** de 28-32 s
  (48 kHz estéreo, corte por silencios, 10 s en CPU). Letras del libreto
  (Dataset_musica/Letras) repartidas con un script: **86 clips con .txt**
  (prefijo numérico → bloque en MAYÚSCULAS del libreto, créditos del libreto
  fuera, citas internas conservadas); «Me estoy quitando» falta en el libreto
  → sus 5 clips sin .txt (condición sin letra).- style_prompt global puesto en [Tags]; letra cruda en .txt (la pre-caché compone
  el prompt: instr + [Tags] + estilo + [Lyrics] + letra).
- Pre-caché: 91/91 latentes + prompts (319-347 ids por clip). Nota del
  autotest del VAE: roundtrip RMS 43 % sobre música real (el latente del
  encoder no reconstruye al 100 %; no afecta al entrenamiento, sí al
  audio decodificado final).
- NAR (rank 16/16, lr 3e-4): 300 pasos loss 1.18→1.08 (0.38-0.5 s/it) y
  **resume exacto 300→1200**: step 301 continúa el schedule (lr 2.69e-04),
  loss → 1.056 en el 1200. Exports: YuE2_FINAL_LoRA.safetensors +
  checkpoints 100...1200 + previews cada 50/100 (20 s WAV 48 kHz).
- **VRAM confirmada por el usuario en vivo: ~8,0 GB sostenidos, picos
  8,7 GB** (previews ODE+VAE y guardado de checkpoint) — bajo el límite de
  9 GB (caché NF4 del backbone solo 3,1 GB; el resto activaciones/optimizer
  con clips de 30 s).
- Resultado del oído (usuario): «el estilo de música se parece, la voz del
  cantante le falta entrenamiento» → NAR captura producción/estilo con
  rapidez; el timbre converge más lento (la referencia de comunidad de
  ~1200 pasos ya corrida).

**Pendiente del usuario:**
- 200–300 pasos NAR con dataset real → preview audible y comprobación del timbre.
- Comparación con la inferencia oficial (`yue2_infer`) con los exports puestos.
- Subida del NF4 a HF (`AcademiaSD/YuE2-3B-NF4-for-LoRA-Training`).
- (Entorno: durante la validación había 2 procesos de OTRA sesión
  (`_test_space_style.py`, Ming-Image) reteniendo ~15 GB de VRAM; cerrarlos
  para medir velocidades reales.)

---

# Fase 0 — Nota técnica (lo ya verificado al código publicado)

## 1. Nombres de modulos para `is_target()` (checkpoints HF)

Los keys del `model.safetensors` son (vía `modeling_yue2.py`, `base_model_prefix = "model"`):

```
model.layers.<N>.self_attn.{q,k,v,o}_proj.weight      # AR path
model.layers.<N>.self_attn.{q,k}_norm.weight          # QK-RMSNorm (NO LoRA)
model.layers.<N>.mlp.{gate,up,down}_proj.weight        # AR path
model.layers.<N>.nar_self_attn.{q,k,v,o}_proj.weight   # NAR path
model.layers.<N>.nar_mlp.{gate,up,down}_proj.weight     # NAR path
model.norm.weight
model.embed_tokens.weight                              # 378MB — NO LoRA
lm_head.weight                                          # 378MB — NO LoRA
llm2vae.weight / vae2llm.weight                         # NAR heads — mirar si dar LoRA (probable NO al principio)
time_embedder.mlp.{0,2}.weight                          # no entrenar
```

⇒ **Dos `is_target` distintos**: las dos etapas están separadas por nombrado, sin superposición. Export del AR con `self_attn./mlp.` y del NAR con `nar_self_attn./nar_mlp.`; en modo "ambos", un solo fichero con todos los keys. Formato PEFT estándar (ver Nota técnica 2 §6: el wheel no trae loader propio).

## 2. Arquitectura exacta cuantifica si el LoRA a rank 16 cabe

Per-layer NAR LoRA (r=16) entrenable ≈ 16 × (2048+1024) × 3 (q,k,v) + 16 × (1024+2048) × 1 (o) + 16 × (2048+6144) × 2 + 16 × (6144+2048) × 1 → de orden **2.6 M/capa × 28 ≈ 73 M params de LoRA**. AdamW 8-bit: ~0.15 GB.

Activaciones del NAR dominadas por la secuencia S ≈ texto (≤1 k) + T_lat + 2 — para clips de 30 s: S ≈ 900 + 752 ≈ 1 700 tokens → con grad checkpointing sobra en 8 GB. El desdoblamiento de tiempo ya está resuelto (Nota técnica 2 §2): `t` se pasa como logit y `σ(raw)` es el t real de interpolación con θ=1.0.

## 3. Riesgos

| Riesgo | Impacto | Mitigación |
| :--- | :--- | :--- |
| Convención FM mal trasladada (target invertido, t mal muestreado) | Entrena mal **sin dar error** | RESUELTA en Nota técnica 2; queda el smoke-test de 1 paso contra `nar_velocity` oficial |
| Grafo de autograd cortado dentro del checkpointing | MSE plateau ~1.38 (bug ya visto en la comunidad) | Prohibido `detach` entre capas; test de gradiente completo |
| Códigos semánticos (codec) no derivables del audio | El dataset NAR no tiene la condición completa que usa el modelo desplegado | Camino A: muestrear los codec con el modelo base sobre el clip (condición autogenerada). Camino B: condicionar solo el texto con `nar_cond_end` (codec dropout). Decidir empíricamente en Fase 4 |
| Tokenización a mano (tokens especiales, formato de letra y ABC) | Rompe el entrenamiento silenciosamente | RESUELTA: usar `YuE2TextTokenizer` del wheel (vendorizado) |
| Preview con canción completa | ~60 s mínimo por preview, latencia en el entrenamiento | Preview = clip corto de ~20 s |
| VAE exige FP32 estricto al cargar | Rompe si hay autocast BF16 | VAE fuera del bucle de entrenamiento; autocast off |
| CC BY-NC 4.0 | Solo no comercial | Mention en README, como el resto de trainers con esta tarjeta |
| Activaciones en secuencias muy largas (covers sin score, `cot=off`) | OOM en 8 GB | Limitar a cortes de ≤60 s; block swap si hace falta (patrón Qwen 2512) |

---

# Fase 0 — Nota técnica 2: wheel `yue2_infer-0.1.5` (verificado)

Extraído y leído: `nar.py` (ODE), `protocol.py` (protocolo/andamio), `tokenization_yue2.py` (BPE), `quantization.py` (FP8, sin NF4), `storage.py`, `pipeline.py`.

## 1. Protocolo de tokens (de `protocol.py`, valores exactos)

```
EOD          = 151643
ABC_START    = 151847        <abc>
ABC_END      = 151848        </abc>
MUSIC_START  = 151851
MUSIC_END    = 151852
CODEC_OFFSET = 151853        códigos semánticos: IDs 151853..184620 (32 768 códigos)
LATENT_START = 184621
LATENT_END   = 184622
LATENT_PAD   = 184623
VOCAB/CONTEXT= 184704 / 24576
```

- Prefijo positivo: `[EOD] + encode(f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n")`; con plan: `+ [ABC_START] + abc_ids + [ABC_END, MUSIC_START]`; sin plan (`cot=off`): `+ [ABC_START, ABC_END, MUSIC_START]`.
- Prefijo negativo (CFG simbólico): solo la instrucción (más el mismo bloque ABC) sin `[Tags]/[Lyrics]`.
- Tras `MUSIC_START` el AR muestrea los **códigos codec** (vocab 32 768; dos fases de sampling: ABC `temp .7 / top_p .9 / top_k 30` y semántico `temp 1.0 / top_p .95 / top_k 100 / rep 1.2`); después el NAR materializa la forma de onda. Los chunks: `ar_tokens = prefix + codec[a:b]+OFFSET + [MUSIC_END]`, y el ruido se sortea **una vez** para toda la canción en CPU (con seed fija) y se trocea entre chunks.
- Texto: `INSTRUCTIONS = {"off": "Generate music with codec tokens from the given conditions.", "melody": "…melody-only ABC transcription without chord symbols…", "full": "…chord-annotated ABC transcription…"}`.
- Tokenizador: Qwen BPE `qwen.tiktoken` (151 643 tokens ordinarios) mediante tiktoken; clase `YuE2TextTokenizer` (~40 líneas, vendorizable tal cual).

## 2. Convención de flow matching — RESUELTA (de `CachedNAR.solve`)

```
estado inicial  x₁ = ε   (ruido; un sorteo por chunk)
integración     t: 1 → 0, 32 pasos midpoint:   t = 1 − step·dt
                raw_t = logit(t).clamp(−20, 20)      # ¡el modelo recibe el LOGIT!
                v1 = velocity(x, raw_t);  x_mid = x − v1·dt/2
                x ← x − velocity(x_mid, raw_mid)·dt
```

Como el estado a t=1 es ruido y a t=0 es latente limpio: `x_t = (1−t)·x0 + t·ε` ⇒ **target de velocidad = ε − x0**. Con `timestep_shift = 1.0`, `_shift_t_value(raw) = σ(raw)` = t exacto: el shift es una identidad en este checkpoint — pero hay que pasar el `raw` **como logit** para respetar el protocolo y futuras variantes con θ≠1. Para entrenar: muestrear `t ~ U(0,1)`, `x_t = (1−t)·x0 + t·ε`, `target = ε − x0`, MSE solo sobre frames de contenido.

Detalles NAR del wheel que el trainer debe replicar:
- `CachedNAR.velocity` **usa solo los módulos NAR** (`nar_self_attn`, `nar_mlp`); las K/V del prefill AR se cachean una vez (el prefill recorre los módulos AR).
- Los tokens NAR son `nar_length = T_lat + 2` (START y END con `x=0`), y la velocidad se Lee de `llm2vae(model.model.norm(x))[0, 1:-1]` — **el `model.norm` se aplica antes del `llm2vae`** (ojo: el `nar_velocity` del HF modeling tiene rutas equivalentes pero opacas; el wheel es la referencia exacta).
- Posiciones RoPE de NAR: `ar_length .. ar_length+nar_length`; PE de latente local 0..nar_length−1 (START=0, contenido 1..T_lat, END=T_lat+1), `clamp(max_latent_frames−1)`.

## 3. Chunking y contexto (de `chunk_ranges`)

`size = min((24576 − prefix − 3) // 2, …)`: cada chunk cabe prefix + 2·size (prefill AR + iteración NAR). Con prefijo <1 k tokens ⇒ chunks de hasta ~12 k frames ≈ 8 min de audio por chunk. Clips de 20–60 s: 1 chunk, sin problema.

## 4. Codec dropout y condición de texto solo (de `nar_cond_end`)

Si `nar_cond_end > 0`, el prefill AR solo cachea K/V hasta esa posición: el NAR **no ve los tokens posteriores** (los códigos codec). Es el mecanismo oficial para Condición sin codec — el camino B del riesgo R2 — y es idéntico al desplegado (la inferencia soporta `nar_cond_end`). Vía implícita: el modelo base se preentrenó con este dropout, luego la condición "solo texto" es legítima ambos lados.

## 5. Cuantización del wheel

`quantization.py` cuantiza **FP8 solo el AR** (E4M3 per-tensor, `torch._scaled_mm`, cc 8.9+) y "NAR always restores exact BF16". No hay NF4. Consecuencias para el Studio: nuestra conversión NF4 es propia; pero su procedencia valida una advertencia: **NAR es la etapa sensible a la precisión** — al NF4 (bitsandbytes) cuantizar NAR SOLO con validación (smoke MSE vs BF16 con x_t fijo). Alternativa segura: NAR en BF16 + block swap (patrón Qwen 2512), AR sí en NF4.

## 6. No hay loader de LoRA en el wheel

Grep: **cero apariciones** de "lora" en `yue2_infer-0.1.5`. El `load_lora` que cita el repo de `monsterovich` viene de otro paquete (`yue2_inference`) o versión futura. ⇒ Export del Studio = **PEFT estándar** (keys `model.layers.N.{self_attn,mlp}.*` y `nar_self_attn/nar_mlp`, `lora_A/B`, `lora_alpha`), cargable con `peft` sobre `YuE2ForCausalLM` (vendorizado), más un loader opcional de strength por etapa de ~20 líneas para inferencia sin PEFT (mismo patrón de `get_peft_model_state_dict`/`set_peft_model_state_dict` de los demás trainers).

## 7. API de inferencia útil para previews

- `synthesize(model, prefix, codec, seed, steps=32, offload_ar=…)`: mueve `embed_tokens`, `lm_head` **y TODOS los módulos AR** de cada capa a CPU durante el ODE NAR — es exactamente la rutina de bajo VRAM para previews.
- Preview del trainer: con la letra/estilo del dataset (o prompt custom), `cot="melody"` (o `off` para clips cortos), `steps` configurable (menos de 32 para rapidez), seed fija, clip corto.

---

# Fase 8 — Por qué el LoRA del 19/10 no clonaba el timbre (y qué se cambió)

## El diagnóstico

El LoRA de 4.200 pasos del 19/10 se entrenó con una versión del trainer que
**condicionaba el NAR solo con el prefijo de texto**: los códigos semánticos
reales (MERT-v2 + head) no existían todavía en la caché, y la fase 3 del
pre-cache (códigos reales) se añadió después, junto con el aviso que hoy impide
entrenar sin ellos.

El NAR es un **decodificador**: recibe códigos y produce audio. La voz vive en
los códigos. Si el NAR se entrena sin ellos, el LoRA no tiene dónde aprender el
timbre (la entrada ya no lo lleva) y en la preview y en ComfyUI el NAR sí los
recibe → el LoRA se aplica sobre una entrada que nunca vio. De ahí el síntoma
exactamente descrito en el run del 19/10: *"el estilo de música se parece, la
voz del cantante le falta"*.

Con la caché regenerada (91/91 ficheros `_realcodes_*.pt`, la fase 3 del
pre-cache) y el NAR condicionando con `prefijo + códigos + MUSIC_END` —igual
que el protocolo publicado del wheel y que el AI Toolkit— el LoRA sí ve la voz
en los datos y puede aprenderla.

## Lo que se cambió en esta fase

1. **El NAR se condiciona con los códigos reales** (ya estaba en el código, pero
   la caché del 19/10 no los tenía: hay que volver a lanzar el pre-cache).
2. **Regularizador del AR (el "minted pack" de la comunidad)**: con un set
   pequeño el AR puede colapsar la gramática de tokens de YuE2. Se descarga
   `Mothersuperior/yue2-minted-corpus` (4.732 canciones generadas por YuE2,
   97 MB) y la etapa AR mezcla un 50 % de ellas con las del artista
   (`regularizer` + `regularizer_ratio` en los ajustes). Es lo que hace la
   comunidad: *"the AR trainer draws 50% of its songs from it so a small artist
   set cannot collapse YuE2's token grammar"*.
3. **El guardado de los ajustes ya no pisa las opciones**: el servidor
   reescribía el JSON entero con lo que mandaba la GUI, así que
   `timestep_sampling`, `max_grad_norm`, `weight_decay` o `min_lr_ratio`
   se perdían en cada guardado. Ahora fusiona (igual que ya hacía Fibo 2.0).
   **Arrreglado en los 17 servidores.**
4. **`timestep_sampling` = `logit_normal`** por defecto en el NAR (es lo que
   usa la referencia; antes era `uniform`), y la opción está en la GUI.
5. **La etapa por defecto** sigue siendo NAR, pero la GUI explica qué hace
   cada una: NAR = voz, timbre y producción; AR = letra y fraseo.
   **Corregido en la Fase 9**: las pruebas reales dicen lo contrario (AR = voz).

## La receta de la comunidad para un LoRA de artista (para comparar)

- LoRA **de la AR** (rank 64, lr 1e-4, 1.600 pasos como máximo: *"do not train
  longer, past ~1,500 steps the model memorises the songs"*), con el
  regularizer pack al 50 % y el *lyric cursor* (Demucs + MMS forced alignment,
  peso 0.08) para que el modelo sepa **cuándo** se canta cada línea.
- El LoRA **de la NAR** de la comunidad (`nar_lora_joint_v4/v5/v8/v9`) no es un
  clonador de voz: es un **adaptador del decodificador** para que renderice bien
  los códigos del head de la comunidad (`tokenizer_head_v5_30k` es el único
  entrenado contra el decodificador base, así que no necesita adaptador NAR).
- En la inferencia se aplican **los dos**: el LoRA de la AR en el lado del CLIP
  y el de la NAR en el del modelo.
- Para el sonido: el VAE merge `Mothersuperior/YuE2-Vae-merge-0.666`.

## Verificado en esta fase

- Pre-cache con la fase 3: 13 temas, 91 clips con sus códigos reales ✓.
- Entrenamiento NAR + AR (40 pasos cada uno) con la mezcla del regularizador:
  NAR 1,31 -> 1,07; 91 muestras; 31 piezas minted tras acotar la longitud a la
  de los clips; LoRA de las dos etapas y previews de audio generados ✓.
- El VAE está bien: el roundtrip de un latente cacheado da RMS 1,03 respecto al
  audio original (la correlación de forma de onda ~0,74 es normal en audio; la
  comunidad mide ~95 % de oído).

---

# Fase 9 — Resultado de las pruebas reales: la voz la clona el AR

Escuchando los LoRA entrenados con datasets reales (ExtremoV4/V5):

- **El LoRA del AR consigue clonar la voz del cantante.**
- **El LoRA del NAR no la clona**, ni siquiera condicionado con los códigos
  reales (Fase 8): cambia cómo se renderizan los códigos, pero la voz sigue
  siendo la del modelo base.

Encaja con la receta de la comunidad: el LoRA de artista es el del AR, y el
del NAR es solo un adaptador del decodificador. La voz vive en los códigos y
los códigos los escribe el AR.

Cambios: la GUI, el lanzador y el README pasan a **AR = voz, timbre y sonido**
y **NAR = letra y fraseo**, y el trainer queda marcado como **experimental /
beta** (los ajustes no son definitivos y cada resultado hay que verificarlo y
ajustarlo). La etapa por defecto pasa a ser **AR**, en la GUI y en el script.

Nombres de salida con la etapa: `YuE2_NAR_LoRA_step_N` / `YuE2_FINAL_NAR_LoRA`
(antes sin sufijo) y `YuE2_AR_LoRA_step_N` / `YuE2_FINAL_AR_LoRA`. Send to
Models y la descarga añaden `_AR` / `_NAR` al nombre final (sale del metadato
`yue2_stage`); Send to Models exporta la etapa elegida y, con `both`, las dos.
No hay `_BOTH`: son dos LoRA que se cargan en sitios distintos.
