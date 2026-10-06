# Informe de cambios propuestos: persistencia del entrenamiento, subidas tolerantes a fallos y notificaciones no bloqueantes

- **Proyecto:** AcademiaSD LoRAlab Trainer Studio
- **Repositorio:** https://github.com/AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio
- **Commit base analizado:** `929f194` (rama `main`, "Fix the missing closing tag in the LoKr row of four trainers")
- **Fecha del informe:** 06/10/2026
- **Entorno de prueba:** CachyOS Linux x86_64, Python 3.13.16 en `.venv` (gestionado con `uv`), GPU RTX 5090 (sm_120), CUDA 13.0, Flask 3.1.3, servidor de desarrollo de Flask con `threaded=True`.
- **Alcance:** tres cambios de comportamiento solicitados por el usuario:

| # | Cambio | Archivos afectados | Complejidad |
|---|--------|--------------------|-------------|
| A | El entrenamiento debe continuar al cerrar la pestaña, y la consola debe poder reanudarse desde cualquier navegador y quedar registrada en disco | 1 módulo nuevo + 9 servidores + 9 interfaces | Alta |
| B | Subida de dataset tolerante a fallos, con resultado por archivo y reintento solo de los fallidos | `scripts/file_transfer.py` + `GUI/file_transfer.js` | Media |
| C | Notificaciones tipo toast en lugar de `alert()`/`confirm()` del navegador | 1 archivo JS nuevo + 9 interfaces + 2 rutas | Media (muchas sustituciones mecánicas) |

## TL;DR (resumen ejecutivo)

- **Problema 1 (Cambio A):** la consola del entrenador era un stream SSE atado a la pestaña; al cerrarla, Flask cerraba el pipe y el proceso de entrenamiento moría (SIGPIPE). Además el estado se borraba y no quedaba registro.
- **Solución A:** el entrenamiento pasa a un módulo compartido (`scripts/trainer_stream.py`) que lee la salida en un hilo propio, la guarda en un backlog + `settings/trainer_console_<id>.log` y la sirve por un SSE reanudable (`/api/console/stream?since=N`). Cerrar la pestaña ya no detiene nada; al reabrir, la consola y el progreso se repintan completos desde cualquier navegador.
- **Problema 2 (Cambio B):** la subida del dataset era una sola petición: un archivo que fallaba abortaba todo el lote y el usuario no sabía cuáles se habían guardado.
- **Solución B:** subida por archivo con resultado individual (✓/✗ y motivo), los buenos se conservan, los malos se listan y hay botón "Reintentar archivos fallidos" que reenvía solo esos.
- **Problema 3 (Cambio C):** `alert()`/`confirm()` nativos bloqueaban la interfaz al guardar configuración o avisar errores.
- **Solución C:** `GUI/notify.js` con toasts no bloqueantes y confirmación propia (`askConfirm`) para acciones destructivas. Se sustituyeron 289 `alert` y 48 `confirm` en las 9 interfaces.
- **Estado:** los tres cambios están implementados y probados (44 comprobaciones automáticas + prueba HTTP en vivo de cierre y reapertura de pestaña). Sin commit, pendiente de revisión del dueño.
- **Impacto en archivos:** 2 módulos nuevos, 9 servidores, 9 interfaces, `file_transfer` (py/js) y 7 archivos de idioma. Sin dependencias nuevas (todo con Flask, stdlib y JS puro).

## Tabla de contenidos

- [TL;DR (resumen ejecutivo)](#tldr-resumen-ejecutivo)
- [0. Metodología y verificación de que este informe NO está contaminado por el `.venv` local](#0-metodología-y-verificación-de-que-este-informe-no-está-contaminado-por-el-venv-local)
- [Cambio A. Entrenamiento persistente y consola reanudable](#cambio-a-entrenamiento-persistente-y-consola-reanudable)
  - [A.1 Comportamiento actual (el problema)](#a1-comportamiento-actual-el-problema)
  - [A.2 Causa raíz (reproducida)](#a2-causa-raíz-reproducida)
  - [A.3 Solución propuesta](#a3-solución-propuesta)
  - [A.4 Código actual](#a4-código-actual-fragmentos-completos-scriptsserver_sdxlpy)
  - [A.5 Código nuevo](#a5-código-nuevo-fragmentos-completos)
  - [A.6 Criterios de aceptación del Cambio A](#a6-criterios-de-aceptación-del-cambio-a)
- [Cambio B. Subida de dataset tolerante a fallos (resultado por archivo)](#cambio-b-subida-de-dataset-tolerante-a-fallos-resultado-por-archivo)
  - [B.1 Comportamiento actual (el problema)](#b1-comportamiento-actual-el-problema)
  - [B.2 Código actual](#b2-código-actual-fragmentos-completos)
  - [B.3 Solución propuesta](#b3-solución-propuesta)
  - [B.4 Código nuevo](#b4-código-nuevo-fragmentos-completos)
  - [B.5 Criterios de aceptación del Cambio B](#b5-criterios-de-aceptación-del-cambio-b)
- [Cambio C. Notificaciones tipo toast (adiós a `alert()`/`confirm()` bloqueantes)](#cambio-c-notificaciones-tipo-toast-adiós-a-alertconfirm-bloqueantes)
  - [C.1 Comportamiento actual (el problema)](#c1-comportamiento-actual-el-problema)
  - [C.2 Solución propuesta](#c2-solución-propuesta)
  - [C.3 Código nuevo](#c3-código-nuevo-fragmento-completo)
  - [C.4 Criterios de aceptación del Cambio C](#c4-criterios-de-aceptación-del-cambio-c)
- [4. Plan de implementación sugerido](#4-plan-de-implementación-sugerido)
- [5. Riesgos y compatibilidad](#5-riesgos-y-compatibilidad)
- [6. Anexo: comandos de verificación usados](#6-anexo-comandos-de-verificación-usados)
- [7. Conclusión](#7-conclusión)
- [8. Estado de implementación y pruebas (06/10/2026)](#8-estado-de-implementación-y-pruebas-06102026)

---

## 0. Metodología y verificación de que este informe NO está contaminado por el `.venv` local

El proyecto local se ejecuta con un entorno `.venv` creado con `uv` y, para no parchear los scripts oficiales, existe un enlace simbólico local `venv -> .venv` (no versionado, incluido en `.git/info/exclude`). Antes de redactar este informe se verificó, archivo por archivo, que **todo el código analizado y propuesto es byte a byte idéntico al de `origin/main`**, de modo que las líneas citadas y los fragmentos "antes" son los del proyecto original y no los de una copia modificada.

Comprobación ejecutada:

```console
$ git diff --name-only origin/main
.gitignore
Install_LoRAlab-TrainerStudio.sh
Install_Triton_SageAtten220.sh
code/Run_LoRAlab-SDXL.sh
docs/Linux.md
```

Ninguno de esos cinco archivos forma parte de la propuesta (son el instalador adaptado a `uv`, el instalador de SageAttention, notas de documentación y el bit de ejecución de un lanzador). Sobre ellos:

- `code/Run_LoRAlab-SDXL.sh`: la única diferencia es el permiso de ejecución (`old mode 100644` -> `new mode 100755`); **el contenido es idéntico**.
- El enlace `venv -> .venv` es un archivo local no versionado; no altera ningún archivo del proyecto.

Verificación de hashes de contenido contra `origin/main` (todos `SAME`):

```console
$ for f in scripts/server_sdxl.py scripts/file_transfer.py GUI/trainer_ui_sdxl.html scripts/launcher.py ... ; do
    h1=$(git hash-object "$f"); h2=$(git rev-parse "origin/main:$f")
    [ "$h1" = "$h2" ] && echo "SAME $f" || echo "DIFF $f"
  done
SAME scripts/server_sdxl.py
SAME scripts/file_transfer.py
SAME GUI/trainer_ui_sdxl.html
SAME scripts/launcher.py
...
```

La comparación se repitió para los 9 `scripts/server_*.py`, los 9 `GUI/trainer_ui_*.html`, los scripts de entrenamiento/caption/pre-caché, `GUI/*.js` y el resto de `scripts/*.py`: **ninguno difiere**. Por eso no fue necesario crear un clon en `OG_project/`: la condición indicada era clonar únicamente si algún archivo implicado interactuara con el `.venv` de forma que pudiera alterar el contenido; ningún archivo de los que se proponen modificar está tocado por la adaptación local (la interacción con el entorno ocurre solo en los lanzadores `code/Run_*.sh`, que usan el Python del entorno, y su contenido es el de upstream).

En resumen: este informe se puede entregar al dueño del proyecto tal cual, porque describe el código original sin cambios.

---

# Cambio A. Entrenamiento persistente y consola reanudable

## A.1 Comportamiento actual (el problema)

1. Al abrir un entrenador en el puerto 5000, la interfaz hace `POST /api/run` y el servidor responde con un **streaming SSE atado a esa petición HTTP** (`Response(stream(), mimetype="text/event-stream")`).
2. Mientras el entrenamiento corre, ese único lector HTTP va pasando las líneas de la consola al navegador. La consola del navegador **es el único lugar** donde se ven los mensajes del script.
3. Si se cierra la pestaña (o el navegador completo), la conexión SSE muere, Flask cierra el generador y el proceso hijo fallece casi de inmediato; el entrenamiento se detiene.
4. Al volver a abrir la página no hay forma de ver lo que ya salió: la consola arranca vacía, el estado puede figurar como "no ejecutándose" y las previews se recargan por polling pero el registro se perdió.
5. La consola del terminal del servidor tampoco muestra la salida del entrenamiento (solo `print()` de mensajes sueltos del propio servidor y del codificador de prompts), y no queda ningún archivo de log del script de entrenamiento.

## A.2 Causa raíz (reproducida)

El problema tiene tres partes, todas en `scripts/server_sdxl.py` (idéntico en los otros 8 servidores):

1. **La lectura del pipe vive en la petición HTTP.** En `/api/run` el servidor hace el `Popen` (línea 721) y devuelve un generador `stream()` (líneas 733-755) que es quien lee `process.stdout` (líneas 740-742). Si el navegador corta, Werkzeug cierra el generador.
2. **Al morir el generador se cierra el pipe y el hijo muere.** El `except GeneratorExit: pass` (líneas 747-748) y el `finally` (líneas 749-753) liberan la referencia al `Popen`; al destruirse el objeto se cierra el extremo de lectura del pipe y el proceso de entrenamiento recibe `EPIPE`/`BrokenPipeError` (SIGPIPE) en la siguiente escritura. Además el `finally` deja `active_process = None`: el servidor "olvida" que había un entrenamiento.
3. **No hay backlog ni archivo.** La consola solo existe como stream efímero; nada guarda lo emitido, así que reabrir no puede reconstruirla.

Reproducción mínima del punto 2, ejecutada en este equipo con el Python del `.venv`:

```python
import subprocess, sys, gc, time, os
proc = subprocess.Popen([sys.executable, "-u", "-c",
    "import time,sys\nfor i in range(1000):\n    print('step', i, flush=True)\n    time.sleep(0.05)"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
pid = proc.pid
for _ in range(3):
    print("leido:", proc.stdout.readline().decode().strip())
proc.stdout.close()          # equivale a que el generador SSE muera y libere el Popen
del proc; gc.collect()
time.sleep(1.0)
state = open(f"/proc/{pid}/stat").read().split(") ", 1)[1].split()[0]
print("estado del hijo tras cerrar el pipe:", state, "(Z = zombi/muerto, S/R = vivo)")
```

Salida observada:

```console
leido: step 0
leido: step 1
leido: step 2
estado del hijo tras cerrar el pipe: Z (Z = zombi/muerto, S/R = vivo)
```

Conclusión: cerrar el pipe de lectura mata al proceso de entrenamiento. Eso es exactamente lo que ocurre al cerrar la pestaña.

## A.3 Solución propuesta

Desacoplar el entrenamiento del navegador con un módulo compartido, siguiendo el estilo de los módulos que ya existen (`console_stream.py`, `file_transfer.py`, `i18n.py`):

1. **`scripts/trainer_stream.py` (nuevo):** un hilo propio del servidor lee `process.stdout` y hace tres cosas a la vez:
   - guarda la salida en un **backlog en memoria** (`deque`, últimas 4000 líneas) numerada con un `seq`;
   - la escribe en **`settings/trainer_console_<id>.log`** (registro en disco, se corta al iniciar cada ejecución);
   - la reenvía a **todos los suscriptores SSE** conectados (puede haber varios navegadores a la vez) y la imprime también en el terminal del servidor.
2. **Nuevos endpoints:**
   - `POST /api/run`: lanza el proceso y responde JSON de inmediato (ya no streamea). Conserva los mismos códigos de error (404 script inexistente, 409 ya en ejecución).
   - `GET /api/console/stream?since=<seq>`: SSE reanudable. Primero reenvía el backlog posterior a `since` y, si la ejecución ya terminó, el evento `done` con el código de salida; después transmite en vivo. Si el cliente se desconecta, **solo se quita ese suscriptor**: el proceso y el hilo lector siguen.
   - `GET /api/console-log`: descarga del archivo `.log` (opcional, útil desde RunPod / acceso remoto).
   - `POST /api/stop`: mismo comportamiento que hoy (SIGINT; CTRL_BREAK en Windows).
3. **Frontend:** `startScript()` pasa a ser "lanzar y engancharse"; nuevo `attachConsole()` que consume el SSE reanudable; `consoleDone()` agrupa el manejo del final (éxito, pausa, código 2 de "falta Pre-Caché", error); nuevo `restoreConsole()` que al cargar la página consulta `/api/status` y, si hay entrenamiento en marcha, repinta la consola completa desde el backlog y sigue enganchado. Así se ve el progreso actual incluso abriéndolo desde otro navegador.
4. **Efectos colaterales deseados:** dos pestañas abiertas pueden ver el mismo entrenamiento; el botón Stop sigue funcionando; cerrar la terminal del entrenador sigue deteniéndolo (eso lo define `scripts/launcher.py:60-64`, es otra decisión y no se toca en esta propuesta).

## A.4 Código actual (fragmentos completos, `scripts/server_sdxl.py`)

### A.4.1 Estado global (líneas 100-102)

```python
active_process = None
active_script = None
process_lock = threading.Lock()
```

### A.4.2 `get_status()` (líneas 187-198)

```python
def get_status():
    global active_process
    global active_script

    with process_lock:
        if active_process is None:
            return {"running": False, "script": None, "pid": None}
        if active_process.poll() is not None:
            active_process = None
            active_script = None
            return {"running": False, "script": None, "pid": None}
        return {"running": True, "script": active_script, "pid": active_process.pid}
```

### A.4.3 `/api/run` con el stream atado a la petición (líneas 697-761)

```python
@app.route("/api/run", methods=["POST"])
def run_script():
    global active_process
    global active_script

    try:
        data = request.get_json(force=True) or {}
        script_name = data.get("script")
        script_path = get_script_for_name(script_name)

        if script_path is None or not script_path.exists():
            return jsonify({"status": "error", "error": t("Script not found: {name}", name=script_name)}), 404

        if encoding_stage() is not None:
            return jsonify({"status": "error", "error": t("Encoding the preview prompt, wait until it finishes.")}), 409

        with process_lock:
            if active_process is not None and active_process.poll() is None:
                return jsonify({"status": "error", "error": t("Process already running: {name}", name=active_script)}), 409

            command = [sys.executable, "-u", str(script_path)]

            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

            process = subprocess.Popen(
                command,
                cwd=str(BASE_DIR),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags
            )

            active_process = process
            active_script = script_name

        def stream():
            global active_process
            global active_script

            yield f"data: {json.dumps({'type': 'start', 'script': script_name}, ensure_ascii=False)}\n\n"

            try:
                if process.stdout is not None:
                    for text, replace in read_console(process.stdout):
                        yield f"data: {json.dumps({'type': 'output', 'text': text, 'replace': replace}, ensure_ascii=False)}\n\n"

                return_code = process.wait()
                yield f"data: {json.dumps({'type': 'done', 'script': script_name, 'code': return_code}, ensure_ascii=False)}\n\n"

            except GeneratorExit:
                pass
            finally:
                with process_lock:
                    if active_process is process:
                        active_process = None
                        active_script = None

        return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    except Exception as exc:
        with process_lock:
            active_process = None
            active_script = None
        return jsonify({"status": "error", "error": str(exc)}), 500
```

### A.4.4 `/api/stop` (líneas 764-791)

```python
@app.route("/api/stop", methods=["POST"])
def stop_script():
    global active_process
    global active_script

    with process_lock:
        process = active_process
        script = active_script

    if process is None or process.poll() is not None:
        with process_lock:
            active_process = None
            active_script = None
        return jsonify({"status": "not_running"})

    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)

        return jsonify({"status": "terminating", "script": script})
    except Exception as exc:
        try:
            process.terminate()
        except Exception:
            pass
        return jsonify({"status": "terminated", "script": script})
```

### A.4.5 Frontend actual: `startScript()` (líneas 1136-1222 de `GUI/trainer_ui_sdxl.html`)

```javascript
        async function startScript(scriptName) {
            document.getElementById('terminal').innerHTML = '';
            appendTerminal(`[SYSTEM] ${t('Launching {script}...', { script: scriptName })}`);
            setRunningState(true);

            userRequestedStop = false;
            lastExecutedStep = 0;
            totalTargetSteps = 0;

            try {
                const response = await fetch('/api/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ script: scriptName })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });

                    const lines = buffer.split('\n\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const msg = JSON.parse(line.slice(6));
                                if (msg.type === 'output') {
                                    appendTerminal(msg.text, msg.replace);

                                    const stepMatch = msg.text.match(stepRe());
                                    if (stepMatch) {
                                        lastExecutedStep = parseInt(stepMatch[1]);
                                        totalTargetSteps = parseInt(stepMatch[2]);
                                    }
                                } else if (msg.type === 'done') {
                                    if (msg.code === 0) {
                                        let successMsg = "";
                                        let msgClass = 'term-success';

                                        if (scriptName === 'train') {
                                            const totalFormSteps = parseInt(document.getElementById('train_total_steps').value) || totalTargetSteps;
                                            const isPausedEarly = userRequestedStop || (lastExecutedStep > 0 && lastExecutedStep < totalFormSteps);

                                            if (isPausedEarly) {
                                                successMsg = '\n[SYSTEM] ⏸ ' + t('Training paused successfully! Checkpoint saved.');
                                                msgClass = 'term-warning';
                                            } else {
                                                successMsg = '\n[SYSTEM] ✓ ' + t('LoRA training completed successfully!');
                                                msgClass = 'term-success';
                                            }
                                        } else if (scriptName === 'precache') {
                                            if (userRequestedStop) {
                                                successMsg = '\n[SYSTEM] ⏸ ' + t('Pre-Cache stopped by user.');
                                                msgClass = 'term-warning';
                                            } else {
                                                successMsg = '\n[SYSTEM] ✓ ' + t('Pre-Cache completed successfully!');
                                                msgClass = 'term-success';
                                            }
                                        } else {
                                            successMsg = `\n[SYSTEM] ✓ ${t('Finished successfully with code')}: ${msg.code}`;
                                        }

                                        appendTerminal(successMsg, false, msgClass);
                                    } else if (scriptName === 'train' && msg.code === 2) {
                                        appendTerminal('\n[SYSTEM] ✗ ' + t('Training could not be run: create the Pre-Cache first.'), false, 'term-danger');
                                    } else {
                                        appendTerminal(`\n[SYSTEM] ⚠ ${t('Process finished with code')}: ${msg.code}`, false, 'term-danger');
                                    }
                                }
                            } catch (e) {}
                        }
                    }
                }
            } catch (err) {
                appendTerminal(`\n[SYSTEM] ${t('Execution error')}: ${err}`, false, 'term-danger');
            } finally {
                setRunningState(false);
                loadPreviews();
                checkCheckpointInfo();
            }
        }
```

## A.5 Código nuevo (fragmentos completos)

### A.5.1 Módulo nuevo: `scripts/trainer_stream.py`

```python
# -*- coding: utf-8 -*-
"""
trainer_stream.py — Entrenamiento desacoplado del navegador.
Training decoupled from the browser.

El script de entrenamiento y la lectura de su consola viven en un hilo del servidor, no en la
petición HTTP que abre la pestaña: cerrarla ya no cierra la tubería (el proceso hijo moría al
cerrarse el pipe) y la salida queda accesible para cualquier navegador (backlog en memoria +
settings/trainer_console_<id>.log en disco).
"""
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from flask import Response, jsonify, request, send_file

from console_stream import read_console
from i18n import t

MAX_LINES = 4000        # backlog en memoria para reengancharse
PING_SECONDS = 15.0     # keep-alive del SSE a través de proxies


def _sse(kind, payload):
    return f"data: {json.dumps({'type': kind, **payload}, ensure_ascii=False)}\n\n"


class TrainerStream:
    """Ejecuta un script a la vez y mantiene su consola accesible para cualquier cliente."""

    def __init__(self, base_dir, settings_dir, trainer_id):
        self.base_dir = Path(base_dir)
        self.log_path = Path(settings_dir) / f"trainer_console_{trainer_id}.log"
        self._proc = None
        self._script = None
        self._lines = deque(maxlen=MAX_LINES)  # (seq, texto, reemplaza)
        self._seq = 0
        self._done = None                      # (seq, script, código) del último proceso
        self._lock = threading.Lock()
        self._subs = set()
        self._log = None

    # ------------------------------------------------------------------ estado
    def status(self):
        with self._lock:
            proc, script = self._proc, self._script
            if proc is None or proc.poll() is not None:
                return {"running": False, "script": None, "pid": None}
            return {"running": True, "script": script, "pid": proc.pid}

    # -------------------------------------------------------- arranque / parada
    def start(self, script_name, script_path):
        """Lanza el script; devuelve (True, None) o (False, error)."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return False, t("Process already running: {name}", name=self._script)
            self._lines.clear()
            self._done = None
            self._script = script_name
            self._proc = subprocess.Popen(
                [sys.executable, "-u", str(script_path)],
                cwd=str(self.base_dir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            proc = self._proc
        self._open_log(script_name)
        threading.Thread(target=self._pump, args=(proc, script_name), daemon=True).start()
        return True, None

    def stop(self):
        """Mismo protocolo que el /api/stop actual (SIGINT; CTRL_BREAK en Windows)."""
        with self._lock:
            proc, script = self._proc, self._script
        if proc is None or proc.poll() is not None:
            return {"status": "not_running"}
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
            return {"status": "terminating", "script": script}
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
            return {"status": "terminated", "script": script}

    # ----------------------------------------------------------------- consola
    def _open_log(self, script_name):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("w", encoding="utf-8")
            self._log.write(f"[SYSTEM] {t('Launching {script}...', script=script_name)}\n")
        except OSError:
            self._log = None

    def _pump(self, proc, script_name):
        # Único lector del pipe: backlog + .log + terminal + suscriptores SSE.
        try:
            if proc.stdout is not None:
                for text, replace in read_console(proc.stdout):
                    self._emit(text, replace)
        finally:
            self._emit_done(script_name, proc.wait())
            if self._log is not None:
                try:
                    self._log.close()
                except OSError:
                    pass
                self._log = None

    def _emit(self, text, replace=False):
        print(text, end="\r" if replace else "\n", flush=True)
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._lines.append((seq, text, replace))
            subs = list(self._subs)
            log = self._log
        if log is not None:
            try:
                log.write(text + ("\r" if replace else "\n"))
                log.flush()
            except OSError:
                pass
        for q in subs:
            q.put(("line", {"seq": seq, "text": text, "replace": replace}))

    def _emit_done(self, script_name, code):
        with self._lock:
            self._seq += 1
            self._done = (self._seq, script_name, code)
            subs = list(self._subs)
            done = self._done
        for q in subs:
            q.put(("done", {"seq": done[0], "script": script_name, "code": code}))

    def sse_stream(self, since=0):
        def gen():
            q = queue.Queue()
            with self._lock:
                self._subs.add(q)
                backlog = [item for item in self._lines if item[0] > since]
                done = self._done
            try:
                for seq, text, replace in backlog:
                    yield _sse("line", {"seq": seq, "text": text, "replace": replace})
                if done is not None and done[0] > since:
                    yield _sse("done", {"seq": done[0], "script": done[1], "code": done[2]})
                    return
                while True:
                    try:
                        kind, payload = q.get(timeout=PING_SECONDS)
                    except queue.Empty:
                        yield ": ping\n\n"
                        continue
                    yield _sse(kind, payload)
                    if kind == "done":
                        return
            except GeneratorExit:
                # La pestaña se cerró: solo se quita este suscriptor, el proceso sigue.
                pass
            finally:
                with self._lock:
                    self._subs.discard(q)
        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ rutas
    def register(self, app):
        @app.route("/api/console/stream")
        def console_stream():
            return self.sse_stream(request.args.get("since", 0, type=int))

        @app.route("/api/console-log")
        def console_log():
            if not self.log_path.exists():
                return jsonify({"status": "error", "error": t("No log yet.")}), 404
            return send_file(self.log_path, as_attachment=True, download_name=self.log_path.name)
```

Notas de diseño:

- `read_console()` se reutiliza tal cual: el parseo de barras `\r` y el filtrado de ANSI no cambian.
- El `finally` de `sse_stream()` solo desregistra al suscriptor. El proceso y el hilo lector no dependen del generador SSE, por lo que cerrar la pestaña ya no puede matarlos.
- El archivo `.log` se corta (`"w"`) al iniciar cada ejecución; para conservar la anterior bastaría con renombrarlo antes (mejora opcional).
- Las líneas transitorias (barras de progreso, `replace=True`) se escriben en el log con `\r` para que se sobrescriban y el archivo no crezca sin control.
- `PING_SECONDS` evita que proxies o túneles corten el SSE por inactividad.

### A.5.2 Integración en cada servidor (ejemplo completo: `scripts/server_sdxl.py`)

Sustituir el estado global (líneas 100-102):

```python
runtime = TrainerStream(BASE_DIR, SETTINGS_DIR, "sdxl")
```

`get_status()` (líneas 187-198) pasa a delegar, conservando el mismo JSON que espera la GUI:

```python
def get_status():
    return runtime.status()
```

`/api/run` (líneas 697-761) se reduce a lanzar y devolver JSON inmediato:

```python
@app.route("/api/run", methods=["POST"])
def run_script():
    data = request.get_json(force=True) or {}
    script_name = data.get("script")
    script_path = get_script_for_name(script_name)

    if script_path is None or not script_path.exists():
        return jsonify({"status": "error", "error": t("Script not found: {name}", name=script_name)}), 404

    if encoding_stage() is not None:
        return jsonify({"status": "error", "error": t("Encoding the preview prompt, wait until it finishes.")}), 409

    ok, error = runtime.start(script_name, script_path)
    if not ok:
        return jsonify({"status": "error", "error": error}), 409
    return jsonify({"status": "ok", "script": script_name})
```

`/api/stop` (líneas 764-791) delega:

```python
@app.route("/api/stop", methods=["POST"])
def stop_script():
    return jsonify(runtime.stop())
```

Y al final del archivo, junto a los otros registros (línea 1072-1073):

```python
i18n.register(app)
file_transfer.register(app, get_dataset_dir, get_train_output_dir, DATASET_EXTS)
runtime.register(app)
```

Deltas por servidor (mismo cambio; `trainer_id` = sufijo del archivo):

| Archivo | Estado global | `get_status` | `/api/run` | `/api/stop` | Observación |
|---|---|---|---|---|---|
| `scripts/server_anima.py` | 100-102 | 194-200 | 648-713 | 715-742 | igual |
| `scripts/server_ideogram4.py` | 100-102 | 194-200 | 648-713 | 715-742 | igual |
| `scripts/server_klein9b.py` | 101-103 | 198-204 | 701-766 | 768-795 | igual |
| `scripts/server_krea2.py` | 101-103 | 198-204 | 651-716 | 718-745 | igual |
| `scripts/server_ltx23.py` | 101-103 | 187-198 | 602-664 | 666-693 | sin `encoding_stage` (nunca tuvo la guarda) |
| `scripts/server_minimaxh3.py` | 85-87 | 234-246 | 752-848 | 850-877 | sin `encoding_stage`; ver nota |
| `scripts/server_qwenimage21.py` | 101-103 | 198-204 | 701-766 | 768-795 | igual |
| `scripts/server_sdxl.py` | 100-102 | 187-198 | 697-761 | 764-791 | ejemplo canónico |
| `scripts/server_zimage.py` | 100-102 | 194-200 | 648-713 | 715-742 | igual |

Nota MiniMax-H3: conserva un endpoint antiguo `GET /api/output` (líneas 738-749) con un `output_buffer` que **nunca se llena** (nadie hace `append`), resto de un diseño anterior. Con `trainer_stream` queda redundante; se propone eliminarlo o dejarlo sin uso, a decisión del dueño.

### A.5.3 Frontend nuevo (ejemplo completo: `GUI/trainer_ui_sdxl.html`)

Reemplaza por completo a `startScript()` (líneas 1136-1222) e incorpora las funciones de reenganche:

```javascript
        let consoleSince = 0;
        let consoleController = null;

        // El servidor guarda la consola: al abrir la página (o reabrirla en cualquier navegador)
        // se vuelve a enganchar y se repinta lo ya emitido, incluido el paso actual.
        async function attachConsole() {
            if (consoleController) consoleController.abort();
            consoleController = new AbortController();
            try {
                const response = await fetch(`/api/console/stream?since=${consoleSince}`, { signal: consoleController.signal });
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });

                    const parts = buffer.split('\n\n');
                    buffer = parts.pop();

                    for (const part of parts) {
                        if (!part.startsWith('data: ')) continue;
                        let msg = null;
                        try { msg = JSON.parse(part.slice(6)); } catch (e) { continue; }

                        if (msg.type === 'line') {
                            consoleSince = msg.seq;
                            appendTerminal(msg.text, msg.replace);

                            const stepMatch = msg.text.match(stepRe());
                            if (stepMatch) {
                                lastExecutedStep = parseInt(stepMatch[1]);
                                totalTargetSteps = parseInt(stepMatch[2]);
                            }
                        } else if (msg.type === 'done') {
                            consoleSince = msg.seq;
                            consoleDone(msg);
                        }
                    }
                }
            } catch (e) {
                // AbortError al cerrar la pestaña: el entrenamiento sigue en el servidor.
            }
        }

        function consoleDone(msg) {
            if (msg.code === 0) {
                let successMsg = "";
                let msgClass = 'term-success';

                if (msg.script === 'train') {
                    const totalFormSteps = totalTargetSteps || parseInt(document.getElementById('train_total_steps').value);
                    const isPausedEarly = userRequestedStop || (lastExecutedStep > 0 && lastExecutedStep < totalFormSteps);

                    if (isPausedEarly) {
                        successMsg = '\n[SYSTEM] ⏸ ' + t('Training paused successfully! Checkpoint saved.');
                        msgClass = 'term-warning';
                    } else {
                        successMsg = '\n[SYSTEM] ✓ ' + t('LoRA training completed successfully!');
                        msgClass = 'term-success';
                    }
                } else if (msg.script === 'precache') {
                    if (userRequestedStop) {
                        successMsg = '\n[SYSTEM] ⏸ ' + t('Pre-Cache stopped by user.');
                        msgClass = 'term-warning';
                    } else {
                        successMsg = '\n[SYSTEM] ✓ ' + t('Pre-Cache completed successfully!');
                        msgClass = 'term-success';
                    }
                } else {
                    successMsg = `\n[SYSTEM] ✓ ${t('Finished successfully with code')}: ${msg.code}`;
                }

                appendTerminal(successMsg, false, msgClass);
            } else if (msg.script === 'train' && msg.code === 2) {
                appendTerminal('\n[SYSTEM] ✗ ' + t('Training could not be run: create the Pre-Cache first.'), false, 'term-danger');
            } else {
                appendTerminal(`\n[SYSTEM] ⚠ ${t('Process finished with code')}: ${msg.code}`, false, 'term-danger');
            }

            setRunningState(false);
            loadPreviews();
            checkCheckpointInfo();
        }

        async function startScript(scriptName) {
            document.getElementById('terminal').innerHTML = '';
            consoleSince = 0;
            appendTerminal(`[SYSTEM] ${t('Launching {script}...', { script: scriptName })}`);
            setRunningState(true);

            userRequestedStop = false;
            lastExecutedStep = 0;
            totalTargetSteps = 0;

            try {
                const res = await fetch('/api/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ script: scriptName })
                });
                const data = await res.json();

                if (data.status !== 'ok') {
                    appendTerminal(`\n[SYSTEM] ${t('Execution error')}: ${data.error}`, false, 'term-danger');
                    setRunningState(false);
                    return;
                }
                attachConsole();
            } catch (err) {
                appendTerminal(`\n[SYSTEM] ${t('Execution error')}: ${err}`, false, 'term-danger');
                setRunningState(false);
            }
        }

        // Al cargar la página: si ya hay un entrenamiento en marcha, repinta la consola
        // completa y el progreso, y sigue enganchado a lo que salga.
        async function restoreConsole() {
            try {
                const st = await (await fetch('/api/status')).json();
                if (st.running) {
                    setRunningState(true);
                    attachConsole();
                }
            } catch (e) {}
        }
```

Punto de inserción de la llamada inicial, junto a las otras cargas del arranque (después de `loadPreviews();`, línea 1593):

```javascript
        setInterval(loadPreviews, 8000);
        loadPreviews();
        restoreConsole();
```

`stopScript()` no cambia. `appendTerminal()` no cambia. `setRunningState()`, `loadPreviews()` y `checkCheckpointInfo()` se reutilizan tal cual.

Ubicación de los bloques a sustituir en cada interfaz:

| Archivo | `startScript()` actual | `stopScript()` actual |
|---|---|---|
| `GUI/trainer_ui_anima.html` | 1123-1209 | 1227-1232 |
| `GUI/trainer_ui_ideogram4.html` | 1077-1163 | 1181-1186 |
| `GUI/trainer_ui_klein9b.html` | 1199-1285 | 1303-1308 |
| `GUI/trainer_ui_krea2.html` | 1131-1217 | 1235-1240 |
| `GUI/trainer_ui_ltx23.html` | 1044-1124 | 1142-1147 |
| `GUI/trainer_ui_minimaxh3.html` | 2027-2175 | 2291-2296 |
| `GUI/trainer_ui_qwenimage21.html` | 1199-1285 | 1303-1308 |
| `GUI/trainer_ui_sdxl.html` | 1136-1222 | 1240-1245 |
| `GUI/trainer_ui_zimage.html` | 1137-1223 | 1241-1246 |

Los rangos se amplían en cada archivo hasta el `finally` completo de `startScript()`; el código es idéntico salvo los números de línea (y el caso de MiniMax-H3, que tiene el bloque algo más largo pero la misma estructura). En todos los casos hay que añadir `restoreConsole();` en la zona de inicialización (junto a `loadPreviews();`).

## A.6 Criterios de aceptación del Cambio A

1. Iniciar un entrenamiento, cerrar todas las pestañas y esperar: el proceso sigue vivo (verificar con `fuser 5000/tcp`, `pgrep -f 2_train_lora` y que la GPU sigue trabajando).
2. Abrir `http://<host>:5000/` en otro navegador (o ventana privada) durante el entrenamiento: se ve la consola completa reconstruida y el progreso actualizado, y sigue saliendo salida en vivo.
3. Al terminar estando la pestaña cerrada, abrirla después muestra el backlog y el mensaje de finalización con el código correcto.
4. `settings/trainer_console_<id>.log` contiene todo lo emitido y se puede descargar con `/api/console-log`.
5. El botón Stop sigue deteniendo el entrenamiento (mismo SIGINT de hoy) y el terminal del servidor muestra también la salida del script.
6. Intentar lanzar un segundo entrenamiento devuelve 409 (el estado ya no se pierde al desconectarse el navegador).

---

# Cambio B. Subida de dataset tolerante a fallos (resultado por archivo)

## B.1 Comportamiento actual (el problema)

- El navegador sube **todos los archivos en una sola petición** (`FormData` con todos los archivos).
- El servidor recorre `request.files.getlist("files")` sin `try/except` por archivo: si un archivo falla (un `.zip` corrupto, un error de disco, un nombre problemático), la excepción corta el bucle y la petición entera termina en error 500.
- El frontend muestra un único `alert()` de error: el usuario no sabe cuáles de los archivos anteriores sí se guardaron, y no hay forma de reintentar solo los fallidos.
- El único informe disponible es `skipped` (tipos no admitidos), que se muestra como texto plano.

## B.2 Código actual (fragmentos completos)

### B.2.1 `scripts/file_transfer.py`, endpoint de subida (líneas 58-83)

```python
    @app.route("/api/upload-dataset", methods=["POST"])
    def upload_dataset():
        folder = get_dataset_dir()
        folder.mkdir(parents=True, exist_ok=True)
        saved, skipped = 0, []
        for item in request.files.getlist("files"):
            if item.filename.lower().endswith(".zip"):
                with zipfile.ZipFile(item.stream) as z:
                    for info in z.infolist():
                        dest = None if info.is_dir() or "__MACOSX" in info.filename else _target(folder, info.filename)
                        if dest is None or dest.suffix.lower() not in allowed:
                            if not info.is_dir():
                                skipped.append(info.filename)
                            continue
                        with z.open(info) as src, open(dest, "wb") as out:
                            while chunk := src.read(1 << 20):
                                out.write(chunk)
                        saved += 1
                continue
            dest = _target(folder, item.filename)
            if dest is None or dest.suffix.lower() not in allowed:
                skipped.append(item.filename)
                continue
            item.save(dest)
            saved += 1
        return jsonify({"status": "ok", "saved": saved, "skipped": skipped[:50], "path": str(folder)})
```

### B.2.2 `GUI/file_transfer.js`, subida (líneas 10-35)

```javascript
function ftSend(files) {
    if (!files || !files.length) return;
    const btn = document.getElementById('btn-ds-upload');
    const label = btn.innerText;
    const form = new FormData();
    for (const f of files) form.append('files', f, f.name);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload-dataset');
    xhr.upload.onprogress = e => { if (e.lengthComputable) btn.innerText = `⬆ ${Math.round(e.loaded / e.total * 100)}%`; };
    xhr.onloadend = () => {
        btn.innerText = label;
        btn.disabled = false;
        let data = null;
        try { data = JSON.parse(xhr.responseText); } catch (e) { }
        if (!data || data.status !== 'ok') {
            alert(t('Upload failed') + ': ' + (data && data.error || xhr.status));
            return;
        }
        let msg = '✓ ' + t('{n} file(s) uploaded', { n: data.saved }) + ` → ${data.path}`;
        if (data.skipped.length) msg += '\n\n' + t('Skipped (unsupported type):') + `\n${data.skipped.join('\n')}`;
        alert(msg);
        if (typeof loadDatasetInfo === 'function') loadDatasetInfo();
    };
    btn.disabled = true;
    xhr.send(form);
}
```

## B.3 Solución propuesta

1. **Servidor (`scripts/file_transfer.py`):** recorrer los archivos con `try/except` individual, y dentro de cada `.zip`, con `try/except` por entrada y por ZIP corrupto. La respuesta nunca es 500 por un archivo fallido; devuelve:
   - `saved` (cantidad, compatible con el código viejo) y `saved_names` (lista),
   - `skipped` (tipo no admitido o ruta no válida),
   - `failed` (lista de `{"name", "error"}`),
   - `status`: `"ok"` o `"partial"`.
2. **Frontend (`GUI/file_transfer.js`):** subir **un archivo por petición, en cola secuencial**, de modo que un fallo no impida el resto. Por cada archivo se pinta una fila en un panel de estado con `✓` (subido) o `✗` (motivo del fallo). Al terminar, un resumen con toast (Cambio C) y un botón **"Reintentar archivos fallidos"** que reencola únicamente esos. Los ya subidos no se vuelven a enviar.
3. **Panel de estado:** se inyecta bajo la grilla del dataset (`#ds-grid`), sin tocar el HTML de cada entrenador (mismo patrón que ya usa `file_transfer.js` con `#ft-modal`).

## B.4 Código nuevo (fragmentos completos)

### B.4.1 `scripts/file_transfer.py`, endpoint nuevo

```python
    @app.route("/api/upload-dataset", methods=["POST"])
    def upload_dataset():
        folder = get_dataset_dir()
        folder.mkdir(parents=True, exist_ok=True)
        saved, skipped, failed = [], [], []

        def save_stream(src, dest, name):
            try:
                with open(dest, "wb") as out:
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                saved.append(dest.name)
            except OSError as exc:
                failed.append({"name": name, "error": str(exc)})

        for item in request.files.getlist("files"):
            name = item.filename or ""
            if name.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(item.stream) as z:
                        for info in z.infolist():
                            if info.is_dir() or "__MACOSX" in info.filename:
                                continue
                            dest = _target(folder, info.filename)
                            if dest is None or dest.suffix.lower() not in allowed:
                                skipped.append(info.filename)
                                continue
                            try:
                                with z.open(info) as src:
                                    save_stream(src, dest, info.filename)
                            except OSError as exc:
                                failed.append({"name": info.filename, "error": str(exc)})
                except (zipfile.BadZipFile, OSError) as exc:
                    failed.append({"name": name, "error": str(exc)})
                continue

            dest = _target(folder, name)
            if dest is None or dest.suffix.lower() not in allowed:
                skipped.append(name)
                continue
            try:
                item.save(dest)
                saved.append(dest.name)
            except OSError as exc:
                failed.append({"name": name, "error": str(exc)})

        return jsonify({
            "status": "ok" if not failed else "partial",
            "saved": len(saved),
            "saved_names": saved,
            "skipped": skipped[:50],
            "failed": failed[:50],
            "path": str(folder),
        })
```

Nota: se conserva `saved` como número para compatibilidad con cualquier interfaz vieja en caché; `saved_names` es la lista nueva.

### B.4.2 `GUI/file_transfer.js`, subida nueva (reemplaza a `ftSend`)

```javascript
// Subida de dataset con resultado por archivo: un fallo no cancela el resto y se puede
// reintentar solo lo fallido. Los avisos usan los toasts de /notify.js (Cambio C).
let ftFailedFiles = [];

function ftUpload() {
    const input = document.getElementById('ds-upload');
    input.value = '';
    input.click();
}

function ftSend(files) {
    const list = [...(files || [])];
    if (!list.length) return;
    ftException();
    ftStatusBox().style.display = 'block';
    ftRunQueue(list);
}

async function ftRunQueue(files) {
    const btn = document.getElementById('btn-ds-upload');
    const label = btn.innerText;
    let ok = 0;
    ftFailedFiles = [];

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        btn.disabled = true;
        btn.innerText = `⬆ ${i + 1}/${files.length}`;
        ftRow(file, '…', t('Uploading {i}/{n}', { i: i + 1, n: files.length }), 'pending');
        try {
            const data = await ftPostOne(file);
            const bad = (data.failed || []).find(f => f.name === file.name);
            if (bad) throw new Error(bad.error);
            if (data.status !== 'ok' && data.status !== 'partial') throw new Error(data.error || `HTTP ${data.http}`);
            ftRow(file, '✓', t('Uploaded'), 'ok');
            ok++;
        } catch (err) {
            ftRow(file, '✗', String(err.message || err), 'fail');
            ftFailedFiles.push(file);
        }
    }

    btn.disabled = false;
    btn.innerText = label;
    ftRetryButton();

    if (typeof loadDatasetInfo === 'function') loadDatasetInfo();

    if (ftFailedFiles.length) {
        toast(t('{ok} uploaded, {n} failed. Use Retry failed.', { ok: ok, n: ftFailedFiles.length }), 'warning');
    } else if (files.length) {
        toast(t('{n} file(s) uploaded', { n: ok }), 'success');
    }
}

function ftPostOne(file) {
    return new Promise((resolve, reject) => {
        const form = new FormData();
        form.append('files', file, file.name);
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/upload-dataset');
        xhr.onloadend = () => {
            let data = null;
            try { data = JSON.parse(xhr.responseText); } catch (e) {}
            if (!data) { reject(new Error(`HTTP ${xhr.status}`)); return; }
            data.http = xhr.status;
            resolve(data);
        };
        xhr.send(form);
    });
}

// --- Panel de estado por archivo -------------------------------------------------
function ftStatusBox() {
    let box = document.getElementById('ft-upload-status');
    if (!box) {
        box = document.createElement('div');
        box.id = 'ft-upload-status';
        box.style.cssText = 'display:none;margin-top:8px;max-height:180px;overflow:auto;font-size:0.82rem;'
            + 'background:var(--bg-dark);border:1px solid var(--panel-border);border-radius:8px;padding:8px;';
        const grid = document.getElementById('ds-grid');
        grid.parentElement.insertBefore(box, grid.nextSibling);
    }
    return box;
}

function ftException() {
    ftFailedFiles = [];
}

function ftRow(file, icon, text, cls) {
    const box = ftStatusBox();
    let row = box.querySelector(`[data-file="${CSS.escape(file.name)}"]`);
    if (!row) {
        row = document.createElement('div');
        row.dataset.file = file.name;
        row.style.cssText = 'display:flex;gap:8px;padding:2px 0;color:var(--text-muted);';
        row.innerHTML = '<span class="ft-icon"></span>'
            + '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>'
            + '<span class="ft-detail"></span>';
        box.appendChild(row);
    }
    row.className = `ft-${cls}`;
    row.querySelector('.ft-icon').innerText = icon;
    row.querySelector('span:nth-child(2)').innerText = file.name;
    row.querySelector('.ft-detail').innerText = text;
}

function ftRetryButton() {
    const box = ftStatusBox();
    let btn = document.getElementById('ft-retry');
    if (!btn) {
        btn = document.createElement('button');
        btn.id = 'ft-retry';
        btn.type = 'button';
        btn.className = 'btn btn-warning btn-sm';
        box.appendChild(btn);
    }
    btn.innerText = t('Retry failed files');
    btn.style.display = ftFailedFiles.length ? '' : 'none';
    btn.onclick = () => {
        const again = ftFailedFiles.slice();
        if (again.length) ftSend(again);
    };
}
```

El resto del archivo (descarga, `#ft-modal`, zona de arrastre) no cambia; la zona de arrastre ya llama a `ftSend(...)`, con lo que hereda el comportamiento nuevo.

## B.5 Criterios de aceptación del Cambio B

1. Subir 5 archivos válidos + 1 rechazado (por ejemplo un `.exe` o un `.zip` corrupto): los 5 válidos quedan en el dataset, el inválido aparece con `✗` y motivo, y el resto se sube igualmente.
2. El contador del Dataset Manager refleja los archivos subidos sin recargar la página.
3. "Reintentar archivos fallidos" solo reenvía los fallidos; los ya subidos no se repiten.
4. Un error de escritura (permiso o disco) en un archivo tampoco cancela el resto.
5. Ningún `alert()` en el flujo de subida: todo se informa con el panel y con toasts.

---

# Cambio C. Notificaciones tipo toast (adiós a `alert()`/`confirm()` bloqueantes)

## C.1 Comportamiento actual (el problema)

Cuando la interfaz guarda configuración, exporta o avisa de un error, usa diálogos nativos del navegador (`alert`/`confirm`). El `alert` congela la pestaña entera hasta pulsar Aceptar, lo que hace incómodo guardar ajustes repetidamente (por ejemplo, cada guardado de configuración de entrenamiento muestra un diálogo modal). Inventario actual (recuento literal sobre el commit `929f194`):

| Archivo | `alert(` | `confirm(` |
|---|---|---|
| `GUI/trainer_ui_anima.html` | 30 | 5 |
| `GUI/trainer_ui_ideogram4.html` | 30 | 5 |
| `GUI/trainer_ui_klein9b.html` | 32 | 5 |
| `GUI/trainer_ui_krea2.html` | 30 | 5 |
| `GUI/trainer_ui_ltx23.html` | 30 | 5 |
| `GUI/trainer_ui_minimaxh3.html` | 41 | 8 |
| `GUI/trainer_ui_qwenimage21.html` | 32 | 5 |
| `GUI/trainer_ui_sdxl.html` | 32 | 5 |
| `GUI/trainer_ui_zimage.html` | 30 | 5 |
| `GUI/file_transfer.js` | 2 | 0 |
| `GUI/launcher.html` | 0 | 0 |
| **Total** | **289** | **48** |

Ejemplos representativos del código actual:

`GUI/trainer_ui_sdxl.html:1089` (guardado de configuración de entrenamiento):

```javascript
                alert('✓ ' + t('Train settings saved to:') + `\n • ${listStr}${live}${encoding}`);
```

`GUI/trainer_ui_sdxl.html:1021` (guardado de Pre-Caché):

```javascript
                if (!silent) alert('✓ ' + t('Pre-Cache settings saved to:') + `\n • ${listStr}`);
```

`GUI/trainer_ui_sdxl.html:1376` (borrado de imagen del dataset, así se usa `confirm`):

```javascript
            if (!confirm(t('Remove "{name}" and its .txt caption from the dataset?', { name: file }) + '\n\n' + t('The files are deleted from disk.'))) return;
```

## C.2 Solución propuesta

1. **`GUI/notify.js` (nuevo, compartido):** define `toast(mensaje, tipo, duración)` (aviso flotante no bloqueante, con cierre manual y autodescarte) y `askConfirm(mensaje, opciones)` (Promise<boolean> con un diálogo propio del proyecto, sin congelar la pestaña; se cierra con Escape, con el botón Cancelar o pulsando fuera). Inyecta su propio CSS, así que no hay que tocar el CSS de cada interfaz.
2. **Ruta `/notify.js`:** se sirve desde `scripts/file_transfer.py` (que ya está registrado en los 9 servidores), con dos líneas:

```python
    @app.route("/notify.js")
    def notify_js():
        return send_from_directory(str(GUI_DIR), "notify.js")
```

3. **Inclusión en cada interfaz:** una etiqueta nueva junto a la de i18n (línea 7 de cada `trainer_ui_*.html`). Debe ir **antes** de `/file_transfer.js` (que ahora usa `toast`):

```html
    <script src="/i18n.js"></script>
    <script src="/notify.js"></script>
```

4. **Conversión mecánica:**
   - `alert(X)` -> `toast(X)` para avisos informativos (los saltos de línea se conservan porque el toast usa `white-space: pre-line`).
   - `alert(X)` de error -> `toast(X, 'error')`; de éxito -> `toast(X, 'success')` (opcional, se puede hacer en una segunda pasada).
   - `confirm(X)` -> `await askConfirm(X)` en funciones que pasan a `async` (las 48 llamadas están en funciones que ya lo son o pueden serlo sin más cambios; ver ejemplo abajo).

## C.3 Código nuevo (fragmento completo)

### C.3.1 `GUI/notify.js`

```javascript
// notify.js — Avisos y confirmaciones propios, compartidos por todas las GUI.
// Sustituyen a alert()/confirm() del navegador: no congelan la página y encajan con el tema.

const NOTIFY_CSS = `
#toast-stack { position: fixed; top: 16px; right: 16px; z-index: 3000; display: flex;
    flex-direction: column; gap: 8px; max-width: min(420px, 90vw); }
.toast { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border-radius: 8px;
    background: var(--panel-bg, #1e293b); border: 1px solid var(--panel-border, #334155);
    color: var(--text-main, #f8fafc); font-size: 0.85rem; white-space: pre-line;
    word-break: break-word; box-shadow: 0 8px 24px rgba(0,0,0,0.45); animation: toast-in 0.18s ease-out; }
.toast.success { border-color: #16a34a; }
.toast.warning { border-color: #d97706; }
.toast.error   { border-color: #dc2626; }
.toast .toast-close { cursor: pointer; opacity: 0.6; border: 0; background: none;
    color: inherit; font-size: 1rem; line-height: 1; }
.toast .toast-close:hover { opacity: 1; }
@keyframes toast-in { from { transform: translateX(12px); opacity: 0; } to { transform: none; opacity: 1; } }
#ask-modal { position: fixed; inset: 0; z-index: 3100; display: none; align-items: center;
    justify-content: center; background: rgba(2,6,23,0.75); }
#ask-modal.active { display: flex; }
#ask-modal .ask-box { width: min(520px, 92vw); display: flex; flex-direction: column; gap: 14px;
    background: var(--panel-bg, #1e293b); border: 1px solid var(--panel-border, #334155);
    border-radius: 12px; padding: 18px; color: var(--text-main, #f8fafc); font-size: 0.9rem;
    white-space: pre-line; }
#ask-modal .ask-actions { display: flex; justify-content: flex-end; gap: 8px; }
`;

function _notifyInit() {
    if (document.getElementById('toast-stack')) return;
    const style = document.createElement('style');
    style.textContent = NOTIFY_CSS;
    document.head.appendChild(style);
    const stack = document.createElement('div');
    stack.id = 'toast-stack';
    document.body.appendChild(stack);
    const modal = document.createElement('div');
    modal.id = 'ask-modal';
    modal.innerHTML = '<div class="ask-box"><div class="ask-text"></div>'
        + '<div class="ask-actions">'
        + '<button type="button" class="btn btn-secondary btn-sm ask-cancel"></button>'
        + '<button type="button" class="btn btn-primary btn-sm ask-ok"></button>'
        + '</div></div>';
    document.body.appendChild(modal);
}

function toast(message, type = 'info', timeout = 5000) {
    _notifyInit();
    const stack = document.getElementById('toast-stack');
    const box = document.createElement('div');
    box.className = `toast ${type}`;
    const text = document.createElement('div');
    text.style.flex = '1';
    text.textContent = String(message);
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'toast-close';
    close.innerText = '×';
    close.onclick = () => box.remove();
    box.append(text, close);
    stack.appendChild(box);
    if (timeout > 0) setTimeout(() => box.remove(), timeout);
    return box;
}

function askConfirm(message, options = {}) {
    _notifyInit();
    const modal = document.getElementById('ask-modal');
    const okLabel = options.okLabel || (typeof t === 'function' ? t('Continue') : 'Continue');
    const cancelLabel = options.cancelLabel || (typeof t === 'function' ? t('Cancel') : 'Cancel');
    modal.querySelector('.ask-text').textContent = String(message);
    const ok = modal.querySelector('.ask-ok');
    const cancel = modal.querySelector('.ask-cancel');
    ok.innerText = okLabel;
    cancel.innerText = cancelLabel;

    return new Promise(resolve => {
        const finish = value => {
            modal.classList.remove('active');
            ok.onclick = cancel.onclick = modal.onclick = null;
            document.removeEventListener('keydown', onKey);
            resolve(value);
        };
        const onKey = e => { if (e.key === 'Escape') finish(false); };
        ok.onclick = () => finish(true);
        cancel.onclick = () => finish(false);
        modal.onclick = e => { if (e.target === modal) finish(false); };
        document.addEventListener('keydown', onKey);
        modal.classList.add('active');
        cancel.focus();
    });
}
```

### C.3.2 Ejemplos de conversión "antes y después"

Guardado de configuración (no bloqueante):

```javascript
// ANTES (GUI/trainer_ui_sdxl.html:1089)
alert('✓ ' + t('Train settings saved to:') + `\n • ${listStr}${live}${encoding}`);

// DESPUÉS
toast('✓ ' + t('Train settings saved to:') + `\n • ${listStr}${live}${encoding}`, 'success');
```

Confirmación de borrado (misma función, ya era `async`):

```javascript
// ANTES (GUI/trainer_ui_sdxl.html:1375-1376)
async function deleteDatasetImage(file) {
    ...
    if (!confirm(t('Remove "{name}" and its .txt caption from the dataset?', { name: file }) + '\n\n' + t('The files are deleted from disk.'))) return;

// DESPUÉS
async function deleteDatasetImage(file) {
    ...
    if (!await askConfirm(t('Remove "{name}" and its .txt caption from the dataset?', { name: file }) + '\n\n' + t('The files are deleted from disk.'),
                          { okLabel: t('Delete'), cancelLabel: t('Cancel') })) return;
```

Para las ~48 confirmaciones, la conversión es: `if (!confirm(X)) return;` -> `if (!await askConfirm(X)) return;` en funciones `async`. En los pocos casos donde la función no sea `async`, se añade `async` (todas son manejadores de eventos o llamadas desde funciones async, no requieren más cambios).

### C.3.3 Claves de traducción nuevas

Claves nuevas a añadir en `GUI/locales/*.json` (la i18n cae al inglés si falta una clave, así que no rompe nada agregarlas después). Ejemplo para `es.json`:

```json
  "Continue": "Continuar",
  "Delete": "Borrar",
  "Uploaded": "Subido",
  "Uploading {i}/{n}": "Subiendo {i}/{n}",
  "Retry failed files": "Reintentar archivos fallidos",
  "{ok} uploaded, {n} failed. Use Retry failed.": "{ok} subidos, {n} fallidos. Usá Reintentar archivos fallidos.",
  "No log yet.": "Todavía no hay registro.",
  "Download console log": "Descargar registro de consola"
```

### C.3.4 Opcional (mejora de A): enlace de descarga del log

Añadir en el bloque de la consola de cada interfaz un enlace discreto:

```html
<a href="/api/console-log" download class="btn btn-secondary btn-sm">⬇ Log</a>
```

## C.4 Criterios de aceptación del Cambio C

1. Guardar configuración de Pre-Caché y de Entrenamiento: aparece un toast arriba a la derecha y la página sigue usable al instante.
2. Error de red al guardar: toast rojo, sin diálogo nativo.
3. Acción destructiva (Clear Dataset, borrar imagen): aparece el diálogo propio del proyecto (mismo look que los modales existentes), con Escape/Cancelar/clic fuera; al aceptar se ejecuta.
4. `grep -c "alert(\|confirm(" GUI/trainer_ui_*.html GUI/file_transfer.js` devuelve 0 en todos.
5. La traducción de los nuevos textos funciona en los 8 idiomas (o cae a inglés si aún no se añadió la clave).

---

# 4. Plan de implementación sugerido

| Fase | Trabajo | Estimación |
|---|---|---|
| 1 | Crear `scripts/trainer_stream.py` y probarlo en `server_sdxl.py` (canónico) | 0.5 día |
| 2 | Portar la integración a los 8 servidores restantes (mecánico, con la tabla de líneas) | 0.5 día |
| 3 | Frontend: `attachConsole`/`consoleDone`/`restoreConsole` en `trainer_ui_sdxl.html`, probar cierre y reapertura de pestaña | 0.5 día |
| 4 | Portar frontend a las 8 interfaces restantes | 0.5 día |
| 5 | Cambio B: endpoint y panel de subida por archivo | 0.5 día |
| 6 | Cambio C: `notify.js`, ruta, inclusión y sustitución mecánica de `alert`/`confirm` (10 archivos, 337 llamadas) | 1 día |
| 7 | Pruebas cruzadas (navegador local, móvil/otro equipo con acceso remoto, RunPod) y ajustes | 0.5 día |

Orden recomendado: A -> B -> C (C depende de A y B solo para las sustituciones dentro de los bloques nuevos; conviene hacer C al final para no repetir sustituciones).

# 5. Riesgos y compatibilidad

- **Compatibilidad de API:** `/api/run` cambia de SSE a JSON. Cualquier cliente externo que dependiera del stream viejo dejaría de funcionar; en la práctica el único consumidor es `trainer_ui_*.html`. `POST /api/stop` y `GET /api/status` mantienen formato.
- **Acceso remoto / RunPod:** al desacoplar el proceso del navegador, un entrenamiento sobrevive a cortes de red del cliente; conviene que el dueño sepa que la sesión SSH/terminal que arrancó el servidor sigue siendo la que mantiene vivo al entrenador (comportamiento de `launcher.py:60-64`, fuera de alcance).
- **Memoria:** el backlog guarda 4000 líneas por entrenador (unas pocas centenas de KB); acotado por diseño.
- **Rotación de log:** el `.log` se corta al iniciar la siguiente ejecución. Si se quiere conservar el historial completo, se puede añadir marca de fecha al nombre; se deja como decisión del dueño.
- **MiniMax-H3:** tiene código muerto (`/api/output` y `output_buffer` nunca alimentado); limpiarlo es opcional y no forma parte estricta de esta propuesta.
- **Rendimiento del SSE en Flask de desarrollo:** cada navegador conectado ocupa un hilo (el servidor ya usa `threaded=True`); es el mismo modelo actual, sin cambios de carga apreciables.

# 6. Anexo: comandos de verificación usados

```console
# 0) Confirmar que el código analizado es el de upstream (sin contaminar por el .venv local)
git -C AcademiaSD_LoRAlab-TrainerStudio log --oneline -1
git -C AcademiaSD_LoRAlab-TrainerStudio diff --name-only origin/main

# A) Reproducción de la muerte del hijo al cerrarse el pipe (usada en A.2)
.venv/bin/python <repro_arriba>

# Recuento actual de diálogos nativos (inventario del Cambio C)
for f in GUI/trainer_ui_*.html GUI/*.js; do
  echo "$f alert=$(grep -c 'alert(' "$f") confirm=$(grep -c 'confirm(' "$f")"
done

# Rutas y líneas citadas del Cambio A
grep -n '@app.route("/api/\(run\|stop\|status\)"' scripts/server_*.py
```

# 7. Conclusión

Los tres cambios son viables sin alterar la arquitectura general del proyecto:

- **A** introduce un único módulo compartido (`scripts/trainer_stream.py`) y una conversión mecánica en 9 servidores y 9 interfaces; es el cambio más grande y el que más valor aporta (entrenamientos que no se pierden y consola reanudable desde cualquier dispositivo, con log en disco).
- **B** es un cambio acotado a dos archivos compartidos, sin duplicación por entrenador.
- **C** es una sustitución masiva pero de bajo riesgo (`alert` -> `toast`, `confirm` -> `askConfirm`) con un solo archivo JS nuevo y una ruta ya registrada en todos los servidores.

Ninguno de los fragmentos "antes" de este informe está afectado por el `.venv` local (ver sección 0), por lo que el documento puede compartirse con el dueño del proyecto sin riesgo de describir código distinto al del repositorio.

---

# 8. Estado de implementación y pruebas (06/10/2026)

Los tres cambios de este informe ya están **implementados y probados** en el árbol de trabajo local (sobre el commit `929f194`), pendientes de revisión y commit por el dueño.

**Archivos nuevos**

- `scripts/trainer_stream.py`: módulo compartido de ejecución y consola reanudable.
- `GUI/notify.js`: toasts y confirmación propia del proyecto.

**Archivos modificados**

- Cambio A: los 9 `scripts/server_*.py` (integración de `TrainerStream`, rutas nuevas, `/api/output` legado de MiniMax reescrito con `last_code()`) y los 9 `GUI/trainer_ui_*.html` (`attachConsole`, `consoleDone`, `startScript` nuevo, `restoreConsole()` al cargar).
- Cambio B: `scripts/file_transfer.py` (subida por archivo con `saved_names`/`failed`/`status: partial`) y `GUI/file_transfer.js` (cola por archivo, panel ✓/✗, botón de reintento).
- Cambio C: los 9 `GUI/trainer_ui_*.html` (0 usos de `alert`/`confirm`, 289 -> `toast` y 48 -> `await askConfirm`), `GUI/file_transfer.js` y las 8 claves nuevas en los 7 `GUI/locales/*.json`.
- Ruta `/notify.js` servida desde `scripts/file_transfer.py` (ya registrado por los 9 servidores).

**Desviaciones menores respecto al borrador de este informe**

1. `TrainerStream.sse_stream()` cierra la conexión en cuanto reenvía el evento `done` de una ejecución terminada (si `since` ya lo vio, cierra sin esperar), para no dejar clientes colgados.
2. Se añadió `TrainerStream.last_code()` para mantener compatibles con el endpoint heredado `/api/output` de MiniMax-H3.
3. En `GUI/file_transfer.js` se eliminó un helper redundante del borrador; el reintento reutiliza `ftSend` con los fallidos.

**Evidencia de pruebas ejecutadas**

- Suite automática (44 comprobaciones, todas OK): supervivencia del proceso tras desconectar el SSE, backlog completo y evento `done` al reengancharse, cursor `since`, log en disco, `/api/console/stream`, `/api/console-log`, 409 al lanzar dos veces, Stop, subida parcial con `saved`/`skipped`/`failed`, y verificación de las 9 interfaces (sin `alert`/`confirm`, con `notify.js`).
- Prueba en vivo por HTTP contra un servidor real en el puerto 5000: corte del stream a 1 s (`curl --max-time 1`), el hijo siguió vivo (`status.running = true`, mismo PID), segundo `POST /api/run` devolvió 409, el reenganche recibió 42 líneas + `done` con código 0, `/api/console-log` devolvió el log completo y `POST /api/stop` detuvo el proceso.
- Validación de sintaxis: todos los `scripts/*.py` compilan, los 9 bloques JS de las interfaces pasan `node --check` (Node 26), `notify.js` y `file_transfer.js` también.
- Backup de cada archivo tocado como `<archivo>.bak_06_10_26_08-13` (y `.bak_06_10_26_08-21` para este informe); sin commits, como pide el flujo del proyecto.

---

*Informe generado el 06/10/2026 sobre el commit `929f194` del repositorio AcademiaSD/AcademiaSD_LoRAlab-TrainerStudio.*
