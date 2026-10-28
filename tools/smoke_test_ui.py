# -*- coding: utf-8 -*-
"""
smoke_test_ui.py — Prueba de humo de las interfaces de todos los entrenadores
Smoke test of every trainer interface

Arranca cada server_*.py y, en un Chrome / Edge sin ventana, hace en su página lo mismo que los botones
Pre-Cache, Start / Resume y Save JSON con el entrenamiento en marcha. No lanza ningún proceso: /api/run y
startScript se sustituyen por un registro. Falla si un botón no llega a lanzar su script, si hay un error
de JavaScript o si un guardado no devuelve 200 / ok. Así se cazan fallos como el de MiniMax-H3 (#48).

Starts every server_*.py and, in a headless Chrome / Edge, does on its page what the Pre-Cache, Start /
Resume and Save JSON (while training) buttons do. Nothing is launched: /api/run and startScript are
replaced by a recorder. It fails when a button never reaches its script, on any JavaScript error, or when
a save does not answer 200 / ok.

Se prueba una copia en una carpeta temporal (git worktree), así los guardados no tocan los ajustes de esta
instalación. Por defecto la copia es el estado actual de los ficheros versionados, cambios sin commitear
incluidos (los ficheros nuevos sin `git add` no entran).
It tests a copy in a temporary folder (git worktree), so the saves never touch this install's settings.
By default the copy is the current state of the tracked files, uncommitted changes included.

Uso / Usage (desde la raíz, con el Python del venv / from the root, with the venv's Python):
    venv\\Scripts\\python tools\\smoke_test_ui.py                 Windows
    venv/bin/python tools/smoke_test_ui.py                      Linux / RunPod
    ... --only minimaxh3,sdxl     solo esos entrenadores / only these trainers
    ... --ref origin/main         probar otro commit o rama / test another commit or branch
    ... --browser <ruta>          Chrome / Edge / Chromium concreto / a specific browser
    ... --keep                    no borrar la carpeta temporal (logs de los servidores) / keep the temp folder

Sin dependencias nuevas: habla con el navegador por su protocolo de depuración con un cliente WebSocket
mínimo. Necesita Chrome, Edge o Chromium instalado.
No new dependencies: it talks to the browser through its debugging protocol with a minimal WebSocket client.
"""
import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lo que se ejecuta en cada página. Devuelve el resultado de cada botón, los scripts que habrían arrancado
# y la respuesta de cada /api/save-*.
PAGE_TEST = r"""
(async () => {
    const api = [];
    const realFetch = window.fetch.bind(window);
    window.fetch = async (url, opts) => {
        const u = String(url);
        if (/\/api\/run(\?|$)/.test(u)) {
            return new Response(JSON.stringify({ status: 'ok' }), { headers: { 'Content-Type': 'application/json' } });
        }
        const res = await realFetch(url, opts);
        if (/\/api\/save-/.test(u)) {
            let status = null;
            try { status = (await res.clone().json()).status; } catch (e) {}
            api.push({ url: u, code: res.status, status });
        }
        return res;
    };
    const launched = [];
    window.startScript = (name) => { launched.push(name); };
    window.askConfirm = async () => true;
    window.confirm = () => true;
    window.prompt = () => null;
    const settle = (p, ms) => Promise.race([Promise.resolve(p).then(() => 'ok'),
                                            new Promise(r => setTimeout(() => r('TIMEOUT'), ms))]);
    // Que termine de cargar los ajustes guardados, como haría alguien antes de pulsar nada.
    await new Promise(r => setTimeout(r, 2000));
    const result = {};
    for (const fn of ['runPrecache', 'runTrain']) {
        if (typeof window[fn] !== 'function') { result[fn] = 'MISSING'; continue; }
        try { result[fn] = await settle(window[fn](), 10000); } catch (e) { result[fn] = 'ERROR: ' + e.message; }
    }
    const badge = document.getElementById('status-badge');
    if (badge) badge.classList.add('badge-running');
    try {
        result.saveTrainLive = typeof saveTrainSettings === 'function' ? await settle(saveTrainSettings(), 10000) : 'MISSING';
    } catch (e) { result.saveTrainLive = 'ERROR: ' + e.message; }
    if (badge) badge.classList.remove('badge-running');
    return { result, launched, api };
})()
"""


class DevTools:
    """Cliente WebSocket mínimo (RFC 6455, solo texto) para el protocolo de depuración de Chromium."""

    def __init__(self, ws_url, timeout=60):
        u = urllib.parse.urlparse(ws_url)
        self.sock = socket.create_connection((u.hostname, u.port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("DevTools handshake failed")
            resp += chunk
        head, self.buf = resp.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise ConnectionError(head.decode(errors="replace"))
        self.next_id = 0
        self.events = []

    def _send(self, text):
        data = text.encode()
        mask = os.urandom(4)
        head = bytearray([0x81])
        if len(data) < 126:
            head.append(0x80 | len(data))
        elif len(data) < 65536:
            head.append(0x80 | 126)
            head += struct.pack(">H", len(data))
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", len(data))
        self.sock.sendall(bytes(head) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("DevTools connection closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _recv(self):
        message = b""
        while True:
            b1, b2 = self._read(2)
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            if b2 & 0x80:
                self._read(4)  # el servidor no enmascara, pero por si acaso
            payload = self._read(n)
            if b1 & 0x0F == 8:
                raise ConnectionError("DevTools connection closed")
            if b1 & 0x0F in (9, 10):  # ping / pong
                continue
            message += payload
            if b1 & 0x80:
                return message.decode()

    def call(self, method, params=None):
        self.next_id += 1
        self._send(json.dumps({"id": self.next_id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self._recv())
            if msg.get("id") == self.next_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]
            self.events.append(msg)

    def wait_event(self, method, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if any(e.get("method") == method for e in self.events):
                return True
            self.sock.settimeout(max(0.1, end - time.time()))
            try:
                self.events.append(json.loads(self._recv()))
            except socket.timeout:
                break
        return False

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_browser(explicit):
    if explicit:
        return explicit
    candidates = [os.environ.get("CHROME_PATH", "")]
    for base in (os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", ""),
                 os.environ.get("LOCALAPPDATA", "")):
        if base:
            candidates += [os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"),
                           os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe")]
    candidates += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def make_copy(ref, tmp):
    """Worktree temporal con el commit pedido o, sin --ref, con los ficheros versionados tal como están ahora."""
    if not ref:
        ref = git("stash", "create") or "HEAD"  # stash create: un commit con los cambios sin commitear, sin tocar nada
    path = os.path.join(tmp, "copy")
    git("worktree", "add", "--detach", path, ref)
    return path


def start_servers(copy, names, logs):
    servers = {}
    for name in names:
        port = free_port()
        code = ("import sys, importlib; sys.path.insert(0, 'scripts'); "
                f"importlib.import_module('server_{name}').app.run(host='127.0.0.1', port={port}, "
                "debug=False, use_reloader=False)")
        log = open(os.path.join(logs, f"server_{name}.log"), "w", encoding="utf-8")
        env = {**os.environ, "LORALAB_SKIP_BROWSER": "1", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=copy, env=env, stdout=log, stderr=subprocess.STDOUT)
        servers[name] = (proc, port, log)
    return servers


def wait_listening(port, proc, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def start_browser(path, tmp):
    profile = os.path.join(tmp, "browser")
    proc = subprocess.Popen([path, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                             "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    active = os.path.join(profile, "DevToolsActivePort")
    end = time.time() + 30
    while time.time() < end and not os.path.exists(active):
        time.sleep(0.2)
    time.sleep(0.3)
    with open(active, encoding="utf-8") as f:
        return proc, int(f.readline().strip())


def devtools_http(port, path, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode() or "null")


def test_page(dport, url):
    """Abre la página, ejecuta PAGE_TEST y devuelve (resultado, errores de JavaScript)."""
    target = devtools_http(dport, "/json/new?about:blank", method="PUT")
    tools = DevTools(target["webSocketDebuggerUrl"])
    try:
        tools.call("Runtime.enable")
        tools.call("Page.enable")
        tools.call("Page.navigate", {"url": url})
        if not tools.wait_event("Page.loadEventFired", 30):
            return None, ["the page did not finish loading in 30 s"]
        tools.sock.settimeout(90)
        out = tools.call("Runtime.evaluate", {"expression": PAGE_TEST, "awaitPromise": True, "returnByValue": True})
        errors = []
        if "exceptionDetails" in out:
            errors.append(out["exceptionDetails"].get("exception", {}).get("description", "test failed"))
        for e in tools.events:
            if e.get("method") == "Runtime.exceptionThrown":
                d = e["params"]["exceptionDetails"]
                errors.append((d.get("exception") or {}).get("description") or d.get("text", "exception"))
            elif e.get("method") == "Runtime.consoleAPICalled" and e["params"].get("type") == "error":
                errors.append("console.error: " + " ".join(str(a.get("value", a.get("description", "")))
                                                           for a in e["params"].get("args", [])))
        return out.get("result", {}).get("value"), [x.splitlines()[0] for x in errors]
    finally:
        tools.close()
        try:
            devtools_http(dport, f"/json/close/{target['id']}")
        except Exception:
            pass


def check(value, errors):
    """Lista de fallos de un entrenador (vacía si todo va bien)."""
    if value is None:
        return errors or ["no result"]
    res, launched, problems = value["result"], value["launched"], list(errors)
    for fn, script in (("runPrecache", "precache"), ("runTrain", "train")):
        if res.get(fn) != "ok":
            problems.append(f"{fn}: {res.get(fn)}")
        elif script not in launched:
            problems.append(f"{fn}: did not launch '{script}'")
    if res.get("saveTrainLive") != "ok":
        problems.append(f"Save JSON while training: {res.get('saveTrainLive')}")
    for call in value["api"]:
        if call["code"] != 200 or call["status"] != "ok":
            problems.append(f"{call['url']}: HTTP {call['code']}, status {call['status']}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", help="comma-separated trainers, e.g. minimaxh3,sdxl")
    ap.add_argument("--ref", help="commit or branch to test (default: the current files)")
    ap.add_argument("--browser", help="path to Chrome / Edge / Chromium")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder with the server logs")
    args = ap.parse_args()

    browser = find_browser(args.browser)
    if not browser:
        sys.exit("[!] Chrome, Edge or Chromium not found: pass --browser <path> / no se encuentra el navegador")

    tmp = tempfile.mkdtemp(prefix="loralab_smoke_")
    logs = os.path.join(tmp, "logs")
    os.makedirs(logs)
    copy, servers, browser_proc = None, {}, None
    failed = {}
    try:
        copy = make_copy(args.ref, tmp)
        names = sorted(f[len("server_"):-3] for f in os.listdir(os.path.join(copy, "scripts"))
                       if f.startswith("server_") and f.endswith(".py"))
        if args.only:
            wanted = {n.strip().lower() for n in args.only.split(",")}
            names = [n for n in names if n.lower() in wanted]
        print(f"Testing {len(names)} trainers from {args.ref or 'the current files'} ...", flush=True)

        servers = start_servers(copy, names, logs)
        browser_proc, dport = start_browser(browser, tmp)
        for name in names:
            proc, port, _ = servers[name]
            if not wait_listening(port, proc):
                failed[name] = [f"the server did not start (see {os.path.join(logs, f'server_{name}.log')})"]
            else:
                try:
                    failed[name] = check(*test_page(dport, f"http://127.0.0.1:{port}/"))
                except Exception as exc:
                    failed[name] = [f"{type(exc).__name__}: {exc}"]
            print(f"  {'OK  ' if not failed[name] else 'FAIL'} {name}", flush=True)
            for problem in failed[name]:
                print(f"         - {problem}", flush=True)
    finally:
        for proc, _, log in servers.values():
            proc.kill()
            proc.wait()
            log.close()
        if browser_proc:
            browser_proc.kill()
            browser_proc.wait()
        if copy:
            try:
                git("worktree", "remove", "--force", copy)
            except subprocess.CalledProcessError:
                pass
        if args.keep:
            print(f"\nTemporary folder kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            git("worktree", "prune")

    bad = [n for n, p in failed.items() if p]
    print(f"\n{len(failed) - len(bad)}/{len(failed)} OK" + (f" - FAIL: {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
