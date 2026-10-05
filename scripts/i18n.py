# -*- coding: utf-8 -*-
"""
i18n.py — Traducciones de la interfaz y de la consola.
Translations for the interface and the console.

Los textos viven en GUI/locales/<idioma>.json, un diccionario plano de claves ("launcher.save": "Save").
El idioma se elige en el lanzador y se guarda en settings/ui.json; el inglés es el de reserva: una clave
que falta en un idioma sale en inglés, y una que falta también en inglés sale como la propia clave.

Python:      from i18n import t;  print(t("launcher.close_window"))   # {name} -> t("x", name=...)
Navegador:   register(app) sirve /i18n.js y /api/i18n; ver GUI/i18n.js.
"""
import json
from pathlib import Path

from flask import jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent.parent
GUI_DIR = BASE_DIR / "GUI"
LOCALES_DIR = GUI_DIR / "locales"
SETTINGS_FILE = BASE_DIR / "settings" / "ui.json"
DEFAULT = "en"
# Nombre de cada idioma en su propio idioma, para el selector del lanzador.
LANGUAGES = {"en": "English", "es": "Español", "de": "Deutsch", "fr": "Français",
             "pt": "Português", "it": "Italiano", "ja": "日本語", "ko": "한국어"}

_cache = {}


def language():
    """Idioma guardado en settings/ui.json, o el inglés."""
    try:
        lang = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")).get("language", DEFAULT)
    except (OSError, ValueError):
        return DEFAULT
    return lang if lang in LANGUAGES else DEFAULT


def save_language(lang):
    if lang not in LANGUAGES:
        raise ValueError(lang)
    try:
        cfg = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = {}
    cfg["language"] = lang
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _load(lang):
    path = LOCALES_DIR / f"{lang}.json"
    mtime = path.stat().st_mtime if path.exists() else 0
    if lang not in _cache or _cache[lang][0] != mtime:
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if mtime else {}
        except ValueError:
            data = {}
        _cache[lang] = (mtime, data)
    return _cache[lang][1]


def strings(lang=None):
    """Todas las claves del idioma, con el inglés debajo para las que falten."""
    lang = lang or language()
    return {**_load(DEFAULT), **(_load(lang) if lang != DEFAULT else {})}


def t(key, **kw):
    lang = language()
    text = _load(lang).get(key) or _load(DEFAULT).get(key) or key
    try:
        return text.format(**kw) if kw else text
    except (KeyError, IndexError, ValueError):
        return text


def register(app):
    """Sirve GUI/i18n.js y /api/i18n (GET: textos del idioma actual; POST {"language"}: lo guarda)."""

    @app.route("/i18n.js")
    def i18n_js():
        return send_from_directory(str(GUI_DIR), "i18n.js")

    @app.route("/api/i18n", methods=["GET", "POST"])
    def i18n_api():
        if request.method == "POST":
            lang = str((request.get_json(force=True) or {}).get("language", ""))
            if lang not in LANGUAGES:
                return jsonify({"status": "error", "error": t("common.invalid_language")}), 400
            save_language(lang)
        lang = request.args.get("lang") if request.args.get("lang") in LANGUAGES else language()
        return jsonify({"status": "ok", "language": lang, "saved": language(),
                        "languages": list(LANGUAGES.items()), "strings": strings(lang)})
