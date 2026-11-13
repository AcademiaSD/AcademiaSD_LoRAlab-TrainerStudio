# -*- coding: utf-8 -*-
"""
text_io.py — Lectura de los .txt del dataset (captions, instrucciones, letras) en cualquier codificación.
Reads the dataset .txt files (captions, instructions, lyrics) in any encoding.

Los captions escritos o editados a mano en Windows no siempre están en UTF-8: el Bloc de notas
antiguo guarda en la codificación del sistema (cp1252 en Europa y América, cp936/GBK en China,
cp932 en Japón...). Leerlos como UTF-8 rompía el pre-caché con UnicodeDecodeError (#71).
"""
import codecs
import locale


def _encodings():
    """
    UTF-8 (con o sin BOM), la codificación del sistema, cp1252 y latin-1, sin repetir.
    La del sistema va antes que cp1252: cada Windows lee lo que guarda su propio Bloc de notas.
    locale.getencoding() da la de Windows aunque Python vaya en modo UTF-8 (PYTHONUTF8=1).
    latin-1 decodifica cualquier byte: es el último recurso y nunca falla.
    """
    try:
        system = locale.getencoding()
    except Exception:
        system = ""
    found, seen = [], set()
    for name in ("utf-8-sig", system, "cp1252", "latin-1"):
        try:
            key = codecs.lookup(name).name
        except LookupError:
            continue
        if key == "utf-8":
            key = "utf-8-sig"  # UTF-8 sin BOM ya lo cubre utf-8-sig
        if key not in seen:
            seen.add(key)
            found.append(name)
    return found


ENCODINGS = _encodings()


def read_text_file(path):
    """El texto de un fichero, en la primera codificación de ENCODINGS que lo decodifique."""
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")
