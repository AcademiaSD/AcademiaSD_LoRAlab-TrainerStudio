# -*- coding: utf-8 -*-
"""
console_stream.py — Lectura de la salida de los scripts para la consola web de los entrenadores.
Reads the scripts' output for the trainers' web console.

Las barras de progreso se reescriben con "\\r" (print("\\r...", end="")). El modo texto de Popen
convierte "\\r" en "\\n", así que cada paso acababa en una línea nueva: aquí se lee el pipe en
binario y una línea cerrada con "\\r" suelto queda provisional, de modo que la siguiente la
reemplaza. "\\r\\n" (fin de línea de Windows) la deja fija.
"""
import io


def read_console(pipe):
    """
    Genera (texto, reemplaza_la_anterior) por cada línea de un pipe abierto en binario.
    Quita los códigos ANSI (el ESC[A de las barras anidadas de tqdm), que la consola web
    mostraría como texto.
    """
    stdout = io.TextIOWrapper(pipe, encoding="utf-8", errors="replace", newline="")
    buffer, pending, transient = "", "", False
    while True:
        char = pending or stdout.read(1)
        pending = ""
        if not char:
            if buffer:
                yield buffer, transient
            return
        if char in "\r\n":
            if char == "\r":
                pending = stdout.read(1)
                if pending == "\n":
                    char, pending = "\n", ""
            if buffer:
                yield buffer, transient
                buffer = ""
                transient = char == "\r"
            elif char == "\n":
                transient = False
        elif char == "\x1b":
            while (char := stdout.read(1)) and not char.isalpha():
                pass
        else:
            buffer += char
