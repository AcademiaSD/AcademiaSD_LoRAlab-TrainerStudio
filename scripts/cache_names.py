# -*- coding: utf-8 -*-
"""
cache_names.py — Qué nombres de la caché son del trainer y cuáles del dataset
Which cache names belong to the trainer and which to the dataset

Los pre-cachés guardan sus propios archivos con un guion bajo delante: el prompt manual de las previews
(_custom*), los negativos (_neg*, _negcfg*), el relleno (_pad*)... Mirar solo el guion bajo dejaba fuera
del recuento y de las previews a las muestras cuyo archivo empieza así (_DSC0161.jpg, de muchas cámaras),
y sus archivos de caché no se borraban al quitarlas del dataset.

The pre-caches store their own files with a leading underscore: the manual preview prompt (_custom*), the
negatives (_neg*, _negcfg*), the padding (_pad*)... Checking only the underscore left dataset samples
whose file starts that way (_DSC0161.jpg, from many cameras) out of the count and of the previews, and
their cache files were never removed when they left the dataset.
"""
import re

_RESERVED = re.compile(r"_(custom\d*|neg|negcfg|pad|vision|realcodes|diagnostics|dop)(_|\.|$)")


def is_reserved(name):
    """True si `name` (un archivo de la caché o el nombre de una muestra) es del trainer, no del dataset."""
    return _RESERVED.match(name) is not None
