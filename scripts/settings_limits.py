# -*- coding: utf-8 -*-
"""
settings_limits.py — Límites de los campos numéricos de los ajustes.
Limits of the number fields in the settings.

La GUI ya no deja escribir valores sin sentido (GUI/number_limits.js, con los mismos límites), pero
un json editado a mano, uno antiguo o la copia de un trabajo de la cola pueden traerlos igual. Cada
script de Pre-Cache y de entrenamiento pasa su configuración por check() nada más cargarla:

    cfg = settings_limits.check(cfg)

Un valor por debajo del mínimo se sube al mínimo y uno que no es un número (campo vacío = null) se
quita, para que el script use su valor por defecto. Cada corrección se avisa en la consola.
"""
from i18n import t

# clave del json: (mínimo, máximo o None, entero)
LIMITS = {
    # Pre-Cache
    "max_side":            (64,   None, True),
    "target_area":         (4096, None, True),
    "max_seq_len":         (1,    None, True),
    # Entrenamiento / training
    "total_steps":         (1,    None, True),
    "batch_size":          (1,    None, True),
    "grad_accum_steps":    (1,    None, True),
    "warmup_steps":        (0,    None, True),
    "lr_cycles":           (1,    None, True),
    "lora_rank":           (1,    None, True),
    "lora_alpha":          (1,    None, False),
    "lokr_factor":         (1,    None, True),
    "save_every":          (0,    None, True),
    "preview_every":       (0,    None, True),
    "preview_steps":       (1,    None, True),
    "preview_cfg":         (0,    None, False),
    "turbo_lora_strength": (0,    None, False),
}


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def check(cfg):
    """Corrige en cfg (y lo devuelve) los valores fuera de límites. Las claves que no están se dejan."""
    if not isinstance(cfg, dict):
        return cfg
    for key, (low, high, integer) in LIMITS.items():
        if key not in cfg:
            continue
        raw = cfg[key]
        value = _number(raw)
        if value is None or value != value:   # no es un número, o NaN
            del cfg[key]
            if raw is not None:
                print("[!] " + t("{name} = {value} is not a number, using the default value.", name=key, value=raw), flush=True)
            continue
        fixed = max(low, value)
        if high is not None:
            fixed = min(high, fixed)
        if integer:
            fixed = int(round(fixed))
        if fixed != value:
            print("[!] " + t("{name} = {value} is not valid, using {fixed}.", name=key, value=raw, fixed=fixed), flush=True)
        cfg[key] = fixed
    return cfg
