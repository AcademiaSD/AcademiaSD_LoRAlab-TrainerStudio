# -*- coding: utf-8 -*-
"""
MiniMax-H3-Image model classes, vendored from diffusers `main` (Apache-2.0,
Copyright 2026 The MiniMax and HuggingFace Teams). The diffusers commit this
project pins predates them; only the relative imports were rewritten to absolute
`diffusers.` ones. Once the pin moves past the commit that adds
`MiniMaxH3ImageTransformer2DModel`, this folder can be deleted and the two names
imported from `diffusers` directly.

Clases de MiniMax-H3-Image copiadas de diffusers `main`: el commit de diffusers
que fija el proyecto es anterior a ellas. Solo se reescribieron los imports
relativos.
"""
try:  # diffusers ya las trae / diffusers already ships them
    from diffusers import MiniMaxH3ImageTransformer2DModel, AutoencoderKLMiniMaxH3Image  # noqa: F401
except Exception:
    from .transformer_minimax_h3_image import MiniMaxH3ImageTransformer2DModel  # noqa: F401
    from .autoencoder_kl_minimax_h3_image import AutoencoderKLMiniMaxH3Image  # noqa: F401

__all__ = ["MiniMaxH3ImageTransformer2DModel", "AutoencoderKLMiniMaxH3Image"]
