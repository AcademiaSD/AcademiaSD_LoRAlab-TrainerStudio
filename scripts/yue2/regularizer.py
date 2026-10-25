# -*- coding: utf-8 -*-
"""
Regularizador del AR de YuE2: el "minted regularizer pack" de la comunidad.

Un LoRA del experto AR sobre un set pequeño (unas 90 canciones) puede colapsar
la gramática de tokens de YuE2: el modelo se memoriza las canciones y deja de
generar música nueva. La comunidad lo evita mezclando en el entrenamiento
canciones generadas por el propio YuE2 ("minted"), que añaden variedad al
vocabulario de tokens sin enseñar la voz del artista.

Fuente: Mothersuperior/yue2-minted-corpus -> regularizer/minted_regularizer_pack.pt
(4.732 canciones: nombre, estilo, letra y sus códigos semánticos).

Referencia de la comunidad (README del tokenizer head v4):
  "The AR trainer draws 50% of its songs from it so a small artist set cannot
   collapse YuE2's token grammar; the minted_val items are the held-out check
   whose loss should stay flat."
"""
import os

import torch

from yue2.protocol import SongRequest, token_prefixes

PACK_REPO = "Mothersuperior/yue2-minted-corpus"
PACK_FILE = "regularizer/minted_regularizer_pack.pt"


def pack_dir(root):
    """Los modelos viven en Models/ desde la migracion del layout."""
    return os.path.join(root, "Models", "YuE2-Regularizer")


def pack_path(root):
    return os.path.join(pack_dir(root), "regularizer", "minted_regularizer_pack.pt")


def download_pack(root, token=None):
    """Descarga el pack (97 MB) y devuelve la ruta del .pt."""
    from huggingface_hub import hf_hub_download

    dest = pack_dir(root)
    os.makedirs(dest, exist_ok=True)
    return hf_hub_download(repo_id=PACK_REPO, filename=PACK_FILE, repo_type="dataset",
                            local_dir=dest, token=token)


class Regularizer:
    """
    Las canciones minted como (prefijo, códigos) listas para el CE del AR.

    min_codes / max_codes acotan la duración de las piezas: los logits del AR
    son [n_codigos, 184704] y con canciones de 4 minutos se van de la VRAM.
    Con 0 no se acota.
    """

    def __init__(self, path, tokenizer, cot="off", min_codes=0, max_codes=0, verbose=True):
        pack = torch.load(path, weights_only=False)
        self.items = []
        for record in pack:
            codes = record.get("codec")
            if codes is None:
                continue
            codes = [int(c) for c in codes]
            n = len(codes)
            if min_codes and n < min_codes:
                continue
            if max_codes and n > max_codes:
                continue
            request = SongRequest(style=str(record.get("style", "")),
                                  lyrics=str(record.get("lyrics", "")),
                                  cot=cot, id=str(record.get("name", "minted")))
            base_ids = token_prefixes(request, tokenizer)
            self.items.append((base_ids, codes))
        if not self.items:
            raise RuntimeError("El regularizer no tiene piezas utilizables "
                               f"(min_codes={min_codes}, max_codes={max_codes})")
        if verbose:
            largo = [len(c) for _, c in self.items]
            print(f"  Regularizer: {len(self.items)} canciones minted "
                  f"({min(largo)}-{max(largo)} codigos, {min(largo)/25:.0f}-{max(largo)/25:.0f} s)")

    def __len__(self):
        return len(self.items)

    def sample(self, rng=None):
        """Una pieza al azar: (base_ids, codigos)."""
        import random
        idx = rng.randrange(len(self.items)) if rng is not None else random.randrange(len(self.items))
        return self.items[idx]
