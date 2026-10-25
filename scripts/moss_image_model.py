# -*- coding: utf-8 -*-
"""
moss_image_model.py — Transformer de MOSS-Image para el entrenador LoRA.
MOSS-Image transformer for the LoRA trainer.

Adaptado de pipeline_moss_image.py de OpenMOSS (https://github.com/OpenMOSS/MOSS-Image,
Apache License 2.0, Copyright OpenMOSS Team). Los nombres de los pesos son los originales, que son
también los que usa el nodo de ComfyUI, así que el LoRA se exporta sin renombrar nada.

Cambios respecto al original:
- Las activaciones se pasan a BF16 con el dtype de las normas, no con el de los pesos lineales:
  en NF4 esos pesos son uint8.
- Gradient checkpointing por bloque (enable_gradient_checkpointing de diffusers).
- Sin las comprobaciones de forma y de máscara del pipeline (la del contenido de la máscara
  sincroniza la GPU en cada paso): los datos vienen siempre del pre-caché.
- Imágenes de referencia para los LoRAs de edición (armV4 y chinoll/Moss-Image-Edit, contrato
  "ref-prefix-only-t0"): sus latentes van delante de la secuencia con el timestep 0, la posición
  (índice de la referencia, 0, fila, columna) y una atención que solo mira a las referencias; el
  objetivo y el texto ven toda la secuencia. reference_layout elige dónde quedan la imagen y el
  texto respecto a ellas (los edit_format de ComfyUI). Sin referencias, el modelo es el original.
"""
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads, prefix=0):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2).to(v.dtype)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2).to(v.dtype)
    v = v.reshape(shape).transpose(1, 2)
    if not prefix:
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
    # Las referencias solo se miran entre sí; el objetivo y el texto miran también a las referencias.
    refs = F.scaled_dot_product_attention(q[:, :, :prefix], k[:, :, :prefix], v[:, :, :prefix])
    body = F.scaled_dot_product_attention(q[:, :, prefix:], k, v)
    return torch.cat((refs, body), 2).transpose(1, 2).flatten(2)


def _segments(x, prefix, fn):
    # fn(trozo, fila de modulación): fila 0 para las referencias y 1 para el resto; sin referencias, solo la 0.
    if not prefix:
        return fn(x, 0)
    return torch.cat((fn(x[:, :prefix], 0), fn(x[:, prefix:], 1)), 1)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, dim, bias=False) for name in ('q', 'k', 'v', 'o')
        })
        self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation, prefix=0):
        # modulation [1, 6, dim]; con referencias [2, 6, dim]: la fila 0 es la de los primeros prefix tokens.
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = _segments(self.norm1(x), prefix, lambda s, i: s * (1 + scale_a[i]) + shift_a[i])
        dtype = self.norm1.weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        attn = self.proj['o'](_attention(q, k, self.proj['v'](y), rope, self.heads, prefix))
        x = x + _segments(attn, prefix, lambda s, i: s * gate_a[i])
        y = _segments(self.norm2(x), prefix, lambda s, i: s * (1 + scale_f[i]) + shift_f[i])
        hidden, gate = self.proj['ffn_up_gate'](y.to(dtype)).chunk(2, -1)
        return x + _segments(self.proj['ffn_down'](_situ(hidden, gate)), prefix, lambda s, i: s * gate_f[i])


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer('bands', self._default_bands(torch.device('cpu')))

    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device='cpu', dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Frecuencias derivadas: .to(bf16) las redondearía.
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device, offset=0):
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        # Coordenadas (r, f, h, w); la secuencia es la imagen seguida del texto. offset desplaza f de
        # los dos (el formato ai-toolkit lo pone a -text_length: la imagen en f = 0, con sus referencias).
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length + offset), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device) + offset
        return self._encode(torch.cat((image, text), 0))

    def reference(self, height, width, index, device):
        # Referencia número index (desde 1): (index, 0, fila, columna), centradas como la imagen.
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        coords = torch.stack((torch.full_like(h, index), torch.zeros_like(h), h, w), -1).flatten(0, 1).float()
        coords[:, 2] -= (height - 1) / 2
        coords[:, 3] -= (width - 1) / 2
        return self._encode(coords)

    def _encode(self, coords):
        phase = coords.unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    # Posición de las referencias de edición, como los edit_format del nodo de ComfyUI:
    # "moss-image-edit" (chinoll: la imagen sigue detrás del texto) o "ai-toolkit" (la imagen y sus
    # referencias en f = 0, el texto antes).
    reference_layout = "moss-image-edit"

    @register_to_config
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42,
                 text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0):
        super().__init__()
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.gradient_checkpointing = False

    def _time(self, times):
        # El entrenamiento original calcula la sinusoide en float64 y pasa float32 al MLP.
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(len(times), 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(len(times), 2, -1)
        return mod, head

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, skip_layers=(),
                references=None):
        """
        hidden_states [B, 128, H, W] (latente FLUX.2 en parches 2x2), timestep [B] en escala 0-1000,
        encoder_hidden_states [B, S, 4096] con encoder_attention_mask [B, S]: cada muestra va con su
        longitud de texto real, así el relleno no influye. skip_layers: bloques que se saltan (la
        rama negativa de las previews salta el 22, como el pipeline). references: para edición, una
        lista por muestra de latentes [1, 128, h, w] limpios (cada uno con su tamaño). Devuelve la
        velocidad [B, 128, H, W].
        """
        batch, _, height, width = hidden_states.shape
        mod, head = self._time(timestep.float().reshape(-1).expand(batch))
        ref_mod = self._time(torch.zeros(1, device=hidden_states.device))[0] if references else None
        lengths = encoder_attention_mask.sum(1).tolist()
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, :lengths[index]]
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())
            refs = references[index] if references else ()
            offset = -context.shape[1] if refs and self.reference_layout == "ai-toolkit" else 0
            rope = self.rope(height, width, context.shape[1], hidden_states.device, offset).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            tokens = [self.video_embedder(r.flatten(2).transpose(1, 2).to(video.dtype).contiguous()) for r in refs]
            ropes = [self.rope.reference(*r.shape[-2:], i + 1, hidden_states.device).to(video.dtype) for i, r in enumerate(refs)]
            prefix = sum(r.shape[1] for r in tokens)
            x = torch.cat((*tokens, video, text), 1)
            rope = torch.cat((*ropes, rope), 1)
            modulation = torch.cat((ref_mod, mod[index:index + 1])) if prefix else mod[index:index + 1]
            for layer_index, layer in enumerate(self.layers):
                if layer_index in skip_layers:
                    continue
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x = self._gradient_checkpointing_func(layer, x, rope, modulation, prefix)
                else:
                    x = layer(x, rope, modulation, prefix)
            x = self.final_norm_video(x[:, prefix:prefix + height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)
