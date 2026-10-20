"""Weight-only int8 quantization: per-tensor (V1) and per-channel (V2)."""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from quantlens.model import GPT2


def quantize_per_tensor(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (q, scale) for a weight tensor using a single global scale."""
    scale = (w.abs().max() / 127).clamp(min=1e-8)
    q = w.div(scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def quantize_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (q, scale) with one scale per output row (dim=0) for a Linear weight [out, in]."""
    scale = (w.abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-8)
    q = w.div(scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct a float32 weight from int8 q and its scale."""
    return q.to(torch.float32) * scale


class QuantLinear(nn.Module):
    """Simulated weight-only int8 quantized linear layer.

    The weight is stored as int8 and dequantized to float32 on every forward pass.
    The bias (if present) is stored as a float32 buffer.
    """

    def __init__(self, linear: nn.Linear, mode: str) -> None:
        super().__init__()
        if mode == "per_tensor":
            q, scale = quantize_per_tensor(linear.weight.data)
        elif mode == "per_channel":
            q, scale = quantize_per_channel(linear.weight.data)
        else:
            raise ValueError(f"Unknown quantization mode: {mode!r}")

        self.register_buffer("weight_q", q)
        self.register_buffer("scale", scale)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone().to(torch.float32))
        else:
            self.register_buffer("bias", None)

        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dequantize(self.weight_q, self.scale)
        return F.linear(x, w, self.bias)


def quantize_model(model: GPT2, mode: str) -> GPT2:
    """Return a deep copy of *model* with every transformer Linear replaced by QuantLinear.

    Layers replaced: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj (inside model.h).
    Token/position embeddings (wte, wpe) and all LayerNorms stay float32.
    The tied output head (model.wte.weight.T) is never an nn.Linear; it stays float32.
    The original model is not modified.
    """
    quantized = copy.deepcopy(model)
    for block in quantized.h:
        block.attn.c_attn = QuantLinear(block.attn.c_attn, mode)
        block.attn.c_proj = QuantLinear(block.attn.c_proj, mode)
        block.mlp.c_fc = QuantLinear(block.mlp.c_fc, mode)
        block.mlp.c_proj = QuantLinear(block.mlp.c_proj, mode)
    return quantized
