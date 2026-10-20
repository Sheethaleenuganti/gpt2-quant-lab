"""Naive W8A8 quantization: activations quantized dynamically per forward pass.

Strategy (V3 – naive):
  - Weights are stored as int8 (same as QuantLinear in int8.py, per-channel).
  - Activations are quantized per-tensor dynamically on each forward call.
  - Integer matmul is *simulated* in float32 (dequantize both operands, then F.linear).
    This measures accuracy impact without requiring a true int8 BLAS kernel.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from quantlens.model import GPT2
from quantlens.quant.int8 import (
    dequantize,
    quantize_per_channel,
    quantize_per_tensor,
)


class W8A8Linear(nn.Module):
    """Weight-and-activation int8 linear layer (simulated, float32 accumulation).

    Weights: quantized per-channel at construction time (stored as int8 buffer).
    Activations: quantized per-tensor dynamically on each forward pass.
    Output: computed in float32 after dequantizing both operands.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight_q, weight_scale = quantize_per_channel(linear.weight.data)
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("weight_scale", weight_scale)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone().to(torch.float32))
        else:
            self.register_buffer("bias", None)

        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations dynamically (per-tensor, over the whole input tensor)
        x_q, x_scale = quantize_per_tensor(x)
        x_dq = dequantize(x_q, x_scale)
        w_dq = dequantize(self.weight_q, self.weight_scale)
        return F.linear(x_dq, w_dq, self.bias)


def quantize_model_w8a8(model: GPT2) -> GPT2:
    """Return a deep copy of *model* with every transformer Linear replaced by W8A8Linear.

    Layers replaced: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj (inside model.h).
    Embeddings (wte, wpe), LayerNorms, and the tied output head remain float32.
    The original model is not modified.
    """
    quantized = copy.deepcopy(model)
    for block in quantized.h:
        block.attn.c_attn = W8A8Linear(block.attn.c_attn)
        block.attn.c_proj = W8A8Linear(block.attn.c_proj)
        block.mlp.c_fc = W8A8Linear(block.mlp.c_fc)
        block.mlp.c_proj = W8A8Linear(block.mlp.c_proj)
    return quantized
