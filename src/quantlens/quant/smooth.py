"""SmoothQuant-style per-channel activation rescaling (V4).

Idea: migrate "difficulty" from activations to weights by computing a per-channel
smoothing factor s from the activation statistics, then:
  - divide each input channel of the weight by s  (weight absorbs the scale)
  - multiply each output channel of the preceding layer-norm by s (or insert a
    Rescale module before the linear), so the net computation is unchanged in fp32.

Here we implement the "insert Rescale" variant: a lightweight nn.Module that
element-wise scales its input before passing to the quantized linear.  This keeps
the model graph simple and does not require touching LayerNorm weights.

Reference: SmoothQuant (Xiao et al., 2022) https://arxiv.org/abs/2211.10438
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from quantlens.model import GPT2
from quantlens.quant.act import W8A8Linear
from quantlens.quant.int8 import dequantize, quantize_per_channel, quantize_per_tensor


# ── Smoothing factor computation ──────────────────────────────────────────────

def compute_smooth_scale(
    act_max: torch.Tensor,   # [in_features] – per-channel activation max
    weight: torch.Tensor,    # [out_features, in_features] – fp32 Linear weight
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute s = act_max^alpha / weight_max^(1-alpha), per SmoothQuant Eq. (5).

    Both numerator and denominator are clamped to 1e-8 to avoid division by zero.
    Returns scale of shape [in_features].
    """
    act_part = act_max.pow(alpha).clamp(min=1e-8)
    w_max = weight.abs().amax(dim=0).clamp(min=1e-8)  # per input-channel weight max
    weight_part = w_max.pow(1 - alpha).clamp(min=1e-8)
    return (act_part / weight_part).clamp(min=1e-8)


# ── Rescale module (inserted before each quantized linear) ───────────────────

class Rescale(nn.Module):
    """Element-wise scale applied to the last dimension of the input tensor."""

    def __init__(self, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale  # divide activations → activations become smoother


class SmoothW8A8Linear(nn.Module):
    """W8A8 linear whose weights have been pre-scaled by s (absorbing the activation scale).

    Equivalent to: F.linear(x / s, w * s, bias)  ≈  F.linear(x, w, bias) in fp32.
    In int8 the activation range is reduced (good) and the weight range is adjusted.
    """

    def __init__(self, linear: nn.Linear, scale: torch.Tensor) -> None:
        """
        Args:
            linear: original fp32 nn.Linear (not modified).
            scale:  [in_features] smoothing scale s.
        """
        super().__init__()
        # Absorb scale into weights: w_smooth = w * s (broadcast over out dim)
        w_smooth = linear.weight.data * scale.unsqueeze(0)
        weight_q, weight_scale = quantize_per_channel(w_smooth)
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("smooth_scale", scale.to(torch.float32))

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone().to(torch.float32))
        else:
            self.register_buffer("bias", None)

        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Activations have already been divided by smooth_scale (via Rescale module).
        x_q, x_scale = quantize_per_tensor(x)
        x_dq = dequantize(x_q, x_scale)
        w_dq = dequantize(self.weight_q, self.weight_scale)
        return F.linear(x_dq, w_dq, self.bias)


# ── Block-level smooth quantization ──────────────────────────────────────────

class SmoothedBlock(nn.Module):
    """A GPT-2 transformer Block with SmoothQuant applied to all four linears."""

    def __init__(self, block, act_stats: dict[str, torch.Tensor], alpha: float = 0.5):
        """
        Args:
            block: a Block from quantlens.model (deepcopy expected).
            act_stats: dict mapping layer name → per-channel activation max tensor.
                       Keys: 'c_attn', 'c_proj_attn', 'c_fc', 'c_proj_mlp'.
            alpha: smoothing exponent (0 = weight-only, 1 = act-only).
        """
        super().__init__()
        self.ln_1 = block.ln_1
        self.ln_2 = block.ln_2

        def _make(linear, act_max):
            s = compute_smooth_scale(act_max, linear.weight.data, alpha)
            return nn.Sequential(Rescale(s), SmoothW8A8Linear(linear, s))

        self.attn_proj = _make(block.attn.c_attn, act_stats["c_attn"])
        self.attn_out = _make(block.attn.c_proj, act_stats["c_proj_attn"])
        self.mlp_fc = _make(block.mlp.c_fc, act_stats["c_fc"])
        self.mlp_out = _make(block.mlp.c_proj, act_stats["c_proj_mlp"])

        # Keep non-quantized parts
        self._n_head = block.attn.n_head
        self._layer_idx = block.attn.layer_idx
        import math
        self._sqrt_hd = math.sqrt(block.attn.c_attn.in_features // block.attn.n_head)

    def _attn_forward(self, x, cache, past_len):
        import math
        import torch.nn.functional as F
        B, T, C = x.shape
        H = self._n_head
        hd = C // H
        qkv = self.attn_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, H, hd).transpose(1, 2)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(self._layer_idx, k, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        causal = torch.tril(
            torch.ones(T, k.size(2), dtype=torch.bool, device=x.device), diagonal=past_len
        )
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_out(out)

    def forward(self, x, cache=None, past_len: int = 0):
        x = x + self._attn_forward(self.ln_1(x), cache, past_len)
        h = F.gelu(self.mlp_fc(self.ln_2(x)), approximate="tanh")
        x = x + self.mlp_out(h)
        return x


# ── Full-model smooth quantization ───────────────────────────────────────────

def quantize_model_smooth(
    model: GPT2,
    all_act_stats: list[dict[str, torch.Tensor]],
    alpha: float = 0.5,
) -> GPT2:
    """Return a deep copy of model with SmoothQuant applied to every transformer block.

    Args:
        model: fp32 GPT2 (not modified).
        all_act_stats: list of length n_layer; each entry is the act_stats dict
                       for that block (see SmoothedBlock).
        alpha: smoothing exponent.
    """
    quantized = copy.deepcopy(model)
    new_h = nn.ModuleList()
    for i, block in enumerate(quantized.h):
        new_h.append(SmoothedBlock(block, all_act_stats[i], alpha=alpha))
    quantized.h = new_h
    return quantized
