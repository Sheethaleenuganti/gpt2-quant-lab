import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from quantlens.model import GPT2, GPT2Config
from quantlens.quant.int8 import (
    QuantLinear,
    dequantize,
    quantize_model,
    quantize_per_channel,
    quantize_per_tensor,
)


def tiny_cfg():
    return GPT2Config(vocab_size=100, n_ctx=64, n_embd=64, n_head=4, n_layer=2)


def tiny_model():
    torch.manual_seed(0)
    return GPT2(tiny_cfg()).eval()


def weight_bytes(model) -> int:
    """Count bytes over parameters AND buffers (mirrors scripts/run_experiments.py)."""
    return (
        sum(p.numel() * p.element_size() for p in model.parameters())
        + sum(b.numel() * b.element_size() for b in model.buffers() if b is not None)
    )


# ── per-tensor quantization ──────────────────────────────────────────────────

def test_per_tensor_dtype_and_range():
    torch.manual_seed(1)
    w = torch.randn(16, 32)
    q, scale = quantize_per_tensor(w)
    assert q.dtype == torch.int8
    assert q.abs().max().item() <= 127


def test_per_tensor_roundtrip_error():
    torch.manual_seed(2)
    w = torch.randn(16, 32)
    q, scale = quantize_per_tensor(w)
    err = (w - dequantize(q, scale)).abs()
    assert (err <= scale.item() / 2 + 1e-6).all(), f"max err {err.max():.3e} > {scale.item()/2:.3e}"


# ── per-channel quantization ─────────────────────────────────────────────────

def test_per_channel_dtype_and_range():
    torch.manual_seed(3)
    w = torch.randn(16, 32)
    q, scale = quantize_per_channel(w)
    assert q.dtype == torch.int8
    assert q.abs().max().item() <= 127


def test_per_channel_roundtrip_error():
    torch.manual_seed(4)
    w = torch.randn(16, 32)
    q, scale = quantize_per_channel(w)
    err = (w - dequantize(q, scale)).abs()
    # each row has its own scale
    assert (err <= scale / 2 + 1e-6).all(), f"max err {err.max():.3e}"


def test_per_channel_beats_per_tensor_on_skewed_matrix():
    """Per-channel MAE must be <= per-tensor MAE on a matrix with wildly different row norms."""
    torch.manual_seed(5)
    base = torch.randn(9, 32)
    # multiply rows by cycling [0.01, 1, 100]
    multipliers = torch.tensor([0.01, 1.0, 100.0]).repeat(3)  # length 9
    w = base * multipliers.unsqueeze(1)

    q_t, s_t = quantize_per_tensor(w)
    q_c, s_c = quantize_per_channel(w)

    mae_tensor = (w - dequantize(q_t, s_t)).abs().mean().item()
    mae_channel = (w - dequantize(q_c, s_c)).abs().mean().item()
    assert mae_channel <= mae_tensor, (
        f"per-channel MAE {mae_channel:.4f} should be <= per-tensor MAE {mae_tensor:.4f}"
    )


# ── QuantLinear ───────────────────────────────────────────────────────────────

def test_quant_linear_forward_per_tensor():
    torch.manual_seed(6)
    linear = nn.Linear(32, 16)
    ql = QuantLinear(linear, "per_tensor")
    x = torch.randn(2, 32)
    with torch.no_grad():
        expected = F.linear(x, dequantize(ql.weight_q, ql.scale), ql.bias)
        actual = ql(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


def test_quant_linear_forward_per_channel():
    torch.manual_seed(7)
    linear = nn.Linear(32, 16)
    ql = QuantLinear(linear, "per_channel")
    x = torch.randn(2, 32)
    with torch.no_grad():
        expected = F.linear(x, dequantize(ql.weight_q, ql.scale), ql.bias)
        actual = ql(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


# ── quantize_model ────────────────────────────────────────────────────────────

def test_quantize_model_output_shape_and_finite():
    model = tiny_model()
    qm = quantize_model(model, "per_tensor")
    ids = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        logits = qm(ids)
    assert logits.shape == (1, 10, 100)
    assert logits.isfinite().all()


def test_quantize_model_leaves_original_unchanged():
    model = tiny_model()
    # snapshot the first block's attn weight before quantization
    original_w = model.h[0].attn.c_attn.weight.data.clone()
    _ = quantize_model(model, "per_channel")
    assert torch.equal(model.h[0].attn.c_attn.weight.data, original_w), (
        "quantize_model must not modify the original model"
    )


def test_quantize_model_is_nn_linear_replaced():
    model = tiny_model()
    qm = quantize_model(model, "per_tensor")
    for block in qm.h:
        for attr in ("c_attn", "c_proj"):
            assert isinstance(getattr(block.attn, attr), QuantLinear)
        for attr in ("c_fc", "c_proj"):
            assert isinstance(getattr(block.mlp, attr), QuantLinear)


def test_quantized_model_smaller_weight_bytes():
    model = tiny_model()
    qm_t = quantize_model(model, "per_tensor")
    qm_c = quantize_model(model, "per_channel")
    fp32_bytes = weight_bytes(model)
    assert weight_bytes(qm_t) < fp32_bytes, "per-tensor model should be smaller than fp32"
    assert weight_bytes(qm_c) < fp32_bytes, "per-channel model should be smaller than fp32"
