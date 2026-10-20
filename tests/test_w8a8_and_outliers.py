"""Tests for W8A8 (act.py) and outlier analysis (eval/outliers.py)."""
import torch
import torch.nn.functional as F
from torch import nn

from quantlens.model import GPT2, GPT2Config
from quantlens.quant.act import W8A8Linear, quantize_model_w8a8
from quantlens.eval.outliers import collect_activation_stats, top_outlier_channels, outlier_summary


def tiny_cfg():
    return GPT2Config(vocab_size=100, n_ctx=64, n_embd=64, n_head=4, n_layer=2)


def tiny_model():
    torch.manual_seed(0)
    return GPT2(tiny_cfg()).eval()


# ── W8A8Linear ────────────────────────────────────────────────────────────────

def test_w8a8_linear_output_shape():
    torch.manual_seed(1)
    linear = nn.Linear(32, 16)
    ql = W8A8Linear(linear)
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        out = ql(x)
    assert out.shape == (2, 5, 16)
    assert out.isfinite().all()


def test_w8a8_linear_close_to_fp32():
    """W8A8 output should be reasonably close to fp32 for small well-conditioned weights."""
    torch.manual_seed(2)
    linear = nn.Linear(64, 32)
    # Scale weights to a reasonable range
    with torch.no_grad():
        linear.weight.data *= 0.1
    ql = W8A8Linear(linear)
    x = torch.randn(1, 8, 64) * 0.5
    with torch.no_grad():
        fp32_out = linear(x)
        q_out = ql(x)
    # Should be within 5% RMSE relative to fp32 output magnitude
    err = (fp32_out - q_out).pow(2).mean().sqrt().item()
    mag = fp32_out.pow(2).mean().sqrt().item()
    assert err / (mag + 1e-8) < 0.05, f"RMSE ratio {err/(mag+1e-8):.3f} too large"


# ── quantize_model_w8a8 ───────────────────────────────────────────────────────

def test_w8a8_model_output_shape_finite():
    model = tiny_model()
    qm = quantize_model_w8a8(model)
    ids = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        logits = qm(ids)
    assert logits.shape == (1, 10, 100)
    assert logits.isfinite().all()


def test_w8a8_model_original_unchanged():
    model = tiny_model()
    original_w = model.h[0].attn.c_attn.weight.data.clone()
    _ = quantize_model_w8a8(model)
    assert torch.equal(model.h[0].attn.c_attn.weight.data, original_w)


def test_w8a8_linears_replaced():
    model = tiny_model()
    qm = quantize_model_w8a8(model)
    for block in qm.h:
        assert isinstance(block.attn.c_attn, W8A8Linear)
        assert isinstance(block.attn.c_proj, W8A8Linear)
        assert isinstance(block.mlp.c_fc, W8A8Linear)
        assert isinstance(block.mlp.c_proj, W8A8Linear)


# ── outlier analysis ──────────────────────────────────────────────────────────

def test_collect_activation_stats_shape():
    model = tiny_model()
    ids = torch.randint(0, 100, (1, 32))
    stats = collect_activation_stats(model, ids, max_tokens=32)
    assert len(stats) == model.cfg.n_layer
    for layer_stats in stats:
        for key in ("c_attn", "c_proj_attn", "c_fc", "c_proj_mlp"):
            assert key in layer_stats
            assert layer_stats[key].ndim == 1
            assert (layer_stats[key] >= 0).all()


def test_top_outlier_channels_count():
    model = tiny_model()
    ids = torch.randint(0, 100, (1, 32))
    stats = collect_activation_stats(model, ids, max_tokens=32)
    outliers = top_outlier_channels(stats, top_k=5)
    assert len(outliers) == model.cfg.n_layer
    for layer in outliers:
        for key in ("c_attn", "c_proj_attn", "c_fc", "c_proj_mlp"):
            assert len(layer[key]) <= 5


def test_outlier_summary_keys():
    model = tiny_model()
    ids = torch.randint(0, 100, (1, 32))
    stats = collect_activation_stats(model, ids, max_tokens=32)
    summary = outlier_summary(stats, top_k=5)
    for key in ("global_act_max", "mean_layer_act_max", "n_layers", "per_layer_outliers"):
        assert key in summary
    assert summary["global_act_max"] >= 0
    assert summary["n_layers"] == model.cfg.n_layer
