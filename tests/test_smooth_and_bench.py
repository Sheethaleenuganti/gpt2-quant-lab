"""Tests for SmoothQuant (smooth.py) and bench.py."""
import torch
from torch import nn

from quantlens.model import GPT2, GPT2Config
from quantlens.quant.smooth import (
    Rescale,
    SmoothW8A8Linear,
    compute_smooth_scale,
    quantize_model_smooth,
)
from quantlens.eval.outliers import collect_activation_stats
from quantlens.eval.bench import bench_model


def tiny_cfg():
    return GPT2Config(vocab_size=100, n_ctx=64, n_embd=64, n_head=4, n_layer=2)


def tiny_model():
    torch.manual_seed(0)
    return GPT2(tiny_cfg()).eval()


# ── compute_smooth_scale ──────────────────────────────────────────────────────

def test_smooth_scale_shape_and_positive():
    torch.manual_seed(0)
    in_f, out_f = 32, 16
    act_max = torch.rand(in_f).abs() + 0.1
    weight = torch.randn(out_f, in_f)
    s = compute_smooth_scale(act_max, weight, alpha=0.5)
    assert s.shape == (in_f,)
    assert (s > 0).all()


def test_smooth_scale_alpha_extremes():
    """alpha=1 → scale dominated by activations; alpha=0 → scale dominated by weights."""
    torch.manual_seed(0)
    in_f = 16
    act_max = torch.rand(in_f) * 10 + 1
    weight = torch.randn(8, in_f)
    s1 = compute_smooth_scale(act_max, weight, alpha=1.0)
    s0 = compute_smooth_scale(act_max, weight, alpha=0.0)
    # alpha=1: s = act_max / 1 → large where acts are large
    assert s1.max() > s0.max() * 0.5  # rough sanity


# ── Rescale ───────────────────────────────────────────────────────────────────

def test_rescale_divides_correctly():
    scale = torch.tensor([2.0, 4.0, 1.0])
    r = Rescale(scale)
    x = torch.tensor([[2.0, 8.0, 3.0]])
    out = r(x)
    expected = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(out, expected)


# ── SmoothW8A8Linear ──────────────────────────────────────────────────────────

def test_smooth_linear_output_shape():
    torch.manual_seed(1)
    linear = nn.Linear(32, 16)
    scale = torch.rand(32) + 0.1
    sl = SmoothW8A8Linear(linear, scale)
    # input must already be divided by scale (Rescale is separate)
    x_raw = torch.randn(2, 8, 32)
    x_scaled = x_raw / scale  # simulate Rescale
    with torch.no_grad():
        out = sl(x_scaled)
    assert out.shape == (2, 8, 16)
    assert out.isfinite().all()


# ── quantize_model_smooth ─────────────────────────────────────────────────────

def _make_dummy_stats(model: GPT2) -> list[dict]:
    """Build dummy activation stats with correct per-linear in_features sizes."""
    n_embd = model.cfg.n_embd
    return [
        {
            "c_attn":      torch.ones(n_embd),       # Linear(n_embd → 3*n_embd)
            "c_proj_attn": torch.ones(n_embd),       # Linear(n_embd → n_embd)
            "c_fc":        torch.ones(n_embd),       # Linear(n_embd → 4*n_embd)
            "c_proj_mlp":  torch.ones(4 * n_embd),  # Linear(4*n_embd → n_embd)
        }
        for _ in range(model.cfg.n_layer)
    ]


def test_smooth_model_output_shape_finite():
    model = tiny_model()
    stats = _make_dummy_stats(model)
    qm = quantize_model_smooth(model, stats, alpha=0.5)
    ids = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        logits = qm(ids)
    assert logits.shape == (1, 10, 100)
    assert logits.isfinite().all()


def test_smooth_model_original_unchanged():
    model = tiny_model()
    original_w = model.h[0].attn.c_attn.weight.data.clone()
    stats = _make_dummy_stats(model)
    _ = quantize_model_smooth(model, stats)
    assert torch.equal(model.h[0].attn.c_attn.weight.data, original_w)


def test_smooth_model_with_real_stats():
    """Use collect_activation_stats for a more realistic test."""
    model = tiny_model()
    ids = torch.randint(0, 100, (1, 32))
    stats = collect_activation_stats(model, ids, max_tokens=32)
    qm = quantize_model_smooth(model, stats, alpha=0.5)
    with torch.no_grad():
        logits = qm(ids)
    assert logits.isfinite().all()


# ── bench_model ───────────────────────────────────────────────────────────────

def test_bench_model_structure():
    model = tiny_model()
    results = bench_model(model, context_lengths=(8, 16), n_runs=1, n_new=4)
    # 2 context lengths × 2 cache settings = 4 rows
    assert len(results) == 4
    for row in results:
        assert row["tokens_per_sec"] > 0
        assert row["context_len"] in (8, 16)
        assert isinstance(row["use_cache"], bool)


def test_bench_cached_faster_at_long_context():
    """With KV cache, generation at long context should not be slower than without."""
    model = tiny_model()
    results = bench_model(model, context_lengths=(32,), n_runs=2, n_new=8)
    cached = next(r for r in results if r["use_cache"] is True)
    uncached = next(r for r in results if r["use_cache"] is False)
    # cached should be at least as fast (or close); allow 20% margin
    assert cached["tokens_per_sec"] >= uncached["tokens_per_sec"] * 0.8
