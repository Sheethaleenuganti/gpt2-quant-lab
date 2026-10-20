"""Token throughput benchmarking for all experiment variants.

Measures median tokens/sec at context lengths 128, 512, 1024,
with and without KV cache (batch size 1, 3 timed runs each).
"""
from __future__ import annotations

import time
from statistics import median
from typing import Any

import torch

from quantlens.generate import generate
from quantlens.model import GPT2


def _warmup(model: GPT2, prompt_len: int = 8) -> None:
    """Run one generation step to warm up the model / JIT."""
    ids = torch.zeros(1, prompt_len, dtype=torch.long)
    with torch.no_grad():
        generate(model, ids, max_new_tokens=4, use_cache=True)


def bench_model(
    model: GPT2,
    context_lengths: tuple[int, ...] = (128, 512, 1024),
    n_runs: int = 3,
    n_new: int = 32,
) -> list[dict[str, Any]]:
    """Benchmark model at several context lengths, with and without KV cache.

    Returns a list of result dicts, one per (context_len, use_cache) combination.
    Each dict has keys: context_len, use_cache, tokens_per_sec (median over n_runs).
    """
    model.eval()
    _warmup(model)
    results = []

    for ctx in context_lengths:
        prompt_len = min(ctx, model.cfg.n_ctx - n_new)
        prompt = torch.zeros(1, prompt_len, dtype=torch.long)
        for use_cache in (True, False):
            times = []
            for _ in range(n_runs):
                torch.manual_seed(0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    generate(model, prompt, max_new_tokens=n_new, use_cache=use_cache)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            elapsed = median(times)
            tps = n_new / elapsed
            results.append({
                "context_len": ctx,
                "use_cache": use_cache,
                "tokens_per_sec": round(tps, 2),
                "median_sec": round(elapsed, 4),
                "n_runs": n_runs,
                "n_new_tokens": n_new,
            })
            print(
                f"  ctx={ctx:4d}  cache={str(use_cache):5s}  "
                f"{tps:6.1f} tok/s  ({elapsed:.3f}s)",
                flush=True,
            )

    return results
