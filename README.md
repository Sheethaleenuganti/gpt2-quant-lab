# quantlens – GPT-2 int8 Quantization Lab

> **Research question:** Which layers and activation channels cause int8 quantization error in GPT-2 small, and does a SmoothQuant-style rescaling recover the lost perplexity?

![CI](https://github.com/Sheethaleenuganti/gpt2-quant-lab/actions/workflows/ci.yml/badge.svg)

---

## Overview

`quantlens` is a from-scratch GPT-2 (124 M) inference engine in pure PyTorch (CPU), used as a research workbench to measure where int8 quantization hurts and whether channel-wise activation smoothing repairs it.

All numbers quoted here come from files in `results/`.

---

## Project Structure

```
src/quantlens/
  model.py          – GPT2Config + GPT2 (forward, tied output head)
  weights.py        – load_pretrained("gpt2") → GPT2 with HF weights
  kvcache.py        – KVCache for incremental decoding
  generate.py       – greedy / top-k generation
  quant/
    int8.py         – weight-only int8 (V1 per-tensor, V2 per-channel)
    act.py          – naive W8A8 (V3)
    smooth.py       – SmoothQuant-style rescaling (V4)
  eval/
    perplexity.py   – sliding-window perplexity on WikiText-2
    outliers.py     – per-channel activation statistics
    bench.py        – tokens/sec benchmark

scripts/
  run_experiments.py  – perplexity for V0–V4
  run_bench.py        – throughput benchmark for V0–V4
  run_outliers.py     – activation outlier collection

tests/              – pytest suite (~25 tests, all fast except slow-marked)
results/            – JSON experiment outputs (no weights committed)
reports/            – final report PDF/figures
docs/hypotheses.md  – pre-registered hypotheses
```

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

**Always use `.venv\Scripts\python.exe`** (or activate first) — `datasets` and `torch` are installed in the venv, not the system Python.

---

## Running Experiments

```powershell
# Perplexity (WikiText-2 test, sliding window 1024, stride 512)
.venv\Scripts\python.exe scripts/run_experiments.py --variant v0   # fp32 baseline
.venv\Scripts\python.exe scripts/run_experiments.py --variant v1   # weight-only per-tensor int8
.venv\Scripts\python.exe scripts/run_experiments.py --variant v2   # weight-only per-channel int8
.venv\Scripts\python.exe scripts/run_experiments.py --variant v3   # naive W8A8
.venv\Scripts\python.exe scripts/run_experiments.py --variant v4   # SmoothQuant W8A8

# Quick smoke test (5 windows only)
.venv\Scripts\python.exe scripts/run_experiments.py --variant v0 --max-windows 5

# Activation outlier analysis
.venv\Scripts\python.exe scripts/run_outliers.py

# Throughput benchmark
.venv\Scripts\python.exe scripts/run_bench.py --variant v0
.venv\Scripts\python.exe scripts/run_bench.py --all
```

---

## Experiment Variants

| Variant | Description |
|---------|-------------|
| V0 | fp32 baseline |
| V1 | Weight-only int8, per-tensor scale |
| V2 | Weight-only int8, per-channel scale |
| V3 | Naive W8A8 (activations + weights quantized) |
| V4 | W8A8 + SmoothQuant-style channel rescaling |

---

## Hypotheses (pre-registered)

See [`docs/hypotheses.md`](docs/hypotheses.md).

- **H1:** Naive W8A8 (V3) degrades perplexity more than weight-only int8 (V1, V2).
- **H2:** A small number of activation channels dominate the activation range in Linear layer inputs.

---

## Tests

```powershell
# Fast suite (runs in ~20s)
.venv\Scripts\pytest -m "not slow" -q

# Full suite including HF oracle comparison (slow, ~5min)
.venv\Scripts\pytest -q
```

---

## Rules (from SPEC.md)

- Engine code in `src/quantlens/` **never imports Hugging Face model classes**.
- `transformers` is allowed only for: `weights.py` weight loading, the tokenizer, and test oracles in `tests/`.
- All quoted numbers come from `results/`.
- No weights, datasets, or `.venv` are committed.