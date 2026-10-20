"""Collect per-channel activation outlier statistics and save to results/outliers.json.

Usage:
  .venv/Scripts/python.exe scripts/run_outliers.py [--calib-tokens 512]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-tokens", type=int, default=512,
                    help="Number of WikiText-2 tokens to use as calibration data")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of top outlier channels to report per linear layer")
    args = ap.parse_args()

    from quantlens.eval.outliers import collect_activation_stats, outlier_summary
    from quantlens.eval.perplexity import load_wikitext2_tokens
    from quantlens.weights import load_pretrained

    torch.manual_seed(0)
    model = load_pretrained("gpt2")
    print(f"Loaded GPT-2, collecting activation stats over {args.calib_tokens} tokens...", flush=True)

    tokens = load_wikitext2_tokens()
    calib = tokens[:, :args.calib_tokens]
    stats = collect_activation_stats(model, calib, max_tokens=args.calib_tokens)

    summary = outlier_summary(stats, top_k=args.top_k)
    summary["calib_tokens"] = args.calib_tokens
    summary["date_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "outliers.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
