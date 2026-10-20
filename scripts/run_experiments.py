"""Run one experiment variant and save the result to results/. Usage:
python scripts/run_experiments.py --variant v0 [--max-windows 5]"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from quantlens.eval.perplexity import load_wikitext2_tokens, perplexity
from quantlens.quant.int8 import quantize_model
from quantlens.weights import load_pretrained

RESULTS = Path(__file__).resolve().parent.parent / "results"

VARIANTS = {
    "v0": lambda: load_pretrained("gpt2"),
    "v1": lambda: quantize_model(load_pretrained("gpt2"), "per_tensor"),
    "v2": lambda: quantize_model(load_pretrained("gpt2"), "per_channel"),
}


def weight_bytes(model) -> int:
    """Count bytes across parameters AND buffers (captures int8 QuantLinear buffers)."""
    return (
        sum(p.numel() * p.element_size() for p in model.parameters())
        + sum(b.numel() * b.element_size() for b in model.buffers() if b is not None)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max-windows", type=int, default=None, help="quick smoke test only")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = VARIANTS[args.variant]()
    tokens = load_wikitext2_tokens()
    print(f"{tokens.size(1)} tokens in WikiText-2 test", flush=True)

    res = perplexity(model, tokens, args.max_length, args.stride, args.max_windows)
    res.update({
        "variant": args.variant,
        "weight_bytes": weight_bytes(model),
        "total_test_tokens": tokens.size(1),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "partial": args.max_windows is not None,
    })

    RESULTS.mkdir(exist_ok=True)
    suffix = "_partial" if args.max_windows is not None else ""
    out = RESULTS / f"{args.variant}_ppl{suffix}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()