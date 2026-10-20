"""Run throughput benchmark for one or all variants. Saves results/bench_<variant>.json.

Usage:
  .venv/Scripts/python.exe scripts/run_bench.py --variant v0
  .venv/Scripts/python.exe scripts/run_bench.py --all
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--variant", choices=["v0", "v1", "v2", "v3", "v4"])
    group.add_argument("--all", action="store_true", help="Benchmark all variants")
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--n-new", type=int, default=32)
    args = ap.parse_args()

    import sys
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.run_experiments import VARIANTS  # reuse variant factory
    from quantlens.eval.bench import bench_model

    variants_to_run = list(VARIANTS.keys()) if args.all else [args.variant]
    RESULTS.mkdir(exist_ok=True)

    for v in variants_to_run:
        print(f"\n=== Benchmarking {v} ===", flush=True)
        torch.manual_seed(0)
        model = VARIANTS[v]()
        rows = bench_model(model, context_lengths=(128, 512, 1024),
                           n_runs=args.n_runs, n_new=args.n_new)
        out_data = {
            "variant": v,
            "rows": rows,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
            "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out = RESULTS / f"bench_{v}.json"
        out.write_text(json.dumps(out_data, indent=2))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
