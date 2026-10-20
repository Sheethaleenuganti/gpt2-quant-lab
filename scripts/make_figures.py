"""Generate publication-ready figures for the research report from results/.

Saves figures to reports/figures/:
  - fig1_perplexity_comparison.png
  - fig2_weight_memory_comparison.png
  - fig3_activation_outliers.png
  - fig4_bench_throughput.png
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def plot_outliers():
    outliers_file = RESULTS / "outliers.json"
    if not outliers_file.exists():
        print(f"Skipping outliers plot: {outliers_file} not found")
        return

    data = json.loads(outliers_file.read_text())
    per_block = data.get("per_layer_outliers", data.get("per_block_top_channels", []))
    
    # Count frequency of top outlier channels across layers for c_fc and c_attn
    attn_counts = {}
    fc_counts = {}
    for block in per_block:
        for ch in block.get("c_attn", []):
            attn_counts[ch] = attn_counts.get(ch, 0) + 1
        for ch in block.get("c_fc", []):
            fc_counts[ch] = fc_counts.get(ch, 0) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    
    # Sort by frequency
    sorted_fc = sorted(fc_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    fc_channels = [str(x[0]) for x in sorted_fc]
    fc_freqs = [x[1] for x in sorted_fc]

    sorted_attn = sorted(attn_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    attn_channels = [str(x[0]) for x in sorted_attn]
    attn_freqs = [x[1] for x in sorted_attn]

    bars1 = ax1.bar(fc_channels, fc_freqs, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax1.set_title("MLP c_fc: Systematic Outlier Channels")
    ax1.set_xlabel("Channel Index (out of 768)")
    ax1.set_ylabel("Occurrence Count Across 12 Transformer Blocks")
    ax1.set_ylim(0, 13)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{yval}/12", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(attn_channels, attn_freqs, color="#c44e52", edgecolor="black", alpha=0.85)
    ax2.set_title("Self-Attention c_attn: Systematic Outlier Channels")
    ax2.set_xlabel("Channel Index (out of 768)")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{yval}/12", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Activation Outlier Channels Emergence in GPT-2 Small", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig3_activation_outliers.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_perplexity():
    variants = ["v0", "v1", "v2", "v3", "v4"]
    labels = ["V0 (FP32)", "V1 (W8-Tensor)", "V2 (W8-Channel)", "V3 (Naive W8A8)", "V4 (SmoothQuant)"]
    ppls = []
    found_labels = []
    
    for v, label in zip(variants, labels):
        f = RESULTS / f"{v}_ppl.json"
        if not f.exists():
            f = RESULTS / f"{v}_ppl_partial.json"
        if f.exists():
            d = json.loads(f.read_text())
            ppls.append(d["perplexity"])
            found_labels.append(label + ("*" if d.get("partial") else ""))

    if not ppls:
        print("Skipping perplexity plot: no perplexity results found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4c72b0", "#55a868", "#2ca02c", "#c44e52", "#8172b3"][:len(ppls)]
    bars = ax.bar(found_labels, ppls, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    ax.set_ylabel("Perplexity on WikiText-2 (Lower is Better)")
    ax.set_title("WikiText-2 Perplexity Across Quantization Variants")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (max(ppls) * 0.02), f"{h:.2f}",
                ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "fig1_perplexity_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_weight_memory():
    variants = ["v0", "v1", "v2", "v3", "v4"]
    labels = ["V0 (FP32)", "V1 (W8-Tensor)", "V2 (W8-Channel)", "V3 (W8A8)", "V4 (Smooth W8A8)"]
    mb = []
    found_labels = []

    for v, label in zip(variants, labels):
        f = RESULTS / f"{v}_ppl.json"
        if not f.exists():
            f = RESULTS / f"{v}_ppl_partial.json"
        if f.exists():
            d = json.loads(f.read_text())
            mb.append(d.get("weight_bytes", 0) / (1024 * 1024))
            found_labels.append(label)

    if not mb:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(found_labels, mb, color="#4c72b0", edgecolor="black", alpha=0.85, width=0.5)
    ax.set_ylabel("Weight Footprint (MB)")
    ax.set_title("GPT-2 Small Model Weight Size")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h:.1f} MB", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "fig2_weight_memory_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_throughput():
    # Load bench results if available
    rows = []
    for f in RESULTS.glob("bench_*.json"):
        data = json.loads(f.read_text())
        v = data["variant"]
        for r in data.get("rows", []):
            rows.append({
                "variant": v,
                "context": r.get("context_len", r.get("context")),
                "cached": r.get("use_cache", r.get("cached")),
                "tok_per_sec": r.get("tokens_per_sec", r.get("tok_per_sec")),
            })

    if not rows:
        print("Skipping throughput plot: no bench results found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    # Group by variant and cache
    contexts = [128, 512, 1024]
    variants = sorted(list(set(r["variant"] for r in rows)))

    for v in variants:
        cached_rates = [next((r["tok_per_sec"] for r in rows if r["variant"] == v and r["context"] == c and r["cached"]), 0) for c in contexts]
        ax.plot(contexts, cached_rates, marker="o", linewidth=2, label=f"{v.upper()} (KV-Cached)")

    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel("Generation Throughput (Tokens / sec)")
    ax.set_title("Inference Throughput vs. Context Length")
    ax.set_xticks(contexts)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    out = FIG_DIR / "fig4_bench_throughput.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    plot_outliers()
    plot_perplexity()
    plot_weight_memory()
    plot_throughput()


if __name__ == "__main__":
    main()
