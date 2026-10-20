"""Activation outlier analysis for GPT-2 linear layers.

Collects per-channel activation statistics by running calibration data through
the model with forward hooks, then identifies which input channels have
disproportionately large ranges.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from quantlens.model import GPT2

# Names of the four linear sub-layers we instrument inside each transformer block
_LINEAR_KEYS = ("c_attn", "c_proj_attn", "c_fc", "c_proj_mlp")


def collect_activation_stats(
    model: GPT2,
    token_ids: torch.Tensor,
    max_tokens: int = 512,
) -> list[dict[str, torch.Tensor]]:
    """Run token_ids through model and record per-channel max |activation| for each linear.

    Args:
        model: fp32 GPT2 in eval mode (not modified).
        token_ids: [1, N] long tensor of calibration tokens.
        max_tokens: maximum context window to use per forward pass.

    Returns:
        all_stats: list of length n_layer; each element is a dict with keys
                   'c_attn', 'c_proj_attn', 'c_fc', 'c_proj_mlp' mapping to
                   a [in_features] float32 tensor of per-channel activation max.
    """
    n_layer = model.cfg.n_layer
    # Accumulators: layer_idx → key → running channel-wise max
    acc: list[dict[str, torch.Tensor | None]] = [
        {k: None for k in _LINEAR_KEYS} for _ in range(n_layer)
    ]

    hooks: list[Any] = []

    def _make_hook(layer_idx: int, key: str):
        def hook(module, inputs, output):
            x = inputs[0].detach()  # [B, T, in_features]
            channel_max = x.abs().amax(dim=(0, 1))  # [in_features]
            prev = acc[layer_idx][key]
            acc[layer_idx][key] = channel_max if prev is None else torch.maximum(prev, channel_max)
        return hook

    # Register hooks on the four linears in each block
    for i, block in enumerate(model.h):
        hooks.append(block.attn.c_attn.register_forward_hook(_make_hook(i, "c_attn")))
        hooks.append(block.attn.c_proj.register_forward_hook(_make_hook(i, "c_proj_attn")))
        hooks.append(block.mlp.c_fc.register_forward_hook(_make_hook(i, "c_fc")))
        hooks.append(block.mlp.c_proj.register_forward_hook(_make_hook(i, "c_proj_mlp")))

    model.eval()
    n = token_ids.size(1)
    with torch.no_grad():
        for begin in range(0, n, max_tokens):
            window = token_ids[:, begin: begin + max_tokens]
            model(window)

    for h in hooks:
        h.remove()

    # Replace any None (unhit layer) with zeros for safety
    result = []
    for layer_acc in acc:
        result.append({
            k: (v if v is not None else torch.zeros(1))
            for k, v in layer_acc.items()
        })
    return result


def top_outlier_channels(
    act_stats: list[dict[str, torch.Tensor]],
    top_k: int = 10,
) -> list[dict[str, list[int]]]:
    """For each layer and linear, return the indices of the top-k outlier input channels.

    Args:
        act_stats: output of collect_activation_stats.
        top_k: number of top channels to report per linear.

    Returns:
        List of dicts (one per layer) mapping linear key → sorted list of channel indices.
    """
    result = []
    for layer_acc in act_stats:
        layer_result: dict[str, list[int]] = {}
        for key, channel_max in layer_acc.items():
            k = min(top_k, channel_max.numel())
            indices = channel_max.topk(k).indices.tolist()
            layer_result[key] = sorted(indices)
        result.append(layer_result)
    return result


def outlier_summary(
    act_stats: list[dict[str, torch.Tensor]],
    top_k: int = 10,
) -> dict[str, Any]:
    """Compute a summary dict with global stats and per-layer outlier info.

    Returns a dict suitable for json.dumps and saving to results/.
    """
    import json

    all_maxes = []
    for layer_acc in act_stats:
        for channel_max in layer_acc.values():
            all_maxes.append(channel_max.max().item())

    outliers = top_outlier_channels(act_stats, top_k=top_k)

    return {
        "global_act_max": max(all_maxes),
        "mean_layer_act_max": sum(all_maxes) / len(all_maxes),
        "n_layers": len(act_stats),
        "top_k": top_k,
        "per_layer_outliers": [
            {k: v for k, v in layer.items()}
            for layer in outliers
        ],
    }
