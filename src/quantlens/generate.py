"""Greedy and top-k generation, with or without the KV cache."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from quantlens.kvcache import KVCache


def _pick(logits, top_k, temperature, generator):
    if top_k is None:  # greedy
        return logits.argmax(dim=-1, keepdim=True)
    vals, idx = torch.topk(logits / temperature, k=min(top_k, logits.size(-1)), dim=-1)
    probs = F.softmax(vals, dim=-1)
    choice = torch.multinomial(probs, 1, generator=generator)
    return idx.gather(-1, choice)


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, top_k=None, temperature=1.0,
             use_cache=True, generator=None):
    """input_ids: [B, T] long. Returns [B, T + max_new_tokens]."""
    model.eval()
    assert input_ids.size(1) + max_new_tokens <= model.cfg.n_ctx, "exceeds context window"
    ids = input_ids
    cache = KVCache(model.cfg.n_layer) if use_cache else None
    step_input = ids
    for _ in range(max_new_tokens):
        logits = model(step_input, cache=cache) if use_cache else model(ids)
        next_id = _pick(logits[:, -1, :], top_k, temperature, generator)
        ids = torch.cat([ids, next_id], dim=1)
        step_input = next_id
    return ids