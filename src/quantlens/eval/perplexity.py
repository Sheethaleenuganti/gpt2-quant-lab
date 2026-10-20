"""Sliding-window perplexity on WikiText-2 (test split)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def load_wikitext2_tokens() -> torch.Tensor:
    """Tokenize the WikiText-2 raw test split into one [1, N] tensor of GPT-2 token ids."""
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast  # tokenizer only, no model class

    try:
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text = "\n\n".join(test["text"])
    return tokenizer(text, return_tensors="pt").input_ids


@torch.no_grad()
def perplexity(model, token_ids: torch.Tensor, max_length: int = 1024, stride: int = 512,
               max_windows: int | None = None, progress: bool = True) -> dict:
    """Perplexity with overlapping windows. Every token after the first is scored exactly once,
    always with as much left context as the window allows."""
    model.eval()
    n = token_ids.size(1)
    total_nll, total_tokens, prev_end, windows = 0.0, 0, 0, 0

    for begin in range(0, n, stride):
        end = min(begin + max_length, n)
        trg_len = end - prev_end  # tokens in this window not scored yet
        window = token_ids[:, begin:end]
        logits = model(window)[:, :-1, :]
        targets = window[:, 1:]
        k = min(trg_len, targets.size(1))  # score only the newest k targets
        nll = F.cross_entropy(logits[:, -k:, :].reshape(-1, logits.size(-1)),
                              targets[:, -k:].reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += k
        prev_end = end
        windows += 1
        if progress and windows % 25 == 0:
            print(f"window {windows}  running ppl {math.exp(total_nll / total_tokens):.3f}", flush=True)
        if end == n or (max_windows is not None and windows >= max_windows):
            break

    nll_per_token = total_nll / total_tokens
    return {
        "perplexity": math.exp(nll_per_token),
        "nll_per_token": nll_per_token,
        "n_tokens": total_tokens,
        "n_windows": windows,
        "max_length": max_length,
        "stride": stride,
    }