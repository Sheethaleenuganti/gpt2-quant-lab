"""GPT-2 forward pass, written from scratch (no Hugging Face model classes)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    eps: float = 1e-5


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPT2Config, layer_idx: int = 0):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.layer_idx = layer_idx
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, cache=None, past_len: int = 0):
        B, T, C = x.shape
        H = self.n_head
        hd = C // H
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, H, hd).transpose(1, 2)  # [B, H, T, hd]
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)  # now [B, H, past_len + T, hd]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)  # [B, H, T, past_len + T]
        # query i sits at absolute position past_len + i and may see keys j <= past_len + i
        causal = torch.tril(
            torch.ones(T, k.size(2), dtype=torch.bool, device=x.device), diagonal=past_len
        )
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config, layer_idx: int = 0):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.eps)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cache=None, past_len: int = 0):
        x = x + self.attn(self.ln_1(x), cache=cache, past_len=past_len)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config | None = None):
        super().__init__()
        self.cfg = cfg or GPT2Config()
        c = self.cfg
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.n_ctx, c.n_embd)
        self.h = nn.ModuleList([Block(c, i) for i in range(c.n_layer)])
        self.ln_f = LayerNorm(c.n_embd, c.eps)

    def forward(self, input_ids: torch.Tensor, cache=None) -> torch.Tensor:
        B, T = input_ids.shape
        past_len = cache.length if cache is not None else 0  # read once, before layers update it
        assert past_len + T <= self.cfg.n_ctx, "sequence longer than context window"
        pos = torch.arange(past_len, past_len + T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.h:
            x = block(x, cache=cache, past_len=past_len)
        x = self.ln_f(x)
        return torch.matmul(x, self.wte.weight.T)  # output head tied to token embedding