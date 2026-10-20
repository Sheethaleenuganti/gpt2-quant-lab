"""Key/value cache for incremental decoding."""
from __future__ import annotations

import torch


class KVCache:
    def __init__(self, n_layer: int):
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer

    @property
    def length(self) -> int:
        """Number of tokens already cached."""
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def update(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append new keys/values [B, H, T_new, hd] for one layer; return the full cache."""
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k_new, v_new
        else:
            self.k[layer] = torch.cat([self.k[layer], k_new], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v_new], dim=2)
        return self.k[layer], self.v[layer]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)