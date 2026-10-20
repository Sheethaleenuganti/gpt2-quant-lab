"""Load pretrained GPT-2 weights into quantlens.model.GPT2."""
from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from quantlens.model import GPT2, GPT2Config

# HF stores these as Conv1D, i.e. [in, out]. nn.Linear wants [out, in].
_TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def load_pretrained(name: str = "gpt2") -> GPT2:
    path = hf_hub_download(repo_id=name, filename="model.safetensors")
    raw = load_file(path)

    state = {}
    for key, tensor in raw.items():
        key = key.removeprefix("transformer.")
        if key.endswith(".attn.bias") or key.endswith("masked_bias") or key.startswith("lm_head"):
            continue  # causal-mask buffers and tied head are not parameters of our model
        if key.endswith(_TRANSPOSED):
            tensor = tensor.t().contiguous()
        state[key] = tensor

    model = GPT2(GPT2Config())
    model.load_state_dict(state, strict=True)
    model.eval()
    return model