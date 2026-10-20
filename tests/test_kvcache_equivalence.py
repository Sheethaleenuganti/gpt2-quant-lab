import pytest
import torch

from quantlens.generate import generate
from quantlens.kvcache import KVCache
from quantlens.model import GPT2, GPT2Config


def tiny_model():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=100, n_ctx=64, n_embd=64, n_head=4, n_layer=2)
    return GPT2(cfg).eval()


def test_incremental_logits_match_full():
    model = tiny_model()
    ids = torch.randint(0, 100, (2, 12))
    with torch.no_grad():
        full = model(ids)
        cache = KVCache(model.cfg.n_layer)
        parts = [model(ids[:, :5], cache=cache)]
        for t in range(5, 12):
            parts.append(model(ids[:, t:t + 1], cache=cache))
        cached = torch.cat(parts, dim=1)
    assert cache.length == 12
    torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)


def test_greedy_generation_same_with_and_without_cache():
    model = tiny_model()
    ids = torch.randint(0, 100, (1, 6))
    a = generate(model, ids, 20, use_cache=True)
    b = generate(model, ids, 20, use_cache=False)
    assert torch.equal(a, b)


@pytest.mark.slow
def test_real_weights_greedy_same_with_and_without_cache():
    from quantlens.weights import load_pretrained
    model = load_pretrained("gpt2")
    ids = torch.tensor([[15496, 11, 616, 1438, 318]])  # "Hello, my name is"
    a = generate(model, ids, 20, use_cache=True)
    b = generate(model, ids, 20, use_cache=False)
    assert torch.equal(a, b)