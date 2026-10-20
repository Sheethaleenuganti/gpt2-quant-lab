import torch
import torch.nn.functional as F

from quantlens.eval.perplexity import perplexity
from quantlens.model import GPT2, GPT2Config


def tiny_model():
    torch.manual_seed(0)
    return GPT2(GPT2Config(vocab_size=100, n_ctx=64, n_embd=64, n_head=4, n_layer=2)).eval()


def test_single_window_matches_manual_cross_entropy():
    model = tiny_model()
    ids = torch.randint(0, 100, (1, 40))
    with torch.no_grad():
        manual = F.cross_entropy(model(ids)[0, :-1], ids[0, 1:]).item()
    res = perplexity(model, ids, max_length=64, stride=64, progress=False)
    assert abs(res["nll_per_token"] - manual) < 1e-5
    assert res["n_tokens"] == 39


def test_sliding_window_scores_every_token_exactly_once():
    model = tiny_model()
    ids = torch.randint(0, 100, (1, 100))
    res = perplexity(model, ids, max_length=32, stride=16, progress=False)
    assert res["n_tokens"] == 99  # all tokens except the first