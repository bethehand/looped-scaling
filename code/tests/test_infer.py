"""KV-cached generation (looped/infer.py) reproduces the full forward pass for every placement and loop count."""
import pytest
import torch

from looped.infer import StaticCache, decode_step, prefill
from looped.model import LoopedLM, ModelConfig


def _model(placement, r):
    torch.manual_seed(0)
    extra = dict(n_prelude=1, n_coda=1) if placement == "middle" else {}
    cfg = ModelConfig(vocab_size=50, d_model=32, head_dim=16, seq_len=24, n_layers=4, placement=placement, r=r, **extra)
    return LoopedLM(cfg).eval()


@pytest.mark.parametrize("placement,r,r_eval", [("dense", 1, None), ("middle", 3, None), ("middle", 3, 5),
                                                ("whole", 2, None)])
def test_cached_decoding_matches_full_forward(placement, r, r_eval):
    model = _model(placement, r)
    idx = torch.randint(0, 50, (2, 20))
    with torch.no_grad():
        full, _ = model(idx, r=r_eval)
    cache = StaticCache.for_model(model, batch=2, r=r_eval)
    n_exec = model.cfg.layer_split[0] + (r_eval or r) * model.cfg.layer_split[1] + model.cfg.layer_split[2]
    assert cache.k.shape == (n_exec, 2, model.cfg.n_heads, 24, model.cfg.head_dim)
    out = prefill(model, idx[:, :7], cache, r=r_eval)
    torch.testing.assert_close(out, full[:, :7], atol=1e-5, rtol=1e-5)
    assert int(cache.pos) == 7 and cache.mask.sum() == 7
    for t in range(7, 20):
        step = decode_step(model, idx[:, t:t + 1], cache, r=r_eval)
        torch.testing.assert_close(step[:, 0], full[:, t], atol=1e-5, rtol=1e-5)
    assert int(cache.pos) == 20


def test_cache_size_grows_with_executed_depth():
    sizes = {r: StaticCache.for_model(_model("middle", r), batch=1).nbytes for r in (1, 2, 4)}
    per_layer = sizes[2] - sizes[1]                    # one extra loop = 2 core layers
    assert sizes[4] - sizes[2] == 2 * per_layer and per_layer == 2 * 2 * 1 * 24 * 32 * 4   # k and v, 2 layers, fp32
