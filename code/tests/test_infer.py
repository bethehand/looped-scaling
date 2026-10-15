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


def test_shared_cache_has_one_slot_per_unique_layer_and_equals_full_when_nothing_is_shared():
    """S6 (2026-10-15): with shared=True a core layer keeps one cache for all loops. For a dense model and for a single
    loop nothing is shared, so the decoded logits equal the full forward pass."""
    model = _model("middle", 3)
    assert StaticCache.for_model(model, batch=1, shared=True).k.shape[0] == 4          # 1 + 2 + 1 unique layers
    assert StaticCache.for_model(model, batch=1).k.shape[0] == 1 + 3 * 2 + 1
    idx = torch.randint(0, 50, (2, 12))
    for placement, r, r_eval in (("dense", 1, None), ("middle", 3, 1)):
        m = _model(placement, r)
        with torch.no_grad():
            full, _ = m(idx, r=r_eval)
        cache = StaticCache.for_model(m, batch=2, r=r_eval, shared=True)
        out = [prefill(m, idx[:, :1], cache, r=r_eval)[:, 0]]
        out += [decode_step(m, idx[:, t:t + 1], cache, r=r_eval)[:, 0] for t in range(1, 12)]
        torch.testing.assert_close(torch.stack(out, 1), full, atol=1e-5, rtol=1e-5)


def test_shared_cache_changes_later_tokens_only():
    """With several loops, sharing changes what later tokens attend to; the first token sees no earlier token."""
    model = _model("middle", 3)
    idx = torch.randint(0, 50, (2, 12))
    with torch.no_grad():
        full, _ = model(idx)
    cache = StaticCache.for_model(model, batch=2, shared=True)
    first = prefill(model, idx[:, :1], cache)[:, 0]
    torch.testing.assert_close(first, full[:, 0], atol=1e-5, rtol=1e-5)
    later = torch.stack([decode_step(model, idx[:, t:t + 1], cache)[:, 0] for t in range(1, 12)], 1)
    assert not torch.allclose(later, full[:, 1:], atol=1e-4)
    assert torch.isfinite(later).all()


def test_kv_share_eval_decode_loss_matches_the_forward_pass():
    """scripts/kv_share_eval.py (S6): the full-cache decoding loss equals the forward-pass loss; sharing changes it."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import kv_share_eval as kse
    model = _model("middle", 3)
    idx = torch.randint(0, 50, (2, 13))
    x, y = idx[:, :-1], idx[:, 1:]
    fw, full, shared = kse.forward_loss(model, x, y, None), kse.decode_loss(model, x, y, None, False), kse.decode_loss(model, x, y, None, True)
    assert abs(fw - full) < 1e-3 and abs(shared - full) > 1e-6
    assert abs(kse.decode_loss(model, x, y, 1, True) - kse.forward_loss(model, x, y, 1)) < 1e-3     # one loop: nothing shared
