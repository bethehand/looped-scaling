"""Pre-run checks 1, 3 and parameter accounting (pre-registration §5). Runs on CPU."""
import copy

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from looped.flops import flops_per_token
from looped.model import LoopedLM, ModelConfig

torch.manual_seed(0)
SMALL = dict(vocab_size=64, d_model=32, head_dim=16, seq_len=16, n_layers=4, n_prelude=1, n_coda=1)


def _batch(cfg, B=2):
    x = torch.randint(0, cfg.vocab_size, (B, cfg.seq_len))
    y = torch.randint(0, cfg.vocab_size, (B, cfg.seq_len))
    return x, y


# ---------- check 1a: r=1 looped model (no adapter) == dense model, bitwise ----------
def test_r1_equivalence_middle_and_whole():
    dense = LoopedLM(ModelConfig(placement="dense", r=1, **SMALL))
    for placement in ("middle", "whole"):
        looped = LoopedLM(ModelConfig(placement=placement, r=1, injection="none", **SMALL))
        # copy dense blocks in execution order into prelude / core / coda
        blocks = list(dense.prelude)
        p, c, q = looped.cfg.layer_split
        order = list(looped.prelude) + list(looped.core) + list(looped.coda)
        assert len(order) == len(blocks)
        for src, dst in zip(blocks, order):
            dst.load_state_dict(src.state_dict())
            assert abs(dst.eps - src.eps) < 1e-12
        looped.tok_emb.load_state_dict(dense.tok_emb.state_dict())
        looped.head.load_state_dict(dense.head.state_dict())
        looped.final_norm.load_state_dict(dense.final_norm.state_dict())
        x, y = _batch(dense.cfg)
        dense.eval(); looped.eval()
        with torch.no_grad():
            ld, _ = dense(x, y)
            ll, _ = looped(x, y)
        assert torch.equal(ld, ll), placement


# ---------- check 1b: truncated backprop gradients == detach-based reference ----------
def _reference_grads(model: LoopedLM, x, y, r, k):
    """Run all loops with autograd but detach the state entering loop r-k (the textbook definition)."""
    model.zero_grad(set_to_none=True)
    h = model.tok_emb(x)
    for b in model.prelude:
        h = b(h, model.rope_cos, model.rope_sin)
    e = h
    s = torch.zeros_like(e) if model.adapter is not None else e
    for i in range(r):
        if k > 0 and i == r - k:
            s = s.detach()
        s = model._core_step(s, e)
    h = s
    for b in model.coda:
        h = b(h, model.rope_cos, model.rope_sin)
    logits = model.head(model.final_norm(h))
    loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("placement", ["middle", "whole"])
@pytest.mark.parametrize("r,k", [(4, 2), (8, 4), (4, 4), (4, 0)])
def test_truncated_backprop_matches_reference(placement, r, k):
    cfg = ModelConfig(placement=placement, r=r, k_bwd=k, **SMALL)
    model = LoopedLM(cfg).double()
    x, y = _batch(cfg)
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    ours = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    ref = _reference_grads(model, x, y, r, k)
    assert ours.keys() == ref.keys()
    for n in ours:
        assert torch.allclose(ours[n], ref[n], atol=1e-10, rtol=1e-8), n
    # a truncated run must differ from the full-backprop run on the core (k in (0, r))
    if 0 < k < r:
        full = _reference_grads(model, x, y, r, 0)
        core_names = [n for n in ours if n.startswith("core.")]
        assert any(not torch.allclose(ours[n], full[n], atol=1e-8) for n in core_names)
    if k == r:  # k == r is identical to full backprop
        full = _reference_grads(model, x, y, r, 0)
        for n in ours:
            assert torch.allclose(ours[n], full[n], atol=1e-10, rtol=1e-8), n


def test_truncation_does_not_touch_eval():
    cfg = ModelConfig(placement="middle", r=4, k_bwd=2, **SMALL)
    model = LoopedLM(cfg)
    x, y = _batch(cfg)
    model.eval()
    with torch.no_grad():
        a, _ = model(x, y)
        b, _ = model(x, y, k_bwd=0)
    assert torch.equal(a, b)


# ---------- check 3: FLOP counter vs measured matmul FLOPs ----------
# FlopCounterMode does not count the fused scaled_dot_product_attention kernel, so the linear-layer part is
# compared exactly and the attention term (4*T*d per executed layer per token) is verified with explicit matmuls.
@pytest.mark.parametrize("placement,r", [("dense", 1), ("middle", 4), ("whole", 2)])
def test_forward_flops_match_measured(placement, r):
    cfg = ModelConfig(placement=placement, r=r, vocab_size=128, d_model=64, head_dim=32, seq_len=32,
                      n_layers=4, n_prelude=1, n_coda=1)
    model = LoopedLM(cfg).eval()
    B = 3
    x, y = _batch(cfg, B)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x, y)
    measured = fc.get_total_flops()
    ours = flops_per_token(cfg)["forward"] * B * cfg.seq_len
    attention = 4 * cfg.seq_len * cfg.d_model * cfg.executed_layers * B * cfg.seq_len
    assert measured == ours - attention, (measured, ours, attention)


def test_attention_flop_term():
    B, H, T, hd = 2, 4, 32, 16
    q, k, v = (torch.randn(B, H, T, hd) for _ in range(3))
    with FlopCounterMode(display=False) as fc:
        att = q @ k.transpose(-1, -2)
        _ = att @ v
    assert fc.get_total_flops() == 4 * T * (H * hd) * B * T


def test_train_flops_truncation_accounting():
    cfg_full = ModelConfig(placement="middle", r=8, k_bwd=0, **SMALL)
    cfg_tr = ModelConfig(placement="middle", r=8, k_bwd=4, **SMALL)
    f, t = flops_per_token(cfg_full), flops_per_token(cfg_tr)
    assert f["forward"] == t["forward"]
    assert t["backward"] < f["backward"]
    assert f["train"] == 3 * f["forward"]


# ---------- parameter accounting ----------
@pytest.mark.parametrize("placement,r", [("dense", 1), ("middle", 4), ("whole", 4)])
def test_param_counts_sum(placement, r):
    cfg = ModelConfig(placement=placement, r=r, **SMALL)
    model = LoopedLM(cfg)
    pc = model.param_counts()
    total = sum(p.numel() for p in model.parameters())
    assert pc["total_with_embedding"] == total
    assert pc["N"] == pc["N_once"] + pc["N_rec"]
    if placement == "whole":
        assert pc["prelude"] == 0 and pc["coda"] == 0
    if placement == "dense":
        assert pc["N_rec"] == 0 and pc["adapter"] == 0


def test_param_groups_cover_all_params():
    cfg = ModelConfig(placement="middle", r=2, **SMALL)
    model = LoopedLM(cfg)
    groups = model.param_groups(1e-3, 0.1)
    n = sum(len(g["params"]) for g in groups)
    assert n == len(list(model.parameters()))
    scale = (cfg.n_layers / cfg.ref_layers) ** -0.5
    by_name = {g["name"]: g for g in groups}
    assert abs(by_name["hidden"]["lr"] - 1e-3 * scale) < 1e-12
    assert abs(by_name["embeddings"]["lr"] - 1e-3) < 1e-12
    assert by_name["norms"]["weight_decay"] == 0.0


# ---------- regression: truncated backprop under bf16 autocast must still train the looped block ----------
# torch <= 2.6 reuses autocast weight casts made under no_grad; without the cache clear in LoopedLM.forward the
# core and adapter get no gradient (silently) and checkpoint recomputation fails.
def _grads(model, x, y, device, autocast):
    model.zero_grad(set_to_none=True)
    ctx = (torch.autocast(device, dtype=torch.bfloat16) if autocast else torch.autocast(device, enabled=False))
    with ctx:
        _, loss = model(x, y)
    loss.backward()
    return {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("placement", ["middle", "whole"])
@pytest.mark.parametrize("k,ckpt", [(2, False), (2, True), (0, True)])
def test_autocast_truncation_keeps_core_gradients(device, placement, k, ckpt):
    torch.manual_seed(0)
    cfg = ModelConfig(placement=placement, r=4, k_bwd=k, ckpt_loops=ckpt, vocab_size=64, d_model=64, head_dim=32,
                      seq_len=32, n_layers=4, n_prelude=1, n_coda=1)
    model = LoopedLM(cfg).to(device).train()
    x = torch.randint(0, 64, (2, 32), device=device)
    y = torch.randint(0, 64, (2, 32), device=device)
    ref = _grads(model, x, y, device, autocast=False)
    got = _grads(model, x, y, device, autocast=True)
    for name in ref:
        assert name in got, f"{name} received no gradient under autocast"
        cos = torch.nn.functional.cosine_similarity(got[name].flatten(), ref[name].flatten(), dim=0)
        assert cos > 0.98, (name, float(cos))
