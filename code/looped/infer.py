"""KV-cached generation for LoopedLM (dense / middle / whole), for the inference-cost measurements (supplementary S3).

The training model (looped/model.py) is used unchanged: this module reuses its modules and weights.
  cache = StaticCache.for_model(model, batch, max_len)
  logits = prefill(model, idx, cache)          # (B, T) prompt -> logits of every prompt position
  logits = decode_step(model, tok, cache)      # (B, 1) next token -> logits of that position
Every executed layer keeps its own keys and values: a block looped r times has r caches per core layer, so the cache
grows with the executed depth, not with the number of unique parameters. With shared=True (S6, 2026-10-15) a core layer
keeps one cache for all loops: every loop reads the earlier tokens' keys and values as written at their last loop and
writes the current token's at the current loop, so after the last loop the cache holds the last loop's values
("terminal KV sharing", Huang et al. 2026; Ouro's last-step reuse). The prompt (prefill) is unchanged by sharing: its
positions attend to each other within the same loop. The cache is allocated once at max_len and a
decode step has static shapes (the position is a tensor and attention is masked to positions <= pos; the step advances
the position itself), so a step can be captured in a CUDA graph and replayed once per token.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import LoopedLM, apply_rope


class StaticCache:
    def __init__(self, n_exec: int, batch: int, n_heads: int, head_dim: int, max_len: int, dtype, device,
                 shared: bool = False):
        self.k = torch.zeros(n_exec, batch, n_heads, max_len, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.shared = shared
        self.pos = torch.zeros(1, dtype=torch.long, device=device)              # next position to write
        self.mask = torch.zeros(1, 1, 1, max_len, dtype=torch.bool, device=device)   # True = may be attended to
        self.max_len = max_len

    @classmethod
    def for_model(cls, model: LoopedLM, batch: int, max_len: int | None = None, r: int | None = None,
                  shared: bool = False) -> "StaticCache":
        cfg = model.cfg
        p, c, q = cfg.layer_split
        n_exec = p + c + q if shared else p + (cfg.r if r is None else r) * c + q
        w = model.head.weight
        return cls(n_exec, batch, cfg.n_heads, cfg.head_dim, max_len or cfg.seq_len, w.dtype, w.device, shared=shared)

    def reset(self, pos: int = 0) -> None:
        self.pos.fill_(pos)
        self.mask.zero_()
        self.mask[..., :pos] = True

    @property
    def nbytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()


def _block(b, x: torch.Tensor, cache: StaticCache, i: int, pos: torch.Tensor, rope_cos, rope_sin,
           prefill: bool) -> torch.Tensor:
    """Block.forward with this execution's keys and values written to cache slot i at positions pos."""
    a = b.attn
    B, T, D = x.shape
    q, k, v = a.qkv(b.n1(x)).split(D, dim=-1)
    q = q.view(B, T, a.n_heads, a.head_dim).transpose(1, 2)
    k = k.view(B, T, a.n_heads, a.head_dim).transpose(1, 2)
    v = v.view(B, T, a.n_heads, a.head_dim).transpose(1, 2)
    cos, sin = rope_cos.index_select(0, pos), rope_sin.index_select(0, pos)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    cache.k[i].index_copy_(2, pos, k)
    cache.v[i].index_copy_(2, pos, v)
    if prefill:            # the prompt starts at position 0, so it only attends to itself
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        y = F.scaled_dot_product_attention(q, cache.k[i], cache.v[i], attn_mask=cache.mask)
    y = y.transpose(1, 2).contiguous().view(B, T, D)
    x = x + b.eps * a.proj(y)
    return x + b.eps * b.mlp(b.n2(x))


def _forward(model: LoopedLM, idx: torch.Tensor, cache: StaticCache, pos: torch.Tensor, r: int, prefill: bool):
    rc, rs = model.rope_cos, model.rope_sin
    x = model.tok_emb(idx)
    p, c = len(model.prelude), len(model.core)
    for j, b in enumerate(model.prelude):
        x = _block(b, x, cache, j, pos, rc, rs, prefill)
    if c > 0:
        e = x
        s = torch.zeros_like(e) if model.adapter is not None else e
        for loop in range(r):
            h = model.adapter(torch.cat([s, e], dim=-1)) if model.adapter is not None else s
            for j, b in enumerate(model.core):
                slot = p + j if cache.shared else p + loop * c + j      # shared: the same slot in every loop
                h = _block(b, h, cache, slot, pos, rc, rs, prefill)
            s = h
        x = s
    base = p + (c if cache.shared else r * c)
    for j, b in enumerate(model.coda):
        x = _block(b, x, cache, base + j, pos, rc, rs, prefill)
    assert base + len(model.coda) == cache.k.shape[0], "cache was built for a different loop count"
    return model.head(model.final_norm(x))


@torch.no_grad()
def prefill(model: LoopedLM, idx: torch.Tensor, cache: StaticCache, r: int | None = None) -> torch.Tensor:
    T = idx.shape[1]
    pos = torch.arange(T, device=idx.device)
    logits = _forward(model, idx, cache, pos, model.cfg.r if r is None else r, prefill=True)
    cache.reset(T)
    return logits


@torch.no_grad()
def decode_step(model: LoopedLM, tok: torch.Tensor, cache: StaticCache, r: int | None = None) -> torch.Tensor:
    """tok: (B, 1), the token at position cache.pos. Static shapes; advances cache.pos by one."""
    cache.mask.index_fill_(-1, cache.pos, True)
    logits = _forward(model, tok, cache, cache.pos, model.cfg.r if r is None else r, prefill=False)
    cache.pos.add_(1)
    return logits
