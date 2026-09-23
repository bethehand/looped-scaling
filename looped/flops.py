"""FLOP and parameter accounting per token (pre-registration §3 and §4).

Conventions (sources [08] Chinchilla App. F, [09] Porian):
  * matmul FLOPs = 2 * multiply-adds; backward = 2x forward for every layer that receives gradients
  * attention score/value products counted at full context: 4 * T * d per executed layer per token
  * output head counted (2 * V * d forward); input embedding lookup counted as 0 FLOPs
  * looped layers counted once per executed loop; the injection adapter (2d x d) once per loop
  * truncated backprop: backward counted only for prelude, coda, head, and the last k loops

Returned quantities are per token.
"""
from __future__ import annotations

from .model import ModelConfig


def per_block_params(cfg: ModelConfig) -> int:
    d, dff = cfg.d_model, cfg.d_ff
    attn = 4 * d * d
    mlp = 3 * d * dff
    norms = 2 * d
    return attn + mlp + norms


def per_block_matmul_params(cfg: ModelConfig) -> int:
    d, dff = cfg.d_model, cfg.d_ff
    return 4 * d * d + 3 * d * dff


def flops_per_token(cfg: ModelConfig, r: int | None = None, k_bwd: int | None = None) -> dict:
    r = cfg.r if r is None else r
    k = cfg.k_bwd if k_bwd is None else k_bwd
    p, c, q = cfg.layer_split
    d, T, V = cfg.d_model, cfg.seq_len, cfg.vocab_size
    f_block = 2 * per_block_matmul_params(cfg) + 4 * T * d
    f_adapter = 2 * (2 * d * d) if (c > 0 and cfg.injection == "concat") else 0
    f_head = 2 * V * d
    f_loop = c * f_block + f_adapter
    fwd = p * f_block + r * f_loop + q * f_block + f_head
    k_eff = r if k == 0 else min(k, r)
    fwd_grad_part = p * f_block + k_eff * f_loop + q * f_block + f_head
    bwd = 2 * fwd_grad_part
    return dict(
        forward=fwd,            # = deploy FLOPs per token
        backward=bwd,
        train=fwd + bwd,        # training FLOPs per token (algorithmic; recompute not included)
        loops_with_grad=k_eff,
        executed_layers=p + r * c + q,
    )


def train_flops(cfg: ModelConfig, tokens: int) -> float:
    return flops_per_token(cfg)["train"] * tokens


def deploy_flops(cfg: ModelConfig) -> float:
    return flops_per_token(cfg)["forward"]
