"""Looped / recurrent-depth transformer.

Three placements (pre-registration §3):
  dense  : 8 unique layers executed once (the r=1 ruler; no injection adapter)
  middle : prelude (n_prelude) -> core (n_core) looped r times with input injection -> coda (n_coda)
  whole  : all n_layers looped r times with input injection (N_once = output head only)

Injection ("concat"): each loop starts from adapter(cat[s, e]) where e is the prelude output
(or the token embedding for "whole") and s is the previous loop's output; s_0 = 0.
Injection ("none"): s_0 = e and each loop is s = core(s) (Ouro-style; used for the r=1 equivalence test).

Truncated backprop (k_bwd > 0): the first r - k_bwd loops run under torch.no_grad(); only the
last k_bwd loops (and prelude/coda/head through them) receive gradients. k_bwd = 0 means full backprop.

Residual scaling (pre-registration §3, source [06]): branch outputs are multiplied by
  eps_once = (L / ref_layers) ** -0.5           for prelude / coda / dense layers
  eps_core = eps_once / r                        for looped core layers
with L = number of unique layers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # explicit: torch 2.6 does not expose torch.utils.checkpoint on import torch


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 448
    n_layers: int = 8
    head_dim: int = 64
    seq_len: int = 1024
    placement: str = "dense"      # dense | middle | whole
    n_prelude: int = 2
    n_coda: int = 2
    r: int = 1                    # loop count used in training and evaluation
    k_bwd: int = 0                # 0 = full backprop; else gradients only through the last k loops
    injection: str = "concat"     # concat | none
    adapter_init: str = "random"  # random | identity
    residual_scale: bool = True
    ref_layers: int = 12
    mlp_mult: float = 8.0 / 3.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    init_std: float = 0.02
    ckpt_loops: bool = False      # gradient checkpointing per loop (memory only; FLOP accounting unaffected)

    def __post_init__(self):
        assert self.placement in ("dense", "middle", "whole"), self.placement
        assert self.d_model % self.head_dim == 0
        if self.placement == "dense":
            assert self.r == 1, "dense placement is the r=1 ruler"
        if self.placement == "middle":
            assert self.n_prelude + self.n_coda < self.n_layers
        assert self.k_bwd >= 0 and self.k_bwd <= self.r

    # ----- derived -----
    @property
    def n_heads(self) -> int:
        return self.d_model // self.head_dim

    @property
    def d_ff(self) -> int:
        h = int(self.mlp_mult * self.d_model)
        return ((h + 63) // 64) * 64

    @property
    def layer_split(self) -> tuple[int, int, int]:
        """(n_prelude, n_core, n_coda) actually used."""
        if self.placement == "dense":
            return self.n_layers, 0, 0
        if self.placement == "middle":
            return self.n_prelude, self.n_layers - self.n_prelude - self.n_coda, self.n_coda
        return 0, self.n_layers, 0

    @property
    def eps_once(self) -> float:
        return (self.n_layers / self.ref_layers) ** -0.5 if self.residual_scale else 1.0

    @property
    def eps_core(self) -> float:
        return self.eps_once / self.r if self.residual_scale else 1.0

    @property
    def executed_layers(self) -> int:
        p, c, q = self.layer_split
        return p + self.r * c + q

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_cache(seq_len: int, head_dim: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                      # (T, hd/2)
    emb = torch.cat([freqs, freqs], dim=-1)          # (T, hd)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, hd)
    T = x.shape[2]
    cos = cos[:T].to(x.dtype)[None, None]
    sin = sin[:T].to(x.dtype)[None, None]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, eps: float):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.eps = eps

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.eps * self.attn(self.n1(x), cos, sin)
        x = x + self.eps * self.mlp(self.n2(x))
        return x


class LoopedLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        p, c, q = cfg.layer_split
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList([Block(cfg, cfg.eps_once) for _ in range(p)])
        self.core = nn.ModuleList([Block(cfg, cfg.eps_core) for _ in range(c)])
        self.coda = nn.ModuleList([Block(cfg, cfg.eps_once) for _ in range(q)])
        self.adapter = (
            nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
            if (c > 0 and cfg.injection == "concat") else None
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        cos, sin = rope_cache(cfg.seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._step_c = None        # compiled loop step (compile mode "region")
        self._ckpt_step_c = None   # compiled checkpointed loop step (compile mode "region")
        self.apply(self._init_weights)
        if self.adapter is not None and cfg.adapter_init == "identity":
            with torch.no_grad():
                self.adapter.weight.zero_()
                self.adapter.weight[:, cfg.d_model:] = torch.eye(cfg.d_model)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    # ----- parameter accounting (pre-registration §3) -----
    def param_counts(self) -> dict:
        def n(mod) -> int:
            return sum(p.numel() for p in mod.parameters())
        emb_in = self.tok_emb.weight.numel()
        head = 0 if self.cfg.tie_embeddings else self.head.weight.numel()
        prelude, core, coda = n(self.prelude), n(self.core), n(self.coda)
        adapter = n(self.adapter) if self.adapter is not None else 0
        norms = self.final_norm.weight.numel()
        n_once = prelude + coda + head + norms
        n_rec = core + adapter
        return dict(
            emb_in=emb_in, head=head, prelude=prelude, core=core, coda=coda, adapter=adapter,
            N_once=n_once, N_rec=n_rec, N=n_once + n_rec,
            total_with_embedding=n_once + n_rec + emb_in,
        )

    # ----- forward -----
    def _core_step(self, s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        if self.adapter is not None:
            h = self.adapter(torch.cat([s, e], dim=-1))
        else:
            h = s
        for b in self.core:
            h = b(h, self.rope_cos, self.rope_sin)
        return h

    def _ckpt_step(self, s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return torch.utils.checkpoint.checkpoint(self._core_step, s, e, use_reentrant=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                r: int | None = None, k_bwd: int | None = None):
        cfg = self.cfg
        r = cfg.r if r is None else r
        k = cfg.k_bwd if k_bwd is None else k_bwd
        x = self.tok_emb(idx)
        for b in self.prelude:
            x = b(x, self.rope_cos, self.rope_sin)
        if len(self.core) > 0:
            e = x
            if self.adapter is not None:
                s = torch.zeros_like(e)
            else:
                s = e
            n_nograd = 0
            if self.training and k > 0:
                n_nograd = max(r - k, 0)
            step = self._step_c if self._step_c is not None else self._core_step
            for i in range(r):
                if i < n_nograd:
                    with torch.no_grad():
                        s = step(s, e)
                    if i == n_nograd - 1:
                        # Autocast caches low-precision copies of the weights. In torch <= 2.6 copies made under
                        # no_grad are reused by the later grad-enabled loops, which silently cuts the gradient to the
                        # looped block (and breaks checkpoint recomputation). Clear the cache before those loops.
                        torch.clear_autocast_cache()
                elif cfg.ckpt_loops and self.training and torch.is_grad_enabled():
                    s = self._ckpt_step_c(s, e) if self._ckpt_step_c is not None else self._ckpt_step(s, e)
                else:
                    s = step(s, e)
            x = s
        for b in self.coda:
            x = b(x, self.rope_cos, self.rope_sin)
        x = self.final_norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ----- torch.compile, per block -----
    def compile_blocks(self, backend: str = "inductor", mode: str = "blocks") -> None:
        """mode "blocks": compile each transformer block in place; the loop over r, truncated backprop, the
        autocast-cache clear and per-loop checkpointing stay in eager Python.
        mode "region": compile prelude/coda blocks individually and one whole loop step (adapter + all core blocks)
        as a single region; checkpointed loops call a compiled function that wraps torch.utils.checkpoint, so the
        recomputation is planned by the compiler instead of eager saved-tensor hooks. The Python loop over r,
        truncation and the autocast-cache clear stay eager in both modes.
        Raises dynamo's recompile limit because the compiled code is called in train/eval, grad/no-grad and several
        batch shapes."""
        import torch._dynamo as dynamo
        import torch._inductor.config as inductor_config
        dc = dynamo.config
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(dc, name):
                setattr(dc, name, max(getattr(dc, name), 64))
        # torch 2.6 bug: inductor's post-grad weight-only-int8 pattern matches "mm(...) * scale" and its extra check
        # reads scale.meta, which crashes when the scale is a Python float -- exactly our residual scaling eps when a
        # graph sees a single eps value (dense ruler, whole-stack looping). The pattern passes only serve cases we do
        # not have (quantized weights, biases, decomposed attention); pointwise fusion is unaffected.
        inductor_config.pattern_matcher = False
        if mode == "blocks":
            for b in list(self.prelude) + list(self.core) + list(self.coda):
                b.compile(backend=backend)
        elif mode == "region":
            for b in list(self.prelude) + list(self.coda):
                b.compile(backend=backend)
            if len(self.core) > 0:
                self._step_c = torch.compile(self._core_step, backend=backend)
                self._ckpt_step_c = torch.compile(self._ckpt_step, backend=backend)
        else:
            raise ValueError(mode)

    # ----- optimizer groups (pre-registration §3: hidden lr = lr0 * (L/12)^-0.5, embeddings/head = lr0) -----
    def param_groups(self, lr0: float, weight_decay: float) -> list[dict]:
        hidden_scale = (self.cfg.n_layers / self.cfg.ref_layers) ** -0.5
        emb_ids = {id(self.tok_emb.weight), id(self.head.weight)}
        hidden_decay, hidden_nodecay, emb = [], [], []
        seen = set()
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if id(p) in emb_ids:
                emb.append(p)
            elif p.ndim >= 2:
                hidden_decay.append(p)
            else:
                hidden_nodecay.append(p)
        return [
            dict(params=hidden_decay, lr=lr0 * hidden_scale, weight_decay=weight_decay, name="hidden"),
            dict(params=hidden_nodecay, lr=lr0 * hidden_scale, weight_decay=0.0, name="norms"),
            dict(params=emb, lr=lr0, weight_decay=weight_decay, name="embeddings"),
        ]
