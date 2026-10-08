"""Per-loop probes (exploratory extension, 2026-10-08; outside the pre-registration).

For a looped checkpoint, run the looped block more times than it was trained for and, after every loop i, read the
state three ways:
  lens    : the frozen rest of the network (coda, final norm, head) on the state after loop i. At i = r this is the
            model itself; elsewhere it is "evaluate with r_eval = i", the depth curve of token_analysis.py.
  adapter : a trained d x d linear map (identity init) in front of the frozen lens -- is the information there up to a
            change of coordinates that the frozen read-out can use?
  linear  : a trained linear head (init: the model's own head) on the RMS-normalised state -- how much of the next
            token is linearly decodable from the state itself, coda or no coda.
Probes are trained on training-stream tokens the checkpoint never saw (the stream continues after its last step) and
evaluated on a validation set. Also per loop: cosine between consecutive states and to the state at the training r,
relative change of the state, agreement of consecutive lens predictions, and on the validation set the loop at which
each token's lens prediction settles to its value at the training r, split by whether that value is right and by the
token's difficulty (decile of its loss at the training r).

    python fit/loop_probe.py --runs 80M_middle_r8_full_s42,40M_middle_r8_full_s42 --r-max 16 --device cuda
Writes results/loop_probe/<run>__<ckpt>.json and results/loop_probe/summary.csv (one row per run and loop).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fit.token_analysis import checkpoint_name, find_config, n_rung_table, pick_device  # noqa: E402
from looped.data import BatchSampler, TokenStream, ValSet  # noqa: E402

DEFAULT_RUNS = "80M_middle_r8_full_s42,40M_middle_r8_full_s42,20M_middle_r8_full_s42,80M_whole_r8_k4_s42"


def _ctx(device, amp):
    return (torch.autocast(device_type=device.type, dtype=amp) if amp is not None
            else torch.autocast(device_type=device.type, enabled=False))


def collect_states(model, idx: torch.Tensor, r_max: int) -> list[torch.Tensor]:
    """Embedding and prelude once, then r_max loop steps; the state after every loop, in order."""
    x = model.tok_emb(idx)
    for b in model.prelude:
        x = b(x, model.rope_cos, model.rope_sin)
    e = x
    s = torch.zeros_like(e) if model.adapter is not None else e
    states = []
    for _ in range(r_max):
        s = model._core_step(s, e)
        states.append(s)
    return states


def lens(model, s: torch.Tensor) -> torch.Tensor:
    """The frozen rest of the network: coda blocks, final norm, head."""
    x = s
    for b in model.coda:
        x = b(x, model.rope_cos, model.rope_sin)
    return model.head(model.final_norm(x))


def rms_normalise(s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return s * torch.rsqrt(s.float().pow(2).mean(-1, keepdim=True) + eps)


class Probes(nn.Module):
    def __init__(self, model, r_max: int):
        super().__init__()
        d, V = model.cfg.d_model, model.cfg.vocab_size
        self.adapters = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(r_max)])
        self.linears = nn.ModuleList([nn.Linear(d, V, bias=False) for _ in range(r_max)])
        with torch.no_grad():
            for a in self.adapters:
                a.weight.copy_(torch.eye(d))
            for lin in self.linears:
                lin.weight.copy_(model.head.weight)


def train_probes(model, probes: Probes, stream: TokenStream, position: int, seq_len: int, batch: int, n_steps: int,
                 r_max: int, device, amp, lr_adapter: float, lr_linear: float, log_every: int = 50) -> list[float]:
    """One pass over n_steps batches starting at `position`; all probes see the same tokens. Returns the logged losses."""
    V = model.cfg.vocab_size
    sampler = BatchSampler(stream, seq_len, batch, position)
    opt = torch.optim.AdamW([dict(params=probes.adapters.parameters(), lr=lr_adapter),
                             dict(params=probes.linears.parameters(), lr=lr_linear)], weight_decay=0.0)
    warm = max(1, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(st, n_steps) / n_steps)))
    probes.train()
    log, acc, t0 = [], np.zeros((2, r_max)), time.time()
    for step in range(1, n_steps + 1):
        x, y = sampler.next()
        x, y = x.to(device), y.to(device)
        with torch.no_grad(), _ctx(device, amp):
            states = collect_states(model, x, r_max)
        for i, s in enumerate(states):
            s = s.detach()
            with _ctx(device, amp):
                la = F.cross_entropy(lens(model, probes.adapters[i](s)).float().view(-1, V), y.view(-1))
                ll = F.cross_entropy(probes.linears[i](rms_normalise(s)).float().view(-1, V), y.view(-1))
            (la + ll).backward()
            acc[0, i] += float(la.detach()); acc[1, i] += float(ll.detach())
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        if step % log_every == 0 or step == n_steps:
            n = log_every if step % log_every == 0 else step % log_every
            a = acc / n
            print(f"  probes step {step}/{n_steps} adapter {np.array2string(a[0], precision=3, max_line_width=200)} "
                  f"linear {np.array2string(a[1], precision=3, max_line_width=200)} ({time.time() - t0:.0f}s)", flush=True)
            log.append(dict(step=step, adapter=a[0].round(4).tolist(), linear=a[1].round(4).tolist()))
            acc[:] = 0
    return log


@torch.no_grad()
def evaluate_probes(model, probes: Probes, val: ValSet, batch: int, r_max: int, r_train: int, device, amp) -> dict:
    V = model.cfg.vocab_size
    probes.eval()
    sums = {k: np.zeros(r_max) for k in ("lens", "adapter", "linear")}
    cos_final, cos_next, rel_change = np.zeros(r_max), np.zeros(r_max - 1), np.zeros(r_max - 1)
    preds, final_loss, targets, n_tok = [], [], [], 0
    for x, y in val.batches(batch):
        x, y = x.to(device), y.to(device)
        with _ctx(device, amp):
            states = collect_states(model, x, r_max)
        n = y.numel()
        yy = y.view(-1)
        s_final = states[r_train - 1].float()
        pr = torch.empty(r_max, n, dtype=torch.int64, device=device)
        for i, s in enumerate(states):
            with _ctx(device, amp):
                lg = lens(model, s).float().view(-1, V)
                la = lens(model, probes.adapters[i](s)).float().view(-1, V)
                ll = probes.linears[i](rms_normalise(s)).float().view(-1, V)
            l_lens = F.cross_entropy(lg, yy, reduction="none")
            sums["lens"][i] += float(l_lens.sum())
            sums["adapter"][i] += float(F.cross_entropy(la, yy, reduction="sum"))
            sums["linear"][i] += float(F.cross_entropy(ll, yy, reduction="sum"))
            pr[i] = lg.argmax(-1)
            if i == r_train - 1:
                final_loss.append(l_lens.cpu().numpy())
            sf = s.float()
            cos_final[i] += float(F.cosine_similarity(sf, s_final, dim=-1).sum())
            if i + 1 < r_max:
                nxt = states[i + 1].float()
                cos_next[i] += float(F.cosine_similarity(sf, nxt, dim=-1).sum())
                rel_change[i] += float(((nxt - sf).norm(dim=-1) / (sf.norm(dim=-1) + 1e-6)).sum())
        preds.append(pr.cpu().numpy().astype(np.uint16))
        targets.append(yy.cpu().numpy().astype(np.uint16))
        n_tok += n
    P = np.concatenate(preds, axis=1)            # (r_max, n_tok) lens argmax per loop
    tgt, fl = np.concatenate(targets), np.concatenate(final_loss)
    final = P[r_train - 1]
    correct = final == tgt
    eq = P[:r_train] == final[None]
    suffix = np.flip(np.cumprod(np.flip(eq, 0), 0), 0).astype(bool)   # suffix[i] = prediction fixed from loop i+1 on
    settle = r_train - suffix.sum(0) + 1                                # first loop (1-based) after which it never changes
    decile = np.minimum((np.argsort(np.argsort(fl)) * 10) // len(fl), 9) + 1
    hist = lambda m: np.bincount(settle[m], minlength=r_train + 1)[1:].tolist()  # noqa: E731
    out = dict(
        n_tokens=int(n_tok),
        loss={k: (v / n_tok).round(5).tolist() for k, v in sums.items()},
        cos_to_final=(cos_final / n_tok).round(5).tolist(),
        match_final=(P == final[None]).mean(1).round(5).tolist(),
        cos_next=(cos_next / n_tok).round(5).tolist(),
        rel_change=(rel_change / n_tok).round(5).tolist(),
        agree_next=(P[1:] == P[:-1]).mean(1).round(5).tolist(),
        top1_acc_at_r=float(correct.mean()),
        settle_hist=dict(all=hist(np.ones_like(correct)), correct=hist(correct), wrong=hist(~correct)),
        settle_mean=dict(all=float(settle.mean()), correct=float(settle[correct].mean()) if correct.any() else None,
                         wrong=float(settle[~correct].mean()) if (~correct).any() else None),
        settle_mean_by_decile=[float(settle[decile == d].mean()) for d in range(1, 11)],
        late_frac_by_decile=[float((settle[decile == d] >= r_train - 1).mean()) for d in range(1, 11)],
    )
    return out


def analyze(model, r_train: int, a, stream: TokenStream, position: int, val: ValSet, device, amp) -> dict:
    """Train the probes on the stream from `position`, evaluate on `val`; returns the report."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    probes = Probes(model, a.r_max).to(device)
    n_steps = int(a.probe_tokens // (a.batch * model.cfg.seq_len))
    log = train_probes(model, probes, stream, position, model.cfg.seq_len, a.batch, n_steps, a.r_max, device, amp,
                       a.lr_adapter, a.lr_linear)
    rep = evaluate_probes(model, probes, val, a.batch, a.r_max, r_train, device, amp)
    rep.update(r_train=r_train, r_max=a.r_max, probe_tokens=int(n_steps * a.batch * model.cfg.seq_len),
               probe_stream_position=int(position), probe_log=log)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--ckpt", default="branch40", help="branch40 (branch closest to --budget-mult x N) or trunk_latest")
    ap.add_argument("--budget-mult", type=float, default=40.0)
    ap.add_argument("--r-max", type=int, default=16)
    ap.add_argument("--probe-tokens", type=float, default=10e6)
    ap.add_argument("--val-tokens", type=float, default=2e6)
    ap.add_argument("--valset", default="fwe")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-adapter", type=float, default=1e-3)
    ap.add_argument("--lr-linear", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/loop_probe")
    a = ap.parse_args()
    from looped.model import LoopedLM
    from looped.train import load_run_config
    device = pick_device(a.device)
    amp = torch.bfloat16 if device.type == "cuda" else None
    os.makedirs(a.out, exist_ok=True)
    n_rung = n_rung_table()
    for run in a.runs.split(","):
        _, _, mcfg, tcfg, dcfg = load_run_config(find_config(run))
        if mcfg.placement == "dense":
            print(f"{run}: dense, skipped", flush=True)
            continue
        ck_name = checkpoint_name(a, tcfg, n_rung[run])
        path = os.path.join(a.out, f"{run}__{a.ckpt}.json")
        if os.path.exists(path):
            print(f"{run}: {path} exists, skipped", flush=True)
            continue
        model = LoopedLM(mcfg).to(device)
        ck = torch.load(os.path.join("runs", run, ck_name + ".pt"), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        position = int(ck["sampler"]["position"])
        print(f"{run}: checkpoint {ck_name} ({ck['tokens_seen'] / 1e6:.0f}M tokens, step {ck['step']}); probes trained "
              f"from stream position {position / 1e6:.0f}M, r_train {mcfg.r}, r_max {a.r_max}", flush=True)
        stream = TokenStream.from_glob(dcfg.train_shards)
        val = ValSet(dcfg.val_sets[a.valset], mcfg.seq_len, int(a.val_tokens))
        t0 = time.time()
        rep = analyze(model, mcfg.r, a, stream, position, val, device, amp)
        rep.update(run=run, ckpt=ck_name, tokens_seen=int(ck["tokens_seen"]), valset=a.valset,
                   placement=mcfg.placement, k_bwd=mcfg.k_bwd, seconds=round(time.time() - t0))
        json.dump(rep, open(path, "w"), indent=1)
        L = rep["loss"]
        print(f"{run}: lens {np.array2string(np.array(L['lens']), precision=3, max_line_width=200)}\n"
              f"{' ' * len(run)}  adapter {np.array2string(np.array(L['adapter']), precision=3, max_line_width=200)}\n"
              f"{' ' * len(run)}  linear {np.array2string(np.array(L['linear']), precision=3, max_line_width=200)}\n"
              f"{' ' * len(run)}  settle mean all/correct/wrong {rep['settle_mean']}  ({rep['seconds']}s)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_summary(a.out)


def write_summary(out: str) -> None:
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "*.json"))):
        rep = json.load(open(p))
        for i in range(rep["r_max"]):
            rows.append(dict(run=rep["run"], ckpt=rep["ckpt"], r_train=rep["r_train"], loop=i + 1,
                             lens=rep["loss"]["lens"][i], adapter=rep["loss"]["adapter"][i], linear=rep["loss"]["linear"][i],
                             cos_to_final=rep["cos_to_final"][i], match_final=rep["match_final"][i],
                             cos_next=rep["cos_next"][i] if i + 1 < rep["r_max"] else "",
                             rel_change=rep["rel_change"][i] if i + 1 < rep["r_max"] else "",
                             agree_next=rep["agree_next"][i] if i + 1 < rep["r_max"] else ""))
    if rows:
        with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"summary: {os.path.join(out, 'summary.csv')} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
