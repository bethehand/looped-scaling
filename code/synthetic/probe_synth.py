"""Intermediate-value probes for the chain task (extension C, 06 v2 section 3).

For a trained model, generate chains with d steps, run it loop by loop (looped models; --unit loop) or layer by layer
(--unit layer, the only choice for dense models) and, after every unit i, train one linear probe per step j = 0..d that
decodes the intermediate value x_j from the state. Two read positions:
  answer : the ARROW position, where the answer is produced -- "what has been computed by unit i"
  local  : the SEP after step j's operation -- "is x_j computed where its operation is read"
The result is a (unit x step) accuracy table per position. A table near the diagonal means one step per loop; a table
that is empty until the last loop means the loop count is a fixed-length program, as in 05 section 5.8.

    python synthetic/probe_synth.py --ckpt runs/synthetic/chain/loop_step_lr0.001_s1.pt --d 8 --r-max 16
Writes results/synthetic/probes/<ckpt name>__d<d>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fit.loop_probe import collect_states  # noqa: E402
from looped.model import LoopedLM, ModelConfig  # noqa: E402
from synthetic.tasks import P, make_batch  # noqa: E402

PROBE_SEED = 777


def collect_layer_states(model, idx: torch.Tensor, r: int | None = None) -> list[torch.Tensor]:
    """State after every executed block, loops unrolled (dense: one per unique layer)."""
    cfg = model.cfg
    r = cfg.r if r is None else r
    x = model.tok_emb(idx)
    out = []
    for b in model.prelude:
        x = b(x, model.rope_cos, model.rope_sin); out.append(x)
    if len(model.core) > 0:
        e = x
        s = torch.zeros_like(e) if model.adapter is not None else e
        for _ in range(r):
            h = model.adapter(torch.cat([s, e], dim=-1)) if model.adapter is not None else s
            for b in model.core:
                h = b(h, model.rope_cos, model.rope_sin); out.append(h)
            s = h
        x = s
    for b in model.coda:
        x = b(x, model.rope_cos, model.rope_sin); out.append(x)
    return out


def gather(states: list[torch.Tensor], positions: torch.Tensor) -> torch.Tensor:
    """(units, B, d_model) at one position per example."""
    ar = torch.arange(positions.size(0), device=positions.device)
    return torch.stack([st[ar, positions] for st in states]).float()


def fit_probes(feat_tr: torch.Tensor, y_tr: torch.Tensor, feat_te: torch.Tensor, y_te: torch.Tensor, steps: int = 300,
               lr: float = 1e-2) -> list[float]:
    """One linear probe per target column; features standardised; returns test accuracy per column."""
    mu, sd = feat_tr.mean(0, keepdim=True), feat_tr.std(0, keepdim=True) + 1e-6
    ftr, fte = (feat_tr - mu) / sd, (feat_te - mu) / sd
    n_col = y_tr.size(1)
    lin = nn.Linear(ftr.size(1), n_col * P).to(ftr.device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        logits = lin(ftr).view(ftr.size(0), n_col, P)
        loss = F.cross_entropy(logits.reshape(-1, P), y_tr.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        pred = lin(fte).view(fte.size(0), n_col, P).argmax(-1)
        return (pred == y_te).float().mean(0).tolist()


@torch.no_grad()
def features(model, task_x: np.ndarray, pos: np.ndarray, d: int, unit: str, r_max: int, device, amp, batch: int = 500):
    """-> answer-position features (units, n, dm) and local features (units, n, d + 1, dm) (x_0 read at the SEP after x_0)."""
    local_pos = np.array([2] + [3 * j + 2 for j in range(1, d + 1)])          # SEP after x0, then SEP after each op
    ans_f, loc_f = [], []
    for s in range(0, len(task_x), batch):
        x = torch.from_numpy(task_x[s:s + batch]).to(device)
        p = torch.from_numpy(pos[s:s + batch]).to(device)
        ctx = torch.autocast(device_type=device.type, dtype=amp) if amp is not None else torch.autocast(device_type=device.type, enabled=False)
        with ctx:
            states = collect_states(model, x, r_max) if unit == "loop" else collect_layer_states(model, x)
        ans_f.append(gather(states, p).cpu())
        loc_f.append(torch.stack([gather(states, torch.full_like(p, int(lp))) for lp in local_pos], dim=2).cpu())
    return torch.cat(ans_f, 1), torch.cat(loc_f, 1)


def probe_checkpoint(model, d: int, unit: str, r_max: int, n_train: int, n_test: int, device, amp, steps: int = 300) -> dict:
    rng = np.random.default_rng(PROBE_SEED)
    x, pos, _, inter = make_batch("chain", d, n_train + n_test, rng)
    model.eval()
    ans_f, loc_f = features(model, x, pos, d, unit, r_max, device, amp)
    y = torch.from_numpy(inter)                                                   # (n, d + 1)
    tr, te = slice(0, n_train), slice(n_train, n_train + n_test)
    table_ans, table_loc = [], []
    for i in range(ans_f.size(0)):
        table_ans.append(fit_probes(ans_f[i, tr].to(device), y[tr].to(device), ans_f[i, te].to(device), y[te].to(device), steps))
        # local: step j's feature comes from its own position
        ftr = torch.stack([loc_f[i, tr, j] for j in range(d + 1)], 1)             # (n_train, d+1, dm)
        fte = torch.stack([loc_f[i, te, j] for j in range(d + 1)], 1)
        accs = []
        for j in range(d + 1):
            accs.append(fit_probes(ftr[:, j].to(device), y[tr, j:j + 1].to(device), fte[:, j].to(device), y[te, j:j + 1].to(device), steps)[0])
        table_loc.append(accs)
    ta, tl = np.array(table_ans), np.array(table_loc)
    first = lambda t: [int(np.argmax(t[:, j] >= 0.9)) + 1 if (t[:, j] >= 0.9).any() else None for j in range(d + 1)]  # noqa: E731
    return dict(unit=unit, d=d, units=int(ta.shape[0]), n_train=n_train, n_test=n_test,
                answer_table=ta.round(4).tolist(), local_table=tl.round(4).tolist(),
                first_unit_decodable_answer=first(ta), first_unit_decodable_local=first(tl),
                final_answer_acc=float(ta[-1, d]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--unit", default="auto", choices=["auto", "loop", "layer"])
    ap.add_argument("--r-max", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/synthetic/probes")
    a = ap.parse_args()
    device = torch.device(a.device)
    amp = torch.bfloat16 if device.type == "cuda" else None
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    from synthetic import tasks
    tasks.set_modulus(ck.get("p", 97))
    tasks.set_family(ck.get("family", "arith"), ck.get("n_ops") or 32)      # the checkpoint's task family
    model = LoopedLM(ModelConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    unit = a.unit if a.unit != "auto" else ("layer" if model.cfg.placement == "dense" else "loop")
    rep = probe_checkpoint(model, a.d, unit, a.r_max, a.n_train, a.n_test, device, amp, a.probe_steps)
    rep.update(ckpt=a.ckpt, cond=ck.get("cond"), task=ck.get("task"), seed=ck.get("seed"), lr=ck.get("lr"))
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.ckpt)[:-3]
    path = os.path.join(a.out, f"{name}__d{a.d}.json")
    json.dump(rep, open(path, "w"), indent=1)
    print(f"{name} ({unit}s x steps, answer position; rows = {unit} 1..{rep['units']}, columns = x_0..x_{a.d}):")
    for i, row in enumerate(rep["answer_table"], start=1):
        print(f"  {unit} {i:2d}: " + " ".join(f"{v:.2f}" for v in row))
    print(f"  first {unit} decodable (>= 0.9) per step: {rep['first_unit_decodable_answer']}")
    print(f"  local position: {rep['first_unit_decodable_local']}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
