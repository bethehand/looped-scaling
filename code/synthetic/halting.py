"""Label-free stopping rules and voting over loop counts on the formal synthetic checkpoints (S8 of
docs/supplementary-experiments-2.md, frozen 2026-10-15).

    python synthetic/halting.py --ckpt-dir runs/synthetic/formal/chain --out results/synthetic/halting --device cuda

For every looped checkpoint: run 32 loops on fresh problems (2000 per step count, d = 1..24; generator seed 777 for
selecting a rule, 888 for reporting), record after every loop the predicted answer, its softmax probability and the
relative change of the answer-position state, and apply the rules of the plan:
  state_<tol>      stop at the first loop whose relative state change is below tol (0.05, 0.02, 0.01, 0.005);
                   state_0.02 is the "adaptive" stop of synthetic/train_synth.py
  pred_<m>         stop when the predicted answer has been the same for m consecutive loops (1, 2, 3)
  conf_<tau>       stop when the top softmax probability is at least tau (0.9, 0.99)
  vote_12_16_24_32 majority over the predictions at loops 12, 16, 24 and 32 (ties: the later loop)
  vote_8_32        majority over the predictions at every loop from 8 to 32
  pred_2_min_own   pred_2, but not before the model's own loop count
References: own (the own loop count: r for fixed loops, 8 for random loops, d for the step-matched model), oracle
(loops = d) and best_fixed (the best of the fixed loop counts 1, 2, 4, 8, 12, 16, 24, 32 per d, chosen with the
answers, an upper bound). A rule that never fires stops at loop 32.
Output: one json per checkpoint: per problem set, per rule, per d: accuracy and mean loops used.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fit.loop_probe import collect_states, lens  # noqa: E402
from looped.model import LoopedLM, ModelConfig  # noqa: E402
from synthetic import tasks  # noqa: E402
from synthetic.tasks import make_batch  # noqa: E402

STATE_TOLS = (0.05, 0.02, 0.01, 0.005)
PRED_MS = (1, 2, 3)
CONF_TAUS = (0.9, 0.99)
FIXED = (1, 2, 4, 8, 12, 16, 24, 32)
VOTE_SETS = {"vote_12_16_24_32": (12, 16, 24, 32), "vote_8_32": tuple(range(8, 33))}
SEEDS = {"select": 777, "report": 888}
OWN_LOOPS = {"loop_r2": 2, "loop_r4": 4, "loop_r8": 8, "loop_rand": 8}      # loop_step: own = d


def own_loops(cond: str, d: int) -> int:
    return OWN_LOOPS.get(cond, d)


def _first(mask: np.ndarray, default: int) -> np.ndarray:
    """Index of the first True along axis 0 of a (R, B) mask, `default` where there is none."""
    hit = mask.any(axis=0)
    return np.where(hit, mask.argmax(axis=0), default)


def _vote(preds: np.ndarray, loops: tuple[int, ...]) -> np.ndarray:
    """Majority over preds[loop - 1] for the given loops; ties go to the value seen at the later loop."""
    sub = preds[[k - 1 for k in loops]]                      # (n, B)
    out = np.empty(preds.shape[1], dtype=np.int64)
    for b in range(preds.shape[1]):
        vals, counts = np.unique(sub[:, b], return_counts=True)
        best = vals[counts == counts.max()]
        last = sub[::-1, b]                                  # later loops first
        out[b] = next(v for v in last if v in set(best.tolist()))
    return out


def apply_rules(preds: np.ndarray, top: np.ndarray, rel: np.ndarray, ans: np.ndarray, own: int, d: int) -> dict:
    """preds, top: (R, B) prediction and its probability after loops 1..R; rel: (R-1, B) relative state change from
    loop i+1 to i+2. Returns rule -> (hits, loops_sum) over the B problems."""
    R, B = preds.shape
    out = {}

    def record(name, stop_idx):
        pred = preds[stop_idx, np.arange(B)]
        out[name] = (int((pred == ans).sum()), int((stop_idx + 1).sum()))

    for tol in STATE_TOLS:
        i = _first(rel < tol, R - 2)                         # the change from loop i+1 to i+2 is small: stop at i+2
        record(f"state_{tol:g}", i + 1)
    same = preds[1:] == preds[:-1]                           # (R-1, B): loop k+1 repeats loop k
    for m in PRED_MS:
        run = np.ones((R - m, B), dtype=bool)
        for j in range(m):
            run &= same[j:j + R - m]                         # m consecutive repeats ending at loop k = j_end + m
        k = _first(run, R - 1 - m) + m
        record(f"pred_{m}", k)
    run2 = same[:-1] & same[1:]                              # two consecutive repeats ending at loop k = idx + 2
    kmin = max(2, own - 1)
    masked = run2.copy()
    masked[: max(0, kmin - 2)] = False
    record("pred_2_min_own", _first(masked, R - 3) + 2)
    for tau in CONF_TAUS:
        record(f"conf_{tau:g}", _first(top >= tau, R - 1))
    for name, loops in VOTE_SETS.items():
        loops = tuple(k for k in loops if k <= R)
        if not loops:                                        # fewer loops than the vote needs (tests)
            continue
        v = _vote(preds, loops)
        out[name] = (int((v == ans).sum()), B * max(loops))
    record("own", np.full(B, min(own, R) - 1))
    record("oracle", np.full(B, min(d, R) - 1))
    for k in FIXED:
        if k <= R:
            record(f"fixed_{k}", np.full(B, k - 1))
    return out


@torch.no_grad()
def loop_traces(model, x: torch.Tensor, pos: torch.Tensor, r_max: int, device, amp):
    """Per loop: predicted answer, its probability, relative state change at the answer position."""
    ctx = (torch.autocast(device_type=device.type, dtype=amp) if amp is not None
           else torch.autocast(device_type=device.type, enabled=False))
    with ctx:
        states = collect_states(model, x, r_max)
        ar = torch.arange(x.size(0), device=device)
        logits = torch.stack([lens(model, st)[ar, pos].float() for st in states])        # (R, B, V)
        sp = torch.stack([st[ar, pos].float() for st in states])                            # (R, B, D)
    probs = torch.softmax(logits, dim=-1)
    top, preds = probs.max(dim=-1)
    rel = (sp[1:] - sp[:-1]).norm(dim=-1) / (sp[:-1].norm(dim=-1) + 1e-6)
    return preds.cpu().numpy(), top.cpu().numpy(), rel.cpu().numpy()


def evaluate_checkpoint(model, cond: str, ds: list[int], n: int, r_max: int, device, amp, batch: int = 500,
                        seeds: dict = SEEDS) -> dict:
    out = {}
    for set_name, seed in seeds.items():
        rng = np.random.default_rng(seed)
        res = {}
        for d in ds:
            x_all, pos_all, ans_all, _ = make_batch("chain", d, n, rng)
            acc = {}
            for s in range(0, n, batch):
                x = torch.from_numpy(x_all[s:s + batch]).to(device)
                pos = torch.from_numpy(pos_all[s:s + batch]).to(device)
                preds, top, rel = loop_traces(model, x, pos, r_max, device, amp)
                for name, (hits, loops) in apply_rules(preds, top, rel, ans_all[s:s + batch], own_loops(cond, d), d).items():
                    h, l = acc.get(name, (0, 0))
                    acc[name] = (h + hits, l + loops)
            res[str(d)] = {name: dict(acc=round(h / n, 4), loops=round(l / n, 2)) for name, (h, l) in acc.items()}
            best = max((k for k in FIXED if k <= r_max), key=lambda k: res[str(d)][f"fixed_{k}"]["acc"])
            res[str(d)]["best_fixed"] = dict(acc=res[str(d)][f"fixed_{best}"]["acc"], loops=float(best))
        out[set_name] = res
    return out


def load_checkpoint(path: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    tasks.set_modulus(ck.get("p", 97))
    tasks.set_family(ck.get("family", "arith"), ck.get("n_ops") or 32)
    model = LoopedLM(ModelConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="runs/synthetic/formal/chain")
    ap.add_argument("--ckpt", nargs="*", default=None, help="checkpoint files instead of every looped one in --ckpt-dir")
    ap.add_argument("--out", default="results/synthetic/halting")
    ap.add_argument("--d", default="1..24")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--r-max", type=int, default=32)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    amp = torch.bfloat16 if device.type == "cuda" else None
    lo, hi = a.d.split("..")
    ds = list(range(int(lo), int(hi) + 1))
    paths = a.ckpt or sorted(p for p in glob.glob(os.path.join(a.ckpt_dir, "*.pt")) if os.path.basename(p).startswith("loop_"))
    os.makedirs(a.out, exist_ok=True)
    for p in paths:
        name = os.path.basename(p)[:-3]
        dst = os.path.join(a.out, name + ".json")
        if os.path.exists(dst):
            print("exists", dst, flush=True)
            continue
        model, ck = load_checkpoint(p, device)
        res = evaluate_checkpoint(model, ck["cond"], ds, a.n, a.r_max, device, amp, a.batch)
        rep = dict(ckpt=p, cond=ck["cond"], seed=ck.get("seed"), lr=ck.get("lr"), n=a.n, r_max=a.r_max, d=ds,
                   seeds=SEEDS, rules=sorted(res["report"][str(ds[0])]), results=res)
        json.dump(rep, open(dst, "w"), indent=1)
        r12 = res["report"].get("12", {})
        print(f"{name}: 12 steps  own {r12.get('own', {}).get('acc')}  state_0.02 {r12.get('state_0.02', {}).get('acc')}  "
              f"pred_2 {r12.get('pred_2', {}).get('acc')} ({r12.get('pred_2', {}).get('loops')} loops)  "
              f"best_fixed {r12.get('best_fixed', {}).get('acc')}", flush=True)


if __name__ == "__main__":
    main()
