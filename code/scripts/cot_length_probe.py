"""Exploratory diagnostic written after S5 (2026-10-17, not in the frozen plan): where does the chain of a chain-of-thought
model break beyond the training range (d > 8)? For each checkpoint and step count d it reports, over 2,000 problems drawn
with the formal evaluation's generator seed (12345) but re-seeded for each d, so not the formal evaluation's own
problems: the per-step accuracy under free generation (token j right given the model's
own earlier tokens), the distribution of the first wrong step, and the per-step accuracy under teacher forcing (token j
right given the true earlier tokens).

    python scripts/cot_length_probe.py --ckpt runs/synthetic/supp2/formal/chain/dense8_cot_lr0.000125_s2_lut32.pt \
        --d 6 8 9 10 12 16 --out results/synthetic/supp2/cot_length_probe.json --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from looped.model import LoopedLM, ModelConfig  # noqa: E402
from synthetic import tasks  # noqa: E402
from synthetic.tasks import make_batch  # noqa: E402

EVAL_SEED = 12345   # synthetic/train_synth.py


def load(path: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    tasks.set_modulus(ck.get("p", 97))
    tasks.set_family(ck.get("family", "arith"), ck.get("n_ops") or 32)
    model = LoopedLM(ModelConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck


@torch.no_grad()
def logits_at(model, x, idx, device, amp):
    with torch.autocast(device_type=device.type, dtype=amp or torch.bfloat16, enabled=amp is not None):
        logits, _ = model(x, None)
    return logits[torch.arange(x.size(0), device=device)[:, None], idx].float()


@torch.no_grad()
def probe(model, d: int, n: int, device, amp, batch: int = 500) -> dict:
    rng = np.random.default_rng(EVAL_SEED)
    x_all, pos_all, ans_all, inter_all = make_batch("chain", d, n, rng, "cot")
    free_ok = np.zeros((n, d), bool)
    tf_ok = np.zeros((n, d), bool)
    for s in range(0, n, batch):
        x = torch.from_numpy(x_all[s:s + batch]).to(device)
        pos = torch.from_numpy(pos_all[s:s + batch]).to(device)
        B = x.size(0)
        ar = torch.arange(B, device=device)
        js = torch.arange(d, device=device)
        truth = torch.from_numpy(inter_all[s:s + batch, 1:]).to(device)
        tf = logits_at(model, x, pos[:, None] + js[None, :], device, amp).argmax(-1)      # true chain in x
        tf_ok[s:s + B] = (tf == truth).cpu().numpy()
        xg = x.clone()
        xg[ar[:, None], pos[:, None] + 1 + js[None, :]] = tasks.PAD                          # hide the chain
        for j in range(d):
            nxt = logits_at(model, xg, (pos + j)[:, None], device, amp)[:, 0].argmax(-1)
            xg[ar, pos + 1 + j] = nxt
        gen = xg[ar[:, None], pos[:, None] + 1 + js[None, :]]
        free_ok[s:s + B] = (gen == truth).cpu().numpy()
    first_wrong = np.where(free_ok.all(1), 0, free_ok.argmin(1) + 1)                          # 0 = whole chain right
    ks, vs = np.unique(first_wrong, return_counts=True)
    return dict(d=d, n=n, final_acc=round(float(free_ok[:, -1].mean()), 4), chain_acc=round(float(free_ok.all(1).mean()), 4),
                free_step_acc=[round(float(a), 4) for a in free_ok.mean(0)],
                teacher_forced_step_acc=[round(float(a), 4) for a in tf_ok.mean(0)],
                first_wrong_step_hist={int(k): int(v) for k, v in zip(ks, vs)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--d", nargs="+", type=int, default=[6, 8, 9, 10, 12, 16])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results/synthetic/supp2/cot_length_probe.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    amp = torch.bfloat16 if device.type == "cuda" else None
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for p in a.ckpt:
        name = os.path.basename(p)[:-3]
        model, ck = load(p, device)
        out[name] = dict(ckpt=p, cond=ck.get("cond"), seed=ck.get("seed"), lr=ck.get("lr"), results={})
        for d in a.d:
            r = probe(model, d, a.n, device, amp)
            out[name]["results"][str(d)] = r
            print(f"{name} d={d:2d}: final {r['final_acc']:.3f} chain {r['chain_acc']:.3f}  free per step {r['free_step_acc']}  "
                  f"teacher-forced {r['teacher_forced_step_acc']}  first wrong {r['first_wrong_step_hist']}", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
