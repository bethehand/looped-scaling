"""Train one model on one synthetic task and evaluate it by step count and inference loop count (extension C).

Conditions (06_合成推理任务设计草案.md v2, section 2): the same 8-unique-layer architecture as the main grid at width 256,
    dense8 / dense20 / dense36      : no loops; 20 and 36 unique layers match the executed depth of r = 4 / 8
    loop_r2 / loop_r4 / loop_r8     : middle block looped a fixed number of times
    loop_rand                       : r = 8 architecture, r drawn from U{1..8} at every step (as in 05 section 5.9)
    loop_step                       : r = 8 architecture, r = the step count d of the batch (Fan et al. 2024 style)
Every batch holds problems of one step count d ~ U{1..d_train}; the loss is taken at the answer position only.
Evaluation: accuracy per step count (d up to 3x the training range) for every inference loop count in --r-eval,
plus for looped models "oracle" (loops = d) and "adaptive" (stop when the relative change of the answer-position
state between loops falls below --tol).

    python synthetic/train_synth.py --task chain --cond loop_step --seed 1 --device cuda
Writes results/synthetic/<task>/<cond>_lr<lr>_s<seed>.json and runs/synthetic/<task>/<cond>_lr<lr>_s<seed>.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fit.loop_probe import collect_states, lens  # noqa: E402
from looped.model import LoopedLM, ModelConfig  # noqa: E402
from synthetic import tasks  # noqa: E402
from synthetic.tasks import SEQ_LEN, VOCAB_SIZE, make_batch  # noqa: E402

CONDITIONS = {
    "dense8": dict(placement="dense", n_layers=8, r=1, regime="fixed"),
    "dense20": dict(placement="dense", n_layers=20, r=1, regime="fixed"),
    "dense36": dict(placement="dense", n_layers=36, r=1, regime="fixed"),
    "loop_r2": dict(placement="middle", n_layers=8, r=2, regime="fixed"),
    "loop_r4": dict(placement="middle", n_layers=8, r=4, regime="fixed"),
    "loop_r8": dict(placement="middle", n_layers=8, r=8, regime="fixed"),
    "loop_rand": dict(placement="middle", n_layers=8, r=8, regime="random"),
    "loop_step": dict(placement="middle", n_layers=8, r=8, regime="step"),
}
EVAL_SEED = 12345     # every condition is tested on the same problems


def build_model(cond: str, task: str, width: int) -> LoopedLM:
    c = CONDITIONS[cond]
    cfg = ModelConfig(vocab_size=VOCAB_SIZE, d_model=width, n_layers=c["n_layers"], head_dim=(64 if width % 64 == 0 else width),
                      seq_len=SEQ_LEN[task],
                      placement=c["placement"], n_prelude=2, n_coda=2, r=c["r"], k_bwd=0)
    return LoopedLM(cfg)


def _ctx(device, amp):
    return (torch.autocast(device_type=device.type, dtype=amp) if amp is not None
            else torch.autocast(device_type=device.type, enabled=False))


def answer_logits(model, x: torch.Tensor, pos: torch.Tensor, r: int | None, device, amp) -> torch.Tensor:
    with _ctx(device, amp):
        logits, _ = model(x, None, r=r)
    return logits[torch.arange(x.size(0), device=device), pos].float()


def parse_range(spec: str) -> list[int]:
    if ".." in spec:
        a, b = spec.split("..")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


@torch.no_grad()
def evaluate(model, task: str, ds: list[int], r_list: list[int] | None, n: int, device, amp, batch: int = 500,
             tol: float | None = None, r_max: int = 32) -> dict:
    """{d: {"r=4": acc, ..., "oracle": acc, "adaptive": acc, "adaptive_loops": mean loops}} ; dense: {d: {"acc": acc}}."""
    model.eval()
    rng = np.random.default_rng(EVAL_SEED)
    out = {}
    for d in ds:
        x_all, pos_all, ans_all, _ = make_batch(task, d, n, rng)
        hits, loops_used = {}, 0.0
        for s in range(0, n, batch):
            x = torch.from_numpy(x_all[s:s + batch]).to(device)
            pos = torch.from_numpy(pos_all[s:s + batch]).to(device)
            ans = torch.from_numpy(ans_all[s:s + batch]).to(device)
            if r_list is None:
                hits["acc"] = hits.get("acc", 0) + int((answer_logits(model, x, pos, None, device, amp).argmax(-1) == ans).sum())
                continue
            for r in r_list:
                key = f"r={r}"
                hits[key] = hits.get(key, 0) + int((answer_logits(model, x, pos, r, device, amp).argmax(-1) == ans).sum())
            hits["oracle"] = hits.get("oracle", 0) + int((answer_logits(model, x, pos, min(d, r_max), device, amp).argmax(-1) == ans).sum())
            if tol is not None:
                with _ctx(device, amp):
                    states = collect_states(model, x, r_max)
                    preds = torch.stack([lens(model, st)[torch.arange(x.size(0), device=device), pos].float().argmax(-1)
                                         for st in states])                                   # (r_max, B)
                sp = torch.stack([st[torch.arange(x.size(0), device=device), pos].float() for st in states])  # (r_max, B, d)
                rel = (sp[1:] - sp[:-1]).norm(dim=-1) / (sp[:-1].norm(dim=-1) + 1e-6)                        # (r_max-1, B)
                stop = torch.full((x.size(0),), r_max - 1, dtype=torch.long, device=device)
                small = rel < tol
                for i in range(r_max - 2, -1, -1):
                    stop = torch.where(small[i], torch.full_like(stop, i + 1), stop)     # first loop whose change is small
                pred = preds[stop, torch.arange(x.size(0), device=device)]
                hits["adaptive"] = hits.get("adaptive", 0) + int((pred == ans).sum())
                loops_used += float((stop + 1).sum())
        row = {k: round(v / n, 4) for k, v in hits.items()}
        if tol is not None and r_list is not None:
            row["adaptive_loops"] = round(loops_used / n, 2)
        out[d] = row
    model.train()
    return out


def train_and_eval(a) -> dict:
    tasks.set_modulus(getattr(a, "p", 97))
    device = torch.device(a.device)
    amp = torch.bfloat16 if device.type == "cuda" else None
    cond = CONDITIONS[a.cond]
    torch.manual_seed(a.seed)
    model = build_model(a.cond, a.task, a.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.param_groups(a.lr, a.weight_decay), lr=a.lr, betas=(0.9, 0.95),
                            fused=(device.type == "cuda"))
    base = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(a.seed + 1000)
    r_rng = np.random.default_rng(a.seed + 2000)
    ds_train = list(range(1, a.d_train + 1))
    ds_test = parse_range(a.d_test)
    r_eval = None if cond["placement"] == "dense" else sorted(set(parse_range(a.r_eval)) | {cond["r"]})   # nominal r always included
    log, curve, t0 = [], [], time.time()
    acc_loss, acc_hit, n_acc = 0.0, 0, 0
    model.train()
    for step in range(1, a.steps + 1):
        frac = min(1.0, step / max(1, a.warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, a.steps) / a.steps)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * frac
        # curriculum (pilot finding 2026-10-10): the largest step count grows linearly from 1 to d_train over the first
        # curriculum_frac of training, then the full mixture; 0 = the flat mixture of the frozen design
        d_max = a.d_train if a.curriculum_frac <= 0 else min(a.d_train, max(1, math.ceil(a.d_train * step / (a.curriculum_frac * a.steps))))
        d = int(rng.integers(1, d_max + 1))
        x, pos, ans, _ = make_batch(a.task, d, a.batch, rng)
        x, pos, ans = torch.from_numpy(x).to(device), torch.from_numpy(pos).to(device), torch.from_numpy(ans).to(device)
        r = {"fixed": None, "random": int(r_rng.integers(1, cond["r"] + 1)), "step": d}[cond["regime"]]
        logits = answer_logits(model, x, pos, r, device, amp)
        loss = F.cross_entropy(logits, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        acc_loss += float(loss.detach()); acc_hit += int((logits.argmax(-1) == ans).sum()); n_acc += 1
        if step % a.log_every == 0 or step == a.steps:
            line = dict(step=step, loss=round(acc_loss / n_acc, 4), acc=round(acc_hit / (n_acc * a.batch), 4),
                        lr=round(a.lr * frac, 6), seconds=round(time.time() - t0))
            log.append(line)
            print(f"[{a.task}/{a.cond} s{a.seed}] step {step} loss {line['loss']:.4f} train-acc {line['acc']:.3f} "
                  f"lr {line['lr']:.2e} ({line['seconds']}s)", flush=True)
            acc_loss, acc_hit, n_acc = 0.0, 0, 0
        if step % a.eval_every == 0 and step < a.steps:
            quick = evaluate(model, a.task, ds_train, None if r_eval is None else [cond["r"]], a.quick_n, device, amp)
            key = "acc" if r_eval is None else f"r={cond['r']}"
            curve.append(dict(step=step, acc_by_d={d: v[key] for d, v in quick.items()}))
            print(f"[{a.task}/{a.cond} s{a.seed}] step {step} quick acc by d: " +
                  " ".join(f"{d}:{v[key]:.2f}" for d, v in quick.items()), flush=True)
    final = evaluate(model, a.task, ds_test, r_eval, a.eval_n, device, amp, tol=(a.tol if r_eval is not None else None))
    rep = dict(task=a.task, cond=a.cond, seed=a.seed, lr=a.lr, steps=a.steps, batch=a.batch, width=a.width, p=tasks.P,
               curriculum_frac=a.curriculum_frac,
               d_train=a.d_train, n_params=n_params, condition=cond, executed_layers=model.cfg.executed_layers,
               train_log=log, quick_curve=curve, final=final, eval_n=a.eval_n, tol=a.tol,
               seconds=round(time.time() - t0), torch_version=torch.__version__)
    name = f"{a.cond}_lr{a.lr:g}_s{a.seed}" + (f"_p{tasks.P}" if tasks.P != 97 else "")
    os.makedirs(os.path.join(a.out, a.task), exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, a.task, name + ".json"), "w"), indent=1)
    if a.ckpt_dir:
        os.makedirs(os.path.join(a.ckpt_dir, a.task), exist_ok=True)
        torch.save(dict(model=model.state_dict(), cfg=model.cfg.to_dict(), cond=a.cond, task=a.task, seed=a.seed, lr=a.lr),
                   os.path.join(a.ckpt_dir, a.task, name + ".pt"))
    key = "acc" if r_eval is None else f"r={cond['r']}"
    print(f"[{a.task}/{a.cond} s{a.seed}] final acc by d ({key}): " + " ".join(f"{d}:{v[key]:.2f}" for d, v in final.items()), flush=True)
    if r_eval is not None:
        print(f"[{a.task}/{a.cond} s{a.seed}] oracle (loops = d): " + " ".join(f"{d}:{v['oracle']:.2f}" for d, v in final.items()), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chain", choices=["chain", "hops"])
    ap.add_argument("--cond", default="loop_r8", choices=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--d-train", type=int, default=8)
    ap.add_argument("--p", type=int, default=97, help="modulus of the chain task (pilot knob; the design says 97)")
    ap.add_argument("--curriculum-frac", type=float, default=0.0,
                    help="> 0: the largest step count grows from 1 to d_train over this fraction of training (0 = flat mixture)")
    ap.add_argument("--d-test", default=None, help="e.g. 1..24 (default: chain 1..24, hops 1..16)")
    ap.add_argument("--r-eval", default="1,2,4,8,12,16,24,32")
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--quick-n", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--tol", type=float, default=0.02, help="adaptive stopping: relative state change below this")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/synthetic")
    ap.add_argument("--ckpt-dir", default="runs/synthetic")
    a = ap.parse_args()
    if a.d_test is None:
        a.d_test = "1..24" if a.task == "chain" else "1..16"
    train_and_eval(a)


if __name__ == "__main__":
    main()
