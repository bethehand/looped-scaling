"""S6 (docs/supplementary-experiments-2.md, frozen 2026-10-15): the loss of token-by-token decoding with one KV cache per
executed layer (as trained) against one cache per unique layer shared across the loops (looped/infer.py shared=True).

    python scripts/kv_share_eval.py --runs 20M_middle_r8_full_s42 20M_middle_r8_full_rs1to8_s42@16 \\
        --ckpt branch_858914816 --out results/supp2/kv_share.json --device cuda

A run is "<name>" or "<name>@<loops>" (inference loop count; default the trained one). The checkpoint is the branch
start of the 40N budget (the 32N trunk at 20M), as in results 5.7-5.9. Loss on the first --windows windows of 1024
tokens of the main validation set, every token predicted after decoding the ones before it; the full-cache loss is
checked against the ordinary forward pass of the same windows.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from looped.data import ValSet  # noqa: E402
from looped.infer import StaticCache, decode_step, prefill  # noqa: E402
from looped.model import LoopedLM  # noqa: E402
from looped.train import load_run_config  # noqa: E402

CONFIG_DIRS = ("configs/runs", "configs/ext", "configs/supp", "configs/supp2")


def find_config(run: str) -> str:
    for d in CONFIG_DIRS:
        p = os.path.join(d, run + ".yaml")
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join("configs", "**", run + ".yaml"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"no config for {run}")
    return hits[0]


@torch.no_grad()
def decode_loss(model: LoopedLM, x: torch.Tensor, y: torch.Tensor, r: int | None, shared: bool) -> float:
    """Sum of the cross-entropy over all positions, every token decoded with the cache (prefill of the first token)."""
    B, T = x.shape
    cache = StaticCache.for_model(model, B, max_len=T, r=r, shared=shared)
    logits = [prefill(model, x[:, :1], cache, r=r)[:, 0]]
    for t in range(1, T):
        logits.append(decode_step(model, x[:, t:t + 1], cache, r=r)[:, 0])
    logits = torch.stack(logits, 1).float()
    return float(F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))


@torch.no_grad()
def forward_loss(model: LoopedLM, x: torch.Tensor, y: torch.Tensor, r: int | None) -> float:
    logits, _ = model(x, None, r=r)
    logits = logits.float()
    return float(F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))


def evaluate_run(run: str, r_eval: int | None, ckpt: str, valset: str, windows: int, batch: int, device) -> dict:
    name, out_dir, mcfg, tcfg, dcfg = load_run_config(find_config(run))
    model = LoopedLM(mcfg).to(device).eval()
    ck = torch.load(os.path.join("runs", run, ckpt + ".pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.float()
    val = ValSet(dcfg.val_sets[valset], mcfg.seq_len, windows * mcfg.seq_len + 1)
    tot = dict(forward=0.0, full=0.0, shared=0.0)
    n = 0
    t0 = time.time()
    for x, y in val.batches(batch):
        x, y = x.to(device), y.to(device)
        tot["forward"] += forward_loss(model, x, y, r_eval)
        tot["full"] += decode_loss(model, x, y, r_eval, shared=False)
        tot["shared"] += decode_loss(model, x, y, r_eval, shared=True)
        n += y.numel()
    r = mcfg.r if r_eval is None else r_eval
    p, c, q = mcfg.layer_split
    res = dict(run=run, ckpt=ckpt, tokens_seen=int(ck.get("tokens_seen", 0)), placement=mcfg.placement, r_train=mcfg.r,
               k_bwd=mcfg.k_bwd, r_sample=tcfg.r_sample, lr0=tcfg.lr0, r_eval=r, valset=valset, n_tokens=n,
               loss_forward=round(tot["forward"] / n, 4), loss_full_cache=round(tot["full"] / n, 4),
               loss_shared_cache=round(tot["shared"] / n, 4), delta_shared_minus_full=round((tot["shared"] - tot["full"]) / n, 4),
               cache_slots_full=p + r * c + q, cache_slots_shared=p + c + q, seconds=round(time.time() - t0))
    print(f"{run}@{r}: forward {res['loss_forward']}  full cache {res['loss_full_cache']}  shared {res['loss_shared_cache']}  "
          f"delta {res['delta_shared_minus_full']:+.4f}  ({res['seconds']}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="<run name>[@<inference loops>]")
    ap.add_argument("--ckpt", default="branch_858914816", help="checkpoint file name without .pt (20M: the 32N trunk)")
    ap.add_argument("--valset", default="fwe")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="results/supp2/kv_share.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    done = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in a.runs:
        run, _, r = spec.partition("@")
        key = spec
        if key in done:
            print("exists", key, flush=True)
            continue
        done[key] = evaluate_run(run, int(r) if r else None, a.ckpt, a.valset, a.windows, a.batch, device)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(done, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
