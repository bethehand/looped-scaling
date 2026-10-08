"""EXPLORATORY per-token analysis: which tokens do loops help, and does looping more at inference help?

Stage "raw" (needs the checkpoints and validation files, i.e. the GPU machine, or a Mac with copies): for every run,
build the model from configs/runs/<run>.yaml, load a checkpoint and store the per-token cross-entropy on the first
--tokens tokens of each validation set, for each inference loop count r_eval (looped runs only; r_eval above the
training r tests "thinking longer" at inference). Checkpoint choice (--ckpt):
  branch40 (default): the branch checkpoint saved at 0.8 x the budget closest to --budget-mult N, i.e. the same token
                      count for the dense and the looped model of a rung (the cooled-down weights were never saved,
                      so models are compared just before the cooldown of the 40N budget);
  trunk_latest      : end of each run's own trunk -- NOT matched (the dense ruler trains to 80N, loops to 40N).
  -> <out>/raw/<run>__<ckpt>__<valset>__r<r>.npy   (float16, one value per predicted token; not tracked by git)

Stage "summary" (CPU): tables that are small enough to commit.
  mean_loss.csv     mean loss per run / valset / r_eval
  delta_stats.csv   looped (at its training r) minus dense of the same rung, per token: mean, quantiles, share of
                    tokens improved, share of the total gain carried by the top 10% of tokens (concentration)
  by_class.csv      the same split by token class (word / number / punctuation / code symbol / whitespace / other);
                    share_of_positive_gain = this class's part of the summed per-token improvements
  by_difficulty.csv the same split by deciles of the two models' average per-token loss (symmetric binning)
  A second dense seed of a rung, when listed, is compared against the first exactly like a looped run: it is the null
  distribution for every statistic (two equally good models already disagree a lot per token).
  depth_curve.csv   loss vs r_eval for looped runs, and the per-token benefit of looping beyond the training r

    python fit/token_analysis.py --stage all --runs 80M_dense_r1_s42 80M_middle_r8_full_s42 ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CLASSES = ["word", "number", "punct", "code", "space", "other"]


def classify_tokens(tokenizer_path: str, vocab_size: int) -> np.ndarray:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    cls = np.full(vocab_size, CLASSES.index("other"), dtype=np.int8)
    code_chars = set("{}()[];=<>+-*/%&|^~#@\\$`_:")
    for i in range(vocab_size):
        s = tok.decode([i])
        t = s.strip()
        if s == "" or t == "":
            cls[i] = CLASSES.index("space")
        elif any(c.isdigit() for c in t):
            cls[i] = CLASSES.index("number")
        elif all(c.isalpha() for c in t):
            cls[i] = CLASSES.index("word")
        elif all((not c.isalnum()) for c in t) and any(c in code_chars for c in t):
            cls[i] = CLASSES.index("code")
        elif all((not c.isalnum()) for c in t):
            cls[i] = CLASSES.index("punct")
    return cls


def pick_device(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_config(run: str) -> str:
    """Main-grid runs live in configs/runs, the 2026-10-08 extension runs in configs/ext, exploratory ones in configs/explore."""
    for d in ("runs", "ext", "explore"):
        p = os.path.join("configs", d, run + ".yaml")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no config for {run} under configs/{{runs,ext,explore}}")


def n_rung_table() -> dict[str, int]:
    """N of the rung per run name, from every manifest."""
    import csv
    import glob
    out = {}
    for m in glob.glob(os.path.join("configs", "manifest*.csv")):
        for row in csv.DictReader(open(m)):
            out[row["name"]] = int(row["N_rung"])
    return out


def checkpoint_name(a, tcfg, n_rung: int) -> str:
    if a.ckpt == "trunk_latest":
        return "trunk_latest"
    b = min(tcfg.budgets, key=lambda x: abs(x / n_rung - a.budget_mult))
    start = int(round(b * (1 - tcfg.cooldown_frac)))
    start -= start % tcfg.batch_tokens
    return f"branch_{start}"


def stage_raw(a) -> None:
    import torch
    import torch.nn.functional as F
    from looped.data import ValSet
    from looped.model import LoopedLM
    from looped.train import load_run_config
    device = pick_device(a.device)
    amp = torch.bfloat16 if device.type == "cuda" else None     # exact fp32 elsewhere
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    n_rung = n_rung_table()
    for run in a.runs:
        _, _, mcfg, tcfg, dcfg = load_run_config(find_config(run))
        model = LoopedLM(mcfg).to(device).eval()
        ck_name = checkpoint_name(a, tcfg, n_rung[run])
        ck = torch.load(os.path.join("runs", run, ck_name + ".pt"), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"{run}: checkpoint {ck_name} ({ck['tokens_seen'] / 1e6:.0f}M tokens, step {ck['step']})", flush=True)
        r_list = [None] if mcfg.placement == "dense" else [int(r) for r in a.r_eval.split(",")]
        for vs in a.valsets.split(","):
            val = ValSet(dcfg.val_sets[vs], mcfg.seq_len, a.tokens)
            targets_path = os.path.join(a.out, "raw", f"targets__{vs}.npy")
            if not os.path.exists(targets_path):
                np.save(targets_path, np.asarray(val.tokens[1 : val.n_windows * mcfg.seq_len + 1]).astype(np.uint16))
            for r in r_list:
                path = os.path.join(a.out, "raw", f"{run}__{a.ckpt}__{vs}__r{r if r is not None else mcfg.r}.npy")
                if os.path.exists(path):
                    continue
                out = np.empty(val.n_windows * mcfg.seq_len, dtype=np.float16)
                pos = 0
                with torch.no_grad():
                    for x, y in val.batches(a.batch):
                        x, y = x.to(device), y.to(device)
                        ctx = torch.autocast(device_type=device.type, dtype=amp) if amp else torch.autocast(device_type=device.type, enabled=False)
                        with ctx:
                            logits, _ = model(x, None, r=r)
                        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="none")
                        n = loss.numel()
                        out[pos:pos + n] = loss.to(torch.float16).cpu().numpy()
                        pos += n
                np.save(path, out[:pos])
                print(f"{run} {vs} r={r}: mean {float(out[:pos].astype(np.float32).mean()):.4f} over {pos} tokens", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _load_raw(out: str, run: str, vs: str, r: int, ckpt: str = "branch40") -> np.ndarray | None:
    p = os.path.join(out, "raw", f"{run}__{ckpt}__{vs}__r{r}.npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


def _meta(run: str) -> dict:
    import yaml
    cfg = yaml.safe_load(open(find_config(run)))["model"]
    rung = run.split("_")[0]
    cell = "dense" if cfg["placement"] == "dense" else f"{cfg['placement']}_r{cfg['r']}" + (f"_k{cfg['k_bwd']}" if cfg["k_bwd"] else "")
    return dict(rung=rung, cell=cell, placement=cfg["placement"], r=cfg["r"], k=cfg["k_bwd"], vocab=cfg["vocab_size"])


def stage_summary(a) -> None:
    import pandas as pd
    metas = {run: _meta(run) for run in a.runs}
    dense_of = {}
    for run, m in metas.items():                       # reference = the first dense run listed for each rung
        if m["placement"] == "dense":
            dense_of.setdefault(m["rung"], run)
    for run, m in metas.items():                       # further dense seeds are compared like looped runs: the null
        if m["placement"] == "dense" and dense_of[m["rung"]] != run:
            m["cell"] = "dense_seed" + run.rsplit("_s", 1)[1]
    cls_table = classify_tokens(a.tokenizer, next(iter(metas.values()))["vocab"]) if os.path.exists(a.tokenizer) else None
    mean_rows, delta_rows, class_rows, diff_rows, depth_rows = [], [], [], [], []
    for run, m in metas.items():
        r_list = [m["r"]] if m["placement"] == "dense" else [int(r) for r in a.r_eval.split(",")]
        for vs in a.valsets.split(","):
            for r in r_list:
                x = _load_raw(a.out, run, vs, r, a.ckpt)
                if x is None:
                    continue
                mean_rows.append(dict(run=run, rung=m["rung"], cell=m["cell"], r_train=m["r"], r_eval=r, valset=vs,
                                      mean_loss=round(float(x.mean()), 5), n_tokens=len(x)))
            if run == dense_of.get(m["rung"]) or m["rung"] not in dense_of:
                continue
            d = _load_raw(a.out, dense_of[m["rung"]], vs, metas[dense_of[m["rung"]]]["r"], a.ckpt)
            l = _load_raw(a.out, run, vs, m["r"], a.ckpt)
            if d is None or l is None:
                continue
            n = min(len(d), len(l)); d, l = d[:n], l[:n]
            delta = d - l                                  # > 0: the looped model is better on this token
            gain = np.clip(delta, 0, None)
            top = np.sort(gain)[::-1]
            k = max(1, n // 10)
            q = np.percentile(delta, [5, 25, 50, 75, 95])
            delta_rows.append(dict(run=run, rung=m["rung"], cell=m["cell"], valset=vs, n_tokens=n,
                                   mean_delta=round(float(delta.mean()), 5), frac_improved=round(float((delta > 0).mean()), 4),
                                   p5=round(float(q[0]), 4), p25=round(float(q[1]), 4), p50=round(float(q[2]), 4),
                                   p75=round(float(q[3]), 4), p95=round(float(q[4]), 4),
                                   top10pct_share_of_gain=round(float(top[:k].sum() / max(top.sum(), 1e-9)), 4),
                                   top1pct_share_of_gain=round(float(top[:max(1, n // 100)].sum() / max(top.sum(), 1e-9)), 4)))
            tgt_path = os.path.join(a.out, "raw", f"targets__{vs}.npy")
            if cls_table is not None and os.path.exists(tgt_path):
                tgt = np.load(tgt_path)[:n].astype(np.int64)
                c = cls_table[tgt]
                for ci, cname in enumerate(CLASSES):
                    sel = c == ci
                    if sel.sum() == 0:
                        continue
                    class_rows.append(dict(run=run, rung=m["rung"], cell=m["cell"], valset=vs, token_class=cname,
                                           share_of_tokens=round(float(sel.mean()), 4), dense_loss=round(float(d[sel].mean()), 4),
                                           looped_loss=round(float(l[sel].mean()), 4), mean_delta=round(float(delta[sel].mean()), 5),
                                           share_of_positive_gain=round(float(gain[sel].sum() / max(gain.sum(), 1e-9)), 4)))
            avg = 0.5 * (d + l)                            # bin by the two models' mean loss: symmetric, no regression to the mean
            edges = np.percentile(avg, np.linspace(0, 100, 11))
            dec = np.clip(np.searchsorted(edges, avg, side="right") - 1, 0, 9)
            for i in range(10):
                sel = dec == i
                diff_rows.append(dict(run=run, rung=m["rung"], cell=m["cell"], valset=vs, avg_loss_decile=i + 1,
                                      avg_loss=round(float(avg[sel].mean()), 4), dense_loss=round(float(d[sel].mean()), 4),
                                      mean_delta=round(float(delta[sel].mean()), 5),
                                      share_of_positive_gain=round(float(gain[sel].sum() / max(gain.sum(), 1e-9)), 4)))
            if m["placement"] == "dense":
                continue
            r_eval = [int(r) for r in a.r_eval.split(",")]
            curve = {r: _load_raw(a.out, run, vs, r, a.ckpt) for r in r_eval}
            curve = {r: v[:n] for r, v in curve.items() if v is not None}
            if len(curve) >= 2:
                r_max = max(curve)
                extra = l - curve[r_max] if r_max > m["r"] else None     # benefit of looping beyond the training depth
                row = dict(run=run, rung=m["rung"], cell=m["cell"], valset=vs, r_train=m["r"],
                           **{f"loss_r{r}": round(float(v.mean()), 5) for r, v in curve.items()})
                if extra is not None:
                    row.update(extra_loop_mean_delta=round(float(extra.mean()), 5),
                               extra_loop_frac_improved=round(float((extra > 0).mean()), 4),
                               spearman_extra_vs_dense_gain=round(float(pd.Series(extra).corr(pd.Series(delta), method="spearman")), 4))
                depth_rows.append(row)
    for name, rows in (("mean_loss", mean_rows), ("delta_stats", delta_rows), ("by_class", class_rows),
                       ("by_difficulty", diff_rows), ("depth_curve", depth_rows)):
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(a.out, name + ".csv"), index=False)
            print(f"{name}: {len(rows)} rows")
    json.dump(dict(runs=a.runs, valsets=a.valsets, tokens=a.tokens, r_eval=a.r_eval, checkpoint=a.ckpt,
                   budget_mult=a.budget_mult,
                   note="exploratory; dense and looped compared at the same token count (branch checkpoint before the"
                        " cooldown of the budget closest to budget_mult x N) unless ckpt=trunk_latest"),
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)


def _stats(delta: np.ndarray) -> dict:
    gain = np.clip(delta, 0, None)
    top = np.sort(gain)[::-1]
    q = np.percentile(delta, [5, 25, 50, 75, 95])
    return dict(mean_delta=round(float(delta.mean()), 5), frac_improved=round(float((delta > 0).mean()), 4),
                p5=round(float(q[0]), 4), p25=round(float(q[1]), 4), p50=round(float(q[2]), 4), p75=round(float(q[3]), 4),
                p95=round(float(q[4]), 4), top10pct_share_of_gain=round(float(top[:max(1, len(delta) // 10)].sum() / max(top.sum(), 1e-9)), 4))


def stage_seedavg(a) -> None:
    """Seed-averaged comparison per (rung, cell): mean over seeds of the per-token loss for the dense and for the looped
    models, then the per-token difference; the null is (i) pairwise differences between single dense seeds and (ii) each
    dense seed minus the mean of the other dense seeds. Written to *_seedavg.csv."""
    import pandas as pd
    metas = {run: _meta(run) for run in a.runs}
    cls_table = classify_tokens(a.tokenizer, next(iter(metas.values()))["vocab"]) if os.path.exists(a.tokenizer) else None
    groups: dict[tuple, list] = {}
    for run, m in metas.items():
        groups.setdefault((m["rung"], m["cell"]), []).append(run)
    delta_rows, diff_rows, class_rows = [], [], []

    def add(kind, rung, cell, vs, d, l, nd, nl):
        n = min(len(d), len(l)); d, l = d[:n], l[:n]; delta = d - l
        delta_rows.append(dict(kind=kind, rung=rung, cell=cell, valset=vs, n_seeds_dense=nd, n_seeds_looped=nl, n_tokens=n, **_stats(delta)))
        avg = 0.5 * (d + l); edges = np.percentile(avg, np.linspace(0, 100, 11))
        dec = np.clip(np.searchsorted(edges, avg, side="right") - 1, 0, 9)
        gain = np.clip(delta, 0, None)
        for i in range(10):
            sel = dec == i
            diff_rows.append(dict(kind=kind, rung=rung, cell=cell, valset=vs, avg_loss_decile=i + 1, avg_loss=round(float(avg[sel].mean()), 4),
                                  mean_delta=round(float(delta[sel].mean()), 5), share_of_positive_gain=round(float(gain[sel].sum() / max(gain.sum(), 1e-9)), 4)))
        tgt_path = os.path.join(a.out, "raw", f"targets__{vs}.npy")
        if cls_table is not None and os.path.exists(tgt_path):
            c = cls_table[np.load(tgt_path)[:n].astype(np.int64)]
            for ci, cname in enumerate(CLASSES):
                sel = c == ci
                if sel.sum():
                    class_rows.append(dict(kind=kind, rung=rung, cell=cell, valset=vs, token_class=cname, share_of_tokens=round(float(sel.mean()), 4),
                                           mean_delta=round(float(delta[sel].mean()), 5), share_of_positive_gain=round(float(gain[sel].sum() / max(gain.sum(), 1e-9)), 4)))

    for vs in a.valsets.split(","):
        for (rung, cell), runs in groups.items():
            if cell == "dense":
                continue
            dense_runs = groups.get((rung, "dense"), [])
            D = [_load_raw(a.out, r, vs, metas[r]["r"], a.ckpt) for r in dense_runs]
            L = [_load_raw(a.out, r, vs, metas[r]["r"], a.ckpt) for r in runs]
            D = [x for x in D if x is not None]; L = [x for x in L if x is not None]
            if not D or not L:
                continue
            n = min(min(len(x) for x in D), min(len(x) for x in L))
            Dm = np.mean([x[:n] for x in D], axis=0); Lm = np.mean([x[:n] for x in L], axis=0)
            add("looped_vs_dense", rung, cell, vs, Dm, Lm, len(D), len(L))
            if len(D) >= 2 and cell == sorted(c for (rg, c) in groups if rg == rung and c != "dense")[0]:   # null once per rung
                for i in range(len(D)):
                    for j in range(i + 1, len(D)):
                        add("null_pairwise", rung, "dense", vs, D[i][:n], D[j][:n], 1, 1)
                    others = np.mean([D[k][:n] for k in range(len(D)) if k != i], axis=0)
                    add("null_loo", rung, "dense", vs, D[i][:n], others, 1, len(D) - 1)
    for name, rows in (("delta_stats_seedavg", delta_rows), ("by_difficulty_seedavg", diff_rows), ("by_class_seedavg", class_rows)):
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(a.out, name + ".csv"), index=False)
            print(f"{name}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["raw", "summary", "all", "seedavg"])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--valsets", default="fwe,second,finemath,code")
    ap.add_argument("--tokens", type=int, default=10_000_000, help="tokens per validation set")
    ap.add_argument("--r-eval", default="1,2,4,8,16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tokenizer", default="data/tokenizer/bpe16k.json")
    ap.add_argument("--out", default="results/token_analysis")
    ap.add_argument("--ckpt", default="branch40", choices=["branch40", "trunk_latest"])
    ap.add_argument("--budget-mult", type=float, default=40.0, help="with --ckpt branch40: budget (in N) whose branch checkpoint is used")
    a = ap.parse_args()
    if a.stage in ("raw", "all"):
        stage_raw(a)
    if a.stage in ("summary", "all"):
        stage_summary(a)
    if a.stage == "seedavg":
        stage_seedavg(a)


if __name__ == "__main__":
    main()
