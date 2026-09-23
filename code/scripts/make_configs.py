"""Generate every run config + manifest from the frozen pre-registration (v1.0).

    python scripts/make_configs.py --out configs [--lr-table configs/lr_table.json] [--sweep]

Grid (pre-registration §3):
  rungs (loop cells)   : 10M / 20M / 40M / 80M  -> widths 320 / 448 / 640 / 896, 8 unique layers
  ruler                : dense r=1 at the four rungs + 160M (width 1280)
  cells per rung       : dense r1 | middle r2 | middle r4 full/trunc(k=2) | middle r8 full/trunc(k=4) | whole (same 5)
  budgets (tokens)     : ruler 5/10/20/40/80 x N ; looped 10/20/40 x N ; truncated: + iso-FLOP budgets x rho
  seeds                : 42/43/44 at 10M/20M/40M ; 42 at 80M ; 42 for the 160M ruler
  N                    : non-embedding params + output head of the rung's DENSE model (same D for all cells of a rung)
--sweep generates the Week-3 learning-rate sweep instead (dense r=1, 5 lr values, budget 10N, one cooldown).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from looped.flops import flops_per_token  # noqa: E402
from looped.model import ModelConfig  # noqa: E402

RUNGS = {"10M": 320, "20M": 448, "40M": 640, "80M": 896}
RULER_ONLY = {"160M": 1280}
BATCH_TOKENS = {320: 81920, 448: 131072, 640: 204800, 896: 327680, 1280: 524288}
MICRO_SEQS = {320: 16, 448: 16, 640: 20, 896: 16, 1280: 16}
SEEDS = {"10M": [42, 43, 44], "20M": [42, 43, 44], "40M": [42, 43, 44], "80M": [42], "160M": [42]}
DEFAULT_LR = {320: 3.0e-3, 448: 3.0e-3, 640: 2.0e-3, 896: 1.5e-3, 1280: 1.0e-3}   # placeholders until the sweep
RULER_BUDGETS = [5, 10, 20, 40, 80]
RULER_BUDGETS_160M = [5, 10, 20, 40]   # deviation 2026-09-24: the FineWeb-Edu 10BT stream (~11B tokens) cannot reach 80N at 160M
LOOP_BUDGETS = [10, 20, 40]
COOLDOWN_FRAC = 0.2
CARD_FLOPS_PER_S = 5e13
VOCAB, SEQ = 16384, 1024

CELLS = [  # (placement, r, k_bwd)
    ("dense", 1, 0),
    ("middle", 2, 0), ("middle", 4, 0), ("middle", 4, 2), ("middle", 8, 0), ("middle", 8, 4),
    ("whole", 2, 0), ("whole", 4, 0), ("whole", 4, 2), ("whole", 8, 0), ("whole", 8, 4),
]


def analytic_N(cfg: ModelConfig) -> dict:
    """Parameter counts without instantiating (mirrors LoopedLM.param_counts)."""
    d, dff, V = cfg.d_model, cfg.d_ff, cfg.vocab_size
    block = 4 * d * d + 3 * d * dff + 2 * d
    p, c, q = cfg.layer_split
    adapter = 2 * d * d if (c > 0 and cfg.injection == "concat") else 0
    N_once = (p + q) * block + V * d + d
    N_rec = c * block + adapter
    return dict(N_once=N_once, N_rec=N_rec, N=N_once + N_rec, emb_in=V * d)


def model_cfg(width: int, placement: str, r: int, k: int) -> ModelConfig:
    return ModelConfig(vocab_size=VOCAB, d_model=width, n_layers=8, head_dim=64, seq_len=SEQ,
                       placement=placement, n_prelude=2, n_coda=2, r=r, k_bwd=k,
                       ckpt_loops=(placement != "dense" and (2 + r * 4 + 2 if placement == "middle" else 8 * r) > 24))


def run_name(rung: str, placement: str, r: int, k: int, seed: int, tag: str = "") -> str:
    bp = "full" if k == 0 else f"k{k}"
    core = f"{rung}_{placement}_r{r}" + ("" if placement == "dense" else f"_{bp}")
    return f"{core}{tag}_s{seed}"


def tokens_processed(budgets: list[int], frac: float) -> int:
    return int(round(max(budgets) * (1 - frac) + sum(frac * b for b in budgets)))


def make_run(rung: str, width: int, placement: str, r: int, k: int, seed: int, N_rung: int,
             lr_table: dict, budgets_mult: list[int], iso_flop: bool, tag: str = "",
             lr_override: float | None = None, betas=(0.9, 0.95)) -> tuple[dict, dict]:
    mcfg = model_cfg(width, placement, r, k)
    fl = flops_per_token(mcfg)
    budgets = [int(m * N_rung) for m in budgets_mult]
    rho = 1.0
    if iso_flop and k > 0:
        full = flops_per_token(model_cfg(width, placement, r, 0))["train"]
        rho = full / fl["train"]
        budgets = sorted(set(budgets + [int(round(m * N_rung * rho)) for m in budgets_mult]))
    bt = BATCH_TOKENS[width]
    budgets = [b - b % bt for b in budgets]
    lr0 = lr_override if lr_override is not None else lr_table.get(str(width), DEFAULT_LR[width])
    name = run_name(rung, placement, r, k, seed, tag)
    cfg = dict(
        name=name, out_dir=f"runs/{name}",
        model=mcfg.to_dict(),
        train=dict(seed=seed, lr0=float(lr0), weight_decay=0.1, betas=list(betas), grad_clip=1.0,
                   batch_tokens=bt, micro_seqs=MICRO_SEQS[width], warmup_tokens=int(2 * N_rung),
                   budgets=budgets, cooldown_frac=COOLDOWN_FRAC, eval_every_steps=200, eval_windows=64,
                   final_eval_points=3, final_eval_gap_steps=25, ckpt_every_seconds=1800, log_every_steps=10,
                   dtype="bf16", compile=False, peak_flops=165e12, eval_batch_seqs=32),
        data=dict(train_shards="data/fwe_train/*.bin",
                  val_sets=dict(fwe="data/val/fwe_val.bin", second="data/val/second_val.bin",
                                finemath="data/val/finemath_val.bin", code="data/val/code_val.bin"),
                  val_main="fwe", val_max_tokens=50_000_000),
    )
    toks = tokens_processed(budgets, COOLDOWN_FRAC)
    flops = fl["train"] * toks
    row = dict(name=name, rung=rung, width=width, placement=placement, r=r, k_bwd=k, seed=seed,
               N_rung=N_rung, budgets=" ".join(str(b) for b in budgets), iso_flop_rho=round(rho, 4),
               tokens_processed=toks, train_flops=f"{flops:.3e}",
               card_days=round(flops / (CARD_FLOPS_PER_S * 86400), 2), status="pending", priority=0,
               config=f"configs/runs/{name}.yaml", tag=tag)
    return cfg, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs")
    ap.add_argument("--lr-table", default=None, help="json {width: lr0} from the Week-3 sweep")
    ap.add_argument("--sweep", action="store_true", help="generate the learning-rate sweep instead of the grid")
    args = ap.parse_args()
    lr_table = json.load(open(args.lr_table)) if args.lr_table else {}
    runs_dir = os.path.join(args.out, "runs" if not args.sweep else "sweep")
    os.makedirs(runs_dir, exist_ok=True)
    rows = []
    rung_order = list(RUNGS) + list(RULER_ONLY)
    for rung in rung_order:
        width = {**RUNGS, **RULER_ONLY}[rung]
        N_rung = analytic_N(model_cfg(width, "dense", 1, 0))["N"]
        if args.sweep:
            center = lr_table.get(str(width), DEFAULT_LR[width])
            grid = [center * f for f in (0.25, 0.5, 1.0, 2.0, 4.0)]
            for lr in grid:
                cfg, row = make_run(rung, width, "dense", 1, 0, 42, N_rung, {}, [10], False,
                                    tag=f"_lr{lr:.2e}", lr_override=lr)
                rows.append(row); yaml.safe_dump(cfg, open(os.path.join(runs_dir, cfg["name"] + ".yaml"), "w"))
            if width == 448:  # beta2 check at 20M
                for lr in grid:
                    cfg, row = make_run(rung, width, "dense", 1, 0, 42, N_rung, {}, [10], False,
                                        tag=f"_lr{lr:.2e}_b2_099", lr_override=lr, betas=(0.9, 0.99))
                    rows.append(row); yaml.safe_dump(cfg, open(os.path.join(runs_dir, cfg["name"] + ".yaml"), "w"))
            if width == 896:  # +-2x cross-check on a looped cell at 80M
                for f in (0.5, 1.0, 2.0):
                    cfg, row = make_run(rung, width, "middle", 4, 0, 42, N_rung, {}, [10], False,
                                        tag=f"_x{f}", lr_override=center * f)
                    rows.append(row); yaml.safe_dump(cfg, open(os.path.join(runs_dir, cfg["name"] + ".yaml"), "w"))
            continue
        cells = CELLS if rung in RUNGS else [("dense", 1, 0)]
        for seed in SEEDS[rung]:
            for placement, r, k in cells:
                budgets = (RULER_BUDGETS_160M if rung == "160M" else RULER_BUDGETS) if placement == "dense" else LOOP_BUDGETS
                cfg, row = make_run(rung, width, placement, r, k, seed, N_rung, lr_table, budgets, iso_flop=True)
                rows.append(row)
                yaml.safe_dump(cfg, open(os.path.join(runs_dir, cfg["name"] + ".yaml"), "w"), sort_keys=False)
    # priority: rung order (10M first), then most expensive first within a rung so long jobs start early
    order = {r: i for i, r in enumerate(rung_order)}
    rows.sort(key=lambda x: (order[x["rung"]], -x["card_days"]))
    for i, r in enumerate(rows):
        r["priority"] = i
    man = os.path.join(args.out, "manifest.csv" if not args.sweep else "manifest_sweep.csv")
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    total = sum(r["card_days"] for r in rows)
    by_rung = {}
    for r in rows:
        by_rung[r["rung"]] = by_rung.get(r["rung"], 0) + r["card_days"]
    print(f"wrote {len(rows)} configs to {runs_dir}; manifest {man}")
    print("card-days by rung:", {k: round(v, 1) for k, v in by_rung.items()}, "total", round(total, 1))


if __name__ == "__main__":
    main()
