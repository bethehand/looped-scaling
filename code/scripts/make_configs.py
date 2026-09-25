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
EVAL_TOKENS_PER_BRANCH = 250e6   # end point on all sets (110M) + 2 tail points on fwe+second (2 x 70M)
EVAL_SPEEDUP = 2.5               # forward-only evaluation vs training tokens/s (estimate)
COMPILE = False                  # set by --compile: per-block torch.compile in every run
LOOP_LR_FACTORS = (0.5, 0.71, 1.0)   # per-cell check for looped cells at 20M, relative to the dense lr0
COMPILE_SPEEDUP = 1.35           # median measured speed-up of compiled over eager (throughput_measured.json)
FIRST_RUNG = "10M"               # main grid: run this rung first (every cell type exercised within ~1 day), then longest-first
THROUGHPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "configs", "throughput_measured.json")

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


def micro_seqs_for(width: int, mcfg: ModelConfig) -> int:
    """Micro-batch (sequences) per forward pass; a pure memory knob, the optimizer batch is unchanged.
    Cells without per-loop checkpointing and more than 16 executed layers peaked at 20.1-21.5 GB on a 24 GB
    RTX 4090 at the default micro-batch (throughput test 2026-09-24); halve it there to keep headroom."""
    micro = MICRO_SEQS[width]
    if not mcfg.ckpt_loops and mcfg.executed_layers > 16 and width >= 640:
        micro //= 2
    return micro


def run_name(rung: str, placement: str, r: int, k: int, seed: int, tag: str = "") -> str:
    bp = "full" if k == 0 else f"k{k}"
    core = f"{rung}_{placement}_r{r}" + ("" if placement == "dense" else f"_{bp}")
    return f"{core}{tag}_s{seed}"


def _measured():
    return json.load(open(THROUGHPUT_FILE)) if os.path.exists(THROUGHPUT_FILE) else {}


# Compile mode of the checkpointed cells, by the rule declared in 03_偏离记录.md on 2026-09-25: per cell, the mode with
# the higher median speed-up over eager across widths 320-896 (results/throughput_ckpt/, GS01, 2026-09-26). Every
# other cell uses per-block compile.
REGION_CELLS = {("whole", 4, 0), ("whole", 8, 0)}


def cell_compiled(mcfg: ModelConfig) -> bool:
    """With --compile every cell is compiled. The checkpointed cells ran eager for a while on numbers measured while
    GS01's CPUs were power-capped; re-measured on the healthy machine, compiling them is 1.39-1.45x faster (median over
    widths) with correct gradients, so they are compiled like the rest."""
    return COMPILE


def cell_compile_mode(placement: str, r: int, k: int) -> str:
    return "region" if (placement, r, k) in REGION_CELLS else "blocks"


def hours_estimate(width: int, placement: str, r: int, k: int, tokens: int, n_branches: int, fallback_card_days: float,
                   compiled: bool | None = None) -> float:
    """Wall-clock hours on one RTX 4090 from measured throughput (training + evaluation); FLOP-based fallback."""
    m = _measured().get(str(width), {})
    tr, ev = m.get(f"{placement}_r{r}_k{k}"), m.get(f"{placement}_r{r}_k0")
    if not tr or not ev:
        return fallback_card_days * 24
    h = (tokens / tr + n_branches * EVAL_TOKENS_PER_BRANCH / (EVAL_SPEEDUP * ev)) / 3600
    compiled = COMPILE if compiled is None else compiled
    return h / COMPILE_SPEEDUP if compiled else h


def makespan_days(hours: list[float], cards: int = 4) -> float:
    free = [0.0] * cards
    for h in hours:
        i = free.index(min(free)); free[i] += h
    return max(free) / 24


def tokens_processed(budgets: list[int], frac: float) -> int:
    return int(round(max(budgets) * (1 - frac) + sum(frac * b for b in budgets)))


def make_run(rung: str, width: int, placement: str, r: int, k: int, seed: int, N_rung: int,
             lr_table: dict, budgets_mult: list[int], iso_flop: bool, tag: str = "",
             lr_override: float | None = None, betas=(0.9, 0.95), cfg_dir: str = "configs/runs") -> tuple[dict, dict]:
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
                   batch_tokens=bt, micro_seqs=micro_seqs_for(width, mcfg), warmup_tokens=int(2 * N_rung),
                   budgets=budgets, cooldown_frac=COOLDOWN_FRAC, eval_every_steps=200, eval_windows=64,
                   final_eval_points=3, final_eval_gap_steps=25, ckpt_every_seconds=1800, log_every_steps=10,
                   dtype="bf16", compile=cell_compiled(mcfg), compile_mode=cell_compile_mode(placement, r, k),
                   peak_flops=165e12, eval_batch_seqs=32),
        data=dict(train_shards="data/fwe_train/*.bin",
                  val_sets=dict(fwe="data/val/fwe_val.bin", second="data/val/second_val.bin",
                                finemath="data/val/finemath_val.bin", code="data/val/code_val.bin"),
                  val_main="fwe", val_max_tokens=50_000_000),
    )
    toks = tokens_processed(budgets, COOLDOWN_FRAC)
    flops = fl["train"] * toks
    card_days = flops / (CARD_FLOPS_PER_S * 86400)
    row = dict(name=name, rung=rung, width=width, placement=placement, r=r, k_bwd=k, seed=seed,
               N_rung=N_rung, budgets=" ".join(str(b) for b in budgets), iso_flop_rho=round(rho, 4),
               tokens_processed=toks, train_flops=f"{flops:.3e}",
               card_days=round(card_days, 2),
               hours_est=round(hours_estimate(width, placement, r, k, toks, len(budgets), card_days, cell_compiled(mcfg)), 2),
               status="pending", priority=0, config=f"{cfg_dir}/{name}.yaml", tag=tag)
    return cfg, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs")
    ap.add_argument("--lr-table", default=None, help="json {width: lr0} from the Week-3 sweep")
    ap.add_argument("--sweep", action="store_true", help="generate the learning-rate sweep instead of the grid")
    ap.add_argument("--compile", action="store_true", help="enable per-block torch.compile in the generated runs")
    ap.add_argument("--loop-lr-check", action="store_true",
                    help="generate the per-cell learning-rate check for looped cells at 20M instead of the grid")
    ap.add_argument("--loop-lr-mult", default=None,
                    help="json from scripts/pick_loop_lr.py; multiplies lr0 of looped cells in the main grid")
    args = ap.parse_args()
    global COMPILE
    COMPILE = args.compile
    lr_table = json.load(open(args.lr_table)) if args.lr_table else {}
    beta2_table = {}
    if "lr0" in lr_table:                       # format written by scripts/pick_lr.py
        beta2_table, lr_table = lr_table.get("beta2", {}), lr_table["lr0"]
    loop_mult = json.load(open(args.loop_lr_mult)) if args.loop_lr_mult else {"mode": "none"}

    def mult_for(placement: str, r: int) -> float:
        if placement == "dense" or loop_mult.get("mode", "none") == "none":
            return 1.0
        if loop_mult["mode"] == "shared":
            return float(loop_mult["shared"])
        return float(loop_mult["per_r"][str(r)])
    if args.loop_lr_check:
        width, rung = 448, "20M"
        runs_dir = os.path.join(args.out, "looplr")
        os.makedirs(runs_dir, exist_ok=True)
        N_rung = analytic_N(model_cfg(width, "dense", 1, 0))["N"]
        base = float(lr_table[str(width)])
        rows = []
        for placement, r, k in CELLS:
            if placement == "dense":
                continue
            for f in LOOP_LR_FACTORS:
                cfg, row = make_run(rung, width, placement, r, k, 42, N_rung, {}, [10], False,
                                    tag=f"_lrx{f}", lr_override=base * f, cfg_dir="configs/looplr",
                                    betas=(0.9, float(beta2_table.get(str(width), 0.95))))
                row["lr_factor"] = f
                rows.append(row)
                yaml.safe_dump(cfg, open(os.path.join(runs_dir, cfg["name"] + ".yaml"), "w"), sort_keys=False)
        rows.sort(key=lambda x: -x["hours_est"])          # longest-first packing
        for i, r in enumerate(rows):
            r["priority"] = i
        man = os.path.join(args.out, "manifest_looplr.csv")
        with open(man, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        hrs = [r["hours_est"] for r in rows]
        print(f"wrote {len(rows)} configs to {runs_dir}; manifest {man}")
        print(f"measured-throughput estimate: {sum(hrs) / 24:.1f} card-days, about {makespan_days(hrs):.2f} days on 4 cards")
        return
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
                                    tag=f"_lr{lr:.2e}", lr_override=lr, cfg_dir="configs/sweep")
                rows.append(row); yaml.safe_dump(cfg, open(os.path.join(runs_dir, cfg["name"] + ".yaml"), "w"))
            if width == 448:  # beta2 check at 20M
                for lr in grid:
                    cfg, row = make_run(rung, width, "dense", 1, 0, 42, N_rung, {}, [10], False,
                                        tag=f"_lr{lr:.2e}_b2_099", lr_override=lr, betas=(0.9, 0.99), cfg_dir="configs/sweep")
                    rows.append(row); yaml.safe_dump(cfg, open(os.path.join(runs_dir, cfg["name"] + ".yaml"), "w"))
            if width == 896:  # +-2x cross-check on a looped cell at 80M
                for f in (0.5, 1.0, 2.0):
                    cfg, row = make_run(rung, width, "middle", 4, 0, 42, N_rung, {}, [10], False,
                                        tag=f"_x{f}", lr_override=center * f, cfg_dir="configs/sweep")
                    rows.append(row); yaml.safe_dump(cfg, open(os.path.join(runs_dir, cfg["name"] + ".yaml"), "w"))
            continue
        cells = CELLS if rung in RUNGS else [("dense", 1, 0)]
        for seed in SEEDS[rung]:
            for placement, r, k in cells:
                budgets = (RULER_BUDGETS_160M if rung == "160M" else RULER_BUDGETS) if placement == "dense" else LOOP_BUDGETS
                lr_cell = float(lr_table.get(str(width), DEFAULT_LR[width])) * mult_for(placement, r)
                cfg, row = make_run(rung, width, placement, r, k, seed, N_rung, lr_table, budgets, iso_flop=True,
                                    betas=(0.9, float(beta2_table.get(str(width), 0.95))), lr_override=lr_cell)
                rows.append(row)
                yaml.safe_dump(cfg, open(os.path.join(runs_dir, cfg["name"] + ".yaml"), "w"), sort_keys=False)
    order = {r: i for i, r in enumerate(rung_order)}
    if args.sweep:   # sweep: 10M-40M first (their lr curves can be checked after ~2 h), then longest-first packing
        early = {"10M", "20M", "40M"}
        rows.sort(key=lambda x: (0, order[x["rung"]], -x["hours_est"]) if x["rung"] in early
                  else (1, 0, -x["hours_est"]))
    else:            # main grid: the cheap FIRST_RUNG exercises every cell type first, then longest-first packing
        rows.sort(key=lambda x: (0 if x["rung"] == FIRST_RUNG else 1, -x["hours_est"]))
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
    hrs = [r["hours_est"] for r in rows]
    print(f"measured-throughput estimate: {sum(hrs) / 24:.1f} card-days, about {makespan_days(hrs):.1f} days on 4 cards in queue order")


if __name__ == "__main__":
    main()
