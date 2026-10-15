"""Configs and manifests of the supplementary experiments S4–S9 (docs/supplementary-experiments-2.md, frozen 2026-10-15).

    python scripts/make_supp2.py --part s4     # S4: configs/supp2/*.yaml + configs/manifest_supp2_s4.csv
    python -u scripts/run_queue.py --manifest configs/manifest_supp2_s4.csv --gpus 0,1,2,3 2>&1 | tee -a logs/queue_supp2_s4.log

S4  text, dense models at high tokens per parameter: the finished dense trunks continued with the training code and
    rates as they are (10M seeds 42/43 from the 128N trunk of the d160 runs to 320N and 640N; 20M seeds 42/43 from the
    64N trunk of the main-grid runs to 160N and 320N), plus one looped run, 10M middle r=4 at 0.35x the dense rate,
    seed 43 (the S1 recipe; S1 has seed 42 only).
S7  text, seeds that change the data order: 20M dense and 20M middle r=4 (full backprop), seed 42, budgets 10/20/40
    tokens per parameter, the training stream read from 3B and 6B tokens in (train.data_offset; the main grid read
    tokens 0-2.1B at 20M, so the three orders share no data).
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts import make_configs as mc  # noqa: E402
from scripts import make_supp as ms  # noqa: E402

SUPP2_DIR = "configs/supp2"
MANIFEST_S4 = "configs/manifest_supp2_s4.csv"
MANIFEST_S7 = "configs/manifest_supp2_s7.csv"
STREAM_TOKENS = 10_410_000_000          # the training stream (FineWeb-Edu sample-10BT files 000-012; deviation log 2026-09-24)

# rung: (source run tag, budget of the source run in N, new budgets in N); the trunk continues from the source run's
# last branch checkpoint, i.e. its trunk at (1 - cooldown_frac) x that budget
S4_DENSE = {"10M": ("_d160", 160, [320, 640]), "20M": ("", 80, [160, 320])}
S4_SEEDS = (42, 43)
S4_LOOPED = ("10M", "middle", 4, 0, 43, 0.35)       # rung, placement, r, k, seed, learning-rate factor (S1 recipe)
S7_OFFSETS = {"3b": 3_000_000_000, "6b": 6_000_000_000}   # tokens into the stream, rounded down to whole batches
S7_CELLS = (("dense", 1, 0), ("middle", 4, 0))
S7_SEED, S7_RUNG, S7_BUDGETS = 42, "20M", [10, 20, 40]


def s4_rows(cfg_dir: str = SUPP2_DIR) -> list[dict]:
    """Write the S4 configs; return the manifest rows (dense continuations first, longest first, then the looped run)."""
    os.makedirs(cfg_dir, exist_ok=True)
    lr0, beta2 = ms._lr_tables()
    rows = []
    for rung, (src_tag, from_mult, mults) in S4_DENSE.items():
        width = mc.RUNGS[rung]
        N_rung = ms.n_rung(rung)
        bt = mc.BATCH_TOKENS[width]
        start = mc.branch_points(mc._budgets(N_rung, [from_mult], bt), mc.COOLDOWN_FRAC, bt)[0][0]
        for seed in S4_SEEDS:
            cfg, row = mc.make_run(rung, width, "dense", 1, 0, seed, N_rung, lr0, mults, iso_flop=False,
                                   tag=f"_d{max(mults)}", lr_override=float(lr0[str(width)]),
                                   betas=(0.9, float(beta2.get(str(width), 0.95))), cfg_dir=cfg_dir)
            src = f"runs/{mc.run_name(rung, 'dense', 1, 0, seed, src_tag)}/branch_{start}.pt"
            cfg["train"]["init_from"] = src
            assert max(cfg["train"]["budgets"]) <= STREAM_TOKENS, (rung, "the stream has too few tokens")
            toks = row["tokens_processed"] - start          # the inherited trunk is not trained again
            fl = mc.flops_per_token(mc.model_cfg(width, "dense", 1, 0))["train"]
            card_days = fl * toks / (mc.CARD_FLOPS_PER_S * 86400)
            row.update(tokens_processed=toks, train_flops=f"{fl * toks:.3e}", card_days=round(card_days, 2),
                       hours_est=round(mc.hours_estimate(width, "dense", 1, 0, toks, len(cfg["train"]["budgets"]),
                                                         card_days, cfg["train"]["compile"]), 2),
                       init_from=src, lr_factor=1.0)
            yaml.safe_dump(cfg, open(os.path.join(cfg_dir, cfg["name"] + ".yaml"), "w"), sort_keys=False)
            rows.append(row)
    rung, placement, r, k, seed, f = S4_LOOPED
    row = ms.s1_run(rung, placement, r, k, seed, f, cfg_dir=cfg_dir)
    row.update(init_from="")
    rows.append(row)
    for i, row in enumerate(rows):
        row["priority"] = i
    return rows


def s7_rows(cfg_dir: str = SUPP2_DIR) -> list[dict]:
    """Write the S7 configs (the main-grid recipe with train.data_offset); return the manifest rows."""
    os.makedirs(cfg_dir, exist_ok=True)
    lr0, beta2 = ms._lr_tables()
    width, N_rung, bt = mc.RUNGS[S7_RUNG], ms.n_rung(S7_RUNG), mc.BATCH_TOKENS[mc.RUNGS[S7_RUNG]]
    rows = []
    for tag, offset in S7_OFFSETS.items():
        for placement, r, k in S7_CELLS:
            cfg, row = mc.make_run(S7_RUNG, width, placement, r, k, S7_SEED, N_rung, lr0, S7_BUDGETS, iso_flop=False,
                                   tag=f"_do{tag}", lr_override=float(lr0[str(width)]),
                                   betas=(0.9, float(beta2.get(str(width), 0.95))), cfg_dir=cfg_dir)
            cfg["train"]["data_offset"] = offset - offset % bt
            assert cfg["train"]["data_offset"] + max(cfg["train"]["budgets"]) + bt <= STREAM_TOKENS
            yaml.safe_dump(cfg, open(os.path.join(cfg_dir, cfg["name"] + ".yaml"), "w"), sort_keys=False)
            row.update(init_from="", lr_factor=1.0, data_offset=cfg["train"]["data_offset"])
            rows.append(row)
    rows.sort(key=lambda r: -r["hours_est"])
    for i, row in enumerate(rows):
        row["priority"] = i
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="s4", choices=["s4", "s7"])
    a = ap.parse_args()
    rows, manifest = (s4_rows(), MANIFEST_S4) if a.part == "s4" else (s7_rows(), MANIFEST_S7)
    ms._write_csv(manifest, rows)
    for row in rows:
        print(f"  {row['name']:36s} budgets {row['budgets']:28s} {row['hours_est']:6.1f} h"
              + (f"  from {row['init_from']}" if row["init_from"] else "")
              + (f"  data_offset {row['data_offset']}" if row.get("data_offset") else ""))
    print(f"{len(rows)} rows -> {manifest}; configs in {SUPP2_DIR}/")


if __name__ == "__main__":
    main()
