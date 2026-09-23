"""Pre-run check 4: measured tokens/s, MFU and peak memory per width and representative cells.

    python scripts/measure_throughput.py --device cuda --steps 20 [--widths 320,448,640,896,1280]
Writes configs/throughput.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from looped.flops import flops_per_token  # noqa: E402
from looped.model import LoopedLM, ModelConfig  # noqa: E402
from scripts.make_configs import BATCH_TOKENS, MICRO_SEQS, model_cfg  # noqa: E402

CELLS = [("dense", 1, 0), ("middle", 4, 0), ("middle", 8, 0), ("middle", 8, 4), ("whole", 4, 0), ("whole", 8, 0)]


def bench(cfg: ModelConfig, micro_seqs: int, device, steps: int, dtype=torch.bfloat16) -> dict:
    model = LoopedLM(cfg).to(device)
    opt = torch.optim.AdamW(model.param_groups(1e-3, 0.1), lr=1e-3, fused=(device.type == "cuda"))
    x = torch.randint(0, cfg.vocab_size, (micro_seqs, cfg.seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (micro_seqs, cfg.seq_len), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for i in range(steps + 3):
        if i == 3:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.time()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            _, loss = model(x, y)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.time() - t0
    toks = micro_seqs * cfg.seq_len * steps
    fl = flops_per_token(cfg)["train"]
    return dict(tok_per_s=toks / dt, tflops=toks / dt * fl / 1e12, mfu_165=toks / dt * fl / 165e12,
                peak_mem_gb=(torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--widths", default="320,448,640,896,1280")
    args = ap.parse_args()
    device = torch.device(args.device)
    out = {}
    for w in [int(x) for x in args.widths.split(",")]:
        for placement, r, k in CELLS:
            if w == 1280 and placement != "dense":
                continue
            cfg = model_cfg(w, placement, r, k)
            key = f"w{w}_{placement}_r{r}_k{k}"
            try:
                res = bench(cfg, MICRO_SEQS[w], device, args.steps)
            except RuntimeError as e:  # OOM etc.
                res = dict(error=str(e)[:120])
            out[key] = res
            print(key, {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in res.items()}, flush=True)
    os.makedirs("configs", exist_ok=True)
    json.dump(out, open("configs/throughput.json", "w"), indent=1)


if __name__ == "__main__":
    main()
