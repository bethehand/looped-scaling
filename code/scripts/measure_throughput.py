"""Pre-run check 4: measured tokens/s, MFU and peak memory per width and representative cells.

    python scripts/measure_throughput.py --device cuda --steps 20 [--widths 320,448,640,896,1280] [--cells middle_r4_k2,...]
Writes configs/throughput.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch._dynamo as dynamo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from looped.flops import flops_per_token  # noqa: E402
from looped.model import LoopedLM, ModelConfig  # noqa: E402
from scripts.make_configs import CELLS, micro_seqs_for, model_cfg  # noqa: E402


def bench(cfg: ModelConfig, micro_seqs: int, device, steps: int, dtype=torch.bfloat16, compile_: bool = False) -> dict:
    if compile_:
        dynamo.reset()               # fresh compile per configuration, so earlier ones cannot exhaust the cache
    model = LoopedLM(cfg).to(device)
    if compile_:
        model.compile_blocks()
    opt = torch.optim.AdamW(model.param_groups(1e-3, 0.1), lr=1e-3, fused=(device.type == "cuda"))
    x = torch.randint(0, cfg.vocab_size, (micro_seqs, cfg.seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (micro_seqs, cfg.seq_len), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    warm = 5 if compile_ else 3          # first steps include compilation
    for i in range(steps + warm):
        if i == warm:
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
    # one more step without zeroing: the looped block (core + adapter) must receive a finite, non-zero gradient
    grad_ok = None
    if len(model.core) > 0:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            _, loss = model(x, y)
        loss.backward()
        ps = list(model.core.parameters()) + (list(model.adapter.parameters()) if model.adapter is not None else [])
        grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in ps)
    return dict(tok_per_s=toks / dt, tflops=toks / dt * fl / 1e12, mfu_165=toks / dt * fl / 165e12,
                peak_mem_gb=(torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None),
                micro=micro_seqs, core_grad_ok=grad_ok, compiled=compile_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--widths", default="320,448,640,896,1280")
    ap.add_argument("--cells", default="", help="comma list like middle_r4_k2,whole_r8_k4 (default: all 11 cells)")
    ap.add_argument("--compile", action="store_true", help="compile each block (results stored under keys ending in _compiled)")
    args = ap.parse_args()
    device = torch.device(args.device)
    want = {c.strip() for c in args.cells.split(",") if c.strip()}
    path = "configs/throughput.json"
    out = json.load(open(path)) if os.path.exists(path) else {}
    for w in [int(x) for x in args.widths.split(",")]:
        for placement, r, k in CELLS:
            if w == 1280 and placement != "dense":
                continue
            if want and f"{placement}_r{r}_k{k}" not in want:
                continue
            cfg = model_cfg(w, placement, r, k)
            key = f"w{w}_{placement}_r{r}_k{k}" + ("_compiled" if args.compile else "")
            try:
                res = bench(cfg, micro_seqs_for(w, cfg), device, args.steps, compile_=args.compile)
            except RuntimeError as e:  # OOM etc.
                res = dict(error=str(e)[:120])
            if device.type == "cuda":
                torch.cuda.empty_cache()
            out[key] = res
            line = {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in res.items()}
            base = out.get(key.replace("_compiled", "")) if args.compile else None
            if base and "tok_per_s" in base and "tok_per_s" in res:
                line["speedup_vs_eager"] = round(res["tok_per_s"] / base["tok_per_s"], 2)
            print(key, line, flush=True)
    os.makedirs("configs", exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
