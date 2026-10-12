"""Inference cost of the study's architectures on one GPU (supplementary S3, docs/supplementary-experiments.md):
weight memory, KV cache, prefill latency and per-token decode latency, eager and with the decode step in a CUDA graph.

    python scripts/bench_inference.py --out results/bench/inference.json
Random weights in bf16 (latency and memory do not depend on the values). For every width and cell: prefill of a
--prompt-token prompt at batch 1, then --gen greedy decode steps at each batch size, from a KV cache allocated once at
prompt + gen positions. Decode times are the mean over the --gen steps after a warm-up; the CUDA-graph path replays one
captured decode step per token, and its first tokens are checked against the eager path.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from looped.infer import StaticCache, decode_step, prefill  # noqa: E402
from looped.model import LoopedLM  # noqa: E402
from scripts.make_configs import model_cfg  # noqa: E402

WIDTHS = (320, 448, 640, 896, 1280)
CELLS = (("dense", 1), ("middle", 2), ("middle", 4), ("middle", 8), ("whole", 4), ("whole", 8))


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _ms(fn, n: int, dev) -> float:
    _sync(dev)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(dev)
    return (time.perf_counter() - t) * 1000 / n


def bench_cell(width: int, placement: str, r: int, prompt: int, gen: int, batches: list[int], dev, graphs: bool) -> dict:
    cfg = model_cfg(width, placement, r, 0)
    assert prompt + gen <= cfg.seq_len
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    torch.manual_seed(0)
    model = LoopedLM(cfg).to(dev, dtype=dtype).eval()
    pc = model.param_counts()
    elt = torch.finfo(dtype).bits // 8
    out = dict(width=width, placement=placement, r=r, executed_layers=cfg.executed_layers, N=pc["N"],
               params_total=pc["total_with_embedding"], weight_mb=round(pc["total_with_embedding"] * elt / 2 ** 20, 2),
               kv_kb_per_token=round(2 * cfg.executed_layers * cfg.d_model * elt / 1024, 2))
    idx = torch.randint(0, cfg.vocab_size, (1, prompt), device=dev)
    cache = StaticCache.for_model(model, 1, prompt + gen)
    for _ in range(2):
        prefill(model, idx, cache)
    out["prefill_ms_b1"] = round(statistics.median(_ms(lambda: prefill(model, idx, cache), 1, dev) for _ in range(7)), 3)
    for B in batches:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        cache = StaticCache.for_model(model, B, prompt + gen)
        idx = torch.randint(0, cfg.vocab_size, (B, prompt), device=dev)
        first = prefill(model, idx, cache)[:, -1:].argmax(-1)
        tok = first.clone()

        def step():
            tok.copy_(decode_step(model, tok, cache).argmax(-1))

        def run(fn, n):      # from the end of the prompt: n tokens, returns them
            cache.reset(prompt); tok.copy_(first)
            seq = []
            for _ in range(n):
                fn(); seq.append(tok.clone())
            return torch.cat(seq, 1)

        n_check = min(16, gen)
        run(step, n_check)                                    # warm-up
        cache.reset(prompt); tok.copy_(first)
        out[f"decode_ms_eager_b{B}"] = round(_ms(step, gen, dev), 4)
        ref = run(step, n_check)
        if graphs:
            s = torch.cuda.Stream(dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                run(step, 3)
            torch.cuda.current_stream(dev).wait_stream(s)
            g = torch.cuda.CUDAGraph()
            cache.reset(prompt); tok.copy_(first)
            with torch.cuda.graph(g):
                step()
            same = bool(torch.equal(run(g.replay, n_check), ref))
            run(g.replay, n_check)
            cache.reset(prompt); tok.copy_(first)
            out[f"decode_ms_graph_b{B}"] = round(_ms(g.replay, gen, dev), 4)
            out[f"graph_matches_eager_b{B}"] = same
            out[f"tokens_per_s_graph_b{B}"] = round(B * 1000 / out[f"decode_ms_graph_b{B}"], 1)
            del g
        if dev.type == "cuda":
            out[f"peak_mb_b{B}"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1)
        out[f"kv_cache_mb_b{B}"] = round(cache.nbytes / 2 ** 20, 2)
        del cache
    del model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench/inference.json")
    ap.add_argument("--widths", default=",".join(map(str, WIDTHS)))
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dev = torch.device(a.device)
    graphs = dev.type == "cuda"
    meta = dict(torch=torch.__version__, python=platform.python_version(), device=str(dev), prompt=a.prompt, gen=a.gen,
                started=time.strftime("%Y-%m-%d %H:%M:%S"), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if dev.type == "cuda":
        meta.update(gpu=torch.cuda.get_device_name(dev), cuda=torch.version.cuda)
        try:
            meta["nvidia_smi_at_start"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,clocks.sm,power.draw", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        except Exception as e:     # recorded, not fatal
            meta["nvidia_smi_at_start"] = repr(e)
    rows = []
    for width in [int(w) for w in a.widths.split(",")]:
        for placement, r in CELLS:
            row = bench_cell(width, placement, r, a.prompt, a.gen, [int(b) for b in a.batches.split(",")], dev, graphs)
            rows.append(row)
            print(json.dumps(row), flush=True)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(meta=meta, rows=rows), open(a.out, "w"), indent=1)
    print(f"{len(rows)} cells -> {a.out}")


if __name__ == "__main__":
    main()
