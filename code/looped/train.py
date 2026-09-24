"""One run = WSD trunk + cooldown branches, sequential on one device.

Usage:  python -m looped.train --config configs/runs/<name>.yaml [--device cuda:0]

Phases
  trunk    : warmup -> constant lr until max(budgets) * (1 - cooldown_frac); at each branch point a
             full checkpoint (model, optimizer, data position) is saved as branch_<D>.pt
  branch D : resume branch_<D>.pt, linearly decay lr to 0 until D tokens, evaluate on all val sets,
             append one JSON line to results.jsonl
Resumable: trunk_latest.pt is written periodically; finished branches are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field

import torch
import yaml

from .data import BatchSampler, TokenStream, ValSet
from .evaluate import evaluate
from .flops import flops_per_token
from .model import LoopedLM, ModelConfig
from .schedule import branch_points, lr_at


def git_commit() -> str:
    """Short commit of the code that produced a run ("+dirty" if the working tree has uncommitted changes)."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=here, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                               cwd=here, timeout=10).stdout.strip()
        return (c or "unknown") + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class TrainConfig:
    seed: int = 42
    lr0: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    batch_tokens: int = 131072
    micro_seqs: int = 16
    warmup_tokens: int = 40_000_000
    budgets: list[int] = field(default_factory=list)   # total tokens at end of each cooldown
    cooldown_frac: float = 0.2
    eval_every_steps: int = 200        # quick validation (first eval_windows of the main set)
    eval_windows: int = 64
    final_eval_points: int = 3         # losses averaged over the last N eval points of a cooldown
    final_eval_gap_steps: int = 25
    ckpt_every_seconds: int = 1800
    log_every_steps: int = 10
    dtype: str = "bf16"                # bf16 | fp32
    compile: bool = False
    compile_mode: str = "blocks"       # blocks | region (see LoopedLM.compile_blocks)
    peak_flops: float = 165e12         # for MFU logging only (RTX 4090 bf16 dense)
    eval_batch_seqs: int = 32


@dataclass
class DataConfig:
    train_shards: str = "data/fwe_train/*.bin"
    val_sets: dict = field(default_factory=dict)   # name -> path
    val_main: str = "fwe"
    val_max_tokens: int | None = 50_000_000


def load_run_config(path: str) -> tuple[str, str, ModelConfig, TrainConfig, DataConfig]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    name = raw["name"]
    out_dir = raw.get("out_dir", os.path.join("runs", name))
    mcfg = ModelConfig(**raw["model"])
    t = dict(raw.get("train", {}))
    if "betas" in t:
        t["betas"] = tuple(t["betas"])
    tcfg = TrainConfig(**t)
    dcfg = DataConfig(**raw.get("data", {}))
    assert tcfg.batch_tokens % (tcfg.micro_seqs * mcfg.seq_len) == 0, "batch_tokens must be a multiple of micro_seqs*seq_len"
    return name, out_dir, mcfg, tcfg, dcfg


class Runner:
    def __init__(self, name: str, out_dir: str, mcfg: ModelConfig, tcfg: TrainConfig, dcfg: DataConfig,
                 device: str = "cuda"):
        self.name, self.out_dir, self.mcfg, self.tcfg, self.dcfg = name, out_dir, mcfg, tcfg, dcfg
        os.makedirs(out_dir, exist_ok=True)
        self.device = torch.device(device)
        self.dtype = {"bf16": torch.bfloat16, "fp32": None}[tcfg.dtype]
        torch.manual_seed(tcfg.seed)
        self.model = LoopedLM(mcfg).to(self.device)
        self.raw_model = self.model
        if tcfg.compile:
            self.raw_model.compile_blocks(mode=tcfg.compile_mode)
        self.opt = torch.optim.AdamW(
            self.raw_model.param_groups(tcfg.lr0, tcfg.weight_decay), lr=tcfg.lr0, betas=tcfg.betas,
            fused=(self.device.type == "cuda"))
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]
        self.stream = TokenStream.from_glob(dcfg.train_shards)
        self.batch_seqs = tcfg.batch_tokens // mcfg.seq_len
        self.sampler = BatchSampler(self.stream, mcfg.seq_len, self.batch_seqs)
        self.n_micro = self.batch_seqs // tcfg.micro_seqs
        if dcfg.val_sets and not os.path.exists(dcfg.val_sets[dcfg.val_main]):
            raise FileNotFoundError(f"main validation set missing: {dcfg.val_sets[dcfg.val_main]}")
        present = {k: v for k, v in dcfg.val_sets.items() if os.path.exists(v)}
        for k in sorted(set(dcfg.val_sets) - set(present)):
            print(f"[{name}] WARNING optional validation set '{k}' not found at {dcfg.val_sets[k]}; skipped", flush=True)
        self.val = {k: ValSet(v, mcfg.seq_len, dcfg.val_max_tokens) for k, v in present.items()}
        # sets that enter scaling-law fits get the extra tail evaluations; exploratory sets only the final one
        self.fit_keys = [k for k in (dcfg.val_main, "second") if k in self.val]
        self.quick_val = ValSet(dcfg.val_sets[dcfg.val_main], mcfg.seq_len,
                                max_tokens=tcfg.eval_windows * (mcfg.seq_len + 1)) if dcfg.val_sets else None
        self.flops = flops_per_token(mcfg)
        self.params = self.raw_model.param_counts()
        self.tokens_seen = 0
        self.step = 0
        self.log_f = open(os.path.join(out_dir, "train_log.csv"), "a")
        if os.path.getsize(os.path.join(out_dir, "train_log.csv")) == 0:
            self.log_f.write("phase,step,tokens,lr,loss,tok_per_s,mfu,quick_val\n")
        self.git_commit = git_commit()
        with open(os.path.join(out_dir, "run_info.json"), "w") as f:
            json.dump(dict(name=name, model=mcfg.to_dict(), train=vars(tcfg), data=vars(dcfg),
                           params=self.params, flops_per_token=self.flops, git_commit=self.git_commit,
                           torch_version=torch.__version__), f, indent=1)

    # ----- checkpointing -----
    def _ckpt_path(self, tag: str) -> str:
        return os.path.join(self.out_dir, f"{tag}.pt")

    def save(self, tag: str) -> None:
        tmp = self._ckpt_path(tag) + ".tmp"
        torch.save(dict(model=self.raw_model.state_dict(), opt=self.opt.state_dict(),
                        sampler=self.sampler.state_dict(), tokens_seen=self.tokens_seen, step=self.step,
                        rng=torch.get_rng_state()), tmp)
        os.replace(tmp, self._ckpt_path(tag))

    def load(self, tag: str) -> None:
        ck = torch.load(self._ckpt_path(tag), map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.sampler.load_state_dict(ck["sampler"])
        self.tokens_seen, self.step = ck["tokens_seen"], ck["step"]
        torch.set_rng_state(ck["rng"].cpu())

    # ----- one optimizer step -----
    def train_step(self, lr: float) -> float:
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = lr * base / self.tcfg.lr0
        x, y = self.sampler.next()
        self.model.train()
        total = 0.0
        for i in range(self.n_micro):
            xs = x[i * self.tcfg.micro_seqs:(i + 1) * self.tcfg.micro_seqs].to(self.device, non_blocking=True)
            ys = y[i * self.tcfg.micro_seqs:(i + 1) * self.tcfg.micro_seqs].to(self.device, non_blocking=True)
            ctx = (torch.autocast(device_type=self.device.type, dtype=self.dtype) if self.dtype is not None
                   else torch.autocast(device_type=self.device.type, enabled=False))
            with ctx:
                _, loss = self.model(xs, ys)
            (loss / self.n_micro).backward()
            total += float(loss.detach()) / self.n_micro
        if self.tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.tcfg.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.tokens_seen += self.tcfg.batch_tokens
        self.step += 1
        return total

    def _quick_eval(self) -> float:
        if self.quick_val is None:
            return float("nan")
        return evaluate(self.model, self.quick_val, self.tcfg.eval_batch_seqs, self.device, self.dtype)

    def _log(self, phase: str, lr: float, loss: float, t0: float, quick: float = float("nan")) -> None:
        dt = max(time.time() - t0, 1e-9)
        tps = self.tcfg.batch_tokens * self.tcfg.log_every_steps / dt
        mfu = tps * self.flops["train"] / self.tcfg.peak_flops
        self.log_f.write(f"{phase},{self.step},{self.tokens_seen},{lr:.3e},{loss:.5f},{tps:.0f},{mfu:.3f},{quick:.5f}\n")
        self.log_f.flush()
        print(f"[{self.name}] {phase} step {self.step} tok {self.tokens_seen/1e6:.1f}M lr {lr:.2e} loss {loss:.4f} "
              f"{tps/1e3:.1f}k tok/s mfu {mfu:.2f}" + (f" qval {quick:.4f}" if not math.isnan(quick) else ""), flush=True)

    # ----- phases -----
    def run(self) -> None:
        t = self.tcfg
        points = branch_points(t.budgets, t.cooldown_frac, t.batch_tokens)
        trunk_end = max(s for s, _ in points)
        results_path = os.path.join(self.out_dir, "results.jsonl")
        done = set()
        if os.path.exists(results_path):
            for line in open(results_path):
                done.add(int(json.loads(line)["budget_tokens"]))
        if all(e in done for _, e in points):
            print(f"[{self.name}] all branches done", flush=True)
            return
        # ---- trunk ----
        if not os.path.exists(os.path.join(self.out_dir, "TRUNK_DONE")):
            if os.path.exists(self._ckpt_path("trunk_latest")):
                self.load("trunk_latest")
                print(f"[{self.name}] resumed trunk at {self.tokens_seen} tokens", flush=True)
            self._train_until(trunk_end, phase="trunk", branch_saves={s for s, _ in points})
            self.save("trunk_latest")
            open(os.path.join(self.out_dir, "TRUNK_DONE"), "w").write(str(self.tokens_seen))
        # ---- branches ----
        for start, end in points:
            if end in done:
                continue
            self.load(f"branch_{start}")
            assert self.tokens_seen == start, (self.tokens_seen, start)
            losses = self._train_until(end, phase=f"cool_{end}", cooldown=(start, end))
            row = self._final_eval(end, losses)
            with open(results_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{self.name}] branch {end} done: {row['val']}", flush=True)

    def _train_until(self, end_tokens: int, phase: str, branch_saves: set[int] | None = None,
                     cooldown: tuple[int, int] | None = None) -> list[dict]:
        t = self.tcfg
        branch_saves = branch_saves or set()
        last_ckpt = time.time()
        t0 = time.time()
        acc = 0.0
        tail_evals: list[dict] = []
        while self.tokens_seen < end_tokens:
            if self.tokens_seen in branch_saves and not os.path.exists(self._ckpt_path(f"branch_{self.tokens_seen}")):
                self.save(f"branch_{self.tokens_seen}")
            cs, ce = cooldown if cooldown else (None, None)
            lr = lr_at(self.tokens_seen + t.batch_tokens, t.lr0, t.warmup_tokens, cs, ce)
            loss = self.train_step(lr)
            acc += loss
            if self.step % t.log_every_steps == 0:
                quick = self._quick_eval() if (self.step % t.eval_every_steps == 0) else float("nan")
                self._log(phase, lr, acc / t.log_every_steps, t0, quick)
                acc, t0 = 0.0, time.time()
            if cooldown is not None:
                remaining_steps = (end_tokens - self.tokens_seen) // t.batch_tokens
                # tail points before the end (e.g. 50 and 25 steps); the end point itself is evaluated in _final_eval
                if (0 < remaining_steps < t.final_eval_points * t.final_eval_gap_steps
                        and remaining_steps % t.final_eval_gap_steps == 0):
                    tail_evals.append(self._full_eval(self.fit_keys))
            if time.time() - last_ckpt > t.ckpt_every_seconds and cooldown is None:
                self.save("trunk_latest")
                last_ckpt = time.time()
        if self.tokens_seen in branch_saves and not os.path.exists(self._ckpt_path(f"branch_{self.tokens_seen}")):
            self.save(f"branch_{self.tokens_seen}")
        return tail_evals

    def _full_eval(self, keys: list[str] | None = None) -> dict:
        keys = list(self.val) if keys is None else keys
        return {k: evaluate(self.model, self.val[k], self.tcfg.eval_batch_seqs, self.device, self.dtype)
                for k in keys}

    def _final_eval(self, budget: int, tail: list[dict]) -> dict:
        """val     = loss at the end of the cooldown on every validation set (primary metric)
           val_avg = mean over the tail points and the end point (robustness metric; fit sets only get tail points)"""
        final = self._full_eval()
        n = self.tcfg.final_eval_points
        pts = (tail[-(n - 1):] if n > 1 else []) + [final]
        avg, n_pts = {}, {}
        for k in final:
            vals = [p[k] for p in pts if k in p]
            avg[k], n_pts[k] = sum(vals) / len(vals), len(vals)
        m = self.mcfg
        return dict(
            name=self.name, seed=self.tcfg.seed, placement=m.placement, r=m.r, k_bwd=m.k_bwd,
            d_model=m.d_model, n_layers=m.n_layers, n_prelude=m.layer_split[0], n_core=m.layer_split[1],
            n_coda=m.layer_split[2], executed_layers=m.executed_layers,
            N_once=self.params["N_once"], N_rec=self.params["N_rec"], N=self.params["N"],
            emb_in=self.params["emb_in"],
            budget_tokens=int(budget), tokens_seen=int(self.tokens_seen), step=int(self.step),
            train_flops=float(self.flops["train"] * self.tokens_seen),
            deploy_flops_per_token=float(self.flops["forward"]),
            train_flops_per_token=float(self.flops["train"]),
            val=final, val_avg=avg, n_avg_points=n_pts.get(self.dcfg.val_main, 1), lr0=self.tcfg.lr0,
            compiled=bool(self.tcfg.compile), torch_version=torch.__version__, git_commit=self.git_commit,
            time=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    name, out_dir, mcfg, tcfg, dcfg = load_run_config(args.config)
    Runner(name, out_dir, mcfg, tcfg, dcfg, device=args.device).run()


if __name__ == "__main__":
    main()
