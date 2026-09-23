"""Fixed-order token stream and validation sets.

All runs read the same contiguous stream of uint16 token ids (pre-registration §3: "固定分片顺序，所有任务同序").
Batch layout follows the contiguous nanoGPT convention: a batch of B sequences of length T is the slice
stream[pos : pos + B*T + 1]; x = slice[:-1].view(B, T), y = slice[1:].view(B, T); pos advances by B*T.
Seeds change initialization only, never the data order.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch


class TokenStream:
    def __init__(self, shard_paths: list[str]):
        if not shard_paths:
            raise ValueError("no shards given")
        self.paths = list(shard_paths)
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.sizes = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total = int(self.offsets[-1])

    @classmethod
    def from_glob(cls, pattern: str) -> "TokenStream":
        paths = sorted(glob.glob(pattern))
        return cls(paths)

    def read(self, start: int, length: int) -> np.ndarray:
        if start + length > self.total:
            raise EOFError(f"stream exhausted: need {start + length} tokens, have {self.total}")
        out = np.empty(length, dtype=np.uint16)
        pos, filled = start, 0
        while filled < length:
            si = int(np.searchsorted(self.offsets, pos, side="right") - 1)
            local = pos - int(self.offsets[si])
            n = int(min(length - filled, self.sizes[si] - local))
            out[filled : filled + n] = self.shards[si][local : local + n]
            filled += n
            pos += n
        return out


class BatchSampler:
    """Sequential batches from a TokenStream; `position` is the token offset and is checkpointed."""

    def __init__(self, stream: TokenStream, seq_len: int, batch_seqs: int, position: int = 0):
        self.stream, self.T, self.B = stream, seq_len, batch_seqs
        self.position = int(position)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.B * self.T
        buf = self.stream.read(self.position, n + 1).astype(np.int64)
        x = torch.from_numpy(buf[:-1]).view(self.B, self.T)
        y = torch.from_numpy(buf[1:]).view(self.B, self.T)
        self.position += n
        return x, y

    def state_dict(self) -> dict:
        return dict(position=self.position)

    def load_state_dict(self, d: dict) -> None:
        self.position = int(d["position"])


class ValSet:
    """A fixed validation token file evaluated as non-overlapping windows of seq_len (+1 target)."""

    def __init__(self, path: str, seq_len: int, max_tokens: int | None = None):
        self.path = path
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        if max_tokens is not None:
            self.tokens = self.tokens[:max_tokens]
        self.T = seq_len
        self.n_windows = (len(self.tokens) - 1) // seq_len

    def batches(self, batch_seqs: int):
        T = self.T
        for start in range(0, self.n_windows, batch_seqs):
            b = min(batch_seqs, self.n_windows - start)
            buf = np.asarray(self.tokens[start * T : start * T + b * T + 1]).astype(np.int64)
            x = torch.from_numpy(buf[:-1]).view(b, T)
            y = torch.from_numpy(buf[1:]).view(b, T)
            yield x, y


def write_shard(path: str, tokens: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tokens.astype(np.uint16).tofile(path)
