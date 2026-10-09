"""Synthetic depth-controlled reasoning tasks (extension C, 06_合成推理任务设计草案.md v2).

chain : modular arithmetic chain, p = 97.  BOS x0 SEP (OP c SEP) x d ARROW answer
        ADD c -> (x + c) mod p ; MUL c (c in 2..p-1) -> x * c mod p ; SQA c -> (x * x + c) mod p
        The operation tables are global (learned in the weights); with SQA in most chains the composition of two
        steps is an arbitrary function on Z_p, so "one layer does two steps" has no cheap representation.
hops  : pointer chasing on n = 32 nodes.  BOS (a GT f(a) SEP) x 32 in random order, QRY x k ARROW answer = f^k(x)
        A log-depth shortcut (prefix doubling of f) exists; the task is the contrast case.
One problem per sequence, PAD after the answer; the answer is predicted at the ARROW position. Problems are drawn
on the fly from a numpy Generator (unlimited, no repetition); the intermediate values x_0..x_d are returned for the
probes. Token ids: numbers 0..p-1, then the special tokens.
"""
from __future__ import annotations

import numpy as np

P = 97
N_NODES = 32
BOS, PAD, SEP, ARROW, ADD, MUL, SQA, GT, QRY = range(P, P + 9)
VOCAB_SIZE = 112                      # P + 9 = 106 used, rounded up
SEQ_LEN = {"chain": 80, "hops": 144}  # chain: 3 d + 5 tokens (d <= 25); hops: 1 + 32 * 4 + 4 + 1 = 134
OPS = (ADD, MUL, SQA)
OP_PROBS = (0.3, 0.3, 0.4)
TASKS = ("chain", "hops")


def apply_op(op: int, c: int, x: int) -> int:
    if op == ADD:
        return (x + c) % P
    if op == MUL:
        return (x * c) % P
    if op == SQA:
        return (x * x + c) % P
    raise ValueError(op)


def gen_chain(rng: np.random.Generator, d: int) -> tuple[list[int], int, list[int]]:
    """-> (prompt tokens ending with ARROW, index of ARROW, intermediate values x_0..x_d)."""
    x = int(rng.integers(P))
    xs, toks = [x], [BOS, x, SEP]
    for o in rng.choice(3, size=d, p=OP_PROBS):
        op = OPS[o]
        c = int(rng.integers(2, P)) if op == MUL else int(rng.integers(P))
        x = apply_op(op, c, x)
        xs.append(x)
        toks += [op, c, SEP]
    toks.append(ARROW)
    return toks, len(toks) - 1, xs


def op_positions(d: int) -> list[int]:
    """Index of the op token of step j = 1..d in a chain prompt."""
    return [3 * j for j in range(1, d + 1)]


def gen_hops(rng: np.random.Generator, k: int) -> tuple[list[int], int, list[int]]:
    f = rng.integers(N_NODES, size=N_NODES)
    toks = [BOS]
    for a in rng.permutation(N_NODES):
        toks += [int(a), GT, int(f[a]), SEP]
    x = int(rng.integers(N_NODES))
    xs = [x]
    for _ in range(k):
        x = int(f[x])
        xs.append(x)
    toks += [QRY, xs[0], k, ARROW]
    return toks, len(toks) - 1, xs


GEN = {"chain": gen_chain, "hops": gen_hops}


def make_batch(task: str, d: int, batch: int, rng: np.random.Generator):
    """-> tokens (batch, T) int64 with the answer after ARROW and PAD beyond; answer positions (batch,);
    answers (batch,); intermediates (batch, d + 1)."""
    T = SEQ_LEN[task]
    x = np.full((batch, T), PAD, dtype=np.int64)
    pos = np.empty(batch, dtype=np.int64)
    ans = np.empty(batch, dtype=np.int64)
    inter = np.empty((batch, d + 1), dtype=np.int64)
    gen = GEN[task]
    for i in range(batch):
        toks, p, xs = gen(rng, d)
        toks = toks + [xs[-1]]
        x[i, : len(toks)] = toks
        pos[i], ans[i], inter[i] = p, xs[-1], xs
    return x, pos, ans, inter


def parse_chain(toks: list[int]) -> int:
    """Independent evaluation of a chain prompt (for tests): returns x_d."""
    assert toks[0] == BOS and toks[2] == SEP
    x, i = toks[1], 3
    while toks[i] != ARROW:
        x = apply_op(toks[i], toks[i + 1], x)
        assert toks[i + 2] == SEP
        i += 3
    return x
