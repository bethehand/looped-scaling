"""WSD learning-rate schedule (pre-registration §3, source [07] Hägele et al.).

  warmup   : linear 0 -> lr0 over `warmup_tokens`
  trunk    : constant lr0
  cooldown : linear lr0 -> 0 from `cooldown_start` to `cooldown_end` tokens
Budgets D_j are total tokens at the end of a cooldown; the branch point is D_j * (1 - cooldown_frac).
"""
from __future__ import annotations


def lr_at(tokens_seen: int, lr0: float, warmup_tokens: int,
          cooldown_start: int | None = None, cooldown_end: int | None = None) -> float:
    if warmup_tokens > 0 and tokens_seen < warmup_tokens:
        return lr0 * tokens_seen / warmup_tokens
    if cooldown_start is None:
        return lr0
    if tokens_seen <= cooldown_start:
        return lr0
    span = max(cooldown_end - cooldown_start, 1)
    frac = (tokens_seen - cooldown_start) / span
    return lr0 * max(0.0, 1.0 - frac)


def branch_points(budgets: list[int], cooldown_frac: float, batch_tokens: int) -> list[tuple[int, int]]:
    """Return [(branch_start_tokens, budget_end_tokens)] rounded down to whole batches."""
    out = []
    for D in budgets:
        start = int(round(D * (1.0 - cooldown_frac)))
        start -= start % batch_tokens
        end = int(D) - int(D) % batch_tokens
        out.append((start, end))
    return out
