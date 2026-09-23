"""Validation loss (nats/token) on fixed validation sets."""
from __future__ import annotations

import torch

from .data import ValSet


@torch.no_grad()
def evaluate(model, valset: ValSet, batch_seqs: int, device: torch.device,
             dtype: torch.dtype | None = None, r: int | None = None) -> float:
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in valset.batches(batch_seqs):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        ctx = torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else torch.autocast(device_type=device.type, enabled=False)
        with ctx:
            _, loss = model(x, y, r=r)
        n = y.numel()
        total_loss += float(loss) * n
        total_tokens += n
    if was_training:
        model.train()
    return total_loss / max(total_tokens, 1)
