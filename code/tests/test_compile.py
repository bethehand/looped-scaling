"""Per-block torch.compile (the optional speed-up) must not change gradients or outputs.
Kept apart from test_model.py so a missing compiler toolchain on some machine does not mask the core checks."""
import pytest
import torch

from looped.model import LoopedLM, ModelConfig
from tests.test_model import _DEVICES, _grads
# ---------- per-block torch.compile must give the same gradients as eager, including truncation + checkpointing ----------


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("placement,k,ckpt", [("middle", 2, False), ("middle", 2, True), ("whole", 2, True), ("middle", 0, False)])
def test_compiled_blocks_match_eager(device, placement, k, ckpt):
    import torch._dynamo
    torch._dynamo.reset()
    torch.manual_seed(0)
    cfg = ModelConfig(placement=placement, r=4, k_bwd=k, ckpt_loops=ckpt, vocab_size=64, d_model=64, head_dim=32,
                      seq_len=32, n_layers=4, n_prelude=1, n_coda=1)
    eager = LoopedLM(cfg).to(device).train()
    comp = LoopedLM(cfg).to(device).train()
    comp.load_state_dict(eager.state_dict())
    # CPU: tracing semantics only (aot_eager, no C++ toolchain needed); CUDA: the real inductor/Triton path
    comp.compile_blocks(backend="aot_eager" if device == "cpu" else "inductor")
    x = torch.randint(0, 64, (2, 32), device=device)
    y = torch.randint(0, 64, (2, 32), device=device)
    ge = _grads(eager, x, y, device, autocast=True)
    for _ in range(2):   # second call exercises the cached compiled graphs
        gc = _grads(comp, x, y, device, autocast=True)
    assert ge.keys() == gc.keys()
    for name in ge:
        cos = torch.nn.functional.cosine_similarity(gc[name].flatten(), ge[name].flatten(), dim=0)
        assert cos > 0.99, (name, float(cos))
    comp.eval()
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        le, _ = eager.eval()(x, y)
        lc, _ = comp(x, y)
    assert torch.allclose(le.float(), lc.float(), atol=5e-2, rtol=5e-2)

