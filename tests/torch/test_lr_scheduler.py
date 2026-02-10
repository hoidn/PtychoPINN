"""Tests for warmup+cosine LR scheduler (Phase 6 of FNO-STABILITY-OVERHAUL-001)."""
import math
import pytest
import torch
from ptycho_torch.schedulers import build_warmup_cosine_scheduler


def test_warmup_cosine_scheduler_progression():
    """Test that warmup+cosine scheduler follows expected LR trajectory."""
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = build_warmup_cosine_scheduler(opt, total_epochs=10, warmup_epochs=2, min_lr_ratio=0.05)
    lrs = []
    for epoch in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]['lr'])
    # After warmup (epoch 1), LR should reach base_lr
    assert lrs[1] >= 0.9 * 1e-3, f"LR after warmup should be near base_lr, got {lrs[1]}"
    # LR should ramp up during warmup
    assert lrs[0] < lrs[1], "LR should increase during warmup"
    # Final LR should be near eta_min = 1e-3 * 0.05 = 5e-5
    assert lrs[-1] == pytest.approx(5e-5, rel=0.1), f"Final LR should be near eta_min, got {lrs[-1]}"


def test_warmup_cosine_no_warmup():
    """Test pure cosine schedule when warmup_epochs=0."""
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = build_warmup_cosine_scheduler(opt, total_epochs=10, warmup_epochs=0, min_lr_ratio=0.05)
    lrs = []
    for epoch in range(10):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]['lr'])
    # First LR should already be near base_lr (cosine starts high)
    assert lrs[0] > 5e-4, f"First LR should be high without warmup, got {lrs[0]}"
    # Final LR near eta_min
    assert lrs[-1] == pytest.approx(5e-5, rel=0.1)
