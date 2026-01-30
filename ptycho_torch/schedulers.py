"""Learning rate scheduler utilities for PyTorch Lightning training.

Provides deterministic warmup+cosine annealing schedule for stable_hybrid
architecture training dynamics (Phase 6 of FNO-STABILITY-OVERHAUL-001).
"""

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def build_warmup_cosine_scheduler(optimizer, total_epochs, warmup_epochs, min_lr_ratio):
    """Build a warmup + cosine annealing LR scheduler.

    Args:
        optimizer: PyTorch optimizer
        total_epochs: Total training epochs
        warmup_epochs: Number of linear warmup epochs
        min_lr_ratio: Minimum LR as fraction of base LR (eta_min = base_lr * ratio)

    Returns:
        LR scheduler (SequentialLR if warmup > 0, else CosineAnnealingLR)
    """
    warmup_epochs = max(0, warmup_epochs)
    eta_min = optimizer.param_groups[0]['lr'] * min_lr_ratio
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
