import torch

from ptycho_torch.train_utils import compute_grad_norm


def test_compute_grad_norm_l2_matches_manual():
    lin = torch.nn.Linear(4, 2, bias=False)
    for p in lin.parameters():
        p.grad = torch.ones_like(p)
    expected = p.grad.numel() ** 0.5
    got = compute_grad_norm(lin.parameters(), norm_type=2.0)
    assert abs(got - expected) < 1e-6
