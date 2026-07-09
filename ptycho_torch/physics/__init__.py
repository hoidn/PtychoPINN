"""Task-local physics surfaces for candidate-lane experiments.

This package collects narrowly scoped differentiable forward operators that
are not part of the production CDI/PtychoPINN runtime contract.
"""

from ptycho_torch.physics.born_rytov_dt import BornRytovForward2D

__all__ = ["BornRytovForward2D"]
