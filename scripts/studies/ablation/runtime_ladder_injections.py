"""Loader-seam injections retained for historical diagnostic evidence.

Each injection swaps a module-attribute subclass of ``TensorDictDataLoader``
in for the duration of ONE training call (``components`` imports the class
from ``ptycho_torch.dataloader`` at call time, so the patch is honored
without touching upstream code) and restores the original unconditionally.
Defaults are structural no-ops so every existing rung is bit-preserved.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .runtime_errors import RuntimeExecutionError

__all__ = [
    "_flatten_probe_batch",
    "_probe_layout_injection",
    "_train_sampler_injection",
]


def _flatten_probe_batch(probe: Any) -> Any:
    """Recreate the historical flat probe rank for sealed rung1f evidence.

    Task 25 migrated current dictionary emission to `(B, C, P, H, W)`. This
    value-preserving reshape maps `(B, C=1, P=1, H, W)` to the old `(B, H, W)`
    rank. A genuine multi-mode probe has no flat representation, so flattening
    it would silently drop physics (fail closed instead).
    """
    if probe.ndim == 3:
        return probe
    if probe.ndim != 5 or probe.shape[1] != 1 or probe.shape[2] != 1:
        raise RuntimeExecutionError(
            "mmap_training",
            "dictionary_flat probe layout requires a single-channel "
            "single-mode (B, 1, 1, H, W) probe batch; got a multi-mode/"
            f"multi-channel shape {tuple(probe.shape)}",
        )
    return probe.reshape(probe.shape[0], probe.shape[-2], probe.shape[-1])


@contextmanager
def _probe_layout_injection(regime: str) -> Iterator[None]:
    """Engage the ``mmap_probe_batch_shape`` regime for one training call.

    ``"modes"`` is a no-op: the real TensorDict loader emits the documented
    (B, C, P, H, W) probe batch (ptycho_torch/dataloader.py __getitem__
    docstring). ``"dictionary_flat"`` re-expresses every emitted batch's
    probe in the historical pre-Task-25 rank (B, H, W), with byte-identical
    values. Mechanism the sealed rung confirmed: ProbeIllumination
    (ptycho_torch/model.py:1261-1268) right-align-broadcasts a (B, H, W)
    probe into P=B pseudo-modes and pad_and_diffract's COHERENT mode sum
    (ptycho_torch/helper.py:670) multiplies the predicted field by B — the
    forward-gain regime the qualified reference was trained under. Applies
    to BOTH train and val loaders in the historical reconstruction.
    """
    if regime == "modes":
        yield
        return
    if regime != "dictionary_flat":  # closed namespace; fail loud on drift
        raise RuntimeExecutionError(
            "mmap_training", f"unknown mmap_probe_batch_shape regime {regime!r}"
        )
    from ptycho_torch import dataloader as loader_module

    original = loader_module.TensorDictDataLoader

    class TensorDictDataLoader(original):  # type: ignore[misc,valid-type]
        """Historical rung1f probe-layout injection; train and validation."""

        def __iter__(self) -> Any:
            for batch in super().__iter__():
                tensor_dict, probe, scaling = batch
                yield tensor_dict, _flatten_probe_batch(probe), scaling

    loader_module.TensorDictDataLoader = TensorDictDataLoader
    try:
        yield
    finally:
        loader_module.TensorDictDataLoader = original


@contextmanager
def _train_sampler_injection(
    train_dataset: Any, regime: str, seed: int
) -> Iterator[None]:
    """Engage the ``mmap_train_sampler`` regime for one training call.

    ``"sequential"`` is a no-op: the real generic-loader path builds a plain
    ``TensorDictDataLoader`` whose default sampler is torch's
    ``SequentialSampler`` (the V3b-diagnosed fixed raster order). Bit
    preservation of every existing rung under the default is therefore
    structural. ``"shuffled"`` swaps in a per-epoch ``RandomSampler``
    deterministically seeded with the rung seed via an explicit
    ``torch.Generator`` — the dictionary path's train regime — for the TRAIN
    dataset only, through the SAME ``TensorDictDataLoader`` machinery
    (``components._build_dataloaders_from_ptycho_dataset`` imports the class
    from ``ptycho_torch.dataloader`` at call time, so a module-attribute
    subclass patch is honored without touching upstream code).
    """
    if regime == "sequential":
        yield
        return
    if regime != "shuffled":  # closed namespace; fail loud on drift
        raise RuntimeExecutionError(
            "mmap_training", f"unknown mmap_train_sampler regime {regime!r}"
        )
    import torch
    from torch.utils.data import RandomSampler

    from ptycho_torch import dataloader as loader_module

    original = loader_module.TensorDictDataLoader

    class TensorDictDataLoader(original):  # type: ignore[misc,valid-type]
        """Sampler-injected variant (rungs 1d/1e); train dataset only."""

        def __init__(self, dataset: Any, *args: Any, **kwargs: Any) -> None:
            if dataset is train_dataset:
                if "sampler" in kwargs or kwargs.get("shuffle"):
                    # S-3: never silently defer to an upstream sampler —
                    # the rung's single-variable claim would be wrong.
                    raise RuntimeExecutionError(
                        "mmap_training",
                        "mmap_train_sampler='shuffled' collided with an "
                        "upstream sampler/shuffle argument "
                        f"(sampler={'sampler' in kwargs}, "
                        f"shuffle={kwargs.get('shuffle')!r}); the injected "
                        "regime would not be in effect",
                    )
                generator = torch.Generator()
                generator.manual_seed(int(seed))
                kwargs["sampler"] = RandomSampler(dataset, generator=generator)
            super().__init__(dataset, *args, **kwargs)

    loader_module.TensorDictDataLoader = TensorDictDataLoader
    try:
        yield
    finally:
        loader_module.TensorDictDataLoader = original
