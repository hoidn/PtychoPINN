"""Loader capture for the V3b batch-consumption diagnostic.

Captures the exact dataloaders each real training flow hands to
``lightning.pytorch.Trainer`` by stubbing ``Trainer`` with a
sentinel-raising recorder: the construction code runs verbatim through
``run_torch_training`` (dictionary flow) respectively
``train_via_generic_loader`` (mmap flow) — nothing is reimplemented and no
training step executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .runtime_errors import RuntimeExecutionError

__all__ = [
    "CapturedLoaders",
    "capture_dictionary_training_loaders",
    "capture_mmap_training_loaders",
]


@dataclass
class CapturedLoaders:
    """The exact loader/model objects a flow would hand to ``Trainer.fit``."""

    flow: str
    train_loader: Any
    val_loader: Any
    trainer_kwargs: Mapping[str, Any]
    model: Any = None
    fit_rng_state_sha256: str | None = None

    def sampler_info(self) -> dict[str, Any]:
        def describe(loader: Any) -> dict[str, Any] | None:
            if loader is None:
                return None
            batch_sampler = getattr(loader, "batch_sampler", None)
            return {
                "loader_class": type(loader).__name__,
                "sampler_class": type(getattr(loader, "sampler", None)).__name__,
                "batch_size": getattr(loader, "batch_size", None),
                "drop_last": getattr(batch_sampler, "drop_last", None),
                "num_workers": getattr(loader, "num_workers", None),
            }

        return {
            "train": describe(self.train_loader),
            "val": describe(self.val_loader),
        }


class _LoaderCaptureRequested(Exception):
    """Sentinel raised inside the stubbed Trainer.fit after capture."""


class _CaptureTrainer:
    """Records the fit arguments, then aborts via the sentinel."""

    holder: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).holder["trainer_kwargs"] = kwargs

    def fit(
        self,
        model: Any,
        train_dataloaders: Any = None,
        val_dataloaders: Any = None,
        datamodule: Any = None,
    ) -> None:
        import hashlib

        import torch

        type(self).holder.update(
            model=model,
            train_loader=train_dataloaders,
            val_loader=val_dataloaders,
            datamodule=datamodule,
            fit_rng_state_sha256=hashlib.sha256(
                torch.random.get_rng_state().numpy().tobytes()
            ).hexdigest(),
        )
        raise _LoaderCaptureRequested()


def _capture_via(flow: str, invoke: Any) -> CapturedLoaders:
    """Run a real training flow with Trainer stubbed; return its loaders."""
    import lightning.pytorch as lightning_module

    _CaptureTrainer.holder = {}
    original = lightning_module.Trainer
    lightning_module.Trainer = _CaptureTrainer
    try:
        invoke()
    except RuntimeError as error:
        if not isinstance(error.__cause__, _LoaderCaptureRequested):
            raise
    except _LoaderCaptureRequested:
        pass
    else:
        raise RuntimeExecutionError(
            "loader_capture",
            f"{flow} flow completed without reaching Trainer.fit; nothing "
            "was captured",
        )
    finally:
        lightning_module.Trainer = original
    holder = _CaptureTrainer.holder
    if holder.get("train_loader") is None and holder.get("datamodule") is None:
        raise RuntimeExecutionError(
            "loader_capture", f"{flow} flow captured no train dataloader"
        )
    return CapturedLoaders(
        flow=flow,
        train_loader=holder.get("train_loader"),
        val_loader=holder.get("val_loader"),
        trainer_kwargs=holder.get("trainer_kwargs", {}),
        model=holder.get("model"),
        fit_rng_state_sha256=holder.get("fit_rng_state_sha256"),
    )


def capture_dictionary_training_loaders(
    config: Mapping[str, Any], train_npz: Path, test_npz: Path, work: Path
) -> CapturedLoaders:
    """Loaders the dictionary flow builds (run_torch_training, verbatim)."""
    from scripts.studies import grid_lines_torch_runner as runner_mod

    from .runtime_ladder_execution import build_runner_config

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    runner_cfg = build_runner_config(
        config, train_npz=Path(train_npz), test_npz=Path(test_npz), output_dir=work
    )
    train_data, train_metadata = runner_mod.load_cached_dataset_with_metadata(
        Path(train_npz)
    )
    test_data, test_metadata = runner_mod.load_cached_dataset_with_metadata(
        Path(test_npz)
    )

    def invoke() -> None:
        runner_mod.run_torch_training(
            runner_cfg,
            train_data,
            test_data,
            train_metadata=train_metadata,
            test_metadata=test_metadata,
        )

    return _capture_via("dictionary", invoke)


def capture_mmap_training_loaders(
    config: Mapping[str, Any],
    recipe: Any,
    train_npz: Path,
    test_npz: Path,
    work: Path,
) -> CapturedLoaders:
    """Loaders the mmap flow builds (train_via_generic_loader, verbatim)."""
    from .runtime_ladder_execution import build_runner_config
    from .runtime_ladder_mmap import train_via_generic_loader

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    runner_cfg = build_runner_config(
        config, train_npz=Path(train_npz), test_npz=Path(test_npz), output_dir=work
    )

    def invoke() -> None:
        train_via_generic_loader(
            runner_cfg, config, recipe, Path(train_npz), Path(test_npz), work
        )

    return _capture_via("mmap", invoke)
