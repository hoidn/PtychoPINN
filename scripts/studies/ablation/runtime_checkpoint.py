"""Strict Lightning checkpoint restoration from persisted hyperparameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .runtime_errors import RuntimeExecutionError, StudyRequestError, sha256_file


def restore_checkpoint_from_hparams(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    device: str = "cpu",
) -> Any:
    """Identity-check and strictly restore a checkpoint using its own hparams."""
    checkpoint_path = Path(checkpoint_path)
    digest = sha256_file(checkpoint_path)
    if digest != expected_sha256:
        raise StudyRequestError(
            f"checkpoint {checkpoint_path} sha256 {digest} does not match "
            f"the recorded evidence hash {expected_sha256}"
        )

    from ptycho_torch.model import PtychoPINN_Lightning

    try:
        model = PtychoPINN_Lightning.load_from_checkpoint(
            str(checkpoint_path), map_location=device, strict=True
        )
    except TypeError as error:
        raise StudyRequestError(
            f"checkpoint {checkpoint_path} carries no persisted "
            "hyper_parameters (config dataclasses); it cannot be restored "
            f"without external configs. Original error: {error}"
        ) from error
    except RuntimeError as error:
        raise RuntimeExecutionError(
            "checkpoint_load",
            "strict weight restoration failed for selected checkpoint "
            f"{checkpoint_path}; do not force with strict=False. "
            f"Original error: {error}",
        ) from error
    model.eval()
    return model
