"""Model loading for PDEBench CNS rollout videos."""

from __future__ import annotations

import json
from dataclasses import fields, dataclass
from pathlib import Path
from typing import Any

import torch

from scripts.studies.pdebench_image128.models import build_model_from_profile
from scripts.studies.pdebench_image128.run_config import ModelProfile, get_model_profile


class MissingCnsCheckpointError(FileNotFoundError):
    """Raised when a CNS rollout has no model state to execute."""


@dataclass(frozen=True)
class CnsPredictor:
    row_id: str
    history_len: int
    field_order: tuple[str, ...]
    device: torch.device
    model: torch.nn.Module

    def __call__(self, history_norm: torch.Tensor) -> torch.Tensor:
        history = history_norm.detach().to(self.device).float()
        if history.ndim == 3:
            model_input = history.unsqueeze(0)
        elif history.ndim == 4:
            model_input = history.reshape(1, history.shape[0] * history.shape[1], history.shape[2], history.shape[3])
        elif history.ndim == 5 and history.shape[0] == 1:
            model_input = history.reshape(1, history.shape[1] * history.shape[2], history.shape[3], history.shape[4])
        else:
            raise ValueError(f"history_norm must have shape (T,C,H,W) or (1,T,C,H,W), got {tuple(history.shape)}")
        with torch.no_grad():
            output = self.model(model_input)
        return output.detach().cpu()[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_first_existing_json(run_root: Path, names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        path = run_root / name
        if path.exists():
            return _read_json(path)
    raise FileNotFoundError(f"none of these metadata files exist under {run_root}: {', '.join(names)}")


def _model_profile_from_payload(row_id: str, payload: dict[str, Any]) -> ModelProfile:
    config = dict(payload.get("profile_config") or {})
    config.setdefault("profile_id", row_id)
    try:
        registered = get_model_profile(row_id)
    except KeyError:
        registered = None
    if registered is not None and not config:
        return registered
    if registered is not None:
        merged = {**registered.to_model_config(), **config}
        return ModelProfile(**{key: value for key, value in merged.items() if key in {field.name for field in fields(ModelProfile)}})
    valid = {field.name for field in fields(ModelProfile)}
    return ModelProfile(**{key: value for key, value in config.items() if key in valid})


def _task_metadata(metadata: dict[str, Any], *, history_len: int, field_order: tuple[str, ...]) -> dict[str, Any]:
    dimensions = metadata.get("dimensions") if isinstance(metadata.get("dimensions"), dict) else {}
    state_shape = list(metadata.get("state_shape", []))
    height = int(dimensions.get("height", state_shape[-2] if len(state_shape) >= 2 else 0))
    width = int(dimensions.get("width", state_shape[-1] if len(state_shape) >= 1 else 0))
    return {
        "task_id": "2d_cfd_cns",
        "spatial_shape": [height, width],
        "state_shape": state_shape,
        "field_order": list(field_order),
        "field_axis_order": str(metadata.get("field_axis_order", metadata.get("axis_order", "NTHW"))),
        "history_len": int(history_len),
        "time_steps": int(metadata.get("time_steps", dimensions.get("time_steps", 0))),
        "trajectory_count": int(metadata.get("trajectory_count", dimensions.get("num_trajectories", 0))),
        "input_channels": int(history_len * len(field_order)),
        "target_channels": int(len(field_order)),
        "dx": float(metadata.get("dx", 1.0)),
        "dy": float(metadata.get("dy", 1.0)),
        "dt": float(metadata.get("dt", 1.0)),
        "eta": float(metadata.get("eta", 0.0)),
        "zeta": float(metadata.get("zeta", 0.0)),
        "boundary_condition": str(metadata.get("boundary_condition", "periodic")),
        "sample_contract": str(metadata.get("dynamic_history_contract", f"concat u[t-{history_len}:t] -> u[t]")),
    }


def load_cns_predictor(
    *,
    run_root: Path,
    row_id: str,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> CnsPredictor:
    """Load a CNS model row as a common rollout predictor callable."""
    run_root = Path(run_root)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else run_root / f"model_state_{row_id}.pt"
    if not checkpoint.exists():
        raise MissingCnsCheckpointError(
            f"CNS rollout video export requires model_state_{row_id}.pt or --checkpoint-path; "
            "one-step comparison NPZ files cannot produce autoregressive videos."
        )
    metadata = _load_first_existing_json(run_root, ("hdf5_metadata.json", "dataset_manifest.json"))
    state_stats = _read_json(run_root / "normalization_stats_state.json")
    profile_payload = _read_json(run_root / f"model_profile_{row_id}.json")
    field_order = tuple(str(item) for item in metadata.get("field_order", state_stats.get("field_order", [])))
    if not field_order:
        raise ValueError("CNS metadata must include field_order")
    history_len = int(metadata.get("history_len", state_stats.get("history_len", 1)))
    dimensions = metadata.get("dimensions") if isinstance(metadata.get("dimensions"), dict) else {}
    state_shape = list(metadata.get("state_shape", []))
    height = int(dimensions.get("height", state_shape[-2] if len(state_shape) >= 2 else 0))
    width = int(dimensions.get("width", state_shape[-1] if len(state_shape) >= 1 else 0))
    if height <= 0 or width <= 0:
        raise ValueError("CNS metadata must include spatial dimensions")
    profile = _model_profile_from_payload(row_id, profile_payload)
    torch_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else device)
    model = build_model_from_profile(
        profile,
        in_channels=history_len * len(field_order),
        out_channels=len(field_order),
        spatial_shape=(height, width),
        task_metadata=_task_metadata(metadata, history_len=history_len, field_order=field_order),
    ).to(torch_device)
    state_dict = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return CnsPredictor(
        row_id=row_id,
        history_len=history_len,
        field_order=field_order,
        device=torch_device,
        model=model,
    )
