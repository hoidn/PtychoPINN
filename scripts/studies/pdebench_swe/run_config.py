"""Run-budget and model-profile contracts for PDEBench SWE longer execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    """Serializable supervised SWE model profile."""

    profile_id: str
    base_model: str
    hidden_channels: int
    fno_modes: int | None = None
    fno_blocks: int | None = None
    hybrid_downsample_steps: int | None = None
    hybrid_resnet_blocks: int | None = None

    def to_model_config(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


_PROFILES: dict[str, ModelProfile] = {
    "hybrid_resnet_base": ModelProfile(
        profile_id="hybrid_resnet_base",
        base_model="hybrid_resnet",
        hidden_channels=16,
        fno_modes=8,
        fno_blocks=4,
        hybrid_downsample_steps=1,
        hybrid_resnet_blocks=2,
    ),
    "fno_base": ModelProfile(
        profile_id="fno_base",
        base_model="fno",
        hidden_channels=16,
        fno_modes=8,
        fno_blocks=4,
    ),
    "unet_base": ModelProfile(
        profile_id="unet_base",
        base_model="unet",
        hidden_channels=16,
    ),
    "hybrid_resnet_spectral_reduced": ModelProfile(
        profile_id="hybrid_resnet_spectral_reduced",
        base_model="hybrid_resnet",
        hidden_channels=16,
        fno_modes=2,
        fno_blocks=4,
        hybrid_downsample_steps=1,
        hybrid_resnet_blocks=2,
    ),
    "hybrid_resnet_local_reduced": ModelProfile(
        profile_id="hybrid_resnet_local_reduced",
        base_model="hybrid_resnet",
        hidden_channels=16,
        fno_modes=8,
        fno_blocks=4,
        hybrid_downsample_steps=1,
        hybrid_resnet_blocks=1,
    ),
}

BUILTIN_PROFILE_IDS = list(_PROFILES)
PRIMARY_PROFILE_IDS = ["hybrid_resnet_base", "fno_base", "unet_base"]
ABLATION_PROFILE_IDS = ["hybrid_resnet_spectral_reduced", "hybrid_resnet_local_reduced"]


def get_model_profile(profile_id: str) -> ModelProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown PDEBench SWE profile_id: {profile_id}") from exc


def parse_profile_ids(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        profile_ids = [item.strip() for item in value.split(",") if item.strip()]
    else:
        profile_ids = [str(item).strip() for item in value if str(item).strip()]
    for profile_id in profile_ids:
        get_model_profile(profile_id)
    return profile_ids


def load_run_budget(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_run_budget(payload)


def validate_run_budget(payload: dict[str, Any]) -> dict[str, Any]:
    budget = dict(payload)
    required_positive_ints = [
        "epochs",
        "batch_size",
        "max_train_trajectories",
        "max_val_trajectories",
        "max_test_trajectories",
        "max_pairs_per_trajectory",
        "normalization_max_samples",
        "num_workers",
    ]
    for key in required_positive_ints:
        if key not in budget:
            raise ValueError(f"run budget missing required field: {key}")
        value = int(budget[key])
        if key == "num_workers":
            if value < 0:
                raise ValueError("run budget num_workers must be non-negative")
        elif value <= 0:
            raise ValueError(f"run budget {key} must be positive")
        budget[key] = value
    if float(budget.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("run budget learning_rate must be positive")
    budget["learning_rate"] = float(budget["learning_rate"])

    primary_profiles = parse_profile_ids(budget.get("primary_profiles"))
    missing_primary = [profile for profile in PRIMARY_PROFILE_IDS if profile not in primary_profiles]
    if missing_primary:
        raise ValueError(f"run budget missing required primary profiles: {missing_primary}")
    budget["primary_profiles"] = primary_profiles

    ablation_profiles = parse_profile_ids(budget.get("ablation_profiles", []))
    budget["ablation_profiles"] = ablation_profiles

    eval_splits = [str(item) for item in budget.get("eval_splits", [])]
    if not eval_splits:
        raise ValueError("run budget eval_splits must not be empty")
    unsupported = sorted(set(eval_splits) - {"train", "val", "test"})
    if unsupported:
        raise ValueError(f"unsupported eval_splits: {unsupported}")
    if "test" not in eval_splits:
        raise ValueError("run budget eval_splits must include test")
    budget["eval_splits"] = eval_splits
    budget["device"] = str(budget.get("device", "cuda"))
    return budget
