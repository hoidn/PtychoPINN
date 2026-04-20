"""Profile and run-budget contracts for OpenFWI FlatVel-A smoke runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PRIMARY_PROFILE_IDS = ["hybrid_resnet_smoke", "unet_smoke"]
OPTIONAL_PROFILE_IDS = ["fno_smoke", "official_inversionnet_probe"]
DEFAULT_RUN_BUDGET = {
    "epochs": 1,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "train_samples": 32,
    "val_samples": 16,
    "test_samples": 16,
    "split_seed": 20260420,
    "device": "cuda",
    "num_workers": 0,
}


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    base_model: str
    hidden_channels: int
    fno_modes: int | None = None
    fno_blocks: int | None = None
    hybrid_downsample_steps: int | None = None
    hybrid_resnet_blocks: int | None = None

    def to_model_config(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


_PROFILES = {
    "hybrid_resnet_smoke": ModelProfile(
        profile_id="hybrid_resnet_smoke",
        base_model="hybrid_resnet",
        hidden_channels=8,
        fno_modes=4,
        fno_blocks=2,
        hybrid_downsample_steps=1,
        hybrid_resnet_blocks=1,
    ),
    "unet_smoke": ModelProfile(profile_id="unet_smoke", base_model="unet", hidden_channels=8),
    "fno_smoke": ModelProfile(
        profile_id="fno_smoke",
        base_model="fno",
        hidden_channels=8,
        fno_modes=4,
        fno_blocks=2,
    ),
    "official_inversionnet_probe": ModelProfile(
        profile_id="official_inversionnet_probe",
        base_model="official_inversionnet",
        hidden_channels=0,
    ),
}


def get_model_profile(profile_id: str) -> ModelProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown OpenFWI FlatVel-A profile_id: {profile_id}") from exc


def parse_profile_ids(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        profile_ids = list(PRIMARY_PROFILE_IDS)
    elif isinstance(value, str):
        profile_ids = [item.strip() for item in value.split(",") if item.strip()]
    else:
        profile_ids = [str(item).strip() for item in value if str(item).strip()]
    for profile_id in profile_ids:
        get_model_profile(profile_id)
    return profile_ids


def validate_run_budget(payload: dict[str, Any]) -> dict[str, Any]:
    budget = {**DEFAULT_RUN_BUDGET, **dict(payload)}
    for key in ["epochs", "batch_size", "train_samples", "val_samples", "test_samples", "split_seed", "num_workers"]:
        value = int(budget[key])
        if key == "num_workers":
            if value < 0:
                raise ValueError("num_workers must be non-negative")
        elif value <= 0:
            raise ValueError(f"{key} must be positive")
        budget[key] = value
    budget["learning_rate"] = float(budget["learning_rate"])
    if budget["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    budget["device"] = str(budget.get("device", "cuda"))
    budget["profiles"] = parse_profile_ids(budget.get("profiles"))
    return budget
