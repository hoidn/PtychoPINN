"""Profile and run-budget contracts for OpenFWI FlatVel-A smoke runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PRIMARY_PROFILE_IDS = ["hybrid_resnet_smoke", "unet_smoke"]
OPTIONAL_PROFILE_IDS = ["fno_smoke", "official_inversionnet_probe"]
DEFAULT_RUN_BUDGET = {
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 2e-4,
    "scheduler": "ReduceLROnPlateau",
    "plateau_factor": 0.5,
    "plateau_patience": 2,
    "plateau_min_lr": 1e-5,
    "plateau_threshold": 0.0,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.999,
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
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
    ),
    "unet_smoke": ModelProfile(profile_id="unet_smoke", base_model="unet", hidden_channels=8),
    "fno_smoke": ModelProfile(
        profile_id="fno_smoke",
        base_model="fno",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
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
    budget["scheduler"] = str(budget.get("scheduler", "ReduceLROnPlateau"))
    if budget["scheduler"] not in {"Default", "ReduceLROnPlateau"}:
        raise ValueError("scheduler must be Default or ReduceLROnPlateau")
    budget["plateau_factor"] = float(budget["plateau_factor"])
    if not 0.0 < budget["plateau_factor"] < 1.0:
        raise ValueError("plateau_factor must be in (0, 1)")
    budget["plateau_patience"] = int(budget["plateau_patience"])
    if budget["plateau_patience"] < 0:
        raise ValueError("plateau_patience must be non-negative")
    budget["plateau_min_lr"] = float(budget["plateau_min_lr"])
    if budget["plateau_min_lr"] < 0:
        raise ValueError("plateau_min_lr must be non-negative")
    if budget["plateau_min_lr"] > 1e-5:
        raise ValueError("plateau_min_lr must be no higher than 1e-5 for PDE studies")
    budget["plateau_threshold"] = float(budget["plateau_threshold"])
    if budget["plateau_threshold"] < 0:
        raise ValueError("plateau_threshold must be non-negative")
    budget["optimizer"] = str(budget.get("optimizer", "adam")).lower()
    if budget["optimizer"] != "adam":
        raise ValueError("optimizer must be adam")
    budget["weight_decay"] = float(budget["weight_decay"])
    if budget["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative")
    budget["beta1"] = float(budget["beta1"])
    budget["beta2"] = float(budget["beta2"])
    if not 0 <= budget["beta1"] < 1:
        raise ValueError("beta1 must be in [0, 1)")
    if not 0 <= budget["beta2"] < 1:
        raise ValueError("beta2 must be in [0, 1)")
    budget["device"] = str(budget.get("device", "cuda"))
    budget["profiles"] = parse_profile_ids(budget.get("profiles"))
    return budget
