"""Run-budget and model-profile contracts for PDEBench 128x128 image-suite tasks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    base_model: str
    hidden_channels: int
    fno_modes: int | None = None
    fno_blocks: int | None = None
    hybrid_downsample_steps: int | None = None
    hybrid_resnet_blocks: int | None = None
    unet_init_features: int | None = None
    evidence_scope: str = "benchmark_candidate"
    strong_baseline: bool = False

    def to_model_config(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


_PROFILES: dict[str, ModelProfile] = {
    "hybrid_resnet_base": ModelProfile(
        profile_id="hybrid_resnet_base",
        base_model="hybrid_resnet",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
    ),
    "fno_base": ModelProfile(
        profile_id="fno_base",
        base_model="fno",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        strong_baseline=True,
    ),
    "unet_strong": ModelProfile(
        profile_id="unet_strong",
        base_model="unet_strong",
        hidden_channels=32,
        unet_init_features=32,
        strong_baseline=True,
    ),
    "unet_tiny_smoke": ModelProfile(
        profile_id="unet_tiny_smoke",
        base_model="unet_tiny",
        hidden_channels=16,
        evidence_scope="readiness-only",
        strong_baseline=False,
    ),
}

PRIMARY_DARCY_PROFILE_IDS = ["hybrid_resnet_base", "fno_base", "unet_strong"]


def get_model_profile(profile_id: str) -> ModelProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown PDEBench image128 profile_id: {profile_id}") from exc


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


def validate_darcy_run_budget(payload: dict[str, Any]) -> dict[str, Any]:
    budget = dict(payload)
    if str(budget.get("task_id")) != "darcy":
        raise ValueError("Darcy run budget must set task_id='darcy'")
    mode = str(budget.get("mode", "readiness"))
    if mode not in {"readiness", "benchmark"}:
        raise ValueError("Darcy run budget mode must be readiness or benchmark")
    budget["mode"] = mode
    for key in ("train_count", "val_count", "test_count", "training_seed", "batch_size", "epochs", "num_workers"):
        if key not in budget:
            raise ValueError(f"run budget missing required field: {key}")
        budget[key] = int(budget[key])
    if budget["batch_size"] <= 0 or budget["epochs"] <= 0:
        raise ValueError("run budget batch_size and epochs must be positive")
    if budget["num_workers"] < 0:
        raise ValueError("run budget num_workers must be non-negative")
    budget["primary_profiles"] = parse_profile_ids(budget.get("primary_profiles"))
    if not budget["primary_profiles"]:
        raise ValueError("run budget primary_profiles must not be empty")
    budget["loss"] = str(budget.get("loss", "mae")).lower()
    if budget["loss"] not in {"mae", "mse", "relative_l2"}:
        raise ValueError("run budget loss must be mae, mse, or relative_l2")
    budget["optimizer"] = str(budget.get("optimizer", "adam")).lower()
    if budget["optimizer"] != "adam":
        raise ValueError("run budget optimizer must be adam")
    budget["learning_rate"] = float(budget.get("learning_rate", 2e-4))
    if budget["learning_rate"] <= 0.0:
        raise ValueError("run budget learning_rate must be positive")
    budget["scheduler"] = str(budget.get("scheduler", "ReduceLROnPlateau"))
    budget["plateau_factor"] = float(budget.get("plateau_factor", 0.5))
    budget["plateau_patience"] = int(budget.get("plateau_patience", 2))
    budget["plateau_min_lr"] = float(budget.get("plateau_min_lr", 1e-4))
    budget["plateau_threshold"] = float(budget.get("plateau_threshold", 0.0))
    budget["precision"] = str(budget.get("precision", "float32"))
    budget["device"] = str(budget.get("device", "cuda"))
    if mode == "benchmark":
        if (budget["train_count"], budget["val_count"], budget["test_count"]) != (8000, 1000, 1000):
            raise ValueError("Darcy benchmark budget must use the full train split 8000/1000/1000")
        missing = [profile for profile in PRIMARY_DARCY_PROFILE_IDS if profile not in budget["primary_profiles"]]
        if missing:
            raise ValueError(f"Darcy benchmark budget missing required primary profiles: {missing}")
    else:
        budget.setdefault("evidence_scope", "smoke_feasibility_only")
    return budget


def default_darcy_benchmark_budget() -> dict[str, Any]:
    return validate_darcy_run_budget(
        {
            "task_id": "darcy",
            "mode": "benchmark",
            "train_count": 8000,
            "val_count": 1000,
            "test_count": 1000,
            "primary_profiles": PRIMARY_DARCY_PROFILE_IDS,
            "training_seed": 20260420,
            "loss": "mae",
            "loss_rationale": "Project grid-lines-derived recipe for same-protocol local comparison.",
            "optimizer": "adam",
            "learning_rate": 2e-4,
            "scheduler": "ReduceLROnPlateau",
            "plateau_factor": 0.5,
            "plateau_patience": 2,
            "plateau_min_lr": 1e-4,
            "plateau_threshold": 0.0,
            "batch_size": 8,
            "epochs": 50,
            "precision": "float32",
            "device": "cuda",
            "num_workers": 2,
            "capped_training_set": False,
        }
    )


def write_default_darcy_benchmark_budget(path: Path) -> Path:
    payload = default_darcy_benchmark_budget()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
