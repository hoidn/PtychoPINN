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
    hybrid_upsampler: str | None = None
    hybrid_skip_connections: bool | None = None
    hybrid_skip_style: str | None = None
    spectral_bottleneck_blocks: int | None = None
    spectral_bottleneck_modes: int | None = None
    spectral_bottleneck_share_weights: bool | None = None
    spectral_bottleneck_gate_init: float | None = None
    spectral_bottleneck_gate_mode: str | None = None
    ffno_bottleneck_blocks: int | None = None
    ffno_bottleneck_modes: int | None = None
    ffno_bottleneck_share_weights: bool | None = None
    ffno_bottleneck_mlp_ratio: float | None = None
    ffno_bottleneck_gate_init: float | None = None
    ffno_bottleneck_norm: str | None = None
    ffno_bottleneck_local_conv: bool | None = None
    ffno_bottleneck_local_conv_kernel_size: int | None = None
    author_ffno_width: int | None = None
    author_ffno_modes: int | None = None
    author_ffno_layers: int | None = None
    author_ffno_share_weight: bool | None = None
    author_ffno_factor: int | None = None
    author_ffno_ff_weight_norm: bool | None = None
    author_ffno_n_ff_layers: int | None = None
    author_ffno_gain: float | None = None
    author_ffno_dropout: float | None = None
    author_ffno_in_dropout: float | None = None
    author_ffno_layer_norm: bool | None = None
    author_ffno_use_position: bool | None = None
    author_ffno_mode: str | None = None
    gnot_hidden: int | None = None
    gnot_layers: int | None = None
    gnot_heads: int | None = None
    gnot_experts: int | None = None
    gnot_inner_multiplier: int | None = None
    gnot_mlp_layers: int | None = None
    gnot_attn_type: str | None = None
    training_loss: str | None = None
    optimizer_name: str | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    scheduler_name: str | None = None
    scheduler_pct_start: float | None = None
    scheduler_div_factor: float | None = None
    scheduler_final_div_factor: float | None = None
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
    "hybrid_resnet_modes24": ModelProfile(
        profile_id="hybrid_resnet_modes24",
        base_model="hybrid_resnet",
        hidden_channels=32,
        fno_modes=24,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
    ),
    "hybrid_resnet_base_down1": ModelProfile(
        profile_id="hybrid_resnet_base_down1",
        base_model="hybrid_resnet",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=1,
        hybrid_resnet_blocks=6,
        evidence_scope="readiness-only",
    ),
    "hybrid_resnet_skip_add": ModelProfile(
        profile_id="hybrid_resnet_skip_add",
        base_model="hybrid_resnet",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        evidence_scope="readiness-only",
    ),
    "hybrid_resnet_cns": ModelProfile(
        profile_id="hybrid_resnet_cns",
        base_model="hybrid_resnet",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="pixelshuffle",
        evidence_scope="benchmark_candidate",
    ),
    "hybrid_resnet_cns_transpose": ModelProfile(
        profile_id="hybrid_resnet_cns_transpose",
        base_model="hybrid_resnet",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="cyclegan_transpose",
        evidence_scope="readiness-only",
    ),
    "hybrid_resnet_interp_bilinear_conv": ModelProfile(
        profile_id="hybrid_resnet_interp_bilinear_conv",
        base_model="hybrid_resnet",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_upsampler="interp_bilinear_conv",
        evidence_scope="readiness-only",
    ),
    "hybrid_resnet_pixelshuffle": ModelProfile(
        profile_id="hybrid_resnet_pixelshuffle",
        base_model="hybrid_resnet",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_upsampler="pixelshuffle",
        evidence_scope="readiness-only",
    ),
    "hybrid_resnet_cns_interp_bilinear_conv": ModelProfile(
        profile_id="hybrid_resnet_cns_interp_bilinear_conv",
        base_model="hybrid_resnet",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="interp_bilinear_conv",
        evidence_scope="readiness-only",
    ),
    "hybrid_resnet_cns_pixelshuffle": ModelProfile(
        profile_id="hybrid_resnet_cns_pixelshuffle",
        base_model="hybrid_resnet",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="pixelshuffle",
        evidence_scope="readiness-only",
    ),
    "spectral_resnet_bottleneck_base": ModelProfile(
        profile_id="spectral_resnet_bottleneck_base",
        base_model="spectral_resnet_bottleneck_net",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="pixelshuffle",
        spectral_bottleneck_blocks=6,
        spectral_bottleneck_modes=12,
        spectral_bottleneck_share_weights=True,
        spectral_bottleneck_gate_init=0.1,
        spectral_bottleneck_gate_mode="shared",
        evidence_scope="manual-only",
    ),
    "spectral_resnet_bottleneck_modes24": ModelProfile(
        profile_id="spectral_resnet_bottleneck_modes24",
        base_model="spectral_resnet_bottleneck_net",
        hidden_channels=32,
        fno_modes=24,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="pixelshuffle",
        spectral_bottleneck_blocks=6,
        spectral_bottleneck_modes=24,
        spectral_bottleneck_share_weights=True,
        spectral_bottleneck_gate_init=0.1,
        spectral_bottleneck_gate_mode="shared",
        evidence_scope="manual-only",
    ),
    "spectral_resnet_bottleneck_modes32": ModelProfile(
        profile_id="spectral_resnet_bottleneck_modes32",
        base_model="spectral_resnet_bottleneck_net",
        hidden_channels=32,
        fno_modes=32,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="pixelshuffle",
        spectral_bottleneck_blocks=6,
        spectral_bottleneck_modes=32,
        spectral_bottleneck_share_weights=True,
        spectral_bottleneck_gate_init=0.1,
        spectral_bottleneck_gate_mode="shared",
        evidence_scope="manual-only",
    ),
    "spectral_resnet_bottleneck_shared_blocks8": ModelProfile(
        profile_id="spectral_resnet_bottleneck_shared_blocks8",
        base_model="spectral_resnet_bottleneck_net",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="pixelshuffle",
        spectral_bottleneck_blocks=8,
        spectral_bottleneck_modes=12,
        spectral_bottleneck_share_weights=True,
        spectral_bottleneck_gate_init=0.1,
        spectral_bottleneck_gate_mode="shared",
        evidence_scope="manual-only",
    ),
    "spectral_resnet_bottleneck_shared_blocks10": ModelProfile(
        profile_id="spectral_resnet_bottleneck_shared_blocks10",
        base_model="spectral_resnet_bottleneck_net",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="pixelshuffle",
        spectral_bottleneck_blocks=10,
        spectral_bottleneck_modes=12,
        spectral_bottleneck_share_weights=True,
        spectral_bottleneck_gate_init=0.1,
        spectral_bottleneck_gate_mode="shared",
        evidence_scope="manual-only",
    ),
    "spectral_resnet_bottleneck_noshare": ModelProfile(
        profile_id="spectral_resnet_bottleneck_noshare",
        base_model="spectral_resnet_bottleneck_net",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="pixelshuffle",
        spectral_bottleneck_blocks=6,
        spectral_bottleneck_modes=12,
        spectral_bottleneck_share_weights=False,
        spectral_bottleneck_gate_init=0.1,
        spectral_bottleneck_gate_mode="shared",
        evidence_scope="manual-only",
    ),
    "spectral_resnet_bottleneck_noshare_blocks8": ModelProfile(
        profile_id="spectral_resnet_bottleneck_noshare_blocks8",
        base_model="spectral_resnet_bottleneck_net",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="pixelshuffle",
        spectral_bottleneck_blocks=8,
        spectral_bottleneck_modes=12,
        spectral_bottleneck_share_weights=False,
        spectral_bottleneck_gate_init=0.1,
        spectral_bottleneck_gate_mode="shared",
        evidence_scope="manual-only",
    ),
    "spectral_resnet_bottleneck_noshare_blocks10": ModelProfile(
        profile_id="spectral_resnet_bottleneck_noshare_blocks10",
        base_model="spectral_resnet_bottleneck_net",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        hybrid_upsampler="pixelshuffle",
        spectral_bottleneck_blocks=10,
        spectral_bottleneck_modes=12,
        spectral_bottleneck_share_weights=False,
        spectral_bottleneck_gate_init=0.1,
        spectral_bottleneck_gate_mode="shared",
        evidence_scope="manual-only",
    ),
    "ffno_bottleneck_base": ModelProfile(
        profile_id="ffno_bottleneck_base",
        base_model="ffno_bottleneck_net",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        ffno_bottleneck_blocks=6,
        ffno_bottleneck_modes=12,
        ffno_bottleneck_share_weights=True,
        ffno_bottleneck_mlp_ratio=2.0,
        ffno_bottleneck_gate_init=0.1,
        ffno_bottleneck_norm="instance",
        evidence_scope="readiness-only",
    ),
    "ffno_bottleneck_localconv_base": ModelProfile(
        profile_id="ffno_bottleneck_localconv_base",
        base_model="ffno_bottleneck_net",
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        hybrid_downsample_steps=2,
        hybrid_resnet_blocks=6,
        hybrid_skip_connections=True,
        hybrid_skip_style="add",
        ffno_bottleneck_blocks=6,
        ffno_bottleneck_modes=12,
        ffno_bottleneck_share_weights=True,
        ffno_bottleneck_mlp_ratio=2.0,
        ffno_bottleneck_gate_init=0.1,
        ffno_bottleneck_norm="instance",
        ffno_bottleneck_local_conv=True,
        ffno_bottleneck_local_conv_kernel_size=3,
        evidence_scope="manual-only",
    ),
    "author_ffno_cns_base": ModelProfile(
        profile_id="author_ffno_cns_base",
        base_model="author_ffno_cns_net",
        hidden_channels=64,
        fno_modes=16,
        fno_blocks=24,
        author_ffno_width=64,
        author_ffno_modes=16,
        author_ffno_layers=24,
        author_ffno_share_weight=True,
        author_ffno_factor=4,
        author_ffno_ff_weight_norm=True,
        author_ffno_n_ff_layers=2,
        author_ffno_gain=0.1,
        author_ffno_dropout=0.0,
        author_ffno_in_dropout=0.0,
        author_ffno_layer_norm=False,
        author_ffno_use_position=True,
        author_ffno_mode="full",
        evidence_scope="readiness-only",
    ),
    "gnot_cns_base": ModelProfile(
        profile_id="gnot_cns_base",
        base_model="gnot_cns_net",
        hidden_channels=128,
        gnot_hidden=128,
        gnot_layers=3,
        gnot_heads=1,
        gnot_experts=1,
        gnot_inner_multiplier=4,
        gnot_mlp_layers=3,
        gnot_attn_type="linear",
        training_loss="relative_l2",
        optimizer_name="AdamW",
        learning_rate=1e-3,
        weight_decay=5e-5,
        scheduler_name="OneCycleLR",
        scheduler_pct_start=0.2,
        scheduler_div_factor=1e4,
        scheduler_final_div_factor=1e4,
        evidence_scope="readiness-only",
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
PRIMARY_CFD_CNS_PROFILE_IDS = ["hybrid_resnet_cns", "fno_base", "unet_strong"]
READINESS_CFD_CNS_PROFILE_IDS = ["hybrid_resnet_cns", "fno_base", "unet_tiny_smoke"]


def required_primary_profiles_for_task(task_id: str) -> list[str]:
    if str(task_id) == "2d_cfd_cns":
        return list(PRIMARY_CFD_CNS_PROFILE_IDS)
    if str(task_id) == "darcy":
        return list(PRIMARY_DARCY_PROFILE_IDS)
    return ["hybrid_resnet_base", "fno_base", "unet_strong"]


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
    budget["loss"] = str(budget.get("loss", "relative_l2")).lower()
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
    budget["plateau_min_lr"] = float(budget.get("plateau_min_lr", 1e-5))
    if budget["plateau_min_lr"] < 0.0:
        raise ValueError("run budget plateau_min_lr must be non-negative")
    if budget["plateau_min_lr"] > 1e-5:
        raise ValueError("run budget plateau_min_lr must be no higher than 1e-5 for PDE studies")
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
            "loss": "relative_l2",
            "loss_rationale": "Aligns Darcy optimization with the relative L2/nRMSE benchmark metric.",
            "optimizer": "adam",
            "learning_rate": 2e-4,
            "scheduler": "ReduceLROnPlateau",
            "plateau_factor": 0.5,
            "plateau_patience": 2,
            "plateau_min_lr": 1e-5,
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
