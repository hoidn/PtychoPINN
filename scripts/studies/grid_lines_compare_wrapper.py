#!/usr/bin/env python3
"""Orchestrate TF grid-lines workflow + Torch runners and merge metrics."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import os
import random
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptycho.image.harmonize import resize_complex_to_shape
from ptycho.workflows.grid_lines_workflow import GridLinesConfig
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig
from scripts.studies.paper_provenance import (
    merge_git_provenance,
    merge_runtime_provenance,
    relative_to_output_dir,
    write_dataset_identity_manifest,
    write_exit_code_proof,
    write_launcher_completion_evidence,
    write_split_manifest,
)


LEGACY_ARCH_TO_MODEL = {
    "cnn": "pinn",
    "baseline": "baseline",
    "ffno": "pinn_ffno",
    "fno": "pinn_fno",
    "hybrid": "pinn_hybrid",
    "stable_hybrid": "pinn_stable_hybrid",
    "fno_vanilla": "pinn_fno_vanilla",
    "hybrid_resnet": "pinn_hybrid_resnet",
    "spectral_resnet_bottleneck_net": "pinn_spectral_resnet_bottleneck_net",
}

MODEL_TO_LEGACY_ARCH = {model_id: arch for arch, model_id in LEGACY_ARCH_TO_MODEL.items()}
MODEL_TO_LEGACY_ARCH["supervised_ffno"] = "ffno"
MODEL_TO_LEGACY_ARCH["supervised_neuralop_uno"] = "neuralop_uno"
MODEL_TO_LEGACY_ARCH["pinn_neuralop_uno"] = "neuralop_uno"
MODEL_DEFAULT_N = {"pinn_ptychovit": 256}
TF_MODEL_IDS = {"pinn", "baseline"}
TORCH_MODEL_IDS = {
    "supervised_ffno",
    "pinn_ffno",
    "pinn_fno",
    "pinn_hybrid",
    "pinn_stable_hybrid",
    "pinn_fno_vanilla",
    "pinn_hybrid_resnet",
    "pinn_hybrid_resnet_ffno_ptychoblock_encoder",
    "pinn_hybrid_resnet_encoder_conv_only",
    "pinn_hybrid_resnet_encoder_spectral_only",
    "supervised_hybrid_resnet",
    "pinn_hybrid_resnet_convnext_bottleneck",
    "pinn_spectral_resnet_bottleneck_net",
    "pinn_spectral_resnet_bottleneck_ds1",
    "pinn_spectral_resnet_bottleneck_linear_decoder",
    "pinn_hybrid_resnet_ffno_bottleneck",
    "pinn_neuralop_uno",
    "supervised_neuralop_uno",
}
SUPPORTED_MODEL_IDS = set(LEGACY_ARCH_TO_MODEL.values()) | TORCH_MODEL_IDS | {"pinn_ptychovit"}
PAPER_MODEL_LABELS = {
    "baseline": "CDI CNN + supervised",
    "pinn": "CDI CNN + PINN",
    "supervised_ffno": "FFNO + supervised",
    "pinn_ffno": "FFNO + PINN",
    "pinn_fno": "FNO + PINN",
    "pinn_hybrid": "Hybrid + PINN",
    "pinn_stable_hybrid": "Stable Hybrid + PINN",
    "pinn_fno_vanilla": "FNO Vanilla + PINN",
    "pinn_hybrid_resnet": "Hybrid ResNet + PINN",
    "pinn_hybrid_resnet_ffno_ptychoblock_encoder": "Hybrid ResNet (FFNO->PtychoBlock encoder) + PINN",
    "pinn_hybrid_resnet_encoder_conv_only": "Hybrid ResNet (conv-only encoder) + PINN",
    "pinn_hybrid_resnet_encoder_spectral_only": "Hybrid ResNet (spectral-only encoder) + PINN",
    "supervised_hybrid_resnet": "Hybrid ResNet + supervised",
    "pinn_hybrid_resnet_convnext_bottleneck": "Hybrid ResNet (ConvNeXt bottleneck) + PINN",
    "pinn_spectral_resnet_bottleneck_net": "Spectral ResNet Bottleneck + PINN",
    "pinn_spectral_resnet_bottleneck_ds1": "Spectral ResNet Bottleneck DS1 + PINN",
    "pinn_spectral_resnet_bottleneck_linear_decoder": "Spectral ResNet Linear Decoder + PINN",
    "pinn_hybrid_resnet_ffno_bottleneck": "Hybrid ResNet FFNO Bottleneck + PINN",
    "pinn_neuralop_uno": "U-NO + PINN",
    "supervised_neuralop_uno": "U-NO + supervised",
    "pinn_ptychovit": "PtychoViT + PINN",
}
PAPER_ARCHITECTURE_OVERRIDES = {
    "baseline": "cnn",
    "pinn": "cnn",
}
PAPER_TRAINING_PROCEDURE_OVERRIDES = {
    "baseline": "supervised",
    "supervised_ffno": "supervised",
    "supervised_neuralop_uno": "supervised",
    "supervised_hybrid_resnet": "supervised",
}

DEFAULT_TORCH_ROW_SPECS: Dict[str, Dict[str, Any]] = {
    "supervised_ffno": {
        "model_id": "supervised_ffno",
        "architecture": "ffno",
        "training_procedure": "supervised",
    },
    "pinn_ffno": {
        "model_id": "pinn_ffno",
        "architecture": "ffno",
        "training_procedure": "pinn",
    },
    "pinn_fno": {
        "model_id": "pinn_fno",
        "architecture": "fno",
        "training_procedure": "pinn",
    },
    "pinn_hybrid": {
        "model_id": "pinn_hybrid",
        "architecture": "hybrid",
        "training_procedure": "pinn",
    },
    "pinn_stable_hybrid": {
        "model_id": "pinn_stable_hybrid",
        "architecture": "stable_hybrid",
        "training_procedure": "pinn",
    },
    "pinn_fno_vanilla": {
        "model_id": "pinn_fno_vanilla",
        "architecture": "fno_vanilla",
        "training_procedure": "pinn",
    },
    "pinn_hybrid_resnet": {
        "model_id": "pinn_hybrid_resnet",
        "architecture": "hybrid_resnet",
        "training_procedure": "pinn",
    },
    "pinn_hybrid_resnet_ffno_ptychoblock_encoder": {
        "model_id": "pinn_hybrid_resnet_ffno_ptychoblock_encoder",
        "architecture": "hybrid_resnet_ffno_ptychoblock_encoder",
        "training_procedure": "pinn",
        "row_status": "decision_support_append_only",
        "lock_row_status": True,
    },
    "pinn_hybrid_resnet_encoder_conv_only": {
        "model_id": "pinn_hybrid_resnet_encoder_conv_only",
        "architecture": "hybrid_resnet",
        "training_procedure": "pinn",
        "overrides": {"hybrid_encoder_branch_select": "conv_only"},
        "row_status": "decision_support_append_only",
        "lock_row_status": True,
    },
    "pinn_hybrid_resnet_encoder_spectral_only": {
        "model_id": "pinn_hybrid_resnet_encoder_spectral_only",
        "architecture": "hybrid_resnet",
        "training_procedure": "pinn",
        "overrides": {"hybrid_encoder_branch_select": "spectral_only"},
        "row_status": "decision_support_append_only",
        "lock_row_status": True,
    },
    "supervised_hybrid_resnet": {
        "model_id": "supervised_hybrid_resnet",
        "architecture": "hybrid_resnet",
        "training_procedure": "supervised",
        "row_status": "decision_support_append_only",
        "lock_row_status": True,
    },
    "pinn_hybrid_resnet_convnext_bottleneck": {
        "model_id": "pinn_hybrid_resnet_convnext_bottleneck",
        "architecture": "hybrid_resnet_convnext_bottleneck",
        "training_procedure": "pinn",
        "row_status": "decision_support_append_only",
        "lock_row_status": True,
    },
    "pinn_spectral_resnet_bottleneck_net": {
        "model_id": "pinn_spectral_resnet_bottleneck_net",
        "architecture": "spectral_resnet_bottleneck_net",
        "training_procedure": "pinn",
    },
    "pinn_spectral_resnet_bottleneck_ds1": {
        "model_id": "pinn_spectral_resnet_bottleneck_ds1",
        "architecture": "spectral_resnet_bottleneck_net",
        "training_procedure": "pinn",
        "overrides": {"hybrid_downsample_steps": 1},
    },
    "pinn_spectral_resnet_bottleneck_linear_decoder": {
        "model_id": "pinn_spectral_resnet_bottleneck_linear_decoder",
        "architecture": "spectral_resnet_bottleneck_linear_decoder",
        "training_procedure": "pinn",
    },
    "pinn_hybrid_resnet_ffno_bottleneck": {
        "model_id": "pinn_hybrid_resnet_ffno_bottleneck",
        "architecture": "hybrid_resnet_ffno_bottleneck",
        "training_procedure": "pinn",
    },
    "pinn_neuralop_uno": {
        "model_id": "pinn_neuralop_uno",
        "architecture": "neuralop_uno",
        "training_procedure": "pinn",
    },
    "supervised_neuralop_uno": {
        "model_id": "supervised_neuralop_uno",
        "architecture": "neuralop_uno",
        "training_procedure": "supervised",
    },
}
MODEL_TO_LEGACY_ARCH.update(
    {
        model_id: str(spec["architecture"])
        for model_id, spec in DEFAULT_TORCH_ROW_SPECS.items()
        if model_id not in MODEL_TO_LEGACY_ARCH
    }
)


def _parse_architectures(value: str) -> Tuple[str, ...]:
    return tuple(a.strip() for a in value.split(",") if a.strip())


def _normalize_row_specs(
    row_specs: Optional[Iterable[Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    normalized = {model_id: dict(spec) for model_id, spec in DEFAULT_TORCH_ROW_SPECS.items()}
    if not row_specs:
        return normalized
    for spec in row_specs:
        spec_dict = dict(spec)
        model_id = str(spec_dict.get("model_id", "")).strip()
        architecture = str(spec_dict.get("architecture", "")).strip()
        if not model_id or not architecture:
            raise ValueError("row_specs entries require non-empty model_id and architecture")
        spec_dict["model_id"] = model_id
        spec_dict["architecture"] = architecture
        spec_dict["training_procedure"] = str(spec_dict.get("training_procedure", "pinn"))
        overrides = spec_dict.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, Mapping):
            raise ValueError(f"row_specs overrides must be a mapping for {model_id}")
        spec_dict["overrides"] = dict(overrides)
        normalized[model_id] = spec_dict
    return normalized


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    with candidate.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, dict) else {}


class _TeeTextStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _write_log_text(path: Path, text: str, *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


@contextlib.contextmanager
def _capture_execution_logs(
    *,
    stdout_overwrite_targets: Iterable[Path],
    stderr_overwrite_targets: Iterable[Path],
    stdout_append_targets: Iterable[Path] = (),
    stderr_append_targets: Iterable[Path] = (),
):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    tee_stdout = _TeeTextStream(sys.stdout, stdout_buffer)
    tee_stderr = _TeeTextStream(sys.stderr, stderr_buffer)
    try:
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            yield
    finally:
        stdout_text = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()
        for target in stdout_overwrite_targets:
            _write_log_text(Path(target), stdout_text, append=False)
        for target in stderr_overwrite_targets:
            _write_log_text(Path(target), stderr_text, append=False)
        for target in stdout_append_targets:
            _write_log_text(Path(target), stdout_text, append=True)
        for target in stderr_append_targets:
            _write_log_text(Path(target), stderr_text, append=True)


def _current_torch_hardware_summary() -> Dict[str, object]:
    try:
        import torch
    except Exception:
        return {"backend": "pytorch", "accelerator": "unknown"}

    accelerator = "cpu"
    if torch.cuda.is_available():
        try:
            accelerator = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            accelerator = "cuda"
    else:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            accelerator = "mps"
    return {
        "backend": "pytorch",
        "accelerator": accelerator,
    }


def _current_tf_hardware_summary() -> Dict[str, object]:
    try:
        import tensorflow as tf
    except Exception:
        return {"backend": "tensorflow", "accelerator": "unknown"}

    gpus = tf.config.list_physical_devices("GPU")
    accelerator = "cpu"
    if gpus:
        accelerator = getattr(gpus[0], "name", None) or "gpu"
    return {
        "backend": "tensorflow",
        "accelerator": accelerator,
    }


def _parse_wall_time_seconds(payload: Mapping[str, Any]) -> Optional[float]:
    started = payload.get("started_at_utc") or payload.get("timestamp_utc")
    finished = payload.get("finished_at_utc")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        start_dt = datetime.fromisoformat(started)
        finish_dt = datetime.fromisoformat(finished)
    except ValueError:
        return None
    return max(0.0, (finish_dt - start_dt).total_seconds())


def _invocation_output_dir_matches_current_root(
    invocation_payload: Mapping[str, Any],
    *,
    current_output_dir: Path,
) -> bool:
    parsed_args = invocation_payload.get("parsed_args")
    if not isinstance(parsed_args, Mapping):
        return False
    raw_output_dir = parsed_args.get("output_dir")
    if not isinstance(raw_output_dir, str) or not raw_output_dir:
        return False
    invocation_output_dir = Path(raw_output_dir)
    if not invocation_output_dir.is_absolute():
        invocation_output_dir = (REPO_ROOT / invocation_output_dir).resolve()
    else:
        invocation_output_dir = invocation_output_dir.resolve()
    return invocation_output_dir == Path(current_output_dir).resolve()


def _recovered_runtime_summary(
    invocation_payload: Mapping[str, Any],
    *,
    current_output_dir: Path,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "recovered_from_existing_artifacts": True,
    }
    if _invocation_output_dir_matches_current_root(
        invocation_payload,
        current_output_dir=current_output_dir,
    ):
        summary["recovered_from_existing_artifacts"] = False
        summary["row_payload_rebuilt_from_row_artifacts"] = True
    wall_time = _parse_wall_time_seconds(invocation_payload)
    if wall_time is not None:
        summary["command_wall_time_sec"] = float(wall_time)
    return summary


def _recovered_tf_runtime_summary(invocation_payload: Mapping[str, Any]) -> Dict[str, object]:
    summary: Dict[str, object] = {"recovered_from_existing_artifacts": True}
    extra = invocation_payload.get("extra")
    if isinstance(extra, Mapping) and extra.get("invocation_mode") == "library":
        summary["runtime_source"] = "unavailable_under_recovery"
        summary["runtime_unavailable_reason"] = (
            "Recovered TensorFlow row invocation records wrapper library completion, "
            "not standalone training runtime; wrapper bundle-repair pass wall time is excluded."
        )
        return summary

    wall_time = _parse_wall_time_seconds(invocation_payload)
    if wall_time is not None:
        summary["runtime_source"] = "row_invocation"
        summary["command_wall_time_sec"] = float(wall_time)
        return summary

    summary["runtime_source"] = "unavailable_under_recovery"
    summary["runtime_unavailable_reason"] = (
        "Recovered TensorFlow row invocation did not provide standalone runtime timestamps."
    )
    return summary


def _count_torch_state_dict_parameters(model_path: Path) -> Optional[int]:
    try:
        import torch
    except Exception:
        return None

    path = Path(model_path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("state_dict") if isinstance(payload, Mapping) and "state_dict" in payload else payload
    if not isinstance(state_dict, Mapping):
        return None
    total = 0
    for tensor in state_dict.values():
        numel = getattr(tensor, "numel", None)
        if callable(numel):
            total += int(numel())
    return total or None


def _recover_tf_final_train_loss(log_text: str, model_id: str) -> Optional[float]:
    if model_id == "pinn":
        matches = re.findall(r"loss:\s*([0-9.eE+-]+)\s*-\s*pred_intensity_loss", log_text)
    else:
        matches = re.findall(
            r"conv2d_[^\n]*?-\s*loss:\s*([0-9.eE+-]+)(?:\s*-\s*val_|$)",
            log_text,
        )
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _history_series(payload: Mapping[str, Any], *keys: str) -> List[float]:
    for key in keys:
        values = payload.get(key)
        if isinstance(values, list) and values:
            try:
                return [float(value) for value in values]
            except Exception:
                return []
    return []


def _load_tf_parameter_count(output_dir: Path, model_id: str) -> Optional[int]:
    if model_id == "baseline":
        model_path = output_dir / "baseline" / "baseline.keras"
        if not model_path.exists():
            return None
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
        count_params = getattr(model, "count_params", None)
        return int(count_params()) if callable(count_params) else None

    archive_zip = output_dir / "pinn" / "wts.h5.zip"
    if not archive_zip.exists():
        return None
    from ptycho import model_manager

    loaded = model_manager.ModelManager.load_multiple_models(
        str(archive_zip.with_suffix("")),
        ["autoencoder"],
    )
    model = loaded.get("autoencoder") if isinstance(loaded, Mapping) else None
    if model is None and isinstance(loaded, Mapping) and loaded:
        model = next(iter(loaded.values()))
    count_params = getattr(model, "count_params", None)
    return int(count_params()) if callable(count_params) else None


def _recover_tf_row_payload(
    *,
    output_dir: Path,
    model_id: str,
    n_value: int,
    epoch_budget: int,
    metrics: Mapping[str, object],
) -> Dict[str, object]:
    log_text = ""
    log_path = output_dir / "live_stdout.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    invocation_payload = _load_json_if_exists(output_dir / "runs" / model_id / "invocation.json")
    history_payload = _load_json_if_exists(output_dir / "runs" / model_id / "history.json")
    loss_series = _history_series(history_payload, "loss")
    val_loss = history_payload.get("val_loss", [])
    return {
        "model_label": PAPER_MODEL_LABELS.get(model_id, model_id),
        "architecture_id": PAPER_ARCHITECTURE_OVERRIDES.get(model_id, MODEL_TO_LEGACY_ARCH.get(model_id, model_id)),
        "training_procedure": PAPER_TRAINING_PROCEDURE_OVERRIDES.get(model_id, "pinn"),
        "N": int(n_value),
        "parameter_count": _load_tf_parameter_count(output_dir, model_id),
        "epoch_budget": int(epoch_budget),
        "final_completed_epoch": int(len(loss_series) or epoch_budget),
        "final_train_loss": (
            float(loss_series[-1])
            if loss_series
            else _recover_tf_final_train_loss(log_text, model_id)
        ),
        "validation_loss": {
            "status": "emitted" if isinstance(val_loss, list) and val_loss else "no_validation_series",
            "value": float(val_loss[-1]) if isinstance(val_loss, list) and val_loss else None,
        },
        "runtime_summary": _recovered_tf_runtime_summary(invocation_payload),
        "hardware_summary": _current_tf_hardware_summary(),
        "row_status": "decision_support",
        "caveats": ["recovered_from_existing_artifacts"],
        "metrics": dict(metrics),
    }


def _recover_torch_row_payload(
    *,
    output_dir: Path,
    model_id: str,
    n_value: int,
    metrics: Mapping[str, object],
) -> Dict[str, object]:
    run_dir = output_dir / "runs" / model_id
    history_payload = _load_json_if_exists(run_dir / "history.json")
    train_loss = history_payload.get("train_loss", [])
    val_loss = history_payload.get("val_loss", [])
    invocation_payload = _load_json_if_exists(run_dir / "invocation.json")
    runtime_summary = _recovered_runtime_summary(
        invocation_payload,
        current_output_dir=output_dir,
    )
    caveats = (
        ["recovered_from_existing_artifacts"]
        if runtime_summary.get("recovered_from_existing_artifacts", True)
        else ["row_payload_rebuilt_from_row_artifacts"]
    )
    parsed_args = invocation_payload.get("parsed_args", {})
    epoch_budget = parsed_args.get("epochs", len(train_loss) or None)
    final_train_loss = float(train_loss[-1]) if isinstance(train_loss, list) and train_loss else None
    return {
        "model_label": PAPER_MODEL_LABELS.get(model_id, model_id),
        "architecture_id": PAPER_ARCHITECTURE_OVERRIDES.get(model_id, MODEL_TO_LEGACY_ARCH.get(model_id, model_id)),
        "training_procedure": PAPER_TRAINING_PROCEDURE_OVERRIDES.get(model_id, "pinn"),
        "N": int(n_value),
        "parameter_count": _count_torch_state_dict_parameters(run_dir / "model.pt"),
        "epoch_budget": int(epoch_budget) if epoch_budget is not None else None,
        "final_completed_epoch": int(len(train_loss) or epoch_budget or 0),
        "final_train_loss": final_train_loss,
        "validation_loss": {
            "status": "emitted" if isinstance(val_loss, list) and val_loss else "no_validation_series",
            "value": float(val_loss[-1]) if isinstance(val_loss, list) and val_loss else None,
        },
        "runtime_summary": runtime_summary,
        "hardware_summary": _current_torch_hardware_summary(),
        "row_status": "decision_support",
        "caveats": caveats,
        "metrics": dict(metrics),
    }


def _default_paper_row_payload(model_id: str, *, n_value: int) -> Dict[str, object]:
    architecture_id = PAPER_ARCHITECTURE_OVERRIDES.get(model_id, MODEL_TO_LEGACY_ARCH.get(model_id, model_id))
    training_procedure = PAPER_TRAINING_PROCEDURE_OVERRIDES.get(model_id, "pinn")
    return {
        "model_label": PAPER_MODEL_LABELS.get(model_id, model_id),
        "architecture_id": architecture_id,
        "training_procedure": training_procedure,
        "N": int(n_value),
        "parameter_count": None,
        "epoch_budget": None,
        "final_completed_epoch": None,
        "final_train_loss": None,
        "validation_loss": {"status": "not_emitted", "value": None},
        "runtime_summary": None,
        "hardware_summary": None,
        "row_status": None,
        "caveats": [],
        "invocation": None,
        "config": None,
        "git": None,
        "environment": None,
        "dataset": None,
        "splits": None,
        "randomness": None,
        "outputs": None,
        "visuals": None,
        "metrics": {},
    }


def _coerce_paper_row_payload(
    model_id: str,
    payload: object,
    *,
    n_value: int,
    metrics: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    normalized = _default_paper_row_payload(model_id, n_value=n_value)
    if isinstance(payload, Mapping):
        normalized.update(dict(payload))
    n_payload = normalized.get("N")
    normalized["N"] = int(n_value if n_payload is None else n_payload)
    normalized["model_label"] = str(normalized.get("model_label") or PAPER_MODEL_LABELS.get(model_id, model_id))
    normalized["architecture_id"] = str(
        normalized.get("architecture_id") or PAPER_ARCHITECTURE_OVERRIDES.get(model_id, MODEL_TO_LEGACY_ARCH.get(model_id, model_id))
    )
    normalized["training_procedure"] = str(
        normalized.get("training_procedure") or PAPER_TRAINING_PROCEDURE_OVERRIDES.get(model_id, "pinn")
    )
    caveats = normalized.get("caveats")
    normalized["caveats"] = list(caveats) if isinstance(caveats, list) else []
    if metrics is not None:
        normalized["metrics"] = dict(metrics)
    elif not isinstance(normalized.get("metrics"), Mapping):
        normalized["metrics"] = {}
    return normalized


def _maybe_write_recovered_torch_config(
    *,
    output_dir: Path,
    model_id: str,
    invocation_payload: Mapping[str, Any],
) -> Path:
    run_dir = output_dir / "runs" / model_id
    config_path = run_dir / "config.json"
    if config_path.exists():
        return config_path
    parsed_args = invocation_payload.get("parsed_args", {})
    payload = {
        "torch_runner_config": parsed_args,
        "model_label": PAPER_MODEL_LABELS.get(model_id, model_id),
        "training_procedure": PAPER_TRAINING_PROCEDURE_OVERRIDES.get(model_id, "pinn"),
        "train_npz": parsed_args.get("train_npz"),
        "test_npz": parsed_args.get("test_npz"),
        "recon_npz": relative_to_output_dir(output_dir, output_dir / "recons" / model_id / "recon.npz"),
    }
    config_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return config_path


def _maybe_materialize_recovered_torch_row_logs(
    *,
    output_dir: Path,
    model_id: str,
    invocation_payload: Mapping[str, Any],
) -> bool:
    run_dir = output_dir / "runs" / model_id
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    if stdout_log.exists() and stderr_log.exists():
        return False
    if not _invocation_output_dir_matches_current_root(
        invocation_payload,
        current_output_dir=output_dir,
    ):
        return False
    if invocation_payload.get("status") != "completed" or invocation_payload.get("exit_code") != 0:
        return False

    timestamp = invocation_payload.get("finished_at_utc") or invocation_payload.get("timestamp_utc") or "unknown"
    note_lines = [
        "Recovered row log placeholder generated during compare-wrapper bundle repair.",
        "Original per-row direct-runner stdout/stderr logs were not captured.",
        f"model_id={model_id}",
        f"timestamp_utc={timestamp}",
    ]
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    if not stdout_log.exists():
        stdout_log.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    if not stderr_log.exists():
        stderr_log.write_text(
            "Recovered row stderr placeholder generated during compare-wrapper bundle repair.\n",
            encoding="utf-8",
        )
    return True


def _enrich_paper_row_payload(
    *,
    model_id: str,
    payload: Mapping[str, Any],
    output_dir: Path,
    train_npz: Path,
    test_npz: Path,
    seed: int,
    nimgs_train: int,
    nimgs_test: int,
    gridsize: int,
    set_phi: bool,
    probe_npz: Path,
    dataset_source: str,
    probe_source: str,
    probe_scale_mode: str,
    row_spec: Optional[Mapping[str, Any]] = None,
) -> Dict[str, object]:
    row = dict(payload)
    run_dir = output_dir / "runs" / model_id
    invocation_path = run_dir / "invocation.json"
    invocation_payload = _load_json_if_exists(invocation_path)
    if model_id in TORCH_MODEL_IDS and invocation_payload:
        _maybe_write_recovered_torch_config(
            output_dir=output_dir,
            model_id=model_id,
            invocation_payload=invocation_payload,
        )
        if _maybe_materialize_recovered_torch_row_logs(
            output_dir=output_dir,
            model_id=model_id,
            invocation_payload=invocation_payload,
        ):
            caveats = list(row.get("caveats")) if isinstance(row.get("caveats"), list) else []
            if "row_output_logs_recovered_from_invocation" not in caveats:
                caveats.append("row_output_logs_recovered_from_invocation")
            row["caveats"] = caveats
    config_path = run_dir / "config.json"
    if not row.get("invocation") and invocation_path.exists():
        row["invocation"] = {
            "json": relative_to_output_dir(output_dir, invocation_path),
            "shell": relative_to_output_dir(output_dir, invocation_path.with_suffix(".sh")),
        }
    if not row.get("config") and config_path.exists():
        row["config"] = {
            "json": relative_to_output_dir(output_dir, config_path),
        }
    extra = invocation_payload.get("extra", {}) if isinstance(invocation_payload, Mapping) else {}
    runtime = extra.get("runtime_provenance", {})
    git_commit = extra.get("git_commit")
    row["environment"] = merge_runtime_provenance(
        row.get("environment") if row.get("environment") else runtime,
        hardware_summary=row.get("hardware_summary") if isinstance(row.get("hardware_summary"), Mapping) else None,
    )
    row["git"] = merge_git_provenance(
        row.get("git"),
        repo_root=REPO_ROOT,
        commit=str(git_commit) if isinstance(git_commit, str) and git_commit else None,
        note_source="recovered_bundle_write",
    )
    dataset_manifest = write_dataset_identity_manifest(
        output_dir,
        train_npz=train_npz,
        test_npz=test_npz,
        dataset_source=dataset_source,
        probe_npz=probe_npz,
        probe_source=probe_source,
        probe_scale_mode=probe_scale_mode,
    )
    dataset_payload = dict(row.get("dataset")) if isinstance(row.get("dataset"), Mapping) else {}
    dataset_payload.update(
        {
            "train_npz": str(train_npz),
            "test_npz": str(test_npz),
            "probe_npz": str(probe_npz),
            "probe_source": probe_source,
            "probe_scale_mode": probe_scale_mode,
            "dataset_source": dataset_source,
            "manifest_json": relative_to_output_dir(output_dir, dataset_manifest),
        }
    )
    row["dataset"] = dataset_payload
    split_manifest = write_split_manifest(
        output_dir,
        train_npz=train_npz,
        test_npz=test_npz,
        seed=seed,
        nimgs_train=nimgs_train,
        nimgs_test=nimgs_test,
        gridsize=gridsize,
        set_phi=set_phi,
    )
    split_payload = dict(row.get("splits")) if isinstance(row.get("splits"), Mapping) else {}
    split_payload.update(
        {
            "nimgs_train": int(nimgs_train),
            "nimgs_test": int(nimgs_test),
            "gridsize": int(gridsize),
            "set_phi": bool(set_phi),
            "seed": int(seed),
            "manifest_json": relative_to_output_dir(output_dir, split_manifest),
        }
    )
    row["splits"] = split_payload
    randomness = row.get("randomness")
    if not isinstance(randomness, Mapping) or not randomness:
        randomness_path = run_dir / "randomness_contract.json"
        if randomness_path.exists():
            row["randomness"] = _load_json_if_exists(randomness_path)
        else:
            row["randomness"] = {"seed": int(seed), "requested_seed": int(seed)}
    outputs_payload = dict(row.get("outputs")) if isinstance(row.get("outputs"), Mapping) else {}
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    exit_code_proof = write_exit_code_proof(
        output_dir,
        model_id=model_id,
        invocation_json=invocation_path if invocation_path.exists() else None,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        proof_source="row_artifacts_present_after_successful_compare_flow",
    )
    outputs_payload.update(
        {
            "metrics_json": relative_to_output_dir(output_dir, run_dir / "metrics.json"),
            "history_json": relative_to_output_dir(output_dir, run_dir / "history.json"),
            "recon_npz": relative_to_output_dir(output_dir, output_dir / "recons" / model_id / "recon.npz"),
            "stdout_log": relative_to_output_dir(output_dir, stdout_log),
            "stderr_log": relative_to_output_dir(output_dir, stderr_log),
        }
    )
    if exit_code_proof is not None:
        outputs_payload["exit_code_proof_json"] = relative_to_output_dir(output_dir, exit_code_proof)
    launcher_completion = None
    if model_id in TORCH_MODEL_IDS:
        launcher_completion = write_launcher_completion_evidence(
            output_dir,
            model_id=model_id,
            wrapper_invocation_json=output_dir / "invocation.json",
            launcher_stderr_log=output_dir / "launcher_stderr.log",
            launcher_stdout_log=output_dir / "launcher_stdout.log",
        )
    if launcher_completion is not None:
        outputs_payload["launcher_completion_json"] = relative_to_output_dir(output_dir, launcher_completion)
    row["outputs"] = outputs_payload
    if not row.get("visuals"):
        row["visuals"] = {
            "amp_phase_png": f"visuals/amp_phase_{model_id}.png",
            "amp_phase_error_png": f"visuals/amp_phase_error_{model_id}.png",
        }
    locked_row_status: Optional[str] = None
    if isinstance(row_spec, Mapping):
        candidate_row_status = row_spec.get("row_status")
        if isinstance(candidate_row_status, str) and candidate_row_status.strip():
            locked_row_status = candidate_row_status.strip()
            row["row_status"] = locked_row_status
    lock_row_status = bool(row_spec.get("lock_row_status", False)) if isinstance(row_spec, Mapping) else False
    if not lock_row_status and row.get("row_status") in {None, "decision_support"}:
        required_paths = [
            invocation_path,
            run_dir / "invocation.sh",
            run_dir / "config.json",
            run_dir / "history.json",
            run_dir / "metrics.json",
            run_dir / "stdout.log",
            run_dir / "stderr.log",
            output_dir / "recons" / model_id / "recon.npz",
        ]
        if exit_code_proof is not None:
            required_paths.append(exit_code_proof)
        visuals = row.get("visuals")
        if isinstance(visuals, Mapping):
            for key in ("amp_phase_png", "amp_phase_error_png"):
                relative_path = visuals.get(key)
                if isinstance(relative_path, str) and relative_path:
                    required_paths.append(output_dir / relative_path)
        if all(path.exists() for path in required_paths):
            row["row_status"] = "paper_grade"
    elif lock_row_status and locked_row_status is not None:
        row["row_status"] = locked_row_status
    return row


def _recover_existing_dataset_paths(output_dir: Path) -> Tuple[Path, Path]:
    dataset_candidates = sorted((output_dir / "datasets").glob("N*/gs*/train.npz"))
    if dataset_candidates:
        train_npz = dataset_candidates[0]
        return train_npz, train_npz.with_name("test.npz")
    for run_dir in sorted((output_dir / "runs").glob("*")):
        config_payload = _load_json_if_exists(run_dir / "config.json")
        train_npz = config_payload.get("train_npz")
        test_npz = config_payload.get("test_npz")
        if isinstance(train_npz, str) and isinstance(test_npz, str):
            return Path(train_npz), Path(test_npz)
        invocation_payload = _load_json_if_exists(run_dir / "invocation.json")
        parsed_args = invocation_payload.get("parsed_args", {}) if isinstance(invocation_payload, Mapping) else {}
        train_npz = parsed_args.get("train_npz")
        test_npz = parsed_args.get("test_npz")
        if isinstance(train_npz, str) and isinstance(test_npz, str):
            return Path(train_npz), Path(test_npz)
    return output_dir / "train.npz", output_dir / "test.npz"


def _parse_models(value: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def _torch_model_route(
    model_id: str,
    row_specs_by_model: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[str, str]:
    spec = (row_specs_by_model or {}).get(model_id)
    if spec is None:
        spec = DEFAULT_TORCH_ROW_SPECS.get(model_id)
    if spec is None:
        raise ValueError(f"Unsupported torch model route for {model_id!r}")
    training_procedure = str(spec.get("training_procedure", PAPER_TRAINING_PROCEDURE_OVERRIDES.get(model_id, "pinn")))
    return str(spec["architecture"]), training_procedure


def _parse_model_n(value: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not value:
        return out
    for chunk in value.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --model-n entry '{item}'; expected model_id=N")
        name, raw_n = item.split("=", 1)
        out[name.strip()] = int(raw_n)
    return out


def _run_tf_workflow_with_selected_models(tf_workflow_module, cfg: GridLinesConfig, tf_models: Tuple[str, ...]):
    """Run TF workflow with explicit model selection when supported.

    Test doubles may still expose the legacy single-arg signature; keep a
    compatibility fallback for those call sites.
    """
    run_fn = tf_workflow_module.run_grid_lines_workflow
    if "tf_models" in inspect.signature(run_fn).parameters:
        return run_fn(cfg, tf_models=tf_models)
    return run_fn(cfg)


def validate_model_specs(
    models: Tuple[str, ...],
    model_n: Dict[str, int],
    row_specs_by_model: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    supported_model_ids = set(SUPPORTED_MODEL_IDS) | set((row_specs_by_model or {}).keys())
    for model_id in models:
        if model_id not in supported_model_ids:
            raise ValueError(f"Unsupported model '{model_id}'")
    for model_id, n_value in model_n.items():
        if model_id not in supported_model_ids:
            raise ValueError(f"Unsupported model '{model_id}' in --model-n")
        if n_value <= 0:
            raise ValueError(f"Invalid N for model '{model_id}': {n_value}")
    if "pinn_ptychovit" in models and model_n.get("pinn_ptychovit", 256) != 256:
        raise ValueError("pinn_ptychovit currently supports only N=256")


def compute_required_ns(models: Tuple[str, ...], model_n: Dict[str, int], default_n: int) -> list[int]:
    return sorted({model_n.get(model_id, default_n) for model_id in models})


def resolve_model_ns(
    models: Tuple[str, ...],
    model_n_overrides: Dict[str, int],
    default_n: int,
) -> Dict[str, int]:
    resolved: Dict[str, int] = {}
    for model_id in models:
        resolved[model_id] = int(model_n_overrides.get(model_id, MODEL_DEFAULT_N.get(model_id, default_n)))
    return resolved


def _load_recon_complex(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "YY_pred" not in data:
            raise KeyError(f"Recon artifact missing YY_pred: {path}")
        return np.squeeze(np.asarray(data["YY_pred"])).astype(np.complex64)


def evaluate_selected_models(
    recon_paths: Dict[str, Path],
    gt_path: Path,
) -> Dict[str, Dict[str, object]]:
    """Evaluate selected model reconstructions on canonical GT object grid."""
    from ptycho.evaluation import eval_reconstruction

    gt_ref = _load_recon_complex(Path(gt_path))
    target_hw = (int(gt_ref.shape[0]), int(gt_ref.shape[1]))
    out: Dict[str, Dict[str, object]] = {}
    for model_id, recon_path in recon_paths.items():
        pred = _load_recon_complex(Path(recon_path))
        pred_ref = resize_complex_to_shape(pred, target_hw)
        metrics = eval_reconstruction(
            pred_ref[..., None],
            gt_ref[..., None],
            label=model_id,
        )
        out[model_id] = {
            "reference_shape": [target_hw[0], target_hw[1]],
            "metrics": metrics,
        }
    return out


def _finalize_compare_outputs(
    *,
    output_dir: Path,
    merged_metrics: Dict[str, dict],
    visual_order: Tuple[str, ...],
    model_ns: Optional[Dict[str, int]] = None,
    model_labels: Optional[Mapping[str, str]] = None,
    row_payloads: Optional[Mapping[str, Mapping[str, object]]] = None,
    manifest_claim_boundary: str = "grid_lines_compare_bundle",
) -> Dict[str, str]:
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(merged_metrics, indent=2, default=_json_default))

    from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals
    from scripts.studies.metrics_tables import write_metrics_tables, write_model_manifest

    render_grid_lines_visuals(output_dir, order=visual_order)
    table_paths = write_metrics_tables(
        output_dir,
        merged_metrics,
        model_ns=model_ns,
        model_labels=model_labels,
    )
    finalized = {
        "metrics_path": str(metrics_path),
        **table_paths,
    }
    if row_payloads:
        model_manifest_path = write_model_manifest(
            output_dir=output_dir,
            row_payloads=row_payloads,
            benchmark_status="decision_support_complete",
            claim_boundary=manifest_claim_boundary,
        )
        finalized["model_manifest_path"] = str(model_manifest_path)
    return finalized


def _build_metrics_model_labels(
    model_ids: Iterable[str],
    *,
    row_specs_by_model: Optional[Mapping[str, Mapping[str, Any]]] = None,
    row_payloads: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for model_id in model_ids:
        label: Optional[object] = None
        if row_specs_by_model:
            spec = row_specs_by_model.get(model_id)
            if spec is not None:
                label = spec.get("display_label") or spec.get("model_label")
                if not label and row_payloads:
                    payload = row_payloads.get(model_id)
                    if isinstance(payload, dict):
                        label = payload.get("model_label")
        if label:
            labels[str(model_id)] = str(label)
    return labels


def _validate_preflight_probe(*, probe_source: str, probe_npz: Path) -> None:
    if probe_source != "custom":
        return
    probe_path = Path(probe_npz)
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe NPZ not found for compare preflight: {probe_path}")


def _run_compare_preflight(
    *,
    N: int,
    gridsize: int,
    output_dir: Path,
    probe_npz: Path,
    selected_models: Tuple[str, ...],
    resolved_model_n: Dict[str, int],
    seed: int,
    dataset_source: str,
    train_data: Optional[Path],
    test_data: Optional[Path],
    probe_source: str,
    set_phi: bool,
    nimgs_train: int,
    nimgs_test: int,
    nphotons: float,
    torch_epochs: Optional[int],
    nepochs: int,
    torch_batch_size: Optional[int],
    batch_size: int,
    torch_learning_rate: float,
    torch_infer_batch_size: int,
    torch_gradient_clip_val: float,
    torch_gradient_clip_algorithm: str,
    torch_output_mode: str,
    torch_loss_mode: str,
    torch_mae_pred_l2_match_target: bool,
    fno_modes: int,
    fno_width: int,
    fno_blocks: int,
    fno_cnn_blocks: int,
    fno_input_transform: str,
    torch_max_hidden_channels: Optional[int],
    torch_resnet_width: Optional[int],
    torch_optimizer: str,
    torch_weight_decay: float,
    torch_momentum: float,
    torch_beta1: float,
    torch_beta2: float,
    torch_scheduler: str,
    torch_lr_warmup_epochs: int,
    torch_lr_min_ratio: float,
    torch_plateau_factor: float,
    torch_plateau_patience: int,
    torch_plateau_min_lr: float,
    torch_plateau_threshold: float,
    torch_position_reassembly_backend: str,
    torch_position_reassembly_batch_size: int,
    torch_position_crop_border: Optional[int],
    probe_scale_mode: str,
    probe_smoothing_sigma: float,
    probe_mask_diameter: Optional[int],
    row_specs_by_model: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict:
    _validate_preflight_probe(probe_source=probe_source, probe_npz=probe_npz)
    if dataset_source == "external_raw_npz":
        if train_data is None or not Path(train_data).exists():
            raise FileNotFoundError(f"Missing external_raw_npz train_data for compare preflight: {train_data}")
        if test_data is None or not Path(test_data).exists():
            raise FileNotFoundError(f"Missing external_raw_npz test_data for compare preflight: {test_data}")

    row_plan: List[Dict[str, object]] = []
    tf_models: List[str] = []
    from scripts.studies import grid_lines_torch_runner as torch_runner

    external_mode = dataset_source == "external_raw_npz"
    for model_id in selected_models:
        n_for_model = int(resolved_model_n.get(model_id, N))
        if model_id in TF_MODEL_IDS:
            tf_models.append(model_id)
            row_plan.append(
                {
                    "model_id": model_id,
                    "architecture": MODEL_TO_LEGACY_ARCH[model_id],
                    "backend": "tf",
                    "N": n_for_model,
                    "status": "supported_for_harness",
                }
            )
            continue

        if model_id in TORCH_MODEL_IDS:
            row_spec = (row_specs_by_model or {}).get(model_id, DEFAULT_TORCH_ROW_SPECS.get(model_id, {}))
            arch, training_procedure = _torch_model_route(model_id, row_specs_by_model)
            torch_cfg_kwargs: Dict[str, Any] = {
                "train_npz": Path(train_data) if train_data is not None else output_dir / "preflight_train.npz",
                "test_npz": Path(test_data) if test_data is not None else output_dir / "preflight_test.npz",
                "output_dir": output_dir,
                "architecture": arch,
                "training_procedure": training_procedure,
                "model_id_override": model_id,
                "model_label_override": row_spec.get("model_label", PAPER_MODEL_LABELS.get(model_id, model_id)),
                "seed": seed,
                "epochs": torch_epochs or nepochs,
                "batch_size": torch_batch_size or batch_size,
                "learning_rate": torch_learning_rate,
                "infer_batch_size": torch_infer_batch_size,
                "gradient_clip_val": torch_gradient_clip_val,
                "gradient_clip_algorithm": torch_gradient_clip_algorithm,
                "generator_output_mode": torch_output_mode,
                "N": n_for_model,
                "gridsize": gridsize,
                "probe_source": probe_source,
                "torch_loss_mode": torch_loss_mode,
                "torch_mae_pred_l2_match_target": torch_mae_pred_l2_match_target,
                "fno_modes": fno_modes,
                "fno_width": fno_width,
                "fno_blocks": fno_blocks,
                "fno_cnn_blocks": fno_cnn_blocks,
                "fno_input_transform": fno_input_transform,
                "max_hidden_channels": torch_max_hidden_channels,
                "resnet_width": torch_resnet_width,
                "optimizer": torch_optimizer,
                "weight_decay": torch_weight_decay,
                "momentum": torch_momentum,
                "adam_beta1": torch_beta1,
                "adam_beta2": torch_beta2,
                "scheduler": torch_scheduler,
                "lr_warmup_epochs": torch_lr_warmup_epochs,
                "lr_min_ratio": torch_lr_min_ratio,
                "plateau_factor": torch_plateau_factor,
                "plateau_patience": torch_plateau_patience,
                "plateau_min_lr": torch_plateau_min_lr,
                "plateau_threshold": torch_plateau_threshold,
                "reassembly_mode": "position" if external_mode else "grid_lines",
                "position_reassembly_backend": torch_position_reassembly_backend,
                "position_reassembly_batch_size": torch_position_reassembly_batch_size,
                "position_crop_border": torch_position_crop_border,
                "probe_mask_diameter": probe_mask_diameter,
            }
            torch_cfg_kwargs.update(dict(row_spec.get("overrides", {})))
            torch_cfg = TorchRunnerConfig(**torch_cfg_kwargs)
            torch_runner.setup_torch_configs(torch_cfg)
            row_plan.append(
                {
                    "model_id": model_id,
                    "architecture": arch,
                    "backend": "torch",
                    "N": n_for_model,
                    "status": "supported_for_harness",
                    "model_label": row_spec.get("model_label", PAPER_MODEL_LABELS.get(model_id, model_id)),
                    "overrides": dict(row_spec.get("overrides", {})),
                }
            )
            continue

        if model_id == "pinn_ptychovit":
            row_plan.append(
                {
                    "model_id": model_id,
                    "architecture": "ptychovit",
                    "backend": "ptychovit",
                    "N": n_for_model,
                    "status": "supported_for_harness",
                }
            )
            continue

        raise ValueError(f"Unsupported model '{model_id}'")

    return {
        "mode": "preflight_only",
        "selected_models": list(selected_models),
        "resolved_model_n": dict(resolved_model_n),
        "tf_models": tf_models,
        "dataset_source": dataset_source,
        "probe_source": probe_source,
        "probe_npz": str(probe_npz),
        "seed": seed,
        "contract": {
            "N": N,
            "gridsize": gridsize,
            "dataset_source": dataset_source,
            "set_phi": bool(set_phi),
            "probe_source": probe_source,
            "probe_npz": str(probe_npz),
            "nimgs_train": int(nimgs_train),
            "nimgs_test": int(nimgs_test),
            "nphotons": float(nphotons),
            "probe_scale_mode": probe_scale_mode,
            "probe_smoothing_sigma": float(probe_smoothing_sigma),
            "probe_mask_diameter": probe_mask_diameter,
            "seed": seed,
            "torch_epochs": int(torch_epochs or nepochs),
            "torch_learning_rate": float(torch_learning_rate),
            "torch_scheduler": torch_scheduler,
            "torch_plateau_factor": float(torch_plateau_factor),
            "torch_plateau_patience": int(torch_plateau_patience),
            "torch_plateau_min_lr": float(torch_plateau_min_lr),
            "torch_plateau_threshold": float(torch_plateau_threshold),
            "torch_loss_mode": torch_loss_mode,
            "torch_mae_pred_l2_match_target": bool(torch_mae_pred_l2_match_target),
            "torch_output_mode": torch_output_mode,
            "fno_modes": int(fno_modes),
            "fno_width": int(fno_width),
            "fno_blocks": int(fno_blocks),
            "fno_cnn_blocks": int(fno_cnn_blocks),
        },
        "row_plan": row_plan,
    }


def _finalize_root_launcher_completion_artifacts(output_dir: Path) -> None:
    launcher_stderr = output_dir / "launcher_stderr.log"
    launcher_stdout = output_dir / "launcher_stdout.log"
    if not launcher_stderr.exists() and not launcher_stdout.exists():
        return
    wrapper_invocation_json = output_dir / "invocation.json"
    for model_id in sorted(TORCH_MODEL_IDS):
        run_dir = output_dir / "runs" / model_id
        if not run_dir.exists():
            continue
        write_launcher_completion_evidence(
            output_dir,
            model_id=model_id,
            wrapper_invocation_json=wrapper_invocation_json,
            launcher_stderr_log=launcher_stderr,
            launcher_stdout_log=launcher_stdout if launcher_stdout.exists() else None,
        )


def run_grid_lines_compare(
    *,
    N: int,
    gridsize: int,
    output_dir: Path,
    probe_npz: Path,
    architectures: Iterable[str],
    models: Optional[Tuple[str, ...]] = None,
    row_specs: Optional[Iterable[Mapping[str, Any]]] = None,
    model_n: Optional[Dict[str, int]] = None,
    reuse_existing_recons: bool = False,
    ptychovit_repo: Optional[Path] = None,
    seed: Optional[int] = None,
    nimgs_train: int = 2,
    nimgs_test: int = 2,
    nphotons: float = 1e9,
    nepochs: int = 60,
    batch_size: int = 16,
    nll_weight: float = 0.0,
    mae_weight: float = 1.0,
    realspace_weight: float = 0.0,
    probe_smoothing_sigma: float = 0.5,
    probe_mask_diameter: Optional[int] = None,
    probe_source: str = "custom",
    probe_scale_mode: str = "pad_preserve",
    set_phi: bool = False,
    torch_epochs: Optional[int] = None,
    torch_batch_size: Optional[int] = None,
    torch_learning_rate: float = 1e-3,
    torch_infer_batch_size: int = 128,
    torch_gradient_clip_val: float = 0.0,
    torch_gradient_clip_algorithm: str = "norm",
    torch_output_mode: str = "real_imag",
    torch_loss_mode: str = "mae",
    torch_mae_pred_l2_match_target: bool = False,
    torch_log_grad_norm: bool = False,
    torch_grad_norm_log_freq: int = 1,
    fno_modes: int = 12,
    fno_width: int = 32,
    fno_blocks: int = 4,
    fno_cnn_blocks: int = 2,
    fno_input_transform: str = "none",
    torch_max_hidden_channels: Optional[int] = None,
    torch_resnet_width: Optional[int] = None,
    torch_optimizer: str = "adam",
    torch_weight_decay: float = 0.0,
    torch_momentum: float = 0.9,
    torch_beta1: float = 0.9,
    torch_beta2: float = 0.999,
    torch_scheduler: str = "Default",
    torch_lr_warmup_epochs: int = 0,
    torch_lr_min_ratio: float = 0.1,
    torch_plateau_factor: float = 0.5,
    torch_plateau_patience: int = 2,
    torch_plateau_min_lr: float = 5e-5,
    torch_plateau_threshold: float = 0.0,
    torch_position_reassembly_backend: str = "auto",
    torch_position_reassembly_batch_size: int = 64,
    torch_position_crop_border: Optional[int] = None,
    dataset_source: str = "synthetic_lines",
    train_data: Optional[Path] = None,
    test_data: Optional[Path] = None,
    preflight_only: bool = False,
    manifest_claim_boundary: str = "grid_lines_compare_bundle",
) -> dict:
    os.environ.setdefault("PTYCHO_MEMOIZE_KEY_MODE", "dataset")
    output_dir = Path(output_dir)
    if seed is None:
        seed = random.SystemRandom().randrange(0, 2**32)
        print(f"[grid_lines_compare_wrapper] Using random seed {seed}")
    architectures = tuple(architectures)
    row_specs_by_model = _normalize_row_specs(row_specs)
    model_n = dict(model_n or {})
    selected_models: Tuple[str, ...]
    if models:
        selected_models = tuple(models)
    elif row_specs is not None:
        selected_models = tuple(
            str(spec["model_id"]) for spec in row_specs if str(spec.get("model_id", "")).strip()
        )
    else:
        selected_models = tuple(
            LEGACY_ARCH_TO_MODEL[arch]
            for arch in architectures
            if arch in LEGACY_ARCH_TO_MODEL
        )
    resolved_model_n = resolve_model_ns(selected_models, model_n, default_n=N)
    validate_model_specs(selected_models, resolved_model_n, row_specs_by_model=row_specs_by_model)
    required_ns = compute_required_ns(selected_models, resolved_model_n, default_n=N)
    external_mode = dataset_source == "external_raw_npz"

    if dataset_source not in {"synthetic_lines", "external_raw_npz"}:
        raise ValueError(f"Unsupported dataset_source '{dataset_source}'")
    if external_mode:
        if train_data is None or test_data is None:
            raise ValueError("external_raw_npz mode requires both train_data and test_data.")
        unsupported = [model_id for model_id in selected_models if model_id not in TORCH_MODEL_IDS]
        if unsupported:
            supported = ", ".join(sorted(TORCH_MODEL_IDS))
            bad = ", ".join(sorted(unsupported))
            raise ValueError(
                f"external_raw_npz supports only Torch model IDs ({supported}); got unsupported: {bad}"
            )
        if len(set(required_ns)) != 1:
            raise ValueError("external_raw_npz mode currently supports only a single N override.")

    if preflight_only:
        return _run_compare_preflight(
            N=N,
            gridsize=gridsize,
            output_dir=output_dir,
            probe_npz=probe_npz,
            selected_models=selected_models,
            resolved_model_n=resolved_model_n,
            seed=seed,
            dataset_source=dataset_source,
            train_data=train_data,
            test_data=test_data,
            probe_source=probe_source,
            set_phi=set_phi,
            nimgs_train=nimgs_train,
            nimgs_test=nimgs_test,
            nphotons=nphotons,
            torch_epochs=torch_epochs,
            nepochs=nepochs,
            torch_batch_size=torch_batch_size,
            batch_size=batch_size,
            torch_learning_rate=torch_learning_rate,
            torch_infer_batch_size=torch_infer_batch_size,
            torch_gradient_clip_val=torch_gradient_clip_val,
            torch_gradient_clip_algorithm=torch_gradient_clip_algorithm,
            torch_output_mode=torch_output_mode,
            torch_loss_mode=torch_loss_mode,
            torch_mae_pred_l2_match_target=torch_mae_pred_l2_match_target,
            fno_modes=fno_modes,
            fno_width=fno_width,
            fno_blocks=fno_blocks,
            fno_cnn_blocks=fno_cnn_blocks,
            fno_input_transform=fno_input_transform,
            torch_max_hidden_channels=torch_max_hidden_channels,
            torch_resnet_width=torch_resnet_width,
            torch_optimizer=torch_optimizer,
            torch_weight_decay=torch_weight_decay,
            torch_momentum=torch_momentum,
            torch_beta1=torch_beta1,
            torch_beta2=torch_beta2,
            torch_scheduler=torch_scheduler,
            torch_lr_warmup_epochs=torch_lr_warmup_epochs,
            torch_lr_min_ratio=torch_lr_min_ratio,
            torch_plateau_factor=torch_plateau_factor,
            torch_plateau_patience=torch_plateau_patience,
            torch_plateau_min_lr=torch_plateau_min_lr,
            torch_plateau_threshold=torch_plateau_threshold,
            torch_position_reassembly_backend=torch_position_reassembly_backend,
            torch_position_reassembly_batch_size=torch_position_reassembly_batch_size,
            torch_position_crop_border=torch_position_crop_border,
            probe_scale_mode=probe_scale_mode,
            probe_smoothing_sigma=probe_smoothing_sigma,
            probe_mask_diameter=probe_mask_diameter,
            row_specs_by_model=row_specs_by_model,
        )

    if models or external_mode:
        from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair
        from ptycho.interop.ptychovit.validate import validate_hdf5_pair
        from ptycho.workflows import grid_lines_workflow as tf_workflow
        from scripts.studies.grid_study_dataset_builder import build_datasets
        from scripts.studies import grid_lines_torch_runner as torch_runner
        from scripts.studies.grid_lines_ptychovit_runner import (
            PtychoViTRunnerConfig,
            run_grid_lines_ptychovit,
        )

        precomputed_gt = output_dir / "recons" / "gt" / "recon.npz"
        precomputed_recons = {
            model_id: output_dir / "recons" / model_id / "recon.npz"
            for model_id in selected_models
            if (output_dir / "recons" / model_id / "recon.npz").exists()
        }
        if reuse_existing_recons and precomputed_gt.exists() and len(precomputed_recons) == len(selected_models):
            train_npz, test_npz = _recover_existing_dataset_paths(output_dir)
            metrics_by_model = evaluate_selected_models(
                precomputed_recons,
                precomputed_gt,
            )
            metrics_by_model_path = output_dir / "metrics_by_model.json"
            metrics_by_model_path.write_text(json.dumps(metrics_by_model, indent=2, default=_json_default))
            legacy_metrics = {
                model_id: payload["metrics"] for model_id, payload in metrics_by_model.items()
            }
            row_payloads: Dict[str, Dict[str, object]] = {}
            for model_id in selected_models:
                n_for_model = int(resolved_model_n.get(model_id, N))
                metric_payload = legacy_metrics.get(model_id, {})
                try:
                    if model_id in TF_MODEL_IDS:
                        recovered = _recover_tf_row_payload(
                            output_dir=output_dir,
                            model_id=model_id,
                            n_value=n_for_model,
                            epoch_budget=int(torch_epochs or nepochs),
                            metrics=metric_payload,
                        )
                    elif model_id in TORCH_MODEL_IDS:
                        recovered = _recover_torch_row_payload(
                            output_dir=output_dir,
                            model_id=model_id,
                            n_value=n_for_model,
                            metrics=metric_payload,
                        )
                    else:
                        recovered = {"metrics": metric_payload}
                except Exception:
                    recovered = {
                        "caveats": ["recovered_from_existing_artifacts", "row_payload_recovery_failed"],
                        "metrics": metric_payload,
                    }
                row_payloads[model_id] = _coerce_paper_row_payload(
                    model_id,
                    recovered,
                    n_value=n_for_model,
                    metrics=metric_payload,
                )
                row_payloads[model_id] = _enrich_paper_row_payload(
                    model_id=model_id,
                    payload=row_payloads[model_id],
                    output_dir=output_dir,
                    train_npz=train_npz,
                    test_npz=test_npz,
                    seed=int(seed),
                    nimgs_train=int(nimgs_train),
                    nimgs_test=int(nimgs_test),
                    gridsize=int(gridsize),
                    set_phi=bool(set_phi),
                    probe_npz=Path(probe_npz),
                    dataset_source=str(dataset_source),
                    probe_source=str(probe_source),
                    probe_scale_mode=str(probe_scale_mode),
                    row_spec=(row_specs_by_model or {}).get(model_id),
                )

            model_ns_for_metrics = {
                model_id: int(resolved_model_n.get(model_id, N))
                for model_id in legacy_metrics.keys()
            }
            _finalize_compare_outputs(
                output_dir=output_dir,
                merged_metrics=legacy_metrics,
                visual_order=tuple(["gt", *precomputed_recons.keys()]),
                model_ns=model_ns_for_metrics,
                model_labels=_build_metrics_model_labels(
                    legacy_metrics.keys(),
                    row_specs_by_model=row_specs_by_model,
                    row_payloads=row_payloads,
                ),
                row_payloads=row_payloads,
                manifest_claim_boundary=manifest_claim_boundary,
            )
            return {
                "train_npz": str(train_npz),
                "test_npz": str(test_npz),
                "metrics": legacy_metrics,
                "metrics_by_model": metrics_by_model,
                "gt_recon": str(precomputed_gt),
                "recon_paths": {k: str(v) for k, v in precomputed_recons.items()},
                "row_payloads": row_payloads,
            }

        tf_cfg = GridLinesConfig(
            N=N,
            gridsize=gridsize,
            output_dir=output_dir,
            probe_npz=probe_npz,
            seed=seed,
            nimgs_train=nimgs_train,
            nimgs_test=nimgs_test,
            nphotons=nphotons,
            nepochs=nepochs,
            batch_size=batch_size,
            nll_weight=nll_weight,
            mae_weight=mae_weight,
            realspace_weight=realspace_weight,
            probe_smoothing_sigma=probe_smoothing_sigma,
            probe_mask_diameter=probe_mask_diameter,
            probe_source=probe_source,
            probe_scale_mode=probe_scale_mode,
            set_phi=set_phi,
        )
        bundles_by_n = build_datasets(
            dataset_source=dataset_source,
            cfg=tf_cfg,
            required_ns=required_ns,
            train_data=train_data,
            test_data=test_data,
            n_groups=max(1, int(nimgs_train)),
            n_subsample=None,
            neighbor_count=7,
            subsample_seed=seed,
        )
        gt_candidates = {bundle["gt_recon"] for bundle in bundles_by_n.values() if "gt_recon" in bundle}
        if not gt_candidates:
            raise RuntimeError("Dataset builders did not provide canonical gt_recon path")
        if len(gt_candidates) != 1:
            raise RuntimeError("Multiple canonical GT paths detected across N bundles")
        gt_path = Path(next(iter(gt_candidates)))

        recon_paths: Dict[str, Path] = {}
        row_payloads: Dict[str, Dict[str, object]] = {}
        tf_models_by_n: Dict[int, Tuple[str, ...]] = {}
        for model_id in selected_models:
            if model_id in TF_MODEL_IDS:
                n_for_model = resolved_model_n[model_id]
                existing = list(tf_models_by_n.get(n_for_model, ()))
                if model_id not in existing:
                    existing.append(model_id)
                tf_models_by_n[n_for_model] = tuple(existing)

        # Run TF workflow at most once per N for the selected TF models.
        for n_for_model, tf_models_for_n in tf_models_by_n.items():
            bundle = bundles_by_n[n_for_model]
            tf_train_npz = Path(bundle["train_npz"])
            tf_test_npz = Path(bundle["test_npz"])
            existing_tf_recons = {
                model_id: output_dir / "recons" / model_id / "recon.npz"
                for model_id in tf_models_for_n
                if (output_dir / "recons" / model_id / "recon.npz").exists()
            }
            if reuse_existing_recons and len(existing_tf_recons) == len(tf_models_for_n):
                recon_paths.update(existing_tf_recons)
                continue

            tf_model_cfg = GridLinesConfig(
                N=n_for_model,
                gridsize=gridsize,
                output_dir=output_dir,
                probe_npz=probe_npz,
                seed=seed,
                nimgs_train=nimgs_train,
                nimgs_test=nimgs_test,
                nphotons=nphotons,
                nepochs=torch_epochs or nepochs,
                batch_size=batch_size,
                nll_weight=nll_weight,
                mae_weight=mae_weight,
                realspace_weight=realspace_weight,
                probe_smoothing_sigma=probe_smoothing_sigma,
                probe_mask_diameter=probe_mask_diameter,
                probe_source=probe_source,
                probe_scale_mode=probe_scale_mode,
                set_phi=set_phi,
            )
            with _capture_execution_logs(
                stdout_overwrite_targets=(),
                stderr_overwrite_targets=(),
                stdout_append_targets=(output_dir / "live_stdout.log",),
                stderr_append_targets=(output_dir / "live_stderr.log",),
            ):
                tf_result = _run_tf_workflow_with_selected_models(tf_workflow, tf_model_cfg, tf_models_for_n)
            tf_row_payloads = tf_result.get("row_payloads", {})
            for model_id in tf_models_for_n:
                recon_path = output_dir / "recons" / model_id / "recon.npz"
                if not recon_path.exists():
                    raise RuntimeError(
                        f"Expected recon artifact missing for {model_id}: {recon_path}"
                    )
                recon_paths[model_id] = recon_path
                row_payloads[model_id] = _coerce_paper_row_payload(
                    model_id,
                    tf_row_payloads.get(model_id),
                    n_value=int(resolved_model_n.get(model_id, n_for_model)),
                )
                row_payloads[model_id] = _enrich_paper_row_payload(
                    model_id=model_id,
                    payload=row_payloads[model_id],
                    output_dir=output_dir,
                    train_npz=tf_train_npz,
                    test_npz=tf_test_npz,
                    seed=int(seed),
                    nimgs_train=int(nimgs_train),
                    nimgs_test=int(nimgs_test),
                    gridsize=int(gridsize),
                    set_phi=bool(set_phi),
                    probe_npz=Path(probe_npz),
                    dataset_source=str(dataset_source),
                    probe_source=str(probe_source),
                    probe_scale_mode=str(probe_scale_mode),
                    row_spec=(row_specs_by_model or {}).get(model_id),
                )

        for model_id in selected_models:
            if model_id in recon_paths:
                continue
            n_for_model = resolved_model_n[model_id]
            bundle = bundles_by_n[n_for_model]
            train_npz = Path(bundle["train_npz"])
            test_npz = Path(bundle["test_npz"])
            existing_recon = output_dir / "recons" / model_id / "recon.npz"
            if reuse_existing_recons and existing_recon.exists():
                recon_paths[model_id] = existing_recon
                continue

            if model_id in TORCH_MODEL_IDS:
                row_spec = row_specs_by_model.get(model_id, DEFAULT_TORCH_ROW_SPECS.get(model_id, {}))
                arch, training_procedure = _torch_model_route(model_id, row_specs_by_model)
                torch_cfg_kwargs: Dict[str, Any] = {
                    "train_npz": train_npz,
                    "test_npz": test_npz,
                    "output_dir": output_dir,
                    "architecture": arch,
                    "training_procedure": training_procedure,
                    "model_id_override": model_id,
                    "model_label_override": row_spec.get("model_label", PAPER_MODEL_LABELS.get(model_id, model_id)),
                    "seed": seed,
                    "epochs": torch_epochs or nepochs,
                    "batch_size": torch_batch_size or batch_size,
                    "learning_rate": torch_learning_rate,
                    "infer_batch_size": torch_infer_batch_size,
                    "gradient_clip_val": torch_gradient_clip_val,
                    "gradient_clip_algorithm": torch_gradient_clip_algorithm,
                    "generator_output_mode": torch_output_mode,
                    "N": n_for_model,
                    "gridsize": gridsize,
                    "probe_source": probe_source,
                    "torch_loss_mode": torch_loss_mode,
                    "torch_mae_pred_l2_match_target": torch_mae_pred_l2_match_target,
                    "fno_modes": fno_modes,
                    "fno_width": fno_width,
                    "fno_blocks": fno_blocks,
                    "fno_cnn_blocks": fno_cnn_blocks,
                    "fno_input_transform": fno_input_transform,
                    "max_hidden_channels": torch_max_hidden_channels,
                    "resnet_width": torch_resnet_width,
                    "optimizer": torch_optimizer,
                    "weight_decay": torch_weight_decay,
                    "momentum": torch_momentum,
                    "adam_beta1": torch_beta1,
                    "adam_beta2": torch_beta2,
                    "log_grad_norm": torch_log_grad_norm,
                    "grad_norm_log_freq": torch_grad_norm_log_freq,
                    "scheduler": torch_scheduler,
                    "lr_warmup_epochs": torch_lr_warmup_epochs,
                    "lr_min_ratio": torch_lr_min_ratio,
                    "plateau_factor": torch_plateau_factor,
                    "plateau_patience": torch_plateau_patience,
                    "plateau_min_lr": torch_plateau_min_lr,
                    "plateau_threshold": torch_plateau_threshold,
                    "reassembly_mode": "position" if external_mode else "grid_lines",
                    "position_reassembly_backend": torch_position_reassembly_backend,
                    "position_reassembly_batch_size": torch_position_reassembly_batch_size,
                    "position_crop_border": torch_position_crop_border,
                }
                torch_cfg_kwargs.update(dict(row_spec.get("overrides", {})))
                torch_cfg = TorchRunnerConfig(**torch_cfg_kwargs)
                with _capture_execution_logs(
                    stdout_overwrite_targets=(output_dir / "runs" / model_id / "stdout.log",),
                    stderr_overwrite_targets=(output_dir / "runs" / model_id / "stderr.log",),
                    stdout_append_targets=(output_dir / "live_stdout.log",),
                    stderr_append_targets=(output_dir / "live_stderr.log",),
                ):
                    torch_result = torch_runner.run_grid_lines_torch(torch_cfg)
                recon_path = torch_result.get("recon_npz")
                if recon_path is None:
                    recon_path = output_dir / "recons" / model_id / "recon.npz"
                recon_paths[model_id] = Path(recon_path)
                row_payloads[model_id] = _coerce_paper_row_payload(
                    model_id,
                    torch_result.get("paper_row_payload"),
                    n_value=int(n_for_model),
                )
                row_payloads[model_id] = _enrich_paper_row_payload(
                    model_id=model_id,
                    payload=row_payloads[model_id],
                    output_dir=output_dir,
                    train_npz=train_npz,
                    test_npz=test_npz,
                    seed=int(seed),
                    nimgs_train=int(nimgs_train),
                    nimgs_test=int(nimgs_test),
                    gridsize=int(gridsize),
                    set_phi=bool(set_phi),
                    probe_npz=Path(probe_npz),
                    dataset_source=str(dataset_source),
                    probe_source=str(probe_source),
                    probe_scale_mode=str(probe_scale_mode),
                    row_spec=(row_specs_by_model or {}).get(model_id),
                )
                continue

            if model_id == "pinn_ptychovit":
                interop_dir = output_dir / "interop" / "pinn_ptychovit" / f"N{n_for_model}"
                train_pair = convert_npz_split_to_hdf5_pair(
                    npz_path=train_npz,
                    out_dir=interop_dir / "train",
                    object_name=f"grid_lines_train_N{n_for_model}",
                )
                test_pair = convert_npz_split_to_hdf5_pair(
                    npz_path=test_npz,
                    out_dir=interop_dir / "test",
                    object_name=f"grid_lines_test_N{n_for_model}",
                )
                validate_hdf5_pair(train_pair.dp_hdf5, train_pair.para_hdf5)
                validate_hdf5_pair(test_pair.dp_hdf5, test_pair.para_hdf5)

                canonical_interop_dir = output_dir / "interop"
                canonical_interop_dir.mkdir(parents=True, exist_ok=True)
                canonical_train_dp = canonical_interop_dir / "train_dp.hdf5"
                canonical_train_para = canonical_interop_dir / "train_para.hdf5"
                canonical_test_dp = canonical_interop_dir / "test_dp.hdf5"
                canonical_test_para = canonical_interop_dir / "test_para.hdf5"
                shutil.copy2(train_pair.dp_hdf5, canonical_train_dp)
                shutil.copy2(train_pair.para_hdf5, canonical_train_para)
                shutil.copy2(test_pair.dp_hdf5, canonical_test_dp)
                shutil.copy2(test_pair.para_hdf5, canonical_test_para)

                pvit_cfg = PtychoViTRunnerConfig(
                    ptychovit_repo=Path(ptychovit_repo or "/home/ollie/Documents/ptycho-vit"),
                    output_dir=output_dir,
                    train_dp=train_pair.dp_hdf5,
                    test_dp=test_pair.dp_hdf5,
                    train_para=train_pair.para_hdf5,
                    test_para=test_pair.para_hdf5,
                    model_n=n_for_model,
                    mode="inference",
                )
                pvit_result = run_grid_lines_ptychovit(pvit_cfg)
                recon_paths[model_id] = Path(pvit_result["recon_npz"])
                row_payloads[model_id] = _coerce_paper_row_payload(
                    model_id,
                    pvit_result.get("paper_row_payload"),
                    n_value=int(n_for_model),
                )
                continue

            if model_id in TF_MODEL_IDS:
                # Selected TF model should have been handled by grouped per-N flow above.
                recon_path = output_dir / "recons" / model_id / "recon.npz"
                if recon_path.exists():
                    recon_paths[model_id] = recon_path
                    continue
                raise RuntimeError(
                    f"Expected recon artifact missing for {model_id}: {recon_path}"
                )

            if model_id == "gt":
                continue

            raise ValueError(f"Unsupported model '{model_id}'")

        metrics_by_model = evaluate_selected_models(
            recon_paths,
            gt_path,
        )
        metrics_by_model_path = output_dir / "metrics_by_model.json"
        metrics_by_model_path.write_text(json.dumps(metrics_by_model, indent=2, default=_json_default))

        legacy_metrics = {
            model_id: payload["metrics"] for model_id, payload in metrics_by_model.items()
        }
        for model_id in selected_models:
            bundle = bundles_by_n[int(resolved_model_n.get(model_id, N))]
            row_train_npz = Path(bundle["train_npz"])
            row_test_npz = Path(bundle["test_npz"])
            row_payloads[model_id] = _coerce_paper_row_payload(
                model_id,
                row_payloads.get(model_id),
                n_value=int(resolved_model_n.get(model_id, N)),
                metrics=legacy_metrics.get(model_id, {}),
            )
            row_payloads[model_id] = _enrich_paper_row_payload(
                model_id=model_id,
                payload=row_payloads[model_id],
                output_dir=output_dir,
                train_npz=row_train_npz,
                test_npz=row_test_npz,
                seed=int(seed),
                nimgs_train=int(nimgs_train),
                nimgs_test=int(nimgs_test),
                gridsize=int(gridsize),
                set_phi=bool(set_phi),
                probe_npz=Path(probe_npz),
                dataset_source=str(dataset_source),
                probe_source=str(probe_source),
                probe_scale_mode=str(probe_scale_mode),
                row_spec=(row_specs_by_model or {}).get(model_id),
            )
        model_ns_for_metrics = {
            model_id: int(resolved_model_n.get(model_id, N))
            for model_id in legacy_metrics.keys()
        }
        _finalize_compare_outputs(
            output_dir=output_dir,
            merged_metrics=legacy_metrics,
            visual_order=tuple(["gt", *recon_paths.keys()]),
            model_ns=model_ns_for_metrics,
            model_labels=_build_metrics_model_labels(
                legacy_metrics.keys(),
                row_specs_by_model=row_specs_by_model,
                row_payloads=row_payloads,
            ),
            row_payloads=row_payloads,
            manifest_claim_boundary=manifest_claim_boundary,
        )
        first_bundle = bundles_by_n[required_ns[0]]
        return {
            "train_npz": str(first_bundle["train_npz"]),
            "test_npz": str(first_bundle["test_npz"]),
            "metrics": legacy_metrics,
            "metrics_by_model": metrics_by_model,
            "gt_recon": str(gt_path),
            "recon_paths": {k: str(v) for k, v in recon_paths.items()},
            "row_payloads": row_payloads,
        }

    dataset_dir = output_dir / "datasets" / f"N{N}" / f"gs{gridsize}"
    train_npz = dataset_dir / "train.npz"
    test_npz = dataset_dir / "test.npz"

    tf_metrics = {}
    row_payloads: Dict[str, Dict[str, object]] = {}
    selected_architectures = architectures if not models else tuple(
        MODEL_TO_LEGACY_ARCH[m]
        for m in selected_models
        if m in MODEL_TO_LEGACY_ARCH
    )
    selected_tf_models = tuple(
        model_id
        for model_id in ("pinn", "baseline")
        if MODEL_TO_LEGACY_ARCH.get(model_id) in selected_architectures
    )

    if selected_tf_models:
        tf_cfg = GridLinesConfig(
            N=N,
            gridsize=gridsize,
            output_dir=output_dir,
            probe_npz=probe_npz,
            seed=seed,
            nimgs_train=nimgs_train,
            nimgs_test=nimgs_test,
            nphotons=nphotons,
            nepochs=torch_epochs or nepochs,
            batch_size=batch_size,
            nll_weight=nll_weight,
            mae_weight=mae_weight,
            realspace_weight=realspace_weight,
            probe_smoothing_sigma=probe_smoothing_sigma,
            probe_mask_diameter=probe_mask_diameter,
            probe_source=probe_source,
            probe_scale_mode=probe_scale_mode,
            set_phi=set_phi,
        )
        from ptycho.workflows import grid_lines_workflow as tf_workflow
        tf_result = _run_tf_workflow_with_selected_models(tf_workflow, tf_cfg, selected_tf_models)
        train_npz = Path(tf_result["train_npz"])
        test_npz = Path(tf_result["test_npz"])
        tf_row_payloads = tf_result.get("row_payloads", {})
        for model_id in selected_tf_models:
            row_payloads[model_id] = _coerce_paper_row_payload(
                model_id,
                tf_row_payloads.get(model_id),
                n_value=int(N),
            )
            row_payloads[model_id] = _enrich_paper_row_payload(
                model_id=model_id,
                payload=row_payloads[model_id],
                output_dir=output_dir,
                train_npz=train_npz,
                test_npz=test_npz,
                seed=int(seed),
                nimgs_train=int(nimgs_train),
                nimgs_test=int(nimgs_test),
                gridsize=int(gridsize),
                set_phi=bool(set_phi),
                probe_npz=Path(probe_npz),
                dataset_source=str(dataset_source),
                probe_source=str(probe_source),
                probe_scale_mode=str(probe_scale_mode),
                row_spec=(row_specs_by_model or {}).get(model_id),
            )
    elif not train_npz.exists() or not test_npz.exists():
        tf_cfg = GridLinesConfig(
            N=N,
            gridsize=gridsize,
            output_dir=output_dir,
            probe_npz=probe_npz,
            seed=seed,
            nimgs_train=nimgs_train,
            nimgs_test=nimgs_test,
            nphotons=nphotons,
            nepochs=nepochs,
            batch_size=batch_size,
            nll_weight=nll_weight,
            mae_weight=mae_weight,
            realspace_weight=realspace_weight,
            probe_smoothing_sigma=probe_smoothing_sigma,
            probe_mask_diameter=probe_mask_diameter,
            probe_source=probe_source,
            probe_scale_mode=probe_scale_mode,
            set_phi=set_phi,
        )
        from ptycho.workflows import grid_lines_workflow as tf_workflow
        datasets = tf_workflow.build_grid_lines_datasets(tf_cfg)
        train_npz = Path(datasets["train_npz"])
        test_npz = Path(datasets["test_npz"])
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        tf_metrics = json.loads(metrics_path.read_text())

    merged = {}
    if "cnn" in selected_architectures and "pinn" in tf_metrics:
        merged["pinn"] = tf_metrics["pinn"]
    if "baseline" in selected_architectures and "baseline" in tf_metrics:
        merged["baseline"] = tf_metrics["baseline"]

    for arch in selected_architectures:
        if arch in (
            "ffno",
            "fno",
            "hybrid",
            "stable_hybrid",
            "fno_vanilla",
            "hybrid_resnet",
            "spectral_resnet_bottleneck_net",
        ):
            torch_cfg = TorchRunnerConfig(
                train_npz=train_npz,
                test_npz=test_npz,
                output_dir=output_dir,
                architecture=arch,
                training_procedure="pinn",
                seed=seed,
                epochs=torch_epochs or nepochs,
                batch_size=torch_batch_size or batch_size,
                learning_rate=torch_learning_rate,
                infer_batch_size=torch_infer_batch_size,
                gradient_clip_val=torch_gradient_clip_val,
                gradient_clip_algorithm=torch_gradient_clip_algorithm,
                generator_output_mode=torch_output_mode,
                N=N,
                gridsize=gridsize,
                torch_loss_mode=torch_loss_mode,
                torch_mae_pred_l2_match_target=torch_mae_pred_l2_match_target,
                fno_modes=fno_modes,
                fno_width=fno_width,
                fno_blocks=fno_blocks,
                fno_cnn_blocks=fno_cnn_blocks,
                fno_input_transform=fno_input_transform,
                max_hidden_channels=torch_max_hidden_channels,
                resnet_width=torch_resnet_width,
                optimizer=torch_optimizer,
                weight_decay=torch_weight_decay,
                momentum=torch_momentum,
                adam_beta1=torch_beta1,
                adam_beta2=torch_beta2,
                log_grad_norm=torch_log_grad_norm,
                grad_norm_log_freq=torch_grad_norm_log_freq,
                scheduler=torch_scheduler,
                lr_warmup_epochs=torch_lr_warmup_epochs,
                lr_min_ratio=torch_lr_min_ratio,
                plateau_factor=torch_plateau_factor,
                plateau_patience=torch_plateau_patience,
                plateau_min_lr=torch_plateau_min_lr,
                plateau_threshold=torch_plateau_threshold,
                position_reassembly_backend=torch_position_reassembly_backend,
                position_reassembly_batch_size=torch_position_reassembly_batch_size,
                position_crop_border=torch_position_crop_border,
            )
            from scripts.studies import grid_lines_torch_runner as torch_runner
            torch_result = torch_runner.run_grid_lines_torch(torch_cfg)
            if "metrics" in torch_result:
                model_id = f"pinn_{arch}"
                merged[model_id] = torch_result["metrics"]
                row_payloads[model_id] = _coerce_paper_row_payload(
                    model_id,
                    torch_result.get("paper_row_payload"),
                    n_value=int(N),
                    metrics=torch_result["metrics"],
                )
                row_payloads[model_id] = _enrich_paper_row_payload(
                    model_id=model_id,
                    payload=row_payloads[model_id],
                    output_dir=output_dir,
                    train_npz=train_npz,
                    test_npz=test_npz,
                    seed=int(seed),
                    nimgs_train=int(nimgs_train),
                    nimgs_test=int(nimgs_test),
                    gridsize=int(gridsize),
                    set_phi=bool(set_phi),
                    probe_npz=Path(probe_npz),
                    dataset_source=str(dataset_source),
                    probe_source=str(probe_source),
                    probe_scale_mode=str(probe_scale_mode),
                    row_spec=(row_specs_by_model or {}).get(model_id),
                )

    order = ["gt"]
    if "cnn" in selected_architectures:
        order.append("pinn")
    if "baseline" in selected_architectures:
        order.append("baseline")
    if "ffno" in selected_architectures:
        order.append("pinn_ffno")
    if "fno" in selected_architectures:
        order.append("pinn_fno")
    if "hybrid" in selected_architectures:
        order.append("pinn_hybrid")
    if "stable_hybrid" in selected_architectures:
        order.append("pinn_stable_hybrid")
    if "fno_vanilla" in selected_architectures:
        order.append("pinn_fno_vanilla")
    if "hybrid_resnet" in selected_architectures:
        order.append("pinn_hybrid_resnet")
    if "spectral_resnet_bottleneck_net" in selected_architectures:
        order.append("pinn_spectral_resnet_bottleneck_net")

    model_ns_for_metrics = {model_id: int(N) for model_id in merged.keys()}
    _finalize_compare_outputs(
        output_dir=output_dir,
        merged_metrics=merged,
        visual_order=tuple(order),
        model_ns=model_ns_for_metrics,
        model_labels=_build_metrics_model_labels(
            merged.keys(),
            row_specs_by_model=row_specs_by_model,
            row_payloads=row_payloads,
        ),
        row_payloads=row_payloads,
        manifest_claim_boundary=manifest_claim_boundary,
    )
    return {
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "metrics": merged,
        "row_payloads": row_payloads,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run grid-lines comparison across backends")
    parser.add_argument("--N", type=int, required=True, choices=[64, 128])
    parser.add_argument("--gridsize", type=int, required=True, choices=[1, 2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-npz",
        type=Path,
        default=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
    )
    parser.add_argument(
        "--architectures",
        type=str,
        default="cnn,fno,hybrid,stable_hybrid,fno_vanilla,hybrid_resnet",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional explicit model IDs (overrides --architectures).",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="synthetic_lines",
        choices=["synthetic_lines", "external_raw_npz"],
        help="Dataset source for study inputs.",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Training NPZ path for external_raw_npz mode.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=None,
        help="Test NPZ path for external_raw_npz mode.",
    )
    parser.add_argument(
        "--model-n",
        type=str,
        default="",
        help="Optional per-model N overrides as comma-separated model_id=N entries.",
    )
    parser.add_argument(
        "--reuse-existing-recons",
        action="store_true",
        help="Reuse existing recon artifacts in output-dir instead of re-running selected backends.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate dataset/probe/row routing without launching backend training or inference.",
    )
    parser.add_argument(
        "--manifest-claim-boundary",
        type=str,
        default="grid_lines_compare_bundle",
        help=(
            "Claim boundary recorded in the run-level model_manifest.json. "
            "Use 'decision_support_append_only' for ablation/decision-support runs that must "
            "not be promoted as paper-grade evidence."
        ),
    )
    parser.add_argument(
        "--ptychovit-repo",
        type=Path,
        default=Path("/home/ollie/Documents/ptycho-vit"),
        help="Path to local ptycho-vit checkout for subprocess execution.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (random if omitted)")
    parser.add_argument("--nimgs-train", type=int, default=2)
    parser.add_argument("--nimgs-test", type=int, default=2)
    parser.add_argument("--nphotons", type=float, default=1e9)
    parser.add_argument("--nepochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nll-weight", type=float, default=0.0)
    parser.add_argument("--mae-weight", type=float, default=1.0)
    parser.add_argument("--realspace-weight", type=float, default=0.0)
    parser.add_argument("--probe-smoothing-sigma", type=float, default=0.5)
    parser.add_argument("--probe-mask-diameter", type=int, default=None)
    parser.add_argument(
        "--probe-source",
        choices=["custom", "ideal_disk"],
        default="custom",
        help="Probe source for grid-lines datasets.",
    )
    parser.add_argument(
        "--probe-scale-mode",
        choices=["pad_preserve", "pad_extrapolate", "interpolate"],
        default="pad_preserve",
    )
    parser.add_argument("--set-phi", action="store_true", help="Enable non-zero phase in synthetic grid data.")
    parser.add_argument("--torch-epochs", type=int, default=None)
    parser.add_argument("--torch-batch-size", type=int, default=None)
    parser.add_argument("--torch-learning-rate", type=float, default=1e-3)
    parser.add_argument("--torch-infer-batch-size", type=int, default=128)
    parser.add_argument(
        "--torch-grad-clip",
        type=float,
        default=0.0,
        help="Torch gradient clipping max norm (<=0 disables clipping).",
    )
    parser.add_argument(
        "--torch-grad-clip-algorithm",
        type=str,
        default="norm",
        choices=["norm", "value", "agc"],
        help="Torch gradient clipping algorithm.",
    )
    parser.add_argument(
        "--torch-output-mode",
        type=str,
        default="real_imag",
        choices=["real_imag", "amp_phase_logits", "amp_phase"],
        help="Torch generator output mode.",
    )
    parser.add_argument("--torch-loss-mode", type=str, default="mae", choices=["poisson", "mae"])
    parser.add_argument(
        "--torch-mae-pred-l2-match-target",
        dest="torch_mae_pred_l2_match_target",
        action="store_true",
        default=False,
        help="Enable prediction L2 matching to target in Torch MAE mode.",
    )
    parser.add_argument(
        "--torch-no-mae-pred-l2-match-target",
        dest="torch_mae_pred_l2_match_target",
        action="store_false",
        help="Disable prediction L2 matching to target in Torch MAE mode.",
    )
    parser.add_argument("--torch-log-grad-norm", action="store_true")
    parser.add_argument("--torch-grad-norm-log-freq", type=int, default=1)
    parser.add_argument("--fno-modes", type=int, default=12)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-blocks", type=int, default=4)
    parser.add_argument("--fno-cnn-blocks", type=int, default=2)
    parser.add_argument(
        "--fno-input-transform",
        type=str,
        default="none",
        choices=["none", "sqrt", "log1p", "instancenorm"],
    )
    parser.add_argument("--torch-max-hidden-channels", type=int, default=None,
                        help="Cap on hidden channels in Hybrid encoder (default: no cap)")
    parser.add_argument("--torch-resnet-width", type=int, default=None,
                        help="Fixed bottleneck width for hybrid_resnet (must be divisible by 4)")
    parser.add_argument("--torch-optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"], help="Optimizer algorithm")
    parser.add_argument("--torch-weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--torch-momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--torch-beta1", type=float, default=0.9, help="Adam/AdamW beta1")
    parser.add_argument("--torch-beta2", type=float, default=0.999, help="Adam/AdamW beta2")
    parser.add_argument("--torch-scheduler", type=str, default="Default",
                        choices=["Default", "Exponential", "WarmupCosine", "ReduceLROnPlateau"])
    parser.add_argument("--torch-lr-warmup-epochs", type=int, default=0)
    parser.add_argument("--torch-lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--torch-plateau-factor", type=float, default=0.5)
    parser.add_argument("--torch-plateau-patience", type=int, default=2)
    parser.add_argument("--torch-plateau-min-lr", type=float, default=5e-5)
    parser.add_argument("--torch-plateau-threshold", type=float, default=0.0)
    parser.add_argument(
        "--torch-position-reassembly-backend",
        type=str,
        default="auto",
        choices=["auto", "shift_sum", "batched"],
    )
    parser.add_argument("--torch-position-reassembly-batch-size", type=int, default=64)
    parser.add_argument("--torch-position-crop-border", type=int, default=None)
    args = parser.parse_args(argv)
    args.architectures = _parse_architectures(args.architectures)
    args.models = _parse_models(args.models) if args.models else None
    args.model_n = _parse_model_n(args.model_n)
    if args.dataset_source == "external_raw_npz":
        if args.train_data is None or args.test_data is None:
            parser.error("--dataset-source external_raw_npz requires --train-data and --test-data")
    if args.models:
        validate_model_specs(args.models, resolve_model_ns(args.models, args.model_n, default_n=args.N))
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    from scripts.studies.invocation_logging import (
        capture_runtime_provenance,
        get_git_commit,
        update_invocation_artifacts,
        write_invocation_artifacts,
    )

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    invocation_json, _ = write_invocation_artifacts(
        output_dir=args.output_dir,
        script_path="scripts/studies/grid_lines_compare_wrapper.py",
        argv=raw_argv,
        parsed_args=vars(args),
        extra={
            "runtime_provenance": capture_runtime_provenance(),
            "git_commit": get_git_commit(Path(__file__).resolve().parents[2]),
        },
    )
    try:
        with _capture_execution_logs(
            stdout_overwrite_targets=(args.output_dir / "launcher_stdout.log",),
            stderr_overwrite_targets=(args.output_dir / "launcher_stderr.log",),
        ):
            run_grid_lines_compare(
                N=args.N,
                gridsize=args.gridsize,
                output_dir=args.output_dir,
                probe_npz=args.probe_npz,
                architectures=args.architectures,
                models=args.models,
                model_n=args.model_n,
                reuse_existing_recons=args.reuse_existing_recons,
                ptychovit_repo=args.ptychovit_repo,
                seed=args.seed,
                nimgs_train=args.nimgs_train,
                nimgs_test=args.nimgs_test,
                nphotons=args.nphotons,
                nepochs=args.nepochs,
                batch_size=args.batch_size,
                nll_weight=args.nll_weight,
                mae_weight=args.mae_weight,
                realspace_weight=args.realspace_weight,
                probe_smoothing_sigma=args.probe_smoothing_sigma,
                probe_mask_diameter=args.probe_mask_diameter,
                probe_source=args.probe_source,
                probe_scale_mode=args.probe_scale_mode,
                set_phi=args.set_phi,
                torch_epochs=args.torch_epochs,
                torch_batch_size=args.torch_batch_size,
                torch_learning_rate=args.torch_learning_rate,
                torch_infer_batch_size=args.torch_infer_batch_size,
                torch_gradient_clip_val=args.torch_grad_clip,
                torch_gradient_clip_algorithm=args.torch_grad_clip_algorithm,
                torch_output_mode=args.torch_output_mode,
                torch_loss_mode=args.torch_loss_mode,
                torch_mae_pred_l2_match_target=args.torch_mae_pred_l2_match_target,
                torch_log_grad_norm=args.torch_log_grad_norm,
                torch_grad_norm_log_freq=args.torch_grad_norm_log_freq,
                fno_modes=args.fno_modes,
                fno_width=args.fno_width,
                fno_blocks=args.fno_blocks,
                fno_cnn_blocks=args.fno_cnn_blocks,
                fno_input_transform=args.fno_input_transform,
                torch_max_hidden_channels=args.torch_max_hidden_channels,
                torch_resnet_width=args.torch_resnet_width,
                torch_optimizer=args.torch_optimizer,
                torch_weight_decay=args.torch_weight_decay,
                torch_momentum=args.torch_momentum,
                torch_beta1=args.torch_beta1,
                torch_beta2=args.torch_beta2,
                torch_scheduler=args.torch_scheduler,
                torch_lr_warmup_epochs=args.torch_lr_warmup_epochs,
                torch_lr_min_ratio=args.torch_lr_min_ratio,
                torch_plateau_factor=args.torch_plateau_factor,
                torch_plateau_patience=args.torch_plateau_patience,
                torch_plateau_min_lr=args.torch_plateau_min_lr,
                torch_plateau_threshold=args.torch_plateau_threshold,
                torch_position_reassembly_backend=args.torch_position_reassembly_backend,
                torch_position_reassembly_batch_size=args.torch_position_reassembly_batch_size,
                torch_position_crop_border=args.torch_position_crop_border,
                dataset_source=args.dataset_source,
                train_data=args.train_data,
                test_data=args.test_data,
                preflight_only=args.preflight_only,
                manifest_claim_boundary=args.manifest_claim_boundary,
            )
        update_invocation_artifacts(
            invocation_json,
            status="completed",
            exit_code=0,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _finalize_root_launcher_completion_artifacts(args.output_dir)
    except Exception as exc:
        update_invocation_artifacts(
            invocation_json,
            status="failed",
            exit_code=1,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
