#!/usr/bin/env python3
"""Preflight the external NeuralOperator UNO surface for the lines128 CDI contract."""

from __future__ import annotations

import argparse
import importlib
import inspect
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.invocation_logging import get_git_commit
from scripts.studies.paper_provenance import current_runtime_provenance, git_dirty, write_json

READY_STATUS = "ready_for_uno_generator_integration"
BLOCKED_PACKAGE_STATUS = "blocked_neuraloperator_missing_or_incompatible"
BLOCKED_SHAPE_STATUS = "blocked_uno_shape_contract_mismatch"

LOCKED_CONTRACT: dict[str, Any] = {
    "N": 128,
    "gridsize": 1,
    "set_phi": True,
    "probe_source": "custom Run1084 probe",
    "probe_scale_mode": "pad_extrapolate",
    "probe_smoothing_sigma": 0.5,
    "nimgs_train": 2,
    "nimgs_test": 2,
    "nphotons": 1e9,
    "seed": 3,
    "torch_epochs": 40,
    "torch_learning_rate": 2e-4,
    "torch_scheduler": "ReduceLROnPlateau",
    "torch_plateau_factor": 0.5,
    "torch_plateau_patience": 2,
    "torch_plateau_min_lr": 1e-4,
    "torch_plateau_threshold": 0.0,
    "torch_loss_mode": "mae",
    "torch_output_mode": "real_imag",
    "selected_fno_comparator": "fno_vanilla",
    "fixed_sample_ids": [0, 1],
}

BASE_FROZEN_SETTINGS: dict[str, Any] = {
    "in_channels": 1,
    "out_channels": 2,
    "hidden_channels": 32,
    "lifting_channels": 128,
    "projection_channels": 128,
    "n_layers": 4,
    "uno_out_channels": [32, 64, 64, 32],
    "uno_scalings": [[1.0, 1.0], [0.5, 0.5], [1, 1], [2, 2]],
    "positional_embedding": "grid",
    "channel_mlp_skip": "linear",
    "generator_output_mode": "real_imag",
}


def _capture_pip_show_text() -> str:
    try:
        result = subprocess.run(
            ["python", "-m", "pip", "show", "neuraloperator"],
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        return f"pip show neuraloperator failed: {exc}\n"
    text = result.stdout.strip()
    if text:
        return text + "\n"
    stderr = result.stderr.strip()
    if stderr:
        return stderr + "\n"
    return ""


def _import_uno_dependencies():
    neuralop_module = importlib.import_module("neuralop")
    models_module = importlib.import_module("neuralop.models")
    uno_cls = getattr(models_module, "UNO")
    return neuralop_module, uno_cls


def _signature_payload(uno_cls: type[Any]) -> dict[str, Any]:
    signature = inspect.signature(uno_cls)
    parameters: list[dict[str, Any]] = []
    for name, parameter in signature.parameters.items():
        default = None if parameter.default is inspect._empty else repr(parameter.default)
        annotation = None if parameter.annotation is inspect._empty else repr(parameter.annotation)
        parameters.append(
            {
                "name": name,
                "kind": str(parameter.kind),
                "default": default,
                "annotation": annotation,
            }
        )
    return {
        "text": str(signature),
        "parameters": parameters,
        "parameter_names": [item["name"] for item in parameters],
    }


def _parameter_names(signature_payload: dict[str, Any]) -> set[str]:
    names = signature_payload.get("parameter_names")
    if isinstance(names, list):
        return {str(name) for name in names}
    return set()


def _build_model_kwargs(signature_payload: dict[str, Any], *, uno_n_modes: Any) -> dict[str, Any]:
    parameter_names = _parameter_names(signature_payload)
    kwargs = {key: value for key, value in BASE_FROZEN_SETTINGS.items() if key != "generator_output_mode"}
    if "uno_n_modes" in parameter_names:
        kwargs["uno_n_modes"] = uno_n_modes
    else:
        kwargs.pop("uno_out_channels", None)
        kwargs.pop("uno_scalings", None)
    return kwargs


def _instantiate_with_mode_probe(uno_cls: type[Any], signature_payload: dict[str, Any]):
    preferred = [[12, 12], [12, 12], [12, 12], [12, 12]]
    fallback = 12
    preferred_kwargs = _build_model_kwargs(signature_payload, uno_n_modes=preferred)
    try:
        model = uno_cls(**preferred_kwargs)
        settings = dict(BASE_FROZEN_SETTINGS)
        settings["uno_n_modes"] = preferred
        settings["uno_n_modes_form"] = "per_layer_sequence"
        settings["uno_n_modes_note"] = "Accepted explicit four-entry per-layer mode sequence."
        return model, settings
    except TypeError as exc:
        fallback_kwargs = _build_model_kwargs(signature_payload, uno_n_modes=fallback)
        try:
            model = uno_cls(**fallback_kwargs)
        except TypeError:
            raise exc
        settings = dict(BASE_FROZEN_SETTINGS)
        settings["uno_n_modes"] = fallback
        settings["uno_n_modes_form"] = "scalar_fallback"
        settings["uno_n_modes_note"] = (
            "Preferred explicit per-layer mode sequence was rejected; "
            "accepted scalar fallback after constructor retry."
        )
        return model, settings


def _assess_output_shape(output: torch.Tensor) -> dict[str, Any]:
    shape = [int(dim) for dim in output.shape]
    if len(shape) == 4 and shape[1] == 2 and shape[2] == LOCKED_CONTRACT["N"] and shape[3] == LOCKED_CONTRACT["N"]:
        return {
            "attempted": True,
            "accepted": True,
            "raw_output_shape": shape,
            "mapping": "bchw_real_imag",
            "normalized_output_shape": [shape[0], LOCKED_CONTRACT["N"], LOCKED_CONTRACT["N"], 1, 2],
            "adapter_required": False,
            "reason": "Raw UNO output already exposes the expected real/imag channel count.",
        }
    if len(shape) == 5 and shape[1] == 1 and shape[2] == LOCKED_CONTRACT["N"] and shape[3] == LOCKED_CONTRACT["N"] and shape[4] == 2:
        return {
            "attempted": True,
            "accepted": True,
            "raw_output_shape": shape,
            "mapping": "bchw2_to_bhwc1r2",
            "normalized_output_shape": [shape[0], LOCKED_CONTRACT["N"], LOCKED_CONTRACT["N"], 1, 2],
            "adapter_required": True,
            "reason": "A thin layout adapter can losslessly map the raw UNO output into the real/imag wrapper contract.",
        }
    return {
        "attempted": True,
        "accepted": False,
        "raw_output_shape": shape,
        "mapping": None,
        "normalized_output_shape": None,
        "adapter_required": None,
        "reason": "Raw UNO output cannot be mapped losslessly into the locked real_imag CDI contract.",
    }


def _package_provenance(neuralop_module: Any | None, pip_show_text: str, *, repo_root: Path) -> dict[str, Any]:
    return {
        "distribution_name": "neuraloperator",
        "module_name": "neuralop",
        "module_version": getattr(neuralop_module, "__version__", None) if neuralop_module is not None else None,
        "module_file": str(Path(neuralop_module.__file__).resolve()) if neuralop_module is not None and getattr(neuralop_module, "__file__", None) else None,
        "pip_show_present": bool(pip_show_text.strip()),
        "git_commit": get_git_commit(repo_root),
        "git_dirty": git_dirty(repo_root),
    }


def _environment_payload(runtime_provenance: dict[str, Any], package_provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_provenance": runtime_provenance,
        "package_provenance": package_provenance,
    }


def _write_artifacts(
    output_root: Path,
    *,
    runtime_provenance: dict[str, Any],
    package_provenance: dict[str, Any],
    pip_show_text: str,
    uno_signature: dict[str, Any],
    shape_probe: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    write_json(output_root / "environment_probe.json", _environment_payload(runtime_provenance, package_provenance))
    (output_root / "pip_show_neuraloperator.txt").write_text(pip_show_text, encoding="utf-8")
    write_json(output_root / "uno_signature.json", uno_signature)
    write_json(output_root / "uno_shape_probe.json", shape_probe)
    write_json(output_root / "preflight_decision.json", decision)


def _blocked_decision(
    *,
    status: str,
    package_status: str,
    runtime_provenance: dict[str, Any],
    package_provenance: dict[str, Any],
    uno_signature: dict[str, Any],
    shape_probe: dict[str, Any],
    blocker_reason: str,
    next_item_recommendation: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "package_status": package_status,
        "runtime_provenance": runtime_provenance,
        "package_provenance": package_provenance,
        "uno_signature": uno_signature,
        "locked_contract": dict(LOCKED_CONTRACT),
        "frozen_uno_settings": dict(BASE_FROZEN_SETTINGS),
        "shape_probe": shape_probe,
        "next_item_recommendation": next_item_recommendation,
        "blocker_reason": blocker_reason,
    }


def run_lines128_uno_preflight(*, output_root: Path, repo_root: Path | None = None) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root or Path.cwd())

    pip_show_text = _capture_pip_show_text()
    runtime_provenance = current_runtime_provenance()

    empty_signature = {"text": None, "parameters": [], "parameter_names": [], "error": None}
    not_attempted_shape = {
        "attempted": False,
        "accepted": False,
        "raw_output_shape": None,
        "mapping": None,
        "normalized_output_shape": None,
        "adapter_required": None,
        "reason": "UNO forward probe was not attempted.",
    }

    try:
        neuralop_module, uno_cls = _import_uno_dependencies()
    except Exception as exc:
        package_provenance = _package_provenance(None, pip_show_text, repo_root=repo_root)
        empty_signature["error"] = f"{type(exc).__name__}: {exc}"
        decision = _blocked_decision(
            status=BLOCKED_PACKAGE_STATUS,
            package_status="missing_or_incompatible",
            runtime_provenance=runtime_provenance,
            package_provenance=package_provenance,
            uno_signature=empty_signature,
            shape_probe=not_attempted_shape,
            blocker_reason=f"Unable to import neuralop.models.UNO: {exc}",
            next_item_recommendation="resolve_neuraloperator_environment",
        )
        _write_artifacts(
            output_root,
            runtime_provenance=runtime_provenance,
            package_provenance=package_provenance,
            pip_show_text=pip_show_text,
            uno_signature=empty_signature,
            shape_probe=not_attempted_shape,
            decision=decision,
        )
        return decision

    package_provenance = _package_provenance(neuralop_module, pip_show_text, repo_root=repo_root)
    signature_payload = _signature_payload(uno_cls)

    try:
        model, frozen_settings = _instantiate_with_mode_probe(uno_cls, signature_payload)
    except TypeError as exc:
        blocked_shape = dict(not_attempted_shape)
        decision = _blocked_decision(
            status=BLOCKED_PACKAGE_STATUS,
            package_status="constructor_incompatible",
            runtime_provenance=runtime_provenance,
            package_provenance=package_provenance,
            uno_signature=signature_payload,
            shape_probe=blocked_shape,
            blocker_reason=f"UNO constructor could not satisfy the frozen lines128 preflight settings: {exc}",
            next_item_recommendation="resolve_uno_constructor_contract",
        )
        _write_artifacts(
            output_root,
            runtime_provenance=runtime_provenance,
            package_provenance=package_provenance,
            pip_show_text=pip_show_text,
            uno_signature=signature_payload,
            shape_probe=blocked_shape,
            decision=decision,
        )
        return decision

    with torch.no_grad():
        dummy = torch.zeros((2, 1, LOCKED_CONTRACT["N"], LOCKED_CONTRACT["N"]), dtype=torch.float32)
        output = model(dummy)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"UNO forward probe returned {type(output)!r}, expected torch.Tensor")

    shape_probe = _assess_output_shape(output)
    if not shape_probe["accepted"]:
        decision = {
            "status": BLOCKED_SHAPE_STATUS,
            "package_status": "ok",
            "runtime_provenance": runtime_provenance,
            "package_provenance": package_provenance,
            "uno_signature": signature_payload,
            "locked_contract": dict(LOCKED_CONTRACT),
            "frozen_uno_settings": frozen_settings,
            "shape_probe": shape_probe,
            "next_item_recommendation": "resolve_uno_output_contract_before_generator_integration",
            "blocker_reason": shape_probe["reason"],
        }
        _write_artifacts(
            output_root,
            runtime_provenance=runtime_provenance,
            package_provenance=package_provenance,
            pip_show_text=pip_show_text,
            uno_signature=signature_payload,
            shape_probe=shape_probe,
            decision=decision,
        )
        return decision

    decision = {
        "status": READY_STATUS,
        "package_status": "ok",
        "runtime_provenance": runtime_provenance,
        "package_provenance": package_provenance,
        "uno_signature": signature_payload,
        "locked_contract": dict(LOCKED_CONTRACT),
        "frozen_uno_settings": frozen_settings,
        "shape_probe": shape_probe,
        "next_item_recommendation": "2026-04-30-cdi-lines128-uno-generator-integration",
        "blocker_reason": None,
    }
    _write_artifacts(
        output_root,
        runtime_provenance=runtime_provenance,
        package_provenance=package_provenance,
        pip_show_text=pip_show_text,
        uno_signature=signature_payload,
        shape_probe=shape_probe,
        decision=decision,
    )
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for preflight artifacts.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    decision = run_lines128_uno_preflight(output_root=args.output_root)
    print(
        f"{decision['status']}: "
        f"{decision['shape_probe'].get('raw_output_shape') or decision['blocker_reason']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
