#!/usr/bin/env python3
"""Preflight and bounded validation helpers for the lines128 paper benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare
from scripts.studies.metrics_tables import write_paper_benchmark_bundle


FNO_COMPARATOR_MODEL_IDS = {
    "fno": "pinn_fno",
    "fno_vanilla": "pinn_fno_vanilla",
}
ALLOWED_GO_NO_GO_STATE = "go_for_harness_preflight_only"
MINIMUM_SUBSET_EXECUTION_STATE = "go_for_minimum_subset_execution"
AUTHORITY_JSON_START = "<!-- lines128_execution_authority_json:start -->"
AUTHORITY_JSON_END = "<!-- lines128_execution_authority_json:end -->"
REQUIRED_FIXED_CONTRACT_FIELDS = (
    "N",
    "gridsize",
    "dataset_source",
    "set_phi",
    "probe_source",
    "probe_npz",
    "probe_scale_mode",
    "probe_smoothing_sigma",
    "probe_mask_diameter",
    "nimgs_train",
    "nimgs_test",
    "nphotons",
    "seed",
    "torch_epochs",
    "torch_learning_rate",
    "torch_scheduler",
    "torch_plateau_factor",
    "torch_plateau_patience",
    "torch_plateau_min_lr",
    "torch_plateau_threshold",
    "torch_loss_mode",
    "torch_mae_pred_l2_match_target",
    "torch_output_mode",
    "fno_modes",
    "fno_width",
    "fno_blocks",
    "fno_cnn_blocks",
)


def _normalize_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Decision artifact fixed contract field '{field_name}' must be a boolean")
    return value


def _normalize_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Decision artifact fixed contract field '{field_name}' must be an integer")
    return int(value)


def _normalize_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Decision artifact fixed contract field '{field_name}' must be numeric")
    return float(value)


def _normalize_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _normalize_int(value, field_name=field_name)


def _normalize_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Decision artifact fixed contract field '{field_name}' must be a non-empty string")
    return value


def _load_decision_artifact(path: Path) -> Dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing decision artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Decision artifact must be a JSON object")
    return payload


def _normalize_relaxed_json_payload(payload: object, *, source_name: str) -> Dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source_name} must be a JSON object")
    return dict(payload)


def _resolve_repo_path(path_value: object) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Decision artifact contract note is missing a valid path")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _validate_contract_note(payload: Mapping[str, object]) -> Path:
    contract_note = payload.get("contract_note")
    if not isinstance(contract_note, Mapping):
        raise ValueError("Decision artifact missing contract note metadata")
    if contract_note.get("status") != "resolved":
        raise ValueError("Preflight contract note is unresolved")
    contract_note_path = _resolve_repo_path(contract_note.get("path"))
    if not contract_note_path.exists():
        raise FileNotFoundError(f"Missing resolved contract note for harness preflight: {contract_note_path}")
    return contract_note_path


def _validate_fixed_contract(payload: Mapping[str, object]) -> Dict[str, Any]:
    contract = payload.get("fixed_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Decision artifact missing fixed contract metadata")
    missing = [field for field in REQUIRED_FIXED_CONTRACT_FIELDS if field not in contract]
    if missing:
        raise ValueError(f"Decision artifact fixed contract is missing required fields: {', '.join(missing)}")

    normalized = {
        "N": _normalize_int(contract["N"], field_name="N"),
        "gridsize": _normalize_int(contract["gridsize"], field_name="gridsize"),
        "dataset_source": _normalize_str(contract["dataset_source"], field_name="dataset_source"),
        "set_phi": _normalize_bool(contract["set_phi"], field_name="set_phi"),
        "probe_source": _normalize_str(contract["probe_source"], field_name="probe_source"),
        "probe_npz": _normalize_str(contract["probe_npz"], field_name="probe_npz"),
        "probe_scale_mode": _normalize_str(contract["probe_scale_mode"], field_name="probe_scale_mode"),
        "probe_smoothing_sigma": _normalize_float(
            contract["probe_smoothing_sigma"], field_name="probe_smoothing_sigma"
        ),
        "probe_mask_diameter": _normalize_optional_int(
            contract["probe_mask_diameter"],
            field_name="probe_mask_diameter",
        ),
        "nimgs_train": _normalize_int(contract["nimgs_train"], field_name="nimgs_train"),
        "nimgs_test": _normalize_int(contract["nimgs_test"], field_name="nimgs_test"),
        "nphotons": _normalize_float(contract["nphotons"], field_name="nphotons"),
        "seed": _normalize_int(contract["seed"], field_name="seed"),
        "torch_epochs": _normalize_int(contract["torch_epochs"], field_name="torch_epochs"),
        "torch_learning_rate": _normalize_float(
            contract["torch_learning_rate"],
            field_name="torch_learning_rate",
        ),
        "torch_scheduler": _normalize_str(contract["torch_scheduler"], field_name="torch_scheduler"),
        "torch_plateau_factor": _normalize_float(
            contract["torch_plateau_factor"],
            field_name="torch_plateau_factor",
        ),
        "torch_plateau_patience": _normalize_int(
            contract["torch_plateau_patience"],
            field_name="torch_plateau_patience",
        ),
        "torch_plateau_min_lr": _normalize_float(
            contract["torch_plateau_min_lr"],
            field_name="torch_plateau_min_lr",
        ),
        "torch_plateau_threshold": _normalize_float(
            contract["torch_plateau_threshold"],
            field_name="torch_plateau_threshold",
        ),
        "torch_loss_mode": _normalize_str(contract["torch_loss_mode"], field_name="torch_loss_mode"),
        "torch_mae_pred_l2_match_target": _normalize_bool(
            contract["torch_mae_pred_l2_match_target"],
            field_name="torch_mae_pred_l2_match_target",
        ),
        "torch_output_mode": _normalize_str(contract["torch_output_mode"], field_name="torch_output_mode"),
        "fno_modes": _normalize_int(contract["fno_modes"], field_name="fno_modes"),
        "fno_width": _normalize_int(contract["fno_width"], field_name="fno_width"),
        "fno_blocks": _normalize_int(contract["fno_blocks"], field_name="fno_blocks"),
        "fno_cnn_blocks": _normalize_int(contract["fno_cnn_blocks"], field_name="fno_cnn_blocks"),
    }
    return normalized


def _validate_fixed_contract_provenance(payload: Mapping[str, object], fixed_contract: Mapping[str, object]) -> None:
    provenance = payload.get("fixed_contract_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Decision artifact missing fixed contract provenance metadata")
    missing = [field for field in fixed_contract if field not in provenance]
    if missing:
        raise ValueError(
            "Decision artifact fixed contract provenance is missing fields: " + ", ".join(missing)
        )
    for field in fixed_contract:
        record = provenance.get(field)
        if not isinstance(record, Mapping):
            raise ValueError(f"Decision artifact provenance for '{field}' must be an object")
        confidence = record.get("confidence")
        sources = record.get("sources")
        if not isinstance(confidence, str) or not confidence.strip():
            raise ValueError(f"Decision artifact provenance for '{field}' must include confidence")
        if not isinstance(sources, list) or not sources or not all(
            isinstance(source, str) and source.strip() for source in sources
        ):
            raise ValueError(f"Decision artifact provenance for '{field}' must include non-empty sources")


def _validate_seed_policy(payload: Mapping[str, object], fixed_contract: Mapping[str, object]) -> int:
    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, Mapping):
        raise ValueError("Decision artifact missing seed policy metadata")
    if seed_policy.get("type") != "fixed":
        raise ValueError("Harness preflight requires a fixed seed policy")
    seed = _normalize_int(seed_policy.get("seed"), field_name="seed_policy.seed")
    if seed != int(fixed_contract["seed"]):
        raise ValueError("Decision artifact seed policy must match the fixed contract seed")
    return seed


def _validate_go_no_go(payload: Mapping[str, object]) -> None:
    go_no_go = payload.get("go_no_go")
    if not isinstance(go_no_go, Mapping):
        raise ValueError("Decision artifact missing go/no-go metadata")
    if go_no_go.get("state") != ALLOWED_GO_NO_GO_STATE:
        raise ValueError("Decision artifact go/no-go state does not authorize harness preflight only")
    if go_no_go.get("full_benchmark_launch_authorized") is not False:
        raise ValueError("Decision artifact go/no-go must keep full benchmark launch disabled")


def _validate_approved_deviations(payload: Mapping[str, object]) -> None:
    deviations = payload.get("approved_deviations")
    if not isinstance(deviations, list):
        raise ValueError("Decision artifact must include an approved_deviations list")


def _expected_runtime_contract(fixed_contract: Mapping[str, object]) -> Dict[str, Any]:
    expected = dict(fixed_contract)
    expected["probe_npz"] = str(_resolve_repo_path(fixed_contract["probe_npz"]))
    return expected


def _validate_compare_preflight_contract(
    compare_preflight: Mapping[str, object],
    fixed_contract: Mapping[str, object],
    *,
    seed: int,
) -> None:
    if compare_preflight.get("seed") != seed:
        raise ValueError("Compare-wrapper preflight seed drifted from the decision artifact seed policy")
    runtime_contract = compare_preflight.get("contract")
    if not isinstance(runtime_contract, Mapping):
        raise ValueError("Compare-wrapper preflight did not return contract metadata")
    expected_contract = _expected_runtime_contract(fixed_contract)
    drift = [
        field
        for field, expected_value in expected_contract.items()
        if runtime_contract.get(field) != expected_value
    ]
    if drift:
        raise ValueError("Compare-wrapper preflight contract drift detected for: " + ", ".join(drift))


def _validate_compare_preflight_rows(
    compare_preflight: Mapping[str, object],
    *,
    supported_model_ids: Iterable[str],
) -> None:
    expected_models = list(supported_model_ids)

    selected_models = compare_preflight.get("selected_models")
    if not isinstance(selected_models, list) or any(not isinstance(model_id, str) for model_id in selected_models):
        raise ValueError("Compare-wrapper preflight did not return selected_models")
    if selected_models != expected_models:
        raise ValueError("Compare-wrapper preflight selected_models drifted from supported decision rows")

    row_plan = compare_preflight.get("row_plan")
    if not isinstance(row_plan, list):
        raise ValueError("Compare-wrapper preflight did not return row_plan")

    row_plan_ids: List[str] = []
    for item in row_plan:
        if not isinstance(item, Mapping) or not isinstance(item.get("model_id"), str):
            raise ValueError("Compare-wrapper preflight row_plan contains an invalid row entry")
        model_id = str(item["model_id"])
        row_plan_ids.append(model_id)
        if item.get("status") != "supported_for_harness":
            raise ValueError(f"Compare-wrapper preflight row_plan has unsupported status for {model_id}")

    if row_plan_ids != expected_models:
        raise ValueError("Compare-wrapper preflight row_plan drifted from supported decision rows")


def _validate_fno_comparator(payload: Mapping[str, object], rows_by_id: Mapping[str, Mapping[str, object]]) -> str:
    comparator = payload.get("selected_fno_comparator")
    if comparator not in FNO_COMPARATOR_MODEL_IDS:
        raise ValueError("Decision artifact must select an explicit FNO comparator")
    comparator_model_id = FNO_COMPARATOR_MODEL_IDS[str(comparator)]
    comparator_row = rows_by_id.get(comparator_model_id)
    if comparator_row is None or comparator_row.get("status") != "supported_for_harness":
        raise ValueError("Selected FNO comparator is not supported_for_harness")
    return str(comparator)


def _extract_authority_payload(path: Path) -> Dict[str, object]:
    note_path = Path(path)
    if not note_path.exists():
        raise FileNotFoundError(f"Missing execution authority note: {note_path}")
    text = note_path.read_text(encoding="utf-8")
    try:
        start = text.index(AUTHORITY_JSON_START) + len(AUTHORITY_JSON_START)
        end = text.index(AUTHORITY_JSON_END, start)
    except ValueError as exc:
        raise ValueError("Execution authority note is missing the embedded JSON payload") from exc
    payload_text = text[start:end].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Execution authority note contains invalid JSON") from exc
    return _normalize_relaxed_json_payload(payload, source_name="Execution authority note")


def _minimum_subset_row_key(row: Mapping[str, object]) -> tuple[str, str, str, bool]:
    model_id = row.get("model_id")
    model_label = row.get("model_label")
    architecture_id = row.get("architecture_id")
    training_procedure = row.get("training_procedure")
    required = row.get("required_for_minimum_subset")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Minimum-subset execution row is missing model_id")
    if not isinstance(model_label, str) or not model_label.strip():
        raise ValueError(f"Minimum-subset execution row {model_id} is missing model_label")
    if not isinstance(architecture_id, str) or not architecture_id.strip():
        raise ValueError(f"Minimum-subset execution row {model_id} is missing architecture_id")
    if training_procedure not in {"supervised", "pinn"}:
        raise ValueError(
            f"Minimum-subset execution row {model_id} has unsupported training_procedure={training_procedure!r}"
        )
    if not isinstance(required, bool):
        raise ValueError(f"Minimum-subset execution row {model_id} must declare required_for_minimum_subset")
    return str(model_id), str(model_label), str(architecture_id), bool(required)


def _normalize_execution_surface(
    payload: Mapping[str, object],
    *,
    fixed_contract: Mapping[str, object],
    source_name: str,
) -> Dict[str, object]:
    state = payload.get("state")
    if state != MINIMUM_SUBSET_EXECUTION_STATE:
        raise ValueError(f"{source_name} does not authorize minimum-subset execution")
    comparator = payload.get("selected_fno_comparator")
    if comparator not in FNO_COMPARATOR_MODEL_IDS:
        raise ValueError(f"{source_name} must select an explicit FNO comparator")

    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, Mapping):
        raise ValueError(f"{source_name} is missing seed_policy")
    normalized_seed = _validate_seed_policy({"seed_policy": seed_policy, "fixed_contract": fixed_contract}, fixed_contract)

    payload_contract = _validate_fixed_contract({"fixed_contract": payload.get("fixed_contract")})
    if payload_contract != dict(fixed_contract):
        raise ValueError(f"{source_name} fixed contract drifted from the harness decision artifact")

    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError(f"{source_name} must authorize exactly four minimum-subset rows")
    normalized_rows: List[Dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{source_name} row entries must be JSON objects")
        model_id, model_label, architecture_id, required = _minimum_subset_row_key(row)
        normalized_rows.append(
            {
                "model_id": model_id,
                "model_label": model_label,
                "architecture_id": architecture_id,
                "training_procedure": str(row["training_procedure"]),
                "required_for_minimum_subset": required,
            }
        )
    required_rows = [row["model_id"] for row in normalized_rows if row["required_for_minimum_subset"]]
    if required_rows != ["baseline", "pinn", "pinn_hybrid_resnet", "pinn_fno_vanilla"]:
        raise ValueError(
            f"{source_name} must lock the required four-row roster in order; got {required_rows}"
        )

    baseline_row = normalized_rows[0]
    pinn_row = normalized_rows[1]
    if baseline_row["architecture_id"] != "cnn" or pinn_row["architecture_id"] != "cnn":
        raise ValueError(f"{source_name} must keep the paired CDI cnn supervised/PINN rows")
    if baseline_row["training_procedure"] != "supervised" or pinn_row["training_procedure"] != "pinn":
        raise ValueError(f"{source_name} must distinguish the CDI cnn supervised and PINN rows")
    if baseline_row["model_label"] == pinn_row["model_label"]:
        raise ValueError(f"{source_name} must not collapse the supervised and PINN CDI cnn labels")

    fixed_sample_ids = payload.get("fixed_sample_ids")
    if not isinstance(fixed_sample_ids, list) or any(not isinstance(item, int) for item in fixed_sample_ids):
        raise ValueError(f"{source_name} must provide integer fixed_sample_ids")
    shared_visual_scales = payload.get("shared_visual_scales")
    if not isinstance(shared_visual_scales, Mapping):
        raise ValueError(f"{source_name} must provide shared_visual_scales")
    later_rows = payload.get("later_complete_table_rows")
    if later_rows != ["pinn_spectral_resnet_bottleneck_net", "pinn_ffno"]:
        raise ValueError(
            f"{source_name} must record spectral and FFNO as the later complete-table rows"
        )

    return {
        "state": MINIMUM_SUBSET_EXECUTION_STATE,
        "selected_fno_comparator": str(comparator),
        "seed_policy": {"type": "fixed", "seed": normalized_seed},
        "fixed_contract": dict(payload_contract),
        "fixed_sample_ids": list(fixed_sample_ids),
        "shared_visual_scales": dict(shared_visual_scales),
        "rows": normalized_rows,
        "later_complete_table_rows": list(later_rows),
    }


def _load_execution_manifest(
    path: Path,
    *,
    fixed_contract: Mapping[str, object],
) -> Dict[str, object]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing execution manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _normalize_execution_surface(payload, fixed_contract=fixed_contract, source_name="Execution manifest")


def _run_compare_execution(
    *,
    execution_surface: Mapping[str, object],
    fixed_contract: Mapping[str, object],
    output_dir: Path,
    reuse_existing_recons: bool = False,
) -> Dict[str, object]:
    rows = execution_surface["rows"]
    selected_models = tuple(str(row["model_id"]) for row in rows if row["required_for_minimum_subset"])
    model_n = {model_id: int(fixed_contract["N"]) for model_id in selected_models}
    return run_grid_lines_compare(
        N=int(fixed_contract["N"]),
        gridsize=int(fixed_contract["gridsize"]),
        output_dir=output_dir,
        probe_npz=_resolve_repo_path(fixed_contract["probe_npz"]),
        architectures=(),
        models=selected_models,
        model_n=model_n,
        seed=int(fixed_contract["seed"]),
        nimgs_train=int(fixed_contract["nimgs_train"]),
        nimgs_test=int(fixed_contract["nimgs_test"]),
        nphotons=float(fixed_contract["nphotons"]),
        set_phi=bool(fixed_contract["set_phi"]),
        probe_source=str(fixed_contract["probe_source"]),
        probe_scale_mode=str(fixed_contract["probe_scale_mode"]),
        probe_smoothing_sigma=float(fixed_contract["probe_smoothing_sigma"]),
        probe_mask_diameter=fixed_contract["probe_mask_diameter"],
        torch_epochs=int(fixed_contract["torch_epochs"]),
        torch_learning_rate=float(fixed_contract["torch_learning_rate"]),
        torch_scheduler=str(fixed_contract["torch_scheduler"]),
        torch_plateau_factor=float(fixed_contract["torch_plateau_factor"]),
        torch_plateau_patience=int(fixed_contract["torch_plateau_patience"]),
        torch_plateau_min_lr=float(fixed_contract["torch_plateau_min_lr"]),
        torch_plateau_threshold=float(fixed_contract["torch_plateau_threshold"]),
        torch_loss_mode=str(fixed_contract["torch_loss_mode"]),
        torch_mae_pred_l2_match_target=bool(fixed_contract["torch_mae_pred_l2_match_target"]),
        torch_output_mode=str(fixed_contract["torch_output_mode"]),
        fno_modes=int(fixed_contract["fno_modes"]),
        fno_width=int(fixed_contract["fno_width"]),
        fno_blocks=int(fixed_contract["fno_blocks"]),
        fno_cnn_blocks=int(fixed_contract["fno_cnn_blocks"]),
        dataset_source=str(fixed_contract["dataset_source"]),
        reuse_existing_recons=reuse_existing_recons,
    )


def _row_statuses(rows_by_id: Mapping[str, Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    statuses: Dict[str, Dict[str, object]] = {}
    for model_id, row in rows_by_id.items():
        status = str(row.get("status", "row_blocker"))
        statuses[model_id] = {"status": status}
        if "blocker_reason" in row:
            statuses[model_id]["reason"] = row["blocker_reason"]
    return statuses


def _required_rows(rows_by_id: Mapping[str, Mapping[str, object]]) -> List[str]:
    return [
        model_id
        for model_id, row in rows_by_id.items()
        if row.get("required_for_minimum_subset")
    ]


def _supported_rows(rows_by_id: Mapping[str, Mapping[str, object]]) -> List[Mapping[str, object]]:
    return [
        row
        for row in rows_by_id.values()
        if row.get("status") == "supported_for_harness"
    ]


def _validate_required_rows(required_rows: Iterable[str], row_statuses: Mapping[str, Mapping[str, object]]) -> None:
    for required_model in required_rows:
        status = row_statuses.get(required_model, {}).get("status")
        if status != "supported_for_harness":
            raise ValueError(f"Minimum required row is not supported_for_harness: {required_model}")


def _run_compare_preflight(
    *,
    supported_rows: Iterable[Mapping[str, object]],
    fixed_contract: Mapping[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    supported_rows = list(supported_rows)
    selected_models = tuple(str(row["model_id"]) for row in supported_rows)
    model_n = {
        str(row["model_id"]): int(row.get("N", fixed_contract["N"]))
        for row in supported_rows
    }
    return run_grid_lines_compare(
        N=int(fixed_contract["N"]),
        gridsize=int(fixed_contract["gridsize"]),
        output_dir=output_dir / "compare_wrapper_preflight",
        probe_npz=_resolve_repo_path(fixed_contract["probe_npz"]),
        architectures=(),
        models=selected_models,
        model_n=model_n,
        seed=int(fixed_contract["seed"]),
        nimgs_train=int(fixed_contract["nimgs_train"]),
        nimgs_test=int(fixed_contract["nimgs_test"]),
        nphotons=float(fixed_contract["nphotons"]),
        set_phi=bool(fixed_contract["set_phi"]),
        probe_source=str(fixed_contract["probe_source"]),
        probe_scale_mode=str(fixed_contract["probe_scale_mode"]),
        probe_smoothing_sigma=float(fixed_contract["probe_smoothing_sigma"]),
        probe_mask_diameter=fixed_contract["probe_mask_diameter"],
        torch_epochs=int(fixed_contract["torch_epochs"]),
        torch_learning_rate=float(fixed_contract["torch_learning_rate"]),
        torch_scheduler=str(fixed_contract["torch_scheduler"]),
        torch_plateau_factor=float(fixed_contract["torch_plateau_factor"]),
        torch_plateau_patience=int(fixed_contract["torch_plateau_patience"]),
        torch_plateau_min_lr=float(fixed_contract["torch_plateau_min_lr"]),
        torch_plateau_threshold=float(fixed_contract["torch_plateau_threshold"]),
        torch_loss_mode=str(fixed_contract["torch_loss_mode"]),
        torch_mae_pred_l2_match_target=bool(fixed_contract["torch_mae_pred_l2_match_target"]),
        torch_output_mode=str(fixed_contract["torch_output_mode"]),
        fno_modes=int(fixed_contract["fno_modes"]),
        fno_width=int(fixed_contract["fno_width"]),
        fno_blocks=int(fixed_contract["fno_blocks"]),
        fno_cnn_blocks=int(fixed_contract["fno_cnn_blocks"]),
        dataset_source=str(fixed_contract["dataset_source"]),
        preflight_only=True,
    )


def _build_row_payloads(
    *,
    rows_by_id: Mapping[str, Mapping[str, object]],
    compare_preflight: Mapping[str, object],
    fixed_contract: Mapping[str, object],
) -> Dict[str, Dict[str, object]]:
    row_plan = compare_preflight.get("row_plan")
    if not isinstance(row_plan, list):
        raise ValueError("Compare wrapper preflight did not return row_plan")
    preflight_rows = {
        str(item["model_id"]): item
        for item in row_plan
        if isinstance(item, Mapping) and "model_id" in item
    }

    row_payloads: Dict[str, Dict[str, object]] = {}
    for model_id, row in rows_by_id.items():
        if row.get("status") != "supported_for_harness":
            continue
        preflight_row = preflight_rows.get(model_id, {})
        row_payloads[model_id] = {
            "model_label": row.get("model_label", model_id),
            "N": int(preflight_row.get("N", row.get("N", fixed_contract["N"]))),
            # Readiness bundles must use the real preflight route but remain
            # benchmark_incomplete until later execution produces row metrics.
            "metrics": {},
        }
    return row_payloads


def run_lines128_paper_benchmark_preflight(
    *,
    decision_artifact: Path,
    output_dir: Path,
) -> Dict[str, object]:
    payload = _load_decision_artifact(decision_artifact)
    contract_note_path = _validate_contract_note(payload)
    fixed_contract = _validate_fixed_contract(payload)
    _validate_fixed_contract_provenance(payload, fixed_contract)
    seed = _validate_seed_policy(payload, fixed_contract)
    _validate_go_no_go(payload)
    _validate_approved_deviations(payload)

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Decision artifact must define at least one row")
    rows_by_id = {
        str(row["model_id"]): row
        for row in rows
        if isinstance(row, Mapping) and "model_id" in row
    }
    comparator = _validate_fno_comparator(payload, rows_by_id)
    row_statuses = _row_statuses(rows_by_id)
    required_rows = _required_rows(rows_by_id)
    _validate_required_rows(required_rows, row_statuses)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    supported_model_ids = [str(row["model_id"]) for row in _supported_rows(rows_by_id)]
    compare_preflight = _run_compare_preflight(
        supported_rows=[rows_by_id[model_id] for model_id in supported_model_ids],
        fixed_contract=fixed_contract,
        output_dir=output_dir,
    )
    _validate_compare_preflight_contract(compare_preflight, fixed_contract, seed=seed)
    _validate_compare_preflight_rows(compare_preflight, supported_model_ids=supported_model_ids)
    compare_preflight_path = output_dir / "compare_wrapper_preflight.json"
    compare_preflight_path.write_text(json.dumps(compare_preflight, indent=2), encoding="utf-8")

    row_payloads = _build_row_payloads(
        rows_by_id=rows_by_id,
        compare_preflight=compare_preflight,
        fixed_contract=fixed_contract,
    )
    bundle_paths = write_paper_benchmark_bundle(
        output_dir=output_dir,
        row_payloads=row_payloads,
        required_rows=tuple(required_rows),
        fixed_sample_ids=payload.get("fixed_sample_ids", []),
        shared_visual_scales=payload.get("shared_visual_scales", {}),
        selected_fno_comparator=comparator,
        row_statuses=row_statuses,
        evidence_scope="readiness_only_not_benchmark_performance",
    )
    bundle_paths["compare_preflight_json"] = str(compare_preflight_path)
    return {
        "contract_note_path": str(contract_note_path),
        "fixed_contract": dict(fixed_contract),
        "selected_models": list(compare_preflight.get("selected_models", [])),
        "required_rows": required_rows,
        "compare_preflight": compare_preflight,
        "bundle_paths": bundle_paths,
    }


def run_lines128_paper_benchmark(
    *,
    decision_artifact: Path,
    execution_authority_note: Path,
    execution_manifest: Path,
    output_dir: Path,
    reuse_existing_recons: bool = False,
) -> Dict[str, object]:
    payload = _load_decision_artifact(decision_artifact)
    fixed_contract = _validate_fixed_contract(payload)
    comparator = _validate_fno_comparator(
        payload,
        {
            str(row["model_id"]): row
            for row in payload.get("rows", [])
            if isinstance(row, Mapping) and "model_id" in row
        },
    )

    authority_payload = _extract_authority_payload(execution_authority_note)
    authority_surface = _normalize_execution_surface(
        authority_payload,
        fixed_contract=fixed_contract,
        source_name="Execution authority note",
    )
    if authority_surface["selected_fno_comparator"] != comparator:
        raise ValueError("Execution authority note selected_fno_comparator drifted from the harness decision artifact")

    manifest_surface = _load_execution_manifest(
        execution_manifest,
        fixed_contract=fixed_contract,
    )
    if manifest_surface != authority_surface:
        raise ValueError("Derived execution manifest drifted from the checked-in execution authority note")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_result = _run_compare_execution(
        execution_surface=authority_surface,
        fixed_contract=fixed_contract,
        output_dir=output_dir,
        reuse_existing_recons=reuse_existing_recons,
    )

    row_payloads = compare_result.get("row_payloads")
    if not isinstance(row_payloads, Mapping):
        raise ValueError("Compare wrapper did not return row_payloads for minimum-subset execution")

    required_rows = [str(row["model_id"]) for row in authority_surface["rows"] if row["required_for_minimum_subset"]]
    missing_rows = [model_id for model_id in required_rows if model_id not in row_payloads]
    if missing_rows:
        raise ValueError(
            "Compare wrapper row_payloads are missing required rows: " + ", ".join(missing_rows)
        )

    bundle_paths = write_paper_benchmark_bundle(
        output_dir=output_dir,
        row_payloads=row_payloads,
        required_rows=required_rows,
        fixed_sample_ids=authority_surface["fixed_sample_ids"],
        shared_visual_scales=authority_surface["shared_visual_scales"],
        selected_fno_comparator=authority_surface["selected_fno_comparator"],
        evidence_scope="minimum_subset_benchmark_execution",
        claim_boundary="minimum_draftable_cdi_subset",
    )
    return {
        "required_rows": required_rows,
        "bundle_paths": bundle_paths,
        "compare_result": compare_result,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run lines128 paper benchmark utilities")
    parser.add_argument("--mode", choices=("preflight", "minimum_subset"), default="preflight")
    parser.add_argument("--decision-artifact", type=Path, required=True)
    parser.add_argument("--execution-authority-note", type=Path)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-existing-recons",
        action="store_true",
        help="Recover the minimum-subset bundle from existing recon and row-local artifacts in output-dir.",
    )
    return parser.parse_args(argv)


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
        script_path="scripts/studies/lines128_paper_benchmark.py",
        argv=raw_argv,
        parsed_args=vars(args),
        extra={
            "runtime_provenance": capture_runtime_provenance(),
            "git_commit": get_git_commit(REPO_ROOT),
        },
    )
    try:
        if args.mode == "preflight":
            run_lines128_paper_benchmark_preflight(
                decision_artifact=args.decision_artifact,
                output_dir=args.output_dir,
            )
        else:
            if args.execution_authority_note is None or args.execution_manifest is None:
                raise ValueError(
                    "minimum_subset mode requires --execution-authority-note and --execution-manifest"
                )
            run_lines128_paper_benchmark(
                decision_artifact=args.decision_artifact,
                execution_authority_note=args.execution_authority_note,
                execution_manifest=args.execution_manifest,
                output_dir=args.output_dir,
                reuse_existing_recons=args.reuse_existing_recons,
            )
        update_invocation_artifacts(
            invocation_json,
            status="completed",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        update_invocation_artifacts(
            invocation_json,
            status="failed",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
