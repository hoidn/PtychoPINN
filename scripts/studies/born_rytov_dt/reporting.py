"""Reporting helpers for BRDT adapter sanity runs.

Emits a stable, machine-readable adapter-contract payload plus a
per-run sanity summary so the later four-row preflight can consume the
adapter contract without re-reading backlog prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt.run_config import (
    DEFAULT_TRAINING_LABEL,
    REJECTED_INPUT_MODES,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_INPUT_MODES,
    required_row_fields,
)


ADAPTER_CONTRACT_SCHEMA_VERSION = "brdt_adapter_contract_v1"


def build_adapter_contract(
    *,
    dataset_id: str,
    operator_version: str,
    rows: Sequence[Mapping[str, Any]],
    classical_backend: Mapping[str, Any],
    loss_contract: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the durable adapter-contract payload.

    Downstream consumers (the four-row preflight) may rely on the keys
    returned here; new keys may be added but existing keys must not be
    renamed without an approved follow-up.
    """
    payload: Dict[str, Any] = {
        "schema_version": ADAPTER_CONTRACT_SCHEMA_VERSION,
        "dataset_id": str(dataset_id),
        "operator_version": str(operator_version),
        "row_schema": {
            "required_fields": list(required_row_fields()),
            "supported_architectures": list(SUPPORTED_ARCHITECTURES),
            "supported_input_modes": list(SUPPORTED_INPUT_MODES),
            "rejected_input_modes": list(REJECTED_INPUT_MODES),
            "default_training_label": DEFAULT_TRAINING_LABEL,
        },
        "rows": [dict(row) for row in rows],
        "classical_backend": dict(classical_backend),
        "loss_contract": dict(loss_contract),
        "physics_loss_rule": dc.PHYSICS_LOSS_RULE,
        "operator_geometry": {
            "grid_size": dc.LOCKED_GRID_SIZE,
            "detector_size": dc.LOCKED_DETECTOR_SIZE,
            "angle_count": dc.LOCKED_ANGLE_COUNT,
            "wavelength_px": dc.LOCKED_WAVELENGTH_PX,
            "medium_ri": dc.LOCKED_MEDIUM_RI,
            "mode": dc.LOCKED_OPERATOR_MODE,
            "normalize": dc.LOCKED_NORMALIZE,
        },
        "claim_boundary": (
            "Adapter readiness only. NOT manuscript evidence. The four-row "
            "preflight item owns benchmark-grade rows; this payload exists "
            "so that later item can consume one shared row contract."
        ),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Serialize ``payload`` as pretty JSON with sorted keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def rows_with_sanity_summary(
    rows: List[Any],
    *,
    selected_row_id: str,
    summary: "SanitySummary",
) -> List[Dict[str, Any]]:
    """Return ``[row.to_dict()...]`` with the selected row annotated.

    Attaches ``sanity_summary`` and overrides ``row_status`` on the row
    matching ``selected_row_id`` so the durable adapter contract carries
    the per-run outcome alongside the static row metadata.
    """
    payload: List[Dict[str, Any]] = []
    for row in rows:
        row_dict = row.to_dict()
        if row_dict.get("row_id") == selected_row_id:
            row_dict["row_status"] = summary.row_status
            row_dict["sanity_summary"] = summary.to_dict()
        payload.append(row_dict)
    return payload


@dataclass(frozen=True)
class SanitySummary:
    """Per-run sanity summary."""

    row_id: str
    architecture: str
    parameter_count: int
    train_steps: int
    final_loss_total: float
    final_loss_breakdown: Dict[str, float]
    eval_image_mae_norm: Optional[float]
    eval_relative_physics: Optional[float]
    row_status: str
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "row_id": self.row_id,
            "architecture": self.architecture,
            "parameter_count": int(self.parameter_count),
            "train_steps": int(self.train_steps),
            "final_loss_total": float(self.final_loss_total),
            "final_loss_breakdown": dict(self.final_loss_breakdown),
            "row_status": self.row_status,
        }
        if self.eval_image_mae_norm is not None:
            payload["eval_image_mae_norm"] = float(self.eval_image_mae_norm)
        if self.eval_relative_physics is not None:
            payload["eval_relative_physics"] = float(self.eval_relative_physics)
        if self.note is not None:
            payload["note"] = self.note
        return payload
