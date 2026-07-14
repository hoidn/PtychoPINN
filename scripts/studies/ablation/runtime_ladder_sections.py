"""Section parsers for the bridge-ladder checked spec.

Extracted from :mod:`runtime_ladder_spec` (module-budget extraction planned
in the task-21c review, m5). Pure parsing/validation of the ``gate``,
``baseline``, ``groups``, and ``residuals`` tables; no I/O beyond path
resolution.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .runtime_errors import StudyRequestError
from .runtime_ladder_config import (
    CONFIG_FIELDS,
    INVARIANT_CONFIG_FIELDS,
    LadderGate,
    closed_table,
    parse_config,
    required_text,
)

from .runtime_ladder_spec_types import LadderBaseline, LadderResidual

__all__ = ["parse_gate", "parse_baseline", "parse_groups", "parse_residuals"]

_SHA256_ALPHABET = set("0123456789abcdef")  # baseline evidence pin format


def parse_gate(value: Any) -> LadderGate:
    if not isinstance(value, Mapping):
        raise StudyRequestError("gate must be a table")
    policy = required_text(value.get("policy"), "gate.policy")
    common = {"policy", "threshold_provenance"}
    if policy == "absolute_ssim_delta_v1":
        fields = {"max_abs_amp_ssim_delta", "max_abs_phase_ssim_delta"}
    elif policy == "retained_ssim_v1":
        fields = {
            "retained_amp_ssim_min_fraction",
            "retained_phase_ssim_min_fraction",
            "absolute_amp_ssim_floor",
        }
    else:
        raise StudyRequestError(f"unsupported gate.policy {policy!r}")
    table = closed_table(
        value, path="gate", allowed=common | fields, required=common | fields
    )
    values = {field: table[field] for field in fields}
    return LadderGate(
        policy=policy,
        threshold_provenance=required_text(
            table["threshold_provenance"], "gate.threshold_provenance"
        ),
        **values,
    )

def parse_baseline(value: Any, base_dir: Path) -> LadderBaseline:
    keys = {
        "id",
        "status",
        "reference_spec",
        "reference_id",
        "evidence",
        "evidence_sha256",
        "historical_evidence",
        "historical_evidence_sha256",
        "dataset",
        "config",
    }
    required = keys - {
        "status",
        "evidence_sha256",
        "historical_evidence",
        "historical_evidence_sha256",
    }
    table = closed_table(value, path="baseline", allowed=keys, required=required)
    status = table.get("status", "current")
    if status not in {"current", "pending", "superseded"}:
        raise StudyRequestError(
            "baseline.status must be 'current', 'pending', or 'superseded'"
        )

    evidence_sha = table.get("evidence_sha256")
    if evidence_sha is not None:
        evidence_sha = required_text(evidence_sha, "baseline.evidence_sha256")
        if len(evidence_sha) != 64 or set(evidence_sha) - _SHA256_ALPHABET:
            raise StudyRequestError(
                "baseline.evidence_sha256 must be a lowercase SHA-256"
            )
    if status == "current" and evidence_sha is None:
        raise StudyRequestError(
            "baseline.evidence_sha256 is required when baseline.status is current"
        )
    if status != "current" and evidence_sha is not None:
        raise StudyRequestError(
            "baseline.evidence_sha256 must be omitted unless baseline.status is current"
        )

    historical_evidence = table.get("historical_evidence")
    historical_sha = table.get("historical_evidence_sha256")
    if (historical_evidence is None) != (historical_sha is None):
        raise StudyRequestError(
            "baseline historical evidence path and SHA-256 must be declared together"
        )
    if historical_evidence is not None:
        historical_evidence = required_text(
            historical_evidence, "baseline.historical_evidence"
        )
        historical_sha = required_text(
            historical_sha, "baseline.historical_evidence_sha256"
        )
        if len(historical_sha) != 64 or set(historical_sha) - _SHA256_ALPHABET:
            raise StudyRequestError(
                "baseline.historical_evidence_sha256 must be a lowercase SHA-256"
            )
    reference_spec = required_text(table["reference_spec"], "baseline.reference_spec")
    evidence = required_text(table["evidence"], "baseline.evidence")
    return LadderBaseline(
        id=required_text(table["id"], "baseline.id"),
        status=status,
        reference_spec=(Path(base_dir) / reference_spec).resolve(),
        reference_spec_declared=reference_spec,
        reference_id=required_text(table["reference_id"], "baseline.reference_id"),
        evidence=(Path(base_dir) / evidence).resolve(),
        evidence_declared=evidence,
        evidence_sha256=evidence_sha,
        historical_evidence=(
            None
            if historical_evidence is None
            else (Path(base_dir) / historical_evidence).resolve()
        ),
        historical_evidence_declared=historical_evidence,
        historical_evidence_sha256=historical_sha,
        dataset=required_text(table["dataset"], "baseline.dataset"),
        config=MappingProxyType(parse_config(table["config"], "baseline.config")),
    )

def parse_groups(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise StudyRequestError("groups must be a nonempty table")
    groups: dict[str, tuple[str, ...]] = {}
    for name, fields in value.items():
        if not isinstance(fields, list) or not fields:
            raise StudyRequestError(f"groups.{name} must be a nonempty field list")
        for field in fields:
            if field not in CONFIG_FIELDS:
                raise StudyRequestError(
                    f"groups.{name} names unknown config field {field!r}"
                )
            if field in INVARIANT_CONFIG_FIELDS:
                raise StudyRequestError(
                    f"groups.{name} may not change ladder invariant {field!r}"
                )
        if len(set(fields)) != len(fields):
            raise StudyRequestError(f"groups.{name} contains duplicate fields")
        groups[name] = tuple(fields)
    return groups

def parse_residuals(value: Any) -> tuple[LadderResidual, ...]:
    if not isinstance(value, list) or not value:
        raise StudyRequestError("residuals must be a nonempty array of tables")
    residuals = []
    for index, entry in enumerate(value):
        table = closed_table(
            entry,
            path=f"residuals[{index}]",
            allowed={"id", "description"},
            required={"id", "description"},
        )
        residuals.append(
            LadderResidual(
                id=required_text(table["id"], f"residuals[{index}].id"),
                description=required_text(
                    table["description"], f"residuals[{index}].description"
                ),
            )
        )
    ids = [residual.id for residual in residuals]
    if len(set(ids)) != len(ids):
        raise StudyRequestError("residuals declare duplicate ids")
    return tuple(residuals)
