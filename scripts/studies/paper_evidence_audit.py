"""Build the repo-local NeurIPS paper evidence audit surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.studies.paper_provenance import load_json_if_exists, write_json


NEURIPS_MANUSCRIPT_ROOT = Path("/home/ollie/Documents/neurips")
DEFAULT_AUDIT_ARTIFACT_ROOT = (
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit"
)
DEFAULT_MANIFEST_PATH = "docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json"
DEFAULT_SUMMARY_PATH = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md"
)

STATUS_VOCABULARY = {
    "paper_grade": "Evidence satisfies the current paper-facing provenance contract and may support headline claims.",
    "full_training": "Evidence proves a same-protocol full-training benchmark lane rather than a capped substitute.",
    "capped_decision_support": "Evidence is coherent and manuscript-usable only with explicit bounded capped wording.",
    "decision_support": "Evidence is useful for local comparison or continuity context but is not a current headline authority.",
    "blocked": "A required row or claim cannot be promoted under the present authority set.",
    "not_protocol_compatible": "A result exists but cannot be used as the same-contract production answer for this paper lane.",
}


def _verification_log_paths(artifact_root: str) -> dict[str, str]:
    verification_root = f"{artifact_root}/verification"
    return {
        "required_inputs_check_log": f"{verification_root}/required_inputs_check.log",
        "pytest_log": f"{verification_root}/pytest_paper_evidence_audit.log",
        "audit_output_validation_log": f"{verification_root}/audit_output_validation.log",
        "audit_direct_entrypoint_log": f"{verification_root}/audit_direct_entrypoint.log",
    }


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(repo_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _relative(repo_root: Path, path_value: str | Path) -> str:
    path = _resolve(repo_root, path_value)
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _read_text(repo_root: Path, path_value: str | Path) -> str:
    return _resolve(repo_root, path_value).read_text(encoding="utf-8")


def _load_json(repo_root: Path, path_value: str | Path) -> dict[str, Any]:
    payload = load_json_if_exists(_resolve(repo_root, path_value))
    if not payload:
        raise ValueError(f"missing required JSON payload: {path_value}")
    return payload


def _assert_paths_match_authoritative_root(
    *,
    repo_root: Path,
    authoritative_root: str,
    path_values: dict[str, str],
    pillar_label: str,
) -> None:
    authoritative_root_path = _resolve(repo_root, authoritative_root).resolve()
    for label, path_value in path_values.items():
        resolved_path = _resolve(repo_root, path_value).resolve()
        if resolved_path.parent != authoritative_root_path:
            raise ValueError(
                f"{pillar_label} authoritative root identity mismatch for {label}: "
                f"expected parent {authoritative_root_path}, got {resolved_path.parent}"
            )


def ensure_repo_local_output_path(path_value: str | Path, *, repo_root: Path | None = None) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    path = Path(path_value)
    resolved = path if path.is_absolute() else (repo_root / path)
    resolved = resolved.resolve()
    if str(resolved).startswith(str(NEURIPS_MANUSCRIPT_ROOT.resolve())):
        raise ValueError(f"output path must stay repo-local and not target {NEURIPS_MANUSCRIPT_ROOT}: {resolved}")
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"output path must stay under repo root {repo_root}: {resolved}") from exc
    return resolved


def build_default_input_manifest(repo_root: Path | str) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    cdi_root = (
        ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/"
        "2026-04-29-cdi-lines128-paper-benchmark-execution/runs/"
        "complete_table_20260430T150757Z_repair_tmux"
    )
    cns_bundle_root = (
        ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
        "2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap"
    )
    cns_historical_bundle_root = (
        ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle"
    )
    artifact_root = DEFAULT_AUDIT_ARTIFACT_ROOT
    manifest_path = DEFAULT_MANIFEST_PATH
    summary_path = DEFAULT_SUMMARY_PATH
    validation_path = f"{artifact_root}/audit_validation.json"

    for output_path in (artifact_root, manifest_path, summary_path, validation_path):
        ensure_repo_local_output_path(output_path, repo_root=repo_root)

    return {
        "generated_at_utc": _now_utc(),
        "status_vocabulary": deepcopy(STATUS_VOCABULARY),
        "output_targets": {
            "artifact_root": artifact_root,
            "audit_inputs_path": f"{artifact_root}/audit_inputs.json",
            "audit_validation_path": validation_path,
            "manifest_path": manifest_path,
            "summary_path": summary_path,
            "verification_logs": _verification_log_paths(artifact_root),
        },
        "cdi": {
            "pillar_id": "cdi",
            "summary_path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md",
            "authoritative_root": cdi_root,
            "paper_manifest_path": f"{cdi_root}/paper_benchmark_manifest.json",
            "metrics_path": f"{cdi_root}/metrics.json",
            "model_manifest_path": f"{cdi_root}/model_manifest.json",
            "metric_schema_path": f"{cdi_root}/metric_schema.json",
            "adjacent_context": [
                {
                    "artifact_id": "cdi_lines128_minimum_subset",
                    "summary_path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_summary.md",
                    "source_root": ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z",
                    "row_status": "decision_support",
                    "claim_boundary": "minimum_draftable_cdi_subset",
                    "draftability": "draftable_context_only",
                },
                {
                    "artifact_id": "cdi_lines128_ffno_prerequisite_pair",
                    "summary_path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md",
                    "source_root": ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet",
                    "row_status": "decision_support",
                    "claim_boundary": "lines128_ffno_vs_hybrid_prerequisite_pair",
                    "draftability": "draftable_context_only",
                },
            ],
        },
        "cns": {
            "pillar_id": "cns",
            "contract_decision_path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md",
            "row_lock_summary_path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md",
            "locked_rows_path": ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/cns_paper_locked_rows_2048cap.json",
            "bundle_summary_path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md",
            "bundle_root": cns_bundle_root,
            "table_rows_path": f"{cns_bundle_root}/cns_paper_table_rows.json",
            "bundle_validation_path": f"{cns_bundle_root}/bundle_validation.json",
            "figure_manifest_path": f"{cns_bundle_root}/figure_manifest.json",
            "fixed_sample_manifest_path": f"{cns_bundle_root}/fixed_sample_manifest.json",
            "historical_bundle_summary_path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_table_figure_bundle_summary.md",
            "historical_locked_rows_path": ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json",
            "historical_bundle_root": cns_historical_bundle_root,
        },
        "index_surfaces": {
            "paper_evidence_index_path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md",
            "docs_index_path": "docs/index.md",
        },
    }


def _normalize_cdi_row(
    row_id: str,
    row_payload: dict[str, Any],
    *,
    repo_root: Path,
    cdi_inputs: dict[str, Any],
) -> dict[str, Any]:
    outputs = row_payload.get("outputs", {})
    recon_npz = outputs.get("recon_npz")
    source_roots = [str(Path(recon_npz).parent)] if isinstance(recon_npz, str) else []
    figure_artifacts = [
        f"{cdi_inputs['authoritative_root']}/visuals/amp_phase_{row_id}.png",
        f"{cdi_inputs['authoritative_root']}/visuals/amp_phase_error_{row_id}.png",
    ]
    return {
        "pillar_id": "cdi",
        "artifact_kind": "benchmark_row",
        "artifact_id": f"cdi:{row_id}",
        "row_id": row_id,
        "row_role": "headline",
        "row_status": row_payload["row_status"],
        "claim_boundary": "complete_lines128_cdi_benchmark",
        "evidence_tier": row_payload["row_status"],
        "source_summary": cdi_inputs["summary_path"],
        "source_root": cdi_inputs["authoritative_root"],
        "metric_schema": {"path": cdi_inputs["metric_schema_path"]},
        "table_artifacts": [
            f"{cdi_inputs['authoritative_root']}/metrics.json",
            f"{cdi_inputs['authoritative_root']}/metric_schema.json",
            f"{cdi_inputs['authoritative_root']}/model_manifest.json",
            f"{cdi_inputs['authoritative_root']}/metrics_table.csv",
            f"{cdi_inputs['authoritative_root']}/metrics_table.tex",
            f"{cdi_inputs['authoritative_root']}/metrics_table_best.tex",
        ],
        "figure_artifacts": figure_artifacts,
        "source_array_roots": source_roots,
        "provenance_gaps": list(row_payload.get("missing_fields", [])),
        "draftability": "draftable_now",
        "blocked_claims": [],
        "notes": {
            "model_label": row_payload.get("model_label"),
            "architecture_id": row_payload.get("architecture_id"),
            "training_procedure": row_payload.get("training_procedure"),
            "parameter_count": row_payload.get("parameter_count"),
            "hardware_summary": row_payload.get("hardware_summary"),
        },
    }


def load_cdi_authority(cdi_inputs: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    summary_text = _read_text(repo_root, cdi_inputs["summary_path"])
    authoritative_root = cdi_inputs["authoritative_root"]
    if authoritative_root not in summary_text:
        raise ValueError(f"CDI summary does not name authoritative root {authoritative_root}")
    _assert_paths_match_authoritative_root(
        repo_root=repo_root,
        authoritative_root=authoritative_root,
        path_values={
            "paper_manifest_path": cdi_inputs["paper_manifest_path"],
            "metrics_path": cdi_inputs["metrics_path"],
            "model_manifest_path": cdi_inputs["model_manifest_path"],
        },
        pillar_label="CDI",
    )

    paper_manifest = _load_json(repo_root, cdi_inputs["paper_manifest_path"])
    metrics_payload = _load_json(repo_root, cdi_inputs["metrics_path"])
    model_manifest = _load_json(repo_root, cdi_inputs["model_manifest_path"])

    status_values = {
        str(paper_manifest.get("benchmark_status")),
        str(metrics_payload.get("benchmark_status")),
        str(model_manifest.get("benchmark_status")),
    }
    if len(status_values) != 1:
        raise ValueError(f"CDI source disagreement on benchmark status: {sorted(status_values)}")

    claim_boundaries = {
        str(paper_manifest.get("claim_boundary")),
        str(metrics_payload.get("claim_boundary")),
        str(model_manifest.get("claim_boundary")),
    }
    if claim_boundaries != {"complete_lines128_cdi_benchmark"}:
        raise ValueError(f"CDI source disagreement on claim boundary: {sorted(claim_boundaries)}")

    comparator_values = {
        str(paper_manifest.get("selected_fno_comparator")),
        str(metrics_payload.get("selected_fno_comparator")),
    }
    if comparator_values != {"fno_vanilla"}:
        raise ValueError(f"CDI source disagreement on selected comparator: {sorted(comparator_values)}")

    paper_rows = {
        str(row["model_id"])
        for row in paper_manifest.get("rows", [])
        if isinstance(row, dict) and row.get("model_id")
    }
    metric_row_ids = {str(key) for key in metrics_payload.get("rows", {}).keys()}
    model_rows = {
        str(row["model_id"])
        for row in model_manifest.get("rows", [])
        if isinstance(row, dict) and row.get("model_id")
    }
    if paper_rows != metric_row_ids or paper_rows != model_rows:
        raise ValueError(
            "CDI source disagreement on row roster: "
            f"paper={sorted(paper_rows)} metrics={sorted(metric_row_ids)} model={sorted(model_rows)}"
        )

    metric_rows = metrics_payload["rows"]
    row_registry = [
        _normalize_cdi_row(row_id, row_payload, repo_root=repo_root, cdi_inputs=cdi_inputs)
        for row_id, row_payload in metric_rows.items()
    ]

    return {
        "pillar_id": "cdi",
        "headline_status": "paper_grade",
        "bundle_status": str(metrics_payload["benchmark_status"]),
        "claim_boundary": "complete_lines128_cdi_benchmark",
        "selected_fno_comparator": "fno_vanilla",
        "seed_policy": paper_manifest.get("seed_policy", {}),
        "headline_row_ids": sorted(metric_rows.keys()),
        "headline_rows": row_registry,
        "table_artifacts": [
            f"{authoritative_root}/metrics.json",
            f"{authoritative_root}/metric_schema.json",
            f"{authoritative_root}/model_manifest.json",
            f"{authoritative_root}/metrics_table.csv",
            f"{authoritative_root}/metrics_table.tex",
            f"{authoritative_root}/metrics_table_best.tex",
        ],
        "figure_artifacts": [
            f"{authoritative_root}/visuals/amp_phase_gt.png",
            f"{authoritative_root}/visuals/compare_amp_phase.png",
            f"{authoritative_root}/visuals/frc_curves.png",
        ],
        "source_summary": cdi_inputs["summary_path"],
        "source_root": authoritative_root,
        "adjacent_context": deepcopy(cdi_inputs["adjacent_context"]),
        "provenance_gaps": [],
    }


def _normalize_cns_row(
    row_payload: dict[str, Any],
    *,
    cns_inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pillar_id": "cns",
        "artifact_kind": "benchmark_row",
        "artifact_id": f"cns:{row_payload['row_id']}",
        "row_id": row_payload["row_id"],
        "row_role": row_payload["row_role"],
        "row_status": row_payload["row_status"],
        "claim_boundary": "bounded_capped_decision_support_only",
        "evidence_tier": row_payload["row_status"],
        "source_summary": cns_inputs["row_lock_summary_path"],
        "source_root": row_payload["run_root"],
        "metric_schema": {
            "family": row_payload.get("metric_family", []),
            "table_json_path": cns_inputs["table_rows_path"],
        },
        "table_artifacts": [
            cns_inputs["table_rows_path"],
            cns_inputs["bundle_root"] + "/cns_paper_table_rows.csv",
            cns_inputs["bundle_root"] + "/cns_paper_table_rows.tex",
        ],
        "figure_artifacts": [
            cns_inputs["figure_manifest_path"],
            cns_inputs["fixed_sample_manifest_path"],
            cns_inputs["bundle_root"] + "/figures",
        ],
        "source_array_roots": [
            cns_inputs["bundle_root"] + "/figure_sources",
            row_payload["run_root"],
        ],
        "provenance_gaps": list(row_payload.get("known_provenance_gaps", [])),
        "draftability": "draftable_with_bounded_wording",
        "blocked_claims": ["same_protocol_full_training_cns_competitiveness"],
        "notes": {
            "task_id": row_payload.get("task_id"),
            "parameter_count": row_payload.get("parameter_count"),
            "runtime_sec": row_payload.get("runtime_sec"),
        },
    }


def _normalize_cns_adjacent_context(context_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(context_payload)
    source_status = str(normalized.get("status", ""))
    if source_status == "excluded_adjacent_context":
        normalized["status"] = "not_protocol_compatible"
    if source_status:
        normalized["source_status"] = source_status
    return normalized


def load_cns_authority(cns_inputs: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    contract_text = _read_text(repo_root, cns_inputs["contract_decision_path"])
    row_lock_text = _read_text(repo_root, cns_inputs["row_lock_summary_path"])
    bundle_text = _read_text(repo_root, cns_inputs["bundle_summary_path"])
    for required_text in (
        "bounded_capped_decision_support",
        cns_inputs["bundle_root"],
        cns_inputs["locked_rows_path"],
    ):
        if required_text not in (contract_text + row_lock_text + bundle_text):
            raise ValueError(f"CNS authorities do not consistently mention {required_text}")

    _assert_paths_match_authoritative_root(
        repo_root=repo_root,
        authoritative_root=cns_inputs["bundle_root"],
        path_values={
            "table_rows_path": cns_inputs["table_rows_path"],
            "bundle_validation_path": cns_inputs["bundle_validation_path"],
            "figure_manifest_path": cns_inputs["figure_manifest_path"],
            "fixed_sample_manifest_path": cns_inputs["fixed_sample_manifest_path"],
        },
        pillar_label="CNS",
    )

    locked_rows = _load_json(repo_root, cns_inputs["locked_rows_path"])
    table_rows = _load_json(repo_root, cns_inputs["table_rows_path"])
    bundle_validation = _load_json(repo_root, cns_inputs["bundle_validation_path"])
    figure_manifest = _load_json(repo_root, cns_inputs["figure_manifest_path"])
    fixed_sample_manifest = _load_json(repo_root, cns_inputs["fixed_sample_manifest_path"])

    if locked_rows.get("contract_type") != "bounded_capped_decision_support":
        raise ValueError(f"CNS locked rows changed contract type: {locked_rows.get('contract_type')}")
    if table_rows.get("claim_boundary") != "capped_decision_support_only":
        raise ValueError(f"CNS table rows changed claim boundary: {table_rows.get('claim_boundary')}")
    if not bundle_validation.get("all_rows_capped_decision_support", False):
        raise ValueError("CNS bundle validation no longer confirms capped row statuses")
    if not bundle_validation.get("no_paper_grade_or_full_training_labels", False):
        raise ValueError("CNS bundle validation no longer rejects paper-grade/full-training promotion")
    if table_rows.get("benchmark_status") != bundle_validation.get("benchmark_status"):
        raise ValueError(
            "CNS source disagreement on benchmark status: "
            f"table={table_rows.get('benchmark_status')} "
            f"validation={bundle_validation.get('benchmark_status')}"
        )

    locked_row_ids = [str(row["row_id"]) for row in locked_rows.get("rows", [])]
    table_row_ids = [str(row["row_id"]) for row in table_rows.get("rows", [])]
    validation_row_ids = [str(row_id) for row_id in bundle_validation.get("table_row_ids", [])]
    figure_row_ids = [str(row_id) for row_id in figure_manifest.get("rows_in_visual_bundle", [])]
    fixed_sample_row_ids = [str(row_id) for row_id in fixed_sample_manifest.get("rows_in_visual_bundle", [])]
    validation_figure_row_ids = [str(row_id) for row_id in bundle_validation.get("figure_manifest_row_ids", [])]
    validation_sample_row_ids = [str(row_id) for row_id in bundle_validation.get("sample_manifest_row_ids", [])]
    validation_visual_row_ids = [str(row_id) for row_id in bundle_validation.get("visual_bundle_row_ids", [])]
    if not (
        locked_row_ids
        == table_row_ids
        == validation_row_ids
        == figure_row_ids
        == fixed_sample_row_ids
        == validation_figure_row_ids
        == validation_sample_row_ids
        == validation_visual_row_ids
    ):
        raise ValueError(
            "CNS source disagreement on visual roster: "
            f"locked={locked_row_ids} table={table_row_ids} validation={validation_row_ids} "
            f"figure={figure_row_ids} sample={fixed_sample_row_ids} "
            f"validation_figure={validation_figure_row_ids} "
            f"validation_sample={validation_sample_row_ids} "
            f"validation_visual={validation_visual_row_ids}"
        )

    figure_sample_ids = list(figure_manifest.get("sample_ids", []))
    fixed_sample_ids = list(fixed_sample_manifest.get("sample_ids", []))
    if figure_sample_ids != fixed_sample_ids:
        raise ValueError(
            "CNS source disagreement on visual sample ids: "
            f"figure={figure_sample_ids} sample={fixed_sample_ids}"
        )

    figure_field_order = list(figure_manifest.get("field_order", []))
    fixed_field_order = list(fixed_sample_manifest.get("field_order", []))
    if figure_field_order != fixed_field_order:
        raise ValueError(
            "CNS source disagreement on visual field order: "
            f"figure={figure_field_order} sample={fixed_field_order}"
        )

    locked_headline = list(locked_rows.get("headline_row_ids", []))
    table_headline = list(table_rows.get("headline_row_ids", []))
    validation_headline = list(bundle_validation.get("table_headline_row_ids", []))
    if locked_headline != table_headline or locked_headline != validation_headline:
        raise ValueError(
            "CNS source disagreement on headline roster: "
            f"locked={locked_headline} table={table_headline} validation={validation_headline}"
        )

    row_registry = [_normalize_cns_row(row, cns_inputs=cns_inputs) for row in locked_rows.get("rows", [])]
    headline_rows = [row for row in row_registry if row["row_role"] == "headline"]
    continuity_rows = [row for row in row_registry if row["row_role"] != "headline"]

    return {
        "pillar_id": "cns",
        "headline_status": "capped_decision_support",
        "bundle_status": str(bundle_validation["benchmark_status"]),
        "claim_boundary": "bounded_capped_decision_support_only",
        "headline_row_ids": locked_headline,
        "headline_rows": headline_rows,
        "continuity_rows": continuity_rows,
        "continuity_row_ids": list(locked_rows.get("continuity_row_ids", [])),
        "source_summary": cns_inputs["bundle_summary_path"],
        "source_root": cns_inputs["bundle_root"],
        "table_artifacts": [
            cns_inputs["table_rows_path"],
            cns_inputs["bundle_root"] + "/cns_paper_table_rows.csv",
            cns_inputs["bundle_root"] + "/cns_paper_table_rows.tex",
        ],
        "figure_artifacts": [
            cns_inputs["figure_manifest_path"],
            cns_inputs["fixed_sample_manifest_path"],
            cns_inputs["bundle_root"] + "/figures",
        ],
        "adjacent_context": [
            _normalize_cns_adjacent_context(context)
            for context in locked_rows.get("excluded_adjacent_context", [])
        ],
        "provenance_gaps": sorted(
            {
                gap
                for row in locked_rows.get("rows", [])
                for gap in row.get("known_provenance_gaps", [])
            }
        ),
    }


def _build_claim_boundary_registry(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cdi = manifest["pillar_summaries"]["cdi"]
    cns = manifest["pillar_summaries"]["cns"]
    registry = [
        {
            "claim_boundary": cdi["claim_boundary"],
            "pillar_id": "cdi",
            "headline_status": cdi["headline_status"],
            "source_summary": cdi["source_summary"],
            "source_root": cdi["source_root"],
        },
        {
            "claim_boundary": cns["claim_boundary"],
            "pillar_id": "cns",
            "headline_status": cns["headline_status"],
            "source_summary": cns["source_summary"],
            "source_root": cns["source_root"],
        },
    ]
    for context in cdi["adjacent_context"]:
        registry.append(
            {
                "claim_boundary": context["claim_boundary"],
                "pillar_id": "cdi",
                "headline_status": context["row_status"],
                "source_summary": context["summary_path"],
                "source_root": context["source_root"],
            }
        )
    return registry


def _build_provenance_gap_registry(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cdi = manifest["pillar_summaries"]["cdi"]
    cns = manifest["pillar_summaries"]["cns"]
    gaps: list[dict[str, Any]] = []
    for gap in cdi.get("provenance_gaps", []):
        gaps.append({"pillar_id": "cdi", "gap": gap, "blocking": False})
    for gap in cns.get("provenance_gaps", []):
        gaps.append({"pillar_id": "cns", "gap": gap, "blocking": True})
    return gaps


def _build_draftability_matrix(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cdi = manifest["pillar_summaries"]["cdi"]
    cns = manifest["pillar_summaries"]["cns"]
    return [
        {
            "section_id": "results_cdi_lines128",
            "draftability": "draftable_now",
            "supporting_status": cdi["headline_status"],
            "allowed_claims": [
                "The paper-grade CDI anchor is the complete six-row Lines128 benchmark.",
                "Hybrid-family CDI claims may use the current paper-grade bundle.",
            ],
            "blocked_claims": [],
        },
        {
            "section_id": "results_pdebench_cns",
            "draftability": "draftable_with_bounded_wording",
            "supporting_status": cns["headline_status"],
            "allowed_claims": [
                "The CNS pillar is a coherent bounded capped comparison under one fixed contract.",
                "Continuity/support context may cite hybrid_resnet_cns without promoting it to the headline roster.",
            ],
            "blocked_claims": ["same_protocol_full_training_cns_competitiveness"],
        },
        {
            "section_id": "results_cross_pillar_takeaway",
            "draftability": "draftable_with_asymmetric_evidence",
            "supporting_status": "mixed",
            "allowed_claims": [
                "CDI is the paper-grade anchor while CNS is bounded capped decision-support generalization evidence.",
            ],
            "blocked_claims": ["symmetrical_full_training_claim_across_both_pillars"],
        },
        {
            "section_id": "results_cns_full_training_competitiveness",
            "draftability": "placeholder_only",
            "supporting_status": "blocked",
            "allowed_claims": [],
            "blocked_claims": ["same_protocol_full_training_cns_competitiveness"],
        },
        {
            "section_id": "results_inverse_wave_candidate_lane",
            "draftability": "placeholder_only",
            "supporting_status": "blocked",
            "allowed_claims": [],
            "blocked_claims": ["candidate_lane_not_promoted_into_required_evidence_package"],
        },
    ]


def _build_blocked_claims(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "same_protocol_full_training_cns_competitiveness",
            "status": "blocked",
            "reason": "The current CNS paper authority is capped decision-support only.",
            "source_summary": manifest["pillar_summaries"]["cns"]["source_summary"],
        },
        {
            "claim_id": "history_len3_locked_cns_headline_lane",
            "status": "blocked",
            "reason": "The locked CNS headline contract remains history_len=2 with authored FFNO present only there.",
            "source_summary": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md",
        },
        {
            "claim_id": "minimum_subset_is_current_cdi_headline_authority",
            "status": "blocked",
            "reason": "The complete six-row Lines128 bundle supersedes the earlier minimum subset as the CDI headline authority.",
            "source_summary": manifest["pillar_summaries"]["cdi"]["source_summary"],
        },
        {
            "claim_id": "paper_facing_neurips_bundle_emitted",
            "status": "blocked",
            "reason": "This item is repo-local only and must not populate /home/ollie/Documents/neurips/ yet.",
            "source_summary": manifest["pillar_summaries"]["cdi"]["source_summary"],
        },
    ]


def build_manifest(inputs: dict[str, Any], *, repo_root: Path | str) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    outputs = deepcopy(inputs["output_targets"])
    for key in ("artifact_root", "audit_inputs_path", "audit_validation_path", "manifest_path", "summary_path"):
        ensure_repo_local_output_path(outputs[key], repo_root=repo_root)

    cdi = load_cdi_authority(inputs["cdi"], repo_root=repo_root)
    cns = load_cns_authority(inputs["cns"], repo_root=repo_root)
    row_registry = cdi["headline_rows"] + cns["headline_rows"] + cns["continuity_rows"]

    manifest = {
        "generated_at_utc": _now_utc(),
        "status_vocabulary": deepcopy(inputs["status_vocabulary"]),
        "output_targets": outputs,
        "authoritative_inputs": {
            "cdi": deepcopy(inputs["cdi"]),
            "cns": deepcopy(inputs["cns"]),
            "index_surfaces": deepcopy(inputs["index_surfaces"]),
        },
        "pillar_summaries": {"cdi": cdi, "cns": cns},
        "row_registry": row_registry,
        "claim_boundary_registry": [],
        "provenance_gap_registry": [],
        "manuscript_draftability": [],
        "blocked_claims": [],
    }
    manifest["claim_boundary_registry"] = _build_claim_boundary_registry(manifest)
    manifest["provenance_gap_registry"] = _build_provenance_gap_registry(manifest)
    manifest["manuscript_draftability"] = _build_draftability_matrix(manifest)
    manifest["blocked_claims"] = _build_blocked_claims(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> dict[str, bool]:
    outputs = manifest["output_targets"]
    output_values = [
        outputs["artifact_root"],
        outputs["audit_inputs_path"],
        outputs["audit_validation_path"],
        outputs["manifest_path"],
        outputs["summary_path"],
        *outputs["verification_logs"].values(),
    ]
    status_values = {
        manifest["pillar_summaries"]["cdi"]["headline_status"],
        manifest["pillar_summaries"]["cns"]["headline_status"],
        *(context["row_status"] for context in manifest["pillar_summaries"]["cdi"]["adjacent_context"]),
        *(context["status"] for context in manifest["pillar_summaries"]["cns"]["adjacent_context"]),
        *(entry["status"] for entry in manifest["blocked_claims"]),
    }
    return {
        "cdi_headline_is_paper_grade": manifest["pillar_summaries"]["cdi"]["headline_status"] == "paper_grade",
        "cns_headline_is_capped_decision_support": (
            manifest["pillar_summaries"]["cns"]["headline_status"] == "capped_decision_support"
        ),
        "no_current_full_training_pillar": all(
            pillar["headline_status"] != "full_training" for pillar in manifest["pillar_summaries"].values()
        ),
        "adjacent_cdi_context_not_promoted": all(
            context["row_status"] != "paper_grade"
            for context in manifest["pillar_summaries"]["cdi"]["adjacent_context"]
        ),
        "all_statuses_within_frozen_vocabulary": status_values <= set(manifest["status_vocabulary"]),
        "no_output_targets_under_neurips": not any(path.startswith(str(NEURIPS_MANUSCRIPT_ROOT)) for path in output_values),
    }


def render_audit_summary(manifest: dict[str, Any]) -> str:
    cdi = manifest["pillar_summaries"]["cdi"]
    cns = manifest["pillar_summaries"]["cns"]
    cns_inputs = manifest["authoritative_inputs"]["cns"]
    outputs = manifest["output_targets"]
    adjacent_cdi_lines = "\n".join(
        f"- `{context['artifact_id']}`: `{context['row_status']}` under `{context['claim_boundary']}` from `{context['source_root']}`."
        for context in cdi["adjacent_context"]
    )
    cns_adjacent_lines = "\n".join(
        f"- `{context['context_id']}`: `{context['status']}` because {context['reason']}"
        for context in cns["adjacent_context"]
    )
    verification_lines = "\n".join(
        f"- `{label}`: `{path}`" for label, path in outputs["verification_logs"].items()
    )
    vocabulary_lines = "\n".join(
        f"- `{status}`: {description}" for status, description in manifest["status_vocabulary"].items()
    )
    draftability_lines = "\n".join(
        f"- `{entry['section_id']}`: `{entry['draftability']}` with supporting status `{entry['supporting_status']}`."
        for entry in manifest["manuscript_draftability"]
    )
    blocked_lines = "\n".join(
        f"- `{entry['claim_id']}`: {entry['reason']}" for entry in manifest["blocked_claims"]
    )
    return f"""# NeurIPS Hybrid ResNet Paper Evidence Package Audit Summary

## Status Vocabulary

{vocabulary_lines}

## Current Authorities

- CDI headline authority: `{cdi['headline_status']}` under `{cdi['claim_boundary']}` from `{cdi['source_root']}`.
- CDI bundle status: `{cdi['bundle_status']}` with selected comparator `{cdi['selected_fno_comparator']}` and fixed seed `{cdi['seed_policy'].get('seed')}`.
- CNS headline authority: `{cns['headline_status']}` under `{cns['claim_boundary']}` from `{cns['source_root']}`.
- CNS bundle status: `{cns['bundle_status']}` reflects table/figure assembly completeness only; it does not upgrade the pillar beyond `{cns['headline_status']}`.
- Historical CNS fallback bundle preserved for provenance: `{cns_inputs['historical_bundle_root']}` under the same capped claim boundary; it is no longer the current discoverability target.
- No outputs from this item target `/home/ollie/Documents/neurips/`; all emitted paths stay repo-local.

## Emitted Outputs

- Manifest path: `{outputs['manifest_path']}`
- Summary path: `{outputs['summary_path']}`
- Validation payload: `{outputs['audit_validation_path']}`
Verification logs:
{verification_lines}

## Draftable Now

- The CDI pillar is draftable now as the current paper-grade anchor because the complete six-row Lines128 bundle is the headline authority.
- The CNS pillar is draftable only with bounded capped wording because every current headline row remains `{cns['headline_status']}`.
- Cross-pillar manuscript language is draftable only if it keeps the asymmetry explicit: paper-grade CDI anchor plus bounded capped CNS support.
- full-training CNS competitiveness claims remain blocked.

{draftability_lines}

## Placeholder-Only Or Blocked Claims

{blocked_lines}

## Adjacent Context

### CDI Continuity Context

{adjacent_cdi_lines}

### CNS Continuity Context

- `hybrid_resnet_cns`: continuity/support only under the same capped contract.
{cns_adjacent_lines}
"""


def validate_summary_sync(manifest: dict[str, Any], summary_text: str) -> dict[str, bool]:
    cdi = manifest["pillar_summaries"]["cdi"]
    cns = manifest["pillar_summaries"]["cns"]
    return {
        "cdi_root_present": cdi["source_root"] in summary_text,
        "cns_root_present": cns["source_root"] in summary_text,
        "cdi_status_present": cdi["headline_status"] in summary_text,
        "cns_status_present": cns["headline_status"] in summary_text,
        "claim_boundaries_present": (
            cdi["claim_boundary"] in summary_text and cns["claim_boundary"] in summary_text
        ),
    }


def _write_summary(path: Path, summary_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_text, encoding="utf-8")


def run_audit(*, repo_root: Path | str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    inputs = deepcopy(inputs or build_default_input_manifest(repo_root))
    outputs = inputs["output_targets"]

    audit_inputs_path = ensure_repo_local_output_path(outputs["audit_inputs_path"], repo_root=repo_root)
    audit_inputs_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(audit_inputs_path, inputs)

    manifest = build_manifest(inputs, repo_root=repo_root)
    summary_text = render_audit_summary(manifest)

    manifest_path = ensure_repo_local_output_path(outputs["manifest_path"], repo_root=repo_root)
    summary_path = ensure_repo_local_output_path(outputs["summary_path"], repo_root=repo_root)
    write_json(manifest_path, manifest)
    _write_summary(summary_path, summary_text)

    validation_payload = {
        "generated_at_utc": _now_utc(),
        "manifest_path": _relative(repo_root, manifest_path),
        "summary_path": _relative(repo_root, summary_path),
        "manifest_validation": validate_manifest(manifest),
        "summary_sync_validation": validate_summary_sync(manifest, summary_text),
    }
    validation_payload["all_checks_pass"] = all(validation_payload["manifest_validation"].values()) and all(
        validation_payload["summary_sync_validation"].values()
    )
    validation_path = ensure_repo_local_output_path(outputs["audit_validation_path"], repo_root=repo_root)
    write_json(validation_path, validation_payload)
    return {
        "inputs_path": _relative(repo_root, audit_inputs_path),
        "manifest_path": _relative(repo_root, manifest_path),
        "summary_path": _relative(repo_root, summary_path),
        "validation_path": _relative(repo_root, validation_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--inputs-path", help="Optional JSON inputs manifest")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    inputs = None
    if args.inputs_path:
        inputs = _load_json(repo_root, args.inputs_path)
    result = run_audit(repo_root=repo_root, inputs=inputs)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
