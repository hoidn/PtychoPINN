"""Preflight helpers for the CDI hybrid-spectral to FFNO parameter-space study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


REUSED_ROWS = [
    {
        "model_id": "pinn_hybrid_resnet",
        "model_label": "Hybrid ResNet + PINN",
        "architecture": "hybrid_resnet",
        "row_kind": "reused_anchor",
        "nearest_anchor": None,
    },
    {
        "model_id": "pinn_spectral_resnet_bottleneck_net",
        "model_label": "Spectral ResNet Bottleneck + PINN",
        "architecture": "spectral_resnet_bottleneck_net",
        "row_kind": "reused_anchor",
        "nearest_anchor": None,
    },
    {
        "model_id": "pinn_ffno",
        "model_label": "FFNO + PINN",
        "architecture": "ffno",
        "row_kind": "reused_anchor",
        "nearest_anchor": None,
    },
]

FRESH_ROWS = [
    {
        "model_id": "pinn_spectral_resnet_bottleneck_ds1",
        "model_label": "Spectral ResNet Bottleneck DS1 + PINN",
        "architecture": "spectral_resnet_bottleneck_net",
        "row_kind": "fresh_bridge",
        "nearest_anchor": "pinn_spectral_resnet_bottleneck_net",
        "overrides": {"hybrid_downsample_steps": 1},
    },
    {
        "model_id": "pinn_spectral_resnet_bottleneck_linear_decoder",
        "model_label": "Spectral ResNet Linear Decoder + PINN",
        "architecture": "spectral_resnet_bottleneck_linear_decoder",
        "row_kind": "fresh_bridge",
        "nearest_anchor": "pinn_spectral_resnet_bottleneck_net",
        "overrides": {},
    },
    {
        "model_id": "pinn_hybrid_resnet_ffno_bottleneck",
        "model_label": "Hybrid ResNet FFNO Bottleneck + PINN",
        "architecture": "hybrid_resnet_ffno_bottleneck",
        "row_kind": "fresh_bridge",
        "nearest_anchor": "pinn_hybrid_resnet",
        "overrides": {},
    },
]


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _required_row_paths(authoritative_root: Path, model_id: str) -> Dict[str, str]:
    run_dir = authoritative_root / "runs" / model_id
    recon_path = authoritative_root / "recons" / model_id / "recon.npz"
    return {
        "run_dir": str(run_dir),
        "recon_npz": str(recon_path),
        "invocation_json": str(run_dir / "invocation.json"),
        "config_json": str(run_dir / "config.json"),
        "history_json": str(run_dir / "history.json"),
        "metrics_json": str(run_dir / "metrics.json"),
    }


def build_study_matrix_payload(*, authoritative_root: Path, artifact_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for row in [*REUSED_ROWS, *FRESH_ROWS]:
        row_payload = dict(row)
        row_payload["analysis_output_root"] = str(artifact_root / "runs" / row["model_id"])
        rows.append(row_payload)
    return {
        "schema_version": "cdi_hybrid_spectral_ffno_parameter_space_v1",
        "study_scope": "cdi_only_decision_support",
        "authoritative_anchor_root": str(authoritative_root),
        "rows": rows,
    }


def build_reference_runs_payload(*, authoritative_root: Path) -> Dict[str, Any]:
    reused_rows = []
    for row in REUSED_ROWS:
        row_payload = dict(row)
        row_payload.update(_required_row_paths(authoritative_root, row["model_id"]))
        reused_rows.append(row_payload)
    return {
        "schema_version": "cdi_hybrid_spectral_ffno_reference_runs_v1",
        "authoritative_root": str(authoritative_root),
        "reused_rows": reused_rows,
    }


def render_preflight_note(
    *,
    authoritative_root: Path,
    matrix_path: Path,
    reference_runs_path: Path,
) -> str:
    lines = [
        "# CDI Hybrid-Spectral to FFNO Parameter-Space Preflight",
        "",
        "- Scope: CDI-only decision-support evidence under the opened Phase 2/Phase 3 parallel gate.",
        "- Phase accounting: this remains Phase 3 CDI work and does not satisfy remaining Phase 2 PDEBench requirements.",
        f"- Authoritative reused-anchor root: `{authoritative_root}`",
        f"- Frozen study matrix: `{matrix_path}`",
        f"- Frozen reference-run manifest: `{reference_runs_path}`",
        "",
        "## Frozen Rows",
        "",
    ]
    for row in [*REUSED_ROWS, *FRESH_ROWS]:
        lines.append(f"- `{row['model_id']}` -> `{row['architecture']}`")
    lines.append("")
    return "\n".join(lines)


def build_preflight_artifacts(
    *,
    authoritative_root: Path,
    artifact_root: Path,
    note_path: Path,
    matrix_path: Path,
    reference_runs_path: Path,
) -> Dict[str, Path]:
    matrix_payload = build_study_matrix_payload(
        authoritative_root=Path(authoritative_root),
        artifact_root=Path(artifact_root),
    )
    reference_payload = build_reference_runs_payload(authoritative_root=Path(authoritative_root))
    _write_json(Path(matrix_path), matrix_payload)
    _write_json(Path(reference_runs_path), reference_payload)
    Path(note_path).parent.mkdir(parents=True, exist_ok=True)
    Path(note_path).write_text(
        render_preflight_note(
            authoritative_root=Path(authoritative_root),
            matrix_path=Path(matrix_path),
            reference_runs_path=Path(reference_runs_path),
        ),
        encoding="utf-8",
    )
    return {
        "study_matrix_path": Path(matrix_path),
        "reference_runs_path": Path(reference_runs_path),
        "note_path": Path(note_path),
    }
