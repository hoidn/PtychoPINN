"""Preflight helpers for the CDI hybrid-spectral to FFNO parameter-space study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_FIXED_CONTRACT: Dict[str, Any] = {
    "N": 128,
    "gridsize": 1,
    "dataset_source": "synthetic_lines",
    "set_phi": True,
    "probe_source": "custom",
    "probe_npz": "datasets/Run1084_recon3_postPC_shrunk_3.npz",
    "probe_scale_mode": "pad_extrapolate",
    "probe_smoothing_sigma": 0.5,
    "probe_mask_diameter": None,
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
    "torch_mae_pred_l2_match_target": False,
    "torch_output_mode": "real_imag",
    "fno_modes": 12,
    "fno_width": 32,
    "fno_blocks": 4,
    "fno_cnn_blocks": 2,
}

EXPECTED_ROW_ARGS: Dict[str, Any] = {
    "seed": 3,
    "epochs": 40,
    "learning_rate": 2e-4,
    "generator_output_mode": "real_imag",
    "N": 128,
    "gridsize": 1,
    "torch_loss_mode": "mae",
    "torch_mae_pred_l2_match_target": False,
    "probe_mask": False,
    "fno_modes": 12,
    "fno_width": 32,
    "fno_blocks": 4,
    "fno_cnn_blocks": 2,
    "scheduler": "ReduceLROnPlateau",
    "plateau_factor": 0.5,
    "plateau_patience": 2,
    "plateau_min_lr": 1e-4,
    "plateau_threshold": 0.0,
    "hybrid_downsample_steps": 2,
    "hybrid_downsample_op": "stride_conv",
    "spectral_bottleneck_blocks": 6,
    "spectral_bottleneck_modes": 12,
    "spectral_bottleneck_share_weights": True,
    "spectral_bottleneck_gate_init": 0.1,
    "spectral_bottleneck_gate_mode": "shared",
}

STUDY_CLAIM_BOUNDARY = "no_paper_promotion_without_later_authority"

REUSED_ROWS = [
    {
        "model_id": "pinn_hybrid_resnet",
        "model_label": "Hybrid ResNet + PINN",
        "architecture": "hybrid_resnet",
        "row_kind": "reused_anchor",
        "nearest_anchor": None,
        "expression_path": "authoritative row copy from the fixed complete-table bundle",
    },
    {
        "model_id": "pinn_spectral_resnet_bottleneck_net",
        "model_label": "Spectral ResNet Bottleneck + PINN",
        "architecture": "spectral_resnet_bottleneck_net",
        "row_kind": "reused_anchor",
        "nearest_anchor": None,
        "expression_path": "authoritative row copy from the fixed complete-table bundle",
    },
    {
        "model_id": "pinn_ffno",
        "model_label": "FFNO + PINN",
        "architecture": "ffno",
        "row_kind": "reused_anchor",
        "nearest_anchor": None,
        "expression_path": "authoritative row copy from the fixed complete-table bundle",
    },
]

FRESH_ROWS = [
    {
        "model_id": "pinn_spectral_resnet_bottleneck_ds1",
        "model_label": "Spectral ResNet Bottleneck DS1 + PINN",
        "architecture": "spectral_resnet_bottleneck_net",
        "row_kind": "fresh_bridge",
        "nearest_anchor": "pinn_spectral_resnet_bottleneck_net",
        "expression_path": "runner override on `spectral_resnet_bottleneck_net`",
        "overrides": {"hybrid_downsample_steps": 1},
        "row_status": "decision_support",
        "lock_row_status": True,
    },
    {
        "model_id": "pinn_spectral_resnet_bottleneck_linear_decoder",
        "model_label": "Spectral ResNet Linear Decoder + PINN",
        "architecture": "spectral_resnet_bottleneck_linear_decoder",
        "row_kind": "fresh_bridge",
        "nearest_anchor": "pinn_spectral_resnet_bottleneck_net",
        "expression_path": "generator registry entry `spectral_resnet_bottleneck_linear_decoder`",
        "overrides": {},
        "row_status": "decision_support",
        "lock_row_status": True,
    },
    {
        "model_id": "pinn_hybrid_resnet_ffno_bottleneck",
        "model_label": "Hybrid ResNet FFNO Bottleneck + PINN",
        "architecture": "hybrid_resnet_ffno_bottleneck",
        "row_kind": "fresh_bridge",
        "nearest_anchor": "pinn_hybrid_resnet",
        "expression_path": "generator registry entry `hybrid_resnet_ffno_bottleneck`",
        "overrides": {},
        "row_status": "decision_support",
        "lock_row_status": True,
    },
]

REUSED_PROVENANCE_RULES = {
    "library": "accepted from a completed row-local launcher record with git/runtime provenance",
    "backfilled_from_wrapper_contract": (
        "accepted from a completed promotion/backfill record with parent invocation "
        "and recovered exit-code proof"
    ),
}

AUTHORITATIVE_BUNDLE_FILES = (
    "paper_benchmark_manifest.json",
    "dataset_identity_manifest.json",
    "split_manifest.json",
    "invocation.json",
    "metrics.json",
    "model_manifest.json",
)


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_equal(*, scope: str, key: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{scope} contract mismatch for {key}: expected {expected!r}, found {actual!r}")


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


def _load_authoritative_manifests(authoritative_root: Path) -> Dict[str, Dict[str, Any]]:
    manifests: Dict[str, Dict[str, Any]] = {}
    missing = [name for name in AUTHORITATIVE_BUNDLE_FILES if not (authoritative_root / name).exists()]
    if missing:
        raise ValueError(
            "authoritative bundle is missing required manifests: "
            + ", ".join(str(authoritative_root / name) for name in missing)
        )
    for name in AUTHORITATIVE_BUNDLE_FILES:
        manifests[name] = _load_json(authoritative_root / name)
    return manifests


def _validate_authoritative_bundle(authoritative_root: Path) -> Dict[str, Any]:
    manifests = _load_authoritative_manifests(authoritative_root)
    fixed_contract = manifests["paper_benchmark_manifest.json"].get("fixed_contract", {})
    for key, expected in EXPECTED_FIXED_CONTRACT.items():
        _expect_equal(
            scope="authoritative fixed contract",
            key=key,
            actual=fixed_contract.get(key),
            expected=expected,
        )

    split_manifest = manifests["split_manifest.json"]
    for key in ("seed", "nimgs_train", "nimgs_test", "gridsize", "set_phi"):
        _expect_equal(
            scope="split manifest",
            key=key,
            actual=split_manifest.get(key),
            expected=EXPECTED_FIXED_CONTRACT[key],
        )

    dataset_manifest = manifests["dataset_identity_manifest.json"]
    for key in ("dataset_source", "probe_source", "probe_scale_mode"):
        _expect_equal(
            scope="dataset identity manifest",
            key=key,
            actual=dataset_manifest.get(key),
            expected=EXPECTED_FIXED_CONTRACT[key],
        )

    validation = {
        "status": "accepted",
        "authoritative_bundle_sha256": {
            name: _sha256(authoritative_root / name) for name in AUTHORITATIVE_BUNDLE_FILES
        },
        "bundle_manifests": {
            name: str(authoritative_root / name) for name in AUTHORITATIVE_BUNDLE_FILES
        },
        "fixed_contract": fixed_contract,
    }
    return validation


def _validate_reused_row(*, authoritative_root: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    model_id = str(row["model_id"])
    run_dir = authoritative_root / "runs" / model_id
    recon_path = authoritative_root / "recons" / model_id / "recon.npz"
    required = {
        "invocation_json": run_dir / "invocation.json",
        "config_json": run_dir / "config.json",
        "history_json": run_dir / "history.json",
        "metrics_json": run_dir / "metrics.json",
        "recon_npz": recon_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise ValueError(f"{model_id} is missing required authoritative artifacts: {', '.join(missing)}")

    invocation = _load_json(required["invocation_json"])
    parsed_args = invocation.get("parsed_args", {})
    _expect_equal(
        scope=model_id,
        key="architecture",
        actual=parsed_args.get("architecture"),
        expected=row["architecture"],
    )
    for key, expected in EXPECTED_ROW_ARGS.items():
        _expect_equal(scope=model_id, key=key, actual=parsed_args.get(key), expected=expected)
    _expect_equal(scope=model_id, key="status", actual=invocation.get("status"), expected="completed")
    _expect_equal(scope=model_id, key="exit_code", actual=invocation.get("exit_code"), expected=0)

    extra = invocation.get("extra", {})
    invocation_mode = str(extra.get("invocation_mode", "library"))
    if invocation_mode not in REUSED_PROVENANCE_RULES:
        raise ValueError(f"{model_id} has unsupported invocation_mode={invocation_mode!r}")
    if invocation_mode == "backfilled_from_wrapper_contract":
        if not extra.get("parent_invocation_json"):
            raise ValueError(f"{model_id} is missing parent_invocation_json for backfilled provenance")
        if not extra.get("recovered_exit_code_from_completed_promotion"):
            raise ValueError(f"{model_id} is missing recovered exit-code proof for backfilled provenance")
    else:
        runtime = extra.get("runtime_provenance", {})
        if not extra.get("git_commit"):
            raise ValueError(f"{model_id} is missing git_commit provenance")
        for key in ("python_executable", "cwd", "ptycho_torch_file"):
            if not runtime.get(key):
                raise ValueError(f"{model_id} is missing runtime_provenance.{key}")

    return {
        "status": "accepted",
        "reuse_acceptability": REUSED_PROVENANCE_RULES[invocation_mode],
        "contract_projection": {key: parsed_args.get(key) for key in ["architecture", *EXPECTED_ROW_ARGS.keys()]},
        "source_sha256": {name: _sha256(path) for name, path in required.items()},
        "invocation_mode": invocation_mode,
    }


def _shared_contract_projection(artifact_root: Path) -> Dict[str, Any]:
    dataset_root = Path(artifact_root) / "datasets" / f"N{EXPECTED_FIXED_CONTRACT['N']}" / f"gs{EXPECTED_FIXED_CONTRACT['gridsize']}"
    return {
        **EXPECTED_FIXED_CONTRACT,
        "train_npz": str(dataset_root / "train.npz"),
        "test_npz": str(dataset_root / "test.npz"),
    }


def _fresh_row_contract_projection(row: Dict[str, Any], *, artifact_root: Path) -> Dict[str, Any]:
    shared_contract = _shared_contract_projection(artifact_root)
    projection = {
        "architecture": row["architecture"],
        "train_npz": shared_contract["train_npz"],
        "test_npz": shared_contract["test_npz"],
        "probe_source": shared_contract["probe_source"],
        **EXPECTED_ROW_ARGS,
    }
    projection.update(row.get("overrides", {}))
    return projection


def build_study_matrix_payload(*, authoritative_root: Path, artifact_root: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for row in [*REUSED_ROWS, *FRESH_ROWS]:
        model_id = str(row["model_id"])
        row_payload = dict(row)
        row_payload["display_label"] = row["model_label"]
        row_payload["analysis_output_root"] = str(artifact_root / "runs" / model_id)
        row_payload["analysis_run_dir"] = str(artifact_root / "runs" / model_id)
        row_payload["analysis_recon_dir"] = str(artifact_root / "recons" / model_id)
        if row["row_kind"] == "fresh_bridge":
            row_payload["contract_projection"] = _fresh_row_contract_projection(row, artifact_root=artifact_root)
        rows.append(row_payload)
    return {
        "schema_version": "cdi_hybrid_spectral_ffno_parameter_space_v2",
        "study_scope": "cdi_only_decision_support",
        "claim_boundary": STUDY_CLAIM_BOUNDARY,
        "authoritative_anchor_root": str(authoritative_root),
        "materialization_policy": "copy_on_write",
        "shared_contract": _shared_contract_projection(artifact_root),
        "rows": rows,
    }


def build_reference_runs_payload(*, authoritative_root: Path) -> Dict[str, Any]:
    bundle_validation = _validate_authoritative_bundle(authoritative_root)
    reused_rows = []
    for row in REUSED_ROWS:
        row_payload = dict(row)
        row_payload["display_label"] = row["model_label"]
        row_payload.update(_required_row_paths(authoritative_root, row["model_id"]))
        row_payload["validation"] = _validate_reused_row(
            authoritative_root=authoritative_root,
            row=row,
        )
        reused_rows.append(row_payload)
    return {
        "schema_version": "cdi_hybrid_spectral_ffno_reference_runs_v2",
        "authoritative_root": str(authoritative_root),
        "fixed_contract": bundle_validation["fixed_contract"],
        "bundle_validation": bundle_validation,
        "reused_rows": reused_rows,
    }


def render_preflight_note(
    *,
    authoritative_root: Path,
    matrix_path: Path,
    reference_runs_path: Path,
    matrix_payload: Dict[str, Any],
    reference_payload: Dict[str, Any],
) -> str:
    fixed_contract = reference_payload["fixed_contract"]
    lines = [
        "# CDI Hybrid-Spectral to FFNO Parameter-Space Preflight",
        "",
        "- Scope: CDI-only decision-support evidence under the opened Phase 2/Phase 3 parallel gate.",
        "- Phase accounting: this remains Phase 3 CDI work and does not satisfy remaining Phase 2 PDEBench requirements.",
        "- Claim boundary: no reused or fresh row is paper-facing without a later checked-in promotion authority.",
        f"- Authoritative reused-anchor root: `{authoritative_root}`",
        f"- Frozen study matrix: `{matrix_path}`",
        f"- Frozen reference-run manifest: `{reference_runs_path}`",
        "",
        "## Fixed Contract",
        "",
        f"- `N={fixed_contract['N']}`, `gridsize={fixed_contract['gridsize']}`, `set_phi={fixed_contract['set_phi']}`, `seed={fixed_contract['seed']}`",
        f"- probe: `{fixed_contract['probe_source']}`, `{fixed_contract['probe_scale_mode']}`, `sigma={fixed_contract['probe_smoothing_sigma']}`",
        f"- split: `nimgs_train={fixed_contract['nimgs_train']}`, `nimgs_test={fixed_contract['nimgs_test']}`, `nphotons={fixed_contract['nphotons']}`",
        (
            f"- training: `epochs={fixed_contract['torch_epochs']}`, "
            f"`lr={fixed_contract['torch_learning_rate']}`, `{fixed_contract['torch_scheduler']}`, "
            f"`plateau_factor={fixed_contract['torch_plateau_factor']}`, "
            f"`plateau_patience={fixed_contract['torch_plateau_patience']}`, "
            f"`plateau_min_lr={fixed_contract['torch_plateau_min_lr']}`, "
            f"`plateau_threshold={fixed_contract['torch_plateau_threshold']}`, "
            f"`loss={fixed_contract['torch_loss_mode']}`, "
            f"`output={fixed_contract['torch_output_mode']}`"
        ),
        (
            f"- spectral shell: `fno_modes={fixed_contract['fno_modes']}`, "
            f"`fno_width={fixed_contract['fno_width']}`, "
            f"`fno_blocks={fixed_contract['fno_blocks']}`, "
            f"`fno_cnn_blocks={fixed_contract['fno_cnn_blocks']}`"
        ),
        "",
        "## Output-root layout",
        "",
        "- Live study root layout:",
        f"  - `runs/<model_id>/` under `{Path(matrix_payload['rows'][0]['analysis_run_dir']).parents[1]}`",
        f"  - `recons/<model_id>/` under `{Path(matrix_payload['rows'][0]['analysis_recon_dir']).parents[1]}`",
        "- Display labels are frozen per row and must be reused in summaries and collated outputs.",
        "",
        "## Reused Anchors",
        "",
    ]
    for row in reference_payload["reused_rows"]:
        validation = row["validation"]
        lines.extend(
            [
                f"### `{row['model_id']}`",
                "",
                f"- Display label: `{row['display_label']}`",
                f"- Architecture: `{row['architecture']}`",
                f"- Expression path: {row['expression_path']}",
                f"- Run dir: `{row['run_dir']}`",
                f"- Recon path: `{row['recon_npz']}`",
                f"- Reuse acceptability: {validation['reuse_acceptability']}",
                "",
            ]
        )
    lines.extend(["## Fresh Bridge Rows", ""])
    for row in matrix_payload["rows"]:
        if row["row_kind"] != "fresh_bridge":
            continue
        lines.extend(
            [
                f"### `{row['model_id']}`",
                "",
                f"- Display label: `{row['display_label']}`",
                f"- Nearest anchor: `{row['nearest_anchor']}`",
                f"- Architecture: `{row['architecture']}`",
                f"- Expression path: {row['expression_path']}",
                f"- Output run dir: `{row['analysis_run_dir']}`",
                f"- Output recon dir: `{row['analysis_recon_dir']}`",
                f"- Overrides: `{json.dumps(row.get('overrides', {}), sort_keys=True)}`",
                "",
            ]
        )
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
            matrix_payload=matrix_payload,
            reference_payload=reference_payload,
        ),
        encoding="utf-8",
    )
    return {
        "study_matrix_path": Path(matrix_path),
        "reference_runs_path": Path(reference_runs_path),
        "note_path": Path(note_path),
    }
