#!/usr/bin/env python
"""Regenerate the run-level ``model_manifest.json`` for the lines128 SRU-Net branch
/ objective ablation backlog item with the post-fix wrapper helpers.

This utility exists because the original launcher invocation for
``runs/ablation_20260505T010316Z`` predates the
``--manifest-claim-boundary``/locked-row-status fix in
``scripts/studies/grid_lines_compare_wrapper.py``. The originally-emitted
manifest defaulted ``claim_boundary`` to ``grid_lines_compare_bundle`` and let
auto-promotion widen the new ablation rows to ``paper_grade``.

Running this script re-derives the manifest from the existing fresh per-row
artifacts using the now-locked row specs, writes a fresh
``model_manifest.json`` next to those artifacts, and emits a
``model_manifest_regeneration_log.json`` audit record that captures the
original launcher invocation, the regeneration command, the input artifact
paths, the resulting ``claim_boundary``/per-row ``row_status``, and the git
commit at regeneration time.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.grid_lines_compare_wrapper import (  # noqa: E402
    DEFAULT_TORCH_ROW_SPECS,
    _enrich_paper_row_payload,
    _recover_torch_row_payload,
)
from scripts.studies.metrics_tables import write_model_manifest  # noqa: E402

ABLATION_ROW_IDS = (
    "pinn_hybrid_resnet_encoder_conv_only",
    "pinn_hybrid_resnet_encoder_spectral_only",
    "supervised_hybrid_resnet",
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def regenerate(run_root: Path, *, n_value: int = 128) -> dict:
    run_root = run_root.resolve()
    invocation_path = run_root / "invocation.json"
    invocation_payload = json.loads(invocation_path.read_text(encoding="utf-8"))
    parsed_args = invocation_payload.get("parsed_args", {})
    dataset_manifest = json.loads(
        (run_root / "dataset_identity_manifest.json").read_text(encoding="utf-8")
    )
    train_npz = Path(dataset_manifest["train_npz"]["path"])
    test_npz = Path(dataset_manifest["test_npz"]["path"])
    probe_npz = Path(dataset_manifest["probe_npz"]["path"])

    row_payloads: dict[str, dict] = {}
    for model_id in ABLATION_ROW_IDS:
        metrics_path = run_root / "runs" / model_id / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        recovered = _recover_torch_row_payload(
            output_dir=run_root,
            model_id=model_id,
            n_value=n_value,
            metrics=metrics,
        )
        row_payloads[model_id] = _enrich_paper_row_payload(
            model_id=model_id,
            payload=recovered,
            output_dir=run_root,
            train_npz=train_npz,
            test_npz=test_npz,
            seed=int(parsed_args.get("seed", 3)),
            nimgs_train=int(parsed_args.get("nimgs_train", 2)),
            nimgs_test=int(parsed_args.get("nimgs_test", 2)),
            gridsize=int(parsed_args.get("gridsize", 1)),
            set_phi=bool(parsed_args.get("set_phi", True)),
            probe_npz=probe_npz,
            dataset_source=str(parsed_args.get("dataset_source", "synthetic_lines")),
            probe_source=str(parsed_args.get("probe_source", "custom")),
            probe_scale_mode=str(parsed_args.get("probe_scale_mode", "pad_extrapolate")),
            row_spec=DEFAULT_TORCH_ROW_SPECS.get(model_id, {}),
        )

    manifest_path = write_model_manifest(
        output_dir=run_root,
        row_payloads=row_payloads,
        benchmark_status="decision_support_complete",
        claim_boundary="decision_support_append_only",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    log_path = run_root / "model_manifest_regeneration_log.json"
    log_payload = {
        "regeneration_kind": "post_run_manifest_rebuild",
        "purpose": (
            "Original launcher invocation predates the "
            "--manifest-claim-boundary / locked-row-status fix in "
            "scripts/studies/grid_lines_compare_wrapper.py. The originally "
            "emitted run-level manifest used the default "
            "grid_lines_compare_bundle claim boundary and let auto-promotion "
            "widen the ablation rows to paper_grade. This regeneration replays "
            "_recover_torch_row_payload + _enrich_paper_row_payload + "
            "write_model_manifest with the now-locked DEFAULT_TORCH_ROW_SPECS "
            "and an explicit decision_support_append_only claim boundary."
        ),
        "regenerated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "regeneration_command": (
            "python "
            + shlex.quote(
                str(Path(__file__).resolve().relative_to(REPO_ROOT))
            )
            + " --run-root "
            + shlex.quote(str(run_root.relative_to(REPO_ROOT)))
        ),
        "original_launcher_invocation": {
            "invocation_path": str(invocation_path.relative_to(REPO_ROOT)),
            "command": invocation_payload.get("command"),
            "argv": invocation_payload.get("argv"),
            "manifest_claim_boundary_in_argv": (
                "--manifest-claim-boundary"
                in (invocation_payload.get("argv") or [])
            ),
            "git_commit_at_launch": (
                invocation_payload.get("extra", {}).get("git_commit")
            ),
        },
        "inputs": {
            "row_metrics_paths": [
                str((run_root / "runs" / m / "metrics.json").relative_to(REPO_ROOT))
                for m in ABLATION_ROW_IDS
            ],
            "row_invocation_paths": [
                str((run_root / "runs" / m / "invocation.json").relative_to(REPO_ROOT))
                for m in ABLATION_ROW_IDS
            ],
            "row_completion_proof_paths": [
                str((run_root / "runs" / m / "exit_code_proof.json").relative_to(REPO_ROOT))
                for m in ABLATION_ROW_IDS
            ],
        },
        "output": {
            "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "claim_boundary": manifest.get("claim_boundary"),
            "benchmark_status": manifest.get("benchmark_status"),
            "row_statuses": [
                {"model_id": row.get("model_id"), "row_status": row.get("row_status")}
                for row in manifest.get("rows", [])
            ],
        },
    }
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    return log_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Compare-wrapper run root containing runs/<model_id>/ subdirs.",
    )
    parser.add_argument("--N", type=int, default=128, help="Locked sample size N.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    log_payload = regenerate(args.run_root, n_value=args.N)
    print(json.dumps(log_payload, indent=2))


if __name__ == "__main__":
    main()
