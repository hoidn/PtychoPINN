#!/usr/bin/env python
"""Collate the SRU-Net ConvNeXt-bottleneck ablation bundle for the lines128 backlog item.

This helper produces an append-only ablation bundle that:
- promotes the existing ``pinn_hybrid_resnet`` baseline row by lineage reference
  (no rerun) from the authoritative six-row CDI bundle root, and
- ingests fresh row-local artifacts for ``pinn_hybrid_resnet_convnext_bottleneck``
  from the grid-lines compare wrapper output, then writes a merged metrics
  payload + ablation manifest under the new artifact root.

The bundle is decision-support only; it does not replace the locked six-row CDI
authority.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

ABLATION_ROW_IDS = ("pinn_hybrid_resnet_convnext_bottleneck",)

ABLATION_LABELS = {
    "pinn_hybrid_resnet": "Hybrid ResNet + PINN",
    "pinn_hybrid_resnet_convnext_bottleneck": "Hybrid ResNet (ConvNeXt bottleneck) + PINN",
}

ABLATION_ARCHITECTURES = {
    "pinn_hybrid_resnet": "hybrid_resnet",
    "pinn_hybrid_resnet_convnext_bottleneck": "hybrid_resnet_convnext_bottleneck",
}

CLAIM_BOUNDARY = "decision_support_append_only"
EVIDENCE_SCOPE = "lines128_srunet_convnext_bottleneck_decision_support"
FIXED_CONTRACT_ID = "cdi_lines128_seed3"

COMPLETION_PROOF_CANDIDATES = ("exit_code_proof.json", "launcher_completion.json")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _required_path(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {what}: {path}")
    return path


def _row_metrics_from_runner(run_dir: Path) -> Dict[str, Any]:
    metrics_path = _required_path(run_dir / "metrics.json", "row metrics.json")
    return _load_json(metrics_path)


def _row_completion_proof(run_dir: Path) -> Dict[str, Any]:
    """Load the row-local completion-proof artifact, failing loudly if absent."""
    for name in COMPLETION_PROOF_CANDIDATES:
        candidate = run_dir / name
        if candidate.exists():
            payload = _load_json(candidate)
            return {"path": candidate, "payload": payload}
    raise FileNotFoundError(
        "Missing row-local completion-proof artifact in "
        f"{run_dir}; expected one of {COMPLETION_PROOF_CANDIDATES}"
    )


def _baseline_row_payload(baseline_root: Path) -> Dict[str, Any]:
    row_dir = _required_path(
        baseline_root / "runs" / "pinn_hybrid_resnet",
        "baseline pinn_hybrid_resnet runs/<row> dir",
    )
    metrics_payload = _row_metrics_from_runner(row_dir)
    invocation_path = row_dir / "invocation.json"
    baseline_completion: Optional[Dict[str, Any]] = None
    baseline_completion_filename: Optional[str] = None
    for name in COMPLETION_PROOF_CANDIDATES:
        candidate = row_dir / name
        if candidate.exists():
            baseline_completion = _load_json(candidate)
            baseline_completion_filename = name
            break
    return {
        "model_id": "pinn_hybrid_resnet",
        "model_label": ABLATION_LABELS["pinn_hybrid_resnet"],
        "training_procedure": "pinn",
        "architecture_id": ABLATION_ARCHITECTURES["pinn_hybrid_resnet"],
        "row_provenance": {
            "evidence_source": "promoted_by_lineage",
            "row_dir": str(row_dir),
            "metrics_path": str(row_dir / "metrics.json"),
            "invocation_present": invocation_path.exists(),
            "completion_proof_present": baseline_completion is not None,
            "completion_proof_filename": baseline_completion_filename,
        },
        "metrics": metrics_payload,
        "completion_proof": baseline_completion,
    }


def _fresh_row_payload(run_root: Path, model_id: str) -> Dict[str, Any]:
    if model_id not in ABLATION_ROW_IDS:
        raise ValueError(f"Unknown ablation row id {model_id!r}")
    row_dir = _required_path(
        run_root / "runs" / model_id,
        f"fresh row directory for {model_id}",
    )
    metrics_payload = _row_metrics_from_runner(row_dir)
    completion = _row_completion_proof(row_dir)
    completion_path = completion["path"]
    return {
        "model_id": model_id,
        "model_label": ABLATION_LABELS[model_id],
        "training_procedure": "pinn",
        "architecture_id": ABLATION_ARCHITECTURES[model_id],
        "overrides": {
            "architecture": ABLATION_ARCHITECTURES[model_id],
            "convnext_bottleneck_layerscale_init": 0.1,
            "changed_factor": "bottleneck_block_family_only",
        },
        "row_provenance": {
            "evidence_source": "fresh_run",
            "row_dir": str(row_dir),
            "invocation_present": (row_dir / "invocation.json").exists(),
            "completion_proof_present": True,
            "completion_proof_filename": completion_path.name,
            "completion_proof_path": str(completion_path),
        },
        "metrics": metrics_payload,
        "completion_proof": completion["payload"],
    }


BUNDLE_OUTPUT_FILENAMES = ("ablation_manifest.json", "ablation_metrics.json")


def build_ablation_bundle(
    *,
    run_root: Path,
    baseline_root: Path,
    bundle_dir: Path,
) -> Dict[str, Any]:
    existing = [
        bundle_dir / name
        for name in BUNDLE_OUTPUT_FILENAMES
        if (bundle_dir / name).exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite append-only ablation bundle artifacts at "
            f"{bundle_dir}: {[str(path) for path in existing]}. "
            "Point --bundle-dir at a fresh leaf instead."
        )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    baseline_payload = _baseline_row_payload(baseline_root)
    fresh_payloads = [_fresh_row_payload(run_root, model_id) for model_id in ABLATION_ROW_IDS]

    rows = [baseline_payload] + fresh_payloads
    metrics_by_model: Dict[str, Any] = {row["model_id"]: row["metrics"] for row in rows}

    manifest = {
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_scope": EVIDENCE_SCOPE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_contract_id": FIXED_CONTRACT_ID,
        "baseline_lineage": {
            "model_id": "pinn_hybrid_resnet",
            "evidence_source": "promoted_by_lineage",
            "authoritative_root": str(baseline_root),
        },
        "fresh_run_root": str(run_root),
        "rows": [
            {
                "model_id": row["model_id"],
                "model_label": row["model_label"],
                "training_procedure": row["training_procedure"],
                "architecture_id": row["architecture_id"],
                "evidence_source": row["row_provenance"]["evidence_source"],
                "overrides": row.get("overrides", {}),
                "completion_proof_present": row["row_provenance"]["completion_proof_present"],
                "completion_proof_filename": row["row_provenance"].get("completion_proof_filename"),
            }
            for row in rows
        ],
    }

    metrics_payload = {
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_scope": EVIDENCE_SCOPE,
        "fixed_contract_id": FIXED_CONTRACT_ID,
        "rows": rows,
        "metrics_by_model": metrics_by_model,
    }

    (bundle_dir / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
    )
    (bundle_dir / "ablation_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    return manifest


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Compare-wrapper output root containing runs/<model_id>/ subdirs.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="Authoritative bundle root that holds runs/pinn_hybrid_resnet/.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Destination directory for ablation_manifest.json and ablation_metrics.json.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    manifest = build_ablation_bundle(
        run_root=args.run_root,
        baseline_root=args.baseline_root,
        bundle_dir=args.bundle_dir,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
