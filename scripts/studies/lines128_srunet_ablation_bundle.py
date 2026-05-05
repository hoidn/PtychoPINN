#!/usr/bin/env python
"""Collate the SRU-Net branch / objective ablation bundle for the lines128 backlog item.

This helper produces an append-only ablation bundle that:
- promotes the existing ``pinn_hybrid_resnet`` baseline row by lineage reference (no rerun),
- ingests fresh row-local artifacts for ``pinn_hybrid_resnet_encoder_conv_only``,
  ``pinn_hybrid_resnet_encoder_spectral_only``, and ``supervised_hybrid_resnet`` from the
  grid-lines compare wrapper output, and
- writes a merged metrics payload + ablation manifest under the new artifact root.

The bundle is decision-support only and does not replace the locked six-row CDI authority.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

ABLATION_ROW_IDS = (
    "pinn_hybrid_resnet_encoder_conv_only",
    "pinn_hybrid_resnet_encoder_spectral_only",
    "supervised_hybrid_resnet",
)

ABLATION_LABELS = {
    "pinn_hybrid_resnet": "Hybrid ResNet + PINN",
    "pinn_hybrid_resnet_encoder_conv_only": "Hybrid ResNet (conv-only encoder) + PINN",
    "pinn_hybrid_resnet_encoder_spectral_only": "Hybrid ResNet (spectral-only encoder) + PINN",
    "supervised_hybrid_resnet": "Hybrid ResNet + supervised",
}

CLAIM_BOUNDARY = "decision_support_append_only"


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


def _row_completion_proof(run_dir: Path) -> Optional[Dict[str, Any]]:
    proof = run_dir / "launcher_completion.json"
    if not proof.exists():
        return None
    return _load_json(proof)


def _baseline_row_payload(baseline_root: Path) -> Dict[str, Any]:
    """Load the baseline pinn_hybrid_resnet row metrics from the authoritative root.

    ``baseline_root`` should be the merged-bundle root that contains the row-local
    metrics under ``runs/pinn_hybrid_resnet/metrics.json``.
    """
    row_dir = _required_path(
        baseline_root / "runs" / "pinn_hybrid_resnet",
        "baseline pinn_hybrid_resnet runs/<row> dir",
    )
    metrics_payload = _row_metrics_from_runner(row_dir)
    invocation_path = row_dir / "invocation.json"
    return {
        "model_id": "pinn_hybrid_resnet",
        "model_label": ABLATION_LABELS["pinn_hybrid_resnet"],
        "training_procedure": "pinn",
        "architecture_id": "hybrid_resnet",
        "row_provenance": {
            "evidence_source": "promoted_by_lineage",
            "row_dir": str(row_dir.relative_to(REPO_ROOT)) if row_dir.is_absolute() and REPO_ROOT in row_dir.parents else str(row_dir),
            "metrics_path": str((row_dir / "metrics.json").relative_to(REPO_ROOT)) if (row_dir / "metrics.json").is_absolute() and REPO_ROOT in (row_dir / "metrics.json").parents else str(row_dir / "metrics.json"),
            "invocation_present": invocation_path.exists(),
        },
        "metrics": metrics_payload,
    }


def _fresh_row_payload(run_root: Path, model_id: str) -> Dict[str, Any]:
    row_dir = _required_path(
        run_root / "runs" / model_id,
        f"fresh row directory for {model_id}",
    )
    metrics_payload = _row_metrics_from_runner(row_dir)
    completion = _row_completion_proof(row_dir)

    overrides: Dict[str, Any] = {}
    if model_id == "pinn_hybrid_resnet_encoder_conv_only":
        overrides["hybrid_encoder_branch_select"] = "conv_only"
        training_procedure = "pinn"
    elif model_id == "pinn_hybrid_resnet_encoder_spectral_only":
        overrides["hybrid_encoder_branch_select"] = "spectral_only"
        training_procedure = "pinn"
    elif model_id == "supervised_hybrid_resnet":
        training_procedure = "supervised"
    else:
        raise ValueError(f"Unknown ablation row id {model_id!r}")

    return {
        "model_id": model_id,
        "model_label": ABLATION_LABELS[model_id],
        "training_procedure": training_procedure,
        "architecture_id": "hybrid_resnet",
        "overrides": overrides,
        "row_provenance": {
            "evidence_source": "fresh_run",
            "row_dir": str(row_dir),
            "invocation_present": (row_dir / "invocation.json").exists(),
            "launcher_completion_present": completion is not None,
        },
        "metrics": metrics_payload,
        "launcher_completion": completion,
    }


def build_ablation_bundle(
    *,
    run_root: Path,
    baseline_root: Path,
    bundle_dir: Path,
) -> Dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    baseline_payload = _baseline_row_payload(baseline_root)
    fresh_payloads = [_fresh_row_payload(run_root, model_id) for model_id in ABLATION_ROW_IDS]

    rows = [baseline_payload] + fresh_payloads
    metrics_by_model: Dict[str, Any] = {row["model_id"]: row["metrics"] for row in rows}

    manifest = {
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_scope": "lines128_srunet_branch_objective_ablation_decision_support",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_contract_id": "cdi_lines128_seed3",
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
            }
            for row in rows
        ],
    }

    metrics_payload = {
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_scope": manifest["evidence_scope"],
        "fixed_contract_id": manifest["fixed_contract_id"],
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
