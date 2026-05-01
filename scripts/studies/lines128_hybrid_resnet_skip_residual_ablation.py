#!/usr/bin/env python3
"""Lines128 Hybrid ResNet skip/residual ablation helper."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.grid_lines_compare_wrapper import (
    _coerce_paper_row_payload,
    _enrich_paper_row_payload,
    _recover_torch_row_payload,
)
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, run_grid_lines_torch
from scripts.studies.metrics_tables import write_paper_benchmark_bundle


DEFAULT_PLAN_PATH = Path(
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation/execution_plan.md"
)
DEFAULT_SUMMARY_PATH = Path(
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/"
    "lines128_hybrid_resnet_skip_residual_ablation_summary.md"
)
DEFAULT_BASELINE_ROOT = Path(
    ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-29-cdi-lines128-paper-benchmark-execution/runs/"
    "complete_table_20260430T150757Z_repair_tmux"
)
DEFAULT_ARTIFACT_ROOT = Path(
    ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation"
)
BASELINE_ROW_ID = "pinn_hybrid_resnet"
MANDATORY_FRESH_ROWS = [
    "pinn_hybrid_resnet_skip_add",
    "pinn_hybrid_resnet_residual_fixed",
    "pinn_hybrid_resnet_skip_add_residual_fixed",
]
OPTIONAL_FRESH_ROWS = ["pinn_hybrid_resnet_skip_gated_add"]
SUMMARY_CROSS_REFERENCES = {
    "legacy_skip_mode_study": {
        "path": "docs/studies/index.md#hybrid-resnet-mode-skip-sweep",
        "context": "legacy CDI architecture-search context only",
    },
    "cns_skip_add_context": {
        "path": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md",
        "context": "non-CDI CNS skip-add context only",
    },
    "encoder_fusion_backlog": {
        "path": "docs/backlog/in_progress/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation.md",
        "context": "active separate future-work surface",
    },
}
ROW_SPECS = {
    BASELINE_ROW_ID: {
        "model_label": "Hybrid ResNet + PINN (reused baseline)",
        "changed_factor": "reused baseline row only",
        "fresh": False,
        "lock_row_status": True,
        "row_status": "decision_support",
    },
    "pinn_hybrid_resnet_skip_add": {
        "model_label": "Hybrid ResNet + PINN + skip-add",
        "changed_factor": "enable decoder skip fusion with add style",
        "fresh": True,
        "lock_row_status": True,
        "row_status": "decision_support",
    },
    "pinn_hybrid_resnet_residual_fixed": {
        "model_label": "Hybrid ResNet + PINN + fixed residual scale",
        "changed_factor": "fix shared bottleneck residual multiplier at 1.0",
        "fresh": True,
        "lock_row_status": True,
        "row_status": "decision_support",
    },
    "pinn_hybrid_resnet_skip_add_residual_fixed": {
        "model_label": "Hybrid ResNet + PINN + skip-add + fixed residual scale",
        "changed_factor": "combine skip-add with fixed bottleneck residual multiplier",
        "fresh": True,
        "lock_row_status": True,
        "row_status": "decision_support",
    },
    "pinn_hybrid_resnet_skip_gated_add": {
        "model_label": "Hybrid ResNet + PINN + skip-gated-add",
        "changed_factor": "optional gated-add skip comparison only",
        "fresh": True,
        "lock_row_status": True,
        "row_status": "decision_support",
    },
}


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=_json_default), encoding="utf-8")
    return path


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source artifact: {src}")
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _baseline_invocation_args(baseline_root: Path) -> Dict[str, Any]:
    payload = _load_json(baseline_root / "runs" / BASELINE_ROW_ID / "invocation.json")
    parsed_args = payload.get("parsed_args")
    if not isinstance(parsed_args, dict):
        raise ValueError("Baseline invocation is missing parsed_args")
    return dict(parsed_args)


def _baseline_manifest(baseline_root: Path) -> Dict[str, Any]:
    return _load_json(baseline_root / "paper_benchmark_manifest.json")


def _baseline_visual_policy(baseline_root: Path) -> Dict[str, Any]:
    metrics_payload = _load_json(baseline_root / "metrics.json")
    visual = metrics_payload.get("visual_collation")
    if not isinstance(visual, dict):
        raise ValueError("Baseline metrics.json is missing visual_collation")
    return dict(visual)


def _build_row_metadata() -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for row_id, row_spec in ROW_SPECS.items():
        metadata[row_id] = {
            "model_id": row_id,
            "model_label": row_spec["model_label"],
            "architecture_id": "hybrid_resnet",
            "training_procedure": "pinn",
            "changed_factor": row_spec["changed_factor"],
            "fresh": bool(row_spec["fresh"]),
            "required": row_id in MANDATORY_FRESH_ROWS or row_id == BASELINE_ROW_ID,
        }
    return metadata


def _execution_manifest_path(artifact_root: Path) -> Path:
    return artifact_root / "execution_manifest.json"


def _row_contract_audit_path(artifact_root: Path) -> Path:
    return artifact_root / "row_contract_audit.json"


def _cross_reference_manifest_path(artifact_root: Path) -> Path:
    return artifact_root / "cross_reference_manifest.json"


def _write_summary_scaffold(
    *,
    summary_path: Path,
    baseline_root: Path,
    plan_path: Path,
) -> None:
    text = "\n".join(
        [
            "# Lines128 Hybrid ResNet Skip/Residual Ablation Summary",
            "",
            f"- Date: `{datetime.now(timezone.utc).date().isoformat()}`",
            "- Backlog item: `2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation`",
            "- State: `in_progress`",
            f"- Plan: `{plan_path}`",
            f"- Baseline source root: `{baseline_root}`",
            "",
            "## Intended Fresh Row Roster",
            "",
            "- Reused baseline: `pinn_hybrid_resnet`",
            "- Fresh: `pinn_hybrid_resnet_skip_add`",
            "- Fresh: `pinn_hybrid_resnet_residual_fixed`",
            "- Fresh: `pinn_hybrid_resnet_skip_add_residual_fixed`",
            "- Optional after mandatory rows only: `pinn_hybrid_resnet_skip_gated_add`",
            "",
            "## Results",
            "",
            "Pending fresh-row execution and collation.",
            "",
            "## Cross-References",
            "",
            "- Legacy skip/mode study: pending final summary link insertion.",
            "- CNS skip-add context: pending final summary link insertion.",
            "- Encoder-fusion follow-up: pending final summary link insertion.",
            "",
            "## Claim Boundary",
            "",
            "Append-only same-contract CDI ablation. This does not replace the completed six-row CDI headline bundle.",
            "",
            "## Residual Risks",
            "",
            "- Pending execution and verification.",
            "",
        ]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(text, encoding="utf-8")


def prepare_execution_scaffold(
    *,
    baseline_root: Path,
    artifact_root: Path,
    summary_path: Path,
    plan_path: Path,
) -> Dict[str, str]:
    baseline_root = Path(baseline_root)
    artifact_root = Path(artifact_root)
    summary_path = Path(summary_path)
    plan_path = Path(plan_path)
    baseline_manifest = _baseline_manifest(baseline_root)
    visual_policy = _baseline_visual_policy(baseline_root)
    baseline_args = _baseline_invocation_args(baseline_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "runs").mkdir(parents=True, exist_ok=True)
    (artifact_root / "recons").mkdir(parents=True, exist_ok=True)
    (artifact_root / "verification").mkdir(parents=True, exist_ok=True)

    row_metadata = _build_row_metadata()
    execution_manifest = {
        "baseline_source_root": str(baseline_root),
        "baseline_row_id": BASELINE_ROW_ID,
        "baseline_invocation_args": baseline_args,
        "selected_fno_comparator": baseline_manifest["selected_fno_comparator"],
        "fixed_sample_ids": visual_policy["fixed_sample_ids"],
        "fixed_contract": baseline_manifest["fixed_contract"],
        "claim_boundary": "same_contract_cdi_append_only_ablation",
        "mandatory_fresh_rows": list(MANDATORY_FRESH_ROWS),
        "optional_fresh_rows": list(OPTIONAL_FRESH_ROWS),
        "row_metadata": row_metadata,
    }
    row_contract_audit = {
        "baseline_source_root": str(baseline_root),
        "baseline_row_id": BASELINE_ROW_ID,
        "fixed_contract": baseline_manifest["fixed_contract"],
        "frozen_runner_arguments": baseline_args,
        "fresh_row_mutations": {
            "pinn_hybrid_resnet_skip_add": {
                "hybrid_skip_connections": True,
                "hybrid_skip_style": "add",
            },
            "pinn_hybrid_resnet_residual_fixed": {
                "hybrid_resnet_bottleneck_layerscale_mode": "fixed",
                "hybrid_resnet_bottleneck_layerscale_value": 1.0,
            },
            "pinn_hybrid_resnet_skip_add_residual_fixed": {
                "hybrid_skip_connections": True,
                "hybrid_skip_style": "add",
                "hybrid_resnet_bottleneck_layerscale_mode": "fixed",
                "hybrid_resnet_bottleneck_layerscale_value": 1.0,
            },
            "pinn_hybrid_resnet_skip_gated_add": {
                "hybrid_skip_connections": True,
                "hybrid_skip_style": "gated_add",
            },
        },
    }
    cross_reference_manifest = dict(SUMMARY_CROSS_REFERENCES)

    _write_summary_scaffold(
        summary_path=summary_path,
        baseline_root=baseline_root,
        plan_path=plan_path,
    )
    execution_manifest_path = _write_json(_execution_manifest_path(artifact_root), execution_manifest)
    row_contract_audit_path = _write_json(_row_contract_audit_path(artifact_root), row_contract_audit)
    cross_reference_path = _write_json(_cross_reference_manifest_path(artifact_root), cross_reference_manifest)
    return {
        "summary_path": str(summary_path),
        "execution_manifest_path": str(execution_manifest_path),
        "row_contract_audit_path": str(row_contract_audit_path),
        "cross_reference_manifest_path": str(cross_reference_path),
    }


def _row_mutations(row_id: str) -> Dict[str, Any]:
    mutations: Dict[str, Any] = {}
    if row_id == "pinn_hybrid_resnet_skip_add":
        mutations.update(hybrid_skip_connections=True, hybrid_skip_style="add")
    elif row_id == "pinn_hybrid_resnet_residual_fixed":
        mutations.update(
            hybrid_resnet_bottleneck_layerscale_mode="fixed",
            hybrid_resnet_bottleneck_layerscale_value=1.0,
        )
    elif row_id == "pinn_hybrid_resnet_skip_add_residual_fixed":
        mutations.update(
            hybrid_skip_connections=True,
            hybrid_skip_style="add",
            hybrid_resnet_bottleneck_layerscale_mode="fixed",
            hybrid_resnet_bottleneck_layerscale_value=1.0,
        )
    elif row_id == "pinn_hybrid_resnet_skip_gated_add":
        mutations.update(hybrid_skip_connections=True, hybrid_skip_style="gated_add")
    elif row_id != BASELINE_ROW_ID:
        raise ValueError(f"Unsupported ablation row_id: {row_id}")
    return mutations


def build_row_runner_config(*, artifact_root: Path, row_id: str) -> TorchRunnerConfig:
    artifact_root = Path(artifact_root)
    manifest = _load_json(_execution_manifest_path(artifact_root))
    baseline_args = dict(manifest["baseline_invocation_args"])
    row_spec = manifest["row_metadata"][row_id]
    cfg_kwargs: Dict[str, Any] = {
        "train_npz": Path(baseline_args["train_npz"]),
        "test_npz": Path(baseline_args["test_npz"]),
        "output_dir": artifact_root,
        "architecture": str(baseline_args["architecture"]),
        "training_procedure": str(baseline_args.get("training_procedure", "pinn")),
        "model_id_override": row_id,
        "model_label_override": str(row_spec["model_label"]),
        "seed": int(baseline_args["seed"]),
        "epochs": int(baseline_args["epochs"]),
        "batch_size": int(baseline_args["batch_size"]),
        "learning_rate": float(baseline_args["learning_rate"]),
        "infer_batch_size": int(baseline_args["infer_batch_size"]),
        "gradient_clip_val": float(baseline_args.get("gradient_clip_val", 0.0)),
        "gradient_clip_algorithm": str(baseline_args.get("gradient_clip_algorithm", "norm")),
        "generator_output_mode": str(baseline_args["generator_output_mode"]),
        "N": int(baseline_args["N"]),
        "gridsize": int(baseline_args["gridsize"]),
        "probe_source": baseline_args.get("probe_source"),
        "torch_loss_mode": str(baseline_args["torch_loss_mode"]),
        "torch_mae_pred_l2_match_target": bool(
            baseline_args.get("torch_mae_pred_l2_match_target", False)
        ),
        "probe_mask": bool(baseline_args.get("probe_mask", False)),
        "probe_mask_sigma": float(baseline_args.get("probe_mask_sigma", 1.0)),
        "probe_mask_diameter": baseline_args.get("probe_mask_diameter"),
        "fno_modes": int(baseline_args["fno_modes"]),
        "fno_width": int(baseline_args["fno_width"]),
        "fno_blocks": int(baseline_args["fno_blocks"]),
        "fno_cnn_blocks": int(baseline_args["fno_cnn_blocks"]),
        "optimizer": str(baseline_args.get("optimizer", "adam")),
        "weight_decay": float(baseline_args.get("weight_decay", 0.0)),
        "momentum": float(baseline_args.get("momentum", 0.9)),
        "adam_beta1": float(baseline_args.get("adam_beta1", 0.9)),
        "adam_beta2": float(baseline_args.get("adam_beta2", 0.999)),
        "scheduler": str(baseline_args.get("scheduler", "Default")),
        "lr_warmup_epochs": int(baseline_args.get("lr_warmup_epochs", 0)),
        "lr_min_ratio": float(baseline_args.get("lr_min_ratio", 0.1)),
        "plateau_factor": float(baseline_args.get("plateau_factor", 0.5)),
        "plateau_patience": int(baseline_args.get("plateau_patience", 2)),
        "plateau_min_lr": float(baseline_args.get("plateau_min_lr", 5e-5)),
        "plateau_threshold": float(baseline_args.get("plateau_threshold", 0.0)),
        "hybrid_skip_connections": bool(baseline_args.get("hybrid_skip_connections", False)),
        "hybrid_downsample_steps": int(baseline_args.get("hybrid_downsample_steps", 2)),
        "hybrid_downsample_op": str(baseline_args.get("hybrid_downsample_op", "stride_conv")),
        "hybrid_encoder_conv_hidden_scale": float(
            baseline_args.get("hybrid_encoder_conv_hidden_scale", 2.0)
        ),
        "hybrid_encoder_spectral_hidden_scale": float(
            baseline_args.get("hybrid_encoder_spectral_hidden_scale", 1.0)
        ),
        "hybrid_resnet_blocks": int(baseline_args.get("hybrid_resnet_blocks", 6)),
        "hybrid_skip_style": str(baseline_args.get("hybrid_skip_style", "add")),
        "hybrid_resnet_bottleneck_layerscale_mode": str(
            baseline_args.get("hybrid_resnet_bottleneck_layerscale_mode", "learned")
        ),
        "hybrid_resnet_bottleneck_layerscale_value": baseline_args.get(
            "hybrid_resnet_bottleneck_layerscale_value"
        ),
        "logger_backend": "csv",
        "recon_log_num_patches": 4,
    }
    cfg_kwargs.update(_row_mutations(row_id))
    return TorchRunnerConfig(**cfg_kwargs)


def _ensure_reused_baseline_assets(*, baseline_root: Path, artifact_root: Path) -> None:
    _copy_tree(
        baseline_root / "runs" / BASELINE_ROW_ID,
        artifact_root / "runs" / BASELINE_ROW_ID,
    )
    _copy_tree(
        baseline_root / "recons" / BASELINE_ROW_ID,
        artifact_root / "recons" / BASELINE_ROW_ID,
    )
    _copy_tree(
        baseline_root / "recons" / "gt",
        artifact_root / "recons" / "gt",
    )


def _row_metrics_map(run_dir: Path) -> Dict[str, Any]:
    payload = _load_json(run_dir / "metrics.json")
    return dict(payload)


def _row_payloads(
    *,
    artifact_root: Path,
    baseline_root: Path,
    included_rows: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    manifest = _load_json(_execution_manifest_path(artifact_root))
    baseline_manifest = _baseline_manifest(baseline_root)
    fixed_contract = baseline_manifest["fixed_contract"]
    row_payloads: Dict[str, Dict[str, Any]] = {}
    for row_id in included_rows:
        recovered = _recover_torch_row_payload(
            output_dir=artifact_root,
            model_id=row_id,
            n_value=int(fixed_contract["N"]),
            metrics=_row_metrics_map(artifact_root / "runs" / row_id),
        )
        payload = _coerce_paper_row_payload(
            row_id,
            recovered,
            n_value=int(fixed_contract["N"]),
            metrics=_row_metrics_map(artifact_root / "runs" / row_id),
        )
        payload["model_label"] = manifest["row_metadata"][row_id]["model_label"]
        payload["architecture_id"] = "hybrid_resnet"
        payload["training_procedure"] = "pinn"
        caveats = list(payload.get("caveats", []))
        caveats.append("same_contract_cdi_ablation")
        caveats.append("reused_baseline_row" if row_id == BASELINE_ROW_ID else "fresh_ablation_row")
        payload["caveats"] = caveats
        row_payloads[row_id] = _enrich_paper_row_payload(
            model_id=row_id,
            payload=payload,
            output_dir=artifact_root,
            train_npz=Path(manifest["baseline_invocation_args"]["train_npz"]),
            test_npz=Path(manifest["baseline_invocation_args"]["test_npz"]),
            seed=int(fixed_contract["seed"]),
            nimgs_train=int(fixed_contract["nimgs_train"]),
            nimgs_test=int(fixed_contract["nimgs_test"]),
            gridsize=int(fixed_contract["gridsize"]),
            set_phi=bool(fixed_contract["set_phi"]),
            probe_npz=REPO_ROOT / str(fixed_contract["probe_npz"]),
            dataset_source=str(fixed_contract["dataset_source"]),
            probe_source=str(fixed_contract["probe_source"]),
            probe_scale_mode=str(fixed_contract["probe_scale_mode"]),
            row_spec=ROW_SPECS[row_id],
        )
    return row_payloads


def _required_row_files(run_dir: Path) -> None:
    required = [
        run_dir / "invocation.json",
        run_dir / "config.json",
        run_dir / "history.json",
        run_dir / "metrics.json",
        run_dir / "model.pt",
        run_dir / "randomness_contract.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing row artifacts: " + ", ".join(missing))


def _row_recon_path(*, artifact_root: Path, row_id: str) -> Path:
    return Path(artifact_root) / "recons" / row_id / "recon.npz"


def _row_completion_summary(*, artifact_root: Path, row_id: str, reused: bool) -> Dict[str, Any]:
    artifact_root = Path(artifact_root)
    run_dir = artifact_root / "runs" / row_id
    recon_path = _row_recon_path(artifact_root=artifact_root, row_id=row_id)
    _required_row_files(run_dir)
    if not recon_path.exists():
        raise FileNotFoundError(f"Missing row recon artifact: {recon_path}")
    return {
        "row_id": row_id,
        "status": "reused_completed_row" if reused else "completed_row",
        "run_dir": str(run_dir),
        "recon_npz": str(recon_path),
        "metrics": _row_metrics_map(run_dir),
    }


def run_fresh_row(*, artifact_root: Path, row_id: str) -> Dict[str, Any]:
    artifact_root = Path(artifact_root)
    run_dir = artifact_root / "runs" / row_id
    recon_path = _row_recon_path(artifact_root=artifact_root, row_id=row_id)
    if run_dir.exists() and recon_path.exists():
        return _row_completion_summary(artifact_root=artifact_root, row_id=row_id, reused=True)
    cfg = build_row_runner_config(artifact_root=artifact_root, row_id=row_id)
    run_grid_lines_torch(
        cfg,
        invocation_extra={
            "ablation_row_id": row_id,
            "ablation_bundle_root": str(artifact_root),
        },
    )
    return _row_completion_summary(artifact_root=artifact_root, row_id=row_id, reused=False)


def _comparison_summary(
    *,
    artifact_root: Path,
    included_rows: Iterable[str],
) -> Dict[str, Any]:
    baseline_metrics = _row_metrics_map(artifact_root / "runs" / BASELINE_ROW_ID)
    summary_rows: Dict[str, Any] = {}
    for row_id in included_rows:
        metrics = _row_metrics_map(artifact_root / "runs" / row_id)
        deltas: Dict[str, Any] = {}
        if row_id != BASELINE_ROW_ID:
            for metric_name in ("mae", "mse", "psnr", "ssim", "ms_ssim", "frc50", "frc1over7"):
                current = metrics.get(metric_name)
                baseline = baseline_metrics.get(metric_name)
                if isinstance(current, list) and isinstance(baseline, list) and len(current) >= 2 and len(baseline) >= 2:
                    deltas[metric_name] = [
                        float(current[0]) - float(baseline[0]),
                        float(current[1]) - float(baseline[1]),
                    ]
        summary_rows[row_id] = {
            "fresh": bool(ROW_SPECS[row_id]["fresh"]),
            "changed_factor": str(ROW_SPECS[row_id]["changed_factor"]),
            "metrics": metrics,
            "delta_vs_baseline": deltas,
        }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_row_id": BASELINE_ROW_ID,
        "rows": summary_rows,
        "interaction_row_executed": (artifact_root / "runs" / "pinn_hybrid_resnet_skip_add_residual_fixed").exists(),
        "non_cdi_context": dict(SUMMARY_CROSS_REFERENCES),
        "claim_boundary": "same_contract_cdi_append_only_ablation",
    }


def collate_ablation_bundle(*, artifact_root: Path, baseline_root: Path) -> Dict[str, str]:
    artifact_root = Path(artifact_root)
    baseline_root = Path(baseline_root)
    manifest = _load_json(_execution_manifest_path(artifact_root))
    _ensure_reused_baseline_assets(baseline_root=baseline_root, artifact_root=artifact_root)
    included_rows = [BASELINE_ROW_ID]
    for row_id in list(MANDATORY_FRESH_ROWS) + list(OPTIONAL_FRESH_ROWS):
        run_dir = artifact_root / "runs" / row_id
        if run_dir.exists():
            _required_row_files(run_dir)
            included_rows.append(row_id)
    row_payloads = _row_payloads(
        artifact_root=artifact_root,
        baseline_root=baseline_root,
        included_rows=included_rows,
    )
    visual_policy = _baseline_visual_policy(baseline_root)
    bundle_paths = write_paper_benchmark_bundle(
        output_dir=artifact_root,
        row_payloads=row_payloads,
        required_rows=included_rows,
        fixed_sample_ids=visual_policy["fixed_sample_ids"],
        shared_visual_scales=visual_policy["shared_visual_scales"],
        selected_fno_comparator=manifest["selected_fno_comparator"],
        row_statuses={row_id: "decision_support" for row_id in included_rows},
        evidence_scope="same_contract_cdi_ablation",
        claim_boundary="same_contract_cdi_append_only_ablation",
        require_row_provenance=True,
    )
    comparison_summary_path = _write_json(
        artifact_root / "comparison_summary.json",
        _comparison_summary(artifact_root=artifact_root, included_rows=included_rows),
    )
    return {
        "metrics_json": str(bundle_paths["metrics_json"]),
        "model_manifest_json": str(bundle_paths["model_manifest_json"]),
        "metrics_table_csv": str(bundle_paths["metrics_table_csv"]),
        "metrics_table_tex": str(bundle_paths["metrics_table_tex"]),
        "comparison_summary_json": str(comparison_summary_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    prepare_parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    prepare_parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    prepare_parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)

    run_parser = subparsers.add_parser("run-row")
    run_parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    run_parser.add_argument("--row-id", required=True, choices=MANDATORY_FRESH_ROWS + OPTIONAL_FRESH_ROWS)

    collate_parser = subparsers.add_parser("collate")
    collate_parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    collate_parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_execution_scaffold(
            baseline_root=args.baseline_root,
            artifact_root=args.artifact_root,
            summary_path=args.summary_path,
            plan_path=args.plan_path,
        )
    elif args.command == "run-row":
        result = run_fresh_row(artifact_root=args.artifact_root, row_id=args.row_id)
    else:
        result = collate_ablation_bundle(
            artifact_root=args.artifact_root,
            baseline_root=args.baseline_root,
        )
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
