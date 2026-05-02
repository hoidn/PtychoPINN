#!/usr/bin/env python3
"""Lines128 Hybrid ResNet encoder-fusion variants helper."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, run_grid_lines_torch


DEFAULT_PLAN_PATH = Path(
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-21-hybrid-resnet-encoder-fusion-variants/execution_plan.md"
)
DEFAULT_SUMMARY_PATH = Path(
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/"
    "lines128_hybrid_resnet_encoder_fusion_variants_summary.md"
)
DEFAULT_BASELINE_ROOT = Path(
    ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-29-cdi-lines128-paper-benchmark-execution/runs/"
    "complete_table_20260430T150757Z_repair_tmux"
)
DEFAULT_ARTIFACT_ROOT = Path(
    ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-21-hybrid-resnet-encoder-fusion-variants"
)
BASELINE_ROW_ID = "pinn_hybrid_resnet"
RUN_ID = "encoder_fusion_20260502T104230Z"
SCALAR_SCOPE_DECISION = "per_block"
MANDATORY_FRESH_ROWS = [
    "pinn_hybrid_resnet_encoder_layerscale",
    "pinn_hybrid_resnet_encoder_branch_gated",
    "pinn_hybrid_resnet_encoder_branch_gated_layerscale",
]
OPTIONAL_ROWS = ["pinn_hybrid_resnet_encoder_fusion_norm"]
ROW_SPECS: Dict[str, Dict[str, Any]] = {
    BASELINE_ROW_ID: {
        "fresh": False,
        "row_status": "reused_baseline",
        "model_label": "Hybrid ResNet + PINN (reused baseline)",
        "changed_factor": "none (reused baseline)",
        "fusion_mode": "baseline",
    },
    "pinn_hybrid_resnet_encoder_layerscale": {
        "fresh": True,
        "row_status": "fresh",
        "model_label": "Hybrid ResNet + per-block encoder LayerScale + PINN",
        "changed_factor": "encoder_fusion_mode=layerscale; per-block outer LayerScale on the fused update",
        "fusion_mode": "layerscale",
    },
    "pinn_hybrid_resnet_encoder_branch_gated": {
        "fresh": True,
        "row_status": "fresh",
        "model_label": "Hybrid ResNet + per-block encoder branch gates + PINN",
        "changed_factor": "encoder_fusion_mode=branch_gated; per-block spectral and local branch gates",
        "fusion_mode": "branch_gated",
    },
    "pinn_hybrid_resnet_encoder_branch_gated_layerscale": {
        "fresh": True,
        "row_status": "fresh",
        "model_label": "Hybrid ResNet + per-block encoder branch gates + LayerScale + PINN",
        "changed_factor": "encoder_fusion_mode=branch_gated_layerscale; per-block branch gates + per-block LayerScale",
        "fusion_mode": "branch_gated_layerscale",
    },
}


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=_json_default), encoding="utf-8")
    return path


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _relative_to(root: Path, path: Path) -> str:
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def _resolve_run_id(artifact_root: Path, explicit_run_id: str | None = None) -> str:
    if explicit_run_id:
        return explicit_run_id
    manifest_path = _execution_manifest_path(artifact_root)
    if manifest_path.exists():
        payload = _load_json(manifest_path)
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    return RUN_ID


def _artifact_run_root(artifact_root: Path, run_id: str | None = None) -> Path:
    return Path(artifact_root) / "runs" / _resolve_run_id(artifact_root, explicit_run_id=run_id)


def _execution_manifest_path(artifact_root: Path) -> Path:
    return Path(artifact_root) / "execution_manifest.json"


def _row_contract_audit_path(artifact_root: Path) -> Path:
    return Path(artifact_root) / "row_contract_audit.json"


def _baseline_invocation_args(baseline_root: Path) -> Dict[str, Any]:
    payload = _load_json(Path(baseline_root) / "runs" / BASELINE_ROW_ID / "invocation.json")
    parsed_args = payload.get("parsed_args")
    if not isinstance(parsed_args, dict):
        raise ValueError("Baseline invocation is missing parsed_args")
    return dict(parsed_args)


def _baseline_manifest(baseline_root: Path) -> Dict[str, Any]:
    return _load_json(Path(baseline_root) / "paper_benchmark_manifest.json")


def _canonical_dataset_input_paths(baseline_root: Path) -> Dict[str, str]:
    manifest = _baseline_manifest(baseline_root)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("Baseline paper_benchmark_manifest.json is missing dataset paths")
    train_npz = dataset.get("train_npz")
    test_npz = dataset.get("test_npz")
    if not isinstance(train_npz, str) or not train_npz:
        raise ValueError("Baseline dataset manifest is missing dataset.train_npz")
    if not isinstance(test_npz, str) or not test_npz:
        raise ValueError("Baseline dataset manifest is missing dataset.test_npz")
    return {
        "train_npz": train_npz,
        "test_npz": test_npz,
    }


def _write_summary_scaffold(
    *,
    summary_path: Path,
    baseline_root: Path,
    artifact_root: Path,
    plan_path: Path,
    run_id: str,
) -> None:
    lines = [
        "# Lines128 Hybrid ResNet Encoder Fusion Variants Summary",
        "",
        f"- Date: `{datetime.now(timezone.utc).date().isoformat()}`",
        "- Status: `in_progress`",
        "- Backlog item: `2026-04-21-hybrid-resnet-encoder-fusion-variants`",
        f"- Plan path: `{plan_path}`",
        f"- Fixed baseline source root: `{baseline_root}`",
        f"- Authoritative ablation run root: `{_artifact_run_root(artifact_root, run_id=run_id)}`",
        "",
        "## Intended Fresh Row Roster",
        "",
        "- Reused baseline: `pinn_hybrid_resnet`",
        "- Fresh: `pinn_hybrid_resnet_encoder_layerscale`",
        "- Fresh: `pinn_hybrid_resnet_encoder_branch_gated`",
        "- Fresh: `pinn_hybrid_resnet_encoder_branch_gated_layerscale`",
        "- Optional follow-up only: `pinn_hybrid_resnet_encoder_fusion_norm`",
        "",
        "## Results",
        "",
        "Pending row-local provenance repair and/or fresh-row execution through the checked-in helper.",
        "",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_execution_scaffold(
    *,
    baseline_root: Path,
    artifact_root: Path,
    summary_path: Path,
    plan_path: Path,
    run_id: str = RUN_ID,
) -> Dict[str, str]:
    baseline_root = Path(baseline_root)
    artifact_root = Path(artifact_root)
    summary_path = Path(summary_path)
    plan_path = Path(plan_path)
    run_root = _artifact_run_root(artifact_root, run_id=run_id)
    baseline_args = _baseline_invocation_args(baseline_root)
    baseline_manifest = _baseline_manifest(baseline_root)
    dataset_paths = _canonical_dataset_input_paths(baseline_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "runs").mkdir(parents=True, exist_ok=True)
    (run_root / "recons").mkdir(parents=True, exist_ok=True)
    (run_root / "training_runs").mkdir(parents=True, exist_ok=True)

    execution_manifest = {
        "baseline_source_root": str(baseline_root),
        "baseline_row_id": BASELINE_ROW_ID,
        "baseline_invocation_args": baseline_args,
        "baseline_fixed_contract": baseline_manifest.get("fixed_contract", {}),
        "canonical_dataset_input_paths": dataset_paths,
        "run_id": run_id,
        "run_root": str(run_root),
        "scalar_scope_decision": SCALAR_SCOPE_DECISION,
        "mandatory_fresh_rows": list(MANDATORY_FRESH_ROWS),
        "optional_rows": list(OPTIONAL_ROWS),
        "row_metadata": {
            row_id: {
                "model_label": spec["model_label"],
                "changed_factor": spec["changed_factor"],
                "fresh": bool(spec["fresh"]),
                "fusion_mode": spec["fusion_mode"],
                "row_status": spec["row_status"],
            }
            for row_id, spec in ROW_SPECS.items()
        },
    }
    row_contract_audit = {
        "authoritative_baseline_source_root": str(baseline_root),
        "ablation_root": str(artifact_root),
        "ablation_run_root": str(run_root),
        "reused_baseline_row_id": BASELINE_ROW_ID,
        "mandatory_fresh_rows": list(MANDATORY_FRESH_ROWS),
        "scalar_scope_decision": SCALAR_SCOPE_DECISION,
        "canonical_dataset_input_paths": {
            "train_npz": dataset_paths["train_npz"],
            "test_npz": dataset_paths["test_npz"],
        },
        "regenerated_row_local_output_paths_template": {
            "run_root": str(run_root),
            "row_local": {
                "output_dir": str(run_root / "training_runs" / "<row_id>"),
                "row_root": str(run_root / "runs" / "<row_id>"),
                "recon_npz": str(run_root / "recons" / "<row_id>" / "recon.npz"),
                "lightning_logs": str(run_root / "training_runs" / "<row_id>" / "lightning_logs" / "version_0"),
                "checkpoints": str(run_root / "training_runs" / "<row_id>" / "checkpoints"),
            },
        },
        "denylisted_path_keys_from_recovered_baseline_row_config": {
            "keys": ["train_npz", "test_npz", "recon_npz", "output_dir"],
        },
    }

    _write_summary_scaffold(
        summary_path=summary_path,
        baseline_root=baseline_root,
        artifact_root=artifact_root,
        plan_path=plan_path,
        run_id=run_id,
    )
    _write_json(_execution_manifest_path(artifact_root), execution_manifest)
    _write_json(_row_contract_audit_path(artifact_root), row_contract_audit)
    return {
        "summary_path": str(summary_path),
        "execution_manifest_path": str(_execution_manifest_path(artifact_root)),
        "row_contract_audit_path": str(_row_contract_audit_path(artifact_root)),
    }


def build_row_runner_config(*, artifact_root: Path, row_id: str) -> TorchRunnerConfig:
    if row_id not in ROW_SPECS or row_id == BASELINE_ROW_ID:
        raise ValueError(f"Unsupported fresh row_id: {row_id}")
    artifact_root = Path(artifact_root)
    manifest = _load_json(_execution_manifest_path(artifact_root))
    baseline_args = dict(manifest["baseline_invocation_args"])
    dataset_paths = dict(manifest["canonical_dataset_input_paths"])
    run_root = _artifact_run_root(artifact_root)
    spec = ROW_SPECS[row_id]
    return TorchRunnerConfig(
        train_npz=Path(dataset_paths["train_npz"]),
        test_npz=Path(dataset_paths["test_npz"]),
        output_dir=run_root / "training_runs" / row_id,
        artifact_root=run_root,
        architecture=str(baseline_args["architecture"]),
        training_procedure=str(baseline_args.get("training_procedure", "pinn")),
        model_id_override=row_id,
        model_label_override=str(spec["model_label"]),
        seed=int(baseline_args["seed"]),
        epochs=int(baseline_args["epochs"]),
        batch_size=int(baseline_args["batch_size"]),
        learning_rate=float(baseline_args["learning_rate"]),
        infer_batch_size=int(baseline_args["infer_batch_size"]),
        gradient_clip_val=float(baseline_args.get("gradient_clip_val", 0.0)),
        gradient_clip_algorithm=str(baseline_args.get("gradient_clip_algorithm", "norm")),
        generator_output_mode=str(baseline_args["generator_output_mode"]),
        N=int(baseline_args["N"]),
        gridsize=int(baseline_args["gridsize"]),
        probe_source=baseline_args.get("probe_source"),
        torch_loss_mode=str(baseline_args["torch_loss_mode"]),
        torch_mae_pred_l2_match_target=bool(
            baseline_args.get("torch_mae_pred_l2_match_target", False)
        ),
        probe_mask=bool(baseline_args.get("probe_mask", False)),
        probe_mask_sigma=float(baseline_args.get("probe_mask_sigma", 1.0)),
        probe_mask_diameter=baseline_args.get("probe_mask_diameter"),
        fno_modes=int(baseline_args["fno_modes"]),
        fno_width=int(baseline_args["fno_width"]),
        fno_blocks=int(baseline_args["fno_blocks"]),
        fno_cnn_blocks=int(baseline_args["fno_cnn_blocks"]),
        optimizer=str(baseline_args.get("optimizer", "adam")),
        weight_decay=float(baseline_args.get("weight_decay", 0.0)),
        momentum=float(baseline_args.get("momentum", 0.9)),
        adam_beta1=float(baseline_args.get("adam_beta1", 0.9)),
        adam_beta2=float(baseline_args.get("adam_beta2", 0.999)),
        scheduler=str(baseline_args.get("scheduler", "Default")),
        lr_warmup_epochs=int(baseline_args.get("lr_warmup_epochs", 0)),
        lr_min_ratio=float(baseline_args.get("lr_min_ratio", 0.1)),
        plateau_factor=float(baseline_args.get("plateau_factor", 0.5)),
        plateau_patience=int(baseline_args.get("plateau_patience", 2)),
        plateau_min_lr=float(baseline_args.get("plateau_min_lr", 5e-5)),
        plateau_threshold=float(baseline_args.get("plateau_threshold", 0.0)),
        hybrid_skip_connections=bool(baseline_args.get("hybrid_skip_connections", False)),
        hybrid_downsample_steps=int(baseline_args.get("hybrid_downsample_steps", 2)),
        hybrid_downsample_op=str(baseline_args.get("hybrid_downsample_op", "stride_conv")),
        hybrid_encoder_conv_hidden_scale=float(
            baseline_args.get("hybrid_encoder_conv_hidden_scale", 2.0)
        ),
        hybrid_encoder_spectral_hidden_scale=float(
            baseline_args.get("hybrid_encoder_spectral_hidden_scale", 1.0)
        ),
        hybrid_resnet_blocks=int(baseline_args.get("hybrid_resnet_blocks", 6)),
        hybrid_skip_style=str(baseline_args.get("hybrid_skip_style", "add")),
        hybrid_encoder_fusion_mode=str(spec["fusion_mode"]),
        hybrid_encoder_layerscale_init=float(
            baseline_args.get("hybrid_encoder_layerscale_init", 0.1)
        ),
        hybrid_encoder_branch_gate_init=float(
            baseline_args.get("hybrid_encoder_branch_gate_init", 0.1)
        ),
        logger_backend=str(baseline_args.get("torch_logger", "csv")),
        recon_log_num_patches=int(baseline_args.get("recon_log_num_patches", 4)),
    )


def _missing_required_row_files(run_root: Path, row_id: str) -> list[str]:
    row_root = Path(run_root) / "runs" / row_id
    required = [
        row_root / "invocation.json",
        row_root / "invocation.sh",
        row_root / "config.json",
        row_root / "history.json",
        row_root / "metrics.json",
        row_root / "model.pt",
        row_root / "randomness_contract.json",
        Path(run_root) / "recons" / row_id / "recon.npz",
    ]
    return [str(path) for path in required if not path.exists()]


def _required_row_files(run_root: Path, row_id: str) -> None:
    missing = _missing_required_row_files(run_root, row_id)
    if missing:
        raise FileNotFoundError("Missing row artifacts: " + ", ".join(missing))


def run_fresh_row(*, artifact_root: Path, row_id: str) -> Dict[str, Any]:
    run_root = _artifact_run_root(artifact_root)
    row_root = run_root / "runs" / row_id
    if row_root.exists() and not _missing_required_row_files(run_root, row_id):
        raise FileExistsError(
            f"Fresh row {row_id} already has completed artifacts under {row_root}; "
            "use a new run_id instead of relaunching into the same row root."
        )
    cfg = build_row_runner_config(artifact_root=artifact_root, row_id=row_id)
    result = run_grid_lines_torch(
        cfg,
        invocation_extra={
            "ablation_row_id": row_id,
            "ablation_bundle_root": str(artifact_root),
            "ablation_run_root": str(run_root),
        },
    )
    return {
        "row_id": row_id,
        "run_dir": result["run_dir"],
        "recon_npz": result["recon_npz"],
    }


def _extract_fusion_mode_from_hparams(path: Path) -> str | None:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"hybrid_encoder_fusion_mode:\s*([A-Za-z0-9_]+)", text)
    return match.group(1) if match else None


def _shared_version_dirs_by_mode(run_root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for candidate in sorted((Path(run_root) / "lightning_logs").glob("version_*")):
        mode = _extract_fusion_mode_from_hparams(candidate / "hparams.yaml")
        if mode:
            mapping[mode] = candidate
    return mapping


def _shared_checkpoints_by_mode(run_root: Path) -> Dict[str, Dict[str, Path]]:
    mapping: Dict[str, Dict[str, Path]] = {}
    checkpoint_root = Path(run_root) / "checkpoints"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is expected here
        raise RuntimeError("Torch is required to repair checkpoint provenance") from exc

    for candidate in sorted(checkpoint_root.glob("*.ckpt")):
        payload = torch.load(candidate, map_location="cpu")
        hyper = payload.get("hyper_parameters", {}) if isinstance(payload, Mapping) else {}
        overrides = hyper.get("generator_overrides", {}) if isinstance(hyper, Mapping) else {}
        mode = overrides.get("hybrid_encoder_fusion_mode") if isinstance(overrides, Mapping) else None
        if not isinstance(mode, str) or not mode:
            continue
        slot = mapping.setdefault(mode, {})
        if candidate.name.startswith("last"):
            slot["last"] = candidate
        elif "best" not in slot:
            slot["best"] = candidate
    return mapping


def _extract_row_log_section(launch_log: Path, row_id: str) -> str:
    text = Path(launch_log).read_text(encoding="utf-8", errors="ignore")
    marker = f"Launching row: {row_id}"
    start = text.find(marker)
    if start < 0:
        return f"Recovered row-local log section missing explicit marker for {row_id}.\n"
    next_marker = text.find("Launching row:", start + len(marker))
    if next_marker < 0:
        next_marker = len(text)
    return text[start:next_marker].strip() + "\n"


def _write_launcher_completion_json(
    *,
    run_root: Path,
    row_id: str,
    training_root: Path,
    version_source: Path,
    last_checkpoint: Path,
    best_checkpoint: Path | None,
) -> Path:
    payload = {
        "model_id": row_id,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_source": "deterministic_shared_run_repair",
        "run_root": str(run_root),
        "row_root": str(run_root / "runs" / row_id),
        "training_output_dir": str(training_root),
        "checkpoint_dir": str(training_root / "checkpoints"),
        "lightning_log_source": str(version_source),
        "last_checkpoint": str(last_checkpoint),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
        "launch_log": str(run_root / "launch_rows.log"),
        "invocation_json": str(run_root / "runs" / row_id / "invocation.json"),
    }
    return _write_json(run_root / "runs" / row_id / "launcher_completion.json", payload)


def repair_existing_run_provenance(*, artifact_root: Path) -> Dict[str, object]:
    artifact_root = Path(artifact_root)
    run_root = _artifact_run_root(artifact_root)
    version_dirs = _shared_version_dirs_by_mode(run_root)
    checkpoint_dirs = _shared_checkpoints_by_mode(run_root)
    repaired_rows: Dict[str, object] = {}
    for row_id in MANDATORY_FRESH_ROWS:
        spec = ROW_SPECS[row_id]
        mode = str(spec["fusion_mode"])
        if mode not in version_dirs:
            raise FileNotFoundError(f"Missing lightning_logs provenance for fusion mode {mode}")
        if mode not in checkpoint_dirs or "last" not in checkpoint_dirs[mode]:
            raise FileNotFoundError(f"Missing checkpoint provenance for fusion mode {mode}")
        training_root = run_root / "training_runs" / row_id
        (training_root / "lightning_logs").mkdir(parents=True, exist_ok=True)
        (training_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        _copy_path(version_dirs[mode], training_root / "lightning_logs" / "version_0")
        _copy_path(checkpoint_dirs[mode]["last"], training_root / "checkpoints" / "last.ckpt")
        best_checkpoint = checkpoint_dirs[mode].get("best")
        if best_checkpoint is not None:
            _copy_path(best_checkpoint, training_root / "checkpoints" / "best.ckpt")
        manifest_path = _write_json(
            training_root / "training_output_manifest.json",
            {
                "row_id": row_id,
                "repaired_at_utc": datetime.now(timezone.utc).isoformat(),
                "repaired_from_shared_output_dir": str(run_root),
                "source_lightning_version_dir": str(version_dirs[mode]),
                "source_last_checkpoint": str(checkpoint_dirs[mode]["last"]),
                "source_best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
            },
        )
        row_root = run_root / "runs" / row_id
        row_root.mkdir(parents=True, exist_ok=True)
        stdout_log = row_root / "stdout.log"
        if not stdout_log.exists():
            stdout_log.write_text(
                _extract_row_log_section(run_root / "launch_rows.log", row_id),
                encoding="utf-8",
            )
        stderr_log = row_root / "stderr.log"
        if not stderr_log.exists():
            stderr_log.write_text(
                "Direct row stderr was not captured separately during the shared-run launch; "
                "see launcher_completion.json and stdout.log for repaired provenance.\n",
                encoding="utf-8",
            )
        completion_path = _write_launcher_completion_json(
            run_root=run_root,
            row_id=row_id,
            training_root=training_root,
            version_source=version_dirs[mode],
            last_checkpoint=checkpoint_dirs[mode]["last"],
            best_checkpoint=best_checkpoint,
        )
        repaired_rows[row_id] = {
            "training_output_manifest": str(manifest_path),
            "launcher_completion_json": str(completion_path),
        }
    audit_path = _row_contract_audit_path(artifact_root)
    if audit_path.exists():
        audit_payload = _load_json(audit_path)
        audit_payload["ablation_run_root"] = str(run_root)
        audit_payload["regenerated_row_local_output_paths_template"] = {
            "run_root": str(run_root),
            "row_local": {
                "output_dir": str(run_root / "training_runs" / "<row_id>"),
                "row_root": str(run_root / "runs" / "<row_id>"),
                "recon_npz": str(run_root / "recons" / "<row_id>" / "recon.npz"),
                "lightning_logs": str(run_root / "training_runs" / "<row_id>" / "lightning_logs" / "version_0"),
                "checkpoints": str(run_root / "training_runs" / "<row_id>" / "checkpoints"),
            },
        }
        audit_payload["historical_shared_output_dir_repair"] = {
            "repaired_at_utc": datetime.now(timezone.utc).isoformat(),
            "shared_output_dir": str(run_root),
            "repair_reason": (
                "The original launch used a shared output_dir for all fresh rows. "
                "Row-local training roots were reconstructed deterministically from "
                "preserved Lightning fusion-mode metadata and per-row invocation artifacts."
            ),
            "repaired_rows": repaired_rows,
        }
        _write_json(audit_path, audit_payload)
    return {
        "run_root": str(run_root),
        "repaired_rows": repaired_rows,
    }


def _ensure_promoted_baseline(*, artifact_root: Path, baseline_root: Path) -> None:
    run_root = _artifact_run_root(artifact_root)
    baseline_row_root = run_root / "promoted_baseline" / "runs" / BASELINE_ROW_ID
    baseline_recon_root = run_root / "promoted_baseline" / "recons" / BASELINE_ROW_ID
    if not baseline_row_root.exists():
        _copy_path(Path(baseline_root) / "runs" / BASELINE_ROW_ID, baseline_row_root)
    if not baseline_recon_root.exists():
        _copy_path(Path(baseline_root) / "recons" / BASELINE_ROW_ID, baseline_recon_root)
    if not (run_root / "recons" / "gt" / "recon.npz").exists():
        _copy_path(Path(baseline_root) / "recons" / "gt", run_root / "recons" / "gt")


def collate_ablation_bundle(*, artifact_root: Path, baseline_root: Path) -> Dict[str, str]:
    artifact_root = Path(artifact_root)
    baseline_root = Path(baseline_root)
    run_root = _artifact_run_root(artifact_root)
    _ensure_promoted_baseline(artifact_root=artifact_root, baseline_root=baseline_root)
    rows: Dict[str, Dict[str, Any]] = {}
    rows[BASELINE_ROW_ID] = {
        "row_status": ROW_SPECS[BASELINE_ROW_ID]["row_status"],
        "model_label": ROW_SPECS[BASELINE_ROW_ID]["model_label"],
        "changed_factor": ROW_SPECS[BASELINE_ROW_ID]["changed_factor"],
        "artifacts": {
            "config.json": _relative_to(artifact_root, run_root / "promoted_baseline" / "runs" / BASELINE_ROW_ID / "config.json"),
            "history.json": _relative_to(artifact_root, run_root / "promoted_baseline" / "runs" / BASELINE_ROW_ID / "history.json"),
            "metrics.json": _relative_to(artifact_root, run_root / "promoted_baseline" / "runs" / BASELINE_ROW_ID / "metrics.json"),
            "model.pt": _relative_to(artifact_root, run_root / "promoted_baseline" / "runs" / BASELINE_ROW_ID / "model.pt"),
            "invocation.json": _relative_to(artifact_root, run_root / "promoted_baseline" / "runs" / BASELINE_ROW_ID / "invocation.json"),
            "stdout.log": _relative_to(artifact_root, run_root / "promoted_baseline" / "runs" / BASELINE_ROW_ID / "stdout.log"),
            "stderr.log": _relative_to(artifact_root, run_root / "promoted_baseline" / "runs" / BASELINE_ROW_ID / "stderr.log"),
            "launcher_completion.json": _relative_to(artifact_root, run_root / "promoted_baseline" / "runs" / BASELINE_ROW_ID / "launcher_completion.json"),
            "randomness_contract.json": _relative_to(artifact_root, run_root / "promoted_baseline" / "runs" / BASELINE_ROW_ID / "randomness_contract.json"),
            "recon_npz": _relative_to(artifact_root, run_root / "promoted_baseline" / "recons" / BASELINE_ROW_ID / "recon.npz"),
        },
    }
    for row_id in MANDATORY_FRESH_ROWS:
        _required_row_files(run_root, row_id)
        artifacts = {
            "config.json": _relative_to(artifact_root, run_root / "runs" / row_id / "config.json"),
            "history.json": _relative_to(artifact_root, run_root / "runs" / row_id / "history.json"),
            "metrics.json": _relative_to(artifact_root, run_root / "runs" / row_id / "metrics.json"),
            "model.pt": _relative_to(artifact_root, run_root / "runs" / row_id / "model.pt"),
            "invocation.json": _relative_to(artifact_root, run_root / "runs" / row_id / "invocation.json"),
            "randomness_contract.json": _relative_to(artifact_root, run_root / "runs" / row_id / "randomness_contract.json"),
            "recon_npz": _relative_to(artifact_root, run_root / "recons" / row_id / "recon.npz"),
        }
        stdout_log = run_root / "runs" / row_id / "stdout.log"
        stderr_log = run_root / "runs" / row_id / "stderr.log"
        completion_json = run_root / "runs" / row_id / "launcher_completion.json"
        training_root = run_root / "training_runs" / row_id
        if stdout_log.exists():
            artifacts["stdout_log"] = _relative_to(artifact_root, stdout_log)
        if stderr_log.exists():
            artifacts["stderr_log"] = _relative_to(artifact_root, stderr_log)
        if completion_json.exists():
            artifacts["launcher_completion_json"] = _relative_to(artifact_root, completion_json)
        if training_root.exists():
            artifacts["training_output_dir"] = _relative_to(artifact_root, training_root)
            artifacts["training_output_manifest.json"] = _relative_to(
                artifact_root,
                training_root / "training_output_manifest.json",
            )
            artifacts["checkpoint_dir"] = _relative_to(artifact_root, training_root / "checkpoints")
            artifacts["lightning_metrics_csv"] = _relative_to(
                artifact_root,
                training_root / "lightning_logs" / "version_0" / "metrics.csv",
            )
        rows[row_id] = {
            "row_status": ROW_SPECS[row_id]["row_status"],
            "model_label": ROW_SPECS[row_id]["model_label"],
            "changed_factor": ROW_SPECS[row_id]["changed_factor"],
            "artifacts": artifacts,
        }
    manifest = {
        "ablation_root": str(artifact_root.resolve()),
        "ablation_run_root": str(run_root.resolve()),
        "fixed_baseline_source_root": str(baseline_root.resolve()),
        "scalar_scope_decision": SCALAR_SCOPE_DECISION,
        "rows": rows,
    }
    return {
        "model_manifest_json": str(_write_json(artifact_root / "model_manifest.json", manifest)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    prepare_parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    prepare_parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    prepare_parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN_PATH)
    prepare_parser.add_argument("--run-id", type=str, default=RUN_ID)

    run_parser = subparsers.add_parser("run-row")
    run_parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    run_parser.add_argument("--row-id", required=True, choices=MANDATORY_FRESH_ROWS)

    repair_parser = subparsers.add_parser("repair")
    repair_parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)

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
            run_id=args.run_id,
        )
    elif args.command == "run-row":
        result = run_fresh_row(artifact_root=args.artifact_root, row_id=args.row_id)
    elif args.command == "repair":
        result = repair_existing_run_provenance(artifact_root=args.artifact_root)
    else:
        result = collate_ablation_bundle(
            artifact_root=args.artifact_root,
            baseline_root=args.baseline_root,
        )
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
