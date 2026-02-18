#!/usr/bin/env python3
"""NERSC scan807 + cameraman orchestration helpers."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Tuple

import numpy as np

from ptycho.workflows.grid_lines_workflow import GridLinesConfig, save_recon_artifact
from scripts.studies.grid_lines_compare_wrapper import (
    _finalize_compare_outputs,
    evaluate_selected_models,
)
from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig, run_grid_lines_torch
from scripts.studies.grid_study_dataset_builder import build_datasets
from scripts.studies.hybrid_checkpoint_inference import run_cross_dataset_hybrid_inference
from scripts.studies.nersc_pair_adapter import materialize_pair_working_copy, pair_to_external_npz
from scripts.studies.prepare_nersc_hybrid_dataset import (
    DOWNSAMPLE_POLICY_CHOICES,
    _downsample_external_payload,
    prepare_hybrid_dataset,
)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _run_and_capture(cmd: list[str], logs_dir: Path) -> subprocess.CompletedProcess[str]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    (logs_dir / "stdout.log").write_text(completed.stdout)
    (logs_dir / "stderr.log").write_text(completed.stderr)
    return completed


def run_ptychovit_inference_stage(
    *,
    dataset_pairs: Dict[str, Tuple[Path, Path]],
    ptychovit_repo: Path,
    checkpoint: Path,
    output_dir: Path,
    python_bin: str = "python",
) -> Dict[str, Dict[str, str]]:
    """Run checkpoint-restored PtychoViT inference per dataset pair."""
    output_dir = Path(output_dir)
    outputs: Dict[str, Dict[str, str]] = {}
    for dataset_name, (dp_h5, para_h5) in dataset_pairs.items():
        dataset_root = output_dir / dataset_name
        run_dir = dataset_root / "runs" / "pinn_ptychovit"
        recon_npz = dataset_root / "recons" / "pinn_ptychovit" / "recon.npz"
        cmd = [
            python_bin,
            "scripts/studies/ptychovit_bridge_entrypoint.py",
            "--ptychovit-repo",
            str(ptychovit_repo),
            "--train-dp",
            str(dp_h5),
            "--train-para",
            str(para_h5),
            "--test-dp",
            str(dp_h5),
            "--test-para",
            str(para_h5),
            "--mode",
            "inference",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(run_dir),
            "--recon-npz",
            str(recon_npz),
        ]
        completed = _run_and_capture(cmd, run_dir)
        if completed.returncode != 0:
            raise RuntimeError(
                f"PtychoViT inference failed for {dataset_name} (exit={completed.returncode})"
            )
        if not recon_npz.exists():
            raise RuntimeError(f"PtychoViT recon missing for {dataset_name}: {recon_npz}")
        invocation_json = run_dir / "invocation.json"
        invocation_sh = run_dir / "invocation.sh"
        if not invocation_json.exists() or not invocation_sh.exists():
            raise RuntimeError(
                f"PtychoViT child invocation artifacts missing for {dataset_name}: "
                f"{invocation_json}, {invocation_sh}"
            )
        outputs[dataset_name] = {
            "run_dir": str(run_dir),
            "recon_npz": str(recon_npz),
            "stdout_log": str(run_dir / "stdout.log"),
            "stderr_log": str(run_dir / "stderr.log"),
            "invocation_json": str(invocation_json),
            "invocation_sh": str(invocation_sh),
        }
    return outputs


def _convert_pair_to_downsampled_external_npz(
    *,
    dp_h5: Path,
    para_h5: Path,
    out_npz: Path,
    work_dir: Path,
    target_n: int = 128,
    downsample_policy: str = "bin-crop",
) -> Path:
    work_dp, work_para = materialize_pair_working_copy(dp_h5, para_h5, work_dir)
    canonical = pair_to_external_npz(work_dp, work_para, out_npz.with_name(f"{out_npz.stem}_canonical.npz"))
    with np.load(canonical, allow_pickle=True) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    downsampled = _downsample_external_payload(
        payload,
        target_n=target_n,
        downsample_policy=downsample_policy,
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **downsampled)
    return out_npz


def _write_gt_recon_from_external_npz(dataset_output_dir: Path, external_npz: Path) -> Path:
    with np.load(external_npz, allow_pickle=True) as loaded:
        if "objectGuess" not in loaded.files:
            raise KeyError(f"External NPZ missing objectGuess for GT recon: {external_npz}")
        object_guess = np.asarray(loaded["objectGuess"], dtype=np.complex64)
    gt = np.squeeze(object_guess).astype(np.complex64)
    return save_recon_artifact(dataset_output_dir, "gt", gt)


def _build_cached_external_bundle(
    *,
    output_dir: Path,
    train_external_npz: Path,
    test_external_npz: Path,
    seed: int,
) -> Dict[str, str]:
    cfg = GridLinesConfig(
        N=128,
        gridsize=1,
        output_dir=Path(output_dir),
        probe_npz=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
        nimgs_train=1,
        nimgs_test=1,
        nphotons=1e9,
        nepochs=1,
        batch_size=8,
        probe_source="custom",
        probe_scale_mode="pad_extrapolate",
    )
    bundles = build_datasets(
        dataset_source="external_raw_npz",
        cfg=cfg,
        required_ns=[128],
        train_data=Path(train_external_npz),
        test_data=Path(test_external_npz),
        n_groups=None,
        n_subsample=None,
        neighbor_count=7,
        subsample_seed=seed,
    )
    return bundles[128]


def aggregate_metrics_visuals_stage(
    *,
    dataset_output_dir: Path,
    recon_paths: Dict[str, Path],
    gt_recon_path: Path,
    model_ns: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """Run object-space metrics and render comparison visuals/tables for one dataset."""
    dataset_output_dir = Path(dataset_output_dir)
    metrics_by_model = evaluate_selected_models(recon_paths, Path(gt_recon_path))
    metrics_by_model_path = dataset_output_dir / "metrics_by_model.json"
    metrics_by_model_path.write_text(json.dumps(metrics_by_model, indent=2, default=_json_default))

    legacy_metrics = {model_id: payload["metrics"] for model_id, payload in metrics_by_model.items()}
    finalize_out = _finalize_compare_outputs(
        output_dir=dataset_output_dir,
        merged_metrics=legacy_metrics,
        visual_order=tuple(["gt", *recon_paths.keys()]),
        model_ns=model_ns,
    )
    return {
        "metrics_by_model_path": str(metrics_by_model_path),
        "legacy_metrics": legacy_metrics,
        "finalize_outputs": finalize_out,
    }


def run_nersc_scan807_cameraman_study(
    *,
    scan807_dp: Path,
    scan807_para: Path,
    cameraman_dp: Path,
    cameraman_para: Path,
    ptychovit_checkpoint: Path,
    output_dir: Path,
    half: str = "top",
    seed: int = 3,
    ptychovit_repo: Path = Path("/home/ollie/Documents/ptycho-vit"),
    downsample_policy: str = "bin-crop",
    position_reassembly_backend: str = "shift_sum",
) -> Dict[str, Any]:
    """Execute the full scan807 + cameraman orchestration with staged artifacts."""
    if downsample_policy not in DOWNSAMPLE_POLICY_CHOICES:
        raise ValueError(
            f"Unsupported downsample_policy='{downsample_policy}', "
            f"expected one of {DOWNSAMPLE_POLICY_CHOICES}."
        )
    if position_reassembly_backend != "shift_sum":
        raise ValueError(
            "NERSC orchestration requires position_reassembly_backend='shift_sum' "
            "to avoid unsafe auto/batched fallback paths."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    working_pairs = {
        "scan807": materialize_pair_working_copy(
            scan807_dp, scan807_para, output_dir / "scan807" / "working_pair"
        ),
        "cameraman256": materialize_pair_working_copy(
            cameraman_dp, cameraman_para, output_dir / "cameraman256" / "working_pair"
        ),
    }

    ptychovit_outputs = run_ptychovit_inference_stage(
        dataset_pairs=working_pairs,
        ptychovit_repo=Path(ptychovit_repo),
        checkpoint=Path(ptychovit_checkpoint),
        output_dir=output_dir,
    )

    cameraman_prep = prepare_hybrid_dataset(
        dp_h5=working_pairs["cameraman256"][0],
        para_h5=working_pairs["cameraman256"][1],
        output_dir=output_dir / "cameraman256" / "hybrid_dataset",
        half=half,
        target_n=128,
        downsample_policy=downsample_policy,
    )
    scan807_test_npz = _convert_pair_to_downsampled_external_npz(
        dp_h5=working_pairs["scan807"][0],
        para_h5=working_pairs["scan807"][1],
        out_npz=output_dir / "scan807" / "hybrid_dataset" / "scan807_n128_full_test.npz",
        work_dir=output_dir / "scan807" / "hybrid_dataset" / "working_pair",
        target_n=128,
        downsample_policy=downsample_policy,
    )

    training_cfg = GridLinesConfig(
        N=128,
        gridsize=1,
        output_dir=output_dir / "hybrid_training",
        probe_npz=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"),
        nimgs_train=1,
        nimgs_test=1,
        nphotons=1e9,
        nepochs=40,
        batch_size=8,
        probe_source="custom",
        probe_scale_mode="pad_extrapolate",
    )
    bundles = build_datasets(
        dataset_source="external_raw_npz",
        cfg=training_cfg,
        required_ns=[128],
        train_data=Path(cameraman_prep["train_npz"]),
        test_data=Path(cameraman_prep["test_npz"]),
        n_groups=None,
        n_subsample=None,
        neighbor_count=7,
        subsample_seed=seed,
    )
    train_npz = Path(bundles[128]["train_npz"])
    test_npz = Path(bundles[128]["test_npz"])

    common_cfg = TorchRunnerConfig(
        train_npz=train_npz,
        test_npz=test_npz,
        output_dir=output_dir / "hybrid_training",
        architecture="hybrid_resnet",
        seed=seed,
        epochs=40,
        batch_size=8,
        learning_rate=2e-4,
        infer_batch_size=128,
        generator_output_mode="real_imag",
        N=128,
        gridsize=1,
        probe_source="custom",
        torch_loss_mode="mae",
        scheduler="ReduceLROnPlateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_min_lr=5e-5,
        plateau_threshold=0.0,
        reassembly_mode="position",
        position_reassembly_backend=position_reassembly_backend,
    )
    train_result = run_grid_lines_torch(common_cfg)
    model_pt = Path(train_result["run_dir"]) / "model.pt"
    if not model_pt.exists():
        raise RuntimeError(f"Hybrid checkpoint missing after training: {model_pt}")

    scan807_cached_bundle = _build_cached_external_bundle(
        output_dir=output_dir / "scan807" / "hybrid_cached",
        train_external_npz=scan807_test_npz,
        test_external_npz=scan807_test_npz,
        seed=seed,
    )
    cameraman_cached_bundle = _build_cached_external_bundle(
        output_dir=output_dir / "cameraman256" / "hybrid_cached",
        train_external_npz=Path(cameraman_prep["train_npz"]),
        test_external_npz=Path(cameraman_prep["test_npz"]),
        seed=seed,
    )

    hybrid_results = run_cross_dataset_hybrid_inference(
        model_pt=model_pt,
        dataset_npzs={
            "scan807": Path(scan807_cached_bundle["test_npz"]),
            "cameraman256": Path(cameraman_cached_bundle["test_npz"]),
        },
        output_dir=output_dir,
        base_cfg=replace(common_cfg, epochs=1),
        allow_oom_fallback=False,
    )

    gt_paths = {
        "scan807": _write_gt_recon_from_external_npz(output_dir / "scan807", scan807_test_npz),
        "cameraman256": _write_gt_recon_from_external_npz(
            output_dir / "cameraman256", Path(cameraman_prep["downsampled_npz"])
        ),
    }
    metrics_outputs = {}
    for dataset_name in ("scan807", "cameraman256"):
        dataset_root = output_dir / dataset_name
        metrics_outputs[dataset_name] = aggregate_metrics_visuals_stage(
            dataset_output_dir=dataset_root,
            recon_paths={
                "pinn_ptychovit": Path(ptychovit_outputs[dataset_name]["recon_npz"]),
                "pinn_hybrid_resnet": Path(hybrid_results[dataset_name]["recon_npz"]),
            },
            gt_recon_path=gt_paths[dataset_name],
            model_ns={"pinn_ptychovit": 256, "pinn_hybrid_resnet": 128},
        )

    manifest = {
        "output_dir": str(output_dir),
        "half": half,
        "downsample_policy": downsample_policy,
        "position_reassembly_backend": position_reassembly_backend,
        "seed": int(seed),
        "working_pairs": {k: [str(v[0]), str(v[1])] for k, v in working_pairs.items()},
        "ptychovit_outputs": ptychovit_outputs,
        "hybrid_training": {
            "train_npz": str(train_npz),
            "test_npz": str(test_npz),
            "model_pt": str(model_pt),
            "run_dir": str(train_result["run_dir"]),
        },
        "hybrid_cross_dataset_outputs": hybrid_results,
        "hybrid_cached_bundles": {
            "scan807": scan807_cached_bundle,
            "cameraman256": cameraman_cached_bundle,
        },
        "gt_paths": {k: str(v) for k, v in gt_paths.items()},
        "metrics_outputs": metrics_outputs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    return manifest
