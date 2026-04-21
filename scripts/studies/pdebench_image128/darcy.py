"""Darcy static-operator runner for the PDEBench 128x128 image suite."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.studies.invocation_logging import capture_runtime_provenance, write_invocation_artifacts
from scripts.studies.pdebench_image128.data import DarcyStaticOperatorDataset, inspect_darcy_hdf5
from scripts.studies.pdebench_image128.metrics import static_operator_metric_payload
from scripts.studies.pdebench_image128.models import ModelBuildBlocker, build_model_from_profile, describe_model
from scripts.studies.pdebench_image128.normalization import compute_static_operator_stats, denormalize_batch
from scripts.studies.pdebench_image128.reporting import (
    build_comparison_summary,
    write_comparison_summary,
    write_literature_context,
)
from scripts.studies.pdebench_image128.run_config import (
    PRIMARY_DARCY_PROFILE_IDS,
    get_model_profile,
    parse_profile_ids,
    validate_darcy_run_budget,
)
from scripts.studies.pdebench_image128.splits import build_sample_split, capped_split, write_sample_split_manifest


DEFAULT_OUTPUT_ROOT = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark"
)
SCRIPT_PATH = "scripts/studies/run_pdebench_image128_suite.py"
DARCY_FILENAME = "2D_DarcyFlow_beta1.0_Train.hdf5"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _guard_output_root(output_root: Path, *, allow_existing: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing:
        raise FileExistsError(f"refusing to write into non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "sample_index": [int(item["sample_index"]) for item in batch],
    }


def _profile_result_from_metrics(profile_id: str, metrics: dict[str, Any], model_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "status": "completed",
        "err_RMSE": metrics["err_RMSE"],
        "err_nRMSE": metrics["err_nRMSE"],
        "relative_l2": metrics["relative_l2"],
        "parameter_count": model_profile["parameter_count"],
    }


def _write_prediction_comparison(
    *,
    output_root: Path,
    profile_id: str,
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    target_stats: dict[str, Any],
) -> dict[str, str]:
    """Write a first-sample prediction/target/error visual for quick qualitative checks."""
    if not predictions or not targets:
        return {}

    prediction = denormalize_batch(predictions[0][:1].float(), target_stats)[0, 0].numpy()
    target = denormalize_batch(targets[0][:1].float(), target_stats)[0, 0].numpy()
    error = np.abs(prediction - target)

    npz_path = output_root / f"comparison_{profile_id}_sample0.npz"
    np.savez_compressed(npz_path, prediction=prediction, target=target, abs_error=error)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_min = float(min(np.min(prediction), np.min(target)))
    image_max = float(max(np.max(prediction), np.max(target)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        ("Prediction", prediction, image_min, image_max),
        ("Ground truth", target, image_min, image_max),
        ("Abs error", error, 0.0, float(np.max(error))),
    ]
    for axis, (title, image, vmin, vmax) in zip(axes, panels, strict=True):
        handle = axis.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    png_path = output_root / f"comparison_{profile_id}_sample0.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return {"comparison_png": str(png_path), "comparison_npz": str(npz_path)}


def _relative_l2_sample_mean_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, prediction.ndim))
    numerator = torch.sum((prediction - target).square(), dim=reduce_dims)
    denominator = torch.clamp(torch.sum(target.square(), dim=reduce_dims), min=1e-12)
    return torch.mean(torch.sqrt(numerator / denominator))


def _run_profile(
    *,
    profile_id: str,
    train_dataset: DarcyStaticOperatorDataset,
    eval_dataset: DarcyStaticOperatorDataset,
    target_stats: dict[str, Any],
    spatial_shape: tuple[int, int],
    output_root: Path,
    run_id: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loss_name: str,
    scheduler_name: str,
    plateau_factor: float,
    plateau_patience: int,
    plateau_min_lr: float,
    plateau_threshold: float,
    device_name: str,
    num_workers: int,
) -> dict[str, Any]:
    profile = get_model_profile(profile_id)
    started = time.time()
    device = _resolve_device(device_name)
    sample = train_dataset[0]
    in_channels = int(sample["input"].shape[0])
    out_channels = int(sample["target"].shape[0])
    try:
        model = build_model_from_profile(
            profile,
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_shape=spatial_shape,
        ).to(device)
    except ModelBuildBlocker as exc:
        blocker = {
            **exc.to_payload(run_id=run_id),
            "profile_id": profile_id,
            "status": "blocked",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(output_root / f"model_profile_{profile_id}.json", {**profile.to_model_config(), **blocker})
        _write_json(output_root / f"metrics_{profile_id}.json", blocker)
        return {"profile_id": profile_id, "status": "blocked", "blocker_reason": exc.reason}

    model_profile = describe_model(model, profile=profile)
    _write_json(output_root / f"model_profile_{profile_id}.json", model_profile)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    if loss_name == "mae":
        criterion = torch.nn.L1Loss()
    elif loss_name == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_name == "relative_l2":
        def criterion(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return _relative_l2_sample_mean_loss(prediction, target)
    else:
        raise ValueError(f"unsupported loss: {loss_name}")
    scheduler = None
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(plateau_factor),
            patience=int(plateau_patience),
            min_lr=float(plateau_min_lr),
            threshold=float(plateau_threshold),
        )
    model.train()
    train_batches = 0
    epoch_losses: list[float] = []
    for epoch_index in range(int(epochs)):
        epoch_loss = 0.0
        epoch_batches = 0
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_batches += 1
            train_batches += 1
        if epoch_batches:
            mean_epoch_loss = epoch_loss / epoch_batches
            epoch_losses.append(mean_epoch_loss)
            print(
                f"EPOCH_LOSS profile={profile_id} epoch={epoch_index + 1} "
                f"loss={mean_epoch_loss:.10g} loss_name={loss_name}",
                flush=True,
            )
            if scheduler is not None:
                scheduler.step(mean_epoch_loss)

    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            predictions.append(model(x).cpu())
            targets.append(y.cpu())
    metrics = static_operator_metric_payload(predictions, targets, normalized=True, target_stats=target_stats)
    comparison_artifacts = _write_prediction_comparison(
        output_root=output_root,
        profile_id=profile_id,
        predictions=predictions,
        targets=targets,
        target_stats=target_stats,
    )
    runtime_sec = time.time() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    payload = {
        **metrics,
        "run_id": run_id,
        "profile_id": profile_id,
        "train_batches": int(train_batches),
        "train_epoch_losses": epoch_losses,
        "training_loss": loss_name,
        "training_loss_definition": "mean over samples of ||prediction-target||_2 / ||target||_2",
        "scheduler": scheduler_name,
        "runtime_sec": runtime_sec,
        "peak_cuda_memory_bytes": peak_memory,
        "model_profile": model_profile,
        **comparison_artifacts,
    }
    _write_json(output_root / f"metrics_{profile_id}.json", payload)
    return _profile_result_from_metrics(profile_id, payload, model_profile)


def _write_dataset_manifest(output_root: Path, *, metadata: dict[str, Any]) -> Path:
    payload = {
        "schema_version": "pdebench_image128_darcy_dataset_manifest_v1",
        "task_id": "darcy",
        "pde_name": "darcy",
        "dataset_source": "PDEBench",
        "dataset_source_url": "https://github.com/pdebench/PDEBench",
        "data_file": metadata["data_file"],
        "file_size_bytes": metadata["file_size_bytes"],
        "beta": metadata["beta"],
        "input_dataset": metadata["input_dataset"],
        "target_dataset": metadata["target_dataset"],
        "sample_contract": "nu[i] -> tensor[i]",
        "no_time_axis": True,
    }
    return _write_json(output_root / "dataset_manifest.json", payload)


def _split_for_run(metadata: dict[str, Any], *, mode: str) -> dict[str, Any]:
    sample_count = int(metadata["sample_count"])
    if sample_count == 10000:
        return build_sample_split(sample_count, seed=20260420, counts=(8000, 1000, 1000))
    return build_sample_split(sample_count, seed=20260420)


def run_darcy(
    *,
    task_id: str,
    mode: str,
    data_root: Path,
    output_root: Path,
    profile_ids: list[str] | None = None,
    epochs: int = 1,
    batch_size: int = 2,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    device: str = "cuda",
    num_workers: int = 0,
    run_budget: Path | None = None,
    allow_existing_output_root: bool = False,
    raw_argv: list[str] | None = None,
) -> int:
    if task_id != "darcy":
        raise ValueError("run_darcy only supports task_id='darcy'")
    if mode not in {"inspect", "readiness", "benchmark"}:
        raise ValueError("Darcy mode must be inspect, readiness, or benchmark")
    output_root = Path(output_root)
    _guard_output_root(output_root, allow_existing=allow_existing_output_root)
    data_file = Path(data_root) / "darcy" / DARCY_FILENAME
    if not data_file.exists():
        raise FileNotFoundError(f"missing Darcy data file: {data_file}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    write_invocation_artifacts(
        output_dir=output_root,
        script_path=SCRIPT_PATH,
        argv=raw_argv or [],
        parsed_args={
            "task_id": task_id,
            "mode": mode,
            "data_root": str(data_root),
            "output_root": str(output_root),
            "profile_ids": profile_ids,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "max_test_samples": max_test_samples,
            "device": device,
            "num_workers": num_workers,
            "run_budget": str(run_budget) if run_budget else None,
        },
        extra={"run_id": run_id, "runtime_provenance": capture_runtime_provenance(), "pid": os.getpid()},
    )
    metadata = inspect_darcy_hdf5(data_file)
    _write_json(output_root / "hdf5_metadata.json", metadata)
    _write_dataset_manifest(output_root, metadata=metadata)
    write_literature_context(output_root, task_id="darcy")
    if mode == "inspect":
        return 0

    full_split = _split_for_run(metadata, mode=mode)
    if mode == "benchmark":
        budget_payload = json.loads(Path(run_budget).read_text(encoding="utf-8")) if run_budget else {
            "task_id": "darcy",
            "mode": "benchmark",
            "train_count": 8000,
            "val_count": 1000,
            "test_count": 1000,
            "primary_profiles": PRIMARY_DARCY_PROFILE_IDS,
            "training_seed": 20260420,
            "loss": "relative_l2",
            "optimizer": "adam",
            "learning_rate": 2e-4,
            "scheduler": "ReduceLROnPlateau",
            "plateau_factor": 0.5,
            "plateau_patience": 2,
            "plateau_min_lr": 1e-5,
            "plateau_threshold": 0.0,
            "batch_size": batch_size,
            "epochs": epochs,
            "precision": "float32",
            "device": device,
            "num_workers": num_workers,
        }
        budget = validate_darcy_run_budget(budget_payload)
        profile_ids = list(budget["primary_profiles"])
        epochs = int(budget["epochs"])
        batch_size = int(budget["batch_size"])
        device = str(budget["device"])
        num_workers = int(budget["num_workers"])
        learning_rate = float(budget["learning_rate"])
        loss_name = str(budget["loss"])
        scheduler_name = str(budget["scheduler"])
        plateau_factor = float(budget["plateau_factor"])
        plateau_patience = int(budget["plateau_patience"])
        plateau_min_lr = float(budget["plateau_min_lr"])
        plateau_threshold = float(budget["plateau_threshold"])
        run_split = full_split
    else:
        learning_rate = 2e-4
        loss_name = "relative_l2"
        scheduler_name = "ReduceLROnPlateau"
        plateau_factor = 0.5
        plateau_patience = 2
        plateau_min_lr = 1e-5
        plateau_threshold = 0.0
        profile_ids = profile_ids or ["unet_tiny_smoke"]
        run_split = capped_split(
            full_split,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )

    write_sample_split_manifest(
        output_root=output_root,
        data_file=data_file,
        split=run_split,
        beta=metadata["beta"],
        input_dataset=metadata["input_dataset"],
        target_dataset=metadata["target_dataset"],
        extra={
            "full_split_counts": {name: len(full_split[name]) for name in ("train", "val", "test")},
            "run_mode": mode,
        },
    )
    input_stats, target_stats = compute_static_operator_stats(
        data_file=data_file,
        input_dataset=metadata["input_dataset"],
        target_dataset=metadata["target_dataset"],
        train_indices=run_split["train"],
    )
    input_stats = {**input_stats, "run_id": run_id}
    target_stats = {**target_stats, "run_id": run_id}
    _write_json(output_root / "normalization_stats_input.json", input_stats)
    _write_json(output_root / "normalization_stats_target.json", target_stats)

    train_dataset = DarcyStaticOperatorDataset(
        data_file=data_file,
        sample_indices=run_split["train"],
        input_stats=input_stats,
        target_stats=target_stats,
    )
    eval_indices = run_split["test"] or run_split["val"] or run_split["train"]
    eval_dataset = DarcyStaticOperatorDataset(
        data_file=data_file,
        sample_indices=eval_indices,
        input_stats=input_stats,
        target_stats=target_stats,
    )
    sample = train_dataset[0]
    spatial_shape = (int(sample["input"].shape[-2]), int(sample["input"].shape[-1]))
    torch.manual_seed(20260420)
    profile_results = []
    blockers = []
    for profile_id in profile_ids:
        result = _run_profile(
            profile_id=profile_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            target_stats=target_stats,
            spatial_shape=spatial_shape,
            output_root=output_root,
            run_id=run_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_name=loss_name,
            scheduler_name=scheduler_name,
            plateau_factor=plateau_factor,
            plateau_patience=plateau_patience,
            plateau_min_lr=plateau_min_lr,
            plateau_threshold=plateau_threshold,
            device_name=device,
            num_workers=num_workers,
        )
        profile_results.append(result)
        if result.get("status") != "completed":
            blockers.append(result)
    summary = build_comparison_summary(
        task_id="darcy",
        mode=mode,
        output_root=output_root,
        profile_results=profile_results,
        run_id=run_id,
        blockers=blockers,
    )
    write_comparison_summary(summary, output_root)
    return 0 if any(item.get("status") == "completed" for item in profile_results) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="darcy")
    parser.add_argument("--mode", choices=["inspect", "readiness", "benchmark"], default="readiness")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-budget", type=Path, default=None)
    parser.add_argument("--allow-existing-output-root", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(argv) if argv is not None else None
    args = parse_args(raw)
    try:
        return run_darcy(
            task_id=args.task,
            mode=args.mode,
            data_root=args.data_root,
            output_root=args.output_root,
            profile_ids=parse_profile_ids(args.profiles) if args.profiles else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            device=args.device,
            num_workers=args.num_workers,
            run_budget=args.run_budget,
            allow_existing_output_root=args.allow_existing_output_root,
            raw_argv=raw or [],
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 2
