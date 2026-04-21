"""2D compressible Navier-Stokes runner for the PDEBench 128x128 image suite."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.studies.invocation_logging import capture_runtime_provenance, write_invocation_artifacts
from scripts.studies.pdebench_image128.data import (
    CFD_CNS_FIELD_ORDER,
    MultiFieldHistoryWindowDataset,
    inspect_cfd_cns_hdf5,
)
from scripts.studies.pdebench_image128.metrics import dynamic_state_metric_payload
from scripts.studies.pdebench_image128.models import ModelBuildBlocker, build_model_from_profile, describe_model
from scripts.studies.pdebench_image128.normalization import compute_multifield_dynamic_stats, denormalize_batch
from scripts.studies.pdebench_image128.physics_losses import (
    PhysicsRegularizationConfig,
    build_physics_regularizer,
)
from scripts.studies.pdebench_image128.reporting import build_comparison_summary, write_comparison_summary
from scripts.studies.pdebench_image128.run_config import (
    PRIMARY_CFD_CNS_PROFILE_IDS,
    READINESS_CFD_CNS_PROFILE_IDS,
    get_model_profile,
    parse_profile_ids,
)
from scripts.studies.pdebench_image128.splits import (
    build_trajectory_split,
    capped_trajectory_split,
    write_trajectory_split_manifest,
)


DEFAULT_OUTPUT_ROOT = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns")
SCRIPT_PATH = "scripts/studies/run_pdebench_image128_suite.py"
CFD_CNS_FILENAME = "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
CFD_CNS_TRAINING_LOSS = "mse"
CFD_CNS_TRAINING_LOSS_DEFINITION = "mean squared error on normalized CNS fields"
CFD_CNS_TRAINING_LOSS_RATIONALE = (
    "Matches the official PDEBench FNO and U-Net forward baseline training code, "
    "which uses nn.MSELoss(reduction='mean') for the compressible CFD benchmarks."
)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _build_cfd_cns_training_criterion() -> torch.nn.Module:
    return torch.nn.MSELoss()


def _parse_physics_loss_weights(spec: str | None) -> dict[str, float]:
    defaults = {
        "positivity_weight": 1.0,
        "continuity_weight": 1.0,
        "global_mass_weight": 1.0,
    }
    if spec is None:
        return defaults
    alias_map = {
        "pos": "positivity_weight",
        "positivity": "positivity_weight",
        "cont": "continuity_weight",
        "continuity": "continuity_weight",
        "mass": "global_mass_weight",
        "global_mass": "global_mass_weight",
    }
    payload = dict(defaults)
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"invalid physics loss weight token {token!r}; expected name=value")
        name, raw_value = token.split("=", 1)
        normalized_name = alias_map.get(name.strip().lower())
        if normalized_name is None:
            raise ValueError(f"unknown physics loss weight name {name!r}")
        payload[normalized_name] = float(raw_value)
    return payload


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
        "trajectory_id": [int(item["trajectory_id"]) for item in batch],
        "target_time_index": [int(item["target_time_index"]) for item in batch],
    }


def _profile_result_from_metrics(profile_id: str, metrics: dict[str, Any], model_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "status": "completed",
        "err_RMSE": metrics["err_RMSE"],
        "err_nRMSE": metrics["err_nRMSE"],
        "relative_l2": metrics["relative_l2"],
        "fRMSE_low": metrics["fRMSE_low"],
        "fRMSE_mid": metrics["fRMSE_mid"],
        "fRMSE_high": metrics["fRMSE_high"],
        "parameter_count": model_profile["parameter_count"],
    }


def _write_prediction_comparison(
    *,
    output_root: Path,
    profile_id: str,
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    state_stats: dict[str, Any],
    field_order: list[str],
) -> dict[str, str]:
    if not predictions or not targets:
        return {}

    prediction = denormalize_batch(predictions[0][:1].float(), state_stats)[0].numpy()
    target = denormalize_batch(targets[0][:1].float(), state_stats)[0].numpy()
    error = np.abs(prediction - target)
    npz_path = output_root / f"comparison_{profile_id}_sample0.npz"
    np.savez_compressed(npz_path, prediction=prediction, target=target, abs_error=error, field_order=np.array(field_order))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = int(prediction.shape[0])
    fig, axes = plt.subplots(channels, 3, figsize=(12, 3.0 * channels), squeeze=False, constrained_layout=True)
    for channel in range(channels):
        name = field_order[channel] if channel < len(field_order) else f"c{channel}"
        image_min = float(min(np.min(prediction[channel]), np.min(target[channel])))
        image_max = float(max(np.max(prediction[channel]), np.max(target[channel])))
        panels = [
            (f"Prediction {name}", prediction[channel], image_min, image_max),
            (f"Ground truth {name}", target[channel], image_min, image_max),
            (f"Abs error {name}", error[channel], 0.0, float(np.max(error[channel]))),
        ]
        for axis, (title, image, vmin, vmax) in zip(axes[channel], panels, strict=True):
            handle = axis.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    png_path = output_root / f"comparison_{profile_id}_sample0.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return {"comparison_png": str(png_path), "comparison_npz": str(npz_path)}


def _run_profile(
    *,
    profile_id: str,
    train_dataset: MultiFieldHistoryWindowDataset,
    eval_dataset: MultiFieldHistoryWindowDataset,
    state_stats: dict[str, Any],
    spatial_shape: tuple[int, int],
    output_root: Path,
    run_id: str,
    metadata: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
    num_workers: int,
    physics_config: PhysicsRegularizationConfig,
) -> dict[str, Any]:
    profile = get_model_profile(profile_id)
    started = time.time()
    device = _resolve_device(device_name)
    sample = train_dataset[0]
    try:
        model = build_model_from_profile(
            profile,
            in_channels=int(sample["input"].shape[0]),
            out_channels=int(sample["target"].shape[0]),
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = _build_cfd_cns_training_criterion()
    physics_regularizer = build_physics_regularizer(
        task_id="2d_cfd_cns",
        metadata=metadata,
        state_stats=state_stats,
        config=physics_config,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        threshold=0.0,
    )

    model.train()
    train_batches = 0
    epoch_losses: list[float] = []
    supervised_epoch_losses: list[float] = []
    physics_epoch_history: list[dict[str, Any]] = []
    for epoch_index in range(int(epochs)):
        epoch_loss = 0.0
        epoch_supervised = 0.0
        epoch_physics_total = 0.0
        epoch_batches = 0
        epoch_term_totals: defaultdict[str, float] = defaultdict(float)
        epoch_weighted_term_totals: defaultdict[str, float] = defaultdict(float)
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            supervised_loss = criterion(pred, y)
            physics_result = physics_regularizer.compute(x_norm=x, pred_norm=pred, target_norm=y)
            loss = supervised_loss + physics_result.total
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_supervised += float(supervised_loss.detach().cpu().item())
            epoch_physics_total += float(physics_result.total.detach().cpu().item())
            for name, value in physics_result.terms.items():
                epoch_term_totals[name] += float(value.detach().cpu().item())
            for name, value in physics_result.weighted_terms.items():
                epoch_weighted_term_totals[name] += float(value.detach().cpu().item())
            epoch_batches += 1
            train_batches += 1
        if epoch_batches:
            mean_epoch_loss = epoch_loss / epoch_batches
            epoch_losses.append(mean_epoch_loss)
            mean_supervised_loss = epoch_supervised / epoch_batches
            supervised_epoch_losses.append(mean_supervised_loss)
            mean_physics_total = epoch_physics_total / epoch_batches
            epoch_physics_payload = {
                "total": mean_physics_total,
                "terms": {name: value / epoch_batches for name, value in sorted(epoch_term_totals.items())},
                "weighted_terms": {
                    name: value / epoch_batches for name, value in sorted(epoch_weighted_term_totals.items())
                },
            }
            physics_epoch_history.append(epoch_physics_payload)
            print(
                f"EPOCH_LOSS profile={profile_id} epoch={epoch_index + 1} "
                f"loss={mean_epoch_loss:.10g} loss_name={CFD_CNS_TRAINING_LOSS} "
                f"supervised={mean_supervised_loss:.10g} physics_total={mean_physics_total:.10g} "
                f"physics_terms={json.dumps(epoch_physics_payload['terms'], sort_keys=True)}",
                flush=True,
            )
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
    metrics = dynamic_state_metric_payload(predictions, targets, normalized=True, state_stats=state_stats)
    comparison_artifacts = _write_prediction_comparison(
        output_root=output_root,
        profile_id=profile_id,
        predictions=predictions,
        targets=targets,
        state_stats=state_stats,
        field_order=list(train_dataset.field_order),
    )
    payload = {
        **metrics,
        "run_id": run_id,
        "profile_id": profile_id,
        "train_batches": int(train_batches),
        "train_epoch_losses": epoch_losses,
        "train_supervised_epoch_losses": supervised_epoch_losses,
        "training_loss": CFD_CNS_TRAINING_LOSS,
        "training_loss_definition": CFD_CNS_TRAINING_LOSS_DEFINITION,
        "training_loss_rationale": CFD_CNS_TRAINING_LOSS_RATIONALE,
        "physics_regularization_enabled": bool(physics_config.enabled),
        "physics_loss_terms": physics_config.active_terms(),
        "physics_loss_weights": physics_config.to_payload()["weights"],
        "physics_last_epoch": physics_epoch_history[-1] if physics_epoch_history else {"total": 0.0, "terms": {}, "weighted_terms": {}},
        "physics_epoch_history": physics_epoch_history,
        "scheduler": "ReduceLROnPlateau",
        "runtime_sec": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "model_profile": model_profile,
        **comparison_artifacts,
    }
    _write_json(output_root / f"metrics_{profile_id}.json", payload)
    return _profile_result_from_metrics(profile_id, payload, model_profile)


def _write_dataset_manifest(output_root: Path, *, metadata: dict[str, Any]) -> Path:
    payload = {
        "schema_version": "pdebench_image128_cfd_cns_dataset_manifest_v1",
        "task_id": "2d_cfd_cns",
        "pde_name": "2d_cfd",
        "dataset_source": "PDEBench",
        "dataset_source_url": "https://github.com/pdebench/PDEBench",
        "dataset_darus_id": "164690",
        "expected_md5": "21969082d0e9524bcc4708e216148e60",
        "data_file": metadata["data_file"],
        "file_size_bytes": metadata["file_size_bytes"],
        "field_order": metadata["field_order"],
        "field_axis_order": metadata["field_axis_order"],
        "field_shapes": metadata["field_shapes"],
        "dx": metadata["dx"],
        "dy": metadata["dy"],
        "dt": metadata["dt"],
        "eta": metadata["eta"],
        "zeta": metadata["zeta"],
        "boundary_condition": metadata["boundary_condition"],
        "history_len": metadata["history_len"],
        "sample_contract": metadata["dynamic_history_contract"],
    }
    return _write_json(output_root / "dataset_manifest.json", payload)


def _split_for_run(metadata: dict[str, Any]) -> dict[str, Any]:
    return build_trajectory_split(int(metadata["trajectory_count"]), seed=20260420)


def run_cfd_cns(
    *,
    task_id: str,
    mode: str,
    data_root: Path,
    output_root: Path,
    profile_ids: list[str] | None = None,
    history_len: int = 2,
    epochs: int = 1,
    batch_size: int = 2,
    max_train_trajectories: int | None = None,
    max_val_trajectories: int | None = None,
    max_test_trajectories: int | None = None,
    max_windows_per_trajectory: int | None = None,
    device: str = "cuda",
    num_workers: int = 0,
    allow_existing_output_root: bool = False,
    physics_config: PhysicsRegularizationConfig | None = None,
    raw_argv: list[str] | None = None,
) -> int:
    if task_id != "2d_cfd_cns":
        raise ValueError("run_cfd_cns only supports task_id='2d_cfd_cns'")
    if mode not in {"inspect", "readiness", "benchmark"}:
        raise ValueError("2d_cfd_cns mode must be inspect, readiness, or benchmark")
    output_root = Path(output_root)
    _guard_output_root(output_root, allow_existing=allow_existing_output_root)
    physics_config = physics_config or PhysicsRegularizationConfig()
    data_file = Path(data_root) / "2d_cfd_cns" / CFD_CNS_FILENAME
    if not data_file.exists():
        raise FileNotFoundError(f"missing 2D CFD CNS data file: {data_file}")
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
            "history_len": history_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_train_trajectories": max_train_trajectories,
            "max_val_trajectories": max_val_trajectories,
            "max_test_trajectories": max_test_trajectories,
            "max_windows_per_trajectory": max_windows_per_trajectory,
            "device": device,
            "num_workers": num_workers,
            "physics_regularization_enabled": bool(physics_config.enabled),
            "physics_loss_terms": physics_config.active_terms(),
            "physics_loss_weights": physics_config.to_payload()["weights"],
        },
        extra={"run_id": run_id, "runtime_provenance": capture_runtime_provenance(), "pid": os.getpid()},
    )
    metadata = inspect_cfd_cns_hdf5(data_file, history_len=history_len)
    _write_json(output_root / "hdf5_metadata.json", metadata)
    _write_dataset_manifest(output_root, metadata=metadata)
    if mode == "inspect":
        return 0

    full_split = _split_for_run(metadata)
    run_split = capped_trajectory_split(
        full_split,
        max_train_trajectories=max_train_trajectories,
        max_val_trajectories=max_val_trajectories,
        max_test_trajectories=max_test_trajectories,
    )
    write_trajectory_split_manifest(
        output_root=output_root,
        data_file=data_file,
        split=run_split,
        state_dataset=",".join(metadata["field_order"]),
        axis_order=metadata["field_axis_order"],
        shape=metadata["state_shape"],
        history_len=int(history_len),
        max_windows_per_trajectory=max_windows_per_trajectory,
        extra={"full_split_counts": {name: len(full_split[name]) for name in ("train", "val", "test")}, "run_mode": mode},
    )
    state_stats = compute_multifield_dynamic_stats(
        data_file=data_file,
        field_order=metadata["field_order"],
        axis_order=metadata["field_axis_order"],
        train_trajectory_ids=run_split["train"],
    )
    state_stats = {**state_stats, "run_id": run_id, "history_len": int(history_len)}
    _write_json(output_root / "normalization_stats_state.json", state_stats)

    profile_ids = profile_ids or (PRIMARY_CFD_CNS_PROFILE_IDS if mode == "benchmark" else READINESS_CFD_CNS_PROFILE_IDS)
    if mode == "benchmark" and "unet_tiny_smoke" in profile_ids:
        raise ValueError("unet_tiny_smoke is readiness-only and cannot be used for benchmark mode")

    train_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=run_split["train"],
        axis_order=metadata["field_axis_order"],
        history_len=int(history_len),
        normalization=state_stats,
        max_windows_per_trajectory=max_windows_per_trajectory,
    )
    eval_ids = run_split["test"] or run_split["val"] or run_split["train"]
    eval_dataset = MultiFieldHistoryWindowDataset(
        data_file=data_file,
        field_order=metadata["field_order"],
        trajectory_ids=eval_ids,
        axis_order=metadata["field_axis_order"],
        history_len=int(history_len),
        normalization=state_stats,
        max_windows_per_trajectory=max_windows_per_trajectory,
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
            state_stats=state_stats,
            spatial_shape=spatial_shape,
            output_root=output_root,
            run_id=run_id,
            metadata=metadata,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=2e-4,
            device_name=device,
            num_workers=num_workers,
            physics_config=physics_config,
        )
        profile_results.append(result)
        if result.get("status") != "completed":
            blockers.append(result)
    summary = build_comparison_summary(
        task_id="2d_cfd_cns",
        mode=mode,
        output_root=output_root,
        profile_results=profile_results,
        run_id=run_id,
        blockers=blockers,
    )
    summary["history_len"] = int(history_len)
    summary["dynamic_history_contract"] = metadata["dynamic_history_contract"]
    summary["field_order"] = metadata["field_order"]
    write_comparison_summary(summary, output_root)
    return 0 if any(item.get("status") == "completed" for item in profile_results) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="2d_cfd_cns")
    parser.add_argument("--mode", choices=["inspect", "readiness", "benchmark"], default="readiness")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--history-len", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-trajectories", type=int, default=None)
    parser.add_argument("--max-val-trajectories", type=int, default=None)
    parser.add_argument("--max-test-trajectories", type=int, default=None)
    parser.add_argument("--max-windows-per-trajectory", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--physics-regularization", choices=["off", "on"], default="off")
    parser.add_argument("--physics-loss-weights", default=None)
    parser.add_argument("--allow-existing-output-root", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(argv) if argv is not None else None
    args = parse_args(raw)
    if args.physics_regularization == "on":
        weights = _parse_physics_loss_weights(args.physics_loss_weights)
    else:
        weights = {
            "positivity_weight": 0.0,
            "continuity_weight": 0.0,
            "global_mass_weight": 0.0,
        }
    try:
        return run_cfd_cns(
            task_id=args.task,
            mode=args.mode,
            data_root=args.data_root,
            output_root=args.output_root,
            profile_ids=parse_profile_ids(args.profiles) if args.profiles else None,
            history_len=args.history_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_train_trajectories=args.max_train_trajectories,
            max_val_trajectories=args.max_val_trajectories,
            max_test_trajectories=args.max_test_trajectories,
            max_windows_per_trajectory=args.max_windows_per_trajectory,
            device=args.device,
            num_workers=args.num_workers,
            allow_existing_output_root=args.allow_existing_output_root,
            physics_config=PhysicsRegularizationConfig(
                enabled=args.physics_regularization == "on",
                positivity_weight=weights["positivity_weight"],
                continuity_weight=weights["continuity_weight"],
                global_mass_weight=weights["global_mass_weight"],
            ),
            raw_argv=raw or [],
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc))
        return 2
