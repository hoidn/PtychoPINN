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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from scripts.studies.invocation_logging import capture_runtime_provenance, write_invocation_artifacts
from scripts.studies.pdebench_image128.data import (
    CFD_CNS_FIELD_ORDER,
    MultiFieldHistoryWindowDataset,
    inspect_cfd_cns_hdf5,
)
from scripts.studies.pdebench_image128.distributed import DistributedRuntime, initialize_runtime, prepare_output_root
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
from scripts.studies.pdebench_image128.visualization import cfd_cns_field_visual_spec


DEFAULT_OUTPUT_ROOT = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns")
SCRIPT_PATH = "scripts/studies/run_pdebench_image128_suite.py"
CFD_CNS_FILENAME = "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
CFD_CNS_DEFAULT_TRAINING_LOSS = "mse"
CFD_CNS_DEFAULT_TRAINING_LOSS_DEFINITION = "mean squared error on normalized CNS fields"
CFD_CNS_DEFAULT_TRAINING_LOSS_RATIONALE = (
    "Matches the official PDEBench FNO and U-Net forward baseline training code, "
    "which uses nn.MSELoss(reduction='mean') for the compressible CFD benchmarks."
)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _build_cfd_cns_training_criterion() -> torch.nn.Module:
    return torch.nn.MSELoss()


def _relative_l2_sample_mean_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    prediction_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    numerator = torch.linalg.vector_norm(prediction_flat - target_flat, dim=1)
    denominator = torch.clamp(torch.linalg.vector_norm(target_flat, dim=1), min=float(eps))
    return (numerator / denominator).mean()


def _build_cfd_cns_training_recipe(
    *,
    profile,
    default_learning_rate: float,
    steps_per_epoch: int,
    epochs: int,
) -> dict[str, Any]:
    config = profile.to_model_config()
    if profile.base_model == "gnot_cns_net":
        return {
            "loss_name": str(config.get("training_loss", "relative_l2")),
            "loss_definition": "mean over batch of ||pred-target||_2 / ||target||_2 on normalized CNS fields",
            "loss_rationale": (
                "Matches the public GNOT recipe more closely than the local PDEBench fairness recipe: "
                "relative L2 objective, AdamW optimizer, and OneCycleLR schedule."
            ),
            "criterion": _relative_l2_sample_mean_loss,
            "optimizer_name": str(config.get("optimizer_name", "AdamW")),
            "learning_rate": float(config.get("learning_rate", 1e-3)),
            "weight_decay": float(config.get("weight_decay", 5e-5)),
            "scheduler_name": str(config.get("scheduler_name", "OneCycleLR")),
            "scheduler_step_mode": "batch",
            "scheduler_factory": lambda optimizer: torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(config.get("learning_rate", 1e-3)),
                epochs=max(1, int(epochs)),
                steps_per_epoch=max(1, int(steps_per_epoch)),
                pct_start=float(config.get("scheduler_pct_start", 0.2)),
                div_factor=float(config.get("scheduler_div_factor", 1e4)),
                final_div_factor=float(config.get("scheduler_final_div_factor", 1e4)),
            ),
        }
    return {
        "loss_name": CFD_CNS_DEFAULT_TRAINING_LOSS,
        "loss_definition": CFD_CNS_DEFAULT_TRAINING_LOSS_DEFINITION,
        "loss_rationale": CFD_CNS_DEFAULT_TRAINING_LOSS_RATIONALE,
        "criterion": _build_cfd_cns_training_criterion(),
        "optimizer_name": "Adam",
        "learning_rate": float(default_learning_rate),
        "weight_decay": 0.0,
        "scheduler_name": "ReduceLROnPlateau",
        "scheduler_step_mode": "epoch",
        "scheduler_factory": lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            threshold=0.0,
        ),
    }


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
        image_spec = cfd_cns_field_visual_spec(name, [prediction[channel], target[channel]])
        error_spec = cfd_cns_field_visual_spec(name, [error[channel]], is_error=True)
        panels = [
            (
                f"Prediction {name}",
                prediction[channel],
                image_spec["cmap"],
                image_spec["vmin"],
                image_spec["vmax"],
            ),
            (
                f"Ground truth {name}",
                target[channel],
                image_spec["cmap"],
                image_spec["vmin"],
                image_spec["vmax"],
            ),
            (
                f"Abs error {name}",
                error[channel],
                error_spec["cmap"],
                error_spec["vmin"],
                error_spec["vmax"],
            ),
        ]
        for axis, (title, image, cmap, vmin, vmax) in zip(axes[channel], panels, strict=True):
            handle = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    png_path = output_root / f"comparison_{profile_id}_sample0.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return {"comparison_png": str(png_path), "comparison_npz": str(npz_path)}


def _evaluate_loader(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    state_stats: dict[str, Any],
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            predictions.append(model(x).cpu())
            targets.append(y.cpu())
    metrics = dynamic_state_metric_payload(predictions, targets, normalized=True, state_stats=state_stats)
    return predictions, targets, metrics


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
    runtime: DistributedRuntime | None = None,
) -> dict[str, Any]:
    runtime = runtime or DistributedRuntime(requested_device=device_name, device=_resolve_device(device_name))
    profile = get_model_profile(profile_id)
    started = time.time()
    device = runtime.device
    sample = train_dataset[0]
    task_metadata = {
        "task_id": "2d_cfd_cns",
        "spatial_shape": [int(spatial_shape[0]), int(spatial_shape[1])],
        "state_shape": list(metadata["state_shape"]),
        "field_order": list(metadata["field_order"]),
        "field_axis_order": str(metadata["field_axis_order"]),
        "history_len": int(metadata["history_len"]),
        "time_steps": int(metadata["time_steps"]),
        "trajectory_count": int(metadata["trajectory_count"]),
        "input_channels": int(metadata["input_channels"]),
        "target_channels": int(metadata["target_channels"]),
        "dx": float(metadata["dx"]),
        "dy": float(metadata["dy"]),
        "dt": float(metadata["dt"]),
        "eta": float(metadata["eta"]),
        "zeta": float(metadata["zeta"]),
        "boundary_condition": str(metadata["boundary_condition"]),
        "sample_contract": str(metadata["dynamic_history_contract"]),
    }
    try:
        raw_model = build_model_from_profile(
            profile,
            in_channels=int(sample["input"].shape[0]),
            out_channels=int(sample["target"].shape[0]),
            spatial_shape=spatial_shape,
            task_metadata=task_metadata,
        ).to(device)
    except ModelBuildBlocker as exc:
        blocker = {
            **exc.to_payload(run_id=run_id),
            "profile_id": profile_id,
            "status": "blocked",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if runtime.is_rank_zero:
            _write_json(output_root / f"model_profile_{profile_id}.json", {**profile.to_model_config(), **blocker})
            _write_json(output_root / f"metrics_{profile_id}.json", blocker)
        return runtime.broadcast_object(
            {"profile_id": profile_id, "status": "blocked", "blocker_reason": exc.reason},
            src=0,
        )

    model_profile = describe_model(raw_model, profile=profile)
    if runtime.is_rank_zero:
        _write_json(output_root / f"model_profile_{profile_id}.json", model_profile)
    model: torch.nn.Module = raw_model
    if runtime.distributed_enabled:
        ddp_kwargs: dict[str, Any] = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        model = DistributedDataParallel(raw_model, **ddp_kwargs)
    train_loader, train_sampler = runtime.build_training_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        shuffle=False,
    )
    training_recipe = _build_cfd_cns_training_recipe(
        profile=profile,
        default_learning_rate=float(learning_rate),
        steps_per_epoch=len(train_loader),
        epochs=int(epochs),
    )
    if training_recipe["optimizer_name"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_recipe["learning_rate"]),
            weight_decay=float(training_recipe["weight_decay"]),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training_recipe["learning_rate"]),
            weight_decay=float(training_recipe["weight_decay"]),
        )
    criterion = training_recipe["criterion"]
    physics_regularizer = build_physics_regularizer(
        task_id="2d_cfd_cns",
        metadata=metadata,
        state_stats=state_stats,
        config=physics_config,
    )
    scheduler = training_recipe["scheduler_factory"](optimizer)

    runtime.maybe_reset_peak_memory_stats()
    model.train()
    total_train_batches = 0
    epoch_losses: list[float] = []
    supervised_epoch_losses: list[float] = []
    physics_epoch_history: list[dict[str, Any]] = []
    for epoch_index in range(int(epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_index)
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
            if training_recipe["scheduler_step_mode"] == "batch":
                scheduler.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_supervised += float(supervised_loss.detach().cpu().item())
            epoch_physics_total += float(physics_result.total.detach().cpu().item())
            for name, value in physics_result.terms.items():
                epoch_term_totals[name] += float(value.detach().cpu().item())
            for name, value in physics_result.weighted_terms.items():
                epoch_weighted_term_totals[name] += float(value.detach().cpu().item())
            epoch_batches += 1
        global_epoch_loss = runtime.reduce_sum(epoch_loss)
        global_epoch_supervised = runtime.reduce_sum(epoch_supervised)
        global_epoch_physics_total = runtime.reduce_sum(epoch_physics_total)
        global_epoch_batches = int(round(runtime.reduce_sum(epoch_batches)))
        total_train_batches += global_epoch_batches
        global_epoch_terms = runtime.reduce_sum_dict(epoch_term_totals)
        global_epoch_weighted_terms = runtime.reduce_sum_dict(epoch_weighted_term_totals)
        if global_epoch_batches:
            mean_epoch_loss = global_epoch_loss / global_epoch_batches
            epoch_losses.append(mean_epoch_loss)
            mean_supervised_loss = global_epoch_supervised / global_epoch_batches
            supervised_epoch_losses.append(mean_supervised_loss)
            mean_physics_total = global_epoch_physics_total / global_epoch_batches
            epoch_physics_payload = {
                "total": mean_physics_total,
                "terms": {name: value / global_epoch_batches for name, value in sorted(global_epoch_terms.items())},
                "weighted_terms": {
                    name: value / global_epoch_batches for name, value in sorted(global_epoch_weighted_terms.items())
                },
            }
            physics_epoch_history.append(epoch_physics_payload)
            if runtime.is_rank_zero:
                print(
                    f"EPOCH_LOSS profile={profile_id} epoch={epoch_index + 1} "
                    f"loss={mean_epoch_loss:.10g} loss_name={training_recipe['loss_name']} "
                    f"supervised={mean_supervised_loss:.10g} physics_total={mean_physics_total:.10g} "
                    f"physics_terms={json.dumps(epoch_physics_payload['terms'], sort_keys=True)}",
                    flush=True,
                )
            if training_recipe["scheduler_step_mode"] == "epoch":
                scheduler.step(mean_epoch_loss)

    runtime.barrier()
    peak_memory = runtime.max_cuda_memory_bytes()
    result: dict[str, Any] | None = None
    if runtime.is_rank_zero:
        train_eval_loader = DataLoader(
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
        raw_model.eval()
        _, _, train_split_metrics = _evaluate_loader(
            model=raw_model,
            data_loader=train_eval_loader,
            device=device,
            state_stats=state_stats,
        )
        predictions, targets, metrics = _evaluate_loader(
            model=raw_model,
            data_loader=eval_loader,
            device=device,
            state_stats=state_stats,
        )
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
            "train_batches": int(total_train_batches),
            "train_epoch_losses": epoch_losses,
            "train_supervised_epoch_losses": supervised_epoch_losses,
            "training_loss": training_recipe["loss_name"],
            "training_loss_definition": training_recipe["loss_definition"],
            "training_loss_rationale": training_recipe["loss_rationale"],
            "optimizer": training_recipe["optimizer_name"],
            "learning_rate": float(training_recipe["learning_rate"]),
            "weight_decay": float(training_recipe["weight_decay"]),
            "train_split_eval": {
                **train_split_metrics,
                "split_name": "train",
            },
            "physics_regularization_enabled": bool(physics_config.enabled),
            "physics_loss_terms": physics_config.active_terms(),
            "physics_loss_weights": physics_config.to_payload()["weights"],
            "physics_last_epoch": (
                physics_epoch_history[-1] if physics_epoch_history else {"total": 0.0, "terms": {}, "weighted_terms": {}}
            ),
            "physics_epoch_history": physics_epoch_history,
            "scheduler": training_recipe["scheduler_name"],
            "runtime_sec": time.time() - started,
            "peak_cuda_memory_bytes": peak_memory,
            "model_profile": model_profile,
            **runtime.training_runtime_payload(),
            **comparison_artifacts,
        }
        _write_json(output_root / f"metrics_{profile_id}.json", payload)
        result = _profile_result_from_metrics(profile_id, payload, model_profile)
    return runtime.broadcast_object(result, src=0)


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
    runtime: DistributedRuntime | None = None,
) -> int:
    if task_id != "2d_cfd_cns":
        raise ValueError("run_cfd_cns only supports task_id='2d_cfd_cns'")
    if mode not in {"inspect", "readiness", "pilot", "benchmark"}:
        raise ValueError("2d_cfd_cns mode must be inspect, readiness, pilot, or benchmark")
    runtime_owned = runtime is None
    runtime = runtime or initialize_runtime(device)
    output_root = Path(output_root)
    try:
        prepare_output_root(output_root, allow_existing=allow_existing_output_root, runtime=runtime)
        physics_config = physics_config or PhysicsRegularizationConfig()
        data_file = Path(data_root) / "2d_cfd_cns" / CFD_CNS_FILENAME
        if not data_file.exists():
            raise FileNotFoundError(f"missing 2D CFD CNS data file: {data_file}")
        if mode == "pilot" and not profile_ids:
            raise ValueError("2d_cfd_cns pilot mode requires explicit profile_ids")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        if runtime.is_rank_zero:
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
                    **runtime.training_runtime_payload(),
                },
                extra={"run_id": run_id, "runtime_provenance": capture_runtime_provenance(), "pid": os.getpid()},
            )
        metadata = inspect_cfd_cns_hdf5(data_file, history_len=history_len)
        if runtime.is_rank_zero:
            _write_json(output_root / "hdf5_metadata.json", metadata)
            _write_dataset_manifest(output_root, metadata=metadata)

        full_split = _split_for_run(metadata)
        run_split = capped_trajectory_split(
            full_split,
            max_train_trajectories=max_train_trajectories,
            max_val_trajectories=max_val_trajectories,
            max_test_trajectories=max_test_trajectories,
        )
        if runtime.is_rank_zero:
            write_trajectory_split_manifest(
                output_root=output_root,
                data_file=data_file,
                split=run_split,
                state_dataset=",".join(metadata["field_order"]),
                axis_order=metadata["field_axis_order"],
                shape=metadata["state_shape"],
                history_len=int(history_len),
                max_windows_per_trajectory=max_windows_per_trajectory,
                extra={
                    "full_split_counts": {name: len(full_split[name]) for name in ("train", "val", "test")},
                    "run_mode": mode,
                    **runtime.training_runtime_payload(),
                },
            )
        if mode == "inspect":
            return 0
        state_stats = compute_multifield_dynamic_stats(
            data_file=data_file,
            field_order=metadata["field_order"],
            axis_order=metadata["field_axis_order"],
            train_trajectory_ids=run_split["train"],
        )
        state_stats = {**state_stats, "run_id": run_id, "history_len": int(history_len)}
        if runtime.is_rank_zero:
            _write_json(output_root / "normalization_stats_state.json", state_stats)

        if profile_ids:
            profile_ids = list(profile_ids)
        elif mode == "benchmark":
            profile_ids = list(PRIMARY_CFD_CNS_PROFILE_IDS)
        else:
            profile_ids = list(READINESS_CFD_CNS_PROFILE_IDS)
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
                runtime=runtime,
            )
            profile_results.append(result)
            if result.get("status") != "completed":
                blockers.append(result)
        if runtime.is_rank_zero:
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
            summary.update(runtime.training_runtime_payload())
            write_comparison_summary(summary, output_root)
        return 0 if any(item.get("status") == "completed" for item in profile_results) else 1
    finally:
        if runtime_owned:
            runtime.finalize()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="2d_cfd_cns")
    parser.add_argument("--mode", choices=["inspect", "readiness", "pilot", "benchmark"], default="readiness")
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
