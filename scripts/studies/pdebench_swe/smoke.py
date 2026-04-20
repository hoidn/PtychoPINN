"""CLI and training loop for the PDEBench SWE primary smoke gate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from scripts.studies.invocation_logging import (
    capture_runtime_provenance,
    write_invocation_artifacts,
)
from scripts.studies.pdebench_swe.data import SweOneStepDataset
from scripts.studies.pdebench_swe.manifest import (
    ManifestBlocker,
    file_identity,
    inspect_hdf5,
    select_state_dataset,
    write_dataset_manifests,
)
from scripts.studies.pdebench_swe.metrics import compute_channel_stats, metric_payload
from scripts.studies.pdebench_swe.models import (
    ModelBuildBlocker,
    build_model,
    describe_model,
)
from scripts.studies.pdebench_swe.splits import (
    build_trajectory_split,
    infer_dimensions,
    write_split_manifest,
)


DEFAULT_RAW_ROOT = ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate"
SCRIPT_PATH = "scripts/studies/run_pdebench_swe_smoke.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PDEBench SWE one-step smoke gate.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_RAW_ROOT))
    parser.add_argument("--dataset-source", default="PDEBench")
    parser.add_argument("--dataset-source-url", default="https://github.com/pdebench/PDEBench")
    parser.add_argument("--dataset-darus-id", default="133021")
    parser.add_argument("--license-note", default="")
    parser.add_argument("--state-dataset", default=None)
    parser.add_argument("--axis-order", default=None)
    parser.add_argument("--split-seed", type=int, default=20260420)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-trajectories", type=int, default=None)
    parser.add_argument("--max-val-trajectories", type=int, default=None)
    parser.add_argument("--max-test-trajectories", type=int, default=None)
    parser.add_argument("--max-pairs-per-trajectory", type=int, default=None)
    parser.add_argument("--pad-multiple", type=int, default=1)
    parser.add_argument("--models", default="hybrid_resnet,fno,unet")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--allow-existing-output-root", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def capture_smoke_provenance(*, run_id: str, output_root: Path, data_file: Path) -> dict[str, Any]:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except Exception:
        git_commit = None
    try:
        dirty_summary = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
    except Exception:
        dirty_summary = []

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "run_id": run_id,
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "output_root": str(Path(output_root).resolve()),
        "data_file": str(Path(data_file).resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit,
        "git_dirty_summary": dirty_summary,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "packages": {
            "h5py": _package_version("h5py"),
            "torch": _package_version("torch"),
            "neuralop": _package_version("neuralop"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _guard_output_root(output_root: Path, *, allow_existing: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing:
        raise FileExistsError(f"refusing to write into non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def _limit_ids(ids: list[int], limit: int | None) -> list[int]:
    return ids if limit is None else ids[: int(limit)]


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "trajectory_id": [item["trajectory_id"] for item in batch],
        "time_index": [item["time_index"] for item in batch],
        "pad": [item["pad"] for item in batch],
    }


def _run_model(
    *,
    model_name: str,
    train_dataset: SweOneStepDataset,
    eval_dataset: SweOneStepDataset,
    channels: int,
    spatial_shape: tuple[int, int],
    args: argparse.Namespace,
    output_root: Path,
    run_id: str,
    root_provenance: dict[str, Any],
    stats: dict[str, Any],
) -> bool:
    model_root = output_root / "runs" / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    smoke_config = {
        "hidden_channels": 8,
        "fno_modes": 4,
        "fno_blocks": 3,
        "hybrid_downsample_steps": 1,
        "hybrid_resnet_blocks": 1,
    }
    started = time.time()
    device = _resolve_device(args.device)
    try:
        model = build_model(
            model_name,
            in_channels=channels,
            out_channels=channels,
            spatial_shape=spatial_shape,
            smoke_config=smoke_config,
        ).to(device)
    except ModelBuildBlocker as exc:
        blocker = {
            **exc.to_payload(run_id=run_id),
            "pid": os.getpid(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(model_root / "blocker.json", blocker)
        _write_json(model_root / "provenance.json", {**root_provenance, "model": model_name, "pid": os.getpid()})
        return False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    model.train()
    train_batches = 0
    for _epoch in range(args.epochs):
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            train_batches += 1

    model.eval()
    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(eval_loader):
            if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
                break
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            prediction_batches.append(model(x).cpu())
            target_batches.append(y.cpu())
    if not prediction_batches:
        raise RuntimeError(f"{model_name} produced no evaluation batches")

    metrics = metric_payload(prediction_batches, target_batches, normalized=True, stats=stats)
    runtime_sec = time.time() - started
    peak_memory = None
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    model_description = describe_model(model, model_name=model_name, smoke_config=smoke_config)
    payload = {
        **metrics,
        "run_id": run_id,
        "pid": os.getpid(),
        "model": model_name,
        "train_batches": int(train_batches),
        "runtime_sec": runtime_sec,
        "peak_cuda_memory_bytes": peak_memory,
        "model_description": model_description,
    }
    _write_json(model_root / "metrics.json", payload)
    _write_json(
        model_root / "provenance.json",
        {
            **root_provenance,
            "model": model_name,
            "pid": os.getpid(),
            "runtime_sec": runtime_sec,
            "model_description": model_description,
        },
    )
    return True


def run(args: argparse.Namespace, *, raw_argv: list[str]) -> int:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = Path(args.output_root)
    _guard_output_root(output_root, allow_existing=args.allow_existing_output_root)

    root_provenance = capture_smoke_provenance(run_id=run_id, output_root=output_root, data_file=args.data_file)
    write_invocation_artifacts(
        output_dir=output_root,
        script_path=SCRIPT_PATH,
        argv=raw_argv,
        parsed_args=vars(args),
        extra={
            "run_id": run_id,
            "runtime_provenance": capture_runtime_provenance(),
            "smoke_provenance": root_provenance,
        },
    )

    if not Path(args.data_file).exists():
        raise FileNotFoundError(f"missing HDF5 data file: {args.data_file}")
    try:
        write_dataset_manifests(
            data_file=args.data_file,
            output_root=output_root,
            dataset_source=args.dataset_source,
            dataset_source_url=args.dataset_source_url,
            dataset_darus_id=args.dataset_darus_id,
            license_note=args.license_note,
            state_dataset=args.state_dataset,
            axis_order=args.axis_order,
            run_id=run_id,
        )
        metadata = inspect_hdf5(args.data_file)
        selected = select_state_dataset(metadata, requested=args.state_dataset)
    except ManifestBlocker:
        raise
    if args.axis_order:
        selected["axis_order"] = args.axis_order
    if args.inspect_only:
        return 0

    shape = [int(item) for item in selected["shape"]]
    axis_order = str(selected["axis_order"])
    dims = infer_dimensions(shape, axis_order)
    ratios = (args.train_fraction, args.val_fraction, args.test_fraction)
    split = build_trajectory_split(dims["num_trajectories"], seed=args.split_seed, ratios=ratios)
    split["train"] = _limit_ids(split["train"], args.max_train_trajectories)
    split["val"] = _limit_ids(split["val"], args.max_val_trajectories)
    split["test"] = _limit_ids(split["test"], args.max_test_trajectories)
    identity = file_identity(args.data_file)
    write_split_manifest(
        output_root=output_root,
        source_file_identity=identity,
        state_dataset=str(selected["path"]),
        axis_order=axis_order,
        shape=shape,
        split=split,
        max_pairs_per_trajectory=args.max_pairs_per_trajectory,
        run_id=run_id,
    )
    train_dataset = SweOneStepDataset(
        data_file=args.data_file,
        state_dataset=str(selected["path"]),
        trajectory_ids=list(split["train"]),
        axis_order=axis_order,
        max_pairs_per_trajectory=args.max_pairs_per_trajectory,
        pad_multiple=args.pad_multiple,
    )
    stats = compute_channel_stats(train_dataset, max_batches=args.max_train_batches)
    stats = {**stats, "run_id": run_id}
    _write_json(output_root / "normalization_stats.json", stats)
    eval_ids = list(split["val"] or split["test"] or split["train"])
    eval_dataset = SweOneStepDataset(
        data_file=args.data_file,
        state_dataset=str(selected["path"]),
        trajectory_ids=eval_ids,
        axis_order=axis_order,
        normalization=stats,
        max_pairs_per_trajectory=args.max_pairs_per_trajectory,
        pad_multiple=args.pad_multiple,
    )
    train_dataset.normalization = stats

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    successes = 0
    blockers = 0
    for model_name in models:
        try:
            if _run_model(
                model_name=model_name,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                channels=dims["channels"],
                spatial_shape=(dims["height"], dims["width"]),
                args=args,
                output_root=output_root,
                run_id=run_id,
                root_provenance=root_provenance,
                stats=stats,
            ):
                successes += 1
            else:
                blockers += 1
        except Exception as exc:
            blockers += 1
            model_root = output_root / "runs" / model_name
            _write_json(
                model_root / "blocker.json",
                {
                    "run_id": run_id,
                    "pid": os.getpid(),
                    "model": model_name,
                    "reason": "model_smoke_failed",
                    "message": str(exc),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            _write_json(model_root / "provenance.json", {**root_provenance, "model": model_name, "pid": os.getpid()})
    return 0 if successes > 0 else 1


def _extract_run_id(payload: dict[str, Any]) -> str | None:
    for candidate in (
        payload.get("run_id"),
        payload.get("run", {}).get("run_id") if isinstance(payload.get("run"), dict) else None,
        payload.get("provenance", {}).get("run_id") if isinstance(payload.get("provenance"), dict) else None,
    ):
        if candidate is not None:
            return str(candidate)
    return None


def validate_fresh_artifacts(
    *,
    output_root: Path,
    run_id: str,
    tracked_pid: str,
    start_ns: int,
    models: list[str],
) -> list[str]:
    """Return validation errors for run_id/PID/freshness contract artifacts."""
    output_root = Path(output_root)
    errors: list[str] = []

    def load(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            errors.append(f"missing smoke contract artifact: {path}")
            return None
        if path.stat().st_mtime_ns < start_ns:
            errors.append(f"stale smoke artifact predates tracked run start: {path}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON artifact {path}: {exc}")
            return None

    for name in ["dataset_manifest.json", "hdf5_metadata.json", "split_manifest.json", "normalization_stats.json"]:
        path = output_root / name
        payload = load(path)
        if payload is not None and _extract_run_id(payload) != run_id:
            errors.append(f"{path} does not record current run_id {run_id}")
    for model in models:
        model_root = output_root / "runs" / model
        provenance = load(model_root / "provenance.json")
        if provenance is not None:
            if _extract_run_id(provenance) != run_id:
                errors.append(f"{model} provenance does not record current run_id {run_id}")
            pid_values = {
                provenance.get("pid"),
                provenance.get("process_pid"),
                provenance.get("smoke_pid"),
            }
            if tracked_pid not in {str(value) for value in pid_values if value is not None}:
                errors.append(f"{model} provenance does not match tracked PID {tracked_pid}")
        written = [path for path in [model_root / "metrics.json", model_root / "blocker.json"] if path.exists()]
        if not written:
            errors.append(f"{model} wrote neither metrics.json nor blocker.json")
        for path in written:
            payload = load(path)
            if payload is not None and _extract_run_id(payload) != run_id:
                errors.append(f"{path} does not record current run_id {run_id}")
    return errors


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    try:
        return run(args, raw_argv=raw_argv)
    except (FileExistsError, FileNotFoundError, ManifestBlocker, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
