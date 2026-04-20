"""CLI and training loop for PDEBench SWE longer one-step execution."""

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
    build_model_from_profile,
    describe_model,
)
from scripts.studies.pdebench_swe.reporting import collate_comparison
from scripts.studies.pdebench_swe.run_config import (
    ABLATION_PROFILE_IDS,
    PRIMARY_PROFILE_IDS,
    get_model_profile,
    load_run_budget,
    parse_profile_ids,
)
from scripts.studies.pdebench_swe.splits import (
    build_run_subset_split,
    build_trajectory_split,
    infer_dimensions,
    write_longer_split_manifests,
)


DEFAULT_RAW_ROOT = ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution"
SCRIPT_PATH = "scripts/studies/run_pdebench_swe_longer.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDEBench SWE longer one-step execution.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_RAW_ROOT) / "runs" / "manual")
    parser.add_argument("--dataset-source", default="PDEBench")
    parser.add_argument("--dataset-source-url", default="https://github.com/pdebench/PDEBench")
    parser.add_argument("--dataset-darus-id", default="133021")
    parser.add_argument("--license-note", default="")
    parser.add_argument("--license-note-file", type=Path, default=None)
    parser.add_argument("--run-budget-file", type=Path, default=None)
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
    parser.add_argument("--pad-multiple", type=int, default=2)
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--run-ablations-if-viable", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--normalization-max-samples", type=int, default=None)
    parser.add_argument("--eval-splits", default=None)
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


def _module_origin(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(Path(module.__file__).resolve()) if getattr(module, "__file__", None) else None


def capture_longer_provenance(*, run_id: str, output_root: Path, data_file: Path) -> dict[str, Any]:
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
            "neuraloperator": _package_version("neuraloperator"),
            "numpy": _package_version("numpy"),
        },
        "module_origins": {
            "torch": _module_origin("torch"),
            "h5py": _module_origin("h5py"),
            "neuralop": _module_origin("neuralop"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _pid_is_live(pid: str) -> bool:
    return pid.isdigit() and Path(f"/proc/{pid}").exists()


def _contains_only_prelaunch_markers(output_root: Path) -> bool:
    allowed = {"longer.run_id", "longer.started_at_ns", "longer.pid"}
    entries = list(output_root.iterdir()) if output_root.exists() else []
    if not entries:
        return True
    if len(entries) != 1 or entries[0].name != "logs" or not entries[0].is_dir():
        return False
    files = [path for path in entries[0].iterdir()]
    return all(path.is_file() and path.name in allowed for path in files)


def _guard_output_root(output_root: Path, *, allow_existing: bool) -> None:
    if (
        output_root.exists()
        and any(output_root.iterdir())
        and not allow_existing
        and not _contains_only_prelaunch_markers(output_root)
    ):
        raise FileExistsError(f"refusing to write into non-empty output root: {output_root}")
    pid_path = output_root / "logs" / "longer.pid"
    if allow_existing and pid_path.exists():
        pid = pid_path.read_text(encoding="utf-8").strip()
        exit_path = output_root / "logs" / "longer.exit_code"
        if _pid_is_live(pid) and not exit_path.exists():
            raise FileExistsError(f"refusing to write into live output root {output_root}; PID {pid} is active")
    output_root.mkdir(parents=True, exist_ok=True)


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


def _split_arg(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_budget_defaults(args: argparse.Namespace) -> tuple[list[str], list[str], list[str], dict[str, Any] | None]:
    budget = load_run_budget(args.run_budget_file) if args.run_budget_file else None
    if budget is not None:
        defaults = {
            "epochs": budget["epochs"],
            "batch_size": budget["batch_size"],
            "learning_rate": budget["learning_rate"],
            "max_train_trajectories": budget["max_train_trajectories"],
            "max_val_trajectories": budget["max_val_trajectories"],
            "max_test_trajectories": budget["max_test_trajectories"],
            "max_pairs_per_trajectory": budget["max_pairs_per_trajectory"],
            "normalization_max_samples": budget["normalization_max_samples"],
            "num_workers": budget["num_workers"],
            "device": budget["device"],
        }
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        if args.eval_splits is None:
            args.eval_splits = ",".join(budget["eval_splits"])

    fallback_defaults = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "max_train_trajectories": None,
        "max_val_trajectories": None,
        "max_test_trajectories": None,
        "max_pairs_per_trajectory": None,
        "normalization_max_samples": None,
        "num_workers": 0,
        "device": "cuda",
        "eval_splits": "val,test",
    }
    for key, value in fallback_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    primary_profiles = parse_profile_ids(
        args.profiles if args.profiles is not None else (
            budget["primary_profiles"] if budget is not None else PRIMARY_PROFILE_IDS
        )
    )
    ablation_profiles = parse_profile_ids(
        budget["ablation_profiles"] if budget is not None else ABLATION_PROFILE_IDS
    )
    eval_splits = _split_arg(args.eval_splits, ["val", "test"])
    return primary_profiles, ablation_profiles, eval_splits, budget


def _write_start_markers(output_root: Path, run_id: str) -> int:
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    start_ns = time.time_ns()
    (logs / "longer.run_id").write_text(f"{run_id}\n", encoding="utf-8")
    (logs / "longer.started_at_ns").write_text(f"{start_ns}\n", encoding="utf-8")
    (logs / "longer.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    return start_ns


def _make_dataset(
    *,
    data_file: Path,
    state_dataset: str,
    trajectory_ids: list[int],
    axis_order: str,
    stats: dict[str, Any] | None,
    max_pairs_per_trajectory: int | None,
    pad_multiple: int,
) -> SweOneStepDataset:
    return SweOneStepDataset(
        data_file=data_file,
        state_dataset=state_dataset,
        trajectory_ids=trajectory_ids,
        axis_order=axis_order,
        normalization=stats,
        max_pairs_per_trajectory=max_pairs_per_trajectory,
        pad_multiple=pad_multiple,
    )


def _evaluate(
    *,
    model: torch.nn.Module,
    dataset: SweOneStepDataset,
    args: argparse.Namespace,
    device: torch.device,
    stats: dict[str, Any],
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            prediction_batches.append(model(x).cpu())
            target_batches.append(y.cpu())
    if not prediction_batches:
        raise RuntimeError("evaluation produced no batches")
    return metric_payload(prediction_batches, target_batches, normalized=True, stats=stats)


def _run_profile(
    *,
    profile_id: str,
    datasets: dict[str, SweOneStepDataset],
    channels: int,
    spatial_shape: tuple[int, int],
    args: argparse.Namespace,
    output_root: Path,
    run_id: str,
    root_provenance: dict[str, Any],
    stats: dict[str, Any],
    eval_splits: list[str],
    source_paths: dict[str, str],
) -> bool:
    profile = get_model_profile(profile_id)
    profile_root = output_root / "runs" / profile_id
    profile_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model = build_model_from_profile(
            profile,
            in_channels=channels,
            out_channels=channels,
            spatial_shape=spatial_shape,
        ).to(device)
    except ModelBuildBlocker as exc:
        blocker = {
            **exc.to_payload(run_id=run_id),
            "profile_id": profile_id,
            "pid": os.getpid(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(profile_root / "blocker.json", blocker)
        _write_json(profile_root / "provenance.json", {**root_provenance, "profile_id": profile_id, "pid": os.getpid()})
        return False

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    train_batches = 0
    model.train()
    for _epoch in range(args.epochs):
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_batches += 1

    eval_payload = {
        split: _evaluate(model=model, dataset=datasets[split], args=args, device=device, stats=stats)
        for split in eval_splits
    }
    runtime_sec = time.time() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    memory_label = "per_profile_cuda_peak" if device.type == "cuda" else "unavailable_cpu"
    model_description = describe_model(model, model_name=profile_id, smoke_config=profile.to_model_config())
    test_payload = eval_payload.get("test", next(iter(eval_payload.values())))
    payload = {
        "schema_version": "pdebench_swe_profile_metrics_v1",
        "run_id": run_id,
        "pid": os.getpid(),
        "profile_id": profile_id,
        "base_model": profile.base_model,
        "data_file": str(Path(args.data_file).resolve()),
        "split_manifest_full": source_paths["split_manifest_full"],
        "split_manifest_run": source_paths["split_manifest_run"],
        "normalization_stats": source_paths["normalization_stats"],
        "horizon": "one_step",
        "metric_units": test_payload.get("metric_units"),
        "eval": eval_payload,
        "err_RMSE": test_payload.get("err_RMSE"),
        "err_nRMSE": test_payload.get("err_nRMSE"),
        "train_batches": int(train_batches),
        "runtime_sec": runtime_sec,
        "peak_cuda_memory_bytes": peak_memory,
        "memory_measurement": memory_label,
        "model_description": model_description,
        "run_budget": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "max_pairs_per_trajectory": args.max_pairs_per_trajectory,
        },
    }
    _write_json(profile_root / "metrics.json", payload)
    _write_json(
        profile_root / "provenance.json",
        {
            **root_provenance,
            "profile_id": profile_id,
            "pid": os.getpid(),
            "runtime_sec": runtime_sec,
            "memory_measurement": memory_label,
            "model_description": model_description,
        },
    )
    return True


def run(args: argparse.Namespace, *, raw_argv: list[str]) -> int:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = Path(args.output_root)
    _guard_output_root(output_root, allow_existing=args.allow_existing_output_root)
    _write_start_markers(output_root, run_id)
    primary_profiles, ablation_profiles, eval_splits, budget = _apply_budget_defaults(args)

    root_provenance = capture_longer_provenance(run_id=run_id, output_root=output_root, data_file=args.data_file)
    write_invocation_artifacts(
        output_dir=output_root,
        script_path=SCRIPT_PATH,
        argv=raw_argv,
        parsed_args=vars(args),
        extra={
            "run_id": run_id,
            "runtime_provenance": capture_runtime_provenance(),
            "longer_provenance": root_provenance,
            "run_budget": budget,
        },
    )

    if not Path(args.data_file).exists():
        raise FileNotFoundError(f"missing HDF5 data file: {args.data_file}")
    write_dataset_manifests(
        data_file=args.data_file,
        output_root=output_root,
        dataset_source=args.dataset_source,
        dataset_source_url=args.dataset_source_url,
        dataset_darus_id=args.dataset_darus_id,
        license_note=args.license_note,
        license_note_file=args.license_note_file,
        state_dataset=args.state_dataset,
        axis_order=args.axis_order,
        run_id=run_id,
    )
    metadata = inspect_hdf5(args.data_file)
    selected = select_state_dataset(metadata, requested=args.state_dataset)
    if args.axis_order:
        selected["axis_order"] = args.axis_order
    if args.inspect_only:
        return 0

    shape = [int(item) for item in selected["shape"]]
    axis_order = str(selected["axis_order"])
    dims = infer_dimensions(shape, axis_order)
    ratios = (args.train_fraction, args.val_fraction, args.test_fraction)
    full_split = build_trajectory_split(dims["num_trajectories"], seed=args.split_seed, ratios=ratios)
    run_split = build_run_subset_split(
        full_split,
        max_train_trajectories=args.max_train_trajectories,
        max_val_trajectories=args.max_val_trajectories,
        max_test_trajectories=args.max_test_trajectories,
    )
    identity = file_identity(args.data_file)
    split_paths = write_longer_split_manifests(
        output_root=output_root,
        source_file_identity=identity,
        state_dataset=str(selected["path"]),
        axis_order=axis_order,
        shape=shape,
        full_split=full_split,
        run_split=run_split,
        max_pairs_per_trajectory=args.max_pairs_per_trajectory,
        run_id=run_id,
    )

    train_dataset_for_stats = _make_dataset(
        data_file=args.data_file,
        state_dataset=str(selected["path"]),
        trajectory_ids=list(run_split["train"]),
        axis_order=axis_order,
        stats=None,
        max_pairs_per_trajectory=args.max_pairs_per_trajectory,
        pad_multiple=args.pad_multiple,
    )
    stats = compute_channel_stats(
        train_dataset_for_stats,
        normalization_max_samples=args.normalization_max_samples,
    )
    train_dataset_for_stats.close()
    stats = {**stats, "run_id": run_id}
    normalization_path = _write_json(output_root / "normalization_stats.json", stats)

    datasets = {
        split: _make_dataset(
            data_file=args.data_file,
            state_dataset=str(selected["path"]),
            trajectory_ids=list(run_split[split]),
            axis_order=axis_order,
            stats=stats,
            max_pairs_per_trajectory=args.max_pairs_per_trajectory,
            pad_multiple=args.pad_multiple,
        )
        for split in ("train", "val", "test")
    }
    source_paths = {
        "split_manifest_full": str(split_paths["full"]),
        "split_manifest_run": str(split_paths["run"]),
        "normalization_stats": str(normalization_path),
    }

    for profile_id in primary_profiles:
        try:
            _run_profile(
                profile_id=profile_id,
                datasets=datasets,
                channels=dims["channels"],
                spatial_shape=(dims["height"], dims["width"]),
                args=args,
                output_root=output_root,
                run_id=run_id,
                root_provenance=root_provenance,
                stats=stats,
                eval_splits=eval_splits,
                source_paths=source_paths,
            )
        except Exception as exc:
            profile_root = output_root / "runs" / profile_id
            _write_json(
                profile_root / "blocker.json",
                {
                    "run_id": run_id,
                    "pid": os.getpid(),
                    "profile_id": profile_id,
                    "reason": "profile_execution_failed",
                    "message": str(exc),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            _write_json(profile_root / "provenance.json", {**root_provenance, "profile_id": profile_id, "pid": os.getpid()})

    comparison = collate_comparison(
        output_root,
        primary_profiles=primary_profiles,
        ablation_profiles=[],
        run_id=run_id,
    )
    ablation_skip_reason = None
    if args.run_ablations_if_viable:
        if comparison["recommended_decision_input"] == "primary_viable":
            for profile_id in ablation_profiles:
                try:
                    _run_profile(
                        profile_id=profile_id,
                        datasets=datasets,
                        channels=dims["channels"],
                        spatial_shape=(dims["height"], dims["width"]),
                        args=args,
                        output_root=output_root,
                        run_id=run_id,
                        root_provenance=root_provenance,
                        stats=stats,
                        eval_splits=eval_splits,
                        source_paths=source_paths,
                    )
                except Exception as exc:
                    profile_root = output_root / "runs" / profile_id
                    _write_json(
                        profile_root / "blocker.json",
                        {
                            "run_id": run_id,
                            "pid": os.getpid(),
                            "profile_id": profile_id,
                            "reason": "ablation_execution_failed",
                            "message": str(exc),
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    _write_json(profile_root / "provenance.json", {**root_provenance, "profile_id": profile_id, "pid": os.getpid()})
        else:
            ablation_skip_reason = comparison["recommended_decision_input"]
    else:
        ablation_skip_reason = "not_requested"

    collate_comparison(
        output_root,
        primary_profiles=primary_profiles,
        ablation_profiles=ablation_profiles if args.run_ablations_if_viable and ablation_skip_reason is None else [],
        run_id=run_id,
        ablation_skip_reason=ablation_skip_reason,
    )
    for dataset in datasets.values():
        dataset.close()
    return 0


def _extract_run_id(payload: dict[str, Any]) -> str | None:
    for candidate in (
        payload.get("run_id"),
        payload.get("extra", {}).get("run_id") if isinstance(payload.get("extra"), dict) else None,
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
    profiles: list[str],
) -> list[str]:
    """Return validation errors for longer-run run_id/PID/freshness artifacts."""
    output_root = Path(output_root)
    errors: list[str] = []

    def load_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            errors.append(f"missing longer contract artifact: {path}")
            return None
        if path.stat().st_mtime_ns < start_ns:
            errors.append(f"stale longer artifact predates tracked run start: {path}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON artifact {path}: {exc}")
            return None

    root_json = [
        "dataset_manifest.json",
        "hdf5_metadata.json",
        "split_manifest_full.json",
        "split_manifest_run.json",
        "normalization_stats.json",
        "comparison_summary.json",
        "invocation.json",
    ]
    for name in root_json:
        path = output_root / name
        payload = load_json(path)
        if payload is not None and _extract_run_id(payload) not in {run_id, None}:
            errors.append(f"{path} does not record current run_id {run_id}")
    for name in ["comparison_summary.csv", "invocation.sh"]:
        path = output_root / name
        if not path.exists():
            errors.append(f"missing longer contract artifact: {path}")
        elif path.stat().st_mtime_ns < start_ns:
            errors.append(f"stale longer artifact predates tracked run start: {path}")

    invocation = load_json(output_root / "invocation.json")
    if invocation is not None:
        if str(invocation.get("pid")) != str(tracked_pid):
            errors.append(f"invocation PID {invocation.get('pid')!r} does not match tracked PID {tracked_pid}")
        parsed_args = invocation.get("parsed_args", {})
        if parsed_args.get("run_id") is not None and str(parsed_args.get("run_id")) != str(run_id):
            errors.append(f"invocation parsed run_id does not match {run_id}")

    for profile_id in profiles:
        profile_root = output_root / "runs" / profile_id
        provenance = load_json(profile_root / "provenance.json")
        if provenance is not None:
            if _extract_run_id(provenance) != run_id:
                errors.append(f"{profile_id} provenance run_id does not match {run_id}")
            pid_values = {provenance.get("pid"), provenance.get("process_pid"), provenance.get("launcher_pid")}
            if tracked_pid not in {str(value) for value in pid_values if value is not None}:
                errors.append(f"{profile_id} provenance PID does not match tracked PID {tracked_pid}")
        written = [path for path in [profile_root / "metrics.json", profile_root / "blocker.json"] if path.exists()]
        if not written:
            errors.append(f"{profile_id} wrote neither metrics.json nor blocker.json")
        for path in written:
            payload = load_json(path)
            if payload is not None:
                if _extract_run_id(payload) != run_id:
                    errors.append(f"{path} does not record current run_id {run_id}")
                if str(payload.get("pid")) != str(tracked_pid):
                    errors.append(f"{path} PID does not match tracked PID {tracked_pid}")
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
