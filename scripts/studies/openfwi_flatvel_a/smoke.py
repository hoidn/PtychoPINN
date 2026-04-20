"""CLI and bounded train/eval loop for the OpenFWI FlatVel-A smoke gate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.studies.invocation_logging import capture_runtime_provenance, write_invocation_artifacts
from scripts.studies.openfwi_flatvel_a.data import (
    EXPECTED_SHARD_SAMPLES,
    OpenFWIShardDataset,
    build_split_manifest,
    compute_normalization_stats,
    inspect_shard_pair,
)
from scripts.studies.openfwi_flatvel_a.manifest import (
    OpenFWIManifestBlocker,
    build_data_manifest,
    resolve_required_shards,
    validate_data_root_policy,
    write_json,
)
from scripts.studies.openfwi_flatvel_a.metrics import metric_payload
from scripts.studies.openfwi_flatvel_a.models import build_model, describe_model, probe_official_inversionnet
from scripts.studies.openfwi_flatvel_a.reporting import collate_comparison
from scripts.studies.openfwi_flatvel_a.run_config import parse_profile_ids, validate_run_budget


DEFAULT_RAW_ROOT = ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate"
SCRIPT_PATH = "scripts/studies/run_openfwi_flatvel_a_smoke.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OpenFWI FlatVel-A fallback smoke gate.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_RAW_ROOT))
    parser.add_argument("--source-url", default="https://openfwi-lanl.github.io/docs/data.html")
    parser.add_argument("--source-access-note", default="")
    parser.add_argument("--license-note", default="")
    parser.add_argument("--profiles", default="hybrid_resnet_smoke,unet_smoke")
    parser.add_argument("--official-openfwi-repo", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=20260420)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--allow-existing-output-root", action="store_true")
    parser.add_argument("--allow-synthetic-shard-samples", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args(argv)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _pid_is_live(pid: str) -> bool:
    return pid.isdigit() and Path(f"/proc/{pid}").exists()


def _pid_markers(output_root: Path) -> list[Path]:
    direct_marker = output_root / "logs" / "smoke.pid"
    markers = [direct_marker] if direct_marker.exists() else []
    markers.extend(output_root.glob("runs/*/logs/smoke.pid"))
    return markers


def _is_current_launcher_marker(marker: Path, pid: str, output_root: Path) -> bool:
    return (
        marker == output_root / "logs" / "smoke.pid"
        and pid == str(os.getpid())
        and not marker.with_name("smoke.exit_code").exists()
    )


def _is_plan_prelaunch_layout(output_root: Path) -> bool:
    if not output_root.exists():
        return False
    entries = list(output_root.iterdir())
    if len(entries) != 1 or entries[0].name != "logs" or not entries[0].is_dir():
        return False
    allowed = {"smoke.pid", "smoke.run_id", "smoke.started_at_ns"}
    log_entries = list(entries[0].iterdir())
    if not all(path.name in allowed for path in log_entries):
        return False
    pid_marker = entries[0] / "smoke.pid"
    if pid_marker.exists():
        return _is_current_launcher_marker(pid_marker, pid_marker.read_text(encoding="utf-8").strip(), output_root)
    return True


def _guard_output_root(output_root: Path, *, allow_existing: bool) -> None:
    output_root = Path(output_root)
    live = []
    incomplete = []
    for marker in _pid_markers(output_root):
        pid = marker.read_text(encoding="utf-8").strip()
        exit_marker = marker.with_name("smoke.exit_code")
        if _is_current_launcher_marker(marker, pid, output_root):
            continue
        if _pid_is_live(pid):
            live.append((str(marker), pid))
        elif not exit_marker.exists():
            incomplete.append((str(marker), pid))
    if live:
        raise FileExistsError(f"live OpenFWI smoke output root exists: {live}")
    if incomplete:
        raise FileExistsError(f"incomplete OpenFWI smoke output root has missing exit code evidence: {incomplete}")
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing and not _is_plan_prelaunch_layout(output_root):
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
        "sample_id": [item["sample_id"] for item in batch],
    }


def capture_smoke_provenance(*, run_id: str, output_root: Path, data_root: Path) -> dict[str, Any]:
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    except Exception:
        git_commit = None
    try:
        git_dirty = subprocess.run(["git", "status", "--short"], check=True, text=True, capture_output=True).stdout.splitlines()
    except Exception:
        git_dirty = []
    cuda_available = torch.cuda.is_available()
    return {
        "run_id": run_id,
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "output_root": str(Path(output_root).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit,
        "git_dirty_summary": git_dirty,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "packages": {
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "scikit-image": _package_version("scikit-image"),
            "neuralop": _package_version("neuralop"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(dataset: OpenFWIShardDataset, *, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)


def _eval_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, stats: dict[str, Any]) -> dict[str, Any]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            predictions.append(model(x).detach().cpu())
            targets.append(y.detach().cpu())
    return metric_payload(predictions, targets, normalized=True, target_stats=stats)


def _run_profile(
    *,
    profile_id: str,
    train_dataset: OpenFWIShardDataset,
    val_dataset: OpenFWIShardDataset,
    test_dataset: OpenFWIShardDataset,
    args: argparse.Namespace,
    output_root: Path,
    run_id: str,
    stats: dict[str, Any],
    root_provenance: dict[str, Any],
) -> None:
    profile_root = output_root / "runs" / profile_id
    profile_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = _resolve_device(args.device)
    try:
        model = build_model(profile_id, in_channels=5, out_channels=1, spatial_shape=(70, 70), profile_config={}).to(device)
    except Exception as exc:
        write_json(
            profile_root / "blocker.json",
            {
                "run_id": run_id,
                "profile_id": profile_id,
                "reason": "model_build_blocked",
                "message": str(exc),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        write_json(profile_root / "provenance.json", {**root_provenance, "profile_id": profile_id, "pid": os.getpid()})
        return

    train_loader = _loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = _loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = _loader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for _epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(x), y)
            loss.backward()
            optimizer.step()

    val_metrics = _eval_model(model, val_loader, device, stats) if len(val_dataset) else {}
    test_metrics = _eval_model(model, test_loader, device, stats) if len(test_dataset) else {}
    runtime_sec = time.time() - started
    description = describe_model(model, profile_id=profile_id, profile_config={})
    payload = {
        **test_metrics,
        "run_id": run_id,
        "profile_id": profile_id,
        "split_manifest": "split_manifest.json",
        "normalization_stats": "normalization_stats.json",
        "eval": {"val": val_metrics, "test": test_metrics},
        "runtime_sec": runtime_sec,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "model_description": description,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(profile_root / "metrics.json", payload)
    write_json(
        profile_root / "provenance.json",
        {
            **root_provenance,
            "profile_id": profile_id,
            "pid": os.getpid(),
            "runtime_sec": runtime_sec,
            "model_description": description,
        },
    )


def run(args: argparse.Namespace, *, raw_argv: list[str]) -> int:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root)
    data_root = Path(args.data_root).expanduser().resolve()
    _guard_output_root(output_root, allow_existing=args.allow_existing_output_root)
    _seed_everything(int(args.split_seed))
    budget = validate_run_budget(vars(args))
    profiles = parse_profile_ids(args.profiles)
    root_provenance = capture_smoke_provenance(run_id=run_id, output_root=output_root, data_root=data_root)
    write_invocation_artifacts(
        output_dir=output_root,
        script_path=SCRIPT_PATH,
        argv=raw_argv,
        parsed_args=vars(args),
        extra={
            "run_id": run_id,
            "runtime_provenance": capture_runtime_provenance(),
            "smoke_provenance": root_provenance,
            "run_budget": budget,
        },
    )
    write_json(output_root / "preflight" / "package_provenance.json", root_provenance)

    try:
        validate_data_root_policy(data_root, repo_root=Path.cwd())
        shards = resolve_required_shards(data_root)
    except OpenFWIManifestBlocker as exc:
        write_json(output_root / "data_access_blocker.json", exc.to_payload(run_id=run_id))
        collate_comparison(output_root, profiles=profiles, run_id=run_id)
        raise

    stale_data_blocker = output_root / "data_access_blocker.json"
    if stale_data_blocker.exists():
        stale_data_blocker.unlink()

    manifest = build_data_manifest(
        data_root=data_root,
        shards=shards,
        source_url=args.source_url,
        license_note=args.license_note,
        access_note=args.source_access_note,
        run_id=run_id,
    )
    write_json(output_root / "data_manifest.json", manifest)
    expected_samples = None if args.allow_synthetic_shard_samples else EXPECTED_SHARD_SAMPLES
    train_shapes = inspect_shard_pair(shards["data1.npy"], shards["model1.npy"], expected_samples=expected_samples)
    test_shapes = inspect_shard_pair(shards["data49.npy"], shards["model49.npy"], expected_samples=expected_samples)
    shape_payload = {
        "schema_version": "openfwi_flatvel_a_shard_shapes_v1",
        "run_id": run_id,
        "pairs": [train_shapes, test_shapes],
        "shape_validation_complete": train_shapes["status"] == "valid" and test_shapes["status"] == "valid",
    }
    write_json(output_root / "shard_shapes.json", shape_payload)
    if not shape_payload["shape_validation_complete"]:
        raise ValueError("OpenFWI FlatVel-A shard shapes do not match the smoke contract")

    split = build_split_manifest(
        train_count=train_shapes["num_samples"],
        test_count=test_shapes["num_samples"],
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        seed=args.split_seed,
    )
    split["run_id"] = run_id
    write_json(output_root / "split_manifest.json", split)

    official_probe = probe_official_inversionnet(args.official_openfwi_repo)
    official_probe["run_id"] = run_id
    if official_probe["status"] == "blocked":
        write_json(output_root / "official_inversionnet_blocker.json", official_probe)
    else:
        write_json(output_root / "official_inversionnet_compatibility.json", official_probe)

    train_dataset = OpenFWIShardDataset(
        data_path=shards["data1.npy"],
        model_path=shards["model1.npy"],
        split_name="train",
        indices=split["train"]["indices"],
    )
    stats = compute_normalization_stats(train_dataset)
    stats["run_id"] = run_id
    write_json(output_root / "normalization_stats.json", stats)
    if args.inspect_only:
        collate_comparison(output_root, profiles=profiles, run_id=run_id)
        return 0

    train_dataset.normalization = stats
    val_dataset = OpenFWIShardDataset(
        data_path=shards["data49.npy"],
        model_path=shards["model49.npy"],
        split_name="val",
        indices=split["val"]["indices"],
        normalization=stats,
    )
    test_dataset = OpenFWIShardDataset(
        data_path=shards["data49.npy"],
        model_path=shards["model49.npy"],
        split_name="test",
        indices=split["test"]["indices"],
        normalization=stats,
    )
    for profile_id in profiles:
        if profile_id == "official_inversionnet_probe":
            continue
        _run_profile(
            profile_id=profile_id,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            args=args,
            output_root=output_root,
            run_id=run_id,
            stats=stats,
            root_provenance=root_provenance,
        )
    collate_comparison(output_root, profiles=profiles, run_id=run_id)
    return 0


def _extract_run_id(payload: dict[str, Any]) -> str | None:
    if payload.get("run_id") is not None:
        return str(payload["run_id"])
    extra = payload.get("extra")
    if isinstance(extra, dict) and extra.get("run_id") is not None:
        return str(extra["run_id"])
    return None


def validate_fresh_artifacts(*, output_root: Path, run_id: str, start_ns: int) -> list[str]:
    """Return freshness/contract validation errors for selected-run artifacts."""
    output_root = Path(output_root)
    errors: list[str] = []
    exit_code = output_root / "logs" / "smoke.exit_code"
    if not exit_code.exists():
        errors.append(f"missing smoke.exit_code: {exit_code}")
    elif exit_code.read_text(encoding="utf-8").strip() != "0":
        errors.append(f"smoke.exit_code is not 0: {exit_code.read_text(encoding='utf-8').strip()}")
    required = [
        "invocation.json",
        "invocation.sh",
        "data_manifest.json",
        "shard_shapes.json",
        "split_manifest.json",
        "normalization_stats.json",
        "comparison_summary.json",
        "comparison_summary.csv",
    ]
    for name in required:
        path = output_root / name
        if not path.exists():
            errors.append(f"missing smoke contract artifact: {path}")
            continue
        if path.stat().st_mtime_ns < start_ns:
            errors.append(f"stale smoke artifact predates tracked run start: {path}")
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON artifact {path}: {exc}")
            else:
                payload_run_id = _extract_run_id(payload)
                if payload_run_id is not None and payload_run_id != run_id:
                    errors.append(f"{path} does not record current run_id {run_id}")
    return errors


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    try:
        return run(args, raw_argv=raw_argv)
    except (FileExistsError, FileNotFoundError, OpenFWIManifestBlocker, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
