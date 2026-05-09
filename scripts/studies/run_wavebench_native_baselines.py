from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROVISIONING_DECISION_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/provisioning_decision.json"
)
NATIVE_BASELINE_PROVENANCE_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/native_baseline_provenance.json"
)
DATASET_MANIFEST_PATH = (
    REPO_ROOT
    / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/dataset_manifest.json"
)
OUTPUT_ROOT_RELATIVE = (
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-29-wavebench-native-baseline-reproduction"
)
SUMMARY_RELATIVE = (
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_native_baseline_summary.md"
)
RUN_ROOTS_FILENAME = "run_roots.json"
MANIFEST_FILENAME = "native_baseline_execution_manifest.json"
TABLE_METRICS_FILENAME = "table_ready_metrics.json"
CSV_FILENAME = "wavebench_native_rows.csv"

ROW_CONFIG = {
    "unet": {
        "row_id": "wavebench_unet_ch32_native",
        "family": "unet-ch-32",
        "result_filename": "native_unet_eval.json",
        "mode": "checkpoint_eval",
    },
    "fno": {
        "row_id": "wavebench_fno_depth4_native",
        "family": "fno-depth-4",
        "result_filename": "native_fno_result.json",
        "mode": "train_and_eval",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.exists(), f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def default_output_root(repo_root: Path) -> Path:
    return repo_root / OUTPUT_ROOT_RELATIVE


def load_contract(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    provisioning = load_json(
        repo_root
        / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/provisioning_decision.json"
    )
    native_baselines = load_json(
        repo_root
        / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/native_baseline_provenance.json"
    )
    dataset_manifest = load_json(
        repo_root
        / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-wavebench-provisioning-decision/dataset_manifest.json"
    )

    observed_checkout = dataset_manifest["observed_wavebench_checkout"]
    native_rows = {
        "unet": {
            "row_id": ROW_CONFIG["unet"]["row_id"],
            "family": ROW_CONFIG["unet"]["family"],
            "route": native_baselines["unet"]["status"],
            "checkpoint_path": native_baselines["unet"]["representative_checkpoint_probe"][
                "local_checkpoint_path"
            ],
            "checkpoint_file_id": native_baselines["unet"]["representative_checkpoint_probe"][
                "file_id"
            ],
            "training_hparams": native_baselines["unet"]["representative_checkpoint_probe"][
                "training_hparams"
            ],
            "model_config": native_baselines["unet"]["representative_checkpoint_probe"][
                "model_config"
            ],
            "retrain_route": native_baselines["unet"]["retrain_route"],
        },
        "fno": {
            "row_id": ROW_CONFIG["fno"]["row_id"],
            "family": ROW_CONFIG["fno"]["family"],
            "route": native_baselines["fno"]["status"],
            "checkpoint_path": native_baselines["fno"]["representative_checkpoint_probe"][
                "local_checkpoint_path"
            ],
            "checkpoint_file_id": native_baselines["fno"]["representative_checkpoint_probe"][
                "file_id"
            ],
            "training_hparams": native_baselines["fno"]["representative_checkpoint_probe"][
                "checkpoint_hyper_parameters"
            ],
            "model_config": native_baselines["fno"]["representative_checkpoint_probe"][
                "checkpoint_hyper_parameters"
            ]["model_config"],
            "retrain_route": native_baselines["fno"]["retrain_route"],
        },
    }

    return {
        "selected_variant": provisioning["selected_variant"],
        "selected_dataset_member": provisioning["selected_dataset_member"],
        "stable_dataset_target": provisioning["stable_dataset_target"],
        "split": {"train": 9000, "val": 500, "test": 500, "seed": 42},
        "wavebench_checkout": {
            "path": str((repo_root / observed_checkout["path"]).resolve())
            if not Path(observed_checkout["path"]).is_absolute()
            else observed_checkout["path"],
            "repo_relative": Path(observed_checkout["path"]).resolve()
            .relative_to(repo_root.resolve())
            .as_posix()
            if Path(observed_checkout["path"]).resolve().is_relative_to(repo_root.resolve())
            else observed_checkout["path"],
            "revision": observed_checkout["repo_revision"],
        },
        "dataset": {
            "actual_path": dataset_manifest["actual_path"],
            "sha256": dataset_manifest["sha256"],
            "file_size_bytes": dataset_manifest["file_size_bytes"],
        },
        "path_normalization_decision": provisioning["path_normalization_decision"],
        "native_rows": native_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native WaveBench reference baselines.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--wavebench-root", default="tmp/wavebench_repo")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_RELATIVE)
    parser.add_argument("--row", required=True, choices=("unet", "fno"))
    parser.add_argument(
        "--mode",
        required=True,
        choices=("smoke", "checkpoint_eval", "train_and_eval"),
    )
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--train-num-epochs", type=int, default=50)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    repo_root = Path(args.repo_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if Path(args.output_root).is_absolute()
        else (repo_root / args.output_root).resolve()
    )
    wavebench_root = (
        Path(args.wavebench_root).resolve()
        if Path(args.wavebench_root).is_absolute()
        else (repo_root / args.wavebench_root).resolve()
    )
    return repo_root, output_root, wavebench_root


def ensure_wavebench_on_path(wavebench_root: Path) -> None:
    root_text = str(wavebench_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def load_wavebench_runtime(wavebench_root: Path) -> dict[str, Any]:
    ensure_wavebench_on_path(wavebench_root)
    pl_model_wrapper = importlib.import_module("wavebench.nn.pl_model_wrapper")
    is_loader = importlib.import_module("wavebench.dataloaders.is_loader")
    torch = importlib.import_module("torch")
    try:
        skimage_metrics = importlib.import_module("skimage.metrics")
    except ModuleNotFoundError:
        skimage_metrics = None
    return {
        "LitModel": pl_model_wrapper.LitModel,
        "get_model": pl_model_wrapper.get_model,
        "get_dataloaders_is_thick_lines": is_loader.get_dataloaders_is_thick_lines,
        "torch": torch,
        "skimage_metrics": skimage_metrics,
    }


def choose_eval_batch_size(test_samples: int, requested_batch_size: int) -> int:
    require(requested_batch_size > 0, "requested_batch_size must be positive")
    upper = min(test_samples, requested_batch_size)
    for candidate in range(upper, 0, -1):
        if test_samples % candidate == 0:
            return candidate
    return 1


def make_device(torch_module: Any, gpu_device: int) -> Any:
    if torch_module.cuda.is_available():
        return torch_module.device(f"cuda:{gpu_device}")
    return torch_module.device("cpu")


def compute_metrics(
    torch_module: Any,
    skimage_metrics: Any,
    predictions: Any,
    targets: Any,
) -> dict[str, float]:
    diff = predictions - targets
    mae = float(torch_module.mean(torch_module.abs(diff)).item())
    rmse = float(torch_module.sqrt(torch_module.mean(diff**2)).item())
    batch_size = predictions.shape[0]
    pred_flat = predictions.reshape(batch_size, -1)
    target_flat = targets.reshape(batch_size, -1)
    numer = torch_module.norm(pred_flat - target_flat, dim=1)
    denom = torch_module.norm(target_flat, dim=1).clamp_min(1e-12)
    rel_l2 = float(torch_module.mean(numer / denom).item())

    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    ssim_values: list[float] = []
    for idx in range(batch_size):
        pred_2d = pred_np[idx, 0]
        target_2d = target_np[idx, 0]
        if skimage_metrics is None:
            pred_centered = pred_2d - pred_2d.mean()
            target_centered = target_2d - target_2d.mean()
            numerator = (2 * (pred_centered * target_centered).mean()) + 1e-8
            denominator = (
                (pred_centered**2).mean() + (target_centered**2).mean() + 1e-8
            )
            ssim_values.append(float(numerator / denominator))
            continue
        data_min = float(min(pred_2d.min(), target_2d.min()))
        data_max = float(max(pred_2d.max(), target_2d.max()))
        data_range = max(data_max - data_min, 1e-6)
        ssim_values.append(
            float(
                skimage_metrics.structural_similarity(
                    target_2d,
                    pred_2d,
                    data_range=data_range,
                )
            )
        )
    ssim = sum(ssim_values) / len(ssim_values)
    return {"MAE": mae, "RMSE": rmse, "RelL2": rel_l2, "SSIM": ssim}


def upsert_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(
    repo_root: Path,
    output_root: Path,
    contract: dict[str, Any],
    row_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "selected_variant": contract["selected_variant"],
        "selected_dataset_member": contract["selected_dataset_member"],
        "stable_dataset_target": contract["stable_dataset_target"],
        "split": contract["split"],
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
        },
        "wavebench_checkout": {
            "repo_relative": contract["wavebench_checkout"]["repo_relative"],
            "revision": contract["wavebench_checkout"]["revision"],
        },
        "native_rows": {
            row_name: {
                "row_id": result["row_id"],
                "status": result["status"],
                "route": contract["native_rows"][row_name]["route"],
                "artifact_path": Path(result["artifact_path"])
                .relative_to(output_root)
                .as_posix(),
            }
            for row_name, result in row_results.items()
        },
        "authoritative_artifacts": {
            "summary": SUMMARY_RELATIVE,
            "table_ready_metrics": str(
                (Path(OUTPUT_ROOT_RELATIVE) / TABLE_METRICS_FILENAME).as_posix()
            ),
        },
    }


def update_table_ready_metrics(
    output_root: Path,
    contract: dict[str, Any],
    row_result: dict[str, Any],
) -> dict[str, Any]:
    table_path = output_root / TABLE_METRICS_FILENAME
    current = load_optional_json(table_path) or {
        "selected_variant": contract["selected_variant"],
        "rows": [],
    }
    rows = [row for row in current["rows"] if row["row_id"] != row_result["row_id"]]
    rows.append(row_result)
    rows.sort(key=lambda item: item["row_id"])
    current["rows"] = rows
    upsert_json(table_path, current)
    return current


def write_csv_projection(output_root: Path, table_ready_metrics: dict[str, Any]) -> None:
    csv_path = output_root / CSV_FILENAME
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "family",
                "status",
                "selected_variant",
                "test_samples",
                "MAE",
                "RMSE",
                "RelL2",
                "SSIM",
                "blocker_reason",
            ],
        )
        writer.writeheader()
        for row in sorted(table_ready_metrics["rows"], key=lambda item: item["row_id"]):
            metrics = row.get("metrics", {})
            writer.writerow(
                {
                    "row_id": row["row_id"],
                    "family": row["family"],
                    "status": row["status"],
                    "selected_variant": row["selected_variant"],
                    "test_samples": row["split"]["test_samples"],
                    "MAE": metrics.get("MAE"),
                    "RMSE": metrics.get("RMSE"),
                    "RelL2": metrics.get("RelL2"),
                    "SSIM": metrics.get("SSIM"),
                    "blocker_reason": row.get("blocker_reason"),
                }
            )


def update_run_roots(output_root: Path, row: str, payload: dict[str, Any]) -> None:
    run_roots_path = output_root / RUN_ROOTS_FILENAME
    current = load_optional_json(run_roots_path) or {}
    current[row] = payload
    upsert_json(run_roots_path, current)


def dataloaders_for_contract(
    runtime: dict[str, Any],
    row: str,
    batch_size: int,
    num_workers: int,
    max_test_samples: int | None,
) -> dict[str, Any]:
    eval_batch_size = batch_size
    if max_test_samples is None:
        eval_batch_size = choose_eval_batch_size(500, batch_size)
    kwargs: dict[str, Any] = {
        "medium_type": "gaussian_lens",
        "train_batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
    }
    if max_test_samples is not None:
        kwargs["num_train_samples"] = min(16, max_test_samples)
        kwargs["num_val_samples"] = min(8, max_test_samples)
        kwargs["num_test_samples"] = max_test_samples
    return runtime["get_dataloaders_is_thick_lines"](**kwargs)


def evaluate_model(
    *,
    runtime: dict[str, Any],
    model: Any,
    loader: Any,
    device: Any,
    max_batches: int | None = None,
) -> tuple[dict[str, float], int]:
    torch_module = runtime["torch"]
    model.eval()
    totals = {"MAE": 0.0, "RMSE": 0.0, "RelL2": 0.0, "SSIM": 0.0}
    sample_count = 0
    batch_count = 0
    with torch_module.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device=device, dtype=torch_module.float32)
            targets = targets.to(device=device, dtype=torch_module.float32)
            predictions = model(inputs)
            metrics = compute_metrics(
                torch_module,
                runtime["skimage_metrics"],
                predictions,
                targets,
            )
            current_batch = int(inputs.shape[0])
            for key, value in metrics.items():
                totals[key] += value * current_batch
            sample_count += current_batch
            batch_count += 1
    require(sample_count > 0, "evaluation produced zero samples")
    return (
        {key: value / sample_count for key, value in totals.items()},
        sample_count,
    )


def run_unet_checkpoint_eval(
    *,
    contract: dict[str, Any],
    runtime: dict[str, Any],
    output_root: Path,
    batch_size: int,
    num_workers: int,
    gpu_device: int,
    max_test_samples: int | None,
    smoke: bool,
) -> dict[str, Any]:
    torch_module = runtime["torch"]
    device = make_device(torch_module, gpu_device)
    checkpoint_path = Path(contract["native_rows"]["unet"]["checkpoint_path"])
    require(checkpoint_path.exists(), f"missing U-Net checkpoint: {checkpoint_path}")
    loaders = dataloaders_for_contract(runtime, "unet", batch_size, num_workers, max_test_samples)
    model = runtime["LitModel"].load_from_checkpoint(str(checkpoint_path), map_location=device)
    model = model.to(device)
    started = time.time()
    metrics, sample_count = evaluate_model(
        runtime=runtime,
        model=model,
        loader=loaders["test"],
        device=device,
        max_batches=1 if smoke else None,
    )
    elapsed = time.time() - started
    payload = {
        "row_id": ROW_CONFIG["unet"]["row_id"],
        "family": ROW_CONFIG["unet"]["family"],
        "status": "completed",
        "selected_variant": contract["selected_variant"],
        "split": {
            "test_samples": sample_count,
            "seed": contract["split"]["seed"],
        },
        "checkpoint_provenance": {
            "path": str(checkpoint_path),
            "file_id": contract["native_rows"]["unet"]["checkpoint_file_id"],
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "device": str(device),
        },
        "metrics": metrics,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "wall_clock_seconds": elapsed,
    }
    result_path = output_root / ROW_CONFIG["unet"]["result_filename"]
    upsert_json(result_path, payload)
    update_run_roots(
        output_root,
        "unet",
        {
            "mode": "smoke" if smoke else "checkpoint_eval",
            "checkpoint_path": str(checkpoint_path),
            "result_path": str(result_path),
        },
    )
    payload["artifact_path"] = str(result_path)
    return payload


def run_fno_smoke(
    *,
    contract: dict[str, Any],
    runtime: dict[str, Any],
    output_root: Path,
    batch_size: int,
    num_workers: int,
    gpu_device: int,
    max_test_samples: int | None,
) -> dict[str, Any]:
    torch_module = runtime["torch"]
    device = make_device(torch_module, gpu_device)
    loaders = dataloaders_for_contract(runtime, "fno", batch_size, num_workers, max_test_samples or 4)
    model = runtime["get_model"](contract["native_rows"]["fno"]["model_config"])
    model = model.to(device)
    metrics, sample_count = evaluate_model(
        runtime=runtime,
        model=model,
        loader=loaders["test"],
        device=device,
        max_batches=1,
    )
    payload = {
        "row_id": ROW_CONFIG["fno"]["row_id"],
        "family": ROW_CONFIG["fno"]["family"],
        "status": "smoke_passed",
        "selected_variant": contract["selected_variant"],
        "split": {
            "test_samples": sample_count,
            "seed": contract["split"]["seed"],
        },
        "metrics": metrics,
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "device": str(device),
        },
    }
    smoke_path = output_root / "fno_smoke_result.json"
    upsert_json(smoke_path, payload)
    update_run_roots(
        output_root,
        "fno_smoke",
        {
            "mode": "smoke",
            "result_path": str(smoke_path),
        },
    )
    return payload


def latest_checkpoint(root: Path, newer_than: float | None = None) -> Path:
    candidates = sorted(root.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if newer_than is not None:
        candidates = [
            path for path in candidates if path.stat().st_mtime >= newer_than
        ]
    require(candidates, f"no checkpoint files found under {root}")
    return candidates[0]


def run_fno_train_and_eval(
    *,
    contract: dict[str, Any],
    runtime: dict[str, Any],
    repo_root: Path,
    wavebench_root: Path,
    output_root: Path,
    batch_size: int,
    num_workers: int,
    gpu_device: int,
    train_num_epochs: int,
) -> dict[str, Any]:
    train_script = wavebench_root / contract["native_rows"]["fno"]["retrain_route"]["entrypoint"].replace(
        "tmp/wavebench_repo/", ""
    )
    require(train_script.exists(), f"missing WaveBench FNO training entrypoint: {train_script}")
    command = [
        sys.executable,
        str(train_script),
        "--medium_type",
        "gaussian_lens",
        "--num_layers",
        "4",
        "--batch_size",
        str(batch_size),
        "--num_epochs",
        str(train_num_epochs),
        "--loss_fun_type",
        "relative_l2",
        "--learning_rate",
        "1e-3",
        "--weight_decay",
        "0.01",
        "--eta_min",
        "1e-5",
        "--num_workers",
        str(num_workers),
        "--gpu_devices",
        str(gpu_device),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(wavebench_root)
    env["WANDB_MODE"] = "offline"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    started = time.time()
    process = subprocess.run(
        command,
        cwd=wavebench_root,
        env=env,
        capture_output=True,
        text=True,
    )
    log_path = output_root / "fno_train.log"
    log_path.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr, encoding="utf-8")
    require(process.returncode == 0, f"FNO training failed; see {log_path}")

    task_root = wavebench_root / "saved_models" / "is_gaussian_lens"
    checkpoint_path = latest_checkpoint(task_root, newer_than=started)
    update_run_roots(
        output_root,
        "fno",
        {
            "mode": "train_and_eval",
            "training_command": command,
            "training_log": str(log_path),
            "checkpoint_path": str(checkpoint_path),
        },
    )

    torch_module = runtime["torch"]
    device = make_device(torch_module, gpu_device)
    loaders = dataloaders_for_contract(runtime, "fno", batch_size, num_workers, None)
    model = runtime["LitModel"].load_from_checkpoint(str(checkpoint_path), map_location=device)
    model = model.to(device)
    metrics, sample_count = evaluate_model(
        runtime=runtime,
        model=model,
        loader=loaders["test"],
        device=device,
    )
    elapsed = time.time() - started
    payload = {
        "row_id": ROW_CONFIG["fno"]["row_id"],
        "family": ROW_CONFIG["fno"]["family"],
        "status": "completed",
        "selected_variant": contract["selected_variant"],
        "split": {
            "test_samples": sample_count,
            "seed": contract["split"]["seed"],
        },
        "training_output_provenance": {
            "checkpoint_path": str(checkpoint_path),
            "training_log": str(log_path),
            "command": command,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "device": str(device),
        },
        "metrics": metrics,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "wall_clock_seconds": elapsed,
    }
    result_path = output_root / ROW_CONFIG["fno"]["result_filename"]
    upsert_json(result_path, payload)
    payload["artifact_path"] = str(result_path)
    return payload


def persist_contract_bundle(
    output_root: Path,
    repo_root: Path,
    contract: dict[str, Any],
    row_result: dict[str, Any],
    row_name: str,
) -> None:
    table_ready_metrics = update_table_ready_metrics(output_root, contract, row_result)
    write_csv_projection(output_root, table_ready_metrics)

    row_results: dict[str, dict[str, Any]] = {}
    for name, config in ROW_CONFIG.items():
        artifact_path = output_root / config["result_filename"]
        payload = load_optional_json(artifact_path)
        if payload is None:
            continue
        payload["artifact_path"] = str(artifact_path)
        row_results[name] = payload

    manifest = build_manifest(repo_root, output_root, contract, row_results)
    upsert_json(output_root / MANIFEST_FILENAME, manifest)


def main() -> None:
    args = parse_args()
    repo_root, output_root, wavebench_root = resolve_paths(args)
    output_root.mkdir(parents=True, exist_ok=True)
    contract = load_contract(repo_root)
    require(
        contract["native_rows"][args.row]["route"]
        == ("checkpoint_reusable" if args.row == "unet" else "retrain_required")
        or args.mode == "smoke",
        f"row {args.row} does not match the locked route for mode {args.mode}",
    )

    runtime = load_wavebench_runtime(wavebench_root)
    if args.row == "unet":
        if args.mode == "train_and_eval":
            raise SystemExit("U-Net native row is locked to checkpoint_eval, not train_and_eval")
        row_result = run_unet_checkpoint_eval(
            contract=contract,
            runtime=runtime,
            output_root=output_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            gpu_device=args.gpu_device,
            max_test_samples=args.max_test_samples,
            smoke=args.mode == "smoke",
        )
    else:
        if args.mode == "checkpoint_eval":
            raise SystemExit("FNO native row is locked to train_and_eval, not checkpoint_eval")
        if args.mode == "smoke":
            row_result = run_fno_smoke(
                contract=contract,
                runtime=runtime,
                output_root=output_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                gpu_device=args.gpu_device,
                max_test_samples=args.max_test_samples,
            )
        else:
            row_result = run_fno_train_and_eval(
                contract=contract,
                runtime=runtime,
                repo_root=repo_root,
                wavebench_root=wavebench_root,
                output_root=output_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                gpu_device=args.gpu_device,
                train_num_epochs=args.train_num_epochs,
            )

    if row_result["status"] in {"completed", "blocked"}:
        persist_contract_bundle(output_root, repo_root, contract, row_result, args.row)
    print(json.dumps(row_result, indent=2))


if __name__ == "__main__":
    main()
