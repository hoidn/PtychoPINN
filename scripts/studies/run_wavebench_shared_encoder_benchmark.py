from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.wavebench_shared_encoder.data import (
    LOCKED_ROWS,
    build_dataloaders,
    load_locked_contract,
    summarize_tensor_batch,
)
from scripts.studies.wavebench_shared_encoder.metrics import compute_reconstruction_metrics
from scripts.studies.wavebench_shared_encoder.models import build_shared_encoder_row, profile_model
from scripts.studies.wavebench_shared_encoder.reporting import (
    read_json_if_exists,
    write_comparison_csv,
    write_json,
)


OUTPUT_ROOT_RELATIVE = (
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-29-wavebench-shared-encoder-supervised-benchmark"
)


def default_output_root(repo_root: Path) -> Path:
    return repo_root / OUTPUT_ROOT_RELATIVE


def build_row_contract(repo_root: Path, wavebench_root: Path) -> dict[str, Any]:
    contract = load_locked_contract(repo_root)
    return {
        "selected_variant": contract["selected_variant"],
        "selected_dataset_member": contract["selected_dataset_member"],
        "stable_dataset_target": contract["stable_dataset_target"],
        "wavebench_root": str(wavebench_root.resolve()),
        "split": contract["split"],
        "latent_channel_settings": contract["latent_channels"],
        "rows": contract["row_roster"],
        "training_recipe": {
            "loss": "l1",
            "optimizer": {"name": "adam", "lr": 2e-4},
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "factor": 0.5,
                "patience": 2,
                "min_lr": 1e-5,
                "threshold": 0.0,
            },
            "seed": 42,
            "batch_size": 32,
            "epochs": 50,
        },
        "row_status_values": ["completed", "blocked", "not_protocol_compatible"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WaveBench shared-encoder benchmark rows.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--wavebench-root", default="tmp/wavebench_repo")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_RELATIVE)
    parser.add_argument("--mode", required=True, choices=("inspect", "smoke", "benchmark"))
    parser.add_argument("--row", default="all")
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
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


def inspect_split_samples(dataloaders: dict[str, Any], output_path: Path) -> dict[str, Any]:
    payload = {"splits": {}}
    for split, loader in dataloaders.items():
        inputs, targets = next(iter(loader))
        payload["splits"][split] = summarize_tensor_batch(inputs, targets)
    write_json(output_path, payload)
    return payload


def _device_from_args(gpu_device: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_device}")
    return torch.device("cpu")


def _collect_predictions(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device=device, dtype=torch.float32)
            batch_targets = batch_targets.to(device=device, dtype=torch.float32)
            batch_predictions = model(batch_inputs)
            predictions.append(batch_predictions.cpu())
            targets.append(batch_targets.cpu())
    return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)


def _train_model(
    model: nn.Module,
    dataloaders: dict[str, Any],
    *,
    device: torch.device,
    learning_rate: float,
    epochs: int,
) -> tuple[float, float]:
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        threshold=0.0,
    )

    train_loss = 0.0
    val_loss = 0.0
    for _ in range(epochs):
        model.train()
        train_total = 0.0
        train_batches = 0
        for batch_inputs, batch_targets in dataloaders["train"]:
            batch_inputs = batch_inputs.to(device=device, dtype=torch.float32)
            batch_targets = batch_targets.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item())
            train_batches += 1
        train_loss = train_total / max(train_batches, 1)

        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in dataloaders["val"]:
                batch_inputs = batch_inputs.to(device=device, dtype=torch.float32)
                batch_targets = batch_targets.to(device=device, dtype=torch.float32)
                predictions = model(batch_inputs)
                val_total += float(criterion(predictions, batch_targets).item())
                val_batches += 1
        val_loss = val_total / max(val_batches, 1)
        scheduler.step(val_loss)
    return train_loss, val_loss


def _update_bundle(
    *,
    output_root: Path,
    contract: dict[str, Any],
    row: str,
    latent_channels: int,
    metrics_payload: dict[str, Any],
) -> None:
    manifest_path = output_root / "shared_encoder_execution_manifest.json"
    manifest = read_json_if_exists(
        manifest_path,
        {
            "selected_variant": contract["selected_variant"],
            "selected_dataset_member": contract["selected_dataset_member"],
            "stable_dataset_target": contract["stable_dataset_target"],
            "authoritative_artifacts": {
                "row_contract": (
                    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
                    "2026-04-29-wavebench-shared-encoder-supervised-benchmark/row_contract.json"
                )
            },
            "rows": {},
        },
    )
    manifest.setdefault("rows", {}).setdefault(row, {})[f"c{latent_channels}"] = {
        "status": metrics_payload["status"],
        "artifact_path": f"rows/{row}/c{latent_channels}/metrics.json",
    }
    write_json(manifest_path, manifest)

    metrics_table_path = output_root / "table_ready_metrics.json"
    table = read_json_if_exists(
        metrics_table_path,
        {"selected_variant": contract["selected_variant"], "rows": []},
    )
    rows = [entry for entry in table["rows"] if not (entry["row"] == row and entry["latent_channels"] == latent_channels)]
    rows.append(metrics_payload)
    rows.sort(key=lambda entry: (entry["row"], entry["latent_channels"]))
    table["rows"] = rows
    write_json(metrics_table_path, table)
    write_json(output_root / "comparison_summary.json", table)
    write_comparison_csv(output_root / "comparison_summary.csv", rows)


def status_for_mode(mode: str) -> str:
    if mode == "benchmark":
        return "completed"
    if mode == "smoke":
        return "smoke_pass"
    raise ValueError(f"unsupported run mode for status mapping: {mode}")


def run_row(
    *,
    row: str,
    latent_channels: int,
    dataloaders: dict[str, Any],
    output_root: Path,
    contract: dict[str, Any],
    mode: str,
    epochs: int,
    device: torch.device,
    learning_rate: float,
    figure_sample_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    torch.manual_seed(int(contract["training_recipe"]["seed"]))
    model = build_shared_encoder_row(row=row, latent_channels=latent_channels).to(device)
    row_dir = output_root / "rows" / row / f"c{latent_channels}"
    row_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    train_loss, val_loss = _train_model(
        model,
        dataloaders,
        device=device,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    runtime_seconds = time.time() - start

    predictions, targets = _collect_predictions(model, dataloaders["test"], device=device)
    metrics = compute_reconstruction_metrics(predictions, targets)
    profile = profile_model(model)
    profile["runtime_seconds"] = runtime_seconds
    profile["peak_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    profile["train_loss"] = train_loss
    profile["val_loss"] = val_loss
    profile["epochs"] = epochs

    status = status_for_mode(mode)
    metrics_payload = {
        "row": row,
        "latent_channels": latent_channels,
        "mode": mode,
        "status": status,
        "metrics": metrics,
    }
    write_json(row_dir / "metrics.json", metrics_payload)
    write_json(row_dir / "model_profile.json", profile)
    _update_bundle(
        output_root=output_root,
        contract=contract,
        row=row,
        latent_channels=latent_channels,
        metrics_payload=metrics_payload,
    )

    if mode == "benchmark":
        from scripts.studies.wavebench_shared_encoder.reporting import write_row_figures

        sample_indices = figure_sample_indices or (0, 1, 2, 3)
        write_row_figures(
            output_root=output_root,
            row=row,
            latent_channels=latent_channels,
            predictions=predictions,
            targets=targets,
            sample_indices=sample_indices,
        )
    return {"status": status, "metrics_path": str(row_dir / "metrics.json")}


def main() -> None:
    args = parse_args()
    repo_root, output_root, wavebench_root = resolve_paths(args)
    output_root.mkdir(parents=True, exist_ok=True)

    contract = build_row_contract(repo_root=repo_root, wavebench_root=wavebench_root)
    write_json(output_root / "row_contract.json", contract)

    dataloaders = build_dataloaders(
        repo_root=repo_root,
        wavebench_root=wavebench_root,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    if args.mode == "inspect":
        inspect_split_samples(dataloaders, output_root / "inspection.json")
        print("wavebench shared-encoder inspect completed")
        return

    rows = list(LOCKED_ROWS) if args.row == "all" else [args.row]
    device = _device_from_args(args.gpu_device)
    epochs = args.epochs
    if epochs is None:
        epochs = 1 if args.mode == "smoke" else int(contract["training_recipe"]["epochs"])

    for row in rows:
        run_row(
            row=row,
            latent_channels=args.latent_channels,
            dataloaders=dataloaders,
            output_root=output_root,
            contract=contract,
            mode=args.mode,
            epochs=epochs,
            device=device,
            learning_rate=args.learning_rate,
        )
    print("wavebench shared-encoder rows completed")


if __name__ == "__main__":
    main()
