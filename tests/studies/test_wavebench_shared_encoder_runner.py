import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.studies import run_wavebench_shared_encoder_benchmark as runner


def _tiny_dataloaders():
    inputs = torch.rand(8, 1, 128, 128)
    targets = torch.rand(8, 1, 128, 128)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return {"train": loader, "val": loader, "test": loader}


def test_run_row_smoke_writes_metrics_with_smoke_pass_status(tmp_path: Path):
    output_root = tmp_path / "wavebench_shared"
    contract = runner.build_row_contract(repo_root=Path("."), wavebench_root=Path("tmp/wavebench_repo"))

    result = runner.run_row(
        row="cnn",
        latent_channels=32,
        dataloaders=_tiny_dataloaders(),
        output_root=output_root,
        contract=contract,
        mode="smoke",
        epochs=1,
        device=torch.device("cpu"),
        learning_rate=2e-4,
    )

    metrics_path = output_root / "rows" / "cnn" / "c32" / "metrics.json"
    profile_path = output_root / "rows" / "cnn" / "c32" / "model_profile.json"
    manifest_path = output_root / "shared_encoder_execution_manifest.json"

    assert result["status"] == "smoke_pass"
    assert metrics_path.exists()
    assert profile_path.exists()
    assert manifest_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert metrics["row"] == "cnn"
    assert metrics["latent_channels"] == 32
    assert metrics["mode"] == "smoke"
    assert metrics["status"] == "smoke_pass"
    assert "MAE" in metrics["metrics"]
    assert profile["total_parameters"] > 0
    assert manifest["rows"]["cnn"]["c32"]["status"] == "smoke_pass"
    assert manifest["rows"]["cnn"]["c32"]["artifact_path"] == "rows/cnn/c32/metrics.json"


def test_run_row_benchmark_records_completed_and_writes_figures(tmp_path: Path):
    output_root = tmp_path / "wavebench_shared"
    contract = runner.build_row_contract(repo_root=Path("."), wavebench_root=Path("tmp/wavebench_repo"))

    result = runner.run_row(
        row="cnn",
        latent_channels=32,
        dataloaders=_tiny_dataloaders(),
        output_root=output_root,
        contract=contract,
        mode="benchmark",
        epochs=1,
        device=torch.device("cpu"),
        learning_rate=2e-4,
        figure_sample_indices=(0, 1),
    )

    assert result["status"] == "completed"
    metrics = json.loads(
        (output_root / "rows" / "cnn" / "c32" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["mode"] == "benchmark"
    assert metrics["status"] == "completed"

    figure_manifest_path = (
        output_root / "figures" / "c32" / "cnn" / "figure_manifest.json"
    )
    arrays_dir = output_root / "figures" / "source_arrays" / "cnn" / "c32"
    assert figure_manifest_path.exists()
    assert (arrays_dir / "sample_000.npz").exists()
    assert (arrays_dir / "sample_001.npz").exists()


def test_inspect_split_samples_writes_one_real_sample_per_split(tmp_path: Path):
    output_path = tmp_path / "inspect.json"
    runner.inspect_split_samples(_tiny_dataloaders(), output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert set(payload["splits"]) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        entry = payload["splits"][split]
        assert entry["input_shape"] == [2, 1, 128, 128]
        assert entry["target_shape"] == [2, 1, 128, 128]
