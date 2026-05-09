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


def test_run_row_writes_metrics_profiles_and_updates_manifest(tmp_path: Path):
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

    assert result["status"] == "completed"
    assert metrics_path.exists()
    assert profile_path.exists()
    assert manifest_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert metrics["row"] == "cnn"
    assert metrics["latent_channels"] == 32
    assert "MAE" in metrics["metrics"]
    assert profile["total_parameters"] > 0
    assert manifest["rows"]["cnn"]["c32"]["status"] == "completed"
    assert manifest["rows"]["cnn"]["c32"]["artifact_path"] == "rows/cnn/c32/metrics.json"


def test_inspect_split_samples_writes_one_real_sample_per_split(tmp_path: Path):
    output_path = tmp_path / "inspect.json"
    runner.inspect_split_samples(_tiny_dataloaders(), output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert set(payload["splits"]) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        entry = payload["splits"][split]
        assert entry["input_shape"] == [2, 1, 128, 128]
        assert entry["target_shape"] == [2, 1, 128, 128]
