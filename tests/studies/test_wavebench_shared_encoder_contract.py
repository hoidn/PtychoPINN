import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "scripts/studies/run_wavebench_shared_encoder_benchmark.py"
VALIDATOR_PATH = REPO_ROOT / "scripts/studies/validate_wavebench_shared_encoder_contract.py"


def test_row_contract_locks_roster_recipe_and_latent_widths():
    namespace = {"__file__": str(RUNNER_PATH)}
    exec(RUNNER_PATH.read_text(encoding="utf-8"), namespace)

    contract = namespace["build_row_contract"](
        repo_root=REPO_ROOT,
        wavebench_root=REPO_ROOT / "tmp/wavebench_repo",
    )

    assert contract["selected_variant"] == "time_varying/is/thick_lines_gaussian_lens"
    assert contract["split"] == {"train": 9000, "val": 500, "test": 500, "seed": 42}
    assert contract["latent_channel_settings"] == [32, 64]
    assert contract["rows"] == [
        "cnn",
        "hybrid_resnet",
        "spectral_resnet_bottleneck_net",
        "fno",
        "ffno",
    ]
    assert contract["training_recipe"]["loss"] == "l1"
    assert contract["training_recipe"]["optimizer"] == {"name": "adam", "lr": 2e-4}
    assert contract["training_recipe"]["scheduler"]["name"] == "ReduceLROnPlateau"


def test_shared_encoder_validator_accepts_consistent_bundle(tmp_path: Path):
    repo_root = tmp_path / "repo"
    output_root = (
        repo_root
        / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark"
    )
    docs_root = repo_root / "docs/plans/NEURIPS-HYBRID-RESNET-2026"
    docs_root.mkdir(parents=True)
    output_root.mkdir(parents=True)
    (output_root / "rows" / "cnn" / "c32").mkdir(parents=True)

    row_contract = {
        "selected_variant": "time_varying/is/thick_lines_gaussian_lens",
        "selected_dataset_member": "wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton",
        "stable_dataset_target": {
            "repo_relative": "wavebench_dataset/time_varying/is/",
            "description": "<wavebench repo>/wavebench_dataset/time_varying/is/",
        },
        "split": {"train": 9000, "val": 500, "test": 500, "seed": 42},
        "latent_channel_settings": [32, 64],
        "rows": [
            "cnn",
            "hybrid_resnet",
            "spectral_resnet_bottleneck_net",
            "fno",
            "ffno",
        ],
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
        },
        "row_status_values": ["completed", "blocked", "not_protocol_compatible"],
    }
    (output_root / "row_contract.json").write_text(
        json.dumps(row_contract, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "selected_variant": row_contract["selected_variant"],
        "selected_dataset_member": row_contract["selected_dataset_member"],
        "stable_dataset_target": row_contract["stable_dataset_target"],
        "authoritative_artifacts": {
            "row_contract": ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/row_contract.json",
        },
        "rows": {
            "cnn": {
                "c32": {
                    "status": "completed",
                    "artifact_path": "rows/cnn/c32/metrics.json",
                }
            }
        },
    }
    (output_root / "shared_encoder_execution_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics = {
        "row": "cnn",
        "latent_channels": 32,
        "mode": "benchmark",
        "status": "completed",
        "metrics": {"MAE": 0.1, "RMSE": 0.2, "RelL2": 0.3, "SSIM": 0.4},
    }
    (output_root / "rows" / "cnn" / "c32" / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--repo-root",
            str(repo_root),
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
