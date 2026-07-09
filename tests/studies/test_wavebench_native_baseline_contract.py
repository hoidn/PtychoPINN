import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/studies/run_wavebench_native_baselines.py"
VALIDATOR_PATH = REPO_ROOT / "scripts/studies/validate_wavebench_native_baseline_contract.py"


def test_wavebench_native_baseline_contract_loader_uses_locked_inputs():
    namespace = {"__file__": str(SCRIPT_PATH)}
    exec(SCRIPT_PATH.read_text(encoding="utf-8"), namespace)

    contract = namespace["load_contract"](REPO_ROOT)
    choose_eval_batch_size = namespace["choose_eval_batch_size"]

    assert contract["selected_variant"] == "time_varying/is/thick_lines_gaussian_lens"
    assert (
        contract["selected_dataset_member"]
        == "wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton"
    )
    assert contract["stable_dataset_target"]["repo_relative"] == "wavebench_dataset/time_varying/is/"
    assert contract["native_rows"]["unet"]["route"] == "checkpoint_reusable"
    assert contract["native_rows"]["fno"]["route"] == "retrain_required"
    assert contract["split"]["train"] == 9000
    assert contract["split"]["val"] == 500
    assert contract["split"]["test"] == 500
    assert choose_eval_batch_size(500, 32) == 25
    assert choose_eval_batch_size(500, 25) == 25


def test_wavebench_native_baseline_validator_accepts_consistent_bundle(tmp_path: Path):
    repo_root = tmp_path / "repo"
    output_root = (
        repo_root
        / ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-native-baseline-reproduction"
    )
    docs_root = repo_root / "docs/plans/NEURIPS-HYBRID-RESNET-2026"
    docs_root.mkdir(parents=True)
    output_root.mkdir(parents=True)

    summary_path = docs_root / "wavebench_native_baseline_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# NeurIPS WaveBench Native Baseline Summary",
                "",
                "- Selected variant: `time_varying/is/thick_lines_gaussian_lens`",
                "- Stable dataset target: `<wavebench repo>/wavebench_dataset/time_varying/is/`",
                "- Native rows remain candidate-lane external references only.",
                "",
                "## Native Rows",
                "",
                "- `wavebench_unet_ch32_native`: `completed` on the full `500`-sample test split.",
                "- `wavebench_fno_depth4_native`: `blocked` after the official retraining route remained unavailable.",
                "",
                "## Residual Risks",
                "",
                "- WaveBench remains outside the manuscript evidence bundle.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (repo_root / "docs/index.md").write_text(
        "wavebench_native_baseline_summary.md\n", encoding="utf-8"
    )
    (docs_root / "evidence_matrix.md").write_text(
        "wavebench_native_baseline_summary.md\n", encoding="utf-8"
    )

    manifest = {
        "selected_variant": "time_varying/is/thick_lines_gaussian_lens",
        "selected_dataset_member": "wavebench_dataset/time_varying/is/thick_lines_gaussian_lens.beton",
        "stable_dataset_target": {
            "repo_relative": "wavebench_dataset/time_varying/is/",
            "description": "<wavebench repo>/wavebench_dataset/time_varying/is/",
        },
        "split": {"train": 9000, "val": 500, "test": 500, "seed": 42},
        "environment": {
            "python_executable": "/tmp/fake/python",
            "python_version": "3.11.0",
        },
        "wavebench_checkout": {
            "repo_relative": "tmp/wavebench_repo",
            "revision": "2bea258d9f05ec7182741293be11be1e545576ae",
        },
        "native_rows": {
            "unet": {
                "row_id": "wavebench_unet_ch32_native",
                "status": "completed",
                "route": "checkpoint_reusable",
                "artifact_path": "native_unet_eval.json",
            },
            "fno": {
                "row_id": "wavebench_fno_depth4_native",
                "status": "blocked",
                "route": "retrain_required",
                "artifact_path": "native_fno_result.json",
            },
        },
        "authoritative_artifacts": {
            "summary": "docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_native_baseline_summary.md",
            "table_ready_metrics": ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-native-baseline-reproduction/table_ready_metrics.json",
        },
    }
    (output_root / "native_baseline_execution_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    unet_row = {
        "row_id": "wavebench_unet_ch32_native",
        "family": "unet-ch-32",
        "status": "completed",
        "selected_variant": "time_varying/is/thick_lines_gaussian_lens",
        "split": {"test_samples": 500, "seed": 42},
        "metrics": {
            "MAE": 0.1,
            "RMSE": 0.2,
            "RelL2": 0.3,
            "SSIM": 0.4,
        },
    }
    (output_root / "native_unet_eval.json").write_text(
        json.dumps(unet_row, indent=2) + "\n",
        encoding="utf-8",
    )

    fno_row = {
        "row_id": "wavebench_fno_depth4_native",
        "family": "fno-depth-4",
        "status": "blocked",
        "selected_variant": "time_varying/is/thick_lines_gaussian_lens",
        "split": {"test_samples": 500, "seed": 42},
        "blocker_reason": "training environment missing required dependency",
    }
    (output_root / "native_fno_result.json").write_text(
        json.dumps(fno_row, indent=2) + "\n",
        encoding="utf-8",
    )

    table_ready_metrics = {
        "selected_variant": "time_varying/is/thick_lines_gaussian_lens",
        "rows": [unet_row, fno_row],
    }
    (output_root / "table_ready_metrics.json").write_text(
        json.dumps(table_ready_metrics, indent=2) + "\n",
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
