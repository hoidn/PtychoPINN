import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.torch]


def _run_command(command, cwd, log_path, env):
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
    return result


@pytest.mark.skipif(
    os.getenv("RUN_LONG_INTEGRATION") != "1",
    reason="Long-running integration test; set RUN_LONG_INTEGRATION=1 to enable.",
)
def test_train_infer_cycle_1000_train_512_test(tmp_path):
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available for long integration test.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for long PyTorch integration test.")

    repo_root = Path(__file__).resolve().parents[2]
    data_file = (
        repo_root
        / ".artifacts"
        / "pytorch_integration_workflow"
        / "canonical"
        / "Run1084_recon3_postPC_shrunk_3_canonical.npz"
    )
    if not data_file.exists():
        pytest.skip(f"Canonical dataset not found at {data_file}")
    train_dir = tmp_path / "training_outputs"
    infer_dir = tmp_path / "pytorch_output"
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    train_log = tmp_path / "train.log"
    train_command = [
        sys.executable,
        "-m",
        "ptycho_torch.train",
        "--train_data_file",
        str(data_file),
        "--test_data_file",
        str(data_file),
        "--output_dir",
        str(train_dir),
        "--max_epochs",
        "50",
        "--n_images",
        "1024",
        "--gridsize",
        "1",
        "--batch_size",
        "4",
        "--accelerator",
        "cuda",
        "--disable_mlflow",
    ]
    train_result = _run_command(train_command, repo_root, train_log, env)
    assert train_result.returncode == 0, (
        "Training failed. See log: "
        f"{train_log}"
    )
    model_artifact_path = train_dir / "wts.h5.zip"
    assert model_artifact_path.exists()

    inference_log = tmp_path / "inference.log"
    inference_command = [
        sys.executable,
        "-m",
        "ptycho_torch.inference",
        "--model_path",
        str(train_dir),
        "--test_data",
        str(data_file),
        "--output_dir",
        str(infer_dir),
        "--n_images",
        "512",
        "--accelerator",
        "cuda",
    ]
    infer_result = _run_command(inference_command, repo_root, inference_log, env)
    assert infer_result.returncode == 0, (
        "Inference failed. See log: "
        f"{inference_log}"
    )

    recon_amp_path = infer_dir / "reconstructed_amplitude.png"
    recon_phase_path = infer_dir / "reconstructed_phase.png"
    assert recon_amp_path.exists()
    assert recon_phase_path.exists()
    assert recon_amp_path.stat().st_size > 1000
    assert recon_phase_path.stat().st_size > 1000
