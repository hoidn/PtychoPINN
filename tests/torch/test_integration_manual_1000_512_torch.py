import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.torch]


def _run_command(command, cwd, log_path, env=None):
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


def _ensure_canonical_dataset(source_path: Path, output_path: Path) -> Path:
    if output_path.exists():
        return output_path

    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    with np.load(source_path) as data:
        if "diff3d" in data:
            diff3d = data["diff3d"]
        elif "diffraction" in data:
            diff3d = data["diffraction"]
        else:
            raise KeyError("Dataset missing required key 'diff3d' or legacy 'diffraction'")

        xcoords = data["xcoords"]
        ycoords = data["ycoords"]

        if diff3d.ndim == 3 and diff3d.shape[0] != len(xcoords) and diff3d.shape[-1] == len(xcoords):
            diff3d = np.moveaxis(diff3d, -1, 0)

        if diff3d.shape[0] != len(xcoords):
            raise ValueError(
                f"diff3d first dimension {diff3d.shape[0]} does not match xcoords length {len(xcoords)}"
            )

        xcoords_start = data["xcoords_start"] if "xcoords_start" in data else xcoords
        ycoords_start = data["ycoords_start"] if "ycoords_start" in data else ycoords
        scan_index = data["scan_index"] if "scan_index" in data else np.zeros(len(xcoords), dtype=int)

        def _cast_complex(arr):
            if arr is None:
                return None
            if np.iscomplexobj(arr) and arr.dtype != np.complex64:
                return arr.astype(np.complex64)
            return arr

        if "probeGuess" not in data:
            raise KeyError("Dataset missing required key 'probeGuess'")

        payload = {
            "xcoords": xcoords.astype(np.float32),
            "ycoords": ycoords.astype(np.float32),
            "xcoords_start": xcoords_start.astype(np.float32),
            "ycoords_start": ycoords_start.astype(np.float32),
            "diff3d": diff3d.astype(np.float32),
            "probeGuess": _cast_complex(data["probeGuess"]),
            "scan_index": scan_index.astype(np.int32),
        }

        object_guess = _cast_complex(data["objectGuess"] if "objectGuess" in data else None)
        if object_guess is not None:
            payload["objectGuess"] = object_guess

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


@pytest.mark.skipif(
    os.getenv("RUN_LONG_INTEGRATION") != "1",
    reason="Long-running integration test; set RUN_LONG_INTEGRATION=1 to enable.",
)
def test_train_infer_cycle_1000_train_512_test(tmp_path):
    try:
        import torch
    except ImportError:  # pragma: no cover - enforced by POLICY-001
        pytest.skip("PyTorch is required for long integration test")

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for long PyTorch integration test")

    repo_root = Path(__file__).resolve().parents[2]
    source_data = repo_root / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"
    canonical_path = (
        repo_root
        / ".artifacts"
        / "pytorch_integration_workflow"
        / "canonical"
        / "Run1084_recon3_postPC_shrunk_3_canonical.npz"
    )
    data_file = _ensure_canonical_dataset(source_data, canonical_path)

    train_dir = tmp_path / "training_outputs"
    infer_dir = tmp_path / "inference_outputs"
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    train_log = tmp_path / "train.log"
    train_command = [
        "python",
        "-m",
        "ptycho_torch.train",
        "--train_data_file",
        str(data_file),
        "--test_data_file",
        str(data_file),
        "--output_dir",
        str(train_dir),
        "--max_epochs",
        "20",
        "--n_images",
        "1000",
        "--gridsize",
        "1",
        "--batch_size",
        "16",
        "--accelerator",
        "cuda",
        "--disable_mlflow",
    ]
    train_result = _run_command(train_command, repo_root, train_log, env=env)
    assert train_result.returncode == 0, (
        "Training failed. See log: "
        f"{train_log}"
    )

    model_artifact_path = train_dir / "wts.h5.zip"
    assert model_artifact_path.exists()

    inference_log = tmp_path / "inference.log"
    inference_command = [
        "python",
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
    infer_result = _run_command(inference_command, repo_root, inference_log, env=env)
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
