import json
import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.tf_integration]

_MAX_VAL_INTENSITY_SCALER_INV_LOSS = 50.0
_METRIC_PATTERN = r"{metric}:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


def _run_command(command, cwd, log_path):
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return result


def _resolve_output_root(tmp_path):
    output_root = os.getenv("INTEGRATION_OUTPUT_DIR")
    if output_root:
        return Path(output_root).expanduser()
    return tmp_path


def _extract_last_metric(log_text, metric_name):
    sanitized = log_text.replace("\r", "\n")
    pattern = re.compile(_METRIC_PATTERN.format(metric=re.escape(metric_name)))
    last_value = None
    for match in pattern.finditer(sanitized):
        try:
            last_value = float(match.group(1))
        except ValueError:
            continue
    return last_value


@pytest.mark.skipif(
    os.getenv("RUN_LONG_INTEGRATION") != "1",
    reason="Long-running integration test; set RUN_LONG_INTEGRATION=1 to enable.",
)
def test_train_infer_cycle_1000_train_512_test(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    data_file = repo_root / "ptycho" / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"
    output_root = _resolve_output_root(tmp_path)
    output_root.mkdir(parents=True, exist_ok=True)
    train_dir = output_root / "training_outputs"
    infer_dir = output_root / "inference_outputs"
    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    train_log = output_root / "train.log"
    train_command = [
        "python",
        str(repo_root / "scripts" / "training" / "train.py"),
        "--train_data_file",
        str(data_file),
        "--test_data_file",
        str(data_file),
        "--output_dir",
        str(train_dir),
        "--nepochs",
        "50",
        "--n_images",
        "1000",
        "--gridsize",
        "1",
    ]
    train_result = _run_command(train_command, repo_root, train_log)
    assert train_result.returncode == 0, (
        "Training failed. See log: "
        f"{train_log}"
    )
    train_log_text = train_log.read_text(encoding="utf-8")
    assert "subsampling 1000 images" in train_log_text
    model_artifact_path = train_dir / "wts.h5.zip"
    assert model_artifact_path.exists()
    train_loss = _extract_last_metric(train_log_text, "intensity_scaler_inv_loss")
    val_loss = _extract_last_metric(train_log_text, "val_intensity_scaler_inv_loss")
    assert train_loss is not None, (
        "Missing intensity_scaler_inv_loss in training log. "
        f"See log: {train_log}"
    )
    assert val_loss is not None, (
        "Missing val_intensity_scaler_inv_loss in training log. "
        f"See log: {train_log}"
    )
    metrics_path = output_root / "train_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "intensity_scaler_inv_loss": train_loss,
                "val_intensity_scaler_inv_loss": val_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert val_loss <= _MAX_VAL_INTENSITY_SCALER_INV_LOSS, (
        "val_intensity_scaler_inv_loss exceeded threshold. "
        f"Got {val_loss:.4f}, max {_MAX_VAL_INTENSITY_SCALER_INV_LOSS:.1f}. "
        f"See log: {train_log}"
    )

    inference_log = output_root / "inference.log"
    inference_command = [
        "python",
        str(repo_root / "scripts" / "inference" / "inference.py"),
        "--model_path",
        str(train_dir),
        "--test_data",
        str(data_file),
        "--output_dir",
        str(infer_dir),
        "--n_images",
        "512",
    ]
    infer_result = _run_command(inference_command, repo_root, inference_log)
    assert infer_result.returncode == 0, (
        "Inference failed. See log: "
        f"{inference_log}"
    )
    inference_log_text = inference_log.read_text(encoding="utf-8")
    assert "using 512 individual images" in inference_log_text
    assert "subsampling 512 images" in inference_log_text

    recon_amp_path = infer_dir / "reconstructed_amplitude.png"
    recon_phase_path = infer_dir / "reconstructed_phase.png"
    assert recon_amp_path.exists()
    assert recon_phase_path.exists()
    assert recon_amp_path.stat().st_size > 1000
    assert recon_phase_path.stat().st_size > 1000
