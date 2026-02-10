import json
from pathlib import Path

import h5py
import numpy as np


def _write_valid_layout(root: Path) -> None:
    (root / "recons" / "pinn_ptychovit").mkdir(parents=True, exist_ok=True)
    (root / "recons" / "gt").mkdir(parents=True, exist_ok=True)
    (root / "visuals").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "pinn_ptychovit").mkdir(parents=True, exist_ok=True)

    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, 16), np.linspace(0.0, 1.0, 16), indexing="ij")
    amp = 1.0 + 0.05 * (yy + xx)
    phase = 0.2 * (yy - xx)
    arr = (amp * np.exp(1j * phase)).astype(np.complex64)
    np.savez(root / "recons" / "pinn_ptychovit" / "recon.npz", YY_pred=arr, amp=np.abs(arr), phase=np.angle(arr))
    np.savez(root / "recons" / "gt" / "recon.npz", YY_pred=arr, amp=np.abs(arr), phase=np.angle(arr))

    (root / "visuals" / "amp_phase_pinn_ptychovit.png").write_bytes(b"stub")
    (root / "visuals" / "compare_amp_phase.png").write_bytes(b"stub")

    metrics = {
        "pinn_ptychovit": {
            "reference_shape": [16, 16],
            "metrics": {
                "mae": [0.01],
                "mse": [0.001],
                "psnr": [70.0],
                "ssim": [0.95],
                "ms_ssim": [0.94],
                "frc50": [10.0],
            },
        }
    }
    (root / "metrics_by_model.json").write_text(json.dumps(metrics, indent=2))

    checkpoint = root / "runs" / "pinn_ptychovit" / "best_model.pth"
    checkpoint.write_bytes(b"stub")

    manifest = {
        "mode": "inference",
        "checkpoint": str(checkpoint),
        "training_returncode": None,
    }
    (root / "runs" / "pinn_ptychovit" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "runs" / "pinn_ptychovit" / "stdout.log").write_text("fresh execution\n")

    bridge_data_dir = root / "runs" / "pinn_ptychovit" / "bridge_work" / "data"
    bridge_data_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(bridge_data_dir / "test_dp.hdf5", "w") as handle:
        handle.create_dataset("dp", data=np.ones((3, 4, 4), dtype=np.float32))
    with h5py.File(bridge_data_dir / "test_para.hdf5", "w") as handle:
        obj = handle.create_dataset("object", data=np.ones((1, 16, 16), dtype=np.complex64))
        obj.attrs["pixel_height_m"] = 1.0
        obj.attrs["pixel_width_m"] = 1.0
        handle.create_dataset("probe_position_x_m", data=np.array([0.0, 1.0, 2.0], dtype=np.float64))
        handle.create_dataset("probe_position_y_m", data=np.array([0.0, -1.0, -2.0], dtype=np.float64))


def test_verifier_requires_metrics_and_recon_and_visuals(tmp_path: Path):
    from scripts.studies.verify_fresh_ptychovit_initial_metrics import main

    rc = main(["--output-dir", str(tmp_path)])
    assert rc == 1


def test_verifier_fails_if_stdout_contains_skipped_backend_execution(tmp_path: Path):
    from scripts.studies.verify_fresh_ptychovit_initial_metrics import main

    _write_valid_layout(tmp_path)
    (tmp_path / "runs" / "pinn_ptychovit" / "stdout.log").write_text(
        "Skipped backend execution; reused existing reconstruction artifact.\n"
    )

    rc = main(["--output-dir", str(tmp_path)])
    assert rc == 1


def test_verifier_fails_if_manifest_indicates_training_bootstrap(tmp_path: Path):
    from scripts.studies.verify_fresh_ptychovit_initial_metrics import main

    _write_valid_layout(tmp_path)
    manifest_path = tmp_path / "runs" / "pinn_ptychovit" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_returncode"] = 0
    manifest_path.write_text(json.dumps(manifest, indent=2))

    rc = main(["--output-dir", str(tmp_path)])
    assert rc == 1


def test_verifier_passes_for_fresh_checkpoint_restored_run(tmp_path: Path):
    from scripts.studies.verify_fresh_ptychovit_initial_metrics import main

    _write_valid_layout(tmp_path)
    rc = main(["--output-dir", str(tmp_path)])
    assert rc == 0


def test_verifier_accepts_nested_numeric_frc_payload(tmp_path: Path):
    from scripts.studies.verify_fresh_ptychovit_initial_metrics import main

    _write_valid_layout(tmp_path)
    metrics_path = tmp_path / "metrics_by_model.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["pinn_ptychovit"]["metrics"]["frc"] = [[1.0, 0.5, 0.25], [1.0, 0.3, 0.1]]
    metrics_path.write_text(json.dumps(metrics, indent=2))

    rc = main(["--output-dir", str(tmp_path)])
    assert rc == 0


def test_verifier_fails_when_probe_positions_constant_zero(tmp_path: Path):
    from scripts.studies.verify_fresh_ptychovit_initial_metrics import main

    _write_valid_layout(tmp_path)
    para_path = tmp_path / "runs" / "pinn_ptychovit" / "bridge_work" / "data" / "test_para.hdf5"
    with h5py.File(para_path, "a") as handle:
        del handle["probe_position_x_m"]
        del handle["probe_position_y_m"]
        handle.create_dataset("probe_position_x_m", data=np.zeros(3, dtype=np.float64))
        handle.create_dataset("probe_position_y_m", data=np.zeros(3, dtype=np.float64))

    rc = main(["--output-dir", str(tmp_path)])
    assert rc == 1


def test_verifier_fails_when_stdout_contains_normalization_fallback_warning(tmp_path: Path):
    from scripts.studies.verify_fresh_ptychovit_initial_metrics import main

    _write_valid_layout(tmp_path)
    (tmp_path / "runs" / "pinn_ptychovit" / "stdout.log").write_text(
        "Warning: Normalization file not found at /tmp/fake/normalization.pkl. Using default: 100000.0\n"
    )

    rc = main(["--output-dir", str(tmp_path)])
    assert rc == 1


def test_verifier_fails_when_recon_covered_region_is_nearly_constant(tmp_path: Path):
    from scripts.studies.verify_fresh_ptychovit_initial_metrics import verify_output

    _write_valid_layout(tmp_path)
    constant = np.ones((16, 16), dtype=np.complex64)
    np.savez(
        tmp_path / "recons" / "pinn_ptychovit" / "recon.npz",
        YY_pred=constant,
        amp=np.abs(constant),
        phase=np.angle(constant),
    )

    errors = verify_output(tmp_path, allow_external_checkpoint=False)
    assert any("covered-region amplitude std" in err for err in errors)
