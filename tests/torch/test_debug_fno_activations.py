import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def test_debug_fno_activations_emits_report(tmp_path: Path) -> None:
    """Smoke test: script writes a JSON report with expected keys."""
    npz_path = tmp_path / "tiny.npz"
    diffraction = np.random.rand(2, 16, 16).astype(np.float32)
    np.savez(npz_path, diffraction=diffraction)

    out_dir = tmp_path / "out"
    cmd = [
        "python",
        "scripts/debug_fno_activations.py",
        "--input",
        str(npz_path),
        "--output",
        str(out_dir),
        "--architecture",
        "fno",
        "--batch-size",
        "1",
        "--max-samples",
        "1",
        "--device",
        "cpu",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    report_path = out_dir / "activation_report.json"
    assert report_path.exists(), "activation_report.json not written"
    report = json.loads(report_path.read_text())
    assert "layers" in report
    assert "summary" in report


def test_debug_fno_activations_stable_hybrid_checkpoint(tmp_path: Path) -> None:
    """Test that script loads a stable_hybrid checkpoint and emits custom JSON name.

    Task ID: FNO-STABILITY-OVERHAUL-001 Phase 8 Task 2
    """
    from ptycho_torch.generators.fno import StableHybridUNOGenerator

    # Create a tiny synthetic checkpoint
    gen = StableHybridUNOGenerator(
        in_channels=1, out_channels=2, hidden_channels=32,
        n_blocks=4, modes=4, C=1,
    )
    ckpt_path = tmp_path / "model.pt"
    torch.save(gen.state_dict(), ckpt_path)

    # Create tiny NPZ
    npz_path = tmp_path / "tiny.npz"
    diffraction = np.random.rand(2, 16, 16).astype(np.float32)
    np.savez(npz_path, diffraction=diffraction)

    out_dir = tmp_path / "out"
    cmd = [
        "python",
        "scripts/debug_fno_activations.py",
        "--input", str(npz_path),
        "--output", str(out_dir),
        "--architecture", "stable_hybrid",
        "--checkpoint", str(ckpt_path),
        "--output-json-name", "activation_report_test.json",
        "--batch-size", "1",
        "--max-samples", "1",
        "--device", "cpu",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    report_path = out_dir / "activation_report_test.json"
    assert report_path.exists(), "Custom JSON filename not written"
    report = json.loads(report_path.read_text())
    assert report["summary"]["architecture"] == "stable_hybrid"
    assert "layers" in report
