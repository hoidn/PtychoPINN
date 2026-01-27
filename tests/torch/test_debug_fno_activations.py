import json
import subprocess
from pathlib import Path

import numpy as np


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
