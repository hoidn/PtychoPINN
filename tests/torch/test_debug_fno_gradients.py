import json
import subprocess
from pathlib import Path


def test_debug_fno_gradients_emits_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    cmd = [
        "python",
        "scripts/debug_fno_gradients.py",
        "--output",
        str(out_dir),
        "--channels",
        "32",
        "--modes",
        "12",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    report_path = out_dir / "gradient_report.json"
    assert report_path.exists(), "gradient_report.json not written"
    report = json.loads(report_path.read_text())
    assert "spectral_grad_mean" in report
    assert "local_grad_mean" in report
    assert "spectral_local_ratio" in report
