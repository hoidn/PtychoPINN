import json
import os
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _subprocess_env() -> dict:
    """``python scripts/*.py`` below is a plain-script launch, so Python only
    auto-adds the script's own directory to sys.path, not the project root.
    Propagate PYTHONPATH explicitly so ``import ptycho``/``import ptycho_torch``
    resolve regardless of the caller's environment."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


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
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
        cwd=_PROJECT_ROOT, env=_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr

    report_path = out_dir / "gradient_report.json"
    assert report_path.exists(), "gradient_report.json not written"
    report = json.loads(report_path.read_text())
    assert "spectral_grad_mean" in report
    assert "local_grad_mean" in report
    assert "spectral_local_ratio" in report
