import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_wavebench_preflight_contract_validator_passes():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/studies/validate_wavebench_preflight_contract.py"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
