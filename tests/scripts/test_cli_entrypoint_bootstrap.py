import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("script_relpath", "required_help_text"),
    [
        ("scripts/training/train.py", "--train_data_file"),
        ("scripts/inference/inference.py", "--model_path"),
    ],
)
def test_cli_entrypoint_bootstraps_repo_root_without_pythonpath(
    script_relpath: str,
    required_help_text: str,
) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    script_path = PROJECT_ROOT / script_relpath

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert required_help_text in result.stdout
