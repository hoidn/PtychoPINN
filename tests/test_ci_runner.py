"""Check local and hosted CI commands without importing or running model tests."""

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def test_ci_runner_worker_policy():
    repo = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as directory:
        fake_python = Path(directory) / "python"
        fake_python.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$@"\n'
        )
        fake_python.chmod(0o755)
        env = dict(os.environ, PATH=f"{directory}:{os.environ['PATH']}")
        env.pop("GITHUB_ACTIONS", None)
        for hosted in (False, True):
            if hosted:
                env["GITHUB_ACTIONS"] = "true"
            result = subprocess.run(
                ["bash", str(repo / "ci/run_ci_tests.sh"), "-k", "worker policy"],
                env=env, check=True, capture_output=True, text=True,
            )
            cuda, *args = result.stdout.splitlines()
            assert cuda == ""
            assert args[:4] == ["-m", "pytest", "tests/torch", "-m"]
            assert args[-2:] == ["-k", "worker policy"]
            if hosted:
                assert "-n" not in args
            else:
                assert "-n" in args, "Local CI must configure eight workers"
                assert args[args.index("-n") + 1] == "8"
                assert args[args.index("--dist") + 1] == "loadfile"


if __name__ == "__main__":
    test_ci_runner_worker_policy()
