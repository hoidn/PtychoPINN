"""Behavioral tests for the generic hydra study runner (ptycho_study).

Uses a fake runner script so no GPU or simulation is involved; the contract
under test is composition, arm.yaml self-containment, resume-by-sentinel,
failure propagation, log capture, and provenance.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

FAKE_RUNNER = '''\
#!/usr/bin/env python
"""Fake synthetic_pipeline stand-in: records calls, writes the sentinel."""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text())
root = Path(cfg["workflow"]["output_root"])
with (root / "calls.txt").open("a") as fh:
    fh.write(json.dumps({"cwd": os.getcwd(), "stages": cfg["workflow"]["stages"]}) + "\\n")
exit_code = int(os.environ.get("FAKE_RUNNER_EXIT", "0"))
if exit_code == 0:
    marker = root / "reconstruction"
    marker.mkdir(parents=True, exist_ok=True)
    (marker / "metrics.json").write_text(json.dumps({"amplitude_ssim": 0.9}))
print("fake runner ran")
sys.exit(exit_code)
'''


def _write_study(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal study conf + fake runner under tmp_path."""
    conf = tmp_path / "conf"
    conf.mkdir()
    out_root = tmp_path / "out"
    (tmp_path / "fake_runner.py").write_text(FAKE_RUNNER)
    (conf / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "defaults": ["_self_"],
                "study": {
                    "name": "fake_study",
                    "output_root": str(out_root),
                    "family_name": "famA",
                    "profile_name": "profB",
                    "runner_root": str(tmp_path),
                    "runner_script": "fake_runner.py",
                    "cuda_visible_devices": None,
                    "sentinel": "reconstruction/metrics.json",
                    "shared_datasets": False,
                },
                "profile": "fake-profile",
                "simulation": {"measurement_domain": "normalized_amplitude"},
                "workflow": {"stages": ["simulate", "train"]},
                "hydra": {"run": {"dir": str(out_root / "arm")}},
            },
            sort_keys=False,
        )
    )
    return conf, out_root


def _invoke(conf: Path, *overrides: str, extra_env: dict | None = None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ptycho.workflows.study_runner",
            "--config-dir",
            str(conf),
            "--config-name",
            "config",
            *overrides,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.study
def test_runs_runner_and_writes_self_contained_arm_config(tmp_path):
    conf, out_root = _write_study(tmp_path)
    proc = _invoke(conf)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    arm_dir = out_root / "arm"
    arm = yaml.safe_load((arm_dir / "arm.yaml").read_text())
    assert "study" not in arm
    assert arm["profile"] == "fake-profile"
    assert arm["workflow"]["output_root"] == str(arm_dir.resolve())
    call = json.loads((arm_dir / "calls.txt").read_text().splitlines()[0])
    assert call["cwd"] == str(tmp_path.resolve())  # subprocess cwd == runner_root
    assert (arm_dir / "reconstruction" / "metrics.json").exists()


@pytest.mark.study
def test_completed_arm_is_skipped_on_rerun(tmp_path):
    conf, out_root = _write_study(tmp_path)
    assert _invoke(conf).returncode == 0
    proc = _invoke(conf)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = (out_root / "arm" / "calls.txt").read_text().splitlines()
    assert len(calls) == 1  # runner not re-invoked
    assert "skip" in proc.stdout


@pytest.mark.study
def test_runner_failure_propagates_and_keeps_log(tmp_path):
    conf, out_root = _write_study(tmp_path)
    proc = _invoke(conf, extra_env={"FAKE_RUNNER_EXIT": "2"})
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "exit 2" in combined
    arm_dir = out_root / "arm"
    assert (arm_dir / "runner.log").read_text().strip() == "fake runner ran"
    assert not (arm_dir / "reconstruction" / "metrics.json").exists()


@pytest.mark.study
def test_provenance_records_overrides_and_runner_hash(tmp_path):
    conf, out_root = _write_study(tmp_path)
    proc = _invoke(conf, "workflow.stages=[train]")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    prov = json.loads((out_root / "arm" / "study_provenance.json").read_text())
    assert prov["study"] == "fake_study"
    assert "workflow.stages=[train]" in prov["overrides"]
    assert len(prov["runner_sha256"]) == 64
    assert prov["python"] == sys.executable


@pytest.mark.study
def test_shared_datasets_symlinks_and_drops_simulate_when_populated(tmp_path):
    conf, out_root = _write_study(tmp_path)
    on = "study.shared_datasets=true"
    proc = _invoke(conf, on, f"hydra.run.dir={out_root / 'armA'}")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    shared = out_root / "datasets_shared" / "famA__normalized_amplitude"
    assert (out_root / "armA" / "datasets").resolve() == shared.resolve()
    first_call = json.loads((out_root / "armA" / "calls.txt").read_text().splitlines()[0])
    assert first_call["stages"] == ["simulate", "train"]  # first arm simulates

    # Populate the shared manifest as a completed simulate stage would.
    (shared / "manifest.json").write_text("{}")

    proc = _invoke(conf, on, f"hydra.run.dir={out_root / 'armB'}")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    second_call = json.loads((out_root / "armB" / "calls.txt").read_text().splitlines()[0])
    assert second_call["stages"] == ["train"]  # simulate dropped, dataset reused
    arm_b = yaml.safe_load((out_root / "armB" / "arm.yaml").read_text())
    assert arm_b["workflow"]["stages"] == ["train"]  # arm.yaml records reality


@pytest.mark.study
def test_shared_datasets_fails_fast_when_sharing_key_missing(tmp_path):
    conf, out_root = _write_study(tmp_path)
    proc = _invoke(
        conf,
        "study.shared_datasets=true",
        "study.family_name=null",
        f"hydra.run.dir={out_root / 'armA'}",
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "study.family_name" in combined  # message names the missing key
    assert not (out_root / "datasets_shared").exists()  # no shared dir created
    link = out_root / "armA" / "datasets"
    assert not link.is_symlink() and not link.exists()  # no symlink created
