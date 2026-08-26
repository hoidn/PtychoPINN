"""Tests for the generic study metrics collator."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "studies" / "collate_study_metrics.py"


def _run(*args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True
    )


@pytest.mark.study
def test_collates_numeric_metrics_across_arms(tmp_path):
    for arm, ssim in [("a__x", 0.91), ("b__y", 0.55)]:
        mdir = tmp_path / arm / "reconstruction"
        mdir.mkdir(parents=True)
        (mdir / "metrics.json").write_text(
            json.dumps(
                {"amplitude_ssim": ssim, "notes": "text", "sub": {"phase_ssim": 0.5}}
            )
        )
    proc = _run(str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "arm\tamplitude_ssim\tsub.phase_ssim"
    assert lines[1] == "a__x\t0.91\t0.5"
    assert lines[2] == "b__y\t0.55\t0.5"


@pytest.mark.study
def test_empty_study_root_exits_nonzero(tmp_path):
    proc = _run(str(tmp_path))
    assert proc.returncode == 1
    assert "no reconstruction/metrics.json" in proc.stderr


@pytest.mark.study
def test_boolean_fields_excluded_from_columns(tmp_path):
    mdir = tmp_path / "a__x" / "reconstruction"
    mdir.mkdir(parents=True)
    (mdir / "metrics.json").write_text(
        json.dumps({"amplitude_ssim": 0.91, "converged": True, "failed": False})
    )
    proc = _run(str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "arm\tamplitude_ssim"
    assert lines[1] == "a__x\t0.91"
    assert "converged" not in lines[0]
    assert "failed" not in lines[0]


@pytest.mark.study
def test_custom_metrics_flag_with_nested_path(tmp_path):
    for arm, ssim in [("a__x", 0.91), ("b__y", 0.55)]:
        mdir = tmp_path / arm / "custom" / "deep" / "path"
        mdir.mkdir(parents=True)
        (mdir / "metrics.json").write_text(json.dumps({"amplitude_ssim": ssim}))
    proc = _run(str(tmp_path), "--metrics", "custom/deep/path/metrics.json")
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "arm\tamplitude_ssim"
    assert lines[1] == "a__x\t0.91"
    assert lines[2] == "b__y\t0.55"


@pytest.mark.study
def test_divergent_key_sets_across_arms_fill_empty_cells(tmp_path):
    mdir_a = tmp_path / "a__x" / "reconstruction"
    mdir_a.mkdir(parents=True)
    (mdir_a / "metrics.json").write_text(
        json.dumps({"amplitude_ssim": 0.91, "mae": 0.1})
    )
    mdir_b = tmp_path / "b__y" / "reconstruction"
    mdir_b.mkdir(parents=True)
    (mdir_b / "metrics.json").write_text(
        json.dumps({"amplitude_ssim": 0.55, "phase_ssim": 0.4})
    )
    proc = _run(str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "arm\tamplitude_ssim\tmae\tphase_ssim"
    assert lines[1] == "a__x\t0.91\t0.1\t"
    assert lines[2] == "b__y\t0.55\t\t0.4"
