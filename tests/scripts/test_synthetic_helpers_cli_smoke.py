import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def _run_help(script_path: Path) -> None:
    result = subprocess.run(
        ["python", str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_dose_response_study_help():
    _run_help(project_root / "scripts" / "studies" / "dose_response_study.py")


def test_sim_lines_gs1_ideal_help():
    _run_help(project_root / "scripts" / "studies" / "sim_lines_4x" / "run_gs1_ideal.py")


def test_run_with_synthetic_lines_help():
    _run_help(project_root / "scripts" / "simulation" / "run_with_synthetic_lines.py")


def test_sim_lines_pipeline_import_smoke():
    from scripts.studies.sim_lines_4x import pipeline

    total_images, train_count, test_count = pipeline.derive_counts(
        pipeline.RunParams(),
        gridsize=1,
    )
    assert total_images == train_count + test_count
