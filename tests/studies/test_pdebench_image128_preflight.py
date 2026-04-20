import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def test_task_specs_define_required_pdebench_image128_tasks():
    from scripts.studies.pdebench_image128.task_specs import TASK_IDS, get_task_spec

    assert TASK_IDS == ("swe", "darcy", "2d_reacdiff")
    assert get_task_spec("swe").expected_filename == "2D_rdb_NA_NA.h5"
    assert get_task_spec("darcy").expected_filename == "2D_DarcyFlow_beta1.0_Train.hdf5"
    assert get_task_spec("2d_reacdiff").expected_filename == "2D_diff-react_NA_NA.h5"
    assert get_task_spec("darcy").task_type == "static_operator"
    assert get_task_spec("2d_reacdiff").listed_size_gb == 13.0


def test_preflight_detects_grouped_swe_native_128_layout(tmp_path):
    from scripts.studies.pdebench_image128.preflight import inspect_task_file

    data_file = tmp_path / "2D_rdb_NA_NA.h5"
    with h5py.File(data_file, "w") as handle:
        for trajectory_id in range(2):
            group = handle.create_group(f"{trajectory_id:04d}")
            group.create_dataset("data", data=np.zeros((3, 128, 128, 1), dtype=np.float32))

    payload = inspect_task_file(task_id="swe", data_file=data_file, compute_sha256=False)

    assert payload["status"] == "ready"
    assert payload["task_id"] == "swe"
    assert payload["is_native_128"] is True
    assert payload["native_spatial_shape"] == [128, 128]
    assert payload["layout"]["kind"] == "dynamic_state"
    assert payload["layout"]["state_dataset"]["path"] == "*/data"
    assert payload["layout"]["state_dataset"]["axis_order"] == "NTHWC"


def test_preflight_detects_darcy_static_operator_pair(tmp_path):
    from scripts.studies.pdebench_image128.preflight import inspect_task_file

    data_file = tmp_path / "2D_DarcyFlow_beta1.0_Train.hdf5"
    with h5py.File(data_file, "w") as handle:
        handle.create_dataset("a", data=np.zeros((4, 128, 128), dtype=np.float32))
        handle.create_dataset("u", data=np.ones((4, 128, 128), dtype=np.float32))

    payload = inspect_task_file(task_id="darcy", data_file=data_file, compute_sha256=False)

    assert payload["status"] == "ready"
    assert payload["is_native_128"] is True
    assert payload["layout"]["kind"] == "static_operator"
    assert payload["layout"]["input_dataset"]["path"] == "a"
    assert payload["layout"]["target_dataset"]["path"] == "u"
    assert payload["available_supervision_units"] == 4


def test_preflight_detects_official_darcy_nu_tensor_schema(tmp_path):
    from scripts.studies.pdebench_image128.preflight import inspect_task_file

    data_file = tmp_path / "2D_DarcyFlow_beta1.0_Train.hdf5"
    with h5py.File(data_file, "w") as handle:
        handle.create_dataset("nu", data=np.zeros((4, 128, 128), dtype=np.float32))
        handle.create_dataset("tensor", data=np.ones((4, 1, 128, 128), dtype=np.float32))
        handle.create_dataset("x-coordinate", data=np.arange(128, dtype=np.float32))
        handle.create_dataset("y-coordinate", data=np.arange(128, dtype=np.float32))

    payload = inspect_task_file(task_id="darcy", data_file=data_file, compute_sha256=False)

    assert payload["status"] == "ready"
    assert payload["is_native_128"] is True
    assert payload["layout"]["input_dataset"]["path"] == "nu"
    assert payload["layout"]["input_dataset"]["axis_order"] == "NHW"
    assert payload["layout"]["target_dataset"]["path"] == "tensor"
    assert payload["layout"]["target_dataset"]["axis_order"] == "NCHW"
    assert payload["available_supervision_units"] == 4


def test_preflight_detects_reaction_diffusion_dynamic_layout(tmp_path):
    from scripts.studies.pdebench_image128.preflight import inspect_task_file

    data_file = tmp_path / "2D_diff-react_NA_NA.h5"
    with h5py.File(data_file, "w") as handle:
        handle.create_dataset("data", data=np.zeros((5, 6, 128, 128, 2), dtype=np.float32))

    payload = inspect_task_file(task_id="2d_reacdiff", data_file=data_file, compute_sha256=False)

    assert payload["status"] == "ready"
    assert payload["is_native_128"] is True
    assert payload["native_spatial_shape"] == [128, 128]
    assert payload["layout"]["kind"] == "dynamic_state"
    assert payload["layout"]["state_dataset"]["path"] == "data"
    assert payload["available_supervision_units"] == 25


def test_suite_preflight_records_missing_task_blockers(tmp_path):
    from scripts.studies.pdebench_image128.preflight import build_suite_preflight

    swe_dir = tmp_path / "swe"
    swe_dir.mkdir()
    with h5py.File(swe_dir / "2D_rdb_NA_NA.h5", "w") as handle:
        for trajectory_id in range(2):
            group = handle.create_group(f"{trajectory_id:04d}")
            group.create_dataset("data", data=np.zeros((3, 128, 128, 1), dtype=np.float32))

    payload = build_suite_preflight(data_root=tmp_path, compute_sha256=False)

    statuses = {task["task_id"]: task["status"] for task in payload["tasks"]}
    assert statuses == {
        "swe": "ready",
        "darcy": "missing_file",
        "2d_reacdiff": "missing_file",
    }
    assert payload["ready_task_count"] == 1
    assert payload["all_tasks_ready"] is False
    assert payload["missing_listed_size_gb"] == 19.2
    assert payload["storage"]["data_root_available_bytes"] > 0
    blocker = next(task for task in payload["tasks"] if task["task_id"] == "darcy")
    assert blocker["blocker"]["reason"] == "missing_file"


def test_write_preflight_artifacts_persists_json_and_markdown(tmp_path):
    from scripts.studies.pdebench_image128.preflight import (
        build_suite_preflight,
        write_preflight_artifacts,
    )

    payload = build_suite_preflight(data_root=tmp_path, compute_sha256=False)
    json_path, markdown_path = write_preflight_artifacts(
        payload,
        output_root=tmp_path / "artifacts",
        markdown_path=tmp_path / "summary.md",
    )

    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert persisted["schema_version"] == "pdebench_image128_suite_preflight_v1"
    assert "PDEBench 128x128 Image-Suite Preflight" in markdown
    assert "darcy" in markdown
    assert "missing_file" in markdown


def test_preflight_markdown_includes_layout_shapes_for_ready_tasks(tmp_path):
    from scripts.studies.pdebench_image128.preflight import (
        build_suite_preflight,
        render_preflight_markdown,
    )

    swe_dir = tmp_path / "swe"
    swe_dir.mkdir()
    with h5py.File(swe_dir / "2D_rdb_NA_NA.h5", "w") as handle:
        for trajectory_id in range(2):
            group = handle.create_group(f"{trajectory_id:04d}")
            group.create_dataset("data", data=np.zeros((3, 128, 128, 1), dtype=np.float32))

    darcy_dir = tmp_path / "darcy"
    darcy_dir.mkdir()
    with h5py.File(darcy_dir / "2D_DarcyFlow_beta1.0_Train.hdf5", "w") as handle:
        handle.create_dataset("nu", data=np.zeros((4, 128, 128), dtype=np.float32))
        handle.create_dataset("tensor", data=np.ones((4, 1, 128, 128), dtype=np.float32))

    payload = build_suite_preflight(data_root=tmp_path, compute_sha256=False)
    markdown = render_preflight_markdown(payload)

    assert "## Dataset Schema Details" in markdown
    assert "`swe`: dynamic state `*/data`, shape `[2, 3, 128, 128, 1]`, axis `NTHWC`" in markdown
    assert (
        "`darcy`: static input `nu`, shape `[4, 128, 128]`, axis `NHW`; "
        "target `tensor`, shape `[4, 1, 128, 128]`, axis `NCHW`"
    ) in markdown


def test_preflight_cli_writes_requested_outputs(tmp_path):
    from scripts.studies.pdebench_image128.preflight import main

    output_root = tmp_path / "artifacts"
    markdown_path = tmp_path / "preflight.md"

    exit_code = main(
        [
            "--data-root",
            str(tmp_path),
            "--output-root",
            str(output_root),
            "--markdown-path",
            str(markdown_path),
            "--no-sha256",
        ]
    )

    assert exit_code == 0
    assert (output_root / "pdebench_image128_suite_preflight.json").exists()
    assert markdown_path.exists()


def test_preflight_script_runs_from_repo_root(tmp_path):
    output_root = tmp_path / "artifacts"
    markdown_path = tmp_path / "preflight.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/studies/run_pdebench_image128_suite.py",
            "--data-root",
            str(tmp_path),
            "--output-root",
            str(output_root),
            "--markdown-path",
            str(markdown_path),
            "--no-sha256",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "pdebench_image128_suite_preflight.json").exists()
    assert markdown_path.exists()
