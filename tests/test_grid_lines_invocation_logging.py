import json


def test_write_invocation_artifacts_writes_json_and_shell(tmp_path):
    from scripts.studies.invocation_logging import write_invocation_artifacts

    out = tmp_path / "runs" / "pinn_hybrid_resnet"
    path_json, path_sh = write_invocation_artifacts(
        output_dir=out,
        script_path="scripts/studies/grid_lines_torch_runner.py",
        argv=["--output-dir", "outputs/x", "--architecture", "hybrid_resnet"],
        parsed_args={"output_dir": "outputs/x", "architecture": "hybrid_resnet"},
    )

    assert path_json.exists()
    assert path_sh.exists()

    payload = json.loads(path_json.read_text())
    assert payload["script"] == "scripts/studies/grid_lines_torch_runner.py"
    assert "--architecture" in payload["argv"]
    assert payload["cwd"]
    assert payload["timestamp_utc"]
    assert payload["command"].startswith("python scripts/studies/grid_lines_torch_runner.py")
    assert "hybrid_resnet" in path_sh.read_text()


def test_write_invocation_artifacts_serializes_paths(tmp_path):
    from pathlib import Path

    from scripts.studies.invocation_logging import write_invocation_artifacts

    out = tmp_path / "root"
    json_path, _ = write_invocation_artifacts(
        output_dir=out,
        script_path="scripts/studies/grid_lines_compare_wrapper.py",
        argv=["--output-dir", str(tmp_path / "outputs")],
        parsed_args={"output_dir": Path("outputs/demo"), "architectures": ("cnn", "hybrid_resnet")},
    )
    payload = json.loads(json_path.read_text())
    assert payload["parsed_args"]["output_dir"] == "outputs/demo"
    assert payload["parsed_args"]["architectures"] == ["cnn", "hybrid_resnet"]

