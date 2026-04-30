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


def test_write_invocation_artifacts_preserves_runtime_provenance(tmp_path):
    from scripts.studies.invocation_logging import write_invocation_artifacts

    out = tmp_path / "runs" / "pinn_hybrid_resnet"
    runtime_provenance = {
        "python_executable": "/usr/bin/python3",
        "cwd": "/tmp/session_repo",
        "pythonpath": "/tmp/session_repo",
        "ptycho_torch_file": "/tmp/session_repo/ptycho_torch/__init__.py",
    }
    path_json, _ = write_invocation_artifacts(
        output_dir=out,
        script_path="scripts/studies/grid_lines_torch_runner.py",
        argv=["--output-dir", "outputs/x"],
        parsed_args={"output_dir": "outputs/x"},
        extra={"runtime_provenance": runtime_provenance},
    )

    payload = json.loads(path_json.read_text())
    assert payload["extra"]["runtime_provenance"]["pythonpath"] == "/tmp/session_repo"
    assert payload["extra"]["runtime_provenance"]["ptycho_torch_file"].endswith(
        "ptycho_torch/__init__.py"
    )


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


def test_write_invocation_artifacts_captures_tmux_launcher_env(tmp_path, monkeypatch):
    from scripts.studies.invocation_logging import write_invocation_artifacts

    monkeypatch.setenv("CODEX_TMUX_SESSION_NAME", "lines128-minimum-subset")
    monkeypatch.setenv("CODEX_TMUX_SOCKET_PATH", "/tmp/codex-lines128.sock")
    monkeypatch.setenv("CODEX_TMUX_ATTACH_COMMAND", "tmux -S /tmp/codex-lines128.sock attach -t lines128-minimum-subset")
    monkeypatch.setenv("CODEX_TMUX_CAPTURE_COMMAND", "tmux -S /tmp/codex-lines128.sock capture-pane -p -t lines128-minimum-subset:0.0")

    out = tmp_path / "root"
    json_path, _ = write_invocation_artifacts(
        output_dir=out,
        script_path="scripts/studies/lines128_paper_benchmark.py",
        argv=["--output-dir", str(tmp_path / "outputs")],
        parsed_args={"output_dir": str(tmp_path / "outputs")},
    )

    payload = json.loads(json_path.read_text())
    tmux_payload = payload["extra"]["tmux"]
    assert tmux_payload["session_name"] == "lines128-minimum-subset"
    assert tmux_payload["socket_path"] == "/tmp/codex-lines128.sock"
    assert "attach -t lines128-minimum-subset" in tmux_payload["attach_command"]
