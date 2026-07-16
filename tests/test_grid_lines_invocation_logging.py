import ast
import json
from pathlib import Path


def test_capture_runtime_provenance_includes_python_version_and_torch_block():
    from scripts.studies.invocation_logging import capture_runtime_provenance

    payload = capture_runtime_provenance()
    assert isinstance(payload.get("python_version"), str)
    assert payload["python_version"]
    torch_block = payload.get("torch")
    assert isinstance(torch_block, dict)
    for key in ("version", "cuda_version", "cuda_available", "device_name"):
        assert key in torch_block, key


def test_capture_neuralop_provenance_returns_required_keys():
    from scripts.studies.invocation_logging import capture_neuralop_provenance

    payload = capture_neuralop_provenance()
    for key in ("neuraloperator_package_version", "neuralop_module_version", "uno_signature"):
        assert key in payload, key


def test_get_git_dirty_returns_bool_or_none():
    from scripts.studies.invocation_logging import get_git_dirty

    value = get_git_dirty()
    assert value is None or isinstance(value, bool)


def test_write_invocation_artifacts_writes_json_and_shell(tmp_path):
    from scripts.studies.invocation_logging import write_invocation_artifacts

    out = tmp_path / "runs" / "pinn_fno"
    path_json, path_sh = write_invocation_artifacts(
        output_dir=out,
        script_path="scripts/studies/grid_lines_torch_runner.py",
        argv=["--output-dir", "outputs/x", "--architecture", "fno"],
        parsed_args={"output_dir": "outputs/x", "architecture": "fno"},
    )

    assert path_json.exists()
    assert path_sh.exists()

    payload = json.loads(path_json.read_text())
    assert payload["script"] == "scripts/studies/grid_lines_torch_runner.py"
    assert "--architecture" in payload["argv"]
    assert payload["cwd"]
    assert payload["timestamp_utc"]
    assert payload["command"].startswith("python scripts/studies/grid_lines_torch_runner.py")
    assert "fno" in path_sh.read_text()


def test_write_invocation_artifacts_preserves_runtime_provenance(tmp_path):
    from scripts.studies.invocation_logging import write_invocation_artifacts

    out = tmp_path / "runs" / "pinn_fno"
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
        parsed_args={"output_dir": Path("outputs/demo"), "architectures": ("cnn", "fno")},
    )
    payload = json.loads(json_path.read_text())
    assert payload["parsed_args"]["output_dir"] == "outputs/demo"
    assert payload["parsed_args"]["architectures"] == ["cnn", "fno"]


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


def test_package_invocation_writer_preserves_schema_and_path_serialization(tmp_path):
    from ptycho.invocation_logging import write_invocation_artifacts

    json_path, shell_path = write_invocation_artifacts(
        output_dir=tmp_path / "run",
        script_path="ptycho/workflows/grid_lines_workflow.py",
        argv=["--output-dir", str(tmp_path / "output")],
        parsed_args={"output_dir": Path("output/demo")},
        extra={"nested": {"path": Path("artifacts/model")}},
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) >= {
        "script",
        "argv",
        "command",
        "parsed_args",
        "cwd",
        "timestamp_utc",
        "pid",
        "extra",
    }
    assert payload["parsed_args"]["output_dir"] == "output/demo"
    assert payload["extra"]["nested"]["path"] == "artifacts/model"
    assert shell_path.read_text(encoding="utf-8") == payload["command"] + "\n"


def test_study_facade_reexports_package_invocation_functions_by_identity():
    from ptycho import invocation_logging as package_logging
    from scripts.studies import invocation_logging as study_logging

    for name in (
        "build_command_line",
        "capture_runtime_provenance",
        "get_git_commit",
        "get_git_dirty",
        "write_invocation_artifacts",
        "update_invocation_artifacts",
    ):
        assert getattr(study_logging, name) is getattr(package_logging, name)


def test_study_facade_retains_neuralop_specific_provenance():
    from scripts.studies import invocation_logging as study_logging

    assert callable(study_logging.capture_neuralop_provenance)
    assert not hasattr(__import__("ptycho.invocation_logging", fromlist=["*"]), "capture_neuralop_provenance")


def test_package_grid_lines_workflow_has_no_scripts_import_edge():
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / "ptycho"
        / "workflows"
        / "grid_lines_workflow.py"
    )
    tree = ast.parse(workflow_path.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert not any(
        module == "scripts" or module.startswith("scripts.")
        for module in imported_modules
    )
