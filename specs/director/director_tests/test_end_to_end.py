import json
import subprocess
import sys
import yaml
from pathlib import Path

def test_end_to_end(tmp_path):
    # Create dummy editable and read-only files that will be used as context.
    editable_file = tmp_path / "fake_editable.py"
    editable_file.write_text("# fake editable content")
    read_only_file = tmp_path / "fake_read_only.py"
    read_only_file.write_text("# fake read-only content")

    # Create a temporary director config file.
    # Setting max_iterations to 0 avoids further LLM calls in the loop.
    config_data = {
        "prompt": "dummy prompt",
        "coder_model": "o3-mini",
        "evaluator_model": "o3-mini",
        "max_iterations": 0,
        "execution_command": "echo dummy",
        "context_editable": [str(editable_file)],  # initial values (will be overridden)
        "context_read_only": [str(read_only_file)],
        "evaluator": "default"
    }
    config_file = tmp_path / "director_config.yaml"
    config_file.write_text(yaml.dump(config_data))

    # Simulate dependency_detector output by writing temporary JSON files.
    # The director expects plain JSON arrays for both CLI overrides.
    editable_dep = json.dumps([str(editable_file)])
    read_only_dep = json.dumps([str(read_only_file)])
    dep_editable_file = tmp_path / "dep_editable.json"
    dep_editable_file.write_text(editable_dep)
    dep_read_only_file = tmp_path / "dep_read_only.json"
    dep_read_only_file.write_text(read_only_dep)

    # Run director.py via subprocess (it prints the loaded configuration on startup).
    # Adjust the path to director.py as needed relative to the current working directory.
    director_py = str(Path("director.py"))

    cmd = [
        sys.executable, director_py,
        "--config", str(config_file),
        "--context-editable", str(dep_editable_file),
        "--context-read-only", str(dep_read_only_file)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Assert that director.py ran successfully.
    assert result.returncode == 0, f"director.py failed. stderr: {result.stderr}"

    # Check that the director printed a line with the loaded configuration.
    # The Director.main() prints a line like:
    # "Loaded configuration: <config dict>"
    loaded_config_line = None
    for line in result.stdout.splitlines():
        if line.startswith("Loaded configuration:"):
            loaded_config_line = line
            break
    assert loaded_config_line, "Loaded configuration not found in director.py output"

    # Option 1: Use substring matching to ensure our filepaths appear.
    assert str(editable_file) in loaded_config_line, "Editable file not found in loaded configuration"
    assert str(read_only_file) in loaded_config_line, "Read-only file not found in loaded configuration"

    # Option 2 (alternative): Attempt to extract the printed dict.
    # Note: The printed format is created via f-string of a pydantic model so it might be eval()-able.
    try:
        config_str = loaded_config_line.replace("Loaded configuration:", "").strip()
        loaded_config = eval(config_str)
    except Exception as e:
        raise AssertionError(f"Failed to parse loaded configuration: {loaded_config_line}. Error: {e}")

    # Verify that the loaded configuration uses the CLI overrides provided by dependency_detector.
    assert loaded_config.get("context_editable") == [str(editable_file)]
    assert loaded_config.get("context_read_only") == [str(read_only_file)]
