import json
import yaml
from pathlib import Path
import pytest
from director import Director, main

def create_temp_config(tmp_path, editable_list, readonly_list=None):
    if readonly_list is None:
        readonly_list = []
    config_data = {
        "prompt": "Dummy prompt",
        "coder_model": "dummy",
        "evaluator_model": "o1-mini",
        "max_iterations": 1,
        "execution_command": "echo test",
        "context_editable": editable_list,
        "context_read_only": readonly_list,
        "evaluator": "default"
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))
    return config_file

def test_director_cli_context_editable_override(tmp_path):
    # Create a temporary config file with a default context_editable list.
    default_editable = [str(tmp_path / "default.py")]
    config_file = create_temp_config(tmp_path, editable_list=default_editable)
    
    # Create dummy default file so the existence check passes.
    (tmp_path / "default.py").write_text("default content")
    
    # Create override files.
    override_files = [str(tmp_path / "override1.py"), str(tmp_path / "override2.py")]
    for f in override_files:
        Path(f).write_text("override content")
    
    # Instantiate Director with the CLI override.
    director = Director(str(config_file), template_values={}, cli_context_editable=override_files)
    
    # The config's context_editable should now equal the CLI provided list.
    assert director.config.context_editable == override_files

def test_main_context_editable_override_with_file(tmp_path, monkeypatch):
    import sys
    # Prepare a temporary YAML config file with default values.
    default_editable = [str(tmp_path / "default.py")]
    config_file = create_temp_config(tmp_path, editable_list=default_editable)
    (tmp_path / "default.py").write_text("default content")
    
    # Create override files.
    override_files = [str(tmp_path / "override1.py"), str(tmp_path / "override2.py")]
    for f in override_files:
        Path(f).write_text("override content")
    
    # Write the override list as JSON into a temporary file.
    override_json_path = tmp_path / "override.json"
    override_json_path.write_text(json.dumps(override_files))
    
    # Simulate CLI arguments where --context-editable points to the override JSON file.
    monkeypatch.setattr(sys, "argv", [
         "director.py",
         "--config", str(config_file),
         "--context-editable", str(override_json_path)
    ])
    
    # Prevent the full execution loop by monkey-patching Director.direct.
    monkeypatch.setattr(Director, "direct", lambda self: None)
    
    # Capture the cli_context_editable value passed into Director.__init__
    captured = {}
    orig_init = Director.__init__
    def fake_init(self, config_path, template_values=None, cli_context_editable=None):
         captured["cli_context_editable"] = cli_context_editable
         orig_init(self, config_path, template_values, cli_context_editable)
    monkeypatch.setattr(Director, "__init__", fake_init)
    
    # Call main() to process the CLI arguments.
    main()
    
    # Verify that the CLI override (parsed from the file) is correctly provided.
    assert captured.get("cli_context_editable") == override_files
