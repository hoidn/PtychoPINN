import pytest
import json
from pathlib import Path
from director import (
    Director, 
    TemplateError, 
    TemplateSyntaxError, 
    TemplateValueError,
    DirectorConfig
)

@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config file with template support"""
    config_content = """
prompt: |
    Hello {{ name }}!
    Details: {{ details }}
coder_model: claude-3-5-haiku-20241022
evaluator_model: gpt-4o
max_iterations: 5
execution_command: echo "test"
context_editable:
    - test.py
context_read_only: []
evaluator: default
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    
    test_file = tmp_path / "test.py"
    test_file.write_text("# Test file")
    
    return config_file

@pytest.fixture
def temp_config_with_values(tmp_path):
    """Create a config file with template values defined"""
    config_content = """
prompt: |
    Hello {{ name }}!
    Details: {{ details }}
coder_model: claude-3-5-haiku-20241022
evaluator_model: gpt-4o
max_iterations: 5
execution_command: echo "test"
context_editable:
    - test.py
context_read_only: []
evaluator: default
template_values:
    name: ConfigName
    details: ConfigDetails
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    
    test_file = tmp_path / "test.py"
    test_file.write_text("# Test file")
    
    return config_file

class TestConfigValidation:
    """Tests for configuration loading and validation"""
    
    def test_missing_config_file(self):
        """Test error when config file doesn't exist"""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Director("nonexistent.yaml")

    def test_invalid_yaml(self, tmp_path):
        """Test error when config file contains invalid YAML"""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: :")
        
        with pytest.raises(ValueError, match="Invalid YAML"):
            Director(str(config_file))

    def test_missing_required_files(self, tmp_path):
        """Test error when referenced files don't exist"""
        config_content = """
prompt: test
coder_model: claude-3-5-haiku-20241022
evaluator_model: gpt-4o
max_iterations: 5
execution_command: echo "test"
context_editable:
    - nonexistent.py
context_read_only: []
evaluator: default
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            Director(str(config_file))

class TestTemplateProcessing:
    """Tests for template processing functionality"""

    def test_basic_template_substitution(self, temp_config):
        """Test basic template substitution works"""
        template_values = {
            "name": "Test",
            "details": "Some details"
        }
        
        director = Director(str(temp_config), template_values)
        assert "Hello Test!" in director.config.prompt
        assert "Details: Some details" in director.config.prompt

    def test_missing_template_value(self, temp_config):
        """Test error raised when template value is missing"""
        template_values = {
            "name": "Test"
            # Missing 'details'
        }
        
        with pytest.raises(ValueError, match="Template processing failed"):
            Director(str(temp_config), template_values)

    def test_template_value_override(self, temp_config_with_values):
        """Test CLI template values override config values"""
        cli_values = {
            "name": "CLIName",
            "details": "CLIDetails"
        }
        
        director = Director(str(temp_config_with_values), cli_values)
        assert "Hello CLIName!" in director.config.prompt
        assert "Details: CLIDetails" in director.config.prompt

    def test_partial_template_override(self, temp_config_with_values):
        """Test partial override of template values"""
        cli_values = {
            "name": "CLIName"
            # details not overridden
        }
        
        director = Director(str(temp_config_with_values), cli_values)
        assert "Hello CLIName!" in director.config.prompt
        assert "Details: ConfigDetails" in director.config.prompt

    def test_multiline_template_value(self, temp_config):
        """Test handling of multiline template values"""
        template_values = {
            "name": "Test",
            "details": "Line 1\nLine 2\nLine 3"
        }
        
        director = Director(str(temp_config), template_values)
        assert "Line 1" in director.config.prompt
        assert "Line 2" in director.config.prompt
        assert "Line 3" in director.config.prompt

    def test_special_characters(self, tmp_path):
        """Test handling of special characters in template values"""
        config_content = """
prompt: |
    {{ special_chars }}
coder_model: claude-3-5-haiku-20241022
evaluator_model: gpt-4o
max_iterations: 5
execution_command: echo "test"
context_editable:
    - test.py
context_read_only: []
evaluator: default
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test file")
        
        special_chars = """<div>
        Test & test
        Lines with "quotes"
        Tab\there
        </div>"""
        
        template_values = {"special_chars": special_chars}
        director = Director(str(config_file), template_values)
        
        # Check that special characters are preserved
        assert "<div>" in director.config.prompt
        assert "Test & test" in director.config.prompt
        assert 'Lines with "quotes"' in director.config.prompt
        assert "Tab\there" in director.config.prompt

class TestIntegration:
    """Integration tests for template processing with Director workflow"""

    def test_template_in_evaluation(self, temp_config):
        """Test template processing works with evaluation workflow"""
        template_values = {
            "name": "Test",
            "details": "Check if x equals 5"
        }
        
        director = Director(str(temp_config), template_values)
        # Verify template is processed before evaluation
        evaluation = director.evaluate("x equals 5")
        assert "Check if x equals 5" in director.config.prompt

    def test_md_file_with_templates(self, tmp_path):
        """Test template processing works with .md files"""
        # Create .md file with templates
        md_content = """# Test
Hello {{ name }}!
Details: {{ details }}
"""
        md_file = tmp_path / "test.md"
        md_file.write_text(md_content)
        
        # Create config referencing .md file
        config_content = f"""
prompt: {md_file}
coder_model: claude-3-5-haiku-20241022
evaluator_model: gpt-4o
max_iterations: 5
execution_command: echo "test"
context_editable:
    - test.py
context_read_only: []
evaluator: default
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test file")
        
        template_values = {
            "name": "Test",
            "details": "MD file test"
        }
        
        director = Director(str(config_file), template_values)
        assert "Hello Test!" in director.config.prompt
        assert "Details: MD file test" in director.config.prompt

def test_cli_template_values_parsing():
    """Test parsing of CLI template values"""
    from director import main
    import sys
    from unittest.mock import patch
    
    test_values = {"name": "Test", "details": "Some details"}
    cli_args = ["director.py", "--config", "test_config.yaml", 
                "--template-values", json.dumps(test_values)]
    
    with patch.object(sys, 'argv', cli_args):
        with patch('director.Director') as mock_director:
            main()
            mock_director.assert_called_once_with("test_config.yaml", test_values)

def test_invalid_cli_template_values():
    """Test handling of invalid CLI template values"""
    from director import main
    import sys
    from unittest.mock import patch
    
    cli_args = ["director.py", "--config", "test_config.yaml", 
                "--template-values", "invalid json"]
    
    with patch.object(sys, 'argv', cli_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
