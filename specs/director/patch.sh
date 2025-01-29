#!/bin/bash

# Create directory structure
mkdir -p src/director
mkdir -p tests

# Create src/director/errors.py
cat > src/director/errors.py << 'EOF'
class TemplateError(Exception):
    """Base class for template-related errors"""
    pass

class TemplateSyntaxError(TemplateError):
    """Raised when template syntax is invalid"""
    pass

class TemplateValueError(TemplateError):
    """Raised when template values are invalid or missing"""
    pass
EOF

# Create src/director/__init__.py
cat > src/director/__init__.py << 'EOF'
from .director import Director
from .errors import TemplateError, TemplateSyntaxError, TemplateValueError

__all__ = ['Director', 'TemplateError', 'TemplateSyntaxError', 'TemplateValueError']
EOF

# Create tests/conftest.py
cat > tests/conftest.py << 'EOF'
import pytest
from pathlib import Path

@pytest.fixture
def test_dir(tmp_path):
    """Create a test directory with necessary structure"""
    (tmp_path / "specs").mkdir()
    (tmp_path / "src").mkdir()
    return tmp_path

@pytest.fixture
def basic_config(test_dir):
    """Create a basic config file for testing"""
    config = test_dir / "specs/basic.yaml"
    config.write_text("""
prompt: "Test prompt"
coder_model: claude-3-5-haiku-20241022
evaluator_model: gpt-4o
max_iterations: 5
execution_command: echo "test"
context_editable:
    - test.py
context_read_only: []
evaluator: default
""")
    return config

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
EOF

# Create tests/test_templates.py
cat > tests/test_templates.py << 'EOF'
import pytest
import json
from pathlib import Path
from director import Director, TemplateError, TemplateSyntaxError, TemplateValueError

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
EOF

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "director"
version = "0.1.0"
dependencies = [
    "pydantic>=2.0",
    "openai",
    "aider-chat",
    "jinja2",
    "pyyaml"
]

[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-cov"
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=director"
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.env
.venv
env/
venv/
ENV/
EOF

# Make script executable
chmod +x src/director/*.py
chmod +x tests/*.py

echo "Project structure created successfully! Now copy director.py to src/director/"

# Make setup script executable
chmod +x setup.sh
