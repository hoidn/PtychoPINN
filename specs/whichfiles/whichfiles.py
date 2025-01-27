import re
from pathlib import Path
import sys
import yaml
from typing import Dict, Any, Optional
import argparse
import subprocess
from datetime import datetime


def load_config(yaml_path: str | Path) -> Dict[str, Any]:
    """
    Load and validate yaml configuration file.
    
    Args:
        yaml_path (str | Path): Path to YAML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary containing description, context_editable, 
                       and context_read_only
                       
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If required keys are missing or invalid
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file {yaml_path} not found"
        )
        
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Validate required keys
    required_keys = {"description", "context_editable", "context_read_only"}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(
            f"Missing required keys in config: {', '.join(missing_keys)}"
        )
        
    # Validate types
    if not isinstance(config["description"], str):
        raise ValueError("description must be a string")
    if not isinstance(config["context_editable"], list):
        raise ValueError("context_editable must be a list")
    if not isinstance(config["context_read_only"], list):
        raise ValueError("context_read_only must be a list")
        
    return config

def process_template(template: str, config: Dict[str, Any]) -> str:
    """
    Replace all <key> placeholders in template with corresponding values from config.
    
    Args:
        template (str): Template string containing <key> placeholders
        config (Dict[str, Any]): Configuration dictionary with replacement values
        
    Returns:
        str: Processed template with all replacements made
    """
    result = template
    for key, value in config.items():
        placeholder = f"<{key}>"
        if isinstance(value, str) and placeholder in result:
            result = result.replace(placeholder, value)
    return result

def setup_argparse() -> argparse.ArgumentParser:
    """
    Create and configure argument parser for CLI.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Process documentation changes based on YAML configuration"
    )
    parser.add_argument(
        "config", 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--spec", 
        help="Path to specification template (defaults to script_name.md)",
        default=None
    )
    return parser

def main(config_path: str | Path, spec_path: Optional[str | Path] = None):
    """
    Main function to process documentation changes.
    
    Args:
        config_path (str): Path to YAML configuration file
        spec_path (Optional[str]): Path to specification template file
    """
    # Load and validate configuration
    config = load_config(config_path)
    
    # Determine spec path
    # Convert paths
    spec_file_path = Path(spec_path if spec_path is not None else Path(sys.argv[0]).with_suffix('.md'))
    
    if not spec_file_path.exists():
        raise FileNotFoundError(
            f"Specification template {spec_file_path} not found"
        )
        
    # Read and process template  
    with open(spec_file_path, "r") as spec_file:
        spec_content = spec_file.read()
    
    spec_prompt = process_template(spec_content, config)

    context = []
    
    # Load editable context files
    for filepath in config["context_editable"]:
        try:
            with open(filepath, "r") as f:
                content = f.read()
                context.append(f"=== {filepath} ===\n{content}\n")
        except FileNotFoundError:
            print(f"Warning: Editable context file {filepath} not found")

    # Load read-only context files 
    for filepath in config["context_read_only"]:
        try:
            with open(filepath, "r") as f:
                content = f.read()
                context.append(f"=== {filepath} ===\n{content}\n")
        except FileNotFoundError:
            print(f"Warning: Read-only context file {filepath} not found")

    # Combine context with prompt
    full_prompt = (
        "Here are the relevant files from the codebase:\n\n" +
        "\n".join(context) +
        "\n\nSpecification:\n" +
        spec_prompt
    )

    # Create temp file and run llm command
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(full_prompt)
        tmp_path = tmp.name

    try:
        with open(tmp_path, 'r') as input_file:
            result = subprocess.run(
                ["llm", "--model", "claude-3-5-sonnet-20241022"],
                stdin=input_file,
                capture_output=True,
                text=True,
                check=True
            )
        response = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running llm command: {e}")
        return
    finally:
        os.unlink(tmp_path)

    # Write response to file
    output_file = "tochange.yaml"
    
    # Extract content between ```yaml ``` markers
    yaml_match = re.search(r'```yaml\s*(.*?)\s*```', response, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1).strip()
    else:
        print("Error: Could not find ```yaml ``` section in LLM response")
        yaml_content = response
        
    with open(output_file, "w") as f:
        f.write(yaml_content)

    # Git commit the new file
    try:
        subprocess.run(["git", "add", output_file], check=True)
        commit_msg = f"Add AI-generated changes from {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during git operations: {e}")

if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    main(args.config, args.spec)
