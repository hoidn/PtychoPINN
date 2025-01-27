from pathlib import Path
import sys
import yaml
from typing import Dict, Any, Optional
import argparse
import subprocess
from datetime import datetime

# TODO: instead of using aider, use the `llm` cli tool, see <ref>. After
# running the tool, git commit the new file addition 
# <ref>
#(ptycho_test) ollie@ollie:~/Documents/scratch/ptycho/specs $ llm --help
#Usage: llm [OPTIONS] COMMAND [ARGS]...
#
#  Access Large Language Models from the command-line
#
#  Documentation: https://llm.datasette.io/
#
#  LLM can run models from many different providers. Consult the plugin
#  directory for a list of available models:
#
#  https://llm.datasette.io/en/stable/plugins/directory.html
#
#  To get started with OpenAI, obtain an API key from them and:
#
#      $ llm keys set openai
#      Enter key: ...
#
#  Then execute a prompt like this:
#
#      llm 'Five outrageous names for a pet pelican'
#
#Options:
#  --version  Show the version and exit.
#  --help     Show this message and exit.
#
#Commands:
#  prompt*       Execute a prompt
#  aliases       Manage model aliases
#  chat          Hold an ongoing chat with a model.
#  collections   View and manage collections of embeddings
#  embed         Embed text and store or return the result
#  embed-models  Manage available embedding models
#  embed-multi   Store embeddings for multiple strings at once
#  install       Install packages from PyPI into the same environment as LLM
#  keys          Manage stored API keys for different models
#  logs          Tools for exploring logged prompts and responses
#  models        Manage available models
#  openai        Commands for working directly with the OpenAI API
#  plugins       List installed plugins
#  similar       Return top N similar IDs from a collection
#  templates     Manage stored prompt templates
#  uninstall     Uninstall Python packages from the LLM environment
#(ptycho_test) ollie@ollie:~/Documents/scratch/ptycho/specs $ llm aliases
#4o                  : gpt-4o
#4o-mini             : gpt-4o-mini
#3.5                 : gpt-3.5-turbo
#chatgpt             : gpt-3.5-turbo
#chatgpt-16k         : gpt-3.5-turbo-16k
#3.5-16k             : gpt-3.5-turbo-16k
#4                   : gpt-4
#gpt4                : gpt-4
#4-32k               : gpt-4-32k
#gpt-4-turbo-preview : gpt-4-turbo
#4-turbo             : gpt-4-turbo
#4t                  : gpt-4-turbo
#3.5-instruct        : gpt-3.5-turbo-instruct
#chatgpt-instruct    : gpt-3.5-turbo-instruct
#ada                 : text-embedding-ada-002 (embedding)
#ada-002             : text-embedding-ada-002 (embedding)
#3-small             : text-embedding-3-small (embedding)
#3-large             : text-embedding-3-large (embedding)
#3-small-512         : text-embedding-3-small-512 (embedding)
#3-large-256         : text-embedding-3-large-256 (embedding)
#3-large-1024        : text-embedding-3-large-1024 (embedding)
#</ref>

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



    # Run llm command and capture output
    try:
        result = subprocess.run(
            ["llm", "--model", "gpt-4-turbo", spec_prompt],
            capture_output=True,
            text=True,
            check=True
        )
        response = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running llm command: {e}")
        return

    # Write response to file
    output_file = "tochange.yaml"
    with open(output_file, "w") as f:
        f.write(response)

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
