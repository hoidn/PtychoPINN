from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
import yaml
from pydantic import BaseModel

class DirectorConfig(BaseModel):
    prompt: str
    coder_model: str
    evaluator_model: Literal["gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview", "o3-mini"]
    max_iterations: int
    execution_command: str
    context_editable: List[str]
    context_read_only: List[str]
    evaluator: Literal["default"]
    template_values: Optional[Dict[str, Any]] = None

def load_and_validate_config(
    config_path: Path, 
    cli_context_editable: Optional[List[str]] = None,
    cli_context_read_only: Optional[List[str]] = None
) -> DirectorConfig:
    # Validate file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load and parse YAML
    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {str(e)}")

    # If prompt is a Markdown file, load its contents.
    if isinstance(config_dict.get("prompt"), str) and config_dict["prompt"].endswith(".md"):
        prompt_path = Path(config_dict["prompt"])
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path) as f:
            config_dict["prompt"] = f.read()

    config = DirectorConfig(**config_dict)

    # Validate evaluator_model
    allowed_evaluator_models = {"gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview", "o3-mini"}
    if config.evaluator_model not in allowed_evaluator_models:
        raise ValueError(
            f"evaluator_model must be one of {allowed_evaluator_models}, got {config.evaluator_model}"
        )

    # Decide which editable files to use
    editable_files = cli_context_editable if cli_context_editable is not None else config.context_editable
    if not editable_files:
        raise ValueError("At least one editable context file must be specified")
    for path in editable_files:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    # Override the YAML value if the CLI override was provided
    if cli_context_editable is not None:
        config.context_editable = cli_context_editable
    if cli_context_read_only is not None:
        config.context_read_only = cli_context_read_only

    for path in config.context_read_only:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    return config
