"""
Utilities for registering PyTorch Lightning models to MLflow server
"""
import os
import json
import dataclasses
from typing import Dict, Optional, Union, Any
from pathlib import Path

import torch
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient


def register_lightning_model_to_mlflow(
    checkpoint_path: str,
    model_class: type,
    experiment_name: str,
    registered_model_name: Optional[str] = None,
    config_map: Optional[Dict[str, Any]] = None,
    config_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    additional_artifacts: Optional[Dict[str, str]] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> str:
    """
    Register a trained PyTorch Lightning model to MLflow server.
    
    This function creates a new MLflow run and logs:
    - The trained model (loaded from checkpoint)
    - Configuration parameters (from config_map or config_dir)
    - Additional artifacts (checkpoints, logs, etc.)
    
    Args:
        checkpoint_path: Path to the .ckpt file from Lightning training
        model_class: The Lightning module class (e.g., PtychoPINN_Lightning)
        experiment_name: Name of the MLflow experiment
        registered_model_name: Optional name to register model in MLflow Model Registry
        config_map: Dictionary mapping config names to config objects (dataclass instances)
                   e.g., {"data_config": data_config, "model_config": model_config}
        config_dir: Alternative to config_map - path to directory containing config JSONs
        run_name: Optional name for this MLflow run
        tags: Optional dictionary of tags to add to the run
        additional_artifacts: Optional dict mapping artifact_path -> local_path
                             e.g., {"logs": "/path/to/logs", "checkpoints": "/path/to/ckpts"}
        mlflow_tracking_uri: Optional MLflow tracking server URI
    
    Returns:
        run_id: The MLflow run ID for this registered model
    
    Example:
        >>> from ptychopinn_torch.models import PtychoPINN_Lightning
        >>> 
        >>> run_id = register_lightning_model_to_mlflow(
        ...     checkpoint_path="outputs/run_20240115/checkpoints/best.ckpt",
        ...     model_class=PtychoPINN_Lightning,
        ...     experiment_name="PtychoPINN Production",
        ...     registered_model_name="ptychopinn_v1",
        ...     config_map={
        ...         "data_config": data_config,
        ...         "model_config": model_config,
        ...         "training_config": training_config,
        ...     },
        ...     tags={"stage": "production", "version": "1.0"},
        ...     additional_artifacts={
        ...         "logs": "outputs/run_20240115/logs",
        ...         "all_checkpoints": "outputs/run_20240115/checkpoints"
        ...     }
        ... )
    """
    
    # Set tracking URI if provided
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Validate inputs
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if config_map is None and config_dir is None:
        raise ValueError("Must provide either config_map or config_dir")
    
    # Load configurations
    configs = _load_configs(config_map, config_dir)
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"Created MLflow run: {run_id}")
        
        # 1. Log configuration parameters
        _log_configs(configs)
        
        # 2. Load and log the model
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = model_class.load_from_checkpoint(checkpoint_path)
        
        # Log model to MLflow
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )
        print(f"Model logged to MLflow (registered_model_name={registered_model_name})")
        
        # 3. Log the checkpoint file itself
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
        
        # 4. Log additional artifacts
        if additional_artifacts:
            _log_additional_artifacts(additional_artifacts)
        
        # 5. Set tags
        default_tags = {
            "framework": "pytorch_lightning",
            "checkpoint_path": checkpoint_path,
        }
        if tags:
            default_tags.update(tags)
        
        for key, value in default_tags.items():
            mlflow.set_tag(key, value)
        
        print(f"Model registration complete!")
        print(f"Run ID: {run_id}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        
        return run_id


def register_from_directory_structure(
    run_dir: str,
    model_class: type,
    experiment_name: str,
    checkpoint_name: str = "best-checkpoint.ckpt",
    registered_model_name: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Register a model from a standard directory structure.
    
    Expected structure:
        run_dir/
            checkpoints/
                best-checkpoint.ckpt
            configs/
                data_config.json
                model_config.json
                training_config.json
                ...
            logs/
                ...
    
    Args:
        run_dir: Root directory of the training run
        model_class: The Lightning module class
        experiment_name: MLflow experiment name
        checkpoint_name: Name of the checkpoint file to use
        registered_model_name: Optional model registry name
        mlflow_tracking_uri: Optional MLflow server URI
        tags: Optional tags for the run
    
    Returns:
        run_id: The MLflow run ID
    
    Example:
        >>> run_id = register_from_directory_structure(
        ...     run_dir="outputs/run_20240115_143022",
        ...     model_class=PtychoPINN_Lightning,
        ...     experiment_name="PtychoPINN Production",
        ...     registered_model_name="ptychopinn_v1",
        ...     mlflow_tracking_uri="http://mlflow-server:5000"
        ... )
    """
    
    run_dir = Path(run_dir)
    
    # Find checkpoint
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        # Try to find any checkpoint
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                checkpoint_path = ckpts[0]
                print(f"Using checkpoint: {checkpoint_path.name}")
            else:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    
    # Config directory
    config_dir = run_dir / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    # Additional artifacts
    additional_artifacts = {}
    
    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        additional_artifacts["logs"] = str(logs_dir)
    
    all_checkpoints_dir = run_dir / "checkpoints"
    if all_checkpoints_dir.exists():
        additional_artifacts["all_checkpoints"] = str(all_checkpoints_dir)
    
    # Register
    return register_lightning_model_to_mlflow(
        checkpoint_path=str(checkpoint_path),
        model_class=model_class,
        experiment_name=experiment_name,
        registered_model_name=registered_model_name,
        config_dir=str(config_dir),
        tags=tags,
        additional_artifacts=additional_artifacts,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )


# Helper functions

def _load_configs(
    config_map: Optional[Dict[str, Any]],
    config_dir: Optional[str]
) -> Dict[str, Dict]:
    """Load configurations from either config_map or config_dir"""
    
    if config_map is not None:
        # Convert dataclass instances to dicts
        configs = {}
        for name, cfg_instance in config_map.items():
            if dataclasses.is_dataclass(cfg_instance):
                configs[name] = dataclasses.asdict(cfg_instance)
            elif isinstance(cfg_instance, dict):
                configs[name] = cfg_instance
            else:
                raise TypeError(f"Config {name} must be dataclass or dict")
        return configs
    
    elif config_dir is not None:
        # Load from JSON files
        configs = {}
        config_path = Path(config_dir)
        
        for json_file in config_path.glob("*.json"):
            config_name = json_file.stem  # filename without .json
            with open(json_file, 'r') as f:
                configs[config_name] = json.load(f)
        
        if not configs:
            raise ValueError(f"No JSON config files found in {config_dir}")
        
        return configs
    
    return {}


def _log_configs(configs: Dict[str, Dict]):
    """Log configuration parameters to MLflow"""
    
    for config_name, config_dict in configs.items():
        # Log as a single JSON parameter
        serializable_dict = _make_serializable(config_dict)
        mlflow.log_param(f"{config_name}_params", json.dumps(serializable_dict))
        
        # Also log individual important parameters (optional, for easier filtering)
        # You can customize this based on which params you want easily searchable
        for key, value in serializable_dict.items():
            if isinstance(value, (int, float, str, bool)):
                param_name = f"{config_name}.{key}"
                # MLflow has a 250 char limit on param values
                str_value = str(value)
                if len(str_value) <= 250:
                    mlflow.log_param(param_name, value)


def _log_additional_artifacts(additional_artifacts: Dict[str, str]):
    """Log additional artifacts to MLflow"""
    
    for artifact_path, local_path in additional_artifacts.items():
        local_path = Path(local_path)
        
        if local_path.is_file():
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        elif local_path.is_dir():
            mlflow.log_artifacts(str(local_path), artifact_path=artifact_path)
        else:
            print(f"Warning: Artifact path does not exist: {local_path}")


def _make_serializable(d: Dict) -> Dict:
    """Recursively convert non-serializable types to JSON-compatible types"""
    
    result = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            # Convert small tensors to lists, large ones to shape description
            result[k] = v.tolist() if v.numel() < 100 else f"Tensor(shape={list(v.shape)})"
        elif isinstance(v, dict):
            result[k] = _make_serializable(v)
        elif isinstance(v, (list, tuple)):
            result[k] = [
                x.tolist() if torch.is_tensor(x) else x 
                for x in v
            ]
        elif isinstance(v, (int, float, str, bool, type(None))):
            result[k] = v
        else:
            # For other types, convert to string
            result[k] = str(v)
    
    return result


def load_model_from_mlflow(
    run_id: str,
    model_class: type,
    mlflow_tracking_uri: Optional[str] = None,
) -> Any:
    """
    Load a registered model from MLflow.
    
    Args:
        run_id: MLflow run ID
        model_class: The Lightning module class
        mlflow_tracking_uri: Optional MLflow server URI
    
    Returns:
        Loaded PyTorch Lightning model
    
    Example:
        >>> model = load_model_from_mlflow(
        ...     run_id="abc123def456",
        ...     model_class=PtychoPINN_Lightning,
        ...     mlflow_tracking_uri="http://mlflow-server:5000"
        ... )
    """
    
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    
    return model

def get_run_id_from_model_version(
    model_name: str,
    version: Optional[Union[int, str]] = None,
    stage: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> str:
    """
    Get the run ID associated with a registered model version.
    
    Args:
        model_name: Name of the registered model
        version: Specific version number (e.g., 1, 2, 3)
        stage: Stage name (e.g., "Production", "Staging") - alternative to version
        mlflow_tracking_uri: Optional MLflow server URI
    
    Returns:
        run_id: The run ID that produced this model version
    
    Example:
        >>> # Get run ID for version 2
        >>> run_id = get_run_id_from_model_version("ptychopinn_v1", version=2)
        
        >>> # Get run ID for production model
        >>> run_id = get_run_id_from_model_version("ptychopinn_v1", stage="Production")
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    client = MlflowClient()
    
    if version is not None:
        # Get specific version
        model_version = client.get_model_version(model_name, version)
    elif stage is not None:
        # Get latest version in stage
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"No model found in stage '{stage}' for model '{model_name}'")
        model_version = versions[0]
    else:
        # Get latest version overall
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        # Sort by version number and get latest
        model_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    
    return model_version.run_id


def load_configs_from_run(
    run_id: str,
    mlflow_tracking_uri: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Load all configuration parameters from an MLflow run.
    
    Args:
        run_id: MLflow run ID
        mlflow_tracking_uri: Optional MLflow server URI
    
    Returns:
        Dictionary mapping config names to config dictionaries
    
    Example:
        >>> configs = load_configs_from_run("abc123def456")
        >>> data_config = configs["data_config"]
        >>> model_config = configs["model_config"]
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    client = MlflowClient()
    run = client.get_run(run_id)
    
    configs = {}
    
    # Extract config parameters (they were logged as JSON strings)
    for param_name, param_value in run.data.params.items():
        if param_name.endswith("_params"):
            config_name = param_name.replace("_params", "")
            try:
                configs[config_name] = json.loads(param_value)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {param_name} as JSON")
    
    return configs


def load_model_and_configs(
    model_name: str,
    version: Optional[Union[int, str]] = None,
    stage: Optional[str] = None,
    model_class: Optional[type] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> Tuple[Any, Dict[str, Dict], str]:
    """
    Load both the model and its associated configurations from MLflow.
    
    This is the main function you'll use - it handles the run ID association automatically.
    
    Args:
        model_name: Name of the registered model
        version: Specific version number
        stage: Stage name (alternative to version)
        model_class: Optional Lightning module class (for type safety)
        mlflow_tracking_uri: Optional MLflow server URI
    
    Returns:
        Tuple of (model, configs, run_id)
        - model: Loaded PyTorch model
        - configs: Dictionary of configuration dictionaries
        - run_id: The associated run ID
    
    Example:
        >>> # Load production model with configs
        >>> model, configs, run_id = load_model_and_configs(
        ...     model_name="ptychopinn_v1",
        ...     stage="Production",
        ...     mlflow_tracking_uri="http://mlflow-server:5000"
        ... )
        >>> 
        >>> print(f"Loaded model from run: {run_id}")
        >>> print(f"Data config: {configs['data_config']}")
        >>> print(f"Model config: {configs['model_config']}")
        
        >>> # Load specific version
        >>> model, configs, run_id = load_model_and_configs(
        ...     model_name="ptychopinn_v1",
        ...     version=3
        ... )
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Step 1: Get run ID from model version
    run_id = get_run_id_from_model_version(
        model_name=model_name,
        version=version,
        stage=stage,
        mlflow_tracking_uri=mlflow_tracking_uri
    )
    
    print(f"Model version maps to run_id: {run_id}")
    
    # Step 2: Load the model
    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
    elif stage is not None:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        # Get latest version
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        model_uri = f"models:/{model_name}/{latest_version.version}"
    
    model = mlflow.pytorch.load_model(model_uri)
    
    # Step 3: Load configs from the run
    configs = load_configs_from_run(run_id, mlflow_tracking_uri)
    
    return model, configs, run_id


def reconstruct_config_objects(
    configs: Dict[str, Dict],
    config_classes: Dict[str, type],
) -> Dict[str, Any]:
    """
    Reconstruct dataclass config objects from dictionaries.
    
    Args:
        configs: Dictionary of config dictionaries (from load_configs_from_run)
        config_classes: Dictionary mapping config names to their dataclass types
    
    Returns:
        Dictionary of reconstructed config objects
    
    Example:
        >>> from ptychopinn_torch.config_params import DataConfig, ModelConfig
        >>> 
        >>> model, configs_dict, run_id = load_model_and_configs("ptychopinn_v1", stage="Production")
        >>> 
        >>> # Reconstruct as dataclass objects
        >>> config_objects = reconstruct_config_objects(
        ...     configs_dict,
        ...     config_classes={
        ...         "data_config": DataConfig,
        ...         "model_config": ModelConfig,
        ...         "training_config": TrainingConfig,
        ...     }
        ... )
        >>> 
        >>> data_config = config_objects["data_config"]  # Now a DataConfig instance
        >>> assert isinstance(data_config, DataConfig)
    """
    reconstructed = {}
    
    for config_name, config_dict in configs.items():
        if config_name in config_classes:
            config_class = config_classes[config_name]
            try:
                # Reconstruct dataclass from dict
                reconstructed[config_name] = config_class(**config_dict)
            except TypeError as e:
                print(f"Warning: Could not reconstruct {config_name}: {e}")
                reconstructed[config_name] = config_dict
        else:
            # Keep as dict if no class provided
            reconstructed[config_name] = config_dict
    
    return reconstructed


def get_model_info(
    model_name: str,
    version: Optional[Union[int, str]] = None,
    stage: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get detailed information about a registered model version.
    
    Args:
        model_name: Name of the registered model
        version: Specific version number
        stage: Stage name
        mlflow_tracking_uri: Optional MLflow server URI
    
    Returns:
        Dictionary with model metadata including run_id, metrics, tags, etc.
    
    Example:
        >>> info = get_model_info("ptychopinn_v1", stage="Production")
        >>> print(f"Run ID: {info['run_id']}")
        >>> print(f"Created: {info['creation_timestamp']}")
        >>> print(f"Metrics: {info['metrics']}")
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    client = MlflowClient()
    
    # Get model version
    if version is not None:
        model_version = client.get_model_version(model_name, version)
    elif stage is not None:
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"No model found in stage '{stage}'")
        model_version = versions[0]
    else:
        versions = client.search_model_versions(f"name='{model_name}'")
        model_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    
    # Get run details
    run = client.get_run(model_version.run_id)
    
    return {
        "model_name": model_name,
        "version": model_version.version,
        "run_id": model_version.run_id,
        "current_stage": model_version.current_stage,
        "creation_timestamp": model_version.creation_timestamp,
        "last_updated_timestamp": model_version.last_updated_timestamp,
        "description": model_version.description,
        "tags": model_version.tags,
        "run_name": run.info.run_name,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "run_tags": run.data.tags,
    }

## Usage Example

# from ptychopinn_torch.mlflow_utils import load_model_and_configs
# from ptychopinn_torch.config_params import DataConfig, ModelConfig, TrainingConfig

# # Load model and configs in one call
# model, configs, run_id = load_model_and_configs(
#     model_name="ptychopinn_v1",
#     stage="Production",
#     mlflow_tracking_uri="http://mlflow-server:5000"
# )

# print(f"Loaded model from run: {run_id}")

# # Access configs as dictionaries
# data_config_dict = configs["data_config"]
# model_config_dict = configs["model_config"]

# # Or reconstruct as dataclass objects
# from ptychopinn_torch.mlflow_utils import reconstruct_config_objects

# config_objects = reconstruct_config_objects(
#     configs,
#     config_classes={
#         "data_config": DataConfig,
#         "model_config": ModelConfig,
#         "training_config": TrainingConfig,
#     }
# )

# data_config = config_objects["data_config"]  # Now a DataConfig instance