"""Metadata management for NPZ files in PtychoPINN.

This module provides utilities for storing, loading, and validating metadata
within NPZ files, particularly focusing on physics parameters like nphotons
that are critical for maintaining consistency across simulation, training,
and inference workflows.

The metadata is stored as a JSON-encoded string in the '_metadata' key of NPZ files,
ensuring backward compatibility with existing code that doesn't expect metadata.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import socket
import warnings

from ptycho.config.config import TrainingConfig, ModelConfig

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manages metadata storage and retrieval for NPZ files.
    
    This class provides static methods for creating, saving, loading, and validating
    metadata associated with ptychography datasets. The metadata includes physics
    parameters (like nphotons), creation information, and transformation history.
    """
    
    METADATA_KEY = '_metadata'
    CURRENT_SCHEMA_VERSION = '1.0.0'
    
    @staticmethod
    def create_metadata(
        config: TrainingConfig,
        script_name: str,
        **additional_params
    ) -> Dict[str, Any]:
        """Create a metadata dictionary from configuration and context.
        
        Args:
            config: Training configuration containing physics parameters
            script_name: Name of the script creating the dataset
            **additional_params: Additional parameters to include (e.g., buffer, seed)
            
        Returns:
            Complete metadata dictionary with all relevant information
        """
        metadata = {
            "schema_version": MetadataManager.CURRENT_SCHEMA_VERSION,
            "creation_info": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "script": script_name,
                "hostname": socket.gethostname(),
                "ptychopinn_version": "2.0.0"  # TODO: Get from package version
            },
            "physics_parameters": {
                "nphotons": config.nphotons,
                "gridsize": config.model.gridsize,
                "N": config.model.N,
                "probe_trainable": config.probe_trainable,
                "intensity_scale_trainable": config.intensity_scale_trainable,
                "nll_weight": config.nll_weight,
                "model_type": config.model.model_type
            },
            "training_parameters": {
                "n_images": config.n_images,
                "batch_size": config.batch_size,
                "nepochs": config.nepochs
            },
            "data_transformations": []
        }
        
        # Add any additional parameters passed in
        if additional_params:
            metadata["additional_parameters"] = additional_params
            
        return metadata
    
    @staticmethod
    def save_with_metadata(
        file_path: str,
        data_dict: Dict[str, np.ndarray],
        metadata: Dict[str, Any]
    ) -> None:
        """Save NPZ file with embedded JSON metadata.
        
        Args:
            file_path: Path where the NPZ file will be saved
            data_dict: Dictionary of arrays to save
            metadata: Metadata dictionary to embed
        """
        # Make a copy to avoid modifying the original
        save_dict = data_dict.copy()
        
        # Convert metadata to JSON string and store as array
        # Use custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        metadata_json = json.dumps(metadata, indent=2, cls=NumpyEncoder)
        # Store as 0-d object array to preserve the string
        save_dict[MetadataManager.METADATA_KEY] = np.array(metadata_json, dtype=object)
        
        # Save the combined dictionary
        np.savez_compressed(file_path, **save_dict)
        logger.debug(f"Saved NPZ with metadata to {file_path}")
    
    @staticmethod
    def load_with_metadata(
        file_path: str
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Any]]]:
        """Load NPZ file and extract metadata if present.
        
        Args:
            file_path: Path to the NPZ file to load
            
        Returns:
            Tuple of (data_dict, metadata_dict)
            metadata_dict will be None if no metadata is present
        """
        with np.load(file_path, allow_pickle=True) as data:
            # Extract all arrays except metadata
            data_dict = {}
            metadata = None
            
            for key in data.files:
                if key == MetadataManager.METADATA_KEY:
                    # Extract and parse metadata
                    try:
                        metadata_json = str(data[key].item())
                        metadata = json.loads(metadata_json)
                        logger.debug(f"Loaded metadata from {file_path}")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Failed to parse metadata from {file_path}: {e}")
                else:
                    data_dict[key] = data[key]
        
        return data_dict, metadata
    
    @staticmethod
    def validate_parameters(
        metadata: Dict[str, Any],
        current_config: TrainingConfig
    ) -> List[str]:
        """Validate current configuration against stored metadata.
        
        Args:
            metadata: Metadata dictionary from NPZ file
            current_config: Current training configuration
            
        Returns:
            List of warning messages for any mismatches found
        """
        warnings_list = []
        
        if not metadata:
            return warnings_list
        
        physics_params = metadata.get("physics_parameters", {})
        
        # Check critical parameters
        if "nphotons" in physics_params:
            stored_nphotons = physics_params["nphotons"]
            if abs(stored_nphotons - current_config.nphotons) > 1e-9:
                warnings_list.append(
                    f"nphotons mismatch: dataset={stored_nphotons}, "
                    f"config={current_config.nphotons}"
                )
        
        if "N" in physics_params:
            stored_N = physics_params["N"]
            if stored_N != current_config.model.N:
                warnings_list.append(
                    f"N mismatch: dataset={stored_N}, "
                    f"config={current_config.model.N}"
                )
        
        if "gridsize" in physics_params:
            stored_gridsize = physics_params["gridsize"]
            if stored_gridsize != current_config.model.gridsize:
                warnings_list.append(
                    f"gridsize mismatch: dataset={stored_gridsize}, "
                    f"config={current_config.model.gridsize}"
                )
        
        return warnings_list
    
    @staticmethod
    def add_transformation_record(
        metadata: Optional[Dict[str, Any]],
        tool_name: str,
        operation: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add a transformation record to metadata.
        
        Args:
            metadata: Existing metadata dictionary (can be None)
            tool_name: Name of the tool performing the transformation
            operation: Description of the operation
            parameters: Parameters used for the transformation
            
        Returns:
            Updated metadata dictionary
        """
        if metadata is None:
            metadata = {
                "schema_version": MetadataManager.CURRENT_SCHEMA_VERSION,
                "data_transformations": []
            }
        
        transformation = {
            "tool": tool_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "operation": operation,
            "parameters": parameters
        }
        
        metadata.setdefault("data_transformations", []).append(transformation)
        return metadata
    
    @staticmethod
    def get_nphotons(metadata: Optional[Dict[str, Any]], default: float = 1e9) -> float:
        """Extract nphotons value from metadata.
        
        Args:
            metadata: Metadata dictionary from NPZ file
            default: Default value if metadata is missing or doesn't contain nphotons
            
        Returns:
            The nphotons value
        """
        if metadata is None:
            return default
        
        physics_params = metadata.get("physics_parameters", {})
        return physics_params.get("nphotons", default)
    
    @staticmethod
    def merge_metadata(
        metadata_list: List[Optional[Dict[str, Any]]],
        merge_strategy: str = "preserve_first"
    ) -> Dict[str, Any]:
        """Merge multiple metadata dictionaries.
        
        Args:
            metadata_list: List of metadata dictionaries to merge
            merge_strategy: Strategy for handling conflicts
                - "preserve_first": Keep parameters from first non-None metadata
                - "validate_consistent": Ensure all parameters match
                
        Returns:
            Merged metadata dictionary
        """
        # Filter out None values
        valid_metadata = [m for m in metadata_list if m is not None]
        
        if not valid_metadata:
            return {
                "schema_version": MetadataManager.CURRENT_SCHEMA_VERSION,
                "merge_info": {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "strategy": merge_strategy,
                    "source_count": 0
                }
            }
        
        if merge_strategy == "preserve_first":
            merged = valid_metadata[0].copy()
        elif merge_strategy == "validate_consistent":
            # Check that all physics parameters match
            merged = valid_metadata[0].copy()
            for i, other in enumerate(valid_metadata[1:], 1):
                other_physics = other.get("physics_parameters", {})
                merged_physics = merged.get("physics_parameters", {})
                
                for key in ["nphotons", "N", "gridsize"]:
                    if key in other_physics and key in merged_physics:
                        if other_physics[key] != merged_physics[key]:
                            raise ValueError(
                                f"Inconsistent {key} in metadata {i}: "
                                f"{other_physics[key]} vs {merged_physics[key]}"
                            )
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Add merge information
        merged["merge_info"] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "strategy": merge_strategy,
            "source_count": len(valid_metadata)
        }
        
        return merged


def load_config_from_metadata(metadata: Dict[str, Any]) -> TrainingConfig:
    """Recreate a TrainingConfig from NPZ metadata.
    
    Args:
        metadata: Metadata dictionary from an NPZ file
        
    Returns:
        TrainingConfig reconstructed from the metadata
    """
    physics_params = metadata.get("physics_parameters", {})
    training_params = metadata.get("training_parameters", {})
    
    model_config = ModelConfig(
        N=physics_params.get("N", 64),
        gridsize=physics_params.get("gridsize", 1),
        model_type=physics_params.get("model_type", "pinn")
    )
    
    return TrainingConfig(
        model=model_config,
        nphotons=physics_params.get("nphotons", 1e9),
        probe_trainable=physics_params.get("probe_trainable", False),
        intensity_scale_trainable=physics_params.get("intensity_scale_trainable", True),
        nll_weight=physics_params.get("nll_weight", 1.0),
        n_images=training_params.get("n_images", 1000),
        batch_size=training_params.get("batch_size", 32),
        nepochs=training_params.get("nepochs", 50)
    )