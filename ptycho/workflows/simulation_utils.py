"""
Utility functions for decoupled probe and object simulation.

This module provides helper functions to support flexible simulation workflows
where probes and objects can be loaded from different sources independently.
"""

import numpy as np
from pathlib import Path
from typing import Union
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


def load_probe_from_source(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Unified probe loading from an .npz file, .npy file, or a direct NumPy array.
    
    Parameters
    ----------
    source : Union[str, Path, np.ndarray]
        The probe source, which can be:
        - A path to an .npy file containing the probe array
        - A path to an .npz file (will look for 'probeGuess' key)
        - A NumPy array to use directly
    
    Returns
    -------
    np.ndarray
        The probe as a 2D complex array with dtype complex64
        
    Raises
    ------
    ValueError
        If the probe is not 2D, not complex, or if required keys are missing
    FileNotFoundError
        If the specified file does not exist
    TypeError
        If the source type is not supported
        
    Examples
    --------
    >>> # Load from NPY file
    >>> probe = load_probe_from_source('probe.npy')
    
    >>> # Load from NPZ file
    >>> probe = load_probe_from_source('dataset.npz')
    
    >>> # Use existing array
    >>> existing_probe = np.ones((64, 64), dtype=np.complex64)
    >>> probe = load_probe_from_source(existing_probe)
    """
    # Handle direct array input
    if isinstance(source, np.ndarray):
        logger.debug("Loading probe from NumPy array")
        probe = source
    
    # Handle file input
    elif isinstance(source, (str, Path)):
        source_path = Path(source)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Probe source file not found: {source_path}")
        
        # Handle .npy file
        if source_path.suffix == '.npy':
            logger.debug(f"Loading probe from NPY file: {source_path}")
            probe = np.load(source_path)
        
        # Handle .npz file
        elif source_path.suffix == '.npz':
            logger.debug(f"Loading probe from NPZ file: {source_path}")
            with np.load(source_path) as data:
                if 'probeGuess' not in data:
                    available_keys = list(data.keys())
                    raise ValueError(
                        f"NPZ file does not contain 'probeGuess' key. "
                        f"Available keys: {available_keys}"
                    )
                probe = data['probeGuess']
        
        else:
            raise ValueError(
                f"Unsupported file format: {source_path.suffix}. "
                "Only .npy and .npz files are supported."
            )
    
    else:
        raise TypeError(
            f"Unsupported source type: {type(source)}. "
            "Expected str, Path, or numpy.ndarray"
        )
    
    # Validate probe
    if probe.ndim != 2:
        raise ValueError(
            f"Probe must be a 2D array, got shape {probe.shape} "
            f"with {probe.ndim} dimensions"
        )
    
    if not np.iscomplexobj(probe):
        raise ValueError(
            f"Probe must be complex-valued, got dtype {probe.dtype}"
        )
    
    # Convert to complex64 if needed
    if probe.dtype != np.complex64:
        logger.debug(f"Converting probe from {probe.dtype} to complex64")
        probe = probe.astype(np.complex64)
    
    logger.info(f"Loaded probe with shape {probe.shape}, dtype {probe.dtype}")
    
    return probe


def validate_probe_object_compatibility(probe: np.ndarray, obj: np.ndarray) -> None:
    """
    Ensures the probe is smaller than the object and can physically scan it.
    
    Parameters
    ----------
    probe : np.ndarray
        The probe array (2D complex)
    obj : np.ndarray
        The object array (2D complex)
        
    Raises
    ------
    ValueError
        If the probe is too large for the object in any dimension
        
    Notes
    -----
    The probe must be smaller than the object in both dimensions to allow
    for scanning across the object with some buffer space at the edges.
    
    Examples
    --------
    >>> probe = np.ones((64, 64), dtype=np.complex64)
    >>> obj = np.ones((256, 256), dtype=np.complex64)
    >>> validate_probe_object_compatibility(probe, obj)  # No error
    
    >>> large_probe = np.ones((300, 300), dtype=np.complex64)
    >>> validate_probe_object_compatibility(large_probe, obj)
    ValueError: Probe (300x300) is too large for object (256x256). Probe must be smaller than object in both dimensions.
    """
    probe_height, probe_width = probe.shape
    obj_height, obj_width = obj.shape
    
    if probe_height >= obj_height or probe_width >= obj_width:
        raise ValueError(
            f"Probe ({probe_height}x{probe_width}) is too large for object "
            f"({obj_height}x{obj_width}). Probe must be smaller than object "
            f"in both dimensions."
        )
    
    logger.debug(
        f"Probe-object compatibility validated: probe {probe.shape} "
        f"can scan object {obj.shape}"
    )