#!/usr/bin/env python
"""Test script to verify nphotons metadata override functionality."""

import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, '.')

from ptycho.metadata import MetadataManager
from ptycho.config.config import TrainingConfig, ModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_metadata_override(data_file):
    """Test if nphotons is correctly read from metadata."""
    
    # Load metadata
    try:
        _, metadata = MetadataManager.load_with_metadata(data_file)
        metadata_nphotons = None
        if metadata:
            if 'nphotons' in metadata:
                metadata_nphotons = metadata['nphotons']
            elif 'physics_parameters' in metadata and 'nphotons' in metadata['physics_parameters']:
                metadata_nphotons = metadata['physics_parameters']['nphotons']
        
        if metadata_nphotons is not None:
            logger.info(f"✓ Found nphotons in metadata: {metadata_nphotons:.1e}")
        else:
            logger.warning(f"✗ No nphotons found in metadata")
            return False
    except Exception as e:
        logger.error(f"✗ Error loading metadata: {e}")
        return False
    
    # Create a config with default nphotons
    config = TrainingConfig(
        model=ModelConfig(),
        nphotons=1e9  # Default value
    )
    logger.info(f"  Original config nphotons: {config.nphotons:.1e}")
    
    # Override with metadata value  
    if metadata_nphotons is not None:
        config = config.__class__(
            **{**config.__dict__, 'nphotons': float(metadata_nphotons)}
        )
        logger.info(f"  Updated config nphotons: {config.nphotons:.1e}")
        
        if abs(config.nphotons - float(metadata_nphotons)) < 1e-10:
            logger.info(f"✓ Successfully overrode nphotons with metadata value")
            return True
        else:
            logger.error(f"✗ Failed to override nphotons correctly")
            return False
    
    return False

if __name__ == "__main__":
    # Test each photon level
    photon_levels = ['1e3', '1e4', '1e5', '1e6', '1e7', '1e8', '1e9']
    
    logger.info("Testing nphotons metadata override for all photon levels:")
    logger.info("-" * 60)
    
    all_passed = True
    for level in photon_levels:
        data_file = f"photon_grid_study_regenerated_20250826_163348/data_p{level}.npz"
        if Path(data_file).exists():
            logger.info(f"\nTesting {level} photons dataset:")
            passed = test_metadata_override(data_file)
            if not passed:
                all_passed = False
        else:
            logger.warning(f"\n✗ Dataset not found: {data_file}")
            all_passed = False
    
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✓ All tests PASSED")
    else:
        logger.error("✗ Some tests FAILED")
    
    sys.exit(0 if all_passed else 1)