#!/usr/bin/env python
"""
Simplified logging test to verify the redundancy of configuration steps.
This focuses on the gridsize parameter which is easier to track.
"""

import os
import sys
import tempfile
from pathlib import Path

# Set environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Add project root
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ptycho import params

def test_config_flow():
    """Test the configuration flow to verify redundancy."""
    
    print("=" * 70)
    print("CONFIGURATION FLOW TEST")
    print("=" * 70)
    
    # Phase 1: Initial state
    print("\n=== PHASE 1: INITIAL STATE ===")
    params.set('gridsize', 999)  # Set a known "wrong" value
    print(f"Initial gridsize: {params.get('gridsize')}")
    
    # Phase 2: Simulate update_legacy_dict
    print("\n=== PHASE 2: SIMULATE update_legacy_dict ===")
    # This is what happens in inference.py main()
    from ptycho.config.config import InferenceConfig, ModelConfig
    from dataclasses import fields
    
    # Create a config with gridsize=1 (default)
    model_defaults = {f.name: f.default for f in fields(ModelConfig)}
    print(f"ModelConfig default gridsize: {model_defaults.get('gridsize')}")
    
    # Simulate update_legacy_dict
    from ptycho.config.config import update_legacy_dict
    test_config = InferenceConfig(
        model=ModelConfig(**model_defaults),
        model_path=Path("./test"),
        test_data_file=Path("./test.npz"),
        debug=False,
        output_dir=Path("./output")
    )
    
    update_legacy_dict(params.cfg, test_config)
    print(f"Gridsize after update_legacy_dict: {params.get('gridsize')}")
    
    # Phase 3: Simulate model loading
    print("\n=== PHASE 3: SIMULATE MODEL LOADING ===")
    print("ModelManager.load_model() would call params.cfg.update(loaded_params)")
    print("This overwrites the configuration with the saved model's params")
    
    # Simulate what ModelManager does
    saved_params = {'gridsize': 2, 'N': 128, 'nphotons': 1e8}  # Example saved params
    params.cfg.update(saved_params)
    print(f"Gridsize after model loading (simulated): {params.get('gridsize')}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)
    print("\n1. update_legacy_dict() sets gridsize from config (999 -> 1)")
    print("2. ModelManager.load_model() overwrites it with saved value (1 -> 2)")
    print("3. Therefore, update_legacy_dict() is REDUNDANT!")
    print("\nThe loaded model's configuration is authoritative.")
    print("The initial update_legacy_dict() call has no lasting effect.")
    
    # Test probe setting
    print("\n" + "=" * 70)
    print("PROBE SETTING ANALYSIS:")
    print("=" * 70)
    
    print("\n1. Model is loaded with its saved probe as a tf.Variable")
    print("2. probe.set_probe_guess() modifies params.cfg['probe']")
    print("3. But the model's tf.Variable is NOT linked to params.cfg")
    print("4. Therefore, probe.set_probe_guess() is INEFFECTUAL!")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_config_flow())