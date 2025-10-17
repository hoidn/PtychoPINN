#!/usr/bin/env python
"""
Verification experiment to test the hypothesis that probe.set_probe_guess() 
in inference.py has no effect on the loaded model's internal state.

Hypothesis: The call to probe.set_probe_guess() occurs after the model has 
already been built and its weights restored. Therefore, modifying the global 
p.cfg['probe'] at this stage has no effect on the tf.Variable inside the 
ProbeIllumination layer of the already-loaded model.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np

# Set environment before ANY imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 70)
    print("PROBE HYPOTHESIS VERIFICATION EXPERIMENT")
    print("=" * 70)
    
    # Import after environment setup
    from ptycho import params as p
    from ptycho import probe
    from ptycho.probe import get_default_probe
    import tensorflow as tf
    
    # Initialize parameters
    print("\n1. Initializing parameters...")
    p.set('N', 64)
    p.set('gridsize', 1)
    p.set('probe.type', 'gaussian')
    p.set('probe.photons', 1e10)
    p.set('nphotons', 1e8)
    p.set('n_filters_scale', 2)
    p.set('offset', 0)
    p.set('gaussian_smoothing_sigma', 0.0)
    p.set('probe.trainable', False)
    p.set('probe.mask', False)
    p.set('intensity_scale', 1.0)
    
    # Set initial probe
    initial_probe = get_default_probe(64)
    p.params()['probe'] = initial_probe
    print(f"Initial probe shape: {initial_probe.shape}")
    print(f"Initial probe mean: {np.mean(np.abs(initial_probe)):.6f}")
    
    # Import model modules after params are set
    from ptycho.model_manager import ModelManager
    from ptycho.model import create_model_with_gridsize
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n2. Creating and saving a test model...")
        
        # Create a model with a specific probe
        model, inference_model = create_model_with_gridsize(gridsize=1, N=64)
        
        # Save the model
        model_path = Path(temp_dir) / "test_model"
        ModelManager.save_model(inference_model, str(model_path), {}, 1.0)
        print(f"Model saved to {model_path}")
        
        # Clear session to ensure clean state
        tf.keras.backend.clear_session()
        
        # Create a completely different probe
        different_probe = np.ones((64, 64), dtype=np.complex64) * 0.5 + 0.5j
        different_probe = different_probe[..., np.newaxis]
        print(f"\n3. Created different probe with mean: {np.mean(np.abs(different_probe)):.6f}")
        
        # Load the model (this will restore the original probe to p.cfg)
        print("\n4. Loading the model...")
        loaded_model = ModelManager.load_model(str(model_path))
        print("Model loaded successfully")
        
        # Step A: Get the probe from inside the model BEFORE set_probe_guess
        probe_layer = None
        for layer in loaded_model.layers:
            if hasattr(layer, 'w') and layer.name == 'probe_illumination':
                probe_layer = layer
                break
        
        if probe_layer is None:
            # Try to find it in nested models
            for layer in loaded_model.layers:
                if hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if hasattr(sublayer, 'w') and sublayer.name == 'probe_illumination':
                            probe_layer = sublayer
                            break
        
        if probe_layer:
            probe_from_model_before = probe_layer.w.numpy()
            print(f"\n5. Probe from model BEFORE set_probe_guess:")
            print(f"   Shape: {probe_from_model_before.shape}")
            print(f"   Mean: {np.mean(np.abs(probe_from_model_before)):.6f}")
        else:
            print("\n5. WARNING: Could not find ProbeIllumination layer in model")
            probe_from_model_before = None
        
        # Get probe from global config BEFORE
        probe_from_cfg_before = p.params()['probe']
        print(f"\n6. Probe from global config BEFORE set_probe_guess:")
        print(f"   Shape: {probe_from_cfg_before.shape}")
        print(f"   Mean: {np.mean(np.abs(probe_from_cfg_before)):.6f}")
        
        # Step B: Call probe.set_probe_guess with the different probe
        print("\n7. Calling probe.set_probe_guess with different probe...")
        probe.set_probe_guess(None, different_probe)
        
        # Get probe from global config AFTER
        probe_from_cfg_after = p.params()['probe']
        print(f"\n8. Probe from global config AFTER set_probe_guess:")
        print(f"   Shape: {probe_from_cfg_after.shape}")
        print(f"   Mean: {np.mean(np.abs(probe_from_cfg_after)):.6f}")
        
        # Step C: Get the probe from inside the model AFTER set_probe_guess
        if probe_layer:
            probe_from_model_after = probe_layer.w.numpy()
            print(f"\n9. Probe from model AFTER set_probe_guess:")
            print(f"   Shape: {probe_from_model_after.shape}")
            print(f"   Mean: {np.mean(np.abs(probe_from_model_after)):.6f}")
        else:
            probe_from_model_after = None
        
        # VERIFICATION
        print("\n" + "=" * 70)
        print("HYPOTHESIS VERIFICATION:")
        print("=" * 70)
        
        # Assertion 1: Global config was changed
        config_changed = not np.allclose(probe_from_cfg_before, probe_from_cfg_after, rtol=1e-5)
        print(f"\n✓ Assertion 1: Global config was changed: {config_changed}")
        if config_changed:
            print(f"  Before mean: {np.mean(np.abs(probe_from_cfg_before)):.6f}")
            print(f"  After mean: {np.mean(np.abs(probe_from_cfg_after)):.6f}")
        
        # Assertion 2: Model's internal state was NOT changed
        if probe_from_model_before is not None and probe_from_model_after is not None:
            model_unchanged = np.allclose(probe_from_model_before, probe_from_model_after, rtol=1e-5)
            print(f"\n✓ Assertion 2: Model's internal state was NOT changed: {model_unchanged}")
            if model_unchanged:
                print(f"  Model probe remains: {np.mean(np.abs(probe_from_model_before)):.6f}")
        else:
            print("\n✗ Assertion 2: Could not verify model's internal state (layer not found)")
            model_unchanged = None
        
        # CONCLUSION
        print("\n" + "=" * 70)
        print("CONCLUSION:")
        print("=" * 70)
        
        if config_changed and model_unchanged:
            print("\n✅ HYPOTHESIS CONFIRMED!")
            print("The probe.set_probe_guess() call in inference.py modifies the global")
            print("configuration but has NO EFFECT on the already-loaded model's internal")
            print("tf.Variable. The model continues to use the probe it was saved with.")
            print("\nRECOMMENDATION: Remove the redundant probe.set_probe_guess() call")
            print("from inference.py as it is misleading and ineffectual.")
            return 0
        elif not config_changed:
            print("\n❌ UNEXPECTED: Global config was not changed by set_probe_guess")
            return 1
        elif not model_unchanged:
            print("\n❌ HYPOTHESIS REJECTED: Model's internal state WAS changed")
            print("Further investigation needed.")
            return 1
        else:
            print("\n⚠️ INCONCLUSIVE: Could not verify model's internal state")
            return 2

if __name__ == "__main__":
    sys.exit(main())