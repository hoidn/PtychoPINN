#!/usr/bin/env python3
"""Investigate why loss is low but reconstruction quality is poor."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the test data to see ground truth
print("Loading test data...")
test_data = np.load('photon_3e3_custom_study/data_3e3_test.npz')

# Check what we have
print("\nAvailable keys in test data:")
for key in test_data.keys():
    if hasattr(test_data[key], 'shape'):
        print(f"  {key}: shape {test_data[key].shape}, dtype {test_data[key].dtype}")

# Load and visualize ground truth if available
if 'objectGuess' in test_data:
    obj_gt = test_data['objectGuess']
    print(f"\nGround truth object shape: {obj_gt.shape}")
    print(f"Ground truth range: [{np.abs(obj_gt).min():.4f}, {np.abs(obj_gt).max():.4f}]")
    
    # Plot ground truth
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(np.abs(obj_gt), cmap='gray')
    axes[0].set_title('Ground Truth Amplitude')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(np.angle(obj_gt), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Ground Truth Phase')
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle('Ground Truth Object (from test data)')
    plt.tight_layout()
    plt.savefig('3e3_ground_truth.png', dpi=150)
    plt.show()

# Check the diffraction patterns
diff_data = test_data['diff3d']
print(f"\nDiffraction data shape: {diff_data.shape}")
print(f"Diffraction data range: [{diff_data.min():.4f}, {diff_data.max():.4f}]")
print(f"Diffraction data mean: {diff_data.mean():.4f}")
print(f"Diffraction data std: {diff_data.std():.4f}")

# Check a few diffraction patterns
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(8):
    ax = axes[i // 4, i % 4]
    im = ax.imshow(np.log10(diff_data[i] + 1e-6), cmap='viridis')
    ax.set_title(f'Pattern {i} (log scale)')
    ax.axis('off')

plt.suptitle('Sample Diffraction Patterns (Input to Model)')
plt.tight_layout()
plt.savefig('3e3_diffraction_samples.png', dpi=150)
plt.show()

# Now let's understand what the loss is measuring
print("\n" + "="*70)
print("LOSS ANALYSIS")
print("="*70)

print("""
The intensity_scaler_inv loss measures the MSE between:
1. INPUT: Measured diffraction patterns (amplitude squared)
2. OUTPUT: Model's predicted diffraction patterns after forward physics

This is a FORWARD MODEL CONSISTENCY loss, not a reconstruction quality loss!

The low loss (0.126) means the model can:
- Take diffraction patterns as input
- Reconstruct an object (even if wrong!)
- Forward propagate through physics
- Match the input diffraction patterns

But this DOESN'T mean the reconstructed object is correct!
A uniform box can still produce diffraction patterns that match
the measured data in terms of MSE, especially at low photon counts.

This is the "trivial solution" problem in phase retrieval!
""")

# Check if we have the actual reconstruction
import tensorflow as tf
from ptycho.workflows.components import load_inference_bundle

print("\nLoading model and getting internal reconstruction...")
model, config = load_inference_bundle(Path('photon_3e3_custom_study/pinn_run'))

# The model we loaded is 'diffraction_to_obj' which should output the object
# Let's check its output structure
print(f"\nModel output names: {model.output_names if hasattr(model, 'output_names') else 'Not available'}")
print(f"Model outputs: {len(model.outputs) if hasattr(model, 'outputs') else 'Unknown'}")

# Get a single sample prediction to analyze
single_diff = diff_data[0:1].reshape(1, 64, 64, 1)
single_pos = np.array([[[[0], [0]]]]) # Dummy position for single patch

print(f"\nRunning single sample through model...")
try:
    prediction = model.predict([single_diff, single_pos], verbose=0)
    
    if isinstance(prediction, list):
        print(f"Model returns {len(prediction)} outputs")
        for i, pred in enumerate(prediction):
            print(f"  Output {i}: shape {pred.shape}, range [{pred.min():.4f}, {pred.max():.4f}]")
    else:
        print(f"Single output: shape {prediction.shape}, range [{prediction.min():.4f}, {prediction.max():.4f}]")
        
except Exception as e:
    print(f"Error during prediction: {e}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("""
The problem is clear:

1. The model minimizes DIFFRACTION MATCHING loss, not OBJECT RECONSTRUCTION loss
2. At low photon counts (3000), many different objects can produce similar 
   diffraction patterns due to noise
3. The model found a "trivial" solution - a uniform object that roughly 
   matches the diffraction statistics
4. The loss is low because it measures diffraction consistency, not object quality

This is why the reconstruction looks bad despite low loss!

SOLUTIONS:
1. Need more photons for better signal-to-noise
2. Need regularization to prevent trivial solutions
3. Need to monitor actual reconstruction metrics (SSIM, PSNR) during training
4. May need to train longer or with different hyperparameters at low photon counts
""")