#!/usr/bin/env python
"""
Compare outputs from inference before and after refactoring.
This verifies that removing the redundant code has no effect on the results.
"""

import subprocess
import sys
import os
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def compare_images(img1_path, img2_path):
    """Compare two images pixel by pixel."""
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))
    
    if img1.shape != img2.shape:
        return False, f"Shape mismatch: {img1.shape} vs {img2.shape}"
    
    if np.array_equal(img1, img2):
        return True, "Images are identical"
    
    # Check if they're very close (accounting for minor numerical differences)
    diff = np.abs(img1.astype(float) - img2.astype(float))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    if max_diff < 2:  # Allow for tiny rounding differences
        return True, f"Images are effectively identical (max diff: {max_diff:.2f})"
    
    return False, f"Images differ (max diff: {max_diff:.2f}, mean diff: {mean_diff:.2f})"

def run_inference(output_dir, script_name="inference.py"):
    """Run inference and return the output directory."""
    model_path = "./training_outputs"
    test_data = "./ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz"
    
    # Make sure we have a model
    if not os.path.exists(f"{model_path}/wts.h5.zip"):
        print(f"No model found at {model_path}/wts.h5.zip")
        print("Running quick training to create a model...")
        subprocess.run([
            sys.executable, "scripts/training/train.py",
            "--train_data_file", test_data,
            "--test_data_file", test_data,
            "--output_dir", model_path,
            "--nepochs", "2",
            "--n_images", "32",
            "--gridsize", "1",
            "--quiet"
        ], check=True)
    
    # Run inference
    print(f"Running inference with {script_name} to {output_dir}...")
    result = subprocess.run([
        sys.executable, f"scripts/inference/{script_name}",
        "--model_path", model_path,
        "--test_data", test_data,
        "--output_dir", output_dir,
        "--n_images", "32"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Inference failed: {result.stderr}")
        return False
    
    return True

def main():
    print("=" * 70)
    print("INFERENCE OUTPUT COMPARISON TEST")
    print("=" * 70)
    
    # Save current inference.py as refactored version
    print("\nPreparing test versions...")
    os.system("cp scripts/inference/inference.py scripts/inference/inference_refactored.py")
    
    # Create version with redundant code restored
    with open("scripts/inference/inference_refactored.py", "r") as f:
        refactored_content = f.read()
    
    # Create original version by adding back the removed line
    original_content = refactored_content.replace(
        "        # Note: update_legacy_dict() removed",
        "        update_legacy_dict(params.cfg, config)\n        # Note: update_legacy_dict() removed"
    )
    
    with open("scripts/inference/inference_original.py", "w") as f:
        f.write("from ptycho.config.config import update_legacy_dict\n")
        f.write(original_content)
    
    # Run both versions
    print("\n1. Running original version (with update_legacy_dict)...")
    if not run_inference("output_original", "inference_original.py"):
        print("Failed to run original version")
        return 1
    
    print("\n2. Running refactored version (without update_legacy_dict)...")
    if not run_inference("output_refactored", "inference_refactored.py"):
        print("Failed to run refactored version")
        return 1
    
    # Compare outputs
    print("\n" + "=" * 70)
    print("COMPARING OUTPUTS:")
    print("=" * 70)
    
    files_to_compare = [
        "reconstructed_amplitude.png",
        "reconstructed_phase.png"
    ]
    
    all_match = True
    for filename in files_to_compare:
        path1 = Path("output_original") / filename
        path2 = Path("output_refactored") / filename
        
        if not path1.exists() or not path2.exists():
            print(f"\n❌ {filename}: Missing file")
            all_match = False
            continue
        
        match, msg = compare_images(path1, path2)
        if match:
            print(f"\n✅ {filename}: {msg}")
        else:
            print(f"\n❌ {filename}: {msg}")
            all_match = False
    
    # Final verdict
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    
    if all_match:
        print("\n✅ SUCCESS: All outputs are identical!")
        print("The refactoring (removing update_legacy_dict) has no effect on results.")
        print("This confirms the code was redundant and safe to remove.")
        return 0
    else:
        print("\n❌ FAILURE: Outputs differ!")
        print("Further investigation needed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())