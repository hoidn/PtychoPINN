#!/usr/bin/env python3
"""
Extract final training losses from the batch training output by re-running 
individual training commands and capturing their output.
"""

import subprocess
import re
import glob
import os

def extract_final_loss_from_output(output_text):
    """Extract the final training loss from TensorFlow training output."""
    # Look for the last epoch's training loss
    # Pattern: "10/10 [==============================] - 4s 420ms/step - loss: 1234.5678"
    pattern = r'\d+/\d+ \[=+\] - \d+s.*?- loss: ([\d.e+-]+)'
    matches = re.findall(pattern, output_text)
    
    if matches:
        # Return the last (final) loss value
        return float(matches[-1])
    
    # Alternative pattern for loss without step info
    alt_pattern = r'loss: ([\d.e+-]+)'
    alt_matches = re.findall(alt_pattern, output_text)
    
    if alt_matches:
        return float(alt_matches[-1])
    
    return None

def run_single_training_for_loss(dataset_path, variant_name):
    """Run a single training and extract the final loss."""
    print(f"Extracting loss for {variant_name}...")
    
    # Use shorter training for quick loss extraction
    args = [
        'python', 'scripts/training/train.py',
        '--train_data_file', dataset_path,
        '--test_data_file', dataset_path,
        '--gridsize', '2',
        '--nepochs', '3',  # Just 3 epochs for quick results
        '--output_dir', f'outputs/loss_extract_{variant_name}',
        '--n_images', '200',  # Smaller dataset for speed
    ]
    
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            # Extract loss from stdout
            final_loss = extract_final_loss_from_output(result.stdout)
            if final_loss is not None:
                print(f"  {variant_name}: {final_loss:.4f}")
                return final_loss
            else:
                print(f"  {variant_name}: Could not parse loss from output")
                return None
        else:
            print(f"  {variant_name}: Training failed (return code {result.returncode})")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  {variant_name}: Training timed out")
        return None
    except Exception as e:
        print(f"  {variant_name}: Error - {e}")
        return None

def main():
    """Extract losses from all 8 coordinate variants."""
    variants_dir = 'datasets/fly64_coord_variants'
    dataset_paths = sorted(glob.glob(os.path.join(variants_dir, '*.npz')))
    
    if not dataset_paths:
        print(f"ERROR: No dataset variants found in {variants_dir}")
        return
    
    print("Extracting final training losses from all coordinate variants...")
    print("=" * 60)
    
    results = {}
    
    for path in dataset_paths:
        filename = os.path.basename(path)
        variant_name = filename.split('converted_')[1].replace('.npz', '')
        
        final_loss = run_single_training_for_loss(path, variant_name)
        results[variant_name] = final_loss
    
    # Print results
    print("\n" + "=" * 60)
    print("COORDINATE TRANSFORMATION LOSS COMPARISON")
    print("=" * 60)
    print(f"{'Transformation':<20} | {'Final Training Loss':<15} | {'Status'}")
    print("-" * 60)
    
    # Sort by loss (valid results first, then None)
    valid_results = [(name, loss) for name, loss in results.items() if loss is not None]
    invalid_results = [(name, loss) for name, loss in results.items() if loss is None]
    
    sorted_valid = sorted(valid_results, key=lambda x: x[1])
    all_results = sorted_valid + invalid_results
    
    for name, loss in all_results:
        if loss is not None:
            status = "✓ BEST" if loss == sorted_valid[0][1] else "✓"
            loss_str = f"{loss:.4f}"
        else:
            status = "FAILED"
            loss_str = "N/A"
        
        print(f"{name:<20} | {loss_str:<15} | {status}")
    
    print("-" * 60)
    
    if sorted_valid:
        best_name, best_loss = sorted_valid[0]
        print(f"\n🎯 BEST RESULT: '{best_name}' with loss {best_loss:.4f}")
        print(f"This suggests the correct coordinate transformation for gridsize=2.")
    else:
        print("\n❌ No valid results obtained.")

if __name__ == '__main__':
    main()