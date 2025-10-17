# run_axis_test_training.py

import os
import glob
import subprocess
import dill
import numpy as np
import time

def run_batch_training():
    """
    Loops through variant datasets, trains the model on each, and reports the
    final loss to identify the best coordinate system.
    """
    variants_dir = 'datasets/fly64_coord_variants'
    output_base_dir = 'outputs/coord_axis_test'
    
    dataset_paths = sorted(glob.glob(os.path.join(variants_dir, '*.npz')))
    
    if not dataset_paths:
        print(f"ERROR: No dataset variants found in {variants_dir}. Did you run generate_coord_variants.py?")
        return

    results = {}

    print("--- Starting Batch Training for Coordinate Axis Test ---")
    print(f"Found {len(dataset_paths)} datasets to test")
    print(f"Using gridsize=2 with 10 epochs each")

    for i, path in enumerate(dataset_paths):
        # Extract the transformation name from the filename for labeling
        filename = os.path.basename(path)
        try:
            # Assumes format like 'fly001_64_train_converted_identity.npz'
            label = filename.split('converted_')[1].replace('.npz', '')
        except IndexError:
            print(f"Warning: Could not parse label from filename {filename}. Skipping.")
            continue
            
        print(f"\n--- [{i+1}/{len(dataset_paths)}] Training on Variant: {label} ---")
        print(f"Dataset: {path}")

        # Create timestamped output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base_dir, f"{timestamp}_{label}")
        
        # Command-line arguments for modern training script
        args = [
            'python', 'scripts/training/train.py',
            '--train_data_file', path,
            '--test_data_file', path,  # Use same file for test split for this experiment
            '--gridsize', '2',
            '--nepochs', '10',
            '--output_dir', output_dir,
            '--n_images', '1000',  # Limit training data for speed
        ]

        print(f"Running: {' '.join(args)}")

        # Run the training as a subprocess
        start_time = time.time()
        try:
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            elapsed_time = time.time() - start_time
            
            print(f"Training completed in {elapsed_time:.1f} seconds")
            
            # After training, find the history file to extract the loss
            history_path = os.path.join(output_dir, 'history.dill')
            
            if os.path.exists(history_path):
                with open(history_path, 'rb') as f:
                    history = dill.load(f)
                
                # Get the final training loss
                final_loss = history['loss'][-1]
                results[label] = final_loss
                print(f"SUCCESS: Variant '{label}' finished with final loss: {final_loss:.4f}")
            else:
                print(f"WARNING: No history.dill found at {history_path}")
                results[label] = np.inf

        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            print(f"ERROR: Training failed for variant '{label}' after {elapsed_time:.1f} seconds")
            print("Command:", ' '.join(args))
            print("Return code:", e.returncode)
            if e.stdout:
                print("Stdout:", e.stdout[-500:])  # Last 500 chars
            if e.stderr:
                print("Stderr:", e.stderr[-500:])  # Last 500 chars
            results[label] = np.inf # Mark as failed
        except Exception as e:
            print(f"An unexpected error occurred for variant '{label}': {e}")
            results[label] = np.inf


    # --- Final Report ---
    print("\n\n" + "="*60)
    print("COORDINATE AXIS TEST RESULTS")
    print("="*60)
    print(f"{'Transformation':<20} | {'Final Training Loss':<15} | {'Status'}")
    print("-" * 60)

    # Sort results by loss value
    sorted_results = sorted(results.items(), key=lambda item: item[1])
    
    for label, loss in sorted_results:
        if loss == np.inf:
            status = "FAILED"
            loss_str = "∞"
        else:
            status = "✓ BEST" if loss == sorted_results[0][1] and loss != np.inf else "✓"
            loss_str = f"{loss:.4f}"
        
        print(f"{label:<20} | {loss_str:<15} | {status}")
    
    print("-" * 60)
    
    # Find the best result
    valid_results = [(label, loss) for label, loss in sorted_results if loss != np.inf]
    if valid_results:
        best_transform, best_loss = valid_results[0]
        print(f"\n🎯 CONCLUSION: The transformation '{best_transform}' yielded the lowest loss ({best_loss:.4f}).")
        print("This is the most likely candidate for the correct coordinate system.")
        
        # Save results to file
        results_file = os.path.join(output_base_dir, 'coordinate_test_results.txt')
        with open(results_file, 'w') as f:
            f.write("Coordinate Axis Test Results\n")
            f.write("="*40 + "\n\n")
            for label, loss in sorted_results:
                f.write(f"{label}: {loss}\n")
            f.write(f"\nBest transformation: {best_transform} (loss: {best_loss:.4f})\n")
        
        print(f"\nResults saved to: {results_file}")
    else:
        print("\n❌ ERROR: All training runs failed. Check the error messages above.")


if __name__ == '__main__':
    run_batch_training()