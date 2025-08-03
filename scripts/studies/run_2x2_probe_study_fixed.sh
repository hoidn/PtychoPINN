#!/bin/bash
#
# Fixed version of run_2x2_probe_study.sh that ensures ground truth is included
# in comparison plots by converting simulated data to standard format
#

set -e  # Exit on error

# Parse command line arguments
OUTPUT_DIR=""
DATASET=""
QUICK_TEST=false
SKIP_COMPLETED=false
PARALLEL_JOBS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --skip-completed)
            SKIP_COMPLETED=true
            shift
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --output-dir <dir> --dataset <dataset.npz> [--quick-test] [--skip-completed] [--parallel-jobs N]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output-dir is required"
    exit 1
fi

if [[ -z "$DATASET" ]] && [[ "$QUICK_TEST" != true ]]; then
    echo "Error: --dataset is required (unless using --quick-test)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set dataset for quick test
if [[ "$QUICK_TEST" == true ]]; then
    DATASET="datasets/fly/fly001_transposed.npz"
    echo "Quick test mode: using $DATASET"
fi

echo "Starting 2x2 Probe Study (Fixed Version)"
echo "Output directory: $OUTPUT_DIR"
echo "Dataset: $DATASET"

# Step 1: Extract default probe
echo "Step 1: Extracting default probe..."
python -c "
import numpy as np
data = np.load('$DATASET')
probe = data['probeGuess']
np.save('$OUTPUT_DIR/default_probe.npy', probe)
print(f'Default probe saved: shape={probe.shape}, dtype={probe.dtype}')
"

# Step 2: Create hybrid probe
echo "Step 2: Creating hybrid probe..."
python scripts/tools/create_hybrid_probe.py \
    "$OUTPUT_DIR/default_probe.npy" \
    "$DATASET" \
    --output "$OUTPUT_DIR/hybrid_probe.npy"

# Function to run one experimental arm
run_experiment() {
    local gridsize=$1
    local probe_type=$2
    local probe_path=$3
    local exp_dir="$OUTPUT_DIR/gs${gridsize}_${probe_type}"
    
    if [[ "$SKIP_COMPLETED" == true ]] && [[ -f "$exp_dir/metrics_summary.csv" ]]; then
        echo "Skipping completed experiment: $exp_dir"
        return
    fi
    
    echo "Running experiment: gridsize=$gridsize, probe=$probe_type"
    mkdir -p "$exp_dir"
    
    # Simulate data with specified probe
    echo "  Simulating data..."
    python scripts/simulation/simulate_and_save.py \
        --input-file "$DATASET" \
        --probe-file "$probe_path" \
        --output-file "$exp_dir/simulated_raw.npz" \
        --n-images $(if [[ "$QUICK_TEST" == true ]]; then echo 500; else echo 5000; fi) \
        --gridsize "$gridsize"
    
    # CRITICAL FIX: Convert to standard format
    echo "  Converting to standard format..."
    python scripts/tools/transpose_rename_convert_tool.py \
        "$exp_dir/simulated_raw.npz" \
        "$exp_dir/simulated_data.npz"
    
    # Split into train/test
    echo "  Splitting dataset..."
    python scripts/tools/split_dataset_tool.py \
        "$exp_dir/simulated_data.npz" \
        "$exp_dir" \
        --split-fraction 0.8
    
    # Train model
    echo "  Training model..."
    ptycho_train \
        --train_data_file "$exp_dir/simulated_data_train.npz" \
        --test_data_file "$exp_dir/simulated_data_test.npz" \
        --output_dir "$exp_dir/model" \
        --gridsize "$gridsize" \
        --nepochs $(if [[ "$QUICK_TEST" == true ]]; then echo 2; else echo 50; fi)
    
    # Evaluate model
    echo "  Evaluating model..."
    mkdir -p "$exp_dir/evaluation"
    python scripts/compare_models.py \
        --pinn_dir "$exp_dir/model" \
        --test_data "$exp_dir/simulated_data_test.npz" \
        --output_dir "$exp_dir/evaluation" \
        --save-debug-images
    
    # Extract metrics
    if [[ -f "$exp_dir/evaluation/comparison_metrics.csv" ]]; then
        cp "$exp_dir/evaluation/comparison_metrics.csv" "$exp_dir/metrics_summary.csv"
    fi
}

# Run experiments (potentially in parallel)
if [[ "$PARALLEL_JOBS" -gt 1 ]]; then
    echo "Running experiments in parallel with $PARALLEL_JOBS jobs..."
    export -f run_experiment
    export OUTPUT_DIR QUICK_TEST SKIP_COMPLETED
    
    parallel -j "$PARALLEL_JOBS" run_experiment ::: 1 2 ::: default hybrid ::: \
        "$OUTPUT_DIR/default_probe.npy" "$OUTPUT_DIR/hybrid_probe.npy" \
        "$OUTPUT_DIR/default_probe.npy" "$OUTPUT_DIR/hybrid_probe.npy"
else
    # Run sequentially
    run_experiment 1 default "$OUTPUT_DIR/default_probe.npy"
    run_experiment 1 hybrid "$OUTPUT_DIR/hybrid_probe.npy"
    run_experiment 2 default "$OUTPUT_DIR/default_probe.npy"
    run_experiment 2 hybrid "$OUTPUT_DIR/hybrid_probe.npy"
fi

# Aggregate results
echo "Aggregating results..."
python -c "
import pandas as pd
import numpy as np
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
results = []

for gridsize in [1, 2]:
    for probe_type in ['default', 'hybrid']:
        exp_dir = output_dir / f'gs{gridsize}_{probe_type}'
        metrics_file = exp_dir / 'metrics_summary.csv'
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            df['gridsize'] = gridsize
            df['probe_type'] = probe_type
            results.append(df)

if results:
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(output_dir / 'study_summary.csv', index=False)
    print('Results summary:')
    print(combined)
else:
    print('No results found!')
"

echo "2x2 Probe Study Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Check comparison_plot.png in each evaluation directory for ground truth comparisons"