#!/bin/bash
#
# run_2x2_probe_study.sh - Automated 2x2 probe parameterization study
#
# This script orchestrates a comprehensive study to investigate the impact of
# different probe functions (default vs. hybrid) on reconstruction quality
# across different overlap constraints (gridsize=1 vs. gridsize=2).
#
# The study creates a 2x2 experimental matrix:
#   - Gridsize 1 with default probe
#   - Gridsize 1 with hybrid probe
#   - Gridsize 2 with default probe
#   - Gridsize 2 with hybrid probe
#
# Each experimental arm includes:
#   1. Probe generation (extracting default or creating hybrid)
#   2. Simulation with the specified probe and gridsize
#   3. Training both PtychoPINN and baseline models
#   4. Evaluation and comparison of results
#
# Usage:
#   ./scripts/studies/run_2x2_probe_study.sh --output-dir DIRECTORY [OPTIONS]
#
# Required Options:
#   --output-dir DIRECTORY     Output directory for study results
#
# Optional Options:
#   --dataset PATH            Source dataset for experimental probe phase (default: datasets/fly/fly001_transposed.npz)
#   --quick-test              Run in quick test mode (fewer images, epochs)
#   --parallel-jobs N         Number of parallel jobs (default: 1)
#   --skip-completed          Skip already completed steps
#   --help                    Show this help message
#
# Quick Test Mode:
#   When --quick-test is specified:
#   - N_TRAIN=256 (instead of 5000)
#   - N_TEST=128 (instead of 1000)  
#   - EPOCHS=5 (instead of 50)
#
# Examples:
#   # Full study
#   ./scripts/studies/run_2x2_probe_study.sh --output-dir probe_study_results
#
#   # Quick test to verify pipeline
#   ./scripts/studies/run_2x2_probe_study.sh --output-dir test_probe_study --quick-test
#
#   # Resume interrupted study
#   ./scripts/studies/run_2x2_probe_study.sh --output-dir probe_study_results --skip-completed
#
#   # Parallel execution with 4 jobs
#   ./scripts/studies/run_2x2_probe_study.sh --output-dir probe_study_results --parallel-jobs 4
#
# Output Structure:
#   output_dir/
#   ├── idealized_probe.npy        # Generated idealized probe
#   ├── hybrid_probe.npy           # Generated hybrid probe  
#   ├── gs1_idealized/             # Gridsize 1, idealized probe
#   │   ├── simulated_data.npz
#   │   ├── model/
#   │   ├── evaluation/
#   │   └── metrics_summary.csv
#   ├── gs1_hybrid/                # Gridsize 1, hybrid probe
#   ├── gs2_idealized/             # Gridsize 2, idealized probe
#   └── gs2_hybrid/                # Gridsize 2, hybrid probe
#
# Requirements:
#   - Properly configured PtychoPINN environment
#   - GPU with sufficient memory for training
#   - ~20GB free disk space for full study
#   - Phase 1 & 2 tools: create_hybrid_probe.py, simulate_and_save.py with --probe-file
#
# Estimated Runtime:
#   - Quick test: ~30 minutes
#   - Full study: 4-6 hours (depending on hardware)
#

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Function to show usage
show_usage() {
    # Extract usage from script header
    sed -n '/^# run_2x2_probe_study.sh/,/^$/p' "$0" | grep '^#' | sed 's/^# *//'
    exit 0
}

# Check if no arguments provided
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided"
    echo "Try '$0 --help' for usage information"
    exit 1
fi

# Default values
OUTPUT_DIR=""
PHASE_SOURCE_DATASET="datasets/fly/fly001_transposed.npz"
QUICK_TEST=false
PARALLEL_JOBS=1
SKIP_COMPLETED=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            if [[ -z "$2" ]] || [[ "$2" =~ ^-- ]]; then
                echo "Error: --output-dir requires a directory argument"
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset)
            if [[ -z "$2" ]] || [[ "$2" =~ ^-- ]]; then
                echo "Error: --dataset requires a file path argument"
                exit 1
            fi
            PHASE_SOURCE_DATASET="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --parallel-jobs)
            if [[ -z "$2" ]] || [[ "$2" =~ ^-- ]]; then
                echo "Error: --parallel-jobs requires a numeric argument"
                exit 1
            fi
            if ! [[ "$2" =~ ^[0-9]+$ ]] || [[ "$2" -le 0 ]]; then
                echo "Error: --parallel-jobs must be a positive integer, got: $2"
                exit 1
            fi
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --skip-completed)
            SKIP_COMPLETED=true
            shift
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Try '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output-dir is required"
    echo "Try '$0 --help' for usage information"
    exit 1
fi

# Check if phase source dataset exists
if [[ ! -f "$PHASE_SOURCE_DATASET" ]]; then
    echo "Error: Phase source dataset file not found: $PHASE_SOURCE_DATASET"
    exit 1
fi

# Add [QUICK TEST] prefix if in quick test mode
if [[ "$QUICK_TEST" == true ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_QUICK_TEST"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log configuration
echo "=========================================="
echo "2x2 Probe Parameterization Study"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Phase source dataset: $PHASE_SOURCE_DATASET"
echo "Quick test mode: $QUICK_TEST"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Skip completed: $SKIP_COMPLETED"
echo "Start time: $(date)"
echo "=========================================="

# Define experimental matrix
GRIDSIZES=(1 2)
PROBE_TYPES=("idealized" "hybrid")

# Set parameters based on mode
if [[ "$QUICK_TEST" == true ]]; then
    N_TRAIN=256
    N_TEST=128
    EPOCHS=5
    echo ""
    echo "Quick test parameters:"
    echo "  N_TRAIN: $N_TRAIN"
    echo "  N_TEST: $N_TEST"
    echo "  EPOCHS: $EPOCHS"
else
    N_TRAIN=5000
    N_TEST=1000
    EPOCHS=50
    echo ""
    echo "Full study parameters:"
    echo "  N_TRAIN: $N_TRAIN"
    echo "  N_TEST: $N_TEST"
    echo "  EPOCHS: $EPOCHS"
fi

echo "  Gridsizes: ${GRIDSIZES[@]}"
echo "  Probe types: ${PROBE_TYPES[@]}"
echo "=========================================="
echo ""

# Checkpoint detection function
is_step_complete() {
    local step_marker="$1"
    if [[ -f "$step_marker" ]]; then
        if [[ "$SKIP_COMPLETED" == true ]]; then
            echo "  [SKIP] Step already completed (found $step_marker)"
            return 0  # true - step is complete
        else
            echo "  [WARN] Found existing marker $step_marker but --skip-completed not set"
            echo "         Re-running this step..."
            return 1  # false - re-run the step
        fi
    else
        return 1  # false - step not complete
    fi
}

# Helper function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ============================================
# SYNTHETIC OBJECT GENERATION
# ============================================

log_message "Generating synthetic lines object..."

# Generate synthetic lines input data
SYNTHETIC_INPUT_PATH="$OUTPUT_DIR/synthetic_lines_input.npz"
if is_step_complete "$SYNTHETIC_INPUT_PATH"; then
    log_message "Synthetic lines object already generated"
else
    log_message "Creating synthetic lines object with probe..."
    python -c "
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '.')))

from ptycho import params as p
from ptycho.diffsim import sim_object_image
from ptycho.probe import get_default_probe

# Set up parameters for synthetic object
N_obj = 224  # Object size (3.5x probe size for 64x64 probe)
# Set data source in global params
p.set('data_source', 'lines')
# Create synthetic lines object
obj = sim_object_image(N_obj)
# Remove extra dimension if present (lines returns (H, W, 1))
if obj.ndim == 3 and obj.shape[2] == 1:
    obj = obj[:, :, 0]
# Get default probe
probe = get_default_probe(64, fmt='np')
# Ensure probe is complex
if not np.iscomplexobj(probe):
    probe = probe.astype(np.complex64)
# Save as NPZ file
np.savez('$SYNTHETIC_INPUT_PATH', 
         objectGuess=obj,
         probeGuess=probe)
print(f'Created synthetic lines object with shape {obj.shape}')
print(f'  Object amplitude range: [{np.abs(obj).min():.3f}, {np.abs(obj).max():.3f}]')
print(f'  Object phase std: {np.angle(obj).std():.3f}')
"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to generate synthetic lines object"
        exit 1
    fi
fi

# ============================================
# PROBE GENERATION
# ============================================

log_message "Starting probe generation..."

# Create idealized probe using synthetic generation
IDEALIZED_PROBE_PATH="$OUTPUT_DIR/idealized_probe.npy"
if is_step_complete "$IDEALIZED_PROBE_PATH"; then
    log_message "Idealized probe already created"
else
    log_message "Extracting idealized probe from synthetic lines input..."
    python -c "
import numpy as np
# Extract probe from synthetic lines input data
data = np.load('$SYNTHETIC_INPUT_PATH')
probe = data['probeGuess']
# Save the idealized probe
np.save('$IDEALIZED_PROBE_PATH', probe)
print(f'Saved idealized probe with shape {probe.shape}, dtype {probe.dtype}')
print(f'  Mean amplitude: {np.abs(probe).mean():.4f}')
print(f'  Phase std: {np.angle(probe).std():.4f}')
"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to create idealized probe"
        exit 1
    fi
fi

# Generate hybrid probe
HYBRID_PROBE_PATH="$OUTPUT_DIR/hybrid_probe.npy"
if is_step_complete "$HYBRID_PROBE_PATH"; then
    log_message "Hybrid probe already generated"
else
    log_message "Generating hybrid probe..."
    # Use idealized probe amplitude with experimental phase from source dataset
    python scripts/tools/create_hybrid_probe.py \
        "$IDEALIZED_PROBE_PATH" \
        "$PHASE_SOURCE_DATASET" \
        --output "$HYBRID_PROBE_PATH"
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to generate hybrid probe"
        exit 1
    fi
    
    # Validate the hybrid probe
    python -c "
import numpy as np
probe = np.load('$HYBRID_PROBE_PATH')
print(f'Loaded hybrid probe with shape {probe.shape}, dtype {probe.dtype}')
print(f'  Mean amplitude: {np.abs(probe).mean():.4f}')
print(f'  Phase std: {np.angle(probe).std():.4f}')
if not np.all(np.isfinite(probe)):
    raise ValueError('Hybrid probe contains non-finite values!')
"
    if [[ $? -ne 0 ]]; then
        echo "Error: Hybrid probe validation failed"
        exit 1
    fi
fi

log_message "Probe generation complete"
echo ""

# ============================================
# SIMULATION FUNCTIONS
# ============================================

run_simulation() {
    local gridsize="$1"
    local probe_type="$2"
    local output_subdir="$3"
    
    log_message "Running simulation for gridsize=$gridsize, probe=$probe_type"
    
    # Determine probe path
    if [[ "$probe_type" == "idealized" ]]; then
        local probe_path="$IDEALIZED_PROBE_PATH"
    else
        local probe_path="$HYBRID_PROBE_PATH"
    fi
    
    # Check if already complete
    local sim_marker="$output_subdir/.simulation_done"
    if is_step_complete "$sim_marker"; then
        return 0
    fi
    
    # Create output subdirectory
    mkdir -p "$output_subdir"
    
    # Build simulation command using synthetic lines input
    local sim_cmd="python scripts/simulation/simulate_and_save.py"
    sim_cmd="$sim_cmd --input-file $SYNTHETIC_INPUT_PATH"
    sim_cmd="$sim_cmd --probe-file $probe_path"
    sim_cmd="$sim_cmd --output-file $output_subdir/simulated_data.npz"
    sim_cmd="$sim_cmd --n-images $N_TRAIN"
    sim_cmd="$sim_cmd --gridsize $gridsize"
    
    # Log the command
    echo "  Command: $sim_cmd"
    
    # Run simulation
    if $sim_cmd > "$output_subdir/simulation.log" 2>&1; then
        touch "$sim_marker"
        log_message "  Simulation completed successfully"
        
        # Extract and log key statistics (optional, don't fail if this errors)
        python -c "
import numpy as np
try:
    data = np.load('$output_subdir/simulated_data.npz')
    # Handle both possible key names for diffraction data
    diff_key = 'diffraction' if 'diffraction' in data else 'diff3d'
    if diff_key in data:
        print(f'  Simulated data shape: {data[diff_key].shape}')
    if 'xcoords' in data:
        print(f'  Number of scan positions: {len(data[\"xcoords\"])}')
except Exception as e:
    print(f'  Note: Could not extract statistics: {e}')
" || true  # Don't fail on statistics extraction
    else
        log_message "  ERROR: Simulation failed! Check $output_subdir/simulation.log"
        return 1
    fi
}

# ============================================
# TRAINING FUNCTIONS
# ============================================

run_training() {
    local output_subdir="$1"
    
    log_message "Running training for $output_subdir"
    
    # Check if already complete
    local train_marker="$output_subdir/.training_done"
    if is_step_complete "$train_marker"; then
        return 0
    fi
    
    # Build training command
    local train_cmd="ptycho_train"
    train_cmd="$train_cmd --train_data_file $output_subdir/simulated_data.npz"
    train_cmd="$train_cmd --output_dir $output_subdir/model"
    train_cmd="$train_cmd --nepochs $EPOCHS"
    train_cmd="$train_cmd --batch_size 32"
    
    # For now, we train only PtychoPINN model
    # Add baseline training later if needed
    train_cmd="$train_cmd --model_type pinn"
    
    # Log the command
    echo "  Command: $train_cmd"
    
    # Run training with progress tracking
    local log_file="$output_subdir/training.log"
    log_message "  Training output will be saved to: $log_file"
    
    # Use tee for interactive mode if not in parallel execution
    if [[ "$PARALLEL_JOBS" -eq 1 ]]; then
        # Interactive mode - show output and save to log
        if $train_cmd 2>&1 | tee "$log_file"; then
            touch "$train_marker"
            log_message "  Training completed successfully"
        else
            log_message "  ERROR: Training failed! Check $log_file"
            return 1
        fi
    else
        # Background mode - only save to log
        if $train_cmd > "$log_file" 2>&1; then
            touch "$train_marker"
            log_message "  Training completed successfully"
        else
            log_message "  ERROR: Training failed! Check $log_file"
            return 1
        fi
    fi
    
    # Extract key metrics from training
    if [[ -f "$output_subdir/model/history.dill" ]]; then
        python -c "
import dill
with open('$output_subdir/model/history.dill', 'rb') as f:
    history = dill.load(f)
final_loss = history['loss'][-1] if 'loss' in history else 'N/A'
print(f'  Final training loss: {final_loss}')
"
    fi
}

# ============================================
# EVALUATION FUNCTIONS
# ============================================

run_evaluation() {
    local output_subdir="$1"
    
    log_message "Running evaluation for $output_subdir"
    
    # Check if already complete
    local eval_marker="$output_subdir/.evaluation_done"
    if is_step_complete "$eval_marker"; then
        return 0
    fi
    
    # Create test data path by replacing train count with test count
    # This assumes the test data follows the same naming pattern
    local test_data_path="${output_subdir}/simulated_data.npz"
    
    # For evaluation, we need to create a test subset
    # For now, we'll use the same data but limit to N_TEST images
    log_message "  Creating test subset with $N_TEST images..."
    python -c "
import numpy as np
# Load full dataset
data = np.load('$test_data_path')
# Create test subset
test_data = {}
# Handle both possible key names for diffraction data
diff_key = 'diffraction' if 'diffraction' in data else 'diff3d'
for key in data.files:
    if key in [diff_key, 'xcoords', 'ycoords']:
        test_data[key] = data[key][:$N_TEST]
    else:
        test_data[key] = data[key]
# Ensure we use standard key name for compatibility
if diff_key == 'diff3d' and 'diff3d' in test_data:
    test_data['diffraction'] = test_data.pop('diff3d')
# Save test subset
test_path = '$output_subdir/test_data.npz'
np.savez(test_path, **test_data)
print(f'Created test subset at {test_path}')
"
    
    # Build evaluation command
    # Single-model evaluation (no baseline comparison)
    local eval_cmd="python scripts/compare_models.py"
    eval_cmd="$eval_cmd --pinn_dir $output_subdir/model"
    eval_cmd="$eval_cmd --test_data $output_subdir/test_data.npz"
    eval_cmd="$eval_cmd --output_dir $output_subdir/evaluation"
    eval_cmd="$eval_cmd --n-test-images $N_TEST"
    
    # Log the command
    echo "  Command: $eval_cmd"
    
    # Run evaluation
    local log_file="$output_subdir/evaluation.log"
    if $eval_cmd > "$log_file" 2>&1; then
        touch "$eval_marker"
        log_message "  Evaluation completed successfully"
        
        # Copy metrics summary
        if [[ -f "$output_subdir/evaluation/comparison_metrics.csv" ]]; then
            cp "$output_subdir/evaluation/comparison_metrics.csv" "$output_subdir/metrics_summary.csv"
            
            # Add experiment metadata to the metrics file
            python -c "
import pandas as pd
metrics = pd.read_csv('$output_subdir/metrics_summary.csv')
# Extract gridsize and probe_type from path
import os
dirname = os.path.basename('$output_subdir')
parts = dirname.split('_')
gridsize = parts[0].replace('gs', '')
probe_type = parts[1]
# Add metadata columns
metrics['gridsize'] = gridsize
metrics['probe_type'] = probe_type
metrics['experiment'] = dirname
# Save updated metrics
metrics.to_csv('$output_subdir/metrics_summary.csv', index=False)
print(f'  Updated metrics with experiment metadata')
"
        fi
    else
        log_message "  ERROR: Evaluation failed! Check $log_file"
        return 1
    fi
}

# ============================================
# MAIN ORCHESTRATION LOOP
# ============================================

log_message "Starting 2x2 experimental matrix..."

# Arrays to track background jobs for parallel execution
declare -a JOB_PIDS=()
declare -a JOB_NAMES=()

# Function to wait for jobs with proper error handling
wait_for_jobs() {
    local failed=0
    log_message "Waiting for ${#JOB_PIDS[@]} parallel jobs to complete..."
    
    for i in "${!JOB_PIDS[@]}"; do
        local pid="${JOB_PIDS[$i]}"
        local name="${JOB_NAMES[$i]}"
        
        if wait "$pid"; then
            log_message "  Job $name (PID $pid) completed successfully"
        else
            log_message "  ERROR: Job $name (PID $pid) failed!"
            failed=$((failed + 1))
        fi
    done
    
    # Clear arrays
    JOB_PIDS=()
    JOB_NAMES=()
    
    if [[ $failed -gt 0 ]]; then
        log_message "ERROR: $failed jobs failed!"
        return 1
    fi
    return 0
}

# Function to run a complete pipeline for one experimental arm
run_experiment_arm() {
    local gridsize="$1"
    local probe_type="$2"
    local arm_name="gs${gridsize}_${probe_type}"
    local output_subdir="$OUTPUT_DIR/$arm_name"
    
    log_message "="
    log_message "Starting experiment: $arm_name"
    log_message "="
    
    # Run simulation
    if ! run_simulation "$gridsize" "$probe_type" "$output_subdir"; then
        log_message "ERROR: Simulation failed for $arm_name"
        return 1
    fi
    
    # Run training
    if ! run_training "$output_subdir"; then
        log_message "ERROR: Training failed for $arm_name"
        return 1
    fi
    
    # Run evaluation
    if ! run_evaluation "$output_subdir"; then
        log_message "ERROR: Evaluation failed for $arm_name"
        return 1
    fi
    
    log_message "Completed experiment: $arm_name"
    return 0
}

# Main execution loop
if [[ "$PARALLEL_JOBS" -gt 1 ]]; then
    log_message "Running experiments in parallel with $PARALLEL_JOBS jobs..."
    
    # Track active jobs
    active_jobs=0
    
    for gridsize in "${GRIDSIZES[@]}"; do
        for probe_type in "${PROBE_TYPES[@]}"; do
            # Wait if we've reached the parallel job limit
            while [[ $active_jobs -ge $PARALLEL_JOBS ]]; do
                # Wait for any job to finish
                for i in "${!JOB_PIDS[@]}"; do
                    if ! kill -0 "${JOB_PIDS[$i]}" 2>/dev/null; then
                        # Job finished
                        wait "${JOB_PIDS[$i]}"
                        local exit_code=$?
                        if [[ $exit_code -ne 0 ]]; then
                            log_message "ERROR: Job ${JOB_NAMES[$i]} failed with exit code $exit_code"
                        fi
                        # Remove from arrays
                        unset 'JOB_PIDS[$i]'
                        unset 'JOB_NAMES[$i]'
                        active_jobs=$((active_jobs - 1))
                        break
                    fi
                done
                sleep 1
            done
            
            # Launch new job
            arm_name="gs${gridsize}_${probe_type}"
            log_message "Launching parallel job for $arm_name..."
            run_experiment_arm "$gridsize" "$probe_type" &
            local pid=$!
            JOB_PIDS+=("$pid")
            JOB_NAMES+=("$arm_name")
            active_jobs=$((active_jobs + 1))
        done
    done
    
    # Wait for all remaining jobs
    wait_for_jobs
else
    # Sequential execution
    log_message "Running experiments sequentially..."
    
    for gridsize in "${GRIDSIZES[@]}"; do
        for probe_type in "${PROBE_TYPES[@]}"; do
            if ! run_experiment_arm "$gridsize" "$probe_type"; then
                log_message "ERROR: Experiment failed, continuing with next..."
                # Continue with other experiments even if one fails
            fi
            echo ""  # Blank line between experiments
        done
    done
fi

# ============================================
# FINAL SUMMARY
# ============================================

log_message "=========================================="
log_message "Study Complete!"
log_message "=========================================="
log_message "Output directory: $OUTPUT_DIR"
log_message "End time: $(date)"
echo ""

# Generate summary of results
log_message "Generating results summary..."
python -c "
import os
import pandas as pd
import glob

output_dir = '$OUTPUT_DIR'
metrics_files = glob.glob(os.path.join(output_dir, '*/metrics_summary.csv'))

if metrics_files:
    # Combine all metrics
    all_metrics = []
    for f in metrics_files:
        try:
            df = pd.read_csv(f)
            all_metrics.append(df)
        except:
            pass
    
    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        summary_path = os.path.join(output_dir, 'study_summary.csv')
        combined.to_csv(summary_path, index=False)
        print(f'Created summary at: {summary_path}')
        print('')
        print('Results Summary:')
        print(combined[['experiment', 'amplitude_psnr', 'phase_psnr']].to_string(index=False))
    else:
        print('No valid metrics found')
else:
    print('No metrics files found')
"

log_message "Done!"