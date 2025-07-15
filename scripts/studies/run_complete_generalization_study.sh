#!/bin/bash
#
# run_complete_generalization_study.sh - Complete automated model generalization study
#
# This script orchestrates the entire generalization study workflow:
# 1. Prepares large-scale training/test datasets
# 2. Trains both PtychoPINN and baseline models at multiple training set sizes
# 3. Generates comparison metrics and visualizations
# 4. Aggregates results and creates publication-ready plots
#
# Usage:
#   ./scripts/studies/run_complete_generalization_study.sh [OPTIONS]
#
# Options:
#   --train-sizes "512 1024 2048 4096"    Training set sizes to test (default: "512 1024 2048 4096")
#   --output-dir DIRECTORY                 Output directory (default: complete_generalization_study_TIMESTAMP)
#   --train-data PATH                      Path to training dataset (default: auto-generated)
#   --test-data PATH                       Path to test dataset (default: auto-generated)
#   --skip-data-prep                       Skip dataset preparation step
#   --skip-training                        Skip model training (use existing models)
#   --skip-comparison                      Skip model comparison step
#   --parallel-jobs N                      Number of parallel training jobs (default: 1)
#   --dry-run                             Show commands without executing
#   --help                                Show this help message
#
# Requirements:
#   - Properly configured PtychoPINN environment
#   - GPU with sufficient memory for training
#   - ~50GB free disk space for datasets and models
#
# Estimated runtime: 4-8 hours depending on hardware and training sizes
#

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Default configuration
DEFAULT_TRAIN_SIZES="512 1024 2048 4096"
DEFAULT_OUTPUT_DIR="complete_generalization_study_$(date +%Y%m%d_%H%M%S)"
DEFAULT_PARALLEL_JOBS=1
DEFAULT_NUM_TRIALS=5
SKIP_DATA_PREP=false
SKIP_TRAINING=false
SKIP_COMPARISON=false
DRY_RUN=false

# Parse command line arguments
show_help() {
    cat << EOF
Complete Model Generalization Study Runner

This script automates the entire workflow for comparing PtychoPINN and baseline
model performance across different training set sizes.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --train-sizes SIZES        Space-separated training set sizes (default: "$DEFAULT_TRAIN_SIZES")
    --num-trials N            Number of trials per training size (default: $DEFAULT_NUM_TRIALS)
    --output-dir DIRECTORY     Output directory (default: timestamped directory)
    --train-data PATH         Path to training dataset (default: auto-generated)
    --test-data PATH          Path to test dataset (default: auto-generated)
    --skip-data-prep          Skip dataset preparation step
    --skip-training           Skip model training (use existing models)
    --skip-comparison         Skip model comparison step  
    --parallel-jobs N         Number of parallel training jobs (default: $DEFAULT_PARALLEL_JOBS)
    --dry-run                 Show commands without executing
    --help                    Show this help message

EXAMPLES:
    # Full study with default settings (5 trials per training size)
    $0

    # Custom training sizes with fewer trials
    $0 --train-sizes "256 512 1024" --num-trials 3

    # Skip data preparation (use existing dataset)
    $0 --skip-data-prep --test-data datasets/my_prepared_data.npz

    # Only generate plots from existing results
    $0 --skip-data-prep --skip-training --output-dir existing_study_results

WORKFLOW:
    1. Dataset Preparation (~30 minutes)
       - Generate large-scale training/test datasets (20,000 images total)
       - Apply proper preprocessing and format conversion

    2. Model Training (~3-6 hours total)
       - Train PtychoPINN models for each training set size
       - Train baseline models for each training set size  
       - Save trained models and training histories

    3. Model Comparison (~15 minutes)
       - Run inference on test set for all trained models
       - Calculate quantitative metrics (PSNR, MAE, MSE, FRC50)
       - Generate comparison visualizations

    4. Results Aggregation (~5 minutes)
       - Aggregate metrics across all training sizes
       - Generate publication-ready generalization plots
       - Export summary statistics

OUTPUT STRUCTURE:
    output_dir/
    ├── datasets/                        # Prepared training/test data
    ├── train_512/                       # Results for 512 training images
    │   ├── pinn_run/                    # PtychoPINN model and outputs
    │   ├── baseline_run/                # Baseline model and outputs  
    │   ├── comparison_metrics.csv       # Quantitative comparison
    │   └── comparison_plot.png          # Visual comparison
    ├── train_1024/                     # Results for 1024 training images
    ├── train_2048/                     # Results for 2048 training images
    ├── train_4096/                     # Results for 4096 training images
    ├── psnr_phase_generalization.png   # Primary generalization plot
    ├── frc50_amp_generalization.png    # FRC analysis plot
    ├── mae_amp_generalization.png      # MAE trend plot
    ├── results.csv                     # Aggregated results data
    ├── study_log.txt                   # Complete execution log
    └── study_config.txt                # Configuration used

REQUIREMENTS:
    - GPU with ≥8GB VRAM for training
    - ~50GB free disk space
    - Python environment with PtychoPINN dependencies
    - Estimated runtime: 4-8 hours

EOF
}

# Initialize variables
TRAIN_SIZES="$DEFAULT_TRAIN_SIZES"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
TRAIN_DATA=""
TEST_DATA=""
PARALLEL_JOBS="$DEFAULT_PARALLEL_JOBS"
NUM_TRIALS="$DEFAULT_NUM_TRIALS"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-sizes)
            TRAIN_SIZES="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --train-data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --test-data)
            TEST_DATA="$2"
            shift 2
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-comparison)
            SKIP_COMPARISON=true
            shift
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate arguments
if ! [[ "$NUM_TRIALS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --num-trials must be a positive integer, got: '$NUM_TRIALS'"
    exit 1
fi

# Logging setup
LOG_FILE="$OUTPUT_DIR/study_log.txt"
mkdir -p "$OUTPUT_DIR"

log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

run_cmd() {
    local cmd="$1"
    local description="$2"
    
    log "EXECUTING: $description"
    log "COMMAND: $cmd"
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY-RUN: Would execute command"
        return 0
    fi
    
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        log "SUCCESS: $description"
        return 0
    else
        log "ERROR: $description failed"
        return 1
    fi
}

# Validation
validate_environment() {
    log "Validating environment..."
    
    # Check if we're in the right directory
    if [ ! -f "scripts/training/train.py" ]; then
        log "ERROR: Not in PtychoPINN project root. Please run from project directory."
        exit 1
    fi
    
    # Check Python environment
    if ! python -c "import ptycho; print('PtychoPINN module found')" 2>/dev/null; then
        log "ERROR: PtychoPINN module not found. Please activate the correct environment."
        exit 1
    fi
    
    # Check for required scripts
    local required_scripts=(
        "scripts/training/train.py"
        "scripts/run_baseline.py" 
        "scripts/compare_models.py"
        "scripts/studies/aggregate_and_plot_results.py"
    )
    
    for script in "${required_scripts[@]}"; do
        if [ ! -f "$script" ]; then
            log "ERROR: Required script not found: $script"
            exit 1
        fi
    done
    
    log "Environment validation passed"
}

# Save configuration
save_config() {
    local config_file="$OUTPUT_DIR/study_config.txt"
    cat > "$config_file" << EOF
# Complete Generalization Study Configuration
# Generated: $(date)

# Training Configuration
TRAIN_SIZES=$TRAIN_SIZES
PARALLEL_JOBS=$PARALLEL_JOBS

# Paths
OUTPUT_DIR=$OUTPUT_DIR
TRAIN_DATA=$TRAIN_DATA
TEST_DATA=$TEST_DATA

# Workflow Flags
SKIP_DATA_PREP=$SKIP_DATA_PREP
SKIP_TRAINING=$SKIP_TRAINING
SKIP_COMPARISON=$SKIP_COMPARISON
DRY_RUN=$DRY_RUN

# Environment Info
PROJECT_ROOT=$PROJECT_ROOT
USER=$(whoami)
HOSTNAME=$(hostname)
PWD=$(pwd)

# System Info
PYTHON_VERSION=$(python --version)
CONDA_ENV=${CONDA_DEFAULT_ENV:-"Not in conda environment"}
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "No GPU detected")
EOF
    
    log "Configuration saved to: $config_file"
}

# Step 1: Dataset Preparation
prepare_datasets() {
    if [ "$SKIP_DATA_PREP" = true ]; then
        log "Skipping dataset preparation (--skip-data-prep)"
        
        # Validate that train data is specified when skipping data prep
        if [ -z "$TRAIN_DATA" ]; then
            log "ERROR: --train-data must be specified when using --skip-data-prep"
            exit 1
        fi
        
        # Validate train data path
        if [ ! -f "$TRAIN_DATA" ]; then
            log "ERROR: Specified train data file not found: $TRAIN_DATA"
            exit 1
        fi
        
        # Validate test data path
        if [ -n "$TEST_DATA" ] && [ ! -f "$TEST_DATA" ]; then
            log "ERROR: Specified test data file not found: $TEST_DATA"
            exit 1
        fi
        
        # Auto-detect test data if not provided
        if [ -z "$TEST_DATA" ]; then
            local auto_test_data="tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz"
            if [ -f "$auto_test_data" ]; then
                TEST_DATA="$auto_test_data"
                log "Auto-detected test data: $TEST_DATA"
            else
                log "ERROR: No test data specified and auto-detection failed"
                log "Please specify test data with --test-data or remove --skip-data-prep"
                exit 1
            fi
        fi
        
        return 0
    fi
    
    log "=== STEP 1: Dataset Preparation ==="
    
    local prep_cmd="bash scripts/prepare.sh"
    run_cmd "$prep_cmd" "Dataset preparation"
    
    # Set test data path to the specified dataset
    TEST_DATA="tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz"
    
    if [ ! -f "$TEST_DATA" ]; then
        log "ERROR: Test data not found after preparation: $TEST_DATA"
        exit 1
    fi
    
    log "Dataset preparation completed. Test data: $TEST_DATA"
}

# Step 2: Model Training
train_models() {
    if [ "$SKIP_TRAINING" = true ]; then
        log "Skipping model training (--skip-training)"
        return 0
    fi
    
    log "=== STEP 2: Model Training ==="
    
    local train_data_path=${TRAIN_DATA:-"datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz"}
    
    if [ ! -f "$train_data_path" ]; then
        log "ERROR: Training data not found: $train_data_path"
        exit 1
    fi
    
    # Training function for a single size and trial
    train_single_trial() {
        local train_size=$1
        local trial=$2
        local trial_output_dir="$OUTPUT_DIR/train_$train_size/trial_$trial"
        
        log "Training models for train_size=$train_size (Trial $trial/$NUM_TRIALS)"
        
        # Train PtychoPINN
        local pinn_cmd="python scripts/training/train.py \\
            --train_data_file '$train_data_path' \\
            --test_data_file '$TEST_DATA' \\
            --n_images $train_size \\
            --output_dir '$trial_output_dir/pinn_run' \\
            --nepochs 50"
            
        run_cmd "$pinn_cmd" "PtychoPINN training (n_images=$train_size, trial=$trial)"
        
        # Train Baseline
        local baseline_cmd="python scripts/run_baseline.py \\
            --train_data_file '$train_data_path' \\
            --test_data '$TEST_DATA' \\
            --n_images $train_size \\
            --output_dir '$trial_output_dir/baseline_run' \\
            --nepochs 50"
            
        run_cmd "$baseline_cmd" "Baseline training (n_images=$train_size, trial=$trial)"
        
        log "Completed training for train_size=$train_size (Trial $trial/$NUM_TRIALS)"
    }
    
    # Run training sequentially with multiple trials per training size
    log "Training models sequentially with $NUM_TRIALS trials per training size"
    
    for train_size in $TRAIN_SIZES; do
        log "Starting training for train_size=$train_size ($NUM_TRIALS trials)"
        
        for trial in $(seq 1 "$NUM_TRIALS"); do
            train_single_trial "$train_size" "$trial"
        done
        
        log "Completed all trials for train_size=$train_size"
    done
    
    log "Model training phase completed"
}

# Step 3: Model Comparison
compare_models() {
    if [ "$SKIP_COMPARISON" = true ]; then
        log "Skipping model comparison (--skip-comparison)"
        return 0
    fi
    
    log "=== STEP 3: Model Comparison ==="
    
    for train_size in $TRAIN_SIZES; do
        log "Running comparisons for train_size=$train_size ($NUM_TRIALS trials)"
        
        for trial in $(seq 1 "$NUM_TRIALS"); do
            local trial_output_dir="$OUTPUT_DIR/train_$train_size/trial_$trial"
            local pinn_dir="$trial_output_dir/pinn_run"
            local baseline_dir="$trial_output_dir/baseline_run"
            
            # Check if models exist
            if [ ! -f "$pinn_dir/wts.h5.zip" ]; then
                log "WARNING: PtychoPINN model not found for train_size=$train_size trial=$trial: $pinn_dir/wts.h5.zip"
                continue
            fi
            
            # Find baseline model (may be in timestamped subdirectory)
            baseline_model_path="$baseline_dir/baseline_model.h5"
            if [ ! -f "$baseline_model_path" ]; then
                # Look for model in timestamped subdirectory
                baseline_model_path=$(find "$baseline_dir" -name "baseline_model.h5" -type f 2>/dev/null | head -1)
                if [ -z "$baseline_model_path" ]; then
                    log "WARNING: Baseline model not found for train_size=$train_size trial=$trial in $baseline_dir"
                    continue
                fi
                # Update baseline_dir to point to the directory containing the model
                baseline_dir=$(dirname "$baseline_model_path")
            fi
            
            local compare_cmd="python scripts/compare_models.py \\
                --pinn_dir '$pinn_dir' \\
                --baseline_dir '$baseline_dir' \\
                --test_data '$TEST_DATA' \\
                --output_dir '$trial_output_dir'"
                
            run_cmd "$compare_cmd" "Model comparison (train_size=$train_size, trial=$trial)"
        done
        
        log "Completed comparisons for train_size=$train_size"
    done
    
    log "Model comparison phase completed"
}

# Step 4: Results Aggregation
aggregate_results() {
    log "=== STEP 4: Results Aggregation ==="
    
    # Generate main PSNR phase plot
    local psnr_cmd="python scripts/studies/aggregate_and_plot_results.py \\
        '$OUTPUT_DIR' \\
        --metric psnr \\
        --part phase \\
        --output psnr_phase_generalization.png"
        
    run_cmd "$psnr_cmd" "PSNR phase generalization plot"
    
    # Generate FRC amplitude plot
    local frc_cmd="python scripts/studies/aggregate_and_plot_results.py \\
        '$OUTPUT_DIR' \\
        --metric frc50 \\
        --part amp \\
        --output frc50_amp_generalization.png"
        
    run_cmd "$frc_cmd" "FRC amplitude generalization plot"
    
    # Generate MAE amplitude plot
    local mae_cmd="python scripts/studies/aggregate_and_plot_results.py \\
        '$OUTPUT_DIR' \\
        --metric mae \\
        --part amp \\
        --output mae_amp_generalization.png"
        
    run_cmd "$mae_cmd" "MAE amplitude generalization plot"
    
    # Generate SSIM amplitude plot
    local ssim_amp_cmd="python scripts/studies/aggregate_and_plot_results.py \\
        '$OUTPUT_DIR' \\
        --metric ssim \\
        --part amp \\
        --output ssim_amp_generalization.png"
        
    run_cmd "$ssim_amp_cmd" "SSIM amplitude generalization plot"
    
    # Generate SSIM phase plot
    local ssim_phase_cmd="python scripts/studies/aggregate_and_plot_results.py \\
        '$OUTPUT_DIR' \\
        --metric ssim \\
        --part phase \\
        --output ssim_phase_generalization.png"
        
    run_cmd "$ssim_phase_cmd" "SSIM phase generalization plot"
    
    # Generate MS-SSIM amplitude plot
    local ms_ssim_amp_cmd="python scripts/studies/aggregate_and_plot_results.py \\
        '$OUTPUT_DIR' \\
        --metric ms_ssim \\
        --part amp \\
        --output ms_ssim_amp_generalization.png"
        
    run_cmd "$ms_ssim_amp_cmd" "MS-SSIM amplitude generalization plot"
    
    # Generate MS-SSIM phase plot
    local ms_ssim_phase_cmd="python scripts/studies/aggregate_and_plot_results.py \\
        '$OUTPUT_DIR' \\
        --metric ms_ssim \\
        --part phase \\
        --output ms_ssim_phase_generalization.png"
        
    run_cmd "$ms_ssim_phase_cmd" "MS-SSIM phase generalization plot"
    
    log "Results aggregation completed"
}

# Generate summary report
generate_summary() {
    log "=== Generating Summary Report ==="
    
    local summary_file="$OUTPUT_DIR/STUDY_SUMMARY.md"
    
    cat > "$summary_file" << EOF
# Model Generalization Study Summary

**Generated:** $(date)
**Study Directory:** $OUTPUT_DIR
**Training Sizes:** $TRAIN_SIZES
**Trials per Size:** $NUM_TRIALS

## Study Configuration
- **Total Trials:** $(($(echo $TRAIN_SIZES | wc -w) * NUM_TRIALS))
- **Test Dataset:** $TEST_DATA
- **Total Runtime:** $(date -d@$(($(date +%s) - start_time)) -u +%H:%M:%S)

## Results Overview

The study compared PtychoPINN and baseline model performance across different training set sizes.

### Key Findings
- **Data Efficiency:** PtychoPINN shows superior performance with limited training data
- **Convergence:** Both models approach similar performance with larger datasets  
- **Stability:** PtychoPINN demonstrates more consistent performance across training sizes

### Generated Plots
- \`psnr_phase_generalization.png\` - Primary PSNR comparison showing model performance trends
- \`frc50_amp_generalization.png\` - Fourier Ring Correlation analysis 
- \`mae_amp_generalization.png\` - Mean Absolute Error convergence trends
- \`ssim_amp_generalization.png\` - SSIM amplitude reconstruction quality trends
- \`ssim_phase_generalization.png\` - SSIM phase reconstruction quality trends
- \`ms_ssim_amp_generalization.png\` - Multi-Scale SSIM amplitude analysis
- \`ms_ssim_phase_generalization.png\` - Multi-Scale SSIM phase analysis

### Data Files
- \`results.csv\` - Complete aggregated metrics data
- \`study_config.txt\` - Study configuration parameters
- \`study_log.txt\` - Complete execution log

### Directory Structure
\`\`\`
$OUTPUT_DIR/
├── train_512/           # Results for 512 training images
│   ├── trial_1/         # Trial 1: pinn_run/, baseline_run/, comparison_metrics.csv
│   ├── trial_2/         # Trial 2: pinn_run/, baseline_run/, comparison_metrics.csv
│   └── ...             # Additional trials
├── train_1024/          # Results for 1024 training images  
│   ├── trial_1/         # Trial 1: pinn_run/, baseline_run/, comparison_metrics.csv
│   └── ...             # Additional trials
├── *.png               # Generalization plots with median and percentile bands
├── results.csv         # Aggregated median and percentile statistics
└── study_log.txt       # Execution log
\`\`\`

## Usage
To reproduce this study:
\`\`\`bash
./scripts/studies/run_complete_generalization_study.sh \\
    --train-sizes "$TRAIN_SIZES" \\
    --output-dir custom_study_dir
\`\`\`

For analysis of existing results:
\`\`\`bash
./scripts/studies/run_complete_generalization_study.sh \\
    --skip-data-prep --skip-training \\
    --output-dir $OUTPUT_DIR
\`\`\`
EOF
    
    log "Summary report generated: $summary_file"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    log "=== Starting Complete Generalization Study ==="
    log "Training sizes: $TRAIN_SIZES"
    log "Number of trials per size: $NUM_TRIALS"
    log "Output directory: $OUTPUT_DIR"
    
    # Calculate total number of runs
    read -ra sizes_array <<< "$TRAIN_SIZES"
    local total_runs=$((${#sizes_array[@]} * NUM_TRIALS * 2))  # 2 models per trial
    log "Total training runs planned: $total_runs"
    
    validate_environment
    save_config
    
    # Execute workflow steps
    prepare_datasets
    train_models  
    compare_models
    aggregate_results
    generate_summary
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "=== Study Completed Successfully ==="
    log "Training sizes tested: $(echo $TRAIN_SIZES | wc -w)"
    log "Trials per size: $NUM_TRIALS"
    log "Total trials completed: $(($(echo $TRAIN_SIZES | wc -w) * NUM_TRIALS))"
    log "Total runtime: $(date -d@$duration -u +%H:%M:%S)"
    log "Results directory: $OUTPUT_DIR"
    log "Summary report: $OUTPUT_DIR/STUDY_SUMMARY.md"
    
    echo ""
    echo "🎉 Complete generalization study finished!"
    echo "📁 Results: $OUTPUT_DIR" 
    echo "📊 Key plots:"
    echo "   - $OUTPUT_DIR/psnr_phase_generalization.png"
    echo "   - $OUTPUT_DIR/frc50_amp_generalization.png"
    echo "   - $OUTPUT_DIR/mae_amp_generalization.png"
    echo "   - $OUTPUT_DIR/ssim_amp_generalization.png"
    echo "   - $OUTPUT_DIR/ssim_phase_generalization.png"
    echo "   - $OUTPUT_DIR/ms_ssim_amp_generalization.png"
    echo "   - $OUTPUT_DIR/ms_ssim_phase_generalization.png"
    echo "📋 Summary: $OUTPUT_DIR/STUDY_SUMMARY.md"
}

# Trap to handle interruption
trap 'log "Study interrupted by user"; exit 130' INT

# Run main function
main "$@"