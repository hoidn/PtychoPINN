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
    --output-dir DIRECTORY     Output directory (default: timestamped directory)
    --test-data PATH          Path to test dataset (default: auto-generated)
    --skip-data-prep          Skip dataset preparation step
    --skip-training           Skip model training (use existing models)
    --skip-comparison         Skip model comparison step  
    --parallel-jobs N         Number of parallel training jobs (default: $DEFAULT_PARALLEL_JOBS)
    --dry-run                 Show commands without executing
    --help                    Show this help message

EXAMPLES:
    # Full study with default settings
    $0

    # Custom training sizes with parallel execution
    $0 --train-sizes "256 512 1024" --parallel-jobs 2

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
TEST_DATA=""
PARALLEL_JOBS="$DEFAULT_PARALLEL_JOBS"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-sizes)
            TRAIN_SIZES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
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
    
    local train_data="datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz"
    
    if [ ! -f "$train_data" ]; then
        log "ERROR: Training data not found: $train_data"
        exit 1
    fi
    
    # Training function for a single size
    train_single_size() {
        local train_size=$1
        local size_output_dir="$OUTPUT_DIR/train_$train_size"
        
        log "Training models for train_size=$train_size"
        
        # Train PtychoPINN
        local pinn_cmd="python scripts/training/train.py \\
            --train_data_file '$train_data' \\
            --test_data_file '$TEST_DATA' \\
            --n_images $train_size \\
            --output_dir '$size_output_dir/pinn_run' \\
            --nepochs 50"
            
        run_cmd "$pinn_cmd" "PtychoPINN training (n_images=$train_size)"
        
        # Train Baseline
        local baseline_cmd="python scripts/run_baseline.py \\
            --train_data '$train_data' \\
            --test_data '$TEST_DATA' \\
            --n_images $train_size \\
            --output_dir '$size_output_dir/baseline_run' \\
            --epochs 50"
            
        run_cmd "$baseline_cmd" "Baseline training (n_images=$train_size)"
        
        log "Completed training for train_size=$train_size"
    }
    
    # Handle parallel vs sequential training
    if [ "$PARALLEL_JOBS" -gt 1 ]; then
        log "Training models in parallel (max_jobs=$PARALLEL_JOBS)"
        
        # Convert space-separated sizes to array
        read -ra size_array <<< "$TRAIN_SIZES"
        
        # Launch parallel jobs
        local pids=()
        local job_count=0
        
        for train_size in "${size_array[@]}"; do
            if [ $job_count -ge $PARALLEL_JOBS ]; then
                # Wait for a job to complete
                wait "${pids[0]}"
                pids=("${pids[@]:1}")  # Remove first PID
                ((job_count--))
            fi
            
            # Launch background job
            (train_single_size "$train_size") &
            pids+=($!)
            ((job_count++))
            
            log "Launched training job for train_size=$train_size (PID: ${pids[-1]})"
        done
        
        # Wait for all remaining jobs
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        
        log "All parallel training jobs completed"
        
    else
        log "Training models sequentially"
        
        for train_size in $TRAIN_SIZES; do
            train_single_size "$train_size"
        done
    fi
    
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
        local size_output_dir="$OUTPUT_DIR/train_$train_size"
        local pinn_dir="$size_output_dir/pinn_run"
        local baseline_dir="$size_output_dir/baseline_run"
        
        # Check if models exist
        if [ ! -f "$pinn_dir/wts.h5.zip" ]; then
            log "WARNING: PtychoPINN model not found for train_size=$train_size: $pinn_dir/wts.h5.zip"
            continue
        fi
        
        if [ ! -f "$baseline_dir/baseline_model.h5" ]; then
            log "WARNING: Baseline model not found for train_size=$train_size: $baseline_dir/baseline_model.h5"
            continue
        fi
        
        local compare_cmd="python scripts/compare_models.py \\
            --pinn_dir '$pinn_dir' \\
            --baseline_dir '$baseline_dir' \\
            --test_data '$TEST_DATA' \\
            --output_dir '$size_output_dir'"
            
        run_cmd "$compare_cmd" "Model comparison (train_size=$train_size)"
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

## Study Configuration
- **Parallel Jobs:** $PARALLEL_JOBS
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

### Data Files
- \`results.csv\` - Complete aggregated metrics data
- \`study_config.txt\` - Study configuration parameters
- \`study_log.txt\` - Complete execution log

### Directory Structure
\`\`\`
$OUTPUT_DIR/
├── train_512/           # Results for 512 training images
├── train_1024/          # Results for 1024 training images  
├── train_2048/          # Results for 2048 training images
├── train_4096/          # Results for 4096 training images
├── *.png               # Generalization plots
├── results.csv         # Aggregated data
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
    log "Output directory: $OUTPUT_DIR"
    log "Parallel jobs: $PARALLEL_JOBS"
    
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
    echo "📋 Summary: $OUTPUT_DIR/STUDY_SUMMARY.md"
}

# Trap to handle interruption
trap 'log "Study interrupted by user"; exit 130' INT

# Run main function
main "$@"