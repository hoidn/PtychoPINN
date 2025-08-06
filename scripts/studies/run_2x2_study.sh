#!/bin/bash
#
# Execute the 2x2 Probe Parameterization Study.
#
# This script implements the second stage of the two-stage workflow, executing
# isolated training and evaluation for each experimental condition. Each condition
# runs in a completely separate subprocess to prevent configuration contamination.
#
# The script automatically detects prepared conditions and validates completeness
# before beginning execution.
#
# Usage:
#   # Quick test run (fewer epochs)
#   bash scripts/studies/run_2x2_study.sh --study-dir study_quick --quick-test
#   
#   # Full study execution
#   bash scripts/studies/run_2x2_study.sh --study-dir study_full
#   
#   # Parallel execution (if supported)
#   bash scripts/studies/run_2x2_study.sh --study-dir study_full --parallel

set -euo pipefail  # Strict error handling

# Default values
STUDY_DIR=""
QUICK_TEST=false
PARALLEL=false
LOG_LEVEL="INFO"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" >&2
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            ;;
        INFO)
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            ;;
        DEBUG)
            if [[ "$LOG_LEVEL" == "DEBUG" ]]; then
                echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
            fi
            ;;
    esac
}

# Error handler
error_exit() {
    log ERROR "$1"
    exit 1
}

# Usage function
usage() {
    cat << EOF
Usage: $0 --study-dir STUDY_DIR [OPTIONS]

Execute the 2x2 Probe Parameterization Study from prepared data.

Required Arguments:
  --study-dir STUDY_DIR    Directory containing prepared study data

Optional Arguments:
  --quick-test            Use quick test mode (fewer epochs)
  --parallel              Execute conditions in parallel (experimental)
  --log-level LEVEL       Logging level: DEBUG, INFO, WARN, ERROR (default: INFO)
  --help                  Show this help message

Example:
  $0 --study-dir study_quick --quick-test
  $0 --study-dir study_full
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --study-dir)
                STUDY_DIR="$2"
                shift 2
                ;;
            --quick-test)
                QUICK_TEST=true
                shift
                ;;
            --parallel)
                PARALLEL=true
                shift
                ;;
            --log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                error_exit "Unknown argument: $1. Use --help for usage information."
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$STUDY_DIR" ]]; then
        error_exit "Required argument --study-dir is missing. Use --help for usage information."
    fi
}

# Validate study directory and detect conditions
detect_conditions() {
    log INFO "Detecting experimental conditions in study directory..."
    
    if [[ ! -d "$STUDY_DIR" ]]; then
        error_exit "Study directory does not exist: $STUDY_DIR"
    fi
    
    # Look for condition directories matching pattern gs[12]_(idealized|hybrid)
    local conditions=()
    for condition_dir in "$STUDY_DIR"/gs*_*; do
        if [[ -d "$condition_dir" ]]; then
            local condition_name=$(basename "$condition_dir")
            if [[ "$condition_name" =~ ^gs[12]_(idealized|hybrid)$ ]]; then
                conditions+=("$condition_name")
                log DEBUG "Found condition directory: $condition_name"
            else
                log WARN "Ignoring non-matching directory: $condition_name"
            fi
        fi
    done
    
    if [[ ${#conditions[@]} -eq 0 ]]; then
        error_exit "No valid condition directories found in $STUDY_DIR"
    fi
    
    log INFO "Detected ${#conditions[@]} experimental conditions: ${conditions[*]}"
    
    # Validate completeness of each condition
    local missing_files=()
    for condition in "${conditions[@]}"; do
        local condition_path="$STUDY_DIR/$condition"
        local train_file="$condition_path/train_data.npz"
        local test_file="$condition_path/test_data.npz"
        
        if [[ ! -f "$train_file" ]]; then
            missing_files+=("$train_file")
        fi
        
        if [[ ! -f "$test_file" ]]; then
            missing_files+=("$test_file")
        fi
        
        log DEBUG "Condition $condition: train=${train_file}, test=${test_file}"
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log ERROR "Missing required data files:"
        for file in "${missing_files[@]}"; do
            log ERROR "  - $file"
        done
        error_exit "Study directory is incomplete. Please run prepare_2x2_study.py first."
    fi
    
    log INFO "All conditions validated successfully"
    
    # Export conditions for use by other functions
    export DETECTED_CONDITIONS="${conditions[*]}"
}

# Extract gridsize from condition name
get_gridsize() {
    local condition="$1"
    if [[ "$condition" =~ ^gs([12])_ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        error_exit "Cannot extract gridsize from condition: $condition"
    fi
}

# Train model for a single condition
train_condition() {
    local condition="$1"
    local condition_path="$STUDY_DIR/$condition"
    local train_file="$condition_path/train_data.npz"
    local test_file="$condition_path/test_data.npz"
    local model_output_dir="$condition_path/trained_model"
    local gridsize=$(get_gridsize "$condition")
    
    log INFO "Training model for condition: $condition (gridsize=$gridsize)"
    
    # Create model output directory
    mkdir -p "$model_output_dir"
    
    # Prepare training arguments
    # Note: --do_stitching is omitted to avoid stitching issues, evaluation handled separately
    local train_args=(
        --train_data_file "$train_file"
        --test_data_file "$test_file"
        --output_dir "$model_output_dir"
        --gridsize "$gridsize"
    )
    
    # Add quick test parameters
    if [[ "$QUICK_TEST" == true ]]; then
        train_args+=(--nepochs 2)
        train_args+=(--n_images 500)
        log DEBUG "Using quick test parameters for training"
    else
        train_args+=(--nepochs 50)
        train_args+=(--n_images 5000)
        log DEBUG "Using full parameters for training"
    fi
    
    # Skip stitching entirely during training - evaluation handled separately by compare_models.py
    log DEBUG "Skipping stitching during training - evaluation will be handled separately"
    
    # Setup logging for this condition
    local log_file="$condition_path/training.log"
    
    log INFO "Starting training subprocess for $condition..."
    log DEBUG "Training command: ptycho_train ${train_args[*]}"
    log DEBUG "Training log: $log_file"
    
    # Execute training in subprocess with complete isolation
    if ptycho_train "${train_args[@]}" > "$log_file" 2>&1; then
        log INFO "✓ Training completed successfully for $condition"
        
        # Verify model files were created
        if [[ -f "$model_output_dir/wts.h5.zip" ]]; then
            log DEBUG "Model weights found: $model_output_dir/wts.h5.zip"
        else
            error_exit "Training claimed success but model weights not found: $model_output_dir/wts.h5.zip"
        fi
    else
        log ERROR "✗ Training failed for $condition"
        log ERROR "Check training log: $log_file"
        error_exit "Training failed for condition: $condition"
    fi
}

# Evaluate model for a single condition
evaluate_condition() {
    local condition="$1"
    local condition_path="$STUDY_DIR/$condition"
    local test_file="$condition_path/test_data.npz"
    local model_dir="$condition_path/trained_model"
    local eval_output_dir="$condition_path/evaluation"
    local gridsize=$(get_gridsize "$condition")
    
    log INFO "Evaluating model for condition: $condition (gridsize=$gridsize)"
    
    # Verify model exists
    if [[ ! -f "$model_dir/wts.h5.zip" ]]; then
        error_exit "Model weights not found for evaluation: $model_dir/wts.h5.zip"
    fi
    
    # Create evaluation output directory
    mkdir -p "$eval_output_dir"
    
    # Prepare evaluation arguments (single-model evaluation as per Task 2.D spec)
    local eval_args=(
        --pinn_dir "$model_dir"
        --test_data "$test_file"
        --output_dir "$eval_output_dir"
        --gridsize "$gridsize"
    )
    
    # Setup logging for evaluation
    local log_file="$condition_path/evaluation.log"
    
    log INFO "Starting evaluation subprocess for $condition..."
    log DEBUG "Evaluation command: python scripts/compare_models.py ${eval_args[*]}"
    log DEBUG "Evaluation log: $log_file"
    
    # Execute evaluation in subprocess with complete isolation
    if python scripts/compare_models.py "${eval_args[@]}" > "$log_file" 2>&1; then
        log INFO "✓ Evaluation completed successfully for $condition"
        
        # Verify evaluation outputs were created
        if [[ -d "$eval_output_dir" ]] && [[ "$(ls -A "$eval_output_dir")" ]]; then
            log DEBUG "Evaluation outputs found in: $eval_output_dir"
        else
            log WARN "Evaluation claimed success but no outputs found in: $eval_output_dir"
        fi
    else
        log ERROR "✗ Evaluation failed for $condition"
        log ERROR "Check evaluation log: $log_file"
        error_exit "Evaluation failed for condition: $condition"
    fi
}

# Execute all conditions sequentially
execute_sequential() {
    local conditions=($DETECTED_CONDITIONS)
    
    log INFO "Executing ${#conditions[@]} conditions sequentially..."
    
    for condition in "${conditions[@]}"; do
        log INFO "Processing condition: $condition"
        
        # Train model
        train_condition "$condition"
        
        # Evaluate model
        evaluate_condition "$condition"
        
        log INFO "Completed condition: $condition"
    done
}

# Execute all conditions in parallel (experimental)
execute_parallel() {
    local conditions=($DETECTED_CONDITIONS)
    
    log INFO "Executing ${#conditions[@]} conditions in parallel..."
    log WARN "Parallel execution is experimental and may cause resource conflicts"
    
    local pids=()
    
    for condition in "${conditions[@]}"; do
        (
            log INFO "Starting parallel processing for: $condition"
            train_condition "$condition"
            evaluate_condition "$condition"
            log INFO "Completed parallel processing for: $condition"
        ) &
        pids+=($!)
    done
    
    # Wait for all parallel processes to complete
    local failed_conditions=()
    for i in "${!pids[@]}"; do
        local pid="${pids[i]}"
        local condition="${conditions[i]}"
        
        if wait "$pid"; then
            log INFO "✓ Parallel execution succeeded for: $condition"
        else
            log ERROR "✗ Parallel execution failed for: $condition"
            failed_conditions+=("$condition")
        fi
    done
    
    if [[ ${#failed_conditions[@]} -gt 0 ]]; then
        error_exit "Parallel execution failed for conditions: ${failed_conditions[*]}"
    fi
}

# Generate execution summary
generate_summary() {
    local conditions=($DETECTED_CONDITIONS)
    
    log INFO "Generating execution summary..."
    
    local summary_file="$STUDY_DIR/execution_summary.log"
    {
        echo "=========================================="
        echo "2x2 PROBE PARAMETERIZATION STUDY SUMMARY"
        echo "=========================================="
        echo "Study directory: $(realpath "$STUDY_DIR")"
        echo "Execution date: $(date)"
        echo "Quick test mode: $QUICK_TEST"
        echo "Parallel execution: $PARALLEL"
        echo "Number of conditions: ${#conditions[@]}"
        echo ""
        echo "Condition Results:"
        
        for condition in "${conditions[@]}"; do
            local condition_path="$STUDY_DIR/$condition"
            local model_file="$condition_path/trained_model/wts.h5.zip"
            local eval_dir="$condition_path/evaluation"
            local gridsize=$(get_gridsize "$condition")
            
            echo "  $condition (gridsize=$gridsize):"
            if [[ -f "$model_file" ]]; then
                echo "    ✓ Training: SUCCESS"
            else
                echo "    ✗ Training: FAILED"
            fi
            
            if [[ -d "$eval_dir" ]] && [[ "$(ls -A "$eval_dir")" ]]; then
                echo "    ✓ Evaluation: SUCCESS"
            else
                echo "    ✗ Evaluation: FAILED"
            fi
        done
        
        echo ""
        echo "=========================================="
    } > "$summary_file"
    
    log INFO "Execution summary saved to: $summary_file"
    
    # Display summary to console
    cat "$summary_file"
}

# Main execution function
main() {
    log INFO "Starting 2x2 Probe Parameterization Study execution"
    log INFO "Study directory: $STUDY_DIR"
    log INFO "Quick test mode: $QUICK_TEST"
    log INFO "Parallel execution: $PARALLEL"
    
    # Detect and validate conditions
    detect_conditions
    
    # Execute based on parallel flag
    if [[ "$PARALLEL" == true ]]; then
        execute_parallel
    else
        execute_sequential
    fi
    
    # Generate summary
    generate_summary
    
    log INFO "2x2 Study execution completed successfully!"
}

# Parse arguments and execute
parse_args "$@"
main