#!/bin/bash
# run_probe_generalization_study.sh - Orchestration script for 2x2 probe generalization study
#
# This script automates the execution of all four experimental arms of the probe
# generalization study, comparing idealized vs experimental probes across gridsize=1 vs gridsize=2.
#
# Usage: ./scripts/studies/run_probe_generalization_study.sh <output_dir> [options]
#
# Experimental Arms:
#   1. Idealized Probe / Gridsize 1
#   2. Idealized Probe / Gridsize 2  
#   3. Experimental Probe / Gridsize 1
#   4. Experimental Probe / Gridsize 2
#
# Each arm generates training comparison results including metrics for PtychoPINN performance analysis.

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Parse command line arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <output_dir> [--skip-if-exists] [--verbose] [--dry-run]"
    echo ""
    echo "Required arguments:"
    echo "  output_dir        Base directory for all experimental outputs"
    echo ""
    echo "Optional arguments:"
    echo "  --skip-if-exists  Skip experimental arms that have already completed"
    echo "  --verbose         Enable detailed logging and progress monitoring"
    echo "  --dry-run         Print commands without executing (for testing)"
    echo ""
    echo "Examples:"
    echo "  $0 probe_generalization_results"
    echo "  $0 probe_generalization_results --skip-if-exists --verbose"
    echo ""
    echo "Expected Input Data:"
    echo "  probe_study_data/ideal_train.npz    - Idealized probe training data"
    echo "  probe_study_data/ideal_test.npz     - Idealized probe test data"
    echo "  probe_study_data/exp_train.npz      - Experimental probe training data"
    echo "  probe_study_data/exp_test.npz       - Experimental probe test data"
    echo ""
    echo "Required Configurations:"
    echo "  configs/comparison_config_gs1.yaml  - Gridsize 1 configuration"
    echo "  configs/comparison_config_gs2.yaml  - Gridsize 2 configuration"
    echo ""
    exit 1
fi

OUTPUT_DIR="$1"
SKIP_IF_EXISTS=false
VERBOSE=false
DRY_RUN=false

# Parse optional arguments
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-if-exists)
            SKIP_IF_EXISTS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Logging function
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message"
    
    # Also log to file if verbose (only if file exists)
    if [ "$VERBOSE" = true ] && [ -f "$OUTPUT_DIR/experiment_log.txt" ]; then
        echo "[$timestamp] [$level] $message" >> "$OUTPUT_DIR/experiment_log.txt"
    fi
}

# Resource validation function
validate_resources() {
    log_message "INFO" "Validating system resources and data files..."
    
    # Check input data files (converted training format)
    local required_files=(
        "probe_study_data/ideal_train_converted.npz"
        "probe_study_data/ideal_test_converted.npz" 
        "probe_study_data/exp_train_converted.npz"
        "probe_study_data/exp_test_converted.npz"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_message "ERROR" "Required data file not found: $file"
            exit 1
        fi
    done
    log_message "INFO" "All required data files found"
    
    # Check configuration files
    local config_files=(
        "configs/comparison_config_gs1.yaml"
        "configs/comparison_config_gs2.yaml"
    )
    
    for config in "${config_files[@]}"; do
        if [ ! -f "$config" ]; then
            log_message "ERROR" "Required configuration file not found: $config"
            exit 1
        fi
        
        # Validate YAML syntax
        python -c "import yaml; yaml.safe_load(open('$config'))" || {
            log_message "ERROR" "Invalid YAML syntax in $config"
            exit 1
        }
    done
    log_message "INFO" "All configuration files validated"
    
    # Check disk space (estimate 2GB per arm = 8GB total, require 10GB minimum)
    available_space=$(df . | tail -1 | awk '{print $4}')
    required_space=10485760  # 10GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        log_message "WARN" "Low disk space: ${available_space}KB available, ${required_space}KB recommended"
    else
        log_message "INFO" "Adequate disk space available: ${available_space}KB"
    fi
    
    # Check write permissions
    mkdir -p "$OUTPUT_DIR"
    if [ ! -w "$OUTPUT_DIR" ]; then
        log_message "ERROR" "Cannot write to output directory: $OUTPUT_DIR"
        exit 1
    fi
    log_message "INFO" "Output directory is writable"
    
    log_message "INFO" "Resource validation complete"
}

# Function to check if experimental arm is already completed
is_arm_completed() {
    local arm_dir="$1"
    if [ -f "$arm_dir/comparison_metrics.csv" ]; then
        return 0  # Completed
    else
        return 1  # Not completed
    fi
}

# Function to execute experimental arm
execute_arm() {
    local arm_name="$1"
    local train_data="$2"
    local test_data="$3"
    local config_file="$4"
    local arm_output_dir="$5"
    
    log_message "INFO" "Starting experimental arm: $arm_name"
    log_message "INFO" "  Train data: $train_data"
    log_message "INFO" "  Test data: $test_data"
    log_message "INFO" "  Config: $config_file"
    log_message "INFO" "  Output: $arm_output_dir"
    
    # Check if arm is already completed
    if [ "$SKIP_IF_EXISTS" = true ] && is_arm_completed "$arm_output_dir"; then
        log_message "INFO" "Arm '$arm_name' already completed, skipping..."
        return 0
    fi
    
    # Backup original config and temporarily use gridsize-specific config
    local original_config="configs/comparison_config.yaml"
    local backup_config="configs/comparison_config.yaml.backup.$$"
    
    log_message "INFO" "Backing up original config and using: $config_file"
    if [ "$DRY_RUN" != true ]; then
        cp "$original_config" "$backup_config"
        cp "$config_file" "$original_config"
    fi
    
    # Construct the comparison command
    local comparison_command="./scripts/run_comparison.sh '$train_data' '$test_data' '$arm_output_dir' --n-train-images 2000"
    
    log_message "INFO" "Executing command: $comparison_command"
    
    if [ "$DRY_RUN" != true ]; then
        # Execute the command
        local start_time=$(date +%s)
        
        if eval "$comparison_command"; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log_message "INFO" "Arm '$arm_name' completed successfully in ${duration} seconds"
            
            # Validate output
            if is_arm_completed "$arm_output_dir"; then
                log_message "INFO" "Arm '$arm_name' validation: PASSED"
            else
                log_message "ERROR" "Arm '$arm_name' validation: FAILED - missing comparison_metrics.csv"
                # Restore original config before returning error
                if [ "$DRY_RUN" != true ] && [ -f "$backup_config" ]; then
                    mv "$backup_config" "$original_config"
                fi
                return 1
            fi
        else
            log_message "ERROR" "Arm '$arm_name' execution FAILED"
            # Restore original config before returning error  
            if [ "$DRY_RUN" != true ] && [ -f "$backup_config" ]; then
                mv "$backup_config" "$original_config"
            fi
            return 1
        fi
        
        # Restore original config after successful execution
        if [ -f "$backup_config" ]; then
            mv "$backup_config" "$original_config"
            log_message "INFO" "Restored original config file"
        fi
    else
        log_message "INFO" "DRY RUN: $comparison_command"
    fi
    
    return 0
}

# Main execution function
main() {
    log_message "INFO" "Starting Probe Generalization Study"
    log_message "INFO" "Output directory: $OUTPUT_DIR"
    log_message "INFO" "Skip if exists: $SKIP_IF_EXISTS"
    log_message "INFO" "Verbose logging: $VERBOSE"
    log_message "INFO" "Dry run mode: $DRY_RUN"
    
    # Create output directory and log file
    mkdir -p "$OUTPUT_DIR"
    if [ "$VERBOSE" = true ]; then
        touch "$OUTPUT_DIR/experiment_log.txt"
        echo "# Probe Generalization Study Execution Log" > "$OUTPUT_DIR/experiment_log.txt"
        echo "# Started: $(date)" >> "$OUTPUT_DIR/experiment_log.txt"
        echo "" >> "$OUTPUT_DIR/experiment_log.txt"
    fi
    
    # Validate resources
    validate_resources
    
    # Track overall success
    local total_arms=4
    local successful_arms=0
    local failed_arms=0
    
    # Define the four experimental arms (using converted datasets)
    declare -A arms=(
        ["ideal_gs1"]="probe_study_data/ideal_train_converted.npz probe_study_data/ideal_test_converted.npz configs/comparison_config_gs1.yaml"
        ["ideal_gs2"]="probe_study_data/ideal_train_converted.npz probe_study_data/ideal_test_converted.npz configs/comparison_config_gs2.yaml"
        ["exp_gs1"]="probe_study_data/exp_train_converted.npz probe_study_data/exp_test_converted.npz configs/comparison_config_gs1.yaml"
        ["exp_gs2"]="probe_study_data/exp_train_converted.npz probe_study_data/exp_test_converted.npz configs/comparison_config_gs2.yaml"
    )
    
    # Execute each experimental arm
    local overall_start_time=$(date +%s)
    
    for arm_name in "${!arms[@]}"; do
        local arm_config=(${arms[$arm_name]})
        local train_data="${arm_config[0]}"
        local test_data="${arm_config[1]}"
        local config_file="${arm_config[2]}"
        local arm_output_dir="$OUTPUT_DIR/$arm_name"
        
        log_message "INFO" "=== Executing Arm $((successful_arms + failed_arms + 1))/$total_arms: $arm_name ==="
        
        if execute_arm "$arm_name" "$train_data" "$test_data" "$config_file" "$arm_output_dir"; then
            ((successful_arms++))
        else
            ((failed_arms++))
            log_message "ERROR" "Arm '$arm_name' failed, but continuing with remaining arms..."
        fi
        
        log_message "INFO" "Progress: $((successful_arms + failed_arms))/$total_arms arms processed"
    done
    
    # Final summary
    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))
    
    log_message "INFO" "=== Probe Generalization Study Summary ==="
    log_message "INFO" "Total duration: ${total_duration} seconds"
    log_message "INFO" "Successful arms: $successful_arms/$total_arms"
    log_message "INFO" "Failed arms: $failed_arms/$total_arms"
    
    if [ "$failed_arms" -eq 0 ]; then
        log_message "INFO" "All experimental arms completed successfully!"
        
        # Create a summary of results
        log_message "INFO" "Generating results summary..."
        if [ "$DRY_RUN" != true ]; then
            create_results_summary
        fi
        
        return 0
    else
        log_message "ERROR" "Some experimental arms failed. Check logs for details."
        return 1
    fi
}

# Function to create basic results summary
create_results_summary() {
    local summary_file="$OUTPUT_DIR/EXPERIMENT_SUMMARY.md"
    
    cat > "$summary_file" << EOF
# Probe Generalization Study Results Summary

**Execution Date:** $(date)
**Output Directory:** $OUTPUT_DIR

## Experimental Configuration

This study evaluated the impact of probe function type (idealized vs. experimental) 
on PtychoPINN performance across different overlap constraints (gridsize=1 vs. gridsize=2).

### Four Experimental Arms:
1. **Idealized Probe / Gridsize 1** - Output: \`ideal_gs1/\`
2. **Idealized Probe / Gridsize 2** - Output: \`ideal_gs2/\`  
3. **Experimental Probe / Gridsize 1** - Output: \`exp_gs1/\`
4. **Experimental Probe / Gridsize 2** - Output: \`exp_gs2/\`

## Quick Results Preview

EOF

    # Add basic metrics summary if files exist
    for arm in ideal_gs1 ideal_gs2 exp_gs1 exp_gs2; do
        if [ -f "$OUTPUT_DIR/$arm/comparison_metrics.csv" ]; then
            echo "### $arm" >> "$summary_file"
            echo "\`\`\`" >> "$summary_file"
            python -c "
import pandas as pd
try:
    df = pd.read_csv('$OUTPUT_DIR/$arm/comparison_metrics.csv')
    if 'pinn_psnr' in df.columns:
        print(f'PINN PSNR: {df[\"pinn_psnr\"].mean():.2f}')
    if 'pinn_ssim' in df.columns:
        print(f'PINN SSIM: {df[\"pinn_ssim\"].mean():.3f}')
except Exception as e:
    print(f'Error reading metrics: {e}')
" >> "$summary_file" 2>/dev/null || echo "Metrics not available" >> "$summary_file"
            echo "\`\`\`" >> "$summary_file"
            echo "" >> "$summary_file"
        fi
    done
    
    cat >> "$summary_file" << EOF

## Next Steps

For detailed analysis, proceed to the Final Phase:
1. Run aggregation script to compile all metrics
2. Generate 2x2 comparison visualization  
3. Create comprehensive analysis report

See \`phase_final_checklist.md\` for detailed instructions.
EOF

    log_message "INFO" "Results summary saved to: $summary_file"
}

# Execute main function
main "$@"