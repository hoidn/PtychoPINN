#!/bin/bash
#
# Full-frame 2x2 probe study with complete object coverage
# This version ensures ground truth is visible in comparison plots
#

set -e  # Exit on error

# Parse command line arguments
OUTPUT_DIR=""
QUICK_TEST=false
SKIP_COMPLETED=false
PARALLEL_JOBS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
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
            echo "Usage: $0 --output-dir <dir> [--quick-test] [--skip-completed] [--parallel-jobs N]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output-dir is required"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting Full-Frame 2x2 Probe Study"
echo "Output directory: $OUTPUT_DIR"

# Set parameters based on mode
if [[ "$QUICK_TEST" == true ]]; then
    N_TRAIN=500
    N_TEST=200
    NEPOCHS=2
    OBJECT_SIZE=128
else
    N_TRAIN=5000
    N_TEST=1000
    NEPOCHS=50
    OBJECT_SIZE=256
fi

# Step 1: Create probes
echo "Step 1: Creating probes..."

# Default probe (idealized)
python -c "
import numpy as np
import sys
sys.path.insert(0, '.')

# Set up params before any imports that might use them
from ptycho import params
params.cfg['N'] = 64
params.cfg['default_probe_scale'] = 0.7

# Now safe to import probe generation
from ptycho.probe import get_default_probe

probe = get_default_probe(64, fmt='np')
probe = probe.astype(np.complex64)
np.save('$OUTPUT_DIR/default_probe.npy', probe)
print(f'Default probe saved: shape={probe.shape}, dtype={probe.dtype}')
"

# Hybrid probe (with phase aberrations)
echo "Creating hybrid probe with phase aberrations..."
python -c "
import numpy as np

# Load default probe
default_probe = np.load('$OUTPUT_DIR/default_probe.npy')

# Create phase aberrations
N = default_probe.shape[0]
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Add coma and astigmatism
phase_aberration = 0.5 * (X**3 + Y**3) + 0.3 * (X**2 - Y**2)

# Apply phase to probe
hybrid_probe = default_probe * np.exp(1j * phase_aberration)

# Normalize power
default_power = np.sum(np.abs(default_probe)**2)
hybrid_power = np.sum(np.abs(hybrid_probe)**2)
hybrid_probe *= np.sqrt(default_power / hybrid_power)

np.save('$OUTPUT_DIR/hybrid_probe.npy', hybrid_probe)
print(f'Hybrid probe saved with phase aberrations')
"

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
    
    # Generate DIFFERENT synthetic objects for train and test
    # Use different seeds to ensure variety
    local train_seed=$((gridsize * 100 + 1))
    local test_seed=$((gridsize * 100 + 2))
    
    # Simulate training data with full-frame coverage
    echo "  Simulating training data (seed=$train_seed)..."
    python scripts/simulation/simulate_full_frame.py \
        --output-file "$exp_dir/train_data.npz" \
        --n-images "$N_TRAIN" \
        --probe-size 64 \
        --object-size "$OBJECT_SIZE" \
        --object-type lines \
        --overlap 0.7 \
        --probe-file "$probe_path" \
        --seed "$train_seed" \
        --gridsize "$gridsize"
    
    # Simulate test data with DIFFERENT object
    echo "  Simulating test data (seed=$test_seed)..."
    python scripts/simulation/simulate_full_frame.py \
        --output-file "$exp_dir/test_data.npz" \
        --n-images "$N_TEST" \
        --probe-size 64 \
        --object-size "$OBJECT_SIZE" \
        --object-type lines \
        --overlap 0.7 \
        --probe-file "$probe_path" \
        --seed "$test_seed" \
        --gridsize "$gridsize"
    
    # Train model
    echo "  Training model..."
    ptycho_train \
        --train_data_file "$exp_dir/train_data.npz" \
        --test_data_file "$exp_dir/test_data.npz" \
        --output_dir "$exp_dir/model" \
        --gridsize "$gridsize" \
        --nepochs "$NEPOCHS"
    
    # Evaluate model
    echo "  Evaluating model..."
    mkdir -p "$exp_dir/evaluation"
    python scripts/compare_models.py \
        --pinn_dir "$exp_dir/model" \
        --test_data "$exp_dir/test_data.npz" \
        --output_dir "$exp_dir/evaluation" \
        --save-debug-images
    
    # Extract metrics
    if [[ -f "$exp_dir/evaluation/comparison_metrics.csv" ]]; then
        cp "$exp_dir/evaluation/comparison_metrics.csv" "$exp_dir/metrics_summary.csv"
    fi
    
    # Also save a verification image showing ground truth coverage
    echo "  Creating coverage verification plot..."
    python -c "
import numpy as np
import matplotlib.pyplot as plt

# Load test data
data = np.load('$exp_dir/test_data.npz')
obj = data['objectGuess']
xcoords = data['xcoords']
ycoords = data['ycoords']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Show object
ax1.imshow(np.abs(obj), cmap='gray')
ax1.set_title('Test Object (Ground Truth)')
ax1.axis('off')

# Show scan coverage
ax2.imshow(np.abs(obj), cmap='gray', alpha=0.3)
# Convert coordinates to pixel positions
px = xcoords + obj.shape[1]/2
py = ycoords + obj.shape[0]/2
ax2.scatter(px, py, c='red', s=1, alpha=0.5)
ax2.set_title(f'Scan Coverage ({len(xcoords)} positions)')
ax2.axis('off')

plt.suptitle(f'Experiment: gridsize=$gridsize, probe=$probe_type')
plt.tight_layout()
plt.savefig('$exp_dir/coverage_verification.png', dpi=150, bbox_inches='tight')
plt.close()

print('Coverage verification saved')
"
}

# Export function for parallel execution
export -f run_experiment
export OUTPUT_DIR N_TRAIN N_TEST NEPOCHS OBJECT_SIZE QUICK_TEST SKIP_COMPLETED

# Run experiments
if [[ "$PARALLEL_JOBS" -gt 1 ]]; then
    echo "Running experiments in parallel with $PARALLEL_JOBS jobs..."
    
    # Create a temporary file with all experiment parameters
    TEMP_FILE=$(mktemp)
    echo "1 default $OUTPUT_DIR/default_probe.npy" >> "$TEMP_FILE"
    echo "1 hybrid $OUTPUT_DIR/hybrid_probe.npy" >> "$TEMP_FILE"
    echo "2 default $OUTPUT_DIR/default_probe.npy" >> "$TEMP_FILE"
    echo "2 hybrid $OUTPUT_DIR/hybrid_probe.npy" >> "$TEMP_FILE"
    
    # Run in parallel
    cat "$TEMP_FILE" | parallel -j "$PARALLEL_JOBS" --colsep ' ' run_experiment {1} {2} {3}
    
    rm "$TEMP_FILE"
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
            df['experiment'] = f'gs{gridsize}_{probe_type}'
            results.append(df)

if results:
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(output_dir / 'study_summary.csv', index=False)
    
    print('\n=== 2x2 Study Results ===')
    print(combined[['experiment', 'amplitude_mae', 'amplitude_psnr', 'amplitude_ssim']].to_string(index=False))
    
    # Create summary plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot MAE
    ax = axes[0, 0]
    for probe in ['default', 'hybrid']:
        data = combined[combined['probe_type'] == probe]
        ax.plot([1, 2], data['amplitude_mae'].values, 'o-', label=probe, markersize=8)
    ax.set_xlabel('Gridsize')
    ax.set_ylabel('MAE')
    ax.set_title('Amplitude MAE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot PSNR
    ax = axes[0, 1]
    for probe in ['default', 'hybrid']:
        data = combined[combined['probe_type'] == probe]
        ax.plot([1, 2], data['amplitude_psnr'].values, 'o-', label=probe, markersize=8)
    ax.set_xlabel('Gridsize')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Amplitude PSNR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot SSIM
    ax = axes[1, 0]
    for probe in ['default', 'hybrid']:
        data = combined[combined['probe_type'] == probe]
        ax.plot([1, 2], data['amplitude_ssim'].values, 'o-', label=probe, markersize=8)
    ax.set_xlabel('Gridsize')
    ax.set_ylabel('SSIM')
    ax.set_title('Amplitude SSIM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary text
    ax = axes[1, 1]
    ax.text(0.1, 0.9, '2x2 Probe Study Summary', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.7, f'Training images: $N_TRAIN', transform=ax.transAxes)
    ax.text(0.1, 0.6, f'Test images: $N_TEST', transform=ax.transAxes)
    ax.text(0.1, 0.5, f'Object size: $OBJECT_SIZE x $OBJECT_SIZE', transform=ax.transAxes)
    ax.text(0.1, 0.4, f'Probe size: 64 x 64', transform=ax.transAxes)
    ax.text(0.1, 0.3, f'Coverage: Full frame', transform=ax.transAxes)
    ax.axis('off')
    
    plt.suptitle('2x2 Probe Parameterization Study Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'study_results_plot.png', dpi=150, bbox_inches='tight')
    
    print(f'\nSummary plot saved to: {output_dir}/study_results_plot.png')
else:
    print('No results found!')
"

echo ""
echo "Full-Frame 2x2 Probe Study Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key outputs:"
echo "  - study_summary.csv: Aggregated metrics"
echo "  - study_results_plot.png: Summary visualization" 
echo "  - */evaluation/comparison_plot.png: Model comparisons WITH ground truth"
echo "  - */coverage_verification.png: Scan coverage verification"