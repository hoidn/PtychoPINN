# Reconstruction Scripts

This directory contains scripts for running traditional iterative reconstruction algorithms, primarily for comparison with neural network approaches.

## Overview

The reconstruction scripts provide implementations of classical ptychographic reconstruction methods that serve as baselines for evaluating the performance of PtychoPINN and other machine learning approaches.

## Scripts

### `run_tike_reconstruction.py`

**Purpose:** Standalone Tike iterative reconstruction script that provides a traditional baseline for comparing against PtychoPINN neural network reconstructions.

**Usage:**
```bash
python scripts/reconstruction/run_tike_reconstruction.py \
    <input_data.npz> \
    <output_dir> \
    [options]
```

**Required Arguments:**
- `input_data.npz` - Input dataset file containing diffraction patterns, probe, and scan coordinates
- `output_dir` - Directory where reconstruction results will be saved

**Key Options:**
- `--iterations N` - Number of Tike algorithm iterations (default: 100)
- `--extra-padding N` - Extra padding pixels for object canvas (default: 32)
- `--visualize` - Generate visualization plots of the reconstruction

**Example Commands:**
```bash
# Quick reconstruction test (100 iterations)
python scripts/reconstruction/run_tike_reconstruction.py \
    datasets/fly64/test_data.npz \
    tike_output_quick/ \
    --iterations 100

# High-quality reconstruction (1000 iterations)
python scripts/reconstruction/run_tike_reconstruction.py \
    datasets/fly64/test_data.npz \
    tike_output/ \
    --iterations 1000 \
    --extra-padding 64 \
    --visualize

# For model comparison studies
python scripts/reconstruction/run_tike_reconstruction.py \
    tike_outputs/fly001_final_downsampled/fly001_final_downsampled_data_transposed.npz \
    tike_reconstruction/ \
    --iterations 1000
```

**Input Requirements:**
The input NPZ file must contain:
- `diffraction` - Stack of diffraction patterns (n_images, N, N) as amplitude values
- `probeGuess` - Initial probe estimate (N, N) complex array  
- `xcoords`, `ycoords` - Scan position coordinates (n_images,) arrays
- `objectGuess` - Initial object estimate (M, M) complex array (optional, for ground truth)

**Output Files:**
- `tike_reconstruction.npz` - Main reconstruction result containing:
  - `psi` - Reconstructed object (complex array)
  - `probe` - Refined probe (complex array)
  - `metadata` - Reconstruction parameters and timing information
- `reconstruction_amplitude.png` - Amplitude visualization (if --visualize)
- `reconstruction_phase.png` - Phase visualization (if --visualize)
- `tike_reconstruction.log` - Detailed algorithm log

**Algorithm Details:**
- Uses Tike's ptychographic reconstruction implementation with automatic padding
- Handles coordinate convention differences (PtychoPINN uses [X,Y], Tike uses [Y,X])
- Implements extra padding strategy to avoid boundary artifacts
- Supports both probe and object refinement
- Records computation time and convergence metrics

## Integration with Model Comparison

The reconstruction results can be directly used in three-way model comparisons:

```bash
# Step 1: Generate Tike reconstruction
python scripts/reconstruction/run_tike_reconstruction.py \
    test_data.npz \
    tike_output/ \
    --iterations 1000

# Step 2: Run three-way comparison
python scripts/compare_models.py \
    --pinn_dir pinn_model/ \
    --baseline_dir baseline_model/ \
    --test_data test_data.npz \
    --output_dir comparison_results \
    --tike_recon_path tike_output/tike_reconstruction.npz
```

This enables comprehensive evaluation of neural network approaches against traditional iterative methods.

## Technical Notes

### Coordinate Convention Handling
- **Critical:** PtychoPINN uses [X,Y] coordinate convention while Tike uses [Y,X]
- The script automatically handles this conversion by swapping coordinate arrays before passing to Tike
- This ensures consistent spatial alignment across all reconstruction methods

### Padding Strategy
- Uses Tike's `get_padded_object()` function with configurable extra padding
- Default 32 pixels of extra padding prevents boundary artifacts
- Padding can be increased for datasets with larger objects or challenging geometry

### Performance Considerations
- Reconstruction time scales with iterations and dataset size
- 100 iterations typically sufficient for quick validation (~1-2 minutes)
- 1000 iterations recommended for high-quality comparisons (~10-15 minutes)
- Memory usage scales with object size and padding amount

### Output Format Compatibility
- Output NPZ format is compatible with PtychoPINN evaluation pipelines
- Reconstruction results can be directly loaded by `compare_models.py`
- Metadata includes timing information for performance comparison studies

## Dependencies

- `tike` - Iterative ptychographic reconstruction library
- `numpy` - Array operations and file I/O
- `matplotlib` - Visualization (optional, for --visualize flag)

## See Also

- **Model Comparison Guide:** `docs/MODEL_COMPARISON_GUIDE.md` - Three-way comparison workflow
- **Commands Reference:** `docs/COMMANDS_REFERENCE.md` - Quick command reference
- **Data Contracts:** `specs/data_contracts.md` - NPZ file format specifications

---

## `ptychi_reconstruct_tike.py`

**Purpose:** Pty-chi reconstruction script that provides an alternative reconstruction method using the pty-chi library for ptychographic data.

**Usage:**
```bash
python scripts/reconstruction/ptychi_reconstruct_tike.py
```

**Configuration:** The script uses hardcoded configuration that can be modified directly in the script:
- `tike_dataset` - Input NPZ file path
- `output_dir` - Output directory name  
- `algorithm` - Reconstruction algorithm ('DM', 'LSQML', 'PIE')
- `num_epochs` - Number of reconstruction epochs
- `n_images` - Number of images to use (for testing)

**Key Features:**
- **Algorithm Selection:** DM (Difference Map), LSQML (Least Squares Maximum Likelihood), PIE (Ptychographic Iterative Engine)
- **Configurable Epochs:** Default 200 epochs for proper convergence
- **Configurable Image Subset:** Default 2000 images (use subset for faster testing)
- **GPU Acceleration:** Automatic CUDA detection and usage when available
- **Data Format Conversion:** Handles amplitude→intensity conversion and coordinate centering

**Example Configuration:**
```python
# Edit these values in main() function:
tike_dataset = "tike_outputs/fly001_reconstructed_final_downsampled/fly001_reconstructed_final_downsampled_data.npz"
output_dir = "ptychi_tike_reconstruction_converged"
algorithm = 'DM'  # Can be 'DM', 'LSQML', 'PIE'
num_epochs = 200  # 200+ recommended for convergence
n_images = 2000   # Use subset for testing, or None for all
```

**Maximum Likelihood (LSQML) Baseline Example (Recommended for studies):**

```python
# In main(): set the algorithm and epochs for a study baseline
algorithm = 'LSQML'
num_epochs = 100  # start with 100; parameterize per study
```

Run the script (uses the hardcoded config above):

```bash
python scripts/reconstruction/ptychi_reconstruct_tike.py
```

Outputs `ptychi_reconstruction.npz` with `reconstructed_object` and `algorithm` fields compatible with `scripts/compare_models.py` via `--tike_recon_path`.

**Input Requirements:**
Same as Tike reconstruction - NPZ file with:
- `diffraction` or `diff3d` - Diffraction patterns as amplitude values
- `probeGuess` - Initial probe estimate (complex)
- `xcoords`, `ycoords` - Scan coordinates
- `objectGuess` - Initial object estimate (complex)

**Output Files:**
- `ptychi_reconstruction.npz` - Main reconstruction result with reconstructed object/probe
- `reconstruction_visualization.png` - 2×2 amplitude/phase visualization plot

**Algorithm Details:**
- **Data Conversion:** Amplitude² → intensity, coordinate centering for pty-chi compatibility
- **Probe Setup:** Automatic dimensionality handling (2D → 4D tensor conversion)
- **Power Constraints:** Probe power constraint based on median total intensity
- **Convergence:** 200+ epochs recommended (15-20 epochs insufficient for quality results)

**Dependencies:**
- `pty-chi` - GPU-accelerated ptychographic reconstruction library (./pty-chi/src)
- `torch` - PyTorch for tensor operations and GPU acceleration
- `numpy` - Array operations and data handling
- `matplotlib` - Result visualization

**Performance:**
- ~8-15 iterations/second on GPU
- 200 epochs ≈ 24 seconds processing time
- Supports both CPU and CUDA GPU acceleration

**Technical Notes:**
- Requires pty-chi library installation and path configuration
- Handles coordinate convention differences (PtychoPINN [X,Y] → pty-chi [Y,X])
- Centers scan positions as required by pty-chi API
- Converts probe from 2D to required 4D format (n_opr_modes, n_modes, h, w)
- Uses torch tensors with automatic device selection
