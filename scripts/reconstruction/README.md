# Reconstruction Scripts Directory

This directory contains scripts for generating ptychographic reconstructions using various traditional algorithms, providing baselines for comparison with machine learning models.

## Available Scripts

### `run_tike_reconstruction.py`

A standalone script that performs ptychographic reconstruction using the Tike library's iterative algorithms.

**Purpose:** Generate traditional algorithm reconstructions that can be integrated into model comparison studies, providing a third arm alongside PtychoPINN and baseline models.

**Usage:**
```bash
python run_tike_reconstruction.py <input_npz> <output_dir> [options]
```

**Arguments:**
- `input_npz`: NPZ file containing `diffraction`, `probeGuess`, `xcoords`, `ycoords`
- `output_dir`: Directory where results will be saved
- `--iterations`: Number of reconstruction iterations (default: 1000)
- `--num-gpu`: Number of GPUs to use (default: 1)
- `--quiet`: Suppress console output (file logging only)
- `--verbose`: Enable DEBUG output to console

**Output Files:**
- `tike_reconstruction.npz`: Standardized NPZ with `reconstructed_object`, `reconstructed_probe`, and `metadata`
- `reconstruction_visualization.png`: 2x2 plot showing amplitude and phase of results
- `logs/debug.log`: Complete execution log

**Examples:**
```bash
# Basic reconstruction
python run_tike_reconstruction.py datasets/fly64/test.npz ./tike_output

# Quick test with fewer iterations
python run_tike_reconstruction.py datasets/fly64/test.npz ./tike_output --iterations 50

# Quiet mode for automation
python run_tike_reconstruction.py datasets/fly64/test.npz ./tike_output --quiet
```

**Integration with Comparison Studies:**
The output `tike_reconstruction.npz` file is designed to integrate seamlessly with the model comparison workflow in Phase 2 of the Tike Comparison Integration initiative.

## Data Contract

All reconstruction scripts in this directory produce NPZ files with the following standardized format:

### Required Arrays
- **`reconstructed_object`**: Complex 2D array containing the final reconstructed object
- **`reconstructed_probe`**: Complex 2D array containing the final reconstructed probe

### Required Metadata
- **`metadata`**: Single-element object array containing a dictionary with:
  - `algorithm`: String identifying the reconstruction algorithm
  - `version`: Version string of the algorithm library
  - `iterations`: Number of iterations performed
  - `computation_time_seconds`: Float timing measurement
  - `parameters`: Dictionary of algorithm-specific parameters
  - `input_file`: Path to the input NPZ file
  - `timestamp`: ISO format timestamp of reconstruction

This standardized format ensures compatibility with existing model comparison and evaluation workflows.

## Development Notes

**Adding New Algorithms:**
1. Create a new script following the naming pattern `run_<algorithm>_reconstruction.py`
2. Follow the same CLI pattern with `argparse` and logging integration
3. Implement the standardized data contract for output NPZ files
4. Include comprehensive docstrings and error handling
5. Add usage examples to this README

**Dependencies:**
- Scripts should integrate with the project's centralized logging system (`ptycho.log_config`)
- Use the standard CLI argument helpers (`ptycho.cli_args`)
- Follow project conventions for error handling and user feedback

## Future Extensions

This directory is designed to accommodate additional reconstruction algorithms:
- EPIE (Extended Ptychographic Iterative Engine)
- RAAR (Relaxed Averaged Alternating Reflections)  
- Other iterative phase retrieval methods
- GPU-accelerated implementations

Each new algorithm should follow the established patterns for consistency and integration with the broader comparison framework.