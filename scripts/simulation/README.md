# PtychoPINN Simulation Workflow

This directory contains tools for generating simulated ptychography datasets. The workflow is designed to be modular, allowing for flexible creation of various types of objects and probes.

## The Two-Stage Simulation Architecture

The simulation process is divided into two distinct stages:

### Stage 1: Input Generation
The first step is to create an `.npz` file containing the ground truth `objectGuess` and `probeGuess`. This file serves as the input for the main simulation. You can obtain this file in several ways:
- Using data from a real experiment.
- Using the output from a previous reconstruction (e.g., from tike).
- Generating a synthetic object and probe programmatically.

### Stage 2: Diffraction Simulation
The core script `simulate_and_save.py` takes the `.npz` file from Stage 1 as input. It simulates the ptychographic scanning process by generating thousands of diffraction patterns and saves the complete, model-ready dataset to a new `.npz` file.

This modular design allows you to simulate diffraction for any object you can create, not just the built-in synthetic types.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `simulate_and_save.py` | **Core Tool.** The main, general-purpose script for running Stage 2. It takes an input `.npz` and generates a full simulated dataset. Now supports `--probe-file` for decoupled probe studies. |
| `run_with_synthetic_lines.py` | **Convenience Wrapper.** A high-level script that automates both Stage 1 and Stage 2 for the specific 'lines' object type. Ideal for quick tests. |

## Workflow Examples

### Simple Workflow: Automated 'Lines' Simulation

For a quick test or a standard 'lines' object simulation, use the `run_with_synthetic_lines.py` wrapper. It handles everything in one command.

```bash
# This command will internally generate a 'lines' object and probe,
# then feed them into the simulation engine.
python scripts/simulation/run_with_synthetic_lines.py \
    --output-dir simulation_lines_output \
    --probe-size 64 \
    --n-images 2000
```

### Advanced Workflow: Simulating a Custom Object (e.g., GRF)

To simulate an object type other than 'lines', you must perform the two stages separately.

**Stage 1: Create the input .npz file programmatically.**

You can do this in a simple Python script. This example generates a Gaussian Random Field (GRF) object.

```python
# File: create_grf_input.py
import numpy as np
from ptycho.diffsim import sim_object_image
from ptycho.probe import get_default_probe
from ptycho import params

# Configure parameters for object and probe generation
params.set('data_source', 'grf')
params.set('N', 64)  # Probe size

# Generate the synthetic object and probe
grf_object = sim_object_image(size=256, which='train')
probe = get_default_probe(N=64, fmt='np')

# Save to an .npz file, which will be the input for Stage 2
np.savez(
    'grf_input.npz',
    objectGuess=grf_object.squeeze().astype(np.complex64),
    probeGuess=probe.astype(np.complex64)
)
print("Created grf_input.npz successfully.")
```

**Stage 2: Run the core simulation script with the generated input.**

After running `python create_grf_input.py`, you can use the output as input for `simulate_and_save.py`.

```bash
# Now, run the main simulation tool with the custom input file
python scripts/simulation/simulate_and_save.py \
    --input-file grf_input.npz \
    --output-file simulation_grf_output/grf_simulated_data.npz \
    --n-images 2000 \
    --visualize
```

### Using External Probe for Studies

The `--probe-file` option allows you to override the probe from the input file with an external probe. This is useful for controlled studies on how probe variations affect reconstruction.

```bash
# Example 1: Use a probe from another dataset
python scripts/simulation/simulate_and_save.py \
    --input-file object_data.npz \
    --output-file output_with_custom_probe.npz \
    --probe-file experimental_probe.npz \
    --n-images 2000

# Example 2: Use a hybrid probe created with the Phase 1 tool
python scripts/tools/create_hybrid_probe.py \
    ideal_probe.npy \
    aberrated_probe.npy \
    --output hybrid_probe.npy

python scripts/simulation/simulate_and_save.py \
    --input-file object_data.npz \
    --output-file output_with_hybrid_probe.npz \
    --probe-file hybrid_probe.npy \
    --n-images 2000
```

The external probe must be:
- A 2D complex array
- Smaller than the object in both dimensions
- Provided as either a `.npy` file or `.npz` file (with 'probeGuess' key)

## Output Data Format

The final output of the simulation workflow is an `.npz` file that conforms to the project's data standard. For full details on the required keys and array shapes, see the [Data Contracts Document](../../docs/data_contracts.md).