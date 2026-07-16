# PtychoPINN Simulation Workflow

This directory contains tools for generating simulated ptychography datasets. The workflow is designed to be modular, allowing for flexible creation of various types of objects and probes.

The canonical generated-data contract is `SimulationConfig`. Both scripts accept `--simulation-config` (TOML, YAML, or JSON), and retained flags are compatibility overrides: explicit CLI value > file value > historical no-file default. The recipe owns probe construction, object description, scan geometry, detector/noise, and seed. Epochs and optimizer settings are not simulation fields.

## The Two-Stage Simulation Architecture

The simulation process is divided into two distinct stages:

### Stage 1: Input Generation
The first step is to create an `.npz` file containing the ground truth `objectGuess` and `probeGuess`. This file serves as the input for the main simulation. You can obtain this file in several ways:
- Using data from a real experiment.
- Using the output from a previous reconstruction (e.g., from tike).
- Generating a synthetic object and probe programmatically.

When `simulation.probe.source_path` is set, that configured archive supplies the probe. The Stage-1 `probeGuess` remains required by the input schema but is not a competing owner.

### Stage 2: Diffraction Simulation
The core script `simulate_and_save.py` takes the `.npz` file from Stage 1 as input. It simulates the ptychographic scanning process by generating thousands of diffraction patterns and saves the complete, model-ready dataset to a new `.npz` file.

This modular design allows you to simulate diffraction for any object you can create, not just the built-in synthetic types.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `simulate_and_save.py` | **Core Tool.** The main, general-purpose script for running Stage 2. It takes an input `.npz` and generates a full simulated dataset. |
| `run_with_synthetic_lines.py` | **Convenience Wrapper.** A high-level script that automates both Stage 1 and Stage 2 for the specific 'lines' object type. Ideal for quick tests. |

## Workflow Examples

### Simple Workflow: Automated 'Lines' Simulation

For a quick test or a standard 'lines' object simulation, use the `run_with_synthetic_lines.py` wrapper. It handles everything in one command.

```bash
# This command will internally generate a 'lines' object and probe,
# then feed them into the simulation engine.
python scripts/simulation/run_with_synthetic_lines.py \
    --simulation-config configs/lines128.toml \
    --output-dir simulation_lines_output \
    --n-images 2000
```

The wrapper requires `simulation.object.kind = "lines"`. It can construct an ideal probe or load `simulation.probe.source_path`, then applies the declared pipeline. A complete lines recipe is:

```toml
[simulation]
N = 128
seed = 3

[simulation.probe]
source = "custom"
source_path = "datasets/Run1084_recon3_postPC_shrunk_3.npz"
transform_pipeline = "smooth:0.5|pad_extrapolate_boundary_matched:128"

[simulation.object]
kind = "lines"
image_size = [392, 392]
objects_per_probe = 4
diffractions_per_object = 2000
set_phi = true

[simulation.scan]
kind = "nongrid"
grid_size = [1, 1]
offset = 4
outer_offset_train = 8
outer_offset_test = 20
train_groups = 2
test_groups = 1
buffer = 64

[simulation.detector]
photons_per_pattern = 1e9
```

### Advanced workflow: prepared supported objects

For `dead_leaves` or `natural_patch`, prepare the Stage-1 `objectGuess` and `probeGuess` separately and pass a matching `SimulationConfig` to Stage 2. The complete dead-leaves example below demonstrates this path. `grf` is not currently a supported `SyntheticObjectConfig.kind` and must not be recorded under another object label.

### Lines and dead-leaves prepared inputs

The general Stage-2 CLI uses the same launch shape for either object family:

```bash
python scripts/simulation/simulate_and_save.py \
  --simulation-config configs/lines64.toml \
  --input-file prepared_lines64.npz \
  --output-file outputs/lines64.npz

python scripts/simulation/simulate_and_save.py \
  --simulation-config configs/dead_leaves64.toml \
  --input-file prepared_dead_leaves64.npz \
  --output-file outputs/dead_leaves64.npz
```

The two config files differ at `simulation.object.kind` (`"lines"` versus `"dead_leaves"`) and may otherwise share probe/scan/detector settings. `simulate_and_save.py` consumes the `objectGuess` already present in `--input-file`; it does not synthesize dead leaves. Use the repository's dead-leaves generator to prepare that Stage-1 NPZ first. The lines convenience wrapper is intentionally lines-only.

A complete prepared dead-leaves recipe is:

```toml
[simulation]
N = 64
seed = 3

[simulation.probe]
source = "custom"
source_path = "prepared_dead_leaves64.npz"
transform_pipeline = "pad_preserve:64"

[simulation.object]
kind = "dead_leaves"
image_size = [256, 256]
objects_per_probe = 4
diffractions_per_object = 2000
set_phi = true

[simulation.scan]
kind = "nongrid"
grid_size = [1, 1]
offset = 4
outer_offset_train = 8
outer_offset_test = 20
train_groups = 2
test_groups = 1
buffer = 32

[simulation.detector]
photons_per_pattern = 1e9
```

Probe transform meanings:

- `pad_extrapolate` is the legacy global quadratic phase, including the center, followed by any later smoothing.
- `smooth:0.5|pad_extrapolate_boundary_matched:128` smooths at source resolution, preserves that complex center exactly, and applies the C0 boundary-conditioned outer phase only outside it.
- `pad_preserve` center-pads the complex probe; `interp` interpolates real/imaginary parts.

Changing the pipeline changes dataset identity. A simulation-time `mask_diameter` is also part of dataset identity; it is distinct from a model-time probe mask.

`simulate_and_save.py` keeps the exact `--output-file` path and records `simulation_config_sha256` plus `dataset_recipe_sha256`; it rejects an existing output when either differs. Use a distinct file or output directory for a changed recipe. Grid-lines generation adds the simulation digest to its dataset directory automatically as `simulation-<simulation_config_sha256>`.

## Output Data Format

The final output of the simulation workflow is an `.npz` file that conforms to the project's data standard. For full details on the required keys and array shapes, see the [Data Contracts Document](../../specs/data_contracts.md).
