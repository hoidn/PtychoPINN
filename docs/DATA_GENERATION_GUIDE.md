# Data Generation Guide: Grid vs Nongrid Simulation

This guide covers the two distinct data generation pipelines in PtychoPINN and when to use each.

## Canonical generated-data recipe

New generation code uses `SimulationConfig` for every property baked into a dataset: detector size, probe construction and simulation mask, object family, scan geometry, photon/beamstop settings, and seed. Training values do not belong in this recipe. A legacy simulator must receive `update_legacy_dict(params.cfg, simulation_config)` before its module is imported or called; the later training bridge is separate.

Probe pipelines are ordered. `pad_extrapolate` is the historical global quadratic phase replacement, including the source footprint. `smooth:0.5|pad_extrapolate_boundary_matched:128` first prepares the source, then copies that complex center exactly and constructs only the outer phase with the C0 harmonic boundary-conditioned solver. The boundary-matched operation is terminal. `pad_preserve` and `interp` retain their established padding/interpolation meanings. Any pipeline change creates a different dataset identity.

Generation metadata records the canonical `SimulationConfig`, its `simulation_config_sha256`, source and probe content hashes, normalized pipeline, transformed-probe hash, `dataset_recipe_sha256`, and boundary solver evidence when applicable. Grid-lines generation writes beneath `<output_dir>/datasets/N<N>/gs<gridsize>/simulation-<simulation_config_sha256>/`. `simulate_and_save.py` preserves the exact `--output-file` and requires a distinct explicit path when either digest changes.

```python
from ptycho.config import load_simulation_config

simulation = load_simulation_config("configs/lines128.toml")
```

## Overview

PtychoPINN has two data generation systems reflecting the broader "two-system" architecture (see `DEVELOPER_GUIDE.md` §1):

| System | Entry Point | Coordinates | Grouping | Primary Use |
|--------|-------------|-------------|----------|-------------|
| **Grid-based** (legacy) | `diffsim.mk_simdata()` | Implicit via `tf.image.extract_patches` | Built-in fixed grid | Notebook workflows, legacy scripts |
| **Nongrid** (modern) | `nongrid_simulation.generate_simulated_data()` | Explicit random arrays | Post-hoc KDTree | Production scripts, flexible layouts |

## Quick Decision Guide

**Use Grid mode when:**
- Reproducing notebook experiments (e.g., `notebooks/dose_dependence.ipynb`)
- Fixed regular scan patterns are acceptable
- You want pre-grouped output ready for training

**Use Nongrid mode when:**
- Random/irregular scan positions needed
- Working with real experimental coordinates
- Using the `SimulationConfig`-driven generation workflow with a derived legacy runtime adapter

---

## Grid-Based Pipeline

### Entry Point
```python
from ptycho.diffsim import mk_simdata

X, Y_I, Y_phi, intensity_scale, YY_full, norm_Y_I, coords = mk_simdata(
    n=2,               # Number of outer grid positions
    size=392,          # Object canvas size
    probe=probeGuess,  # Complex probe array
    outer_offset=8     # Stride between groups
)
```

### Legacy low-level `params.cfg` view

New generation entry points resolve and bridge `SimulationConfig`. The direct `p.set()` calls below document only the state read by historical `mk_simdata()` and notebook callers; they are not a second ownership surface.

```python
from ptycho.params import params as p

p.set('N', 128)                    # Patch size
p.set('gridsize', 2)               # 2x2 = 4 patches per group
p.set('offset', 4)                 # Intra-group spacing
p.set('outer_offset_train', 8)     # Training grid stride
p.set('outer_offset_test', 20)     # Test grid stride
p.set('nphotons', 1e9)             # Photon count for Poisson noise
p.set('size', 392)                 # Object size
p.set('data_source', 'lines')      # Object type
p.set('max_position_jitter', 3)    # Position buffer
p.set('sim_jitter_scale', 0.0)     # Jitter std (0 = no jitter)
```

### Output Format
- `X`: `(n × gridsize², N, N, gridsize²)` — **already grouped** diffraction
- `Y_I`, `Y_phi`: Same shape — amplitude and phase ground truth
- `intensity_scale`, `norm_Y_I`, `YY_full`: physics-scale and full-object simulation outputs
- `coords`: Grid positions

### Key Characteristics
1. **Positions are implicit** — computed via `tf.image.extract_patches`, no coordinate arrays
2. **Output is pre-grouped** — ready for `PtychoDataContainer` without KDTree step
3. **Fixed grid layout** — patches at regular `offset` spacing within groups

### Memoization Notes (Grid-Based)
Grid simulations may be memoized on disk to avoid repeating identical dataset generation.
For stable reuse across grid studies, use dataset-only cache keys:

```bash
PTYCHO_MEMOIZE_KEY_MODE=dataset
```

To disable memoization for parameter sweeps:

```bash
PTYCHO_DISABLE_MEMOIZE=1
```

### Programmatic `N=256` Grid-Lines Builds

The public helper CLI `scripts/studies/grid_lines_workflow.py` currently accepts only `--N 64` or `--N 128`.

If you need a synthetic grid-lines dataset at `N=256`, build it programmatically through `GridLinesConfig` and the workflow helpers in `ptycho.workflows.grid_lines_workflow` instead of that CLI.

Important storage rule:

- Treat `outputs/` as a cleanup-prone compatibility location, not the desired long-term home for pinned datasets.
- If an `N=256` grid-lines pair needs to persist as a durable study input, move or rebuild it under a durable git-ignored dataset location (for example under `datasets/`) and update consumers explicitly.

New `GridLinesConfig` builds write:

- `<output_dir>/datasets/N256/gs1/simulation-<simulation_config_sha256>/train.npz`
- `<output_dir>/datasets/N256/gs1/simulation-<simulation_config_sha256>/test.npz`

The old undigested `outputs/lines_256_arch_improvement/datasets/N256/gs1/{train,test}.npz` pair and its flat metadata describe historical pre-`SimulationConfig` artifacts only.

Older archived `custom_npz_pair_n256` pair:

- `outputs/hybrid_resnet_structural_rerun_20260226T110719Z/datasets/custom_npz_builder_n256/datasets/N256/gs1/train.npz`
- `outputs/hybrid_resnet_structural_rerun_20260226T110719Z/datasets/custom_npz_builder_n256/datasets/N256/gs1/test.npz`

The current `lines_256` working pair uses:

- `probe_scale_mode=pad_preserve`
- `probe_smoothing_sigma=0.5`
- centered complex-probe padding from the 64x64 source probe

Its embedded metadata records the synthetic recipe:

- `N=256`, `gridsize=1`
- `size=392`, `offset=4`, `outer_offset_train=8`, `outer_offset_test=20`
- `nimgs_train=2`, `nimgs_test=1`, `nphotons=1e9`
- `probe_source=custom`, `probe_scale_mode=pad_preserve`, `probe_smoothing_sigma=0.5`, `coords_type=relative`

The listed flat metadata fields remain compatibility provenance for those historical artifacts. New reproduction is owned by the resolved `SimulationConfig`, its `simulation_config_sha256`, and its `dataset_recipe_sha256`.
Do not read this section as approval to store persistent datasets under `outputs/` indefinitely.

### Direct Container Construction
```python
from ptycho.loader import PtychoDataContainer
from ptycho.diffsim import scale_nphotons
import tensorflow as tf

container = PtychoDataContainer(
    X=X,
    Y_I=Y_I,
    Y_phi=Y_phi,
    norm_Y_I=scale_nphotons(tf.convert_to_tensor(X)),
    YY_full=None,
    coords_nominal=coords,
    coords_true=coords,
    nn_indices=None,
    global_offsets=None,
    local_offsets=None,
    probeGuess=probeGuess
)
```

---

## Nongrid Pipeline

### Entry Point
```python
from ptycho import params as p
from ptycho.config import (
    DetectorSimulationConfig,
    ModelConfig,
    ScanSimulationConfig,
    SimulationConfig,
    SyntheticObjectConfig,
    TrainingConfig,
    update_legacy_dict,
)

simulation = SimulationConfig(
    N=64,
    object=SyntheticObjectConfig(diffractions_per_object=2000),
    scan=ScanSimulationConfig(kind="nongrid", grid_size=(2, 2), buffer=15),
    detector=DetectorSimulationConfig(photons_per_pattern=1e9),
)
update_legacy_dict(p.cfg, simulation)

# TrainingConfig remains the compatibility envelope expected by the legacy API.
runtime = TrainingConfig(
    model=ModelConfig(
        N=simulation.N,
        gridsize=simulation.scan.grid_size[0],
    ),
    n_groups=simulation.object.diffractions_per_object,
    nphotons=simulation.detector.photons_per_pattern,
)

from ptycho.nongrid_simulation import generate_simulated_data

raw_data = generate_simulated_data(
    config=runtime,
    objectGuess=objectGuess,
    probeGuess=probeGuess,
    buffer=simulation.scan.buffer,
)
```

### Output Format
- Returns `RawData` container with:
  - `diff3d`: `(n_images, N, N)` — **ungrouped** individual patterns
  - `xcoords`, `ycoords`: `(n_images,)` — explicit position arrays
  - `Y`: Ground truth patches (if simulation)

### Key Characteristics
1. **Positions are explicit** — random uniform within object bounds
2. **Output is ungrouped** — requires `generate_grouped_data()` for training
3. **Flexible layout** — can use any coordinate distribution

### Grouping Step (Required for Training)
```python
# Nongrid output needs grouping before training
grouped_data = raw_data.generate_grouped_data(
    N=runtime.model.N,
    K=runtime.neighbor_count,        # Neighbors to consider
    nsamples=runtime.n_groups,       # Groups to generate
    gridsize=runtime.model.gridsize  # Patterns per group
)

# Then convert to container via loader
from ptycho.loader import load
container = load(lambda: grouped_data, probeGuess, which=None, create_split=False)
```

---

## Parameter Mapping

| Concept | Canonical owner | Grid legacy view | Nongrid compatibility view |
|---|---|---|---|
| Detector/probe size | `simulation.N` | `cfg['N']` | `runtime.model.N`, derived |
| Scan grouping | `simulation.scan.grid_size` | `cfg['gridsize']` | `runtime.model.gridsize`, derived |
| Grid split counts | `simulation.scan.train_groups/test_groups` | `cfg['nimgs_train/test']` | N/A |
| Nongrid pattern count | `simulation.object.diffractions_per_object` | N/A | `runtime.n_groups`, derived |
| Photon count | `simulation.detector.photons_per_pattern` | `cfg['nphotons']` | `runtime.nphotons`, derived |
| Scan offsets/buffer | `simulation.scan.*` | Corresponding legacy fields | `buffer`, derived |
| Object size | `simulation.object.image_size` | `cfg['size']` | `objectGuess.shape`, which must agree |

---

## Object Generation

Canonical `SimulationConfig` entry points support `lines`, `dead_leaves`, and `natural_patch`. Legacy helpers may create other arrays, but unsupported kinds such as `grf` must not be mislabeled or materialized as a canonical simulation recipe.

```python
from ptycho.diffsim import mk_lines_img

# Lines pattern
obj = mk_lines_img(size=392, nlines=400)

# Other supported families are prepared by their dedicated Stage-1 generators.
```

---

## Common Pitfalls

### Grid Mode
- **Must set `params.cfg` before calling `mk_simdata()`** — it reads global state
- **Import-time side effects** — some legacy modules trigger data generation on import (see ANTIPATTERN-001)

### Nongrid Mode
- **Must bridge `SimulationConfig` before legacy generation**; bridge the derived runtime config separately before training (CONFIG-001)
- **Grouping is required** — `RawData` output is ungrouped; training needs grouped data

### Both
- **gridsize must match** — simulation gridsize must equal training gridsize
- **Probe size must match N** — `probeGuess.shape == (N, N)`

---

## Example: Notebook-Compatible Grid Simulation

```python
"""Reproduce notebooks/dose_dependence.ipynb data generation."""
from ptycho.params import params as p
from ptycho.diffsim import mk_simdata
from ptycho.probe import get_default_probe

# Setup (matches dose.py::init())
p.set('N', 128)
p.set('gridsize', 2)
p.set('offset', 4)
p.set('outer_offset_train', 8)
p.set('outer_offset_test', 20)
p.set('nphotons', 1e9)
p.set('size', 392)
p.set('data_source', 'lines')

# Generate probe
probe = get_default_probe(N=128, fmt='np')

# Simulate
X_train, Y_I_train, Y_phi_train, intensity_scale, YY_full, norm_Y_I, coords_train = mk_simdata(
    n=2, size=392, probe=probe, outer_offset=8
)
```

---

## Related Documentation
- `docs/DEVELOPER_GUIDE.md` §1 — Two-system architecture
- `scripts/simulation/README.md` — Stage 1/Stage 2 simulation workflow
- `docs/specs/spec-ptycho-core.md` — standalone-NPZ format specification
- `docs/findings.md` CONFIG-001 — params.cfg initialization requirement
