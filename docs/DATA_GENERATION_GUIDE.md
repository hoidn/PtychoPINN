# Data Generation Guide: Grid vs Nongrid Simulation

This guide covers the two distinct data generation pipelines in PtychoPINN and when to use each.

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
- Using modern `TrainingConfig` dataclass workflow

---

## Grid-Based Pipeline

### Entry Point
```python
from ptycho.diffsim import mk_simdata

X, Y_I, Y_phi, coords = mk_simdata(
    nimgs=2,           # Number of outer grid positions
    size=392,          # Object canvas size
    probe=probeGuess,  # Complex probe array
    outer_offset=8     # Stride between groups
)
```

### Required params.cfg Setup
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
- `X`: `(nimgs × gridsize², N, N, gridsize²)` — **already grouped** diffraction
- `Y_I`, `Y_phi`: Same shape — amplitude and phase ground truth
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

If you need a synthetic grid-lines dataset at `N=256`, build it programmatically through `GridLinesConfig` and the workflow helpers in `ptycho.workflows.grid_lines_workflow` instead of that CLI. The archived `lines_256` / `custom_npz_pair_n256` benchmark pair in:

- `outputs/hybrid_resnet_structural_rerun_20260226T110719Z/datasets/custom_npz_builder_n256/datasets/N256/gs1/train.npz`
- `outputs/hybrid_resnet_structural_rerun_20260226T110719Z/datasets/custom_npz_builder_n256/datasets/N256/gs1/test.npz`

was generated this way. Its embedded metadata records the synthetic recipe:

- `N=256`, `gridsize=1`
- `size=392`, `offset=4`, `outer_offset_train=8`, `outer_offset_test=20`
- `nimgs_train=2`, `nimgs_test=1`, `nphotons=1e9`
- `probe_source=custom`, `coords_type=relative`

The original generation invocation for that archived pair was not preserved, so use the metadata fields above as the authoritative reproduction recipe.

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
from ptycho.nongrid_simulation import generate_simulated_data
from ptycho.config import TrainingConfig, ModelConfig

config = TrainingConfig(
    model=ModelConfig(N=64, gridsize=2),
    n_groups=2000,
    nphotons=1e9,
)

raw_data = generate_simulated_data(
    config=config,
    objectGuess=objectGuess,
    probeGuess=probeGuess,
    buffer=15.0  # Edge buffer for random positions
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
    N=config.model.N,
    K=config.neighbor_count,        # Neighbors to consider
    nsamples=config.n_groups,       # Groups to generate
    gridsize=config.model.gridsize  # Patterns per group
)

# Then convert to container via loader
from ptycho.loader import load
container = load(lambda: grouped_data, probeGuess, which=None, create_split=False)
```

---

## Parameter Mapping

| Concept | Grid (`params.cfg`) | Nongrid (`TrainingConfig`) |
|---------|---------------------|---------------------------|
| Patch size | `cfg['N']` | `config.model.N` |
| Group size | `cfg['gridsize']` | `config.model.gridsize` |
| Sample count | `cfg['nimgs_train']` | `config.n_groups` |
| Photon count | `cfg['nphotons']` | `config.nphotons` |
| Intra-group spacing | `cfg['offset']` | N/A (KDTree determines) |
| Inter-group spacing | `cfg['outer_offset_train']` | N/A (random positions) |
| Object size | `cfg['size']` | Determined by `objectGuess.shape` |

---

## Object Generation

Both systems can use the same object generators:

```python
from ptycho.diffsim import mk_lines_img, sim_object_image

# Lines pattern
obj = mk_lines_img(size=392, nlines=400)

# GRF (Gaussian Random Field)
from ptycho.params import params as p
p.set('data_source', 'grf')
obj = sim_object_image(size=256, which='train')
```

---

## Common Pitfalls

### Grid Mode
- **Must set `params.cfg` before calling `mk_simdata()`** — it reads global state
- **Import-time side effects** — some legacy modules trigger data generation on import (see ANTIPATTERN-001)

### Nongrid Mode
- **Must call `update_legacy_dict(params.cfg, config)` before legacy module usage** (CONFIG-001)
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
X_train, Y_I_train, Y_phi_train, coords_train = mk_simdata(
    nimgs=2, size=392, probe=probe, outer_offset=8
)
```

---

## Related Documentation
- `docs/DEVELOPER_GUIDE.md` §1 — Two-system architecture
- `scripts/simulation/README.md` — Stage 1/Stage 2 simulation workflow
- `specs/data_contracts.md` — NPZ format specification
- `docs/findings.md` CONFIG-001 — params.cfg initialization requirement
