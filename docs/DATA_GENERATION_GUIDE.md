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

### Direct Container Construction

**IMPORTANT:** When constructing containers manually (bypassing `loader.load()`), you MUST attach
`dataset_intensity_stats` to enable the spec-compliant dataset-derived intensity_scale formula.
See `specs/spec-ptycho-core.md §Normalization Invariants`.

```python
from ptycho.loader import PtychoDataContainer, compute_dataset_intensity_stats
from ptycho.diffsim import scale_nphotons
import tensorflow as tf

# Compute stats from raw/normalized diffraction for dataset-derived intensity_scale
# For normalized data (e.g., from mk_simdata), pass is_normalized=True and intensity_scale
dataset_stats = compute_dataset_intensity_stats(
    X, intensity_scale=intensity_scale, is_normalized=True
)

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
    probeGuess=probeGuess,
    dataset_intensity_stats=dataset_stats  # Required for correct intensity_scale
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

## 4. Alternative Data Creation Flows (No NPZ Required)

This section documents ways to create training/inference data programmatically without loading from NPZ files.

### 4.1. Programmatic Nongrid Simulation

Generate synthetic data with random scan coordinates entirely in memory:

```python
from ptycho.config.config import TrainingConfig, ModelConfig
from ptycho.nongrid_simulation import generate_simulated_data
from ptycho.probe import get_default_probe
from ptycho.diffsim import sim_object_image
import numpy as np

# 1. Configure simulation
config = TrainingConfig(
    model=ModelConfig(N=64, gridsize=2),
    n_groups=2000,
    nphotons=1e6
)

# 2. Generate synthetic object (256x256 "lines" pattern)
obj = sim_object_image(256, data_source='lines')  # Returns complex64

# 3. Get default disk probe
probe = get_default_probe(N=64, fmt='np')  # Returns (64, 64) complex

# 4. Generate simulated data with random scan coordinates
raw_data, gt_patches = generate_simulated_data(
    config=config,
    objectGuess=obj,
    probeGuess=probe,
    buffer=20.0,        # Min distance from object edges
    return_patches=True
)

# raw_data contains:
#   .xcoords, .ycoords: random positions within buffer
#   .diff3d: simulated diffraction patterns with Poisson noise
#   .Y: ground truth patches at each position
#   .probeGuess: the probe used for simulation
```

**Supported Object Types:**
- `'lines'`: Random line segments with Gaussian blur
- `'grf'`: Gaussian random field
- `'points'`: Random point sources

### 4.2. Direct RawData Construction

Create `RawData` from arrays without any file I/O:

```python
import numpy as np
from ptycho.raw_data import RawData

n_positions = 1000
N = 64

# Option A: From arrays with pre-computed diffraction
raw_data = RawData(
    xcoords=np.random.uniform(50, 200, n_positions),
    ycoords=np.random.uniform(50, 200, n_positions),
    xcoords_start=np.random.uniform(50, 200, n_positions),  # Can equal xcoords
    ycoords_start=np.random.uniform(50, 200, n_positions),  # Can equal ycoords
    diff3d=np.random.rand(n_positions, N, N).astype(np.float32),
    probeGuess=np.ones((N, N), dtype=np.complex64),
    scan_index=np.zeros(n_positions, dtype=int),
    objectGuess=None,  # Optional full object
    Y=None,            # Optional ground truth patches
)

# Option B: Simplified construction (start coords = end coords)
raw_data = RawData.from_coords_without_pc(
    xcoords=np.linspace(50, 200, n_positions),
    ycoords=np.linspace(50, 200, n_positions),
    diff3d=measured_patterns,
    probeGuess=probe,
    scan_index=np.zeros(n_positions, dtype=int),
    objectGuess=full_object  # Optional
)

# Option C: Simulate diffraction from object (gridsize=1 only)
raw_data = RawData.from_simulation(
    xcoords=np.array([100, 110, 120]),
    ycoords=np.array([100, 105, 115]),
    probeGuess=probe,
    objectGuess=full_object,
    scan_index=None  # Defaults to zeros
)
```

### 4.3. Direct PtychoDataContainer Construction

Skip `RawData` entirely and create model-ready tensors directly.

**IMPORTANT:** When bypassing `loader.load()`, you MUST attach `dataset_intensity_stats` to enable
the spec-compliant dataset-derived intensity_scale formula. Without this, training falls back to
the closed-form 988.21 constant, causing amplitude bias. See `specs/spec-ptycho-core.md §Normalization Invariants`.

```python
from ptycho.loader import PtychoDataContainer, compute_dataset_intensity_stats
import numpy as np

B, N, C = 1000, 64, 4  # batch, patch size, channels (gridsize²)

# Create diffraction data
X = np.random.rand(B, N, N, C).astype(np.float32) + 0.1

# REQUIRED: Compute dataset intensity stats from raw diffraction arrays
# This enables the dataset-derived intensity_scale formula instead of fallback
dataset_stats = compute_dataset_intensity_stats(X, is_normalized=False)
# Or if data is already normalized, pass the recorded intensity_scale:
# dataset_stats = compute_dataset_intensity_stats(X, intensity_scale=recorded_scale, is_normalized=True)

# Direct construction with all required arrays
container = PtychoDataContainer(
    X=X,
    Y_I=np.random.rand(B, N, N, C).astype(np.float32),
    Y_phi=np.random.rand(B, N, N, C).astype(np.float32),
    norm_Y_I=1.0,
    YY_full=None,
    coords_nominal=np.random.rand(B, 1, 2, C).astype(np.float32),
    coords_true=np.random.rand(B, 1, 2, C).astype(np.float32),
    nn_indices=np.zeros((B, C), dtype=np.int32),
    global_offsets=np.random.rand(B, 1, 2, C),
    local_offsets=np.zeros((B, 1, 2, C)),
    probeGuess=np.ones((N, N), dtype=np.complex64),
    dataset_intensity_stats=dataset_stats  # Required for correct intensity_scale
)

# Or via factory (combines RawData creation + grouping + loading)
# Note: The factory path handles stats automatically via loader.load()
container = PtychoDataContainer.from_raw_data_without_pc(
    xcoords=coords_x,
    ycoords=coords_y,
    diff3d=patterns,
    probeGuess=probe,
    scan_index=np.zeros(len(coords_x), dtype=int),
    objectGuess=None,
    N=64,
    K=7,
    nsamples=1000
)
```

### 4.4. Test Fixture Patterns

For unit tests, create deterministic synthetic data:

```python
import numpy as np
from ptycho.raw_data import RawData

def create_synthetic_raw_data(n_points=100, N=64, seed=42):
    """Create synthetic RawData for testing (deterministic, no I/O)."""
    np.random.seed(seed)

    # Deterministic coordinates (reproducible across runs)
    xcoords = np.linspace(50, 150, n_points)
    ycoords = np.linspace(50, 150, n_points)

    # Synthetic diffraction patterns
    diff3d = np.random.rand(n_points, N, N).astype(np.float32)

    # Simple uniform probe
    probe = np.ones((N, N), dtype=np.complex64)

    return RawData.from_coords_without_pc(
        xcoords, ycoords, diff3d, probe,
        scan_index=np.zeros(n_points, dtype=int)
    )

# Usage in tests
def test_my_function():
    raw_data = create_synthetic_raw_data(n_points=50, seed=42)
    # ... test logic using raw_data ...
```

### 4.5. Flow Summary

| Flow | Entry Point | NPZ Required | Global State | Output Type |
|------|-------------|--------------|--------------|-------------|
| NPZ Load | `RawData.from_file()` | Yes | No | `RawData` |
| Nongrid Sim | `generate_simulated_data()` | No | Temporarily | `RawData` |
| Grid Sim | `mk_simdata()` | No | Yes | Tensors |
| Direct RawData | `RawData()` | No | No | `RawData` |
| Direct Container | `PtychoDataContainer()` | No | No | `PtychoDataContainer` |
| From Simulation | `RawData.from_simulation()` | No | Yes | `RawData` |

**Recommendation:** For new code without NPZ files, use `generate_simulated_data()` with programmatic object/probe creation, or construct `RawData` directly from arrays.

---

## Related Documentation
- `docs/DEVELOPER_GUIDE.md` §1 — Two-system architecture
- `docs/DEVELOPER_GUIDE.md` §12 — Inference pipeline patterns
- `specs/spec-inference-pipeline.md` — API contracts for data containers
- `scripts/simulation/README.md` — Stage 1/Stage 2 simulation workflow
- `specs/data_contracts.md` — NPZ format specification
- `docs/findings.md` CONFIG-001 — params.cfg initialization requirement
