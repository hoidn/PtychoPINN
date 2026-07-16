# PtychoPINN Developer Guide

## Document Purpose

This document is the canonical guide to the project's architecture-level rules: which
system a change belongs to, the anti-patterns that have caused real bugs, the data
pipeline's contracts, and the conventions for logging, testing, and evaluation. It
records *why* the rules exist (each grew out of a debugging incident) so that future
work stays robust and consistent with the project's design.

It intentionally does **not** duplicate content owned by other documents:

- **CLI recipes (training, inference, tests):** `docs/COMMANDS_REFERENCE.md`
- **Test commands and suite layout:** `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`
- **PyTorch backend workflows:** `docs/workflows/pytorch.md`
- **Torch architecture & component contracts:** `docs/architecture_torch.md`
- **Known issues, policies, finding IDs:** `docs/findings.md`
- **Data contracts:** `docs/specs/spec-ptycho-core.md` (standalone NPZ), `specs/data_contracts.md` (Ptychodus HDF5)

## Related Documentation

- **<doc-ref type="guide">docs/debugging/TROUBLESHOOTING.md</doc-ref>** — debug shape mismatches and config problems
- **<doc-ref type="guide">docs/debugging/QUICK_REFERENCE_PARAMS.md</doc-ref>** — `params.cfg` initialization patterns
- **<doc-ref type="guide">docs/development/INVOCATION_LOGGING_GUIDE.md</doc-ref>** — CLI/orchestration invocation provenance artifacts
- **<doc-ref type="guide">docs/findings.md</doc-ref>** — the finding/policy registry cited throughout this guide

---

## 1. The Core Concept: A "Two-System" Architecture

**The Lesson:** The repository contains two distinct, semi-independent systems: a
legacy, grid-based system and a modern, coordinate-based system. Many bugs arise from
the friction between them.

| Feature | Legacy "Grid-Based" System | Modern "Coordinate-Based" System |
| --- | --- | --- |
| **Primary Workflow** | Legacy modules invoked through `ptycho.params.cfg` consumers | `ptycho_train` / `ptycho_inference` (see `pyproject.toml [project.scripts]`), `scripts/run_comparison.sh`, `scripts/run_baseline.py` |
| **Configuration** | Legacy `ptycho.params.cfg` global dictionary | `ptycho.config.config` dataclasses (`TrainingConfig`, `InferenceConfig`), bridged via `update_legacy_dict` |
| **Patch Reassembly** | `ptycho.image.stitching.stitch_patches` | `ptycho.tf_helper.reassemble_position` |
| **Characteristic** | Global state and implicit configuration | Explicit data flow; configuration via function arguments |

**The Rule:** Before starting any task, identify which system you are operating in. The
long-term direction is to migrate functionality to the modern, coordinate-based system
and eliminate reliance on global state.

### 1.1. The PyTorch Backend

The modern system has a second implementation: the PyTorch backend under
`ptycho_torch/` (Lightning-based model, memory-mapped dataloader, and a generator
registry — `ptycho_torch.generators.registry.resolve_generator` — that selects among
CNN, FNO, FFNO, and hybrid architectures). PyTorch (torch ≥ 2.2) is a mandatory
dependency.

Rules specific to this backend:

- **Workflow authority:** `docs/workflows/pytorch.md` governs configuration and
  execution (`python -m ptycho_torch.train` / `python -m ptycho_torch.inference`).
- **Config bridging:** torch-side configs (`ptycho_torch/config_params.py`) are bridged
  from the canonical dataclasses via `ptycho_torch/config_bridge.py`. Torch workflows
  must still run `update_legacy_dict(params.cfg, config)` before touching legacy
  modules. `PyTorchExecutionConfig` controls runtime behavior only and must **never**
  populate `params.cfg`.
- **Component contracts:** IDL-style contracts for the core data and workflow APIs
  (`RawData`, `PtychoDataContainer`, `run_grid_lines_torch`, `run_cdi_example_torch`,
  `resolve_generator`) live in <doc-ref type="guide">docs/architecture_torch.md</doc-ref>
  §6 Component Contracts. Docstrings must cross-reference them.

---

## 2. Critical Architectural Principles & Anti-Patterns

Fundamental rules to avoid introducing fragile, difficult-to-debug code.

### 2.1. Anti-Pattern: Side Effects on Import

A module must **never** perform complex, state-dependent operations (loading or
generating data, building models) at the top level. Design functions to receive all
data they need as explicit arguments.

```python
# Incorrect: importing this module re-runs a data pipeline
from ptycho.generate_data import YY_ground_truth

# Correct: the dependency is an explicit argument
def save_recons(model_type, stitched_obj, ground_truth_obj=None):
    ...
```

### 2.2. Anti-Pattern: Implicit Dependencies via Global State

**The Lesson:** Relying on the global configuration dictionary (`ptycho.params.cfg`)
makes the codebase fragile and introduces unsafe initialization-order dependencies.

#### Configuration Migration

Until the codebase is fully refactored, all modern scripts must follow this order
(the "safe initialization pattern"; see
`docs/debugging/QUICK_REFERENCE_PARAMS.md`):

1. Set up configuration using the modern `TrainingConfig`/`InferenceConfig` dataclasses.
2. Update the legacy global dictionary: `update_legacy_dict(params.cfg, config)`
   (defined in `ptycho/config/config.py`).
3. Load any necessary data (e.g., a probe from a file).
4. Use this data to populate any remaining required keys in `params.cfg`
   (e.g., `p.set('probe', ...)`).
5. **Only then**, perform local imports of modules (`ptycho.model`, `ptycho.nbutils`)
   that depend on this global state.

### 2.3. Anti-Pattern: Import-Time Model Construction

**The Lesson:** `ptycho.model` historically built a module-level model at import time
using the then-current `params.cfg['gridsize']`, freezing the architecture with
whatever gridsize happened to be set. Updating `params.cfg` after import cannot fix
this — an already-constructed model is never rebuilt.

**The Rule:** Never construct models at module scope. `ptycho/model.py` now defers
singleton creation via lazy `__getattr__`; custom workflows must use the factory
functions:

- `ptycho.model.create_compiled_model(...)` — compiled model ready for training
- `ptycho.model.create_model_with_gridsize(...)` — uncompiled model

Related initialization-order rules from the same incident class:

- Never use training config loaders for inference — they have different parameter
  priorities.
- Never import model-constructing modules at top level in scripts.
- Never load a saved model before applying its configuration to `params.cfg`
  (`ptycho/model_manager.py` enforces this order).

### 2.4. Interpreter & Subprocess Policy (PYTHON-ENV-001)

**Rule:** invoke Python via the default PATH `python` for code, scripts, and
documentation. This section is the single source of truth for interpreter selection.

**Why:** The default interpreter selected by the user's environment is the single
source of truth for commands and subprocesses. Keeping examples and subprocess
invocations consistent with `python` avoids hidden per-host overrides and simplifies
instructions.

Canonical patterns:
- In Python code spawning Python subprocesses:
  ```python
  import subprocess
  subprocess.run(["python", "-m", "pkg.module", "--flag", "value"], check=True)
  ```
- In shell scripts and docs examples:
  ```bash
  python -m pkg.module --flag value
  ```

**Scope:** applies to new/modified code, orchestration helpers, and command examples.
Keep commands environment-agnostic (no hardcoded conda env names).

**Exceptions:** legacy archives under `archive/` and tests expressly validating legacy
behavior may retain historical commands.

**Enforcement:** avoid introducing repository-specific interpreter indirection
(e.g., `PYTHON_BIN` wrappers) in docs/snippets.

**Session-local workflow rule:** when a controller or orchestration helper launches
child study commands against a dedicated run/session checkout, it must keep plain PATH
`python` while explicitly setting `PYTHONPATH` to that session repo root. Do not
inherit ambient launcher `PYTHONPATH`: cross-checkout module resolution can silently
score the wrong source tree even when the child command string and Git `HEAD` look
correct.

### 2.5. Invocation Provenance for Scripts and Orchestrators

**Rule:** study scripts and wrappers (`scripts/studies/*`) MUST persist invocation
artifacts (`invocation.json`, `invocation.sh`) in deterministic output locations using
the shared helper pattern. This is mandatory for new scripts and for modified existing
scripts. See <doc-ref type="guide">docs/development/INVOCATION_LOGGING_GUIDE.md</doc-ref>
for artifact names, placement conventions, and test requirements.

### 2.6. Anti-Pattern: Implicit Device Inheritance Across Train→Infer Boundaries

**Rule:** do not assume model device placement survives `Trainer.fit(...)`. At every
train→infer boundary, explicitly resolve the target inference device and call
`model.to(device)` before the forward loop. Relying on implicit parameter-device
inheritance can silently force CPU inference.

### 2.7. Principle: Separation of Data Shaping and Model Responsibilities

A model's core architecture should be fixed and define a clear data contract; it is
the responsibility of the data pipeline or calling script to shape data to match that
contract.

```python
# Correct: the calling script shapes the data.
if n_channels > 1:
    X_train_in = _channel_to_flat(X_train_in)

# Anti-pattern: the model adapts its I/O structure to the input shape.
c = X_train.shape[-1]           # BUG
decoded1 = Conv2D(c, ...)(x1)   # BUG
```

---

## 3. The Data Pipeline: Contracts and Bookkeeping

A data pipeline's file formats and loading logic constitute a public API. Its behavior
must be explicit and robust.

**The Canonical Data Format:** All tools that produce or consume standalone NPZ
datasets **MUST** adhere to
**<doc-ref type="contract">docs/specs/spec-ptycho-core.md</doc-ref>** — the single
source of truth for standalone-NPZ array shapes, key names, and dtypes. Ptychodus HDF5
product files are governed by `specs/data_contracts.md`.

### 3.1. Rule: Explicit `dtype` for Non-Default Array Types

Always pass an explicit `dtype` when initializing NumPy arrays that will hold
non-default types (`np.zeros(..., dtype=np.complex64)`). Assigning complex data into a
default `float64` array silently discards the imaginary part — this once caused the
supervised model to train on amplitude only.

### 3.2. Rule: The Data File Format is a Strict API

An inconsistent file format is a bug in the script that **generates** it, never a
problem to be solved by the script that **loads** it. Every per-image array in an NPZ
file must carry its batch dimension in the same position; fix the generator
(`scripts/tools/transpose_rename_convert_tool.py` and friends), not the loader.

### 3.3. Rule: Prioritize Prepared Data; Fail on Ambiguity

A data loader must not be "helpfully" ambiguous: check for the most processed,
prepared version of the data first (the `Y` array), and raise rather than silently
falling back to regenerating data. (`objectGuess` being present for evaluation does
not mean patches should be re-derived from it.)

### 3.4. Core Tensor Formats for gridsize > 1

To handle overlapping patches, the codebase uses three primary tensor formats:

- **Channel Format (`B, N, N, C`)** — the format for **neural network processing**.
  The `C = gridsize**2` neighboring patches are treated as channels. Produced by
  `ptycho.raw_data.get_image_patches`; expected by the U-Net in `ptycho/model.py`.
- **Flat Format (`B*C, N, N, 1`)** — the format for **individual patch physics
  simulation**. Each of the `C` patches is a separate batch item. **Required input
  format for `ptycho.diffsim.illuminate_and_diffract`.**
- **Grid Format (`B, G, G, N, N, 1`)** — a transitional format that makes the physical
  2D grid of patches explicit.

**CRITICAL RULE:** Use `ptycho.tf_helper._channel_to_flat()` to convert from Channel
to Flat Format before calling the core physics simulation engine.

### 3.5. Normalization Architecture: Three Distinct Systems

**The Critical Lesson:** PtychoPINN uses three separate normalization systems that
must never be confused. Mixing them is a recurring source of subtle bugs. (Full
treatment: <doc-ref type="guide">docs/DATA_NORMALIZATION_GUIDE.md</doc-ref>.)

1. **Physics normalization (`intensity_scale`)** — scales simulated data to realistic
   photon counts. `ptycho/diffsim.py` *calculates* the scale but does **not** apply
   it; scaling is applied only in the physics loss layer during training. Internal
   pipeline data stays normalized.
2. **Statistical normalization (`normalize_data`)** — standard ML preprocessing,
   applied in the data loader before model input (`ptycho/loader.py`; the output
   contract is `float32`). Completely independent from physics normalization.
3. **Display/comparison scaling** — visual adjustments in plotting and metric code
   only (`ptycho/image/`, comparison scripts). Never affects training or physics.

**The Rule:** document which normalization you're using, keep them separate, and apply
physics scaling only at the model's physics boundary. Never apply `intensity_scale` in
the data pipeline (double-scaling), and never compose the systems
(`normalize_data(X * intensity_scale)` confuses two of them).

---

## 4. Physical Consistency in Data Preprocessing

### 4.1. Downsampling: Binning vs. Cropping

**The Lesson:** Downsampling must be physically consistent across related arrays. A
downsampled diffraction pixel represents an average over a detector area, so the
corresponding real-space object patch must be downsampled by **binning (averaging)**,
not cropping.

**The Rule:** Real-space arrays (`objectGuess`, `probeGuess`, `Y` patches) are
downsampled with `bin_complex_array`; k-space arrays (`diffraction`) with
`crop_center`. `scripts/tools/downsample_data_tool.py` implements this correctly.

### 4.2. Simulation Consistency

**The Principle:** A dataset's `diffraction` array is only physically valid for the
specific `objectGuess` and `probeGuess` it was generated from.

**The Rule:** If you modify the object or probe in any way (e.g., upsampling or
smoothing via `scripts/tools/prepare_data_tool.py`), the original `diffraction` data
is invalid: you **must** re-simulate a new `diffraction` array. `scripts/prepare.sh`
models this workflow correctly.

### 4.3. Critical Data Flow: The Patch Extraction Pipeline

Patch extraction in `ptycho/raw_data.py` involves a coordinate-order convention that
must be respected:

1. **Offset creation:** `offsets_c` is built by stacking `ycoords` then `xcoords`,
   giving **`[y_offset, x_offset]`** order.
2. **Translation function:** `ptycho.tf_helper.translate` expects translations in
   **`[dx, dy]`** (`[x_offset, y_offset]`) order.
3. **The required swap:** the coordinate vector **must be swapped** before being passed
   to `translate`.

```python
# offsets_yx has shape (batch, 2) and order [y, x]
offsets_yx = tf.reshape(offsets_f, (-1, 2))

# Swap columns to get [x, y] order for the translate function
offsets_xy = tf.gather(offsets_yx, [1, 0], axis=1)

translated_patches = hh.translate(images, -offsets_xy)
```

All new and refactored code must perform the explicit swap. (Omitting it produces
transposed translations that look correct on symmetric data.)

---

## 5. Authoritative Methods for Evaluation

To keep model comparisons fair, the project uses single, authoritative functions per
backend for common evaluation tasks.

### 5.1. Patch Reassembly

Reassembly places a small central region of each predicted patch onto a large canvas
at its real-valued scan coordinates, normalizing overlapping regions. Each backend has
one authoritative implementation:

- **TensorFlow:** `ptycho.tf_helper.reassemble_position` — used by
  `ptycho/workflows/components.py`, `scripts/compare_models.py`, and
  `scripts/run_baseline.py`.
- **PyTorch:** `ptycho_torch/helper.py` (`reassemble_patches_position_real`, plus the
  probe-weighted variant `reassemble_patches_position_real_probe`), with batch
  pipelines in `ptycho_torch/reassembly.py`; the workflow entry point is
  `run_cdi_example_torch(..., do_stitching=True)`. The barycentric implementations in
  `ptycho_torch/reassembly_beta.py` are experimental.

**The Rule:** within any model comparison, every reconstruction must be stitched with
the same reassembly method — never compare outputs stitched by different
implementations. `ptycho.image.stitching.stitch_patches` is the grid-based legacy path
and must not be used for coordinate-based (non-grid) outputs.

### 5.2. Evaluation Alignment

**The Correct Function:** `ptycho.image.cropping.align_for_evaluation`

```python
def align_for_evaluation(
    reconstruction_image: np.ndarray,
    ground_truth_image: np.ndarray,
    scan_coords_yx: np.ndarray,
    stitch_patch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Aligns a reconstruction with ground truth using scan coordinates and the
    stitching patch size ('M' from reassemble_position) to compute the precise
    bounding box for a physically correct comparison."""
```

**The Rule:** Any script that calculates metrics (e.g., `scripts/run_baseline.py`,
`scripts/compare_models.py`) **must** use this function to prepare inputs for
`eval_reconstruction`.

---

## 6. Centralized Logging

**The Golden Rule:** All logs for a run live in a `logs/` subdirectory of that run's
output directory — never in the project root.

`ptycho/log_config.py` (`setup_logging`) is the standard mechanism: it tees all log
messages plus captured stdout to `<output_dir>/logs/debug.log`, with console verbosity
controlled independently. `ptycho/cli_args.py` (`add_logging_arguments`,
`get_logging_config`) provides the standard `--quiet` / `--verbose` /
`--console-level` flags, which the unified CLIs already register.

**Anti-Pattern:** do not add `logging.basicConfig()` or manual `FileHandler`s to
scripts — that scatters log files and bypasses stdout capture. Wire new scripts
through:

```python
setup_logging(output_dir, **get_logging_config(args))
```

---

## 7. Testing Conventions

**Authority:** Repository scope-and-evidence policy determines what is a completion
gate. `docs/TESTING_GUIDE.md` owns test command mechanics and evidence guidance;
`docs/development/TEST_SUITE_INDEX.md` catalogs selectors. This section covers only
the structural conventions. The project encourages test-driven development: write the
failing test that reproduces the bug or specifies the feature before the
implementation.

### 7.1. Test Directory Structure

All tests for library code live in the top-level `tests/` directory, mirroring the
package layout. Tests for the PyTorch backend live under `tests/torch/`.

```
tests/
├── test_baselines.py          # Tests for ptycho/baselines.py
├── image/
│   ├── test_cropping.py       # Tests for ptycho/image/cropping.py
│   └── test_registration.py   # Tests for ptycho/image/registration.py
├── workflows/
│   └── test_components.py     # Tests for ptycho/workflows/components.py
├── scripts/                   # Tests for entry points under scripts/
├── studies/                   # Tests for scripts/studies/*
└── torch/                     # Tests for ptycho_torch/* (CI-gated)
```

**Naming:** `test_<module_name>.py`, matching the module under test.

**Script-level tests:** tests for standalone scripts follow the same mirroring
convention (`tests/scripts/`, `tests/studies/`) — do not co-locate test files inside
`scripts/`.

### 7.2. Running Tests

The project standard is **pytest** (see `docs/TESTING_GUIDE.md` for the full command
library and evidence guidance):

```bash
pytest tests/ -q                  # full suite
pytest tests/torch -m "not slow"  # the fast torch suite
```

For each task, run fresh selectors capable of falsifying the affected acceptance
claim and governing invariants. When the active roadmap or plan names the repository
CI boundary, run the checked-in harness exactly as `bash ci/run_ci_tests.sh` at that
boundary; the public `main` branch's `pytest-cpu` job invokes the same script with the
deselect/ignore lists maintained alongside it. Supplemental checks do not become
completion gates unless the current acceptance scope makes them relevant.

A few legacy suites still use `unittest`; run those modules explicitly per the
Testing Guide. Do not add new `unittest`-style discovery flows.

**Collection caveat:** module-level imports in test files execute at collection time
even for deselected/slow-marked tests, so anything a gated test file imports at module
scope (including from `scripts/studies/`) is a hard CI dependency.

---

## 8. Data Handling for Overlap-Based Training

The `gridsize` parameter controls the use of overlapping scan positions in the physics
model. The data loading pipeline uses one unified sampling strategy for all gridsize
values ("sample-then-group"):

1. **Random sampling of anchor points** from the complete set of scan coordinates.
2. **Neighbor grouping:** for gridsize > 1, the K-nearest neighbors of each anchor
   form a group; for gridsize = 1, each anchor is a single-element group.

Consequences:

- **No manual shuffling is needed for gridsize=1** — sampling is random for all
  gridsize values. Pre-shuffled datasets continue to work.
- **Sequential behavior** (first N images) is available via the
  `sequential_sampling` config field / `--sequential_sampling` flag
  (`ptycho.raw_data.RawData.generate_grouped_data`).
- With gridsize=2, model input tensors have `C = gridsize² = 4` channels; the loader
  (`ptycho/loader.py`) and model handle the multi-channel format, and the training log
  confirms the interpretation:

```
INFO - Parameter interpretation: --n_groups=500 refers to neighbor groups (gridsize=2, total patterns=2000)
```

---

## 9. Code Review Checklist

Before merging any PR that touches data loading or configuration:

### 9.1. Configuration Flow
- [ ] Does the code access `params.get()` or `params.cfg`? If yes, is
      `update_legacy_dict()` called first, and is there a fallback if params are
      uninitialized?
- [ ] Are there hidden dependencies on global state? Prefer explicit parameters.
- [ ] Is the initialization order documented, with a pointer to
      `docs/debugging/TROUBLESHOOTING.md` where relevant?

### 9.2. Shape Validation
- [ ] For gridsize-dependent code, are there tests for gridsize=1 AND gridsize=2?
- [ ] Do integration tests verify expected output shapes?
- [ ] Is there validation that catches shape mismatches early?

### 9.3. Error Messages
- [ ] Do error messages include configuration context?
  ```python
  # Good
  raise ValueError(f"Expected shape (*,*,*,4) for gridsize=2, got {shape}. "
                   f"Check params.cfg['gridsize']={params.cfg.get('gridsize')}")
  # Bad
  raise ValueError("Shape mismatch")
  ```

### 9.4. Documentation
- [ ] Are all `params.cfg` dependencies documented in docstrings?
- [ ] Do docstrings cross-reference the component contracts
      (`docs/architecture_torch.md` §6) where one exists?
