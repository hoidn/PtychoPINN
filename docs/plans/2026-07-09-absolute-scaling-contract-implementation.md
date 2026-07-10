# Absolute Scaling Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manuscript-conformant, NLL-only `ci_intensity_v2` scaling path for rectangular PyTorch training and raw-probe VarPro inference, while retaining old numerical behavior behind explicit `legacy_v1` compatibility.

**Architecture:** A new `ptycho_torch.scaling_contract` module owns profile resolution and dimensioned scale math. Existing mmap, in-memory, and dict adapters emit named CI tensors while retaining tuple aliases at their outer boundary. Training consumes the normalized probe with exact inverse compensation; inference consumes the physical probe directly. The amplitude forward remains unchanged and does not activate CI scaling.

**Tech Stack:** Python 3.11, PyTorch, Lightning, TensorDict, NumPy, pytest.

**Design authority:** `docs/superpowers/specs/2026-07-09-absolute-scaling-contract-design.md`

**Repository constraint:** Do not create git worktrees. Subagents work in the
shared checkout on sequential, non-overlapping tasks. Cross-branch integration
uses a disposable local clone after the fno-stable commit series is complete.

---

## File Map

- Create `ptycho_torch/scaling_contract.py`: profile constants, resolution, validation, CI statistics, amplitude adapter, and loss normalizer.
- Modify `ptycho_torch/config_params.py`: persisted profile fields with CI defaults.
- Modify `ptycho_torch/dataloader.py`: CI statistics, physical/training probe storage, named TensorDict fields, mmap/from-numpy parity.
- Modify `ptycho_torch/workflows/components.py`: dict-container CI adapter and explicit probe shape.
- Modify `ptycho_torch/model.py`: eager profile/loss validation, CI forward scale, count Poisson reduction and logging.
- Modify `ptycho_torch/reassembly.py`: physical-probe, mask-parity VarPro path.
- Modify `ptycho_torch/inference.py`: explicit compatibility overrides and canonical CI inference routing/fail-fast.
- Modify `scripts/studies/grid_lines_torch_runner.py`: pass resolved CI profile and convert normalized amplitude at the boundary.
- Modify `scripts/studies/make_synthetic_truth_datasets.py` and its count-dataset callers: count-calibrated probes and correct Poisson generation.
- Modify `docs/specs/spec-ptycho-core.md`, `docs/specs/spec-ptycho-interfaces.md`, `docs/DATA_NORMALIZATION_GUIDE.md`, `docs/findings.md`, and `docs/index.md`: atomic authority update and discoverability.
- Add focused tests under `tests/torch/test_absolute_scaling_*.py`.

---

### Task 1: Profile Resolution And Configuration Gate

**Files:**
- Create: `ptycho_torch/scaling_contract.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/model.py`
- Modify: `ptycho_torch/train.py`
- Modify: `ptycho_torch/train_lightning_only.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_absolute_scaling_contract.py`

**Prerequisites:** Design specification only.

- [ ] **Step 1: Write failing profile-table tests**

Cover all rows in the design table. Missing fields resolve to
`ci_intensity_v2/count_intensity`; explicit legacy requires both fields;
contradictory and unknown values raise `ValueError`.

- [ ] **Step 2: Write failing activation tests**

Assert that amplitude mode does not activate CI validation. Assert that
rectangular CI accepts only unsupervised `torch_loss_mode="poisson"`; MAE,
supervised mode, and any other primary loss fail during model construction.
Auxiliary object regularizers remain accepted.

- [ ] **Step 3: Run RED tests**

Run:

```bash
pytest -q tests/torch/test_absolute_scaling_contract.py
```

Expected: failures because profile fields and resolver do not exist.

- [ ] **Step 4: Implement the minimal contract API**

Add these public APIs:

```python
CI_SCALE_CONTRACT = "ci_intensity_v2"
LEGACY_SCALE_CONTRACT = "legacy_v1"
COUNT_INTENSITY = "count_intensity"
NORMALIZED_AMPLITUDE = "normalized_amplitude"

@dataclass(frozen=True)
class ResolvedScaleContract:
    version: str
    measurement_domain: str

def resolve_scale_contract(version=None, measurement_domain=None) -> ResolvedScaleContract: ...
def ci_scaling_active(model_config) -> bool: ...
def validate_scale_contract(data_config, model_config, training_config) -> ResolvedScaleContract: ...
```

Add `DataConfig.scale_contract_version` and `DataConfig.measurement_domain` with
CI defaults. `TrainingConfig.torch_loss_mode` is authoritative. Call validation
before any data module/container construction in `train.py`,
`train_lightning_only.py`, `_build_lightning_dataloaders`, and the grid-lines
dict entry point. In `PtychoPINN_Lightning.__init__`, validate before
`_resolve_generator_from_config`.

- [ ] **Step 5: Run GREEN tests and compatibility pins**

```bash
pytest -q tests/torch/test_absolute_scaling_contract.py \
  tests/torch/test_loss_modes.py \
  tests/torch/test_cross_branch_rectangular_parity.py
```

- [ ] **Step 6: Commit**

```bash
git add ptycho_torch/scaling_contract.py ptycho_torch/config_params.py \
  ptycho_torch/model.py ptycho_torch/train.py \
  ptycho_torch/train_lightning_only.py ptycho_torch/workflows/components.py \
  scripts/studies/grid_lines_torch_runner.py \
  tests/torch/test_absolute_scaling_contract.py
git commit -m "feat(torch): define CI absolute scaling profile"
```

### Task 2: Exact CI Statistics And Poisson Normalization

**Files:**
- Modify: `ptycho_torch/scaling_contract.py`
- Test: `tests/torch/test_absolute_scaling_math.py`

**Prerequisites:** Task 1.

- [ ] **Step 1: Write failing closed-form statistics tests**

For `(B,C,H,W)` count intensity, assert one experiment-level scalar:

```text
rms_input_scale = sqrt((N/2)^2 / mean_BC(sum_HW(I^2)))
mean_measured_intensity = mean_BCHW(I)
```

Cover multiple channels, non-square batch counts, non-finite values, negatives,
and all-zero rejection.

- [ ] **Step 2: Write the RED gradient-invariance test**

Use the same normalized prediction/observation at `S=1` and `S=329`. Build
physical counts as `(S*A)^2`; divide the Poisson data term by detached physical
mean intensity. Assert parameter gradients agree within `rtol=1e-5`. Also pin
that the current unnormalized denominator differs by approximately `S^2`.

- [ ] **Step 3: Run RED tests**

```bash
pytest -q tests/torch/test_absolute_scaling_math.py
```

Expected: missing statistics, adapter, and normalizer APIs fail.

- [ ] **Step 4: Implement CI scale math**

Add:

```python
@dataclass(frozen=True)
class CIExperimentStatistics:
    rms_input_scale: torch.Tensor
    mean_measured_intensity: torch.Tensor

def derive_ci_experiment_statistics(measured_intensity, N): ...
def normalize_ci_poisson_per_sample(raw_nll, mean_measured_intensity): ...
def adapt_normalized_amplitude_to_ci(amplitude, probe, count_amplitude_scale): ...
```

The adapter returns `(S*A)**2` and `S*probe` without mutating inputs.

- [ ] **Step 5: Run GREEN tests**

```bash
pytest -q tests/torch/test_absolute_scaling_math.py tests/torch/test_physics_scale.py
```

- [ ] **Step 6: Commit**

```bash
git add ptycho_torch/scaling_contract.py tests/torch/test_absolute_scaling_math.py
git commit -m "feat(torch): add dimensioned CI scaling math"
```

### Task 3: Mmap And In-Memory CI Batch Contract

**Files:**
- Modify: `ptycho_torch/dataloader.py`
- Modify: `ptycho_torch/train_utils.py`
- Modify: `ptycho_torch/lightning_utils.py`
- Modify: `ptycho_torch/model.py`
- Modify: `ptycho_torch/train.py`
- Modify: `ptycho_torch/train_lightning_only.py`
- Test: `tests/torch/test_absolute_scaling_mmap.py`
- Test: `tests/torch/test_multimode_probe_and_from_np.py`

**Prerequisites:** Tasks 1-2.

- [ ] **Step 1: Write failing mmap/from_np parity tests**

Create a tiny count-intensity NPZ with a two-mode physical probe. Assert both
constructors emit identical named fields:

```text
measured_intensity
rms_input_scale
mean_measured_intensity
probe_training
probe_physical
probe_normalization
```

Assert `probe_training == q * probe_physical`, `q` is computed jointly before
channel expansion, and returned probes have `(B,C,P,H,W)`.

- [ ] **Step 2: Write explicit legacy tests**

Set both legacy fields and assert existing `images`,
`rms_scaling_constant`, `physics_scaling_constant`, `batch[1]`, and `batch[2]`
remain byte-identical to the current fixtures.

- [ ] **Step 3: Run RED tests**

```bash
pytest -q tests/torch/test_absolute_scaling_mmap.py \
  tests/torch/test_multimode_probe_and_from_np.py
```

- [ ] **Step 4: Implement dual-profile storage**

Preserve raw probes in `data_dict['probes_physical']`; retain normalized probes
as `data_dict['probes']` for tuple compatibility. Add CI TensorDict fields and
per-experiment statistics. During `PtychoDataModule.setup("fit")`, compute those
statistics from the finalized deterministic `Subset.indices` and store them on
the data module; every DDP rank sees the same mmap and seeded split. Add a
`CIStatisticsCallback.on_fit_start` in `lightning_utils` that reads the prepared
data-module statistics and registers them on the model before the first batch or
checkpoint save. Both `train.py` and `train_lightning_only.py` install this
callback. Attach the frozen training values to validation batches.
`PtychoPINN_Lightning.on_save_checkpoint` persists the registered statistics;
`on_load_checkpoint` restores them. `lightning_utils` also writes them to run
configuration metadata. Add a lifecycle test for both entry points and a
checkpoint round-trip test proving inference receives the finalized training
values. New CI batches neither emit nor consume `physics_scaling_constant`;
only explicit legacy batches retain it.

For scalar and batched indices, named `probe_normalization` is 5-D while tuple
`batch[2]` remains the existing 4-D alias.

- [ ] **Step 5: Run GREEN and loader regression suites**

```bash
pytest -q tests/torch/test_absolute_scaling_mmap.py \
  tests/torch/test_multimode_probe_and_from_np.py \
  tests/torch/test_dataloader_batch_scale_semantics.py \
  tests/torch/test_loader_length_guards.py
```

- [ ] **Step 6: Commit**

```bash
git add ptycho_torch/dataloader.py ptycho_torch/train_utils.py \
  ptycho_torch/lightning_utils.py ptycho_torch/model.py ptycho_torch/train.py \
  ptycho_torch/train_lightning_only.py tests/torch/test_absolute_scaling_mmap.py \
  tests/torch/test_multimode_probe_and_from_np.py
git commit -m "feat(torch): emit physical CI mmap batches"
```

### Task 4: Dict Adapter And Batch-Invariant Probe Shapes

**Files:**
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_absolute_scaling_dict.py`
- Test: `tests/torch/test_inline_dataset_amplitude_scaling_regression.py`

**Prerequisites:** Tasks 1-2.

- [ ] **Step 1: Write failing normalized-amplitude adapter test**

Given `A`, `P`, and known `S`, assert the grid-lines rectangular CI container
emits `measured_intensity=(S*A)^2`, `probe_physical=S*P`, and CI statistics.
Assert the amplitude forward does not invoke this conversion.

- [ ] **Step 2: Write batch-size and multimode invariance tests**

For `B in {1,2,8}`, assert shared-probe forward inputs have
`(B,C,P,H,W)` and predicted diffraction does not acquire a batch-dependent
factor. Cover two incoherent modes and assert per-mode intensities are summed.

- [ ] **Step 3: Run RED tests**

```bash
pytest -q tests/torch/test_absolute_scaling_dict.py \
  tests/torch/test_inline_dataset_amplitude_scaling_regression.py
```

- [ ] **Step 4: Implement explicit dict conversion and collation**

Replace `derive_dict_physics_scale` as the rectangular CI mechanism with a
named adapter that owns `S`, measured intensity, physical probe, and statistics.
Keep the old helper for amplitude/legacy callers. Always return explicit probe
mode and channel axes; remove the raw `(B,H,W)` convention from CI and add a
legacy-only branch for frozen amplitude behavior.

Derive CI statistics once from the training dict. Pass those exact frozen
values into validation/test adapters rather than deriving split-local values.
The test fixture must use deliberately different train/test intensity
distributions and assert identical attached training statistics.

- [ ] **Step 5: Run GREEN tests and runner tests**

```bash
pytest -q tests/torch/test_absolute_scaling_dict.py \
  tests/torch/test_inline_dataset_amplitude_scaling_regression.py \
  tests/torch/test_dict_container_physics_scale.py \
  tests/torch/test_grid_lines_torch_runner.py
```

- [ ] **Step 6: Commit**

```bash
git add ptycho_torch/workflows/components.py scripts/studies/grid_lines_torch_runner.py \
  tests/torch/test_absolute_scaling_dict.py \
  tests/torch/test_inline_dataset_amplitude_scaling_regression.py
git commit -m "fix(torch): make CI dict scaling and probes explicit"
```

### Task 5: NLL-Only CI Training Forward

**Files:**
- Modify: `ptycho_torch/model.py`
- Test: `tests/torch/test_absolute_scaling_loss.py`
- Test: `tests/torch/test_rect_probe_scale_double_div.py`

**Prerequisites:** Tasks 1-4.

- [ ] **Step 1: Write an independent physical-forward oracle**

Construct a calibrated physical probe and known complex object. Compute
`I_ref=sum_p|FFT_ortho(M*P_physical*O)|^2` outside production code. Feed the
normalized training probe plus `q` through `PtychoPINN.compute_loss` helpers and
assert its predicted intensity matches `I_ref`.

- [ ] **Step 2: Write loss reduction and fail-fast tests**

Assert CI Poisson uses `measured_intensity`, clamps rate at `1e-8`, returns the
physical-mean-normalized optimization loss, and exposes/logs raw count NLL.
Assert CI never instantiates or calls `RectangularMAELoss`.

- [ ] **Step 3: Run RED tests**

```bash
pytest -q tests/torch/test_absolute_scaling_loss.py \
  tests/torch/test_rect_probe_scale_double_div.py
```

Expected: new physical-oracle and NLL-only assertions fail on legacy routing;
pre-existing compatibility pins remain green.

- [ ] **Step 4: Implement CI routing**

For CI rectangular training, set the field compensation to `1/q`, use the named
measured intensity, and normalize only the Poisson data term. Keep the current
`sqrt(1/(probe_scaling**2*physics_scale))` and double-square MAE branches only
under explicit legacy. Keep the amplitude branch byte-stable.

- [ ] **Step 5: Run GREEN and forward regressions**

```bash
pytest -q tests/torch/test_absolute_scaling_loss.py \
  tests/torch/test_rect_probe_scale_double_div.py \
  tests/torch/test_rectangular_scaled_forward.py \
  tests/torch/test_compute_loss_c4_regression.py \
  tests/torch/test_loss_modes.py
```

- [ ] **Step 6: Commit**

```bash
git add ptycho_torch/model.py tests/torch/test_absolute_scaling_loss.py \
  tests/torch/test_rect_probe_scale_double_div.py
git commit -m "fix(torch): train CI in physical count units"
```

### Task 6: Raw-Probe VarPro And Mask Parity

**Files:**
- Modify: `ptycho_torch/reassembly.py`
- Test: `tests/torch/test_absolute_scaling_varpro.py`
- Test: `tests/torch/test_varpro_solve_units.py`

**Prerequisites:** Tasks 3 and 5.

- [ ] **Step 1: Write independent VarPro oracle tests**

Generate counts from a known object and calibrated raw probe without calling
`compute_varpro_basis`. Assert CI VarPro recovers known `s1/s2` using the raw
probe, with no training output scale. Repeat with a soft mask and two incoherent
modes.

- [ ] **Step 2: Write dose/gauge tests**

Across three dose levels, scale both counts and physical probe consistently and
assert recovered object scale is invariant. As a negative control, hold the
probe fixed and assert recovered scale follows square-root dose.

- [ ] **Step 3: Run RED tests**

```bash
pytest -q tests/torch/test_absolute_scaling_varpro.py \
  tests/torch/test_varpro_solve_units.py
```

Expected: new raw-probe and mask-parity tests fail; explicit legacy unit tests
remain green.

- [ ] **Step 4: Implement profile-aware VarPro**

CI reads `probe_physical`, applies the same mask resolver as training, and calls
`compute_varpro_basis(..., scale=None)` on the full detector frame. Explicit
legacy retains normalized-probe/output-scale behavior for fixture reproduction.

- [ ] **Step 5: Run GREEN and reassembly suites**

```bash
pytest -q tests/torch/test_absolute_scaling_varpro.py \
  tests/torch/test_varpro_solve_units.py \
  tests/torch/test_varpro_probe_weighted_reassembly.py \
  tests/torch/test_inference_reassembly_parity.py \
  tests/torch/test_inference_reassembly_aggregation.py
```

- [ ] **Step 6: Commit**

```bash
git add ptycho_torch/reassembly.py tests/torch/test_absolute_scaling_varpro.py \
  tests/torch/test_varpro_solve_units.py
git commit -m "fix(torch): solve CI VarPro with the physical probe"
```

### Task 7: Count-Calibrated Synthetic Data

**Files:**
- Modify: `scripts/studies/make_synthetic_truth_datasets.py`
- Modify: `scripts/studies/make_lines_datasets.py`
- Modify: `scripts/studies/make_flux_sweep.py`
- Modify: `scripts/studies/make_dose_ladder_datasets.py`
- Modify: `scripts/studies/make_gridgeom_dataset.py`
- Modify: `scripts/studies/make_weakphase_test.py`
- Test: `tests/torch/test_absolute_scaling_dataset_generation.py`

**Prerequisites:** Tasks 1-2.

- [ ] **Step 1: Write failing physical-consistency tests**

Generate noiseless expected intensity from known object/probe, apply a requested
dose, scale the stored physical probe by the matching amplitude factor, then
draw Poisson counts. Assert the stored probe/object forward reproduces expected
counts in mean and that metadata records both CI profile fields and probe gauge.

- [ ] **Step 2: Preserve a legacy helper explicitly**

If historical scripts/tests require deterministic rescaling of an existing
amplitude, retain it under a clearly named legacy helper. It must not be the CI
builder or claim a fresh Poisson measurement.

- [ ] **Step 3: Run RED tests**

```bash
pytest -q tests/torch/test_absolute_scaling_dataset_generation.py \
  tests/torch/test_make_aligned_count_twin.py
```

Expected: physical probe/count consistency and CI metadata assertions fail;
existing twin arithmetic tests remain green.

- [ ] **Step 4: Implement calibrated generation**

Add the CI generator and migrate the listed callers. Keep the explicitly named
legacy helper for historical artifact reproduction.

- [ ] **Step 5: Run GREEN tests**

```bash
pytest -q tests/torch/test_absolute_scaling_dataset_generation.py \
  tests/torch/test_make_aligned_count_twin.py
```

- [ ] **Step 6: Commit**

```bash
git add scripts/studies/make_synthetic_truth_datasets.py \
  scripts/studies/make_lines_datasets.py scripts/studies/make_flux_sweep.py \
  scripts/studies/make_dose_ladder_datasets.py scripts/studies/make_gridgeom_dataset.py \
  scripts/studies/make_weakphase_test.py \
  tests/torch/test_absolute_scaling_dataset_generation.py
git commit -m "fix(studies): generate count-calibrated CI datasets"
```

### Task 8: Inference And Compatibility Entry Points

**Files:**
- Modify: `ptycho_torch/inference.py`
- Modify: `ptycho_torch/train.py`
- Modify: `ptycho_torch/train_lightning_only.py`
- Modify: `ptycho_torch/config_factory.py`
- Modify: `ptycho_torch/workflows/components.py`
- Test: `tests/torch/test_absolute_scaling_entrypoints.py`
- Test: `tests/torch/test_inference_normalization.py`

**Prerequisites:** Tasks 1 and 6.

- [ ] **Step 1: Write CLI/config tests**

Missing profile flags default to CI for rectangular workflows. Add paired
legacy override flags that must be supplied together. Known legacy checkpoints
without metadata fail unless both overrides are present.

- [ ] **Step 2: Write simplified-inference routing test**

Assert CI checkpoint inference routes through canonical barycentric VarPro or
fails with a message naming that required path. It must not compute and discard
an output scale. Explicit legacy simplified inference remains available.

- [ ] **Step 3: Run RED tests**

```bash
pytest -q tests/torch/test_absolute_scaling_entrypoints.py \
  tests/torch/test_inference_normalization.py \
  tests/torch/test_config_factory.py
```

Expected: paired override, metadata-free checkpoint, and canonical CI inference
routing tests fail.

- [ ] **Step 4: Implement entry-point routing**

Thread profile fields and paired overrides through `config_factory.py`, CLI
entry points, and bundle loading in `components.py`. Recover persisted training
statistics from checkpoint/bundle metadata for CI inference.

- [ ] **Step 5: Run GREEN tests**

```bash
pytest -q tests/torch/test_absolute_scaling_entrypoints.py \
  tests/torch/test_inference_normalization.py \
  tests/torch/test_config_factory.py
```

- [ ] **Step 6: Commit**

```bash
git add ptycho_torch/inference.py ptycho_torch/train.py \
  ptycho_torch/train_lightning_only.py ptycho_torch/config_factory.py \
  ptycho_torch/workflows/components.py \
  tests/torch/test_absolute_scaling_entrypoints.py \
  tests/torch/test_inference_normalization.py
git commit -m "feat(torch): version CI scaling entry points"
```

### Task 9: Normative Documentation And Historical Supersession

**Files:**
- Modify: `docs/specs/spec-ptycho-core.md`
- Modify: `docs/specs/spec-ptycho-interfaces.md`
- Modify: `docs/DATA_NORMALIZATION_GUIDE.md`
- Modify: `docs/findings.md`
- Modify: `docs/index.md`

**Prerequisites:** Tasks 1-8.

- [ ] **Step 1: Update normative profiles atomically**

Separate standalone legacy amplitude and CI count-intensity schemas, document
profile defaults, formulas, NLL-only activation, raw-probe inference, explicit
legacy selection, and tuple aliases.

- [ ] **Step 2: Supersede stale findings without deleting evidence**

Correct claims that normalized-probe/output-scale VarPro proves absolute units,
that the mmap `physics_scaling_constant` is a photon scale, and that Adam epsilon
is the primary `count_scale_mode=auto` explanation. Preserve measured historical
results and label their contract.

- [ ] **Step 3: Validate references and commit**

```bash
rg -n "ci_intensity_v2|legacy_v1|measurement_domain|probe_physical" \
  docs/specs docs/DATA_NORMALIZATION_GUIDE.md docs/findings.md docs/index.md
git diff --check
git add docs/specs/spec-ptycho-core.md docs/specs/spec-ptycho-interfaces.md \
  docs/DATA_NORMALIZATION_GUIDE.md docs/findings.md docs/index.md
git commit -m "docs: publish CI absolute scaling contract"
```

### Task 10: Verification, Review, And Cross-Branch Integration

**Files:**
- Modify: `docs/plans/2026-07-09-absolute-scaling-contract-implementation.md`
- Create: `.artifacts/absolute_scaling_ci_v2/pytest-focused.log` (ignored evidence)
- Create: `.artifacts/absolute_scaling_ci_v2/pytest-nonslow.log` (ignored evidence)

**Prerequisites:** Tasks 1-9.

- [ ] **Step 1: Run focused absolute-scale suite**

```bash
mkdir -p .artifacts/absolute_scaling_ci_v2
pytest -q tests/torch/test_absolute_scaling_*.py \
  tests/torch/test_multimode_probe_and_from_np.py \
  tests/torch/test_varpro_solve_units.py \
  2>&1 | tee .artifacts/absolute_scaling_ci_v2/pytest-focused.log
```

- [ ] **Step 2: Run broader PyTorch non-slow suite in tmux**

Use the registered markers only, activate `ptycho311`, and track the exact
pytest PID inside tmux:

```bash
tmux new-session -d -s absolute_scale_nonslow "bash -lc '
  source /home/ollie/miniconda3/etc/profile.d/conda.sh &&
  conda activate ptycho311 &&
  mkdir -p .artifacts/absolute_scaling_ci_v2 &&
  CUDA_VISIBLE_DEVICES= python -m pytest -q \
    -m \"not slow and not grid_lines_hybrid_resnet_integration and not grid_lines_hybrid_resnet_aligned_ablation\" \
    > >(tee .artifacts/absolute_scaling_ci_v2/pytest-nonslow.log) 2>&1 &
  pid=\$!; echo \$pid > .artifacts/absolute_scaling_ci_v2/pytest-nonslow.pid;
  wait \$pid; rc=\$?;
  echo \$rc > .artifacts/absolute_scaling_ci_v2/pytest-nonslow.exit;
  exit \$rc
'"
```

Wait for that exact tmux command to exit and require
`pytest-nonslow.exit == 0`. Repeat this exact selector and logging contract in
the disposable `main` clone.

- [ ] **Step 3: Run project integration marker**

```bash
pytest -q -m integration
```

- [ ] **Step 4: Run the repository CI gate**

```bash
bash ci/run_ci_tests.sh
```

- [ ] **Step 5: Dispatch final spec and code-quality reviews**

Resolve all blocking findings and rerun affected tests.

- [ ] **Step 6: Port the reviewed commit series to `main` in a disposable clone**

Clone the local repository into `/tmp/ptychopinn-absolute-scale-main`, fetch
`origin/main`, cherry-pick the reviewed implementation commits, resolve against
the consolidated loader commit, and run the focused, non-slow, integration, and
`ci/run_ci_tests.sh` gates there. Push `main` only as a normal fast-forward; do
not force-push. This leaves the user's dirty fno-stable checkout untouched.

- [ ] **Step 7: Record final evidence and commit the completed plan**

```bash
git add docs/plans/2026-07-09-absolute-scaling-contract-implementation.md
git commit -m "docs: record absolute scaling migration evidence"
```

## Acceptance Gate

The work is complete only when:

- CI rectangular + Poisson NLL passes the independent physical oracle.
- CI plus MAE fails before loading data.
- Missing profile fields resolve to CI.
- Explicit legacy reproduces frozen parity behavior.
- Raw-probe VarPro passes calibrated-dose and negative-control sweeps.
- Shared/multimode probes are batch-size invariant.
- mmap, from-numpy, and dict adapters agree.
- Focused, non-slow, integration, and `ci/run_ci_tests.sh` evidence is recorded
  on both target branches.
