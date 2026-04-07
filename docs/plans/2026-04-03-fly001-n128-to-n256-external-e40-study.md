# Fly001 N128-to-N256 External E40 Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a principled manual `40`-epoch external `fly001` study at `256x256` by bridging the canonical `fly001_128` top-half-train/full-object-test split into a physically consistent `N=256` dataset family.

**Architecture:** Keep the existing external-study flow instead of reusing the synthetic `lines_256` wrapper. Add a dedicated preparation helper that loads the canonical `fly001_128` raw NPZ split, upsamples the complex object/probe from `128 -> 256`, scales the existing scan coordinates by `2x`, resimulates diffraction at those explicit coordinates, and emits new external raw NPZs plus a manifest. Then add a one-off manual Torch runbook that uses `build_datasets(dataset_source="external_raw_npz", ...)` and `run_grid_lines_torch(...)` to train a single `hybrid_resnet` run for `40` epochs and write the usual recon/metric/visual outputs.

**Tech Stack:** Python 3.11, NumPy NPZ I/O, SciPy interpolation, TensorFlow-backed `RawData.from_simulation(...)`, `GridLinesConfig`, `grid_study_dataset_builder`, `grid_lines_torch_runner`, pytest, bash runbooks

---

## Design Summary

### Study Contract

This is a new external dataset family, not a variant of the synthetic `lines_256` study:

- Source split stays canonical:
  - train: `datasets/fly001_128/fly001_128_top_half_converted.npz`
  - test: `datasets/fly001_128/fly001_128_full_test_converted.npz`
- Split semantics stay unchanged:
  - train uses the top-half scan subset
  - test uses the full-object scan set
- New family name should be explicit, for example:
  - `fly001_external_n256_upsampled_from_n128_top_train_full_test_v1`

### Why Not Use the Existing `lines_256` Wrapper

`scripts/studies/run_lines_256_arch_experiment.py` is the wrong boundary for this work because it hard-pins:

- the synthetic `lines_256` dataset contract
- `20` epochs
- the current session-controller lineage

This study should instead follow the existing external-fly pattern from:

- `scripts/studies/runbooks/grid_lines_external_fly001_n128_top_train_full_test_e40.sh`

### Simulation Contract

The bridge dataset must be physically consistent. Interpolating `objectGuess` and `probeGuess` alone is not enough because the diffraction in the raw external NPZ would still correspond to the old `128x128` fields.

The principled contract is:

1. Load the canonical `fly001_128` raw train/test NPZs.
2. Upsample `objectGuess` and `probeGuess` from `128 -> 256` using the same cubic real/imag interpolation rule already used by `prepare_data_tool.py`.
3. Scale scan coordinates by `2.0` so they refer to the same physical positions on the enlarged object grid.
4. Resimulate diffraction at those explicit scaled coordinates using `RawData.from_simulation(...)` under `N=256`, `gridsize=1`, and the runbook photon budget.
5. Save new raw external NPZs containing the new diffraction plus the upsampled fields and scaled coordinates.
6. Feed those raw NPZs into `build_datasets(dataset_source="external_raw_npz", ...)` to get grouped Torch train/test bundles.

### Non-Goals

- Do not modify the synthetic `lines_256` controller/workflow for this study.
- Do not create a generic multi-resolution external-study framework beyond what this single bridge needs.
- Do not silently mutate the existing `fly001_128` dataset family in place.

---

### Task 1: Lock the Bridge-Dataset Contract With Failing Tests

**Files:**
- Create: `tests/studies/test_prepare_fly001_256_external_from_n128.py`
- Create: `scripts/studies/prepare_fly001_256_external_from_n128.py`

- [ ] **Step 1: Write the failing tests for the new prep helper**

Add focused tests around a callable helper such as:

```python
def prepare_dataset(
    *,
    train_npz: Path,
    test_npz: Path,
    output_dir: Path,
    target_n: int = 256,
    zoom_factor: float = 2.0,
    nphotons: float = 1e9,
) -> dict[str, str]:
    ...
```

Cover at least:

```python
def test_prepare_dataset_upsamples_object_probe_and_scales_coords(tmp_path):
    result = prepare_dataset(...)
    with np.load(result["train_npz"]) as train:
        assert train["objectGuess"].shape == (256, 256)
        assert train["probeGuess"].shape == (256, 256)
        assert np.allclose(train["xcoords"], source_train["xcoords"] * 2.0)
        assert np.allclose(train["ycoords"], source_train["ycoords"] * 2.0)
```

```python
def test_prepare_dataset_preserves_top_half_train_and_full_test_counts(tmp_path):
    result = prepare_dataset(...)
    with np.load(result["train_npz"]) as train, np.load(result["test_npz"]) as test:
        assert train["xcoords"].shape[0] == source_train["xcoords"].shape[0]
        assert test["xcoords"].shape[0] == source_test["xcoords"].shape[0]
```

```python
def test_prepare_dataset_resimulates_diffraction_at_target_n(tmp_path):
    result = prepare_dataset(...)
    with np.load(result["train_npz"]) as train:
        assert train["diffraction"].shape[1:] == (256, 256)
```

```python
def test_prepare_dataset_rejects_missing_object_or_probe(tmp_path):
    with pytest.raises(KeyError, match="objectGuess|probeGuess"):
        prepare_dataset(...)
```

- [ ] **Step 2: Run collect-only on the new module**

Run: `pytest --collect-only tests/studies/test_prepare_fly001_256_external_from_n128.py -q`

Expected: the new tests collect cleanly.

- [ ] **Step 3: Run the narrow selector to verify failure before implementation**

Run: `pytest tests/studies/test_prepare_fly001_256_external_from_n128.py -v`

Expected: FAIL because the helper script/module does not exist yet.

- [ ] **Step 4: Commit the red tests**

```bash
git add tests/studies/test_prepare_fly001_256_external_from_n128.py
git commit -m "test: lock fly001 n128-to-n256 external bridge contract"
```

---

### Task 2: Implement the N128-to-N256 External Bridge Helper

**Files:**
- Create: `scripts/studies/prepare_fly001_256_external_from_n128.py`
- Modify: `scripts/tools/prepare_data_tool.py` (only if a small reusable interpolation helper extraction materially reduces duplication)
- Test: `tests/studies/test_prepare_fly001_256_external_from_n128.py`

- [ ] **Step 1: Implement a pure helper layer for load -> upsample -> resimulate -> save**

Inside `scripts/studies/prepare_fly001_256_external_from_n128.py`, add small internal helpers:

```python
def _load_raw_npz(path: Path) -> dict[str, np.ndarray]: ...
def _upsample_complex_field(field: np.ndarray, zoom_factor: float) -> np.ndarray: ...
def _scale_coords(data: dict[str, np.ndarray], zoom_factor: float) -> dict[str, np.ndarray]: ...
def _simulate_diffraction(*, xcoords, ycoords, probe_guess, object_guess, nphotons) -> np.ndarray: ...
def _write_raw_npz(path: Path, payload: dict[str, np.ndarray]) -> None: ...
```

Requirements:

- preserve `scan_index` if present; otherwise synthesize zeros
- preserve `xcoords_start` / `ycoords_start` if present, scaled by `zoom_factor`
- resimulate diffraction using explicit coordinates, not random sampling
- write `diffraction` in the same key shape family expected by `external_raw_npz`

- [ ] **Step 2: Use `RawData.from_simulation(...)` under a controlled parameter context**

Implement the resimulation boundary so it:

- sets `params.N = 256`
- sets `params.gridsize = 1`
- sets `params.nphotons = nphotons`
- calls `RawData.from_simulation(scaled_xcoords, scaled_ycoords, probe_256, object_256, scan_index)`
- restores the old global params afterward

Do not route through `simulate_and_save.py`, because that helper generates its own random scan coordinates instead of preserving the fly split layout.

- [ ] **Step 3: Write a manifest and clear output layout**

Persist:

- `output_dir/train_raw.npz`
- `output_dir/test_raw.npz`
- `output_dir/manifest.json`

The manifest should capture:

- source raw NPZ paths
- source SHA-256 hashes
- `source_n = 128`
- `target_n = 256`
- `zoom_factor = 2.0`
- `split_policy = "top_train_full_test"`
- `simulation_mode = "explicit_scaled_coords"`
- `n_train`, `n_test`

- [ ] **Step 4: Add a thin CLI wrapper**

Expose CLI args:

- `--train-npz`
- `--test-npz`
- `--output-dir`
- `--target-n` (default `256`)
- `--zoom-factor` (default `2.0`)
- `--nphotons` (default `1e9`)

The CLI should call `prepare_dataset(...)` and print the manifest path.

- [ ] **Step 5: Run the new test module**

Run: `pytest tests/studies/test_prepare_fly001_256_external_from_n128.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the bridge helper**

```bash
git add scripts/studies/prepare_fly001_256_external_from_n128.py tests/studies/test_prepare_fly001_256_external_from_n128.py
git commit -m "feat: add fly001 n128-to-n256 external bridge prep helper"
```

---

### Task 3: Add the Manual Hybrid-ResNet E40 Runbook

**Files:**
- Create: `scripts/studies/runbooks/grid_lines_external_fly001_n256_upsampled_top_train_full_test_hybrid_resnet_e40.sh`
- Modify: `scripts/studies/README.md`
- Test: `tests/studies/test_prepare_fly001_256_external_from_n128.py` (reuse for prep contract), plus a smoke command against the runbook itself

- [ ] **Step 1: Base the runbook on the existing external-fly N128 flow**

Follow the structure of:

- `scripts/studies/runbooks/grid_lines_external_fly001_n128_top_train_full_test_e40.sh`

But narrow it to one manual run:

- architecture: `hybrid_resnet`
- `N = 256`
- `epochs = 40`
- separate output root, for example:
  - `outputs/grid_lines_external_fly001_n256_upsampled_top_train_full_test_hybrid_resnet_e40`

- [ ] **Step 2: Invoke the new prep helper before dataset grouping**

Inside the runbook:

1. call `prepare_dataset(...)` from the new helper
2. pass the resulting raw train/test NPZs into:

```python
bundles = build_datasets(
    dataset_source="external_raw_npz",
    cfg=cfg,
    required_ns=[256],
    train_data=train_raw,
    test_data=test_raw,
    n_groups=None,
    n_subsample=None,
    neighbor_count=7,
    subsample_seed=3,
)
```

- [ ] **Step 3: Keep the Torch/manual study boundary simple**

Use a single `run_grid_lines_torch(TorchRunnerConfig(...))` call and do not bring back the multi-model comparison wrapper.

The runbook should still persist:

- recon artifact
- `metrics.json`
- `metrics_by_model.json` if convenient
- comparison PNGs via `_finalize_compare_outputs(...)` with:
  - `visual_order=("gt", "pinn_hybrid_resnet")`

- [ ] **Step 4: Choose conservative starting hyperparameters**

Start from the existing external-fly N128 runbook settings unless a value is explicitly tied to `N=128`:

- `batch_size = 8`
- `learning_rate = 2e-4`
- `scheduler = "ReduceLROnPlateau"`
- `plateau_factor = 0.5`
- `plateau_patience = 2`
- `plateau_min_lr = 5e-5`

Document in a comment that this is a first bridge-study baseline, not yet a tuned `N=256` optimum.

- [ ] **Step 5: Run a dry smoke of the runbook prep path**

Run a narrow smoke such as:

```bash
bash scripts/studies/runbooks/grid_lines_external_fly001_n256_upsampled_top_train_full_test_hybrid_resnet_e40.sh
```

If a full run is too expensive for the implementation phase, temporarily add a local-only guard or direct Python snippet that executes just the prep + dataset-build portions and verify they complete cleanly before removing the guard.

- [ ] **Step 6: Commit the runbook**

```bash
git add scripts/studies/runbooks/grid_lines_external_fly001_n256_upsampled_top_train_full_test_hybrid_resnet_e40.sh scripts/studies/README.md
git commit -m "feat: add fly001 external n256 upsampled hybrid_resnet e40 runbook"
```

---

### Task 4: Verify the End-to-End Study Boundary

**Files:**
- Reuse files above

- [ ] **Step 1: Re-run collect-only on any new tests**

Run: `pytest --collect-only tests/studies/test_prepare_fly001_256_external_from_n128.py -q`

Expected: clean collection.

- [ ] **Step 2: Run the targeted pytest module**

Run: `pytest tests/studies/test_prepare_fly001_256_external_from_n128.py -v`

Expected: PASS.

- [ ] **Step 3: Run a dataset-prep CLI smoke**

Run:

```bash
python scripts/studies/prepare_fly001_256_external_from_n128.py \
  --train-npz datasets/fly001_128/fly001_128_top_half_converted.npz \
  --test-npz datasets/fly001_128/fly001_128_full_test_converted.npz \
  --output-dir outputs/fly001_external_n256_bridge_smoke
```

Expected:

- manifest written
- `train_raw.npz` and `test_raw.npz` written
- object/probe shapes are `256x256`
- diffraction shapes are `256x256`

- [ ] **Step 4: Run the manual study**

Run:

```bash
bash scripts/studies/runbooks/grid_lines_external_fly001_n256_upsampled_top_train_full_test_hybrid_resnet_e40.sh
```

Expected:

- grouped datasets created under the runbook output root
- Torch training finishes `40` epochs
- `metrics.json` exists
- recon outputs exist
- comparison PNG exists

- [ ] **Step 5: Record the study result**

Capture in the implementation handoff or follow-up notes:

- final output directory
- amplitude SSIM
- phase SSIM
- dataset manifest path
- whether the `N=256` bridge run beat or trailed the existing `N=128` external baseline

- [ ] **Step 6: Commit the final verified state**

```bash
git add scripts/studies/prepare_fly001_256_external_from_n128.py \
        scripts/studies/runbooks/grid_lines_external_fly001_n256_upsampled_top_train_full_test_hybrid_resnet_e40.sh \
        scripts/studies/README.md \
        tests/studies/test_prepare_fly001_256_external_from_n128.py
git commit -m "feat: add manual fly001 n256 upsampled external e40 study"
```
