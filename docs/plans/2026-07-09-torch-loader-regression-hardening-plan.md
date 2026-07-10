# Torch Loader Regression Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair all five audited `PtychoDataset` defects while preserving multi-mode probe loading, TF-compatible coordinate signs, and memory-map group-count behavior.

**Architecture:** Keep the repair inside the existing loader boundary. Canonicalize a file's diffraction stack once before coordinate alignment, reject invalid files before allocation, make `from_np` calculate factors from the same tensors as the mmap path, and normalize scalar/batched probe indexing through one code path. Extend the two existing loader regression modules rather than creating new test infrastructure.

**Tech Stack:** Python 3.11, NumPy, PyTorch, TensorDict, pytest.

**Design authority:** `docs/plans/2026-07-09-torch-loader-regression-hardening-design.md`

**Execution constraint:** `CLAUDE.md` forbids worktrees. Execute on the current non-main `fno-stable` branch, stage only task-owned files, and preserve all unrelated worktree changes.

---

## File Map

- Modify `ptycho_torch/dataloader.py`: canonical diffraction alignment,
  fail-fast file validation, `from_np` normalization, optional legacy scale
  copying, and scalar/batched indexing.
- Modify `ptycho_torch/helper.py`: preserve Parseval behavior while making Max
  normalization shape-aware and rejecting non-finite or non-positive factors.
- Modify `tests/torch/test_loader_length_guards.py`: integrated legacy-layout,
  coordinate-array validation, fail-fast file-set, and mmap sign tests.
- Modify `tests/torch/test_multimode_probe_and_from_np.py`: grouped `from_np`
  normalization, override behavior, from-numpy sign, and scalar indexing tests.
- Do not modify `ptycho_torch/patch_generator.py`, model or physics modules, or
  `ptycho_torch/dset_loader_pt_mmap.py`.

## Task 1: Canonical Diffraction Alignment

**Files:**
- Modify: `ptycho_torch/dataloader.py:44-66, 687-806`
- Test: `tests/torch/test_loader_length_guards.py`

- [x] **Step 1: Add failing coordinate-array and integrated legacy-layout tests**

Extend `_write_npz` with keyword-only `pattern_size=N_PIX` and
`legacy_hwn=False`. Generate the diffraction array at `pattern_size`, transpose
it with `np.transpose(diff3d, (1, 2, 0))` when `legacy_hwn=True`, and generate
the probe/object at the same spatial size. Add:

```python
def test_align_coords_rejects_unequal_xy_lengths():
    x, _ = _line_scan(20)
    _, y = _line_scan(19)
    with pytest.raises(ValueError, match="xcoords.*20.*ycoords.*19"):
        _align_coords_to_diffraction(x, y, 20, "fixture.npz")


def test_memory_map_loads_legacy_hwn_layout(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(40)
    _write_npz(
        tmp_path / "npz" / "legacy.npz", 40, x, y, legacy_hwn=True
    )
    data_config = DataConfig(
        N=N_PIX, grid_size=(1, 1), C=1, K=4, n_subsample=1,
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0),
    )
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)
    dataset = _build(tmp_path, data_config, model_config)
    assert len(dataset) == 40
    assert dataset.mmap_ptycho["images"].shape == (40, 1, N_PIX, N_PIX)
    assert int(dataset.mmap_ptycho["nn_indices"].max()) < 40
```

- [x] **Step 2: Run the new tests and verify RED**

Run:

```bash
python -m pytest \
  tests/torch/test_loader_length_guards.py::test_align_coords_rejects_unequal_xy_lengths \
  tests/torch/test_loader_length_guards.py::test_memory_map_loads_legacy_hwn_layout -vv
```

Expected: unequal x/y is not rejected at the helper boundary; the integrated
legacy build fails with an out-of-bounds index after truncating 40 coordinates
to 32.

- [x] **Step 3: Implement one canonical stack/count source**

In `_align_coords_to_diffraction`, validate `len(xcoords) == len(ycoords)`
before comparing either array with `n_diff`:

```python
if len(xcoords) != len(ycoords):
    raise ValueError(
        f"{source}: xcoords has {len(xcoords)} entries but "
        f"ycoords has {len(ycoords)} entries."
    )
```

In `memory_map_data`, move canonical diffraction loading before coordinate
alignment and reuse it later:

```python
diff_stack = torch.from_numpy(_get_diffraction_stack(npz_file)).to(torch.float32)
n_diff = diff_stack.shape[0]
xcoords_full, ycoords_full = _align_coords_to_diffraction(
    xcoords_full, ycoords_full, n_diff, str(npz_file)
)
```

Delete the raw `len(npz_data[key])` count and the later duplicate
`_get_diffraction_stack` load. Keep the existing warning suppression because
`npz_headers` has already emitted the trailing-coordinate warning.

- [x] **Step 4: Run focused GREEN verification**

Run:

```bash
python -m pytest tests/torch/test_loader_length_guards.py tests/torch/test_dataloader.py -q
```

Expected: all tests pass; the existing trailing-coordinate warning remains.

- [x] **Step 5: Self-review and commit**

Confirm the stack is loaded once per write pass and no raw diffraction axis is
used as a pattern count. Stage only the task files and commit:

```bash
git add ptycho_torch/dataloader.py tests/torch/test_loader_length_guards.py
git commit -m "fix(torch): align coordinates to canonical diffraction layout"
```

## Task 2: Fail-Fast File-Set Validation

**Files:**
- Modify: `ptycho_torch/dataloader.py:415-428`
- Test: `tests/torch/test_loader_length_guards.py`

- [x] **Step 1: Add failing public-constructor error tests**

Add:

```python
def test_dataset_rejects_fewer_positions_with_file_context(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(15)
    _write_npz(tmp_path / "npz" / "a.npz", 20, x, y)
    data_config = DataConfig(
        N=N_PIX, grid_size=(1, 1), C=1, K=4, n_subsample=1,
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0),
    )
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)
    with pytest.raises(ValueError, match=r"a\.npz.*15 scan positions.*20 diffraction"):
        _build(tmp_path, data_config, model_config)


def test_dataset_rejects_cross_file_image_shape_mismatch(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    _write_npz(tmp_path / "npz" / "a.npz", 20, x, y)
    _write_npz(
        tmp_path / "npz" / "b.npz", 20, x, y, pattern_size=16
    )
    data_config = DataConfig(
        N=N_PIX, grid_size=(1, 1), C=1, K=4, n_subsample=1,
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0),
    )
    model_config = ModelConfig(C_model=1, C_forward=1, object_big=False)
    with pytest.raises(ValueError, match=r"b\.npz.*Expected \(32, 32\).*\(16, 16\)"):
        _build(tmp_path, data_config, model_config)
```

- [x] **Step 2: Run the new tests and verify RED**

Run:

```bash
python -m pytest \
  tests/torch/test_loader_length_guards.py::test_dataset_rejects_fewer_positions_with_file_context \
  tests/torch/test_loader_length_guards.py::test_dataset_rejects_cross_file_image_shape_mismatch -vv
```

Expected: the one-file case raises the generic "Could not determine image
shape" error; the two-file case prints "Skipping file" and later fails with
`IndexError: list index out of range`.

- [x] **Step 3: Remove partial file skipping**

In `calculate_length`, call `npz_headers(npz_file)` without a catch-and-continue
wrapper. Replace the image-shape warning/continue with:

```python
raise ValueError(
    f"{npz_file}: image shape mismatch. Expected {first_im_shape}, "
    f"got {tensor_shape[1:]}."
)
```

Do not filter or rewrite `file_list`; the accepted file set remains exactly the
requested directory set when validation succeeds.

- [x] **Step 4: Run focused GREEN verification**

Run:

```bash
python -m pytest tests/torch/test_loader_length_guards.py -q
```

Expected: all tests pass with direct, filename-bearing `ValueError`s.

- [x] **Step 5: Self-review and commit**

Verify there is no `continue` path that can shorten per-file metadata arrays.
Commit:

```bash
git add ptycho_torch/dataloader.py tests/torch/test_loader_length_guards.py
git commit -m "fix(torch): reject invalid mmap input files before allocation"
```

## Task 3: Group-Aware `from_np` Normalization

**Files:**
- Modify: `ptycho_torch/dataloader.py:813-850, 1015-1046, 1092-1097, 1143-1148`
- Test: `tests/torch/test_multimode_probe_and_from_np.py`

- [x] **Step 1: Add grouped fixture helpers and failing parity tests**

Add `_make_grouped_arrays(n_modes=None)` using an 8x8 raster, the existing
normalized-amplitude construction, and the existing probe/object construction.
Return `(diff3d, xcoords, ycoords, probe, obj)`. Add `_group_configs()`:

```python
def _group_configs(normalize="Group"):
    data_config = DataConfig(
        N=N_PIX, grid_size=(2, 2), C=4, K=6, n_subsample=1,
        neighbor_function="4_quadrant", scan_pattern="Isotropic",
        x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), normalize=normalize,
    )
    model_config = ModelConfig(
        C_model=4, C_forward=4, object_big=True
    )
    return data_config, model_config, TrainingConfig(batch_size=8)
```

Add these executable tests:

```python
def test_from_np_group_normalization_matches_file_dataset(tmp_path):
    payload = _make_grouped_arrays()
    data_config, model_config, training_config = _group_configs()
    np.random.seed(123)
    file_ds = _build_file_dataset(
        tmp_path, {"grouped.npz": payload}, data_config, model_config,
        training_config,
    )
    positions = np.stack([payload[2], payload[1]], axis=1)
    np.random.seed(123)
    mem_ds = PtychoDataset.from_np(
        payload[0], payload[3], positions, model_config, data_config
    )
    for key in (
        "images", "nn_indices", "rms_scaling_constant",
        "physics_scaling_constant",
    ):
        torch.testing.assert_close(
            torch.as_tensor(mem_ds.mmap_ptycho[key]),
            torch.as_tensor(file_ds.mmap_ptycho[key][:]),
            rtol=1e-5, atol=1e-6,
        )


def test_from_np_group_normalization_omits_undefined_legacy_scalar():
    payload = _make_grouped_arrays()
    data_config, model_config, _ = _group_configs()
    positions = np.stack([payload[2], payload[1]], axis=1)
    np.random.seed(123)
    dataset = PtychoDataset.from_np(
        payload[0], payload[3], positions, model_config, data_config
    )
    assert "scaling_constant" not in dataset.data_dict
    assert dataset.mmap_ptycho["rms_scaling_constant"].shape == (len(dataset), 1, 1, 1)
    _, probes, scaling = dataset[torch.arange(4)]
    assert probes.shape[:2] == (4, 4)
    assert scaling.shape == (4, 1, 1, 1)
    subset = dataset.get_experiment_dataset(0)
    assert "scaling_constant" not in subset.data_dict


def test_from_np_group_rms_override_preserves_group_physics_factors():
    payload = _make_grouped_arrays()
    data_config, model_config, _ = _group_configs()
    positions = np.stack([payload[2], payload[1]], axis=1)
    np.random.seed(123)
    baseline = PtychoDataset.from_np(
        payload[0], payload[3], positions, model_config, data_config
    )
    np.random.seed(123)
    overridden = PtychoDataset.from_np(
        payload[0], payload[3], positions, model_config, data_config,
        scaling_constant=7.5,
    )
    torch.testing.assert_close(
        overridden.mmap_ptycho["rms_scaling_constant"],
        torch.full_like(overridden.mmap_ptycho["rms_scaling_constant"], 7.5),
    )
    torch.testing.assert_close(
        overridden.mmap_ptycho["physics_scaling_constant"],
        baseline.mmap_ptycho["physics_scaling_constant"],
    )
    torch.testing.assert_close(
        overridden.data_dict["scaling_constant"], torch.tensor([7.5])
    )


def test_c1_group_uses_effective_batch_factors_in_both_paths(tmp_path):
    payload = _make_arrays()
    data_config, model_config, training_config = _configs(
        normalize="Group", x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0)
    )
    np.random.seed(123)
    file_ds = _build_file_dataset(
        tmp_path, {"c1.npz": payload}, data_config, model_config,
        training_config,
    )
    positions = np.stack([payload[2], payload[1]], axis=1)
    np.random.seed(123)
    mem_ds = PtychoDataset.from_np(
        payload[0], payload[3], positions, model_config, data_config
    )
    torch.testing.assert_close(
        torch.as_tensor(mem_ds.mmap_ptycho["rms_scaling_constant"]),
        torch.as_tensor(file_ds.mmap_ptycho["rms_scaling_constant"][:]),
    )
    torch.testing.assert_close(
        torch.as_tensor(mem_ds.mmap_ptycho["physics_scaling_constant"]),
        torch.as_tensor(file_ds.mmap_ptycho["physics_scaling_constant"][:]),
    )
    expected_scalar = mem_ds.mmap_ptycho[
        "rms_scaling_constant"
    ][0].reshape(1)
    torch.testing.assert_close(
        mem_ds.data_dict["scaling_constant"], expected_scalar
    )
    torch.testing.assert_close(
        file_ds.data_dict["scaling_constant"], expected_scalar
    )


def test_c1_none_uses_unit_factors_and_legacy_scalar_in_both_paths(tmp_path):
    payload = _make_arrays()
    data_config, model_config, training_config = _configs(normalize="None")
    file_ds = _build_file_dataset(
        tmp_path, {"none.npz": payload}, data_config, model_config,
        training_config,
    )
    positions = np.stack([payload[2], payload[1]], axis=1)
    mem_ds = PtychoDataset.from_np(
        payload[0], payload[3], positions, model_config, data_config
    )
    for dataset in (file_ds, mem_ds):
        assert torch.equal(
            dataset.mmap_ptycho["rms_scaling_constant"],
            torch.ones_like(dataset.mmap_ptycho["rms_scaling_constant"]),
        )
        assert torch.equal(
            dataset.mmap_ptycho["physics_scaling_constant"],
            torch.ones_like(dataset.mmap_ptycho["physics_scaling_constant"]),
        )
        torch.testing.assert_close(
            dataset.data_dict["scaling_constant"], torch.tensor([1.0])
        )
```

Seed NumPy before each file/from-numpy build so randomized neighbor choices and
stored `nn_indices` are directly comparable.

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest \
  tests/torch/test_multimode_probe_and_from_np.py::test_from_np_group_normalization_matches_file_dataset \
  tests/torch/test_multimode_probe_and_from_np.py::test_from_np_group_normalization_omits_undefined_legacy_scalar \
  tests/torch/test_multimode_probe_and_from_np.py::test_from_np_group_rms_override_preserves_group_physics_factors \
  tests/torch/test_multimode_probe_and_from_np.py::test_c1_group_uses_effective_batch_factors_in_both_paths \
  tests/torch/test_multimode_probe_and_from_np.py::test_c1_none_uses_unit_factors_and_legacy_scalar_in_both_paths -vv
```

Expected: Group mode fails with `UnboundLocalError`; the key behavior and
override semantics cannot be satisfied.

- [x] **Step 3: Build images before calculating factors**

Move `images`/`nn_indices_tensor` construction before normalization. Import
`replace` from `dataclasses`. Implement the same branch rule in both
`memory_map_data` and `from_np`:

```python
use_batch_factors = (
    data_config.normalize == "Batch"
    or (
        data_config.normalize == "Group"
        and data_config.C == 1
        and model_config.mode != "Supervised"
    )
)

factor_config = (
    data_config
    if not use_batch_factors or data_config.normalize == "Batch"
    else replace(data_config, normalize="Batch")
)

if data_config.normalize == "None":
    rms_factors = torch.ones(N_groups, 1, 1, 1)
    physics_factors = torch.ones_like(rms_factors)
elif use_batch_factors:
    rms_scalar = hh.get_rms_scaling_factor(diff_tensor, factor_config)
    physics_scalar = hh.get_physics_scaling_factor(diff_tensor, factor_config)
    rms_factors = rms_scalar.expand(N_groups, 1, 1, 1).clone()
    physics_factors = physics_scalar.expand(N_groups, 1, 1, 1).clone()
else:
    rms_factors = hh.get_rms_scaling_factor(images, data_config)
    physics_factors = hh.get_physics_scaling_factor(images, data_config)
```

Apply `scaling_constant` only after mode-derived factors are calculated:

```python
if scaling_constant is not None:
    rms_factors = torch.full_like(rms_factors, float(scaling_constant))
```

Populate the legacy key only for Batch, `'None'`, or an explicit override.
Here "Batch" includes `use_batch_factors=True`: a configured C=1 Group dataset
stores the effective-Batch RMS scalar. Group with C>1 and no override omits the
key. `'None'` stores `torch.tensor([1.0])`.
Use `rms_factors` and `physics_factors` directly in the TensorDict instead of
blindly expanding a scalar.

Update `get_experiment_dataset` to copy `scaling_constant` only when present.
Remove the dead unconditional `scaling_constant` lookup from `__getitem__` in
this task so a Group dataset remains indexable at the Task 3 commit boundary.
Remove no other state keys.

For the mmap path, use its local names and preserve chunked C>1 Group writes:

```python
use_batch_factors = (
    self.data_config.normalize == "Batch"
    or (
        self.data_config.normalize == "Group"
        and self.data_config.C == 1
        and self.model_config.mode != "Supervised"
    )
)
factor_config = (
    self.data_config
    if self.data_config.normalize == "Batch"
    else replace(self.data_config, normalize="Batch")
)

if self.data_config.normalize == "None":
    unit_factors = torch.ones(B, 1, 1, 1)
    # write both mmap factor tensors and the legacy scalar 1.0
elif use_batch_factors:
    norm_rms_factor = hh.get_rms_scaling_factor(diff_stack, factor_config)
    norm_physics_factor = hh.get_physics_scaling_factor(diff_stack, factor_config)
    # expand both over B and record norm_rms_factor in the legacy scalar slot
else:
    # retain the existing chunked Group calculation on diff_stack[nn_indices]
```

Do not use `N_groups`, `images`, or `diff_tensor` in the mmap branch.

- [x] **Step 4: Run focused GREEN verification**

Run:

```bash
python -m pytest \
  tests/torch/test_multimode_probe_and_from_np.py \
  tests/torch/test_dataloader_batch_scale_semantics.py \
  tests/torch/test_probe_normalization_parity.py \
  tests/torch/test_parity_probe_normalization.py -q
```

Expected: all tests pass, including existing multi-mode and C=1 parity tests.

- [x] **Step 5: Self-review and commit**

Verify the override does not alter physics factors and Group mode does not
invent a legacy scalar. Commit:

```bash
git add ptycho_torch/dataloader.py tests/torch/test_multimode_probe_and_from_np.py
git commit -m "fix(torch): normalize grouped from-numpy datasets per sample"
```

## Task 4: Loader Coordinate-Sign Regression Pins

**Files:**
- Test: `tests/torch/test_loader_length_guards.py`
- Test: `tests/torch/test_multimode_probe_and_from_np.py`
- Temporary mutation only: `ptycho_torch/patch_generator.py:386`

- [x] **Step 1: Add mmap and `from_np` sign assertions**

In `test_loader_length_guards.py`, add:

```python
def test_mmap_coords_relative_uses_tf_sign(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "sign.npz", len(x), x, y)
    np.random.seed(123)
    dataset = _build(tmp_path, *_quadrant_configs(1))
    coords_global = torch.as_tensor(dataset.mmap_ptycho["coords_global"])
    coords_center = torch.as_tensor(dataset.mmap_ptycho["coords_center"])
    coords_relative = torch.as_tensor(dataset.mmap_ptycho["coords_relative"])
    expected = -(coords_global - coords_center)
    torch.testing.assert_close(coords_relative, expected, rtol=0, atol=1e-6)
    assert float(coords_relative.abs().max()) > 0
```

In `test_multimode_probe_and_from_np.py`, add independently:

```python
def test_from_np_coords_relative_uses_tf_sign():
    payload = _make_grouped_arrays()
    data_config, model_config, _ = _group_configs()
    positions = np.stack([payload[2], payload[1]], axis=1)
    np.random.seed(123)
    dataset = PtychoDataset.from_np(
        payload[0], payload[3], positions, model_config, data_config
    )
    coords_global = torch.as_tensor(dataset.mmap_ptycho["coords_global"])
    coords_center = torch.as_tensor(dataset.mmap_ptycho["coords_center"])
    coords_relative = torch.as_tensor(dataset.mmap_ptycho["coords_relative"])
    expected = -(coords_global - coords_center)
    torch.testing.assert_close(coords_relative, expected, rtol=0, atol=1e-6)
    assert float(coords_relative.abs().max()) > 0
```

- [x] **Step 2: Prove the tests catch the historical regression**

Because the current code is already correct, use mutation verification:

1. Record `git diff --binary -- ptycho_torch/patch_generator.py` in a temporary
   file outside the repo, then run both new tests and confirm they pass on
   `local_offset_sign=-1`.
2. Temporarily change only the default in `get_relative_coords` from `-1` to
   `+1` using `apply_patch`.
3. Run both tests and confirm they fail with the sign assertion.
4. Restore `-1` using `apply_patch` and rerun both tests to green.

Do not commit the temporary mutation. Confirm the final binary diff is
byte-identical to the pre-mutation snapshot.

Run the new tests by exact selectors:

```bash
python -m pytest \
  tests/torch/test_loader_length_guards.py::test_mmap_coords_relative_uses_tf_sign \
  tests/torch/test_multimode_probe_and_from_np.py::test_from_np_coords_relative_uses_tf_sign -vv
```

- [x] **Step 3: Run adjacent coordinate/reassembly verification**

Run:

```bash
python -m pytest \
  tests/torch/test_loader_length_guards.py \
  tests/torch/test_multimode_probe_and_from_np.py \
  tests/torch/test_coords_relative_contract.py \
  tests/torch/test_rawdata_coords_relative_sign.py \
  tests/torch/test_reassembly_sign_parity.py \
  tests/torch/test_object_big_generator_contract.py -q
```

Expected: all tests pass.

- [x] **Step 4: Self-review and commit**

Confirm each test exercises nonzero offsets and does not merely compare two
paths using the same helper. Commit tests only:

```bash
git add tests/torch/test_loader_length_guards.py tests/torch/test_multimode_probe_and_from_np.py
git commit -m "test(torch): pin mmap and from-numpy coordinate signs"
```

## Task 5: Scalar and Batched `__getitem__` Contracts

**Files:**
- Modify: `ptycho_torch/dataloader.py:1075-1099`
- Test: `tests/torch/test_multimode_probe_and_from_np.py`

- [x] **Step 1: Add failing scalar single/multi-mode tests**

Add one C=1 single-mode case and one grouped C=4 multi-mode case:

```python
def test_from_np_scalar_getitem_single_mode_shapes():
    data_config, model_config, _ = _configs()
    model_config = ModelConfig(
        C_model=1, C_forward=1, object_big=False
    )
    diff3d, xcoords, ycoords, probe, _ = _make_arrays(n_images=12)
    positions = np.stack([ycoords, xcoords], axis=1)
    dataset = PtychoDataset.from_np(
        diff3d, probe, positions, model_config, data_config
    )
    td, probe, scale = dataset[0]
    assert td["images"].shape == (1, N_PIX, N_PIX)
    assert probe.shape == (1, 1, N_PIX, N_PIX)
    assert scale.shape == (1, 1, 1)


def test_from_np_scalar_getitem_multimode_expands_channels():
    payload = _make_grouped_arrays(n_modes=3)
    data_config, model_config, _ = _group_configs()
    positions = np.stack([payload[2], payload[1]], axis=1)
    np.random.seed(123)
    dataset = PtychoDataset.from_np(
        payload[0], payload[3], positions, model_config, data_config
    )
    td, probe, scale = dataset[0]
    assert td["images"].shape == (4, N_PIX, N_PIX)
    assert probe.shape == (4, 3, N_PIX, N_PIX)
    assert scale.shape == (1, 1, 1)
```

Retain the existing sliced/batched `(B,C,P,N,N)` assertion.

- [x] **Step 2: Run scalar and existing batched selectors and verify RED**

Run:

```bash
python -m pytest \
  tests/torch/test_multimode_probe_and_from_np.py::test_from_np_scalar_getitem_single_mode_shapes \
  tests/torch/test_multimode_probe_and_from_np.py::test_from_np_scalar_getitem_multimode_expands_channels \
  tests/torch/test_multimode_probe_and_from_np.py::test_from_np_multimode_probe_and_getitem -vv
```

Expected: scalar indexing fails at five-dimensional `expand` because a
zero-dimensional tensor was mistaken for a batch index.

- [x] **Step 3: Normalize indexing through a batched internal representation**

Convert the selected experiment ID to a tensor, detect scalar rank, reshape to
one dimension, and build batch-shaped outputs:

```python
exp_idx = torch.as_tensor(self.mmap_ptycho["experiment_id"][idx])
is_scalar = exp_idx.ndim == 0
exp_idx_batch = exp_idx.reshape(-1)
get_idx = exp_idx_batch if self.n_files > 1 else torch.zeros_like(exp_idx_batch)
probes = self.data_dict["probes"][get_idx].unsqueeze(1).expand(
    -1, channels, -1, -1, -1
)
scales = self.data_dict["probe_scaling"][get_idx].view(-1, 1, 1, 1)
if is_scalar:
    probes = probes[0]
    scales = scales[0]
```

Return TensorDict indexing unchanged. The dead `scaling_const` lookup was
removed in Task 3; `batch[2]` remains probe scaling.

- [x] **Step 4: Run focused GREEN verification**

Run:

```bash
python -m pytest \
  tests/torch/test_multimode_probe_and_from_np.py \
  tests/torch/test_dataloader_batch_scale_semantics.py -q
```

Expected: scalar and batched shapes pass, and `batch[2]` remains probe scaling.

- [x] **Step 5: Self-review and commit**

Confirm list, slice, tensor, and integer indices preserve their documented
rank. Commit:

```bash
git add ptycho_torch/dataloader.py tests/torch/test_multimode_probe_and_from_np.py
git commit -m "fix(torch): support scalar PtychoDataset indexing"
```

## Task 6: Integrated Verification and Final Review

**Files:**
- Review all changes since `d01b875f`.
- No production edits unless verification or review identifies a defect.

- [x] **Step 1: Run the focused loader/probe/coordinate suite serially**

```bash
python -m pytest -q \
  tests/torch/test_loader_length_guards.py \
  tests/torch/test_multimode_probe_and_from_np.py \
  tests/torch/test_dataloader.py \
  tests/torch/test_dataloader_batch_scale_semantics.py \
  tests/torch/test_ptycho_dataset_normalized_amplitude.py \
  tests/torch/test_probe_normalization_parity.py \
  tests/torch/test_parity_probe_normalization.py \
  tests/torch/test_probe_mask_soft_semantics.py \
  tests/torch/test_rect_probe_scale_double_div.py \
  tests/torch/test_coords_relative_contract.py \
  tests/torch/test_rawdata_coords_relative_sign.py \
  tests/torch/test_reassembly_sign_parity.py \
  tests/torch/test_object_big_generator_contract.py \
  tests/torch/test_training_forward_probe_weighted_reassembly.py \
  tests/torch/test_data_pipeline.py
```

Expected: zero failures.

- [x] **Step 2: Run the full non-slow Torch suite serially**

```bash
python -m pytest tests/torch -m "not slow" -q
```

Expected baseline: all loader-related tests pass. The known unrelated
`test_fixture_pytorch_integration.py::TestFixtureContract::test_metadata_content_valid`
checksum mismatch may remain; record its exact result rather than claiming a
fully green suite if it does.

- [x] **Step 3: Inspect scope and repository state**

```bash
git diff --check d01b875f..HEAD
git diff --stat d01b875f..HEAD
git status --short --branch
```

Confirm only the design/plan, `ptycho_torch/dataloader.py`, and the two intended
test files changed in the planned initiative. The final-review amendment also
changes `ptycho_torch/helper.py` for Max scaling. Preserve unrelated
pre-existing status.

- [x] **Step 4: Dispatch final implementation review**

Provide the reviewer the approved design, this plan, base `d01b875f`, and final
HEAD. Resolve every Critical or Important issue and rerun affected tests before
completion.

### Final-Review Amendments

The final review found additional cases under the same loader integrity
contract. They were resolved with RED/GREEN regression coverage before final
approval:

- coordinate-count and square-plane disambiguation for legacy `(H, W, N)`
  stacks, including trailing-coordinate collisions;
- exact `diffraction` then `diff3d` NPZ-key priority in both header and load
  paths, ignoring prefixed decoy members;
- Batch and Group `data_scaling='Max'` parity between mmap and `from_np`, with
  explicit rejection of non-finite or non-positive denominators; and
- independent closed-form Parseval Group assertions, so path parity does not
  share the implementation under test as its only oracle.

The accepted residual is rectangular-layout ambiguity with trailing
coordinates. Integrated loader inputs remain square by `DataConfig.N` and
preflight contract; genuinely ambiguous rectangular arrays retain the
historical heuristic.
