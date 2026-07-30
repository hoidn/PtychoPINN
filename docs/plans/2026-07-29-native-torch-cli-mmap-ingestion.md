# Native Torch CLI Mmap Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `python -m ptycho_torch.train` construct exact, run-local,
mmap-backed train and optional validation datasets while preserving the shared
training, DDP, CI-scaling, checkpoint, and bundle paths.

**Architecture:** The native CLI stages each exact NPZ into an isolated role
workspace and constructs an opt-in capped `PtychoDataset`. The shared workflow
passes that native dataset through unchanged, builds either direct mmap loaders
or a rank-local prebuilt DataModule, and gives validation datasets finalized
training statistics. The native mmap adapter is supported only on Linux with
procfs accessible at `/proc/self/fd`; it proves that capability before any
output or staging mutation and has no pathname fallback. Existing programmatic
in-memory inputs and uncapped mmap callers retain their behavior.

**Tech Stack:** Python 3.11, NumPy, PyTorch, TensorDict,
Lightning, dataclasses, pytest.

**Approved design:**
`docs/superpowers/specs/2026-07-29-native-torch-cli-mmap-ingestion-design.md`

---

## File Structure And Responsibilities

- `ptycho_torch/dataloader.py`
  - Keep standalone-NPZ validation aligned with the normative optional
    `objectGuess` contract.
  - Select an exact number of group records before mmap allocation when a
    caller explicitly opts into a cap.
  - Own validated attachment of externally finalized CI statistics.
- `ptycho_torch/workflows/components.py`
  - Preserve `PtychoDataset` at the shared workflow boundary.
  - Build direct mmap loaders safely with optional validation.
  - Thread explicit validation-map paths into DDP/spawn DataModules.
  - Register finalized DataModule CI statistics on every rank.
- `ptycho_torch/train_utils.py`
  - Reopen train and optional validation maps independently per rank.
  - Preserve the existing seeded 90/10 split only when no explicit validation
    map exists.
- `ptycho_torch/train.py`
  - Replace `RawData.from_file` routing without duplicating model or Trainer
    construction.
- `ptycho_torch/cli/mmap_ingestion.py`
  - Own exact-file staging, role-scoped cleanup, and fresh run-local mmap
    construction so `train.py` remains an orchestration entry point.
  - Preflight Linux, procfs descriptor-path identity, descriptor-relative
    operations, and no-follow support before creating or changing output state.
- `tests/torch/test_loader_length_guards.py`
  - Pin optional-object behavior, exact capping, deterministic selection, and
    uncapped compatibility.
- `tests/torch/test_workflows_components.py`
  - Pin mmap pass-through, direct-loader behavior, DDP path threading, and
    rank-safe CI callback configuration.
- `tests/torch/test_absolute_scaling_mmap.py`
  - Prove explicit validation maps use full train/full validation datasets and
    training-owned CI statistics.
- `tests/torch/test_cli_train_torch.py`
  - Pin native-CLI mmap routing and continued bundle persistence.
- `tests/torch/test_native_train_mmap.py`
  - Pin isolated staging, copy fallback, role-scoped rebuild/cleanup, and real
    native-CLI mmap artifacts.
- `tests/torch/test_ci_profile.py`
  - Keep the named CI profile routing test isolated from physical mmap
    construction.
- `docs/workflows/pytorch.md`
  - Document the native CLI mmap substrate, role workspaces, exact group count,
    and optional-object behavior.
- `docs/architecture_torch.md`
  - Distinguish native CLI mmap ingestion from programmatic in-memory inputs.
- `docs/development/TEST_SUITE_INDEX.md`
  - Catalog the new native mmap test module and added selectors.

No configuration schema, mmap tensor schema, model artifact schema, or
`ModelSpec` change is required.

### Repository Constraints

- Do not create a worktree; project `AGENTS.md` forbids it.
- Invoke Python through PATH as `python`.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or
  `ptycho/tf_helper.py`.
- Preserve unrelated dirty-worktree changes.
- Native CLI mmap ingestion requires Linux with procfs mounted and accessible
  at `/proc/self/fd`; fail before output mutation when this capability is
  unavailable and do not add a pathname fallback.
- For a command that becomes long-running, use the tmux skill with the
  `ptycho311` conda environment and track the exact launched PID.

---

### Task 1: Make `objectGuess` Truly Optional In File-Backed Mmap

**Files:**

- Modify: `tests/torch/test_loader_length_guards.py`
- Modify: `ptycho_torch/dataloader.py:243-289`
- Modify: `ptycho_torch/dataloader.py:979-1110`

- [ ] **Step 1: Write the failing unsupervised optional-object test**

Add a fixture with all normative required fields and no `objectGuess`, then
construct a real file-backed dataset:

```python
def test_unsupervised_dataset_accepts_missing_object_guess(tmp_path):
    (tmp_path / "npz").mkdir()
    x, y = _line_scan(20)
    rng = np.random.default_rng(7)
    raw = rng.random((20, N_PIX, N_PIX)).astype(np.float32)
    np.savez(
        tmp_path / "npz" / "no_object.npz",
        xcoords=x,
        ycoords=y,
        diff3d=raw / np.sqrt(
            (raw ** 2).sum(axis=(-2, -1), keepdims=True)
        ),
        probeGuess=np.ones((N_PIX, N_PIX), dtype=np.complex64),
    )
    dataset = _build(
        tmp_path,
        DataConfig(
            N=N_PIX,
            C=1,
            grid_size=(1, 1),
            n_subsample=1,
            x_bounds=(0.0, 1.0),
            y_bounds=(0.0, 1.0),
        ),
        ModelConfig(C_model=1, C_forward=1, object_big=False),
    )

    assert len(dataset) == 20
    assert dataset.data_dict["objectGuess"] == []
```

- [ ] **Step 2: Write the failing supervised zero-correction test**

Create a labeled NPZ without `objectGuess`, leave
`phase_subtraction=True`, and assert:

```python
assert dataset.data_dict["phase_correction"] == [0.0]
np.testing.assert_allclose(
    dataset.mmap_ptycho["label_phase"].cpu().numpy(),
    np.angle(label)[:, None],
    atol=1e-6,
)
assert dataset.data_dict["objectGuess"] == []
```

Keep the existing malformed-present-object test unchanged; an optional key is
still validated when present.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
python -m pytest \
  tests/torch/test_loader_length_guards.py::test_unsupervised_dataset_accepts_missing_object_guess \
  tests/torch/test_loader_length_guards.py::test_supervised_dataset_without_object_guess_uses_zero_phase_correction \
  -q
```

Expected: both tests fail because `_validate_writer_inputs` and
`memory_map_data` index `objectGuess` unconditionally.

- [ ] **Step 4: Make validation conditional on a present object**

Change `_validate_writer_inputs` to this contract:

```python
required_keys = ["probeGuess"]
if model_config.mode == "Supervised":
    required_keys.append("label")

with np.load(npz_file) as data:
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(
            f"{npz_file}: missing required key(s): {', '.join(missing_keys)}."
        )
    probe_shape = data["probeGuess"].shape
    object_shape = (
        data["objectGuess"].shape if "objectGuess" in data else None
    )
    label_shape = (
        data["label"].shape
        if model_config.mode == "Supervised"
        else None
    )

if object_shape is not None and len(object_shape) != 2:
    raise ValueError(
        f"{npz_file}: objectGuess must be 2D; got shape {object_shape}."
    )
```

Retain all existing probe and label shape checks.

- [ ] **Step 5: Make writing and phase correction conditional**

Load the optional object once per file alongside coordinate metadata:

```python
with np.load(npz_file) as npz_data:
    xcoords_full = npz_data["xcoords"]
    ycoords_full = npz_data["ycoords"]
    object_guess = (
        np.array(npz_data["objectGuess"], copy=True)
        if "objectGuess" in npz_data
        else None
    )
```

In supervised mode:

```python
phase_corr_factor = 0.0
if object_guess is not None:
    obj_phase = np.angle(object_guess)
    phase_corr_factor = obj_phase[
        int(obj_phase.shape[0] / 3):int(obj_phase.shape[0] * 2 / 3),
        int(obj_phase.shape[1] / 3):int(obj_phase.shape[1] * 2 / 3),
    ].mean()
self.data_dict["phase_correction"].append(phase_corr_factor)
```

At full-object persistence:

```python
if object_guess is not None:
    area = object_guess.shape[0] * object_guess.shape[1]
    if int(object_guess.sum().real) != area:
        self.data_dict["objectGuess"].append(object_guess)
```

- [ ] **Step 6: Run focused and nearby guard tests**

Run:

```bash
python -m pytest tests/torch/test_loader_length_guards.py \
  -k 'object_guess or missing_object or supervised_dataset' -q
```

Expected: PASS, including the existing invalid-rank and missing-label guards.

- [ ] **Step 7: Commit**

```bash
git add ptycho_torch/dataloader.py tests/torch/test_loader_length_guards.py
git commit -m "fix(torch): accept mmap inputs without object guess"
```

---

### Task 2: Cap Group Records Before Mmap Allocation

**Files:**

- Modify: `tests/torch/test_loader_length_guards.py`
- Modify: `ptycho_torch/dataloader.py:398-468`
- Modify: `ptycho_torch/dataloader.py:573-693`

- [ ] **Step 1: Extend the test builder without changing existing callers**

Allow test-only constructor keywords:

```python
def _build(tmp_path, data_config, model_config, **dataset_kwargs):
    return PtychoDataset(
        ptycho_dir=str(tmp_path / "npz"),
        model_config=model_config,
        data_config=data_config,
        training_config=TrainingConfig(batch_size=8),
        data_dir=str(tmp_path / "mm"),
        remake_map=True,
        **dataset_kwargs,
    )
```

- [ ] **Step 2: Write exact-cap RED tests**

Add:

```python
def test_group_limit_caps_grouped_mmap_before_allocation(tmp_path):
    # 8x8 raster has 36*n_subsample valid quadrant records.
    (tmp_path / "npz").mkdir()
    x, y = _raster(8)
    _write_npz(tmp_path / "npz" / "groups.npz", len(x), x, y)
    dataset = _build(
        tmp_path,
        *_quadrant_configs(3),
        group_limit=17,
        sequential_sampling=True,
    )
    assert len(dataset) == 17
    assert dataset.cum_length == [0, 17]
    for key in ("images", "coords_relative", "coords_center", "nn_indices"):
        assert dataset.mmap_ptycho[key].shape[0] == 17
```

Add an ungrouped sequential selection test whose unique
`center_scan_id` values equal the first requested source records:

```python
assert dataset.mmap_ptycho["center_scan_id"].tolist() == list(range(6))
```

- [ ] **Step 3: Write deterministic-random RED tests**

Build identical one-file datasets into separate mmap directories and assert:

```python
assert selected_a.tolist() == selected_b.tolist()
assert len(set(selected_a.tolist())) == group_limit
```

Cover:

- explicit `subsample_seed=17`;
- `subsample_seed=0` as a valid seed;
- absent seed matching explicit seed `42`; and
- grouped `C=4` records by comparing the complete stored `nn_indices`, not only
  center IDs.

The grouped comparison catches the legacy global `np.random.choice` calls
inside `group_coords`.

- [ ] **Step 4: Write insufficient-count and uncapped RED/compatibility tests**

Request one more record than exists and assert the exception contains both
counts:

```python
(tmp_path / "npz").mkdir()
x, y = _line_scan(20)
_write_npz(tmp_path / "npz" / "limited.npz", 20, x, y)

with pytest.raises(
    ValueError,
    match=r"requested 21.*available 20",
):
    _build(
        tmp_path,
        DataConfig(
            N=N_PIX,
            C=1,
            grid_size=(1, 1),
            n_subsample=1,
            x_bounds=(0.0, 1.0),
            y_bounds=(0.0, 1.0),
        ),
        ModelConfig(C_model=1, C_forward=1, object_big=False),
        group_limit=21,
    )
```

Assert no manifest, state file, or mmap tensor file was produced. Retain
`test_nearest_gs1_length_unchanged` as the uncapped compatibility pin.

- [ ] **Step 5: Run the cap tests and confirm RED**

Run:

```bash
python -m pytest tests/torch/test_loader_length_guards.py \
  -k 'group_limit or capped_grouped_records or nearest_gs1_length_unchanged' \
  -q
```

Expected: new tests fail because the constructor has no cap API.

- [ ] **Step 6: Add the opt-in constructor contract**

Add trailing keywords:

```python
def __init__(
    self,
    ptycho_dir: str,
    model_config: "ModelConfig",
    data_config: "DataConfig",
    training_config: "TrainingConfig" = None,
    data_dir: str = "data/memmap",
    remake_map: bool = False,
    defer_ci_statistics: bool = False,
    group_limit: int | None = None,
    sequential_sampling: bool = False,
):
```

Validate and store:

```python
if group_limit is not None and (
    isinstance(group_limit, bool)
    or not isinstance(group_limit, int)
    or group_limit <= 0
):
    raise ValueError("group_limit must be a positive integer or None.")
self.group_limit = group_limit
self.sequential_sampling = bool(sequential_sampling)
self.group_selection_seed = (
    42
    if data_config.subsample_seed is None
    else int(data_config.subsample_seed)
)
```

Do not infer the cap from `training_config.n_groups`; only the native CLI opts
in.

- [ ] **Step 7: Isolate deterministic candidate generation**

For capped construction only, save NumPy's legacy RNG state, seed candidate
group generation, and restore the state even on failure:

```python
rng_state = None
if self.group_limit is not None:
    rng_state = np.random.get_state()
    np.random.seed(self.group_selection_seed)
try:
    length_result = self.calculate_length()
finally:
    if rng_state is not None:
        np.random.set_state(rng_state)
```

Uncapped callers must retain current RNG behavior.

- [ ] **Step 8: Apply the cap to cached records**

After all candidate records are built in `calculate_length`, call a private
helper with the complete per-file records. Its core behavior is:

```python
candidate_counts = [
    len(grouping[0]) if grouping is not None else len(valid)
    for valid, grouping in zip(
        valid_indices_per_file,
        grouping_per_file,
    )
]
available = sum(candidate_counts)
requested = self.group_limit
if requested is None:
    selected = None
elif requested > available:
    raise ValueError(
        f"Group limit requested {requested} groups, "
        f"but only {available} are available."
    )
elif requested == available:
    selected = np.arange(available, dtype=np.int64)
elif self.sequential_sampling:
    selected = np.arange(requested, dtype=np.int64)
else:
    selected = np.sort(
        np.random.default_rng(self.group_selection_seed).choice(
            available,
            size=requested,
            replace=False,
        )
    )
```

Partition the selected global record positions by file. For grouped records,
slice all four tuple members with the same local indices. For ungrouped
records, slice `valid_indices_per_file`. Leave
`source_indices_per_file` unchanged. Recompute `cum_length` and total length
from the selected per-file counts before returning.

The approved native CLI always stages one NPZ per role. Do not broaden this
task into new guarantees for capped multi-file directories; a cap can otherwise
leave an experiment with zero selected rows.

- [ ] **Step 9: Run cap and full loader-guard tests**

Run:

```bash
python -m pytest tests/torch/test_loader_length_guards.py -q
```

Expected: PASS. Confirm allocation shapes equal the cap and the uncapped length
test still returns `n_valid * n_subsample`.

- [ ] **Step 10: Commit**

```bash
git add ptycho_torch/dataloader.py tests/torch/test_loader_length_guards.py
git commit -m "feat(torch): cap mmap groups before allocation"
```

---

### Task 3: Preserve Native Datasets And Build Safe Direct Mmap Loaders

**Files:**

- Modify: `tests/torch/test_workflows_components.py`
- Modify: `ptycho_torch/dataloader.py:1196-1302`
- Modify: `ptycho_torch/workflows/components.py:728-868`
- Modify: `ptycho_torch/workflows/components.py:1357-1433`

- [ ] **Step 1: Write the `PtychoDataset` pass-through RED test**

Construct a minimal real dataset (or a `PtychoDataset.__new__` sentinel) and
assert:

```python
assert components._ensure_container(dataset, canonical_config) is dataset
```

Expected RED failure: `_ensure_container` rejects the mmap dataset as an
unknown input type.

- [ ] **Step 2: Write direct-loader optional-validation RED tests**

With an execution payload whose strategy is `auto`, assert:

```python
train_loader, val_loader = (
    components._build_dataloaders_from_ptycho_dataset(
        train_ptycho_dataset=train_dataset,
        test_ptycho_dataset=None,
        payload=payload,
    )
)
assert train_loader.dataset is train_dataset
assert val_loader is None
```

With an explicit test dataset, assert the validation loader exists and wraps
that exact dataset.

- [ ] **Step 3: Write a direct CI-statistics authority RED test**

Build train and validation `PtychoDataset.from_np` objects from deliberately
different count-intensity arrays. Capture the validation dataset's provisional
statistics, build direct loaders, then assert:

```python
train_stats = train_dataset.get_ci_statistics()
for name, expected in train_stats.items():
    torch.testing.assert_close(
        test_dataset.get_ci_statistics()[name],
        expected,
    )
assert any(
    not torch.equal(
        provisional[name],
        train_stats[name],
    )
    for name in provisional
)
```

- [ ] **Step 4: Run the direct workflow tests and confirm RED**

Run:

```bash
python -m pytest tests/torch/test_workflows_components.py \
  -k 'mmap_dataset_passes_through or mmap_direct_loader' -q
```

Expected: pass-through raises `TypeError`, no-test construction tries to wrap
`None`, and test statistics remain test-derived.

- [ ] **Step 5: Add a validated CI-statistics attachment API**

Add `PtychoDataset.set_ci_statistics(statistics)`:

```python
def set_ci_statistics(self, statistics):
    if not self.ci_contract_active:
        raise ValueError(
            "CI statistics can only be attached to an active CI dataset."
        )
    required = {"rms_input_scale", "mean_measured_intensity"}
    if set(statistics) != required:
        raise ValueError(
            "CI statistics must contain exactly rms_input_scale and "
            "mean_measured_intensity."
        )
    resolved = {}
    for name in sorted(required):
        value = torch.as_tensor(statistics[name]).detach().reshape(-1).clone()
        if value.numel() != self.n_files:
            raise ValueError(
                f"{name} has {value.numel()} value(s), "
                f"expected {self.n_files}."
            )
        if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
            raise ValueError(f"{name} must be positive and finite.")
        resolved[name] = value
    self.data_dict["ci_statistics"] = resolved
    return self.get_ci_statistics()
```

End `set_ci_statistics_from_indices` with:

```python
return self.set_ci_statistics({
    "rms_input_scale": rms_values,
    "mean_measured_intensity": mean_values,
})
```

- [ ] **Step 6: Preserve mmap input identity**

Make the first `_ensure_container` branch:

```python
if isinstance(data, PtychoDataset):
    logger.debug("Input is already PtychoDataset, returning as-is")
    return data
```

Update its type annotation and docstring without changing the `RawData`,
`RawDataTorch`, or `PtychoDataContainerTorch` branches.

The file-backed constructor already rejects missing supervised labels. Adjust
the later `DATA-SUP-001` preflight so a valid `PtychoDataset` does not try to
iterate `train_loader=None` on the DDP/DataModule route:

```python
if (
    pt_model_config.mode == "Supervised"
    and not isinstance(train_container, PtychoDataset)
):
    # Keep the existing first-batch label check for in-memory containers.
```

Add a regression test proving a supervised mmap DataModule reaches Trainer
construction without iterating a nonexistent direct loader.

- [ ] **Step 7: Make validation construction conditional and train-owned**

In the direct branch:

```python
val_loader = None
if test_ptycho_dataset is not None:
    if train_ptycho_dataset.ci_contract_active:
        test_ptycho_dataset.set_ci_statistics(
            train_ptycho_dataset.get_ci_statistics()
        )
    val_loader = TensorDictDataLoader(
        test_ptycho_dataset,
        batch_size=training_config.batch_size,
        collate_fn=Collate(device=primary_device),
        **loader_kwargs,
    )
return train_loader, val_loader
```

- [ ] **Step 8: Run focused and regression tests**

Run:

```bash
python -m pytest \
  tests/torch/test_workflows_components.py \
  tests/torch/test_absolute_scaling_mmap.py \
  -k 'mmap or ci_statistics' -q
```

Expected: PASS.

- [ ] **Step 9: Preserve the legacy bundle intensity-scale projection**

`train_cdi_model_torch` currently reads the in-memory container attribute
`physics_scaling_constant`. Add the equivalent mmap projection:

```python
if hasattr(train_container, "physics_scaling_constant"):
    scale_tensor = torch.as_tensor(
        train_container.physics_scaling_constant
    )
elif (
    isinstance(train_container, PtychoDataset)
    and "physics_scaling_constant" in train_container.mmap_ptycho.keys()
):
    scale_tensor = torch.as_tensor(
        train_container.mmap_ptycho["physics_scaling_constant"]
    )
else:
    scale_tensor = None

if scale_tensor is not None:
    results["intensity_scale"] = float(
        scale_tensor.reshape(-1)[0].item()
    )
```

Add a workflow test that stubs Lightning training, supplies a legacy-profile
mmap dataset with a known first physics scale, and asserts the results preserve
that value. CI maps intentionally have no legacy physics-scale field and keep
their named CI-statistics persistence path.

- [ ] **Step 10: Commit**

```bash
git add \
  ptycho_torch/dataloader.py \
  ptycho_torch/workflows/components.py \
  tests/torch/test_workflows_components.py
git commit -m "fix(torch): preserve mmap datasets in shared workflow"
```

---

### Task 4: Reopen Explicit Validation Maps Under DDP And Spawn

**Files:**

- Modify: `tests/torch/test_workflows_components.py`
- Modify: `tests/torch/test_absolute_scaling_mmap.py`
- Modify: `ptycho_torch/workflows/components.py:136-178`
- Modify: `ptycho_torch/workflows/components.py:1357-1410`
- Modify: `ptycho_torch/workflows/components.py:1810-1850`
- Modify: `ptycho_torch/train_utils.py:535-614`

- [ ] **Step 1: Write the DDP path-threading RED test**

Extend the existing `test_mmap_ddp_loader_uses_resolved_execution` or add a
neighboring parametrized test for `ddp` and `ddp_spawn`:

```python
result = components._build_dataloaders_from_ptycho_dataset(
    train_ptycho_dataset=SimpleNamespace(
        data_dir_path=Path("/maps/train/memmap")
    ),
    test_ptycho_dataset=SimpleNamespace(
        data_dir_path=Path("/maps/test/memmap")
    ),
    payload=payload,
)
assert result.map_path == Path("/maps/train/memmap")
assert result.validation_map_path == Path("/maps/test/memmap")
```

Also pin `validation_map_path is None` for the no-test fallback.

- [ ] **Step 2: Write a real explicit-map DataModule RED test**

In `tests/torch/test_absolute_scaling_mmap.py`:

1. Build distinct train and validation maps with different measured
   intensities.
2. Instantiate
   `PrebuiltPtychoDataModule(train_map_path, model_config, data_config,
   training_config, validation_map_path=validation_map_path)`.
3. Call `setup("fit")`.
4. Assert:

```python
assert module.train_dataset is module.dataset
assert module.val_dataset is module.validation_dataset
assert len(module.train_dataset) == len(train_source)
assert len(module.val_dataset) == len(validation_source)
assert not isinstance(module.train_dataset, torch.utils.data.Subset)
assert not isinstance(module.val_dataset, torch.utils.data.Subset)
```

Read train and validation batches and assert both expose the statistics derived
from every train-map index, not the validation map's provisional statistics.

- [ ] **Step 3: Strengthen the no-test split compatibility test**

In the existing
`test_prebuilt_data_module_replaces_provisional_ci_statistics_from_train_split`,
assert the exact seed-42 90/10 indices and that both datasets remain
`torch.utils.data.Subset` instances when no validation path is passed.

- [ ] **Step 4: Run the DataModule tests and confirm RED**

Run:

```bash
python -m pytest \
  tests/torch/test_workflows_components.py \
  tests/torch/test_absolute_scaling_mmap.py \
  -k 'mmap_ddp_loader or prebuilt_data_module' -q
```

Expected: explicit validation-path construction is unsupported and the DDP
dispatcher drops the test map.

- [ ] **Step 5: Add the optional validation-map constructor**

Use an additive signature:

```python
from pathlib import Path


def __init__(
    self,
    map_path,
    model_config,
    data_config,
    training_config,
    validation_map_path=None,
):
    super().__init__()
    self.map_path = Path(map_path)
    self.model_config = model_config
    self.data_config = data_config
    self.training_config = training_config
    self.validation_map_path = (
        Path(validation_map_path)
        if validation_map_path is not None
        else None
    )
    self.validation_dataset = None
```

- [ ] **Step 6: Implement explicit full-map setup**

Reopen the train map first with the existing rank/DDP arguments. If an explicit
validation path exists, reopen it independently using the same rank/DDP
arguments and only then assign public state:

```python
train_dataset = PtychoDataset.from_existing_map(
    self.map_path,
    self.model_config,
    self.data_config,
    current_rank=current_rank,
    is_ddp_active=is_ddp_active,
)
if self.validation_map_path is not None:
    validation_dataset = PtychoDataset.from_existing_map(
        self.validation_map_path,
        self.model_config,
        self.data_config,
        current_rank=current_rank,
        is_ddp_active=is_ddp_active,
    )
    self.dataset = train_dataset
    self.validation_dataset = validation_dataset
    self.train_dataset = train_dataset
    self.val_dataset = validation_dataset
    if train_dataset.ci_contract_active:
        self.ci_statistics = train_dataset.set_ci_statistics_from_indices(
            torch.arange(len(train_dataset))
        )
        validation_dataset.set_ci_statistics(self.ci_statistics)
else:
    # Keep current int(0.1 * size), random_split, and seed 42 verbatim.
```

Do not catch an explicit validation-map reopen error and do not substitute an
internal split.

- [ ] **Step 7: Thread the test map from the workflow**

Pass:

```python
validation_map_path=(
    test_ptycho_dataset.data_dir_path
    if test_ptycho_dataset is not None
    else None
),
```

The `_ResolvedPrebuiltPtychoDataModule` forwarding constructor already accepts
this keyword.

- [ ] **Step 8: Register DataModule CI statistics on every rank**

When the resolved scale contract is active CI and `data_product` is a
`PrebuiltPtychoDataModule`, append the existing
`ptycho_torch.lightning_utils.CIStatisticsCallback` to the Trainer callbacks:

```python
if (
    resolved_scale_contract is not None
    and resolved_scale_contract.version == CI_SCALE_CONTRACT
    and isinstance(data_product, PrebuiltPtychoDataModule)
):
    from ptycho_torch.lightning_utils import CIStatisticsCallback
    callbacks.append(CIStatisticsCallback())
```

Add a focused Trainer-mock assertion in
`tests/torch/test_workflows_components.py` that the callback is present for the
CI prebuilt route. This preserves CI checkpoint/bundle statistics under real
DDP and spawn rather than relying on parent-process setup.

- [ ] **Step 9: Run focused and full affected suites**

Run:

```bash
python -m pytest \
  tests/torch/test_workflows_components.py \
  tests/torch/test_absolute_scaling_mmap.py -q
```

Expected: PASS. The no-test DDP path still uses its seeded 90/10 split; the
explicit-test path uses both complete maps.

- [ ] **Step 10: Commit**

```bash
git add \
  ptycho_torch/train_utils.py \
  ptycho_torch/workflows/components.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_absolute_scaling_mmap.py
git commit -m "feat(torch): use explicit mmap validation under DDP"
```

---

### Task 5: Route The Native CLI Through Fresh Exact-File Mmaps

**Files:**

- Create: `ptycho_torch/cli/mmap_ingestion.py`
- Create: `tests/torch/test_native_train_mmap.py`
- Modify: `tests/torch/test_cli_train_torch.py`
- Modify: `tests/torch/test_ci_profile.py`
- Modify: `ptycho_torch/train.py:24-85`
- Modify: `ptycho_torch/train.py:390-535`

- [ ] **Step 1: Write ingestion-adapter RED tests**

Add tests for `build_cli_mmap_dataset`:

- the constructor sees a staging directory containing exactly the selected
  basename even when the source parent contains unrelated NPZs;
- stale content below the exact role workspace is removed before construction;
- the source is hard-linked when possible;
- an `OSError` from `os.link` invokes `shutil.copy2`;
- the staged entry is gone after success; and
- constructor failure removes the partial role workspace and propagates the
  original exception; and
- a source path located inside the role workspace fails before cleanup rather
  than deleting its own input; and
- simulated non-Linux and inaccessible-procfs environments fail before
  `output_dir` exists and report the supported Linux/procfs contract.

Use a fake `PtychoDataset` constructor that inspects the staging directory at
call time and captures:

```python
assert kwargs["data_dir"] == str(
    output_dir / "mmap_workspace" / "train" / "mmap" / "memmap"
)
assert kwargs["remake_map"] is True
assert kwargs["group_limit"] == payload.pt_training_config.n_groups
assert (
    kwargs["sequential_sampling"]
    is payload.tf_training_config.sequential_sampling
)
```

- [ ] **Step 2: Write native CLI routing RED test**

Mock the configuration factory and ingestion adapter, return distinct train/test
dataset sentinels, then assert `run_cdi_example_torch` receives those exact
objects plus the original resolved payload. Put a failing spy on
`RawData.from_file` so any in-memory fallback fails the test.

- [ ] **Step 3: Update the bundle-persistence test contract**

Replace its `RawData.from_file` mock with
`build_cli_mmap_dataset` side effects. Retain every assertion that
`save_torch_bundle` receives both model roles and writes the
`{output_dir}/wts.h5` base path.

Likewise, update
`tests/torch/test_ci_profile.py::test_cli_profile_reaches_training_execution`
to mock `build_cli_mmap_dataset` instead of `RawData.from_file`. Replace the
stale `captured["execution_config"] is not None` assertion with the current
payload contract:

```python
assert captured["resolved_payload"] is not None
assert captured.get("execution_config") is None
```

Keep all profile-override assertions. Do not modify or adopt the unrelated
pre-existing
`tests/torch/test_ci_profile.py::test_workflow_forwards_torch_overrides_to_lightning`
failure as a completion gate for this change.

- [ ] **Step 4: Run CLI unit tests and confirm RED**

Run:

```bash
python -m pytest \
  tests/torch/test_native_train_mmap.py \
  tests/torch/test_cli_train_torch.py \
  tests/torch/test_ci_profile.py \
  -k 'mmap or bundle_persistence or cli_profile_reaches_training_execution' \
  -q
```

Expected: helper import/attribute failures and continued `RawData` routing.

- [ ] **Step 5: Implement exact role-workspace staging**

Create `ptycho_torch/cli/mmap_ingestion.py` with:

Before resolving the staging/output mutation path, run a live descriptor
capability preflight. It must require Linux, accessible procfs descriptor
aliases at `/proc/self/fd`, descriptor-relative operations, no-follow support,
and symlink-resistant tree removal. It must open a harmless read-only directory
descriptor, prove that `/proc/self/fd/<fd>` identifies the same directory, and
close the descriptor on success or failure. Capability failure must name the
Linux/procfs requirement, occur before `output_dir` creation, and must not
select a pathname fallback.

```python
def build_cli_mmap_dataset(
    npz_file: Path,
    *,
    payload,
    output_dir: Path,
    role: str,
):
    import os
    import shutil
    from ptycho_torch.dataloader import PtychoDataset

    if role not in {"train", "test"}:
        raise ValueError(f"Unsupported mmap dataset role: {role!r}")

    npz_file = Path(npz_file)
    role_workspace = Path(output_dir) / "mmap_workspace" / role
    source_path = npz_file.resolve(strict=True)
    resolved_workspace = role_workspace.resolve(strict=False)
    if (
        source_path == resolved_workspace
        or resolved_workspace in source_path.parents
    ):
        raise ValueError(
            f"Source NPZ {source_path} must be outside the mmap "
            f"role workspace {resolved_workspace}."
        )
    if role_workspace.exists():
        shutil.rmtree(role_workspace)
    staged_dir = role_workspace / "staged"
    staged_dir.mkdir(parents=True)
    staged_file = staged_dir / npz_file.name

    try:
        try:
            os.link(source_path, staged_file)
        except OSError:
            shutil.copy2(source_path, staged_file)

        return PtychoDataset(
            ptycho_dir=str(staged_dir),
            model_config=payload.pt_model_config,
            data_config=payload.pt_data_config,
            training_config=payload.pt_training_config,
            data_dir=str(role_workspace / "mmap" / "memmap"),
            remake_map=True,
            group_limit=payload.pt_training_config.n_groups,
            sequential_sampling=(
                payload.tf_training_config.sequential_sampling
            ),
        )
    except Exception:
        shutil.rmtree(role_workspace, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)
```

The role literal and exact constructed target keep cleanup scoped. There is no
`RawData` fallback.

- [ ] **Step 6: Replace native CLI input construction**

After the payload is finalized:

```python
train_data = build_cli_mmap_dataset(
    train_data_file,
    payload=payload,
    output_dir=output_dir,
    role="train",
)
test_data = (
    build_cli_mmap_dataset(
        test_data_file,
        payload=payload,
        output_dir=output_dir,
        role="test",
    )
    if test_data_file is not None
    else None
)
```

Import the adapter once at the CLI ingestion boundary. Keep the helper out of
`train.py`.

Keep the existing `run_cdi_example_torch` call with
`resolved_payload=payload`, profile overrides, and persistence path unchanged.

- [ ] **Step 7: Preserve optional patch-stat artifacts for mmap input**

The post-training diagnostic currently recognizes only in-memory `Y_I`/`Y`.
Add an mmap fallback:

```python
elif hasattr(train_container, "mmap_ptycho"):
    batch_size = payload.pt_training_config.batch_size
    amp_tensor = torch.as_tensor(
        train_container.mmap_ptycho["images"][:batch_size]
    )
```

Keep existing in-memory behavior first in precedence and retain the current
logger/finalization code.

- [ ] **Step 8: Run all CLI unit tests**

Run:

```bash
python -m pytest \
  tests/torch/test_native_train_mmap.py \
  tests/torch/test_cli_train_torch.py \
  -q
python -m pytest \
  tests/torch/test_ci_profile.py::test_cli_profile_reaches_training_execution \
  -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add \
  ptycho_torch/cli/mmap_ingestion.py \
  ptycho_torch/train.py \
  tests/torch/test_native_train_mmap.py \
  tests/torch/test_cli_train_torch.py \
  tests/torch/test_ci_profile.py
git commit -m "feat(torch): train native CLI from run-local mmap"
```

---

### Task 6: Document And Verify The Complete Migration

**Files:**

- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/architecture_torch.md`
- Modify: `docs/development/TEST_SUITE_INDEX.md`

- [ ] **Step 1: Update the workflow guide**

Document:

- native `python -m ptycho_torch.train` stages only the selected train/test
  NPZs;
- persistent maps live under
  `<output_dir>/mmap_workspace/{train,test}/mmap/memmap`;
- role workspaces are rebuilt per invocation;
- `--n_images=N` is an exact pre-allocation group count and fails when
  candidates are insufficient;
- `sequential_sampling` selects the first stable records; otherwise
  `subsample_seed` or `42` selects deterministically without replacement;
- `objectGuess` is optional; and
- the native mmap path requires Linux with accessible procfs descriptor aliases
  at `/proc/self/fd`, fails before output mutation when unavailable, and has no
  pathname fallback; and
- programmatic `RawData` inputs continue to use the in-memory path.

Correct the prerequisite wording that currently lists `objectGuess` as if it
were required.

- [ ] **Step 2: Update the Torch architecture flow**

Show both supported training substrates:

```text
native train CLI
  -> exact-file run-local PtychoDataset mmap
  -> shared run_cdi_example_torch

programmatic RawData / RawDataTorch
  -> PtychoDataContainerTorch in memory
  -> shared run_cdi_example_torch
```

Describe direct no-test behavior (`val_loader=None`) versus DDP no-test
behavior (existing deterministic 90/10 split), and full-map explicit
validation for either strategy.

- [ ] **Step 3: Refresh the generated test catalog**

Run:

```bash
python scripts/tools/generate_test_index.py \
  > docs/development/TEST_SUITE_INDEX.md
```

Review the diff and retain only changes caused by the touched existing suites
and `tests/torch/test_native_train_mmap.py`. If unrelated concurrent test-tree
changes appear, generate to a temporary file instead and use `apply_patch` to
update only the intended catalog rows.

- [ ] **Step 4: Run formatting and focused RED/GREEN regression evidence**

Run:

```bash
git diff --check
python -m pytest \
  tests/torch/test_loader_length_guards.py \
  tests/torch/test_ptycho_dataset_normalized_amplitude.py \
  tests/torch/test_absolute_scaling_mmap.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_native_train_mmap.py \
  tests/torch/test_cli_train_torch.py \
  tests/torch/test_ci_profile.py::test_cli_profile_reaches_training_execution \
  -q
```

Expected: `git diff --check` exits `0`; all selected tests pass.

- [ ] **Step 5: Run a real native-CLI smoke**

Use the tmux skill and activate `ptycho311` because this may exceed one minute.
Run against the tracked integration fixture:

```bash
python -m ptycho_torch.train \
  --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
  --output_dir .artifacts/native_torch_cli_mmap_smoke \
  --n_images 4 \
  --gridsize 1 \
  --batch_size 2 \
  --max_epochs 1 \
  --accelerator cpu \
  --num-workers 0 \
  --logger none \
  --disable-checkpointing \
  --quiet
```

Track the exact launched PID. Consider the smoke complete only when it exits
`0` and these fresh artifacts exist:

```text
.artifacts/native_torch_cli_mmap_smoke/mmap_workspace/train/mmap/memmap/
.artifacts/native_torch_cli_mmap_smoke/mmap_workspace/train/mmap/state_files.npz
.artifacts/native_torch_cli_mmap_smoke/mmap_workspace/train/mmap/mmap_manifest.json
.artifacts/native_torch_cli_mmap_smoke/wts.h5.zip
```

Inspect the TensorDict itself and assert the map length is exactly `4`:

```bash
python - <<'PY'
from tensordict import TensorDict

dataset = TensorDict.load_memmap(
    ".artifacts/native_torch_cli_mmap_smoke/"
    "mmap_workspace/train/mmap/memmap"
)
assert len(dataset) == 4, len(dataset)
print(f"verified mmap rows: {len(dataset)}")
PY
```

- [ ] **Step 6: Review the final diff for scope and dirty-worktree safety**

Run:

```bash
git status --short
git diff --stat
git diff -- \
  ptycho_torch/dataloader.py \
  ptycho_torch/train_utils.py \
  ptycho_torch/workflows/components.py \
  ptycho_torch/cli/mmap_ingestion.py \
  ptycho_torch/train.py \
  tests/torch/test_loader_length_guards.py \
  tests/torch/test_absolute_scaling_mmap.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_cli_train_torch.py \
  tests/torch/test_native_train_mmap.py \
  tests/torch/test_ci_profile.py \
  docs/workflows/pytorch.md \
  docs/architecture_torch.md \
  docs/development/TEST_SUITE_INDEX.md
```

Confirm no unrelated user-owned paths were staged or changed.

- [ ] **Step 7: Commit documentation**

```bash
git add \
  docs/workflows/pytorch.md \
  docs/architecture_torch.md \
  docs/development/TEST_SUITE_INDEX.md
git commit -m "docs(torch): describe native CLI mmap training"
```

- [ ] **Step 8: Run verification-before-completion**

Use `superpowers:verification-before-completion`. Report exact test counts,
the real CLI exit code, and the four required artifact paths. Do not claim the
task complete from older or partial evidence.
