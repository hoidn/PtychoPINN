# Grid-Lines Torch Runner Shared Position Crop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `grid_lines_torch_runner` use the shared center-cropping utility for position reassembly and change the default from `crop_border=0` to a non-zero, reasonable value.

**Architecture:** Keep the existing hybrid position-reassembly backend (`shift_sum`/`batched`) and insert a shared-crop preprocessing step in the runner before reassembly. Add runner-level crop config with an auto default that resolves to `min(patch_h, patch_w)//4` (clamped to valid bounds) so N128/N256 runs crop by default unless explicitly overridden. Preserve backward compatibility through explicit override flags (including `0`), and record resolved crop behavior in runtime contract artifacts for auditability.

**Tech Stack:** Python, NumPy, pytest, existing `ptycho.image.cropping` utility

---

### Task 1: Lock Behavior Contract With Failing Tests

**Files:**
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `tests/torch/test_grid_lines_position_reassembly_strategy.py`

**Step 1: Write failing tests for crop default and override semantics**

```python
def test_position_crop_auto_default_resolves_to_quarter_patch():
    # N=128 => crop border 32
    ...

def test_position_crop_explicit_zero_disables_crop():
    ...

def test_position_crop_is_clamped_to_nonempty_patch():
    ...
```

**Step 2: Write failing test proving shared crop utility is used**

```python
def test_position_reassembly_uses_shared_center_crop_utility(monkeypatch):
    called = {}
    def fake_crop(arr, border):
        called["border"] = border
        return arr
    monkeypatch.setattr(
        "scripts.studies.grid_lines_torch_runner.center_crop_spatial_by_border",
        fake_crop,
    )
    _reassemble_with_coords_offsets(...)
    assert called["border"] == 32
```

**Step 3: Run tests to verify RED state**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "position_crop or shared_center_crop" -v`
Expected: FAIL because crop fields/resolver/shared utility call do not yet exist in runner path.

**Step 4: Commit test-only RED scaffolding**

```bash
git add tests/torch/test_grid_lines_torch_runner.py tests/torch/test_grid_lines_position_reassembly_strategy.py
git commit -m "test(torch): define position reassembly shared-crop contract"
```

### Task 2: Add Runner Config + CLI For Position Crop Control

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Add config fields in `TorchRunnerConfig`**

```python
position_crop_border: int | None = None
position_crop_mode: str = "auto"  # auto | explicit
```

Note: Keep `None` as "auto" resolver trigger; explicit `0` is valid and means no crop.

**Step 2: Add CLI flags and wire into config object**

```bash
--position-crop-border <int>
```

Parser contract:
- missing flag => auto default
- `--position-crop-border 0` => disable crop
- positive integer => explicit crop
- negative => reject with actionable error

**Step 3: Add resolver helper and validate inputs**

Add a helper such as:

```python
def _resolve_position_crop_border(patch_h: int, patch_w: int, configured: int | None) -> int:
    # auto => min(h,w)//4, clamped
    # explicit => clamp to nonempty patch
```

**Step 4: Run targeted tests**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "position_crop" -v`
Expected: PASS for parser/config/resolver semantics.

**Step 5: Commit config/CLI plumbing**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "feat(torch-runner): add position reassembly crop config with non-zero auto default"
```

### Task 3: Replace Ad Hoc Crop Handling With Shared Utility In Reassembly Path

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Apply shared utility to normalized patches before reassembly**

Implementation direction in `_normalize_position_inputs(...)`:

```python
from ptycho.image.cropping import center_crop_spatial_by_border

patches = ...  # (B, H, W, 1)
patches_2d = np.squeeze(patches, axis=-1)           # (B, H, W)
crop_border = _resolve_position_crop_border(...)
patches_2d = center_crop_spatial_by_border(patches_2d, crop_border)
patches = patches_2d[..., None].astype(np.complex64)
```

**Step 2: Ensure `M` follows effective cropped patch size**

After crop, force reassembly `M` to effective patch size (do not assume `cfg.N` still matches):

```python
effective_m = int(patches.shape[1])
```

Use `effective_m` consistently for `shift_sum` and `batched` calls.

**Step 3: Emit runtime contract evidence**

Extend runtime contract output map with:
- `position_crop_border_configured`
- `position_crop_border_resolved`
- `position_patch_shape_pre_crop`
- `position_patch_shape_post_crop`

**Step 4: Run focused reassembly tests**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "position_reassembly_mode_uses_coords_offsets or shared_center_crop or backend_shift_sum" -v`
Expected: PASS.

**Step 5: Commit shared utility integration**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "refactor(torch-runner): use shared center-crop utility in position reassembly"
```

### Task 4: Verify Backend Strategy Behavior Is Unchanged Except Cropping Contract

**Files:**
- Modify: `tests/torch/test_grid_lines_position_reassembly_strategy.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Add regression tests for backend routing with crop enabled by default**

```python
def test_large_external_n128_auto_still_prefers_shift_sum_with_crop_default(...):
    ...

def test_explicit_batched_backend_still_uses_batched_with_crop_default(...):
    ...
```

**Step 2: Add test for override parity lane (`--position-crop-border 0`)**

```python
def test_position_reassembly_crop_zero_preserves_legacy_m_dimension(...):
    ...
```

**Step 3: Run test subset**

Run: `pytest tests/torch/test_grid_lines_position_reassembly_strategy.py tests/torch/test_grid_lines_torch_runner.py -k "position and (backend or crop)" -v`
Expected: PASS.

**Step 4: Commit strategy regressions**

```bash
git add tests/torch/test_grid_lines_position_reassembly_strategy.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "test(torch-runner): guard backend routing under new position crop defaults"
```

### Task 5: Documentation + End-to-End Verification

**Files:**
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/findings.md` (only if a new invariant needs a formal Finding ID)

**Step 1: Document runner crop behavior and override path**

Add a short section covering:
- auto default (`min(patch_h, patch_w)//4`)
- explicit disable (`--position-crop-border 0`)
- why shared utility is used (cross-pipeline consistency)

**Step 2: Run full verification set**

Run:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py -v
pytest tests/torch/test_grid_lines_position_reassembly_strategy.py -v
pytest tests/studies/test_nersc_ptychovit_orchestration.py -k "position_reassembly_backend" -v
```
Expected: PASS.

**Step 3: Quick static audit for ad hoc crop regressions in runner scope**

Run:
```bash
rg -n "center_crop_spatial_by_border|position_crop_border|_resolve_position_crop_border|reassemble_with_coords_offsets" scripts/studies/grid_lines_torch_runner.py tests/torch
```
Expected: runner crop path references shared utility and tests cover defaults + override.

**Step 4: Commit docs + verification artifacts**

```bash
git add docs/workflows/pytorch.md docs/findings.md scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py tests/torch/test_grid_lines_position_reassembly_strategy.py
git commit -m "docs(torch): document shared position crop defaults and override lane"
```

---

Plan complete and saved to `docs/plans/2026-03-04-grid-lines-torch-runner-shared-position-crop.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
