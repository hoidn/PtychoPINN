# NERSC Downsample Policy Flip (Real-Space Crop + Diffraction Bin) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change the NERSC `256 -> 128` preparation path so real-space arrays (`objectGuess`, `probeGuess`) are center-cropped while diffraction is binned, then re-verify the NERSC smoke workflow/study outputs.

**Architecture:** Keep the change localized to the shared helper `scripts/studies/prepare_nersc_hybrid_dataset.py::_downsample_external_payload`, which is reused by both cameraman prep and scan807 downsample conversion. Replace current transform policy (`diffraction crop + real-space bin + coord scale`) with the requested inverse policy (`diffraction bin + real-space crop`) and adjust coordinate handling to match cropped real-space semantics (no scale-down). Validate with targeted unit tests first, then rerun the orchestration smoke path and a study rerun command.

**Tech Stack:** Python 3.11, NumPy, h5py, pytest, existing `scripts/studies/*` orchestration helpers.

---

## Brainstorming Resolution (Disambiguated Decisions)

1. **Scope of policy flip:** Apply the new policy in the shared `_downsample_external_payload` helper so both cameraman and scan807 N=128 prep paths change together.
2. **Diffraction binning math:** Use deterministic block-average binning over diffraction amplitudes in `[scan, H, W]`.
3. **Real-space resize rule:** Replace real-space binning with center-crop to reduced dimensions (`H//factor`, `W//factor`) so 256→128 remains exact and odd-shape behavior remains defined.
4. **Coordinate handling:** Remove `1/factor` coordinate scaling for this path; real-space crop preserves pixel size, so coordinates remain in the same pixel frame.
5. **Verification depth:** Require both unit-level numeric assertions and an end-to-end smoke/study rerun with artifact checks.

---

### Task 1: Add RED Tests For Flipped Downsample Semantics

**Files:**
- Modify: `tests/studies/test_prepare_nersc_hybrid_dataset.py`
- Test: `tests/studies/test_prepare_nersc_hybrid_dataset.py`

**Step 1: Add failing behavior tests for the transform flip**

Add tests that enforce the new policy:

```python
def test_downsample_external_payload_bins_diffraction_blocks():
    payload = {"diffraction": diffraction_6x6_known_values, ...}
    out = _downsample_external_payload(payload, target_n=3)
    assert np.allclose(out["diffraction"], expected_block_means)


def test_downsample_external_payload_center_crops_object_and_probe_not_bin():
    payload = {"objectGuess": object_odd_shape, "probeGuess": probe_even_shape, ...}
    out = _downsample_external_payload(payload, target_n=3)
    assert np.array_equal(out["objectGuess"], expected_center_crop)
    assert np.array_equal(out["probeGuess"], expected_center_crop_probe)


def test_downsample_external_payload_preserves_coords_when_real_space_is_cropped():
    payload = {"xcoords": x, "ycoords": y, ...}
    out = _downsample_external_payload(payload, target_n=3)
    assert np.array_equal(out["xcoords"], x)
    assert np.array_equal(out["ycoords"], y)
```

**Step 2: Run targeted test selector (expect FAIL)**

Run:

```bash
pytest tests/studies/test_prepare_nersc_hybrid_dataset.py -k "downsample_external_payload" -v
```

Expected: FAIL on old crop/bin/coord-scale assumptions.

**Step 3: Commit RED tests**

```bash
git add tests/studies/test_prepare_nersc_hybrid_dataset.py
git commit -m "test(studies): add red tests for nersc flipped downsample policy"
```

---

### Task 2: Implement Flipped Downsample Policy In Shared Helper

**Files:**
- Modify: `scripts/studies/prepare_nersc_hybrid_dataset.py`
- Test: `tests/studies/test_prepare_nersc_hybrid_dataset.py`

**Step 1: Add diffraction stack binning helper**

Implement a helper for rank-3 diffraction data:

```python
def _bin_real_stack(stack: np.ndarray, factor: int) -> np.ndarray:
    # center-crop to factor-compatible H/W, then mean-pool blocks
    ...
```

Expected behavior:
- Input shape `[N, H, W]`
- Optional center-crop to divisible H/W
- Reshape to `[N, H//f, f, W//f, f]`
- Mean over block axes `(2, 4)`
- Return `float32`

**Step 2: Add real-space center-crop helper for 2D arrays**

```python
def _crop_center_2d(array_2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    ...
```

Use for `objectGuess` and `probeGuess` to resize by factor via crop (not averaging).

**Step 3: Update `_downsample_external_payload`**

Change logic:
- `downsampled["diffraction"] = _bin_real_stack(diffraction, factor)`
- Real-space outputs from center-crop to reduced dimensions.
- Remove coordinate scale multiplication loop.
- Keep `scan_index` behavior unchanged.

**Step 4: Run tests (expect PASS)**

```bash
pytest tests/studies/test_prepare_nersc_hybrid_dataset.py -v
```

**Step 5: Commit GREEN implementation**

```bash
git add scripts/studies/prepare_nersc_hybrid_dataset.py tests/studies/test_prepare_nersc_hybrid_dataset.py
git commit -m "feat(studies): flip nersc downsample to diffraction-bin and real-space-crop"
```

---

### Task 3: Add Integration Guard For Orchestration Usage

**Files:**
- Modify: `tests/studies/test_nersc_ptychovit_orchestration.py`
- Test: `tests/studies/test_nersc_ptychovit_orchestration.py`

**Step 1: Add an orchestration-level contract test**

Add a test that ensures scan807 conversion path uses the updated helper behavior (not stale assumptions):

```python
def test_convert_pair_to_downsampled_external_npz_applies_flipped_policy(monkeypatch, tmp_path):
    ...
    assert converted_payload["diffraction"] == expected_binned
    assert converted_payload["objectGuess"] == expected_cropped
```

Use monkeypatch to avoid expensive full orchestration.

**Step 2: Run selector (expect FAIL initially if needed)**

```bash
pytest tests/studies/test_nersc_ptychovit_orchestration.py -k "downsampled_external_npz" -v
```

**Step 3: Adjust code/tests as needed and run full file**

```bash
pytest tests/studies/test_nersc_ptychovit_orchestration.py -v
```

**Step 4: Commit**

```bash
git add tests/studies/test_nersc_ptychovit_orchestration.py
git commit -m "test(studies): guard nersc orchestration downsample policy"
```

---

### Task 4: Update Docs For New Data-Prep Semantics

**Files:**
- Modify: `docs/plans/2026-02-17-nersc-ptychovit-hybrid-orchestration-plan.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/workflows/ptychovit.md`

**Step 1: Update the old stated semantics**

Replace mentions of:
- `crop diffraction, bin complex object/probe, scale coords`

With:
- `bin diffraction, center-crop real-space object/probe, keep coords in same pixel frame`.

**Step 2: Run docs consistency grep**

```bash
rg -n "crop diffraction|bin complex object|scale coords|bin diffraction|real-space" docs/plans/2026-02-17-nersc-ptychovit-hybrid-orchestration-plan.md docs/studies/index.md docs/workflows/ptychovit.md
```

**Step 3: Commit docs update**

```bash
git add docs/plans/2026-02-17-nersc-ptychovit-hybrid-orchestration-plan.md docs/studies/index.md docs/workflows/ptychovit.md
git commit -m "docs(studies): update nersc downsample semantics after policy flip"
```

---

### Task 5: Post-Fix Verification (Smoke + Study Rerun)

**Files:**
- Runtime artifacts only under `tmp/` or `outputs/`

**Step 1: Run required local smoke tests**

```bash
pytest -q \
  tests/studies/test_prepare_nersc_hybrid_dataset.py \
  tests/studies/test_nersc_ptychovit_orchestration.py
```

Expected: PASS.

**Step 2: Run orchestration smoke rerun in tmux**

Because this is long-running, execute in tmux per repo policy.

```bash
python scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py \
  --scan807-dp /home/ollie/Downloads/nersc/testdata/scan807_dp.hdf5 \
  --scan807-para /home/ollie/Downloads/nersc/testdata/scan807_para.hdf5 \
  --cameraman-dp /home/ollie/Downloads/nersc/testdata/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/testdata/cameraman256_para.hdf5 \
  --ptychovit-checkpoint datasets/run145/best_model.pth \
  --half top \
  --output-dir outputs/nersc_scan807_cameraman_study_downsample_flip_smoke \
  --seed 3
```

Expected artifacts:
- `outputs/nersc_scan807_cameraman_study_downsample_flip_smoke/manifest.json`
- per-dataset recon files for `pinn_ptychovit` and `pinn_hybrid_resnet`
- per-dataset `metrics_by_model.json` and `visuals/`

**Step 3: Validate new prep semantics on produced NPZs**

Run a short inspection snippet (or equivalent script) verifying:
- diffraction is block-binned (not center-cropped)
- object/probe are center-cropped (not binned)
- coordinates are unchanged vs pre-downsample canonical payload

**Step 4: Optional full study rerun for final evidence**

Re-run into a clean output directory with final checkpoint path and archive logs/manifests per testing guide.

**Step 5: Commit verification metadata/docs references (if tracked)**

```bash
git add <any tracked verification notes/log indices>
git commit -m "test(studies): rerun nersc smoke study after downsample policy flip"
```

---

## Final Verification Gate

Run before completion:

```bash
pytest -q tests/studies/test_prepare_nersc_hybrid_dataset.py tests/studies/test_nersc_ptychovit_orchestration.py
pytest -q -m integration
```

Record command outputs in your execution notes and confirm no regression against invocation logging expectations.
