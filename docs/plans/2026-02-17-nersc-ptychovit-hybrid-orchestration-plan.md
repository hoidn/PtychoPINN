# NERSC Scan807 + Cameraman Orchestration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible study orchestration that runs checkpoint-restored PtychoViT inference on `scan807` and `cameraman256`, trains `hybrid_resnet` on half of cameraman at `N=128` for 40 epochs, runs cross-dataset hybrid inference on full `scan807` and full `cameraman256`, then produces comparison metrics and visuals for both datasets.

**Architecture:** Add a thin orchestration layer under `scripts/studies/` that reuses existing conversion/runner/evaluation paths wherever possible. Adapt NERSC paired HDF5 datasets into canonical NPZ inputs for external-raw Torch studies, run PtychoViT via the existing bridge entrypoint in inference mode, run Hybrid ResNet training/inference via existing Torch runner/model-loading paths, and aggregate metrics/visuals via existing grid-lines comparison helpers. Keep model execution paths unchanged; add new abstractions only when they improve maintainability or remove duplication.

**Tech Stack:** Python 3.11, NumPy, h5py, argparse, subprocess, PyTorch/Lightning runtime, existing `scripts/studies/*` wrappers, pytest

---

## Brainstorming Resolution (Pre-Implementation Decisions)

Resolved decisions (no blocker questions required):

1. **Hybrid training split policy:** default to `top` half for cameraman to mirror successful external `N=128 e40` study patterns; support `--half {top,bottom}` for explicit override.
2. **Hybrid training resolution:** downsample cameraman `256->128` using study semantics (bin diffraction, center-crop complex object/probe, keep coords in the same pixel frame).
3. **PtychoViT runs:** inference-only with checkpoint restore (`--mode inference --checkpoint <best_model.pth>`), run independently for scan807 and cameraman pairs.
4. **Scan807/cameraman pair compliance:** enforce required probe/object pixel attrs by preflight patching into working copies (never mutate source files in place), then re-validate pair contract before any model stage.
5. **Position reassembly safety:** pin external reassembly backend to `shift_sum` only (never `auto`/`batched`) due known batched correctness regression.
6. **Hybrid cross-dataset inference:** use the single trained hybrid checkpoint from cameraman-half training and run on both full scan807 and full cameraman test NPZs.
7. **Metrics/visual policy:** evaluate only predicted model IDs (`pinn_ptychovit`, `pinn_hybrid_resnet`) against GT reference; use `gt` only for visual ordering.
8. **Coordinate frame policy:** convert HDF5 world positions to centered pixel coordinates using object geometry (`center_{x,y}_m`, `pixel_{width,height}_m`) explicitly.
9. **CONFIG-001 policy:** any custom PyTorch inference/training helper must call factory/bridge path that guarantees `update_legacy_dict(params.cfg, config)` before legacy-touched modules.
10. **Runtime orchestration:** one deterministic runbook script with explicit stage directories and invocation logs; long runs executed in tmux.

---

### Task 1: Add NERSC Pair Adapter Contract Tests (RED)

**Files:**
- Create: `tests/studies/test_nersc_pair_adapter.py`
- Test: `tests/studies/test_nersc_pair_adapter.py`

**Step 1: Write failing tests for adapter contract**

Add tests that define expected behavior:

```python
def test_pair_preflight_adds_missing_probe_pixel_attrs_to_working_copy():
    ...

def test_pair_to_npz_converts_dp_intensity_to_amplitude_and_positions_to_pixels():
    ...

def test_pair_to_npz_emits_external_raw_required_keys():
    ...

def test_pair_to_npz_applies_object_centered_world_to_pixel_mapping():
    ...
```

**Step 2: Run test to verify failure**

Run: `pytest tests/studies/test_nersc_pair_adapter.py -v`
Expected: FAIL because adapter module does not exist.

**Step 3: Commit RED tests**

```bash
git add tests/studies/test_nersc_pair_adapter.py
git commit -m "test(studies): add red contract tests for nersc hdf5 pair adapter"
```

---

### Task 2: Implement HDF5 Pair Adapter + Compliance Patcher (GREEN)

**Files:**
- Create: `scripts/studies/nersc_pair_adapter.py`
- Test: `tests/studies/test_nersc_pair_adapter.py`

**Step 1: Implement minimal adapter API**

Implement:

```python
def materialize_pair_working_copy(dp_h5: Path, para_h5: Path, out_dir: Path) -> tuple[Path, Path]:
    ...

def ensure_required_para_attrs(para_h5: Path, *, default_pixel_m: float | None = None) -> None:
    ...

def pair_to_external_npz(dp_h5: Path, para_h5: Path, out_npz: Path) -> Path:
    ...
```

Contract requirements:
- `dp` interpreted as intensity; write amplitude as `diff3d = sqrt(max(dp,0))`.
- Convert `probe_position_{x,y}_m` (world meters) to centered pixel coordinates with:
  - `x_px = (x_m - center_x_m) / pixel_width_m`
  - `y_px = (y_m - center_y_m) / pixel_height_m`
- Do not ignore object center attrs (`center_x_m`, `center_y_m`) during conversion.
- Emit NPZ keys: `xcoords`, `ycoords`, `xcoords_start`, `ycoords_start`, `diff3d`, `probeGuess`, `objectGuess`, `scan_index`.
- Preserve complex dtypes for object/probe.
- If probe attrs missing, add `pixel_height_m`/`pixel_width_m` in working copy from object attrs (or explicit fallback).
- Re-validate the working-copy pair after patching and hard-fail if required attrs are still missing.

**Step 2: Re-run tests**

Run: `pytest tests/studies/test_nersc_pair_adapter.py -v`
Expected: PASS.

**Step 3: Commit GREEN implementation**

```bash
git add scripts/studies/nersc_pair_adapter.py tests/studies/test_nersc_pair_adapter.py
git commit -m "feat(studies): add nersc hdf5 pair adapter for external npz workflows"
```

---

### Task 3: Add Cameraman 256→128 Half-Train/Full-Test Prep Tests (RED)

**Files:**
- Create: `tests/studies/test_prepare_nersc_hybrid_dataset.py`
- Test: `tests/studies/test_prepare_nersc_hybrid_dataset.py`

**Step 1: Write failing dataset-prep tests**

```python
def test_prepare_hybrid_dataset_writes_train_half_and_full_test_npz():
    ...

def test_prepare_hybrid_dataset_supports_top_and_bottom_half():
    ...

def test_prepare_hybrid_dataset_records_manifest_and_counts():
    ...
```

**Step 2: Run test to verify failure**

Run: `pytest tests/studies/test_prepare_nersc_hybrid_dataset.py -v`
Expected: FAIL because prep script/module is missing.

**Step 3: Commit RED tests**

```bash
git add tests/studies/test_prepare_nersc_hybrid_dataset.py
git commit -m "test(studies): add red tests for cameraman hybrid dataset prep"
```

---

### Task 4: Implement Cameraman Hybrid Dataset Prep Script (GREEN)

**Files:**
- Create: `scripts/studies/prepare_nersc_hybrid_dataset.py`
- Modify: `scripts/tools/downsample_data_tool.py` (only if unavoidable)
- Test: `tests/studies/test_prepare_nersc_hybrid_dataset.py`

**Step 1: Implement prep flow**

Implement CLI that:
1. Converts cameraman pair -> canonical external NPZ (Task 2 adapter).
2. Downsamples to `N=128` (diffraction binning + real-space center-crop; no coordinate rescaling).
3. Produces train split using half-space mask (`top` default, `bottom` option).
4. Writes full test NPZ (unsplit full object).
5. Writes manifest JSON with source paths, split threshold, counts, SHA256.

**Step 2: Run tests**

Run: `pytest tests/studies/test_prepare_nersc_hybrid_dataset.py -v`
Expected: PASS.

**Step 3: Commit**

```bash
git add scripts/studies/prepare_nersc_hybrid_dataset.py tests/studies/test_prepare_nersc_hybrid_dataset.py
git commit -m "feat(studies): add cameraman half-train full-test prep for hybrid e40 study"
```

---

### Task 5: Add PtychoViT Restored-Inference Orchestration Tests (RED)

**Files:**
- Create: `tests/studies/test_nersc_ptychovit_orchestration.py`
- Test: `tests/studies/test_nersc_ptychovit_orchestration.py`

**Step 1: Write failing tests for bridge invocation contract**

```python
def test_ptychovit_inference_stage_invokes_bridge_entrypoint_with_checkpoint():
    ...

def test_ptychovit_stage_writes_recon_artifact_paths_for_scan807_and_cameraman():
    ...

def test_orchestrator_writes_invocation_artifacts_for_parent_and_children():
    ...

def test_hybrid_external_inference_pins_shift_sum_backend():
    ...
```

**Step 2: Run test to verify failure**

Run: `pytest tests/studies/test_nersc_ptychovit_orchestration.py -v`
Expected: FAIL until orchestration module exists.

**Step 3: Commit RED tests**

```bash
git add tests/studies/test_nersc_ptychovit_orchestration.py
git commit -m "test(studies): add red tests for ptychovit checkpoint inference orchestration"
```

---

### Task 6: Implement End-to-End NERSC Orchestrator Runbook (GREEN)

**Files:**
- Create: `scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py`
- Create: `scripts/studies/nersc_orchestration.py`
- Test: `tests/studies/test_nersc_ptychovit_orchestration.py`

**Step 1: Implement staged orchestrator**

Implement phases:
1. Preflight/working-copy pair materialization for scan807 + cameraman, including mandatory post-patch pair validation.
2. PtychoViT checkpoint-restored inference for both pairs (via bridge entrypoint subprocess).
3. Cameraman hybrid dataset prep (`N=128`, half-train/full-test).
4. Hybrid training for 40 epochs with e40-aligned params.
5. Hybrid inference on full scan807 and full cameraman with trained checkpoint, forcing external position reassembly backend `shift_sum`.
6. Metrics+visual aggregation per dataset.

Implementation guardrails:
- Explicitly pass `torch_position_reassembly_backend=\"shift_sum\"` for external reassembly stages.
- Reject `auto`/`batched` in this orchestration path to avoid silent routing into known-bad batched behavior.
- Ensure all new `scripts/studies/` entrypoints write `invocation.json` and `invocation.sh` at startup.

Expose key CLI args:
- `--scan807-dp`, `--scan807-para`
- `--cameraman-dp`, `--cameraman-para`
- `--ptychovit-checkpoint`
- `--half {top,bottom}` default `top`
- `--position-reassembly-backend` (choices restricted to `shift_sum` in this study path)
- `--output-dir`
- `--seed`

**Step 2: Re-run orchestration tests**

Run: `pytest tests/studies/test_nersc_ptychovit_orchestration.py -v`
Expected: PASS.

**Step 3: Commit**

```bash
git add scripts/studies/nersc_orchestration.py scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py tests/studies/test_nersc_ptychovit_orchestration.py
git commit -m "feat(studies): add nersc multi-stage orchestration runbook"
```

---

### Task 7: Add Hybrid Checkpoint Reuse Inference Tests + Implementation

**Files:**
- Create: `tests/torch/test_hybrid_checkpoint_cross_dataset_inference.py`
- Create: `scripts/studies/hybrid_checkpoint_inference.py`
- Modify: `scripts/studies/nersc_orchestration.py`

**Step 1: Write failing tests**

```python
def test_cross_dataset_inference_loads_single_model_pt_and_runs_two_test_npzs():
    ...

def test_cross_dataset_inference_writes_recon_npz_for_each_dataset():
    ...
```

**Step 2: Run tests to confirm RED**

Run: `pytest tests/torch/test_hybrid_checkpoint_cross_dataset_inference.py -v`
Expected: FAIL until helper exists.

**Step 3: Implement helper + wire into orchestrator**

Implement model reconstruction/loading based on saved training hyperparams + `model.pt` state dict and run inference on both datasets.

Mandatory CONFIG-001 rule for this helper:
- use config-factory/bridge setup that guarantees `update_legacy_dict(params.cfg, config)` before any loader/model path that can touch legacy modules.
- add test assertions (spy/monkeypatch) that bridge update is called before inference execution.

**Step 4: Run tests to confirm GREEN**

Run: `pytest tests/torch/test_hybrid_checkpoint_cross_dataset_inference.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/hybrid_checkpoint_inference.py scripts/studies/nersc_orchestration.py tests/torch/test_hybrid_checkpoint_cross_dataset_inference.py
git commit -m "feat(torch): add checkpoint-reuse hybrid inference for multi-dataset evaluation"
```

---

### Task 8: Add Metrics/Visual Aggregation Tests + Integration

**Files:**
- Create: `tests/studies/test_nersc_metrics_visuals.py`
- Modify: `scripts/studies/nersc_orchestration.py`

**Step 1: Write failing tests for aggregation output contract**

```python
def test_metrics_visual_stage_writes_metrics_json_metrics_by_model_and_tables():
    ...

def test_metrics_visual_stage_renders_compare_amp_phase_png():
    ...
```

**Step 2: Run tests (RED)**

Run: `pytest tests/studies/test_nersc_metrics_visuals.py -v`
Expected: FAIL.

**Step 3: Implement aggregation stage**

Use:
- `evaluate_selected_models(...)`
- `_finalize_compare_outputs(...)`

For each dataset output root with model IDs:
- `pinn_ptychovit`
- `pinn_hybrid_resnet`

GT handling:
- `gt` is reference-only input to `evaluate_selected_models(...)`.
- keep `gt` in `visual_order` but never include it in scored model IDs passed to metrics aggregation/table generation.

**Step 4: Run tests (GREEN)**

Run: `pytest tests/studies/test_nersc_metrics_visuals.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/nersc_orchestration.py tests/studies/test_nersc_metrics_visuals.py
git commit -m "feat(studies): add per-dataset metrics and visuals aggregation for nersc orchestration"
```

---

### Task 9: Verification, Runbook Evidence, and Study Index/Docs Update

**Files:**
- Modify: `docs/studies/index.md`
- Modify: `docs/workflows/ptychovit.md`
- Modify: `docs/workflows/pytorch.md`
- Create: `docs/plans/2026-02-17-nersc-ptychovit-hybrid-orchestration-notes.md` (optional execution notes)

**Step 1: Run targeted tests**

```bash
pytest tests/studies/test_nersc_pair_adapter.py -v
pytest tests/studies/test_prepare_nersc_hybrid_dataset.py -v
pytest tests/studies/test_nersc_ptychovit_orchestration.py -v
pytest tests/torch/test_hybrid_checkpoint_cross_dataset_inference.py -v
pytest tests/studies/test_nersc_metrics_visuals.py -v
pytest -v -m integration
```

Expected: all PASS.

Additional verification expectations:
- Invocation artifacts exist for each new `scripts/studies/` entrypoint (`invocation.json`, `invocation.sh`).
- External hybrid reassembly config in recorded invocations is pinned to `shift_sum`.

**Step 2: Run orchestrator smoke (tmux recommended)**

```bash
python scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py \
  --scan807-dp /home/ollie/Downloads/nersc/testdata/scan807_dp.hdf5 \
  --scan807-para /home/ollie/Downloads/nersc/testdata/scan807_para.hdf5 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --ptychovit-checkpoint datasets/run145/best_model.pth \
  --half top \
  --position-reassembly-backend shift_sum \
  --output-dir outputs/nersc_scan807_cameraman_study
```

Expected artifacts:
- `outputs/nersc_scan807_cameraman_study/scan807/...`
- `outputs/nersc_scan807_cameraman_study/cameraman256/...`
- per-dataset `metrics.json`, `metrics_by_model.json`, visuals and tables.

**Step 3: Update docs index entries + command snippets**

Document the new runbook and dataset-prep path in study/workflow docs.

**Step 4: Commit docs + verification updates**

```bash
git add docs/studies/index.md docs/workflows/ptychovit.md docs/workflows/pytorch.md
git commit -m "docs(studies): add nersc scan807+cameraman orchestration runbook"
```

---

## Notes on Runtime/Scale

- Use tmux for long-running training/inference orchestration.
- Keep all generated datasets/recons under `outputs/` (git-ignored).
- Preserve source HDF5 files as read-only; write patched pair copies under run output workspace.
- Prefer deterministic seeds and write invocation artifacts at each stage.
