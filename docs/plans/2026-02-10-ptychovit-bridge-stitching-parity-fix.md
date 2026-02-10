# PtychoViT Bridge Stitching Parity Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace non-parity scan-wise mean reconstruction in the PtychoViT bridge with position-aware stitching, then regenerate fresh baseline evidence and harden verification against collapsed recon behavior.

**Architecture:** Keep NPZ->HDF5 interop unchanged except for contract-compliant consumption at inference time. During bridge inference, collect per-scan predicted complex patches and stitch them into object space using scan positions plus occupancy normalization (upstream parity behavior). Add verifier checks so future regressions (e.g., accidental mean aggregation or collapsed covered-region outputs) fail fast.

**Tech Stack:** Python 3, NumPy, torch, h5py, pytest, `scripts/studies/ptychovit_bridge_entrypoint.py`, `scripts/studies/verify_fresh_ptychovit_initial_metrics.py`, `tests/test_ptychovit_bridge_entrypoint.py`, `tests/test_verify_fresh_ptychovit_initial_metrics.py`.

---

### Task 0: Bridge Stitching Contract Tests

**Files:**
- Modify: `tests/test_ptychovit_bridge_entrypoint.py`
- Reference: `/home/ollie/Documents/ptycho-vit/training.py`

**Step 1: Write failing tests**

```python
def test_bridge_inference_stitches_patches_using_positions(tmp_path):
    # Arrange synthetic predictions with distinct per-scan values and non-trivial positions.
    # Assert reconstruction matches stitched placement semantics, not arithmetic mean.


def test_bridge_inference_outputs_object_space_shape_not_patch_mean_only(tmp_path):
    # Assert recon shape equals para object shape and honors occupancy normalization.
```

**Step 2: Run tests to verify RED**

Run:
- `pytest tests/test_ptychovit_bridge_entrypoint.py -k "stitches_patches_using_positions or object_space_shape_not_patch_mean_only" -v`

Expected: FAIL (current bridge averages scan predictions).

**Step 3: Write minimal implementation in bridge inference**

- Add helper in `scripts/studies/ptychovit_bridge_entrypoint.py` to:
  - load/cache probe positions from dataset in pixel frame
  - assemble predicted complex patches onto object canvas via Fourier-shift placement
  - build occupancy buffer and normalize by `clip(buffer, min=1)`
- Ensure output `YY_pred` is stitched object-space complex map (not mean over scans).

**Step 4: Run tests to verify GREEN**

Run:
- `pytest tests/test_ptychovit_bridge_entrypoint.py -k "stitches_patches_using_positions or object_space_shape_not_patch_mean_only" -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/ptychovit_bridge_entrypoint.py tests/test_ptychovit_bridge_entrypoint.py
git commit -m "fix: stitch ptychovit bridge predictions with scan-position parity"
```

### Task 1: Collapse Guard in Verifier

**Files:**
- Modify: `scripts/studies/verify_fresh_ptychovit_initial_metrics.py`
- Modify: `tests/test_verify_fresh_ptychovit_initial_metrics.py`

**Step 1: Write failing tests**

```python
def test_verifier_fails_when_recon_covered_region_is_nearly_constant(tmp_path):
    # Build compliant artifact layout but with reconstructed amplitude variance below threshold in covered region.
    # Expect non-zero exit with actionable message.
```

**Step 2: Run tests to verify RED**

Run:
- `pytest tests/test_verify_fresh_ptychovit_initial_metrics.py -k covered_region_is_nearly_constant -v`

Expected: FAIL before verifier hardening.

**Step 3: Write minimal verifier implementation**

- Add covered-region variance check:
  - derive coverage mask from test para positions + patch shape
  - compute amplitude std only on covered pixels
  - fail when below configurable threshold
- Include clear failure text indicating likely assembly/collapse regression.

**Step 4: Run tests to verify GREEN**

Run:
- `pytest tests/test_verify_fresh_ptychovit_initial_metrics.py -k covered_region_is_nearly_constant -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/verify_fresh_ptychovit_initial_metrics.py tests/test_verify_fresh_ptychovit_initial_metrics.py
git commit -m "test: detect collapsed covered-region reconstructions in fresh verifier"
```

### Task 2: Spec + Workflow Sync

**Files:**
- Modify: `docs/workflows/ptychovit.md`
- Modify: `specs/ptychovit_interop_contract.md`
- Modify: `tests/test_docs_ptychovit_workflow.py`

**Step 1: Write failing docs test**

```python
def test_ptychovit_workflow_requires_position_aware_stitching_contract():
    text = Path("docs/workflows/ptychovit.md").read_text()
    assert "position-aware stitching" in text
    assert "mean aggregation" in text
```

**Step 2: Run test to verify RED**

Run:
- `pytest tests/test_docs_ptychovit_workflow.py -k stitching_contract -v`

Expected: FAIL before docs/spec sync.

**Step 3: Update docs/spec language**

- Ensure workflow + spec both require stitched object reconstruction.
- Ensure both explicitly mark scan-wise mean aggregation as non-compliant.

**Step 4: Run test to verify GREEN**

Run:
- `pytest tests/test_docs_ptychovit_workflow.py -k stitching_contract -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add docs/workflows/ptychovit.md specs/ptychovit_interop_contract.md tests/test_docs_ptychovit_workflow.py
git commit -m "docs: codify ptychovit stitching parity reconstruction contract"
```

### Task 3: Fresh Baseline Regeneration + Evidence

**Files:**
- Modify: `docs/plans/2026-02-10-ptychovit-bridge-stitching-parity-fix.md`
- Modify: `docs/plans/2026-02-10-ptychovit-lines-data-contract-fix.md`
- Modify/Create (gitignored): `.artifacts/ptychovit_lines_contract_fix/README.md`

**Step 1: Run fresh baseline after stitch fix**

Run:

```bash
python scripts/studies/run_fresh_ptychovit_initial_metrics.py \
  --checkpoint <absolute-path-to-best_model.pth> \
  --output-dir tmp/ptychovit_initial_fresh_stitched \
  --ptychovit-repo /home/ollie/Documents/ptycho-vit \
  --set-phi --force-clean
```

Expected:
- Exit 0
- `tmp/ptychovit_initial_fresh_stitched/recons/pinn_ptychovit/recon.npz` exists

**Step 2: Run verifier**

Run:

```bash
python scripts/studies/verify_fresh_ptychovit_initial_metrics.py \
  --output-dir tmp/ptychovit_initial_fresh_stitched
```

Expected:
- `Fresh baseline verification PASSED`

**Step 3: Record metrics and diagnostics**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('tmp/ptychovit_initial_fresh_stitched/metrics_by_model.json').read_text())['pinn_ptychovit']['metrics']
for k in ['mae','mse','psnr','ssim','ms_ssim','frc50']:
    print(k, m[k])
PY
```

Expected:
- Finite metric arrays
- Evidence includes whether metrics improved vs pre-stitch baseline

**Step 4: Update plan evidence sections**

- Append RED/GREEN test evidence and fresh run outputs to this plan.
- Add cross-reference in `docs/plans/2026-02-10-ptychovit-lines-data-contract-fix.md`.

**Step 5: Commit**

```bash
git add docs/plans/2026-02-10-ptychovit-bridge-stitching-parity-fix.md docs/plans/2026-02-10-ptychovit-lines-data-contract-fix.md

git commit -m "chore: record stitched ptychovit fresh baseline evidence"
```

## Expected Outputs

- `tmp/ptychovit_initial_fresh_stitched/metrics_by_model.json`
- `tmp/ptychovit_initial_fresh_stitched/recons/pinn_ptychovit/recon.npz`
- `tmp/ptychovit_initial_fresh_stitched/visuals/amp_phase_pinn_ptychovit.png`
- `tmp/ptychovit_initial_fresh_stitched/runs/pinn_ptychovit/manifest.json`
- Updated evidence pointers under `.artifacts/ptychovit_lines_contract_fix/README.md`

## Non-Goals (YAGNI)

- No upstream `ptycho-vit` code modifications
- No backend expansion beyond `pinn_ptychovit`
- No changes to cross-model harmonization policy in `grid_lines_compare_wrapper`

## Execution Evidence (2026-02-10)

- Task 0 RED:
  - `pytest tests/test_ptychovit_bridge_entrypoint.py -k "stitches_patches_using_positions or object_space_shape_not_patch_mean_only" -v`
  - Failed with import error for `_stitch_complex_predictions` before implementation.
- Task 0 GREEN:
  - Same selector passed after bridge stitching implementation.
- Task 1 RED:
  - `pytest tests/test_verify_fresh_ptychovit_initial_metrics.py -k covered_region_is_nearly_constant -v`
  - Failed because verifier did not check covered-region amplitude variance.
- Task 1 GREEN:
  - Same selector passed after covered-region collapse guard was implemented.
- Task 2 RED:
  - `pytest tests/test_docs_ptychovit_workflow.py -k stitching_contract -v`
  - Failed on missing explicit `scan-wise mean aggregation` contract wording.
- Task 2 GREEN:
  - Same selector passed after workflow/spec wording updates.
- Regression suite:
  - `pytest tests/test_ptychovit_bridge_entrypoint.py tests/test_verify_fresh_ptychovit_initial_metrics.py tests/test_docs_ptychovit_workflow.py -v`
  - Result: `25 passed`.
- Task 3 fresh baseline:
  - Command:
    - `python scripts/studies/run_fresh_ptychovit_initial_metrics.py --checkpoint /home/ollie/Documents/tmp/PtychoPINN/tmp/ptychovit_initial_fresh_n8_contractcheck/runs/pinn_ptychovit/best_model.pth --output-dir tmp/ptychovit_initial_fresh_stitched --ptychovit-repo /home/ollie/Documents/ptycho-vit --set-phi --force-clean`
  - Result:
    - `Seeded checkpoint: tmp/ptychovit_initial_fresh_stitched/runs/pinn_ptychovit/best_model.pth`
    - `Run complete: tmp/ptychovit_initial_fresh_stitched`
- Task 3 verifier:
  - `python scripts/studies/verify_fresh_ptychovit_initial_metrics.py --output-dir tmp/ptychovit_initial_fresh_stitched`
  - Result: `Fresh baseline verification PASSED`.
- Task 3 metrics extraction:
  - `mae [0.526398777961731, 0.21783155088676898]`
  - `mse [0.42637017369270325, 0.07172433506488773]`
  - `psnr [51.83293568498609, 59.57413830617984]`
  - `ssim [0.2185700448202355, 0.8263113885699188]`
  - `ms_ssim [0.07722588552600171, 0.004342017606292724]`
  - `frc50 [1, 1]`
- Artifacts pointer:
  - `.artifacts/ptychovit_lines_contract_fix/README.md`
