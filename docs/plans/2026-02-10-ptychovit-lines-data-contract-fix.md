# PtychoViT Lines Data-Contract Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix poor checkpoint-restored `pinn_ptychovit` recon quality on synthetic lines runs by aligning interop data with PtychoViT assumptions (scan positions + normalization), then regenerate fresh baseline metrics/PNGs with provenance checks.

**Architecture:** Keep the existing NPZ→HDF5 adapter boundary, but upgrade the contract at two points: (1) emit and consume true scan positions rather than all-zero relative coords; (2) provide explicit normalization dictionaries to avoid fallback defaults in upstream loader. Preserve existing wrapper contracts (`metrics.json`, `metrics_by_model.json`, `recon_npz`) and strengthen verifier checks to fail fast on broken data semantics.

**Tech Stack:** Python 3, NumPy, h5py, pytest, subprocess, existing `ptycho/workflows/grid_lines_workflow.py`, `ptycho/interop/ptychovit/convert.py`, `scripts/studies/ptychovit_bridge_entrypoint.py`, `scripts/studies/verify_fresh_ptychovit_initial_metrics.py`.

**Execution discipline:**
- Use `@superpowers:test-driven-development` for each task.
- Use `@superpowers:systematic-debugging` if any new smoke run fails unexpectedly.
- Use `@superpowers:verification-before-completion` before final claims.
- Keep docs and tests in same commits as behavior changes.

---

### Task 0: Position Contract Test Gate (No All-Zero Scan Positions)

**Files:**
- Modify: `tests/test_grid_lines_workflow.py`
- Modify: `tests/test_ptychovit_adapter.py`

**Step 1: Write failing tests**

```python
# tests/test_grid_lines_workflow.py

def test_build_grid_lines_datasets_persists_nonconstant_scan_positions(tmp_path):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, build_grid_lines_datasets
    ...
    out = build_grid_lines_datasets(cfg)
    with np.load(out["train_npz"], allow_pickle=True) as d:
        assert "coords_offsets" in d.files
        pos = np.asarray(d["coords_offsets"])
        assert np.unique(pos).size > 1
```

```python
# tests/test_ptychovit_adapter.py

def test_convert_prefers_coords_offsets_for_probe_positions(tmp_path):
    from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair
    # create NPZ with coords_nominal all zeros, coords_offsets non-constant
    ...
    out = convert_npz_split_to_hdf5_pair(...)
    with h5py.File(out.para_hdf5, "r") as f:
        x = np.asarray(f["probe_position_x_m"])
        y = np.asarray(f["probe_position_y_m"])
    assert np.unique(x).size > 1
    assert np.unique(y).size > 1
```

**Step 2: Run tests to verify RED**

Run:
- `pytest tests/test_grid_lines_workflow.py -k persists_nonconstant_scan_positions -v`
- `pytest tests/test_ptychovit_adapter.py -k prefers_coords_offsets_for_probe_positions -v`

Expected: FAIL before position contract implementation.

**Step 3: Write minimal implementation**

- In `build_grid_lines_datasets(...)`, persist `coords_offsets` from container when available:
  - train/test NPZ payload includes `coords_offsets` key.
- In converter, update position source priority:
  - prefer `coords_offsets`
  - fallback to `coords_true`
  - fallback to `coords_nominal`
- Support shape extraction for channel-format offsets `(N,1,2,1)` and legacy forms.

**Step 4: Run tests to verify GREEN**

Run same selectors; expected PASS.

**Step 5: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py ptycho/interop/ptychovit/convert.py tests/test_grid_lines_workflow.py tests/test_ptychovit_adapter.py
git commit -m "fix: use true scan positions for ptychovit interop on lines data"
```

### Task 1: Normalization Contract Fix (No Loader Fallback)

**Files:**
- Modify: `scripts/studies/ptychovit_bridge_entrypoint.py`
- Modify: `tests/test_ptychovit_bridge_entrypoint.py`

**Step 1: Write failing tests**

```python
# tests/test_ptychovit_bridge_entrypoint.py

def test_prepare_runtime_training_config_writes_normalization_dict(tmp_path):
    from scripts.studies.ptychovit_bridge_entrypoint import _prepare_runtime_training_config, parse_args
    ...
    runtime_cfg_path, cfg = _prepare_runtime_training_config(args)
    norm_path = Path(cfg["data"]["normalization_dict_path"])
    test_norm_path = Path(cfg["data"]["test_normalization"])
    assert norm_path.exists()
    assert test_norm_path.exists()
    assert norm_path == test_norm_path
```

```python
# tests/test_ptychovit_bridge_entrypoint.py

def test_generated_normalization_dict_contains_train_and_test_object_names(tmp_path):
    ...
    # assert pickle has keys for both copied train/test object names
```

**Step 2: Run tests to verify RED**

Run:
- `pytest tests/test_ptychovit_bridge_entrypoint.py -k normalization_dict -v`

Expected: FAIL before normalization dict generation.

**Step 3: Write minimal implementation**

- Add helper in bridge entrypoint:
  - read copied `train_dp.hdf5`/`test_dp.hdf5`
  - compute `max(dp)` per object name
  - write pickle dict under `bridge_work/data/normalization.pkl`
- Set both:
  - `config["data"]["normalization_dict_path"]`
  - `config["data"]["test_normalization"]`
  to the generated pickle path.

**Step 4: Run tests to verify GREEN**

Run same selector; expected PASS.

**Step 5: Commit**

```bash
git add scripts/studies/ptychovit_bridge_entrypoint.py tests/test_ptychovit_bridge_entrypoint.py
git commit -m "fix: generate runtime normalization dict for ptychovit bridge datasets"
```

### Task 2: Verifier Hardening For Position/Normalization Assumptions

**Files:**
- Modify: `scripts/studies/verify_fresh_ptychovit_initial_metrics.py`
- Modify: `tests/test_verify_fresh_ptychovit_initial_metrics.py`

**Step 1: Write failing tests**

```python
# tests/test_verify_fresh_ptychovit_initial_metrics.py

def test_verifier_fails_when_probe_positions_constant_zero(tmp_path):
    ...


def test_verifier_fails_when_stdout_contains_normalization_fallback_warning(tmp_path):
    ...
```

**Step 2: Run tests to verify RED**

Run:
- `pytest tests/test_verify_fresh_ptychovit_initial_metrics.py -k "positions_constant_zero or normalization_fallback_warning" -v`

Expected: FAIL before verifier update.

**Step 3: Write minimal implementation**

- Verifier additionally checks:
  - `probe_position_x_m` and `probe_position_y_m` from interop test para file are not constant/degenerate.
  - `stdout.log` does not contain `Normalization file not found` warning.

**Step 4: Run tests to verify GREEN**

Run same selector; expected PASS.

**Step 5: Commit**

```bash
git add scripts/studies/verify_fresh_ptychovit_initial_metrics.py tests/test_verify_fresh_ptychovit_initial_metrics.py
git commit -m "test: fail fresh baseline verification on bad position/normalization contracts"
```

### Task 3: Fresh Baseline Regeneration (Checkpoint-Restored, Lines)

**Files:**
- No code changes required (execution/evidence task)
- Modify: `docs/plans/2026-02-10-ptychovit-lines-data-contract-fix.md` (append evidence)
- Create: `.artifacts/ptychovit_lines_contract_fix/README.md` (gitignored pointer)

**Step 1: Run fresh baseline command**

Run:

```bash
python scripts/studies/run_fresh_ptychovit_initial_metrics.py \
  --checkpoint <absolute-path-to-known-good-best_model.pth> \
  --output-dir tmp/ptychovit_initial_fresh_fixed \
  --ptychovit-repo /home/ollie/Documents/ptycho-vit \
  --set-phi
```

Expected:
- Run exits 0.
- `tmp/ptychovit_initial_fresh_fixed/recons/pinn_ptychovit/recon.npz` exists.
- `tmp/ptychovit_initial_fresh_fixed/visuals/amp_phase_pinn_ptychovit.png` exists.

**Step 2: Run hardened verifier**

Run:

```bash
python scripts/studies/verify_fresh_ptychovit_initial_metrics.py \
  --output-dir tmp/ptychovit_initial_fresh_fixed
```

Expected: exit 0 with `Fresh baseline verification PASSED`.

**Step 3: Record metric summary + PNG paths**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('tmp/ptychovit_initial_fresh_fixed/metrics_by_model.json').read_text())['pinn_ptychovit']['metrics']
print('MAE', m['mae'])
print('MSE', m['mse'])
print('PSNR', m['psnr'])
print('SSIM', m['ssim'])
print('MS-SSIM', m['ms_ssim'])
print('FRC50', m['frc50'])
PY
```

Expected: finite metric arrays; include output in evidence README.

**Step 4: Archive evidence pointers**

Run:
- `mkdir -p .artifacts/ptychovit_lines_contract_fix`
- Write artifact index with command lines, logs, JSON paths, and PNG paths.

**Step 5: Commit evidence update**

```bash
git add docs/plans/2026-02-10-ptychovit-lines-data-contract-fix.md
git commit -m "chore: record fixed lines-contract ptychovit fresh baseline evidence"
```

### Task 4: Docs Update For New Assumptions

**Files:**
- Modify: `docs/workflows/ptychovit.md`
- Modify: `tests/test_docs_ptychovit_workflow.py`

**Step 1: Write failing docs test**

```python
# tests/test_docs_ptychovit_workflow.py

def test_ptychovit_workflow_calls_out_nonzero_scan_positions_and_norm_dict_requirement():
    text = Path("docs/workflows/ptychovit.md").read_text()
    assert "probe_position_x_m/probe_position_y_m must be non-constant" in text
    assert "normalization dictionary" in text
    assert "Normalization file not found" in text
```

**Step 2: Run tests to verify RED**

Run:
- `pytest tests/test_docs_ptychovit_workflow.py -k nonzero_scan_positions_and_norm_dict_requirement -v`

Expected: FAIL before docs update.

**Step 3: Write minimal docs updates**

- Add explicit assumptions section:
  - position vectors must encode scan trajectory (not all zero).
  - runtime normalization dictionary is required; fallback warning indicates invalid baseline.
- Add troubleshooting steps for these failures.

**Step 4: Run tests to verify GREEN**

Run:
- `pytest tests/test_docs_ptychovit_workflow.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add docs/workflows/ptychovit.md tests/test_docs_ptychovit_workflow.py
git commit -m "docs: codify ptychovit lines position and normalization requirements"
```

## Expected Outputs

- `tmp/ptychovit_initial_fresh_fixed/metrics_by_model.json`
- `tmp/ptychovit_initial_fresh_fixed/recons/pinn_ptychovit/recon.npz`
- `tmp/ptychovit_initial_fresh_fixed/visuals/amp_phase_pinn_ptychovit.png`
- `tmp/ptychovit_initial_fresh_fixed/visuals/compare_amp_phase.png`
- `tmp/ptychovit_initial_fresh_fixed/runs/pinn_ptychovit/manifest.json`
- `.artifacts/ptychovit_lines_contract_fix/README.md`

## Execution Evidence (2026-02-10)

- Task 0 RED:
  - `pytest tests/test_grid_lines_workflow.py -k persists_nonconstant_scan_positions -v` (failed: `coords_offsets` missing)
  - `pytest tests/test_ptychovit_adapter.py -k prefers_coords_offsets_for_probe_positions -v` (failed: constant probe positions)
- Task 0 GREEN:
  - same selectors passed after position-contract implementation
- Task 1 RED:
  - `pytest tests/test_ptychovit_bridge_entrypoint.py -k normalization_dict -v` (failed: normalization paths were `None`)
- Task 1 GREEN:
  - same selector passed after runtime `normalization.pkl` generation/wiring
- Task 2 RED:
  - `pytest tests/test_verify_fresh_ptychovit_initial_metrics.py -k "positions_constant_zero or normalization_fallback_warning" -v` (failed)
- Task 2 GREEN:
  - same selector passed after verifier hardening
- Task 4 RED:
  - `pytest tests/test_docs_ptychovit_workflow.py -k nonzero_scan_positions_and_norm_dict_requirement -v` (failed)
- Task 4 GREEN:
  - `pytest tests/test_docs_ptychovit_workflow.py -v` (passed)
- Fresh baseline regeneration:
  - Command:
    - `python scripts/studies/run_fresh_ptychovit_initial_metrics.py --checkpoint /home/ollie/Documents/tmp/PtychoPINN/tmp/ptychovit_initial_fresh_smoke2/runs/pinn_ptychovit/best_model.pth --output-dir tmp/ptychovit_initial_fresh_fixed --ptychovit-repo /home/ollie/Documents/ptycho-vit --set-phi --force-clean`
  - Result:
    - `Seeded checkpoint: tmp/ptychovit_initial_fresh_fixed/runs/pinn_ptychovit/best_model.pth`
    - `Run complete: tmp/ptychovit_initial_fresh_fixed`
- Hardened verifier:
  - `python scripts/studies/verify_fresh_ptychovit_initial_metrics.py --output-dir tmp/ptychovit_initial_fresh_fixed`
  - Result: `Fresh baseline verification PASSED`
- Extracted metrics (`pinn_ptychovit`):
  - `MAE [0.51882404088974, 0.2395657977696754]`
  - `MSE [0.4123847782611847, 0.08664213551357519]`
  - `PSNR [51.97777742267992, 58.75351213105684]`
  - `SSIM [0.21887534208816184, 0.812723282189552]`
  - `MS-SSIM [0.06994964345794019, 0.0037439318039930764]`
  - `FRC50 [2, 1]`
- Artifact index:
  - `.artifacts/ptychovit_lines_contract_fix/README.md`

## Non-Goals (YAGNI)

- No support for `pinn_ptychovit` at `N != 256`.
- No upstream `ptycho-vit` repository code edits.
- No change to harmonization strategy (still object-grid canonical pixel-space metrics).

## Follow-On Resolution Update (2026-02-10)

Status update:

- The previously open stitching-parity gap is now resolved by implementing scan-position-aware
  bridge reconstruction assembly plus covered-region collapse validation.
- Completion evidence is recorded in:
  - `docs/plans/2026-02-10-ptychovit-bridge-stitching-parity-fix.md`
  - `.artifacts/ptychovit_lines_contract_fix/README.md`

Canonical contract references:

- `specs/ptychovit_interop_contract.md`
- `docs/workflows/ptychovit.md`
