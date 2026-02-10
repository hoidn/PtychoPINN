# Fresh Checkpoint-Restored Lines Initial Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce trustworthy "initial" `pinn_ptychovit` metrics and reconstruction PNGs from a fresh execution path that restores a specified checkpoint and evaluates on synthetic lines data.

**Architecture:** Keep orchestration unchanged: use `grid_lines_compare_wrapper.py` in selected-model mode with `pinn_ptychovit` only and `N=256` for that arm. Enforce provenance at boundaries: clean output directory, explicit checkpoint placement, no stale-recon reuse, and manifest/log checks that prove inference executed from restored weights on the new run artifacts. Add small verifier tooling and tests so this baseline procedure is reproducible and auditable.

**Tech Stack:** Python 3, pytest, subprocess, NumPy JSON checks, existing `scripts/studies/grid_lines_compare_wrapper.py`, existing `scripts/studies/ptychovit_bridge_entrypoint.py`.

**Execution discipline:**
- Use `@superpowers:test-driven-development` for each behavior change.
- Use `@superpowers:verification-before-completion` before completion claims.
- Keep docs updated in the same commit as behavior/tooling changes.

---

### Task 0: Fresh-Run Contract Gate (No Silent Reuse, No Scratch "Finetune")

**Files:**
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Modify: `tests/test_ptychovit_bridge_entrypoint.py`

**Step 1: Write failing tests**

```python
# tests/test_grid_lines_compare_wrapper.py

def test_parse_args_reuse_existing_recons_defaults_false(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args
    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
    ])
    assert args.reuse_existing_recons is False


def test_parse_args_reuse_existing_recons_flag_sets_true(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args
    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--reuse-existing-recons",
    ])
    assert args.reuse_existing_recons is True
```

```python
# tests/test_ptychovit_bridge_entrypoint.py

def test_bridge_inference_manifest_records_checkpoint_and_no_training(...):
    # run mode=inference with explicit --checkpoint
    # assert manifest["mode"] == "inference"
    # assert manifest["checkpoint"] equals supplied checkpoint path
    # assert manifest["training_returncode"] is None
```

**Step 2: Run tests to verify RED**

Run:
- `pytest tests/test_grid_lines_compare_wrapper.py -k "reuse_existing_recons_defaults_false or reuse_existing_recons_flag_sets_true" -v`
- `pytest tests/test_ptychovit_bridge_entrypoint.py -k "manifest_records_checkpoint_and_no_training" -v`

Expected: FAIL if contract behavior is not fully asserted.

**Step 3: Minimal implementation (if needed)**

- Ensure parser contract and manifest fields match tests.
- Keep behavior non-breaking for existing options.

**Step 4: Run tests to verify GREEN**

Run same selectors; expected PASS.

**Step 5: Commit**

```bash
git add tests/test_grid_lines_compare_wrapper.py tests/test_ptychovit_bridge_entrypoint.py
# include source file edits only if required by RED/GREEN
git commit -m "test: lock fresh-run provenance contracts for ptychovit initial metrics"
```

### Task 1: Add Reproducible Fresh-Run Driver Script

**Files:**
- Create: `scripts/studies/run_fresh_ptychovit_initial_metrics.py`
- Create: `tests/test_run_fresh_ptychovit_initial_metrics.py`

**Step 1: Write failing tests**

```python
# tests/test_run_fresh_ptychovit_initial_metrics.py

def test_driver_refuses_existing_output_dir_without_force(tmp_path):
    ...


def test_driver_copies_checkpoint_to_expected_bridge_location(tmp_path, monkeypatch):
    ...


def test_driver_invokes_wrapper_without_reuse_flag(monkeypatch, tmp_path):
    ...
```

**Step 2: Run test to verify RED**

Run: `pytest tests/test_run_fresh_ptychovit_initial_metrics.py -v`

Expected: FAIL with missing script/imports.

**Step 3: Write minimal implementation**

Script contract:
- Inputs:
  - `--checkpoint` (required)
  - `--output-dir` (required)
  - `--ptychovit-repo` (default `/home/ollie/Documents/ptycho-vit`)
  - `--force-clean` (optional)
  - lines-run knobs pass-through: `--N`, `--gridsize`, `--nimgs-train`, `--nimgs-test`, `--torch-epochs`, `--set-phi`
- Behavior:
  - fail if output-dir exists and non-empty without `--force-clean`
  - create `${output_dir}/runs/pinn_ptychovit/`
  - copy checkpoint to `${output_dir}/runs/pinn_ptychovit/best_model.pth`
  - run:

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N <N> --gridsize <gridsize> --output-dir <output_dir> \
  --architectures hybrid \
  --models pinn_ptychovit --model-n pinn_ptychovit=256 \
  --ptychovit-repo <repo> --set-phi \
  --nimgs-train <nimgs_train> --nimgs-test <nimgs_test>
```

  - intentionally do not pass `--reuse-existing-recons`

**Step 4: Run test to verify GREEN**

Run: `pytest tests/test_run_fresh_ptychovit_initial_metrics.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/run_fresh_ptychovit_initial_metrics.py tests/test_run_fresh_ptychovit_initial_metrics.py
git commit -m "feat: add reproducible fresh ptychovit initial-metrics driver"
```

### Task 2: Add Fresh-Execution Verifier (Metrics + PNG + Provenance)

**Files:**
- Create: `scripts/studies/verify_fresh_ptychovit_initial_metrics.py`
- Create: `tests/test_verify_fresh_ptychovit_initial_metrics.py`

**Step 1: Write failing tests**

```python
# tests/test_verify_fresh_ptychovit_initial_metrics.py

def test_verifier_requires_metrics_and_recon_and_visuals(tmp_path):
    ...


def test_verifier_fails_if_stdout_contains_skipped_backend_execution(tmp_path):
    ...


def test_verifier_fails_if_manifest_indicates_training_bootstrap(tmp_path):
    ...


def test_verifier_passes_for_fresh_checkpoint_restored_run(tmp_path):
    ...
```

**Step 2: Run tests to verify RED**

Run: `pytest tests/test_verify_fresh_ptychovit_initial_metrics.py -v`

Expected: FAIL with missing script/imports.

**Step 3: Write minimal implementation**

Verifier checks (exit non-zero on any failure):
- required files exist:
  - `metrics_by_model.json`
  - `recons/pinn_ptychovit/recon.npz`
  - `recons/gt/recon.npz`
  - `visuals/amp_phase_pinn_ptychovit.png`
  - `visuals/compare_amp_phase.png`
  - `runs/pinn_ptychovit/manifest.json`
  - `runs/pinn_ptychovit/stdout.log`
- `metrics_by_model.json` contains `pinn_ptychovit` and numeric finite metric payload values.
- `stdout.log` does not contain `Skipped backend execution`.
- manifest checks:
  - `mode == "inference"`
  - `training_returncode is null`
  - `checkpoint` path exists and points at `${output_dir}/runs/pinn_ptychovit/best_model.pth` unless override flag explicitly allows external absolute checkpoint.

**Step 4: Run tests to verify GREEN**

Run: `pytest tests/test_verify_fresh_ptychovit_initial_metrics.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/verify_fresh_ptychovit_initial_metrics.py tests/test_verify_fresh_ptychovit_initial_metrics.py
git commit -m "feat: add verifier for fresh checkpoint-restored ptychovit initial baseline"
```

### Task 3: Workflow Documentation for Initial Baseline Runbook

**Files:**
- Modify: `docs/workflows/ptychovit.md`
- Modify: `scripts/studies/README.md`
- Modify: `tests/test_docs_ptychovit_workflow.py`

**Step 1: Write failing doc tests**

```python
# tests/test_docs_ptychovit_workflow.py

def test_ptychovit_workflow_documents_fresh_initial_metrics_runbook():
    text = Path("docs/workflows/ptychovit.md").read_text()
    assert "Fresh Initial Baseline (Checkpoint-Restored, Lines Synthetic)" in text
    assert "--models pinn_ptychovit" in text
    assert "--model-n pinn_ptychovit=256" in text
    assert "--reuse-existing-recons" in text  # described as optional and disabled by default
```

**Step 2: Run test to verify RED**

Run: `pytest tests/test_docs_ptychovit_workflow.py -k fresh_initial_baseline -v`

Expected: FAIL before docs are updated.

**Step 3: Write minimal docs implementation**

- Add runbook section to `docs/workflows/ptychovit.md` with:
  - clean-output prerequisite
  - checkpoint copy path
  - fresh run command
  - verifier command
  - expected artifact list (metrics/recons/PNGs/logs/manifest)
- Add one-line pointer in `scripts/studies/README.md`.

**Step 4: Run tests to verify GREEN**

Run:
- `pytest tests/test_docs_ptychovit_workflow.py -k fresh_initial_baseline -v`
- `pytest tests/test_docs_ptychovit_workflow.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add docs/workflows/ptychovit.md scripts/studies/README.md tests/test_docs_ptychovit_workflow.py
git commit -m "docs: add fresh checkpoint-restored initial metrics runbook for ptychovit"
```

### Task 4: Execute Fresh Baseline and Archive Evidence

**Files:**
- Modify: `docs/plans/2026-02-10-fresh-checkpoint-lines-initial-metrics.md` (append evidence section)
- Create: `.artifacts/ptychovit_initial_metrics_fresh/README.md` (gitignored pointer)

**Step 1: Run driver command**

Run:

```bash
python scripts/studies/run_fresh_ptychovit_initial_metrics.py \
  --checkpoint <absolute-path-to-best_model.pth> \
  --output-dir tmp/ptychovit_initial_fresh \
  --ptychovit-repo /home/ollie/Documents/ptycho-vit \
  --N 128 --gridsize 1 --nimgs-train 8 --nimgs-test 8 --set-phi
```

Expected:
- fresh run artifacts under `tmp/ptychovit_initial_fresh/`
- no skip-reuse message in `runs/pinn_ptychovit/stdout.log`

**Step 2: Run verifier command**

Run:

```bash
python scripts/studies/verify_fresh_ptychovit_initial_metrics.py \
  --output-dir tmp/ptychovit_initial_fresh
```

Expected: exit 0 with explicit PASS summary.

**Step 3: Archive logs**

Run:
- `mkdir -p .artifacts/ptychovit_initial_metrics_fresh`
- copy command logs + verifier output + key JSON paths into artifact index README.

Expected: reproducible evidence inventory with durable paths.

**Step 4: Run targeted regression tests**

Run:
- `pytest tests/test_ptychovit_bridge_entrypoint.py -v`
- `pytest tests/test_grid_lines_compare_wrapper.py -v`
- `pytest tests/test_run_fresh_ptychovit_initial_metrics.py -v`
- `pytest tests/test_verify_fresh_ptychovit_initial_metrics.py -v`
- `pytest tests/test_docs_ptychovit_workflow.py -v`

Expected: PASS.

**Step 5: Commit evidence update**

```bash
git add docs/plans/2026-02-10-fresh-checkpoint-lines-initial-metrics.md
git commit -m "chore: record fresh checkpoint-restored ptychovit initial baseline evidence"
```

## Expected Outputs

- `tmp/ptychovit_initial_fresh/metrics_by_model.json`
- `tmp/ptychovit_initial_fresh/metrics.json`
- `tmp/ptychovit_initial_fresh/recons/gt/recon.npz`
- `tmp/ptychovit_initial_fresh/recons/pinn_ptychovit/recon.npz`
- `tmp/ptychovit_initial_fresh/visuals/amp_phase_pinn_ptychovit.png`
- `tmp/ptychovit_initial_fresh/visuals/compare_amp_phase.png`
- `tmp/ptychovit_initial_fresh/runs/pinn_ptychovit/manifest.json`
- `.artifacts/ptychovit_initial_metrics_fresh/README.md`

## Non-Goals

- No new support for `pinn_ptychovit` at `N != 256`.
- No modifications to upstream `ptycho-vit` internals.
- No changes to physical-unit harmonization policy (pixel-space canonicalization remains).
