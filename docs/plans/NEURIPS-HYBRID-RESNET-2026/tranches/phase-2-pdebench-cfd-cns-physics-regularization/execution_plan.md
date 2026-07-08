# PDEBench 2D CFD CNS Physics Regularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable physics-loss framework with a `2d_cfd_cns` backend that supports positivity, continuity residual, and global mass regularization in the shared PDEBench CNS training path.

**Architecture:** Keep model architectures unchanged. Add a generic physics-regularizer builder and CNS backend under `scripts/studies/pdebench_image128/`, extend CNS metadata with grid/time spacing, wire the shared runner to combine supervised and physics losses, and prove behavior with focused synthetic tests plus a small readiness smoke.

**Tech Stack:** Python 3.11 via PATH `python`, PyTorch, h5py, pytest, existing PDEBench image-suite study code.

**Execution Status (2026-04-21):** Implemented in-session. Focused verification passed and the bounded enabled smoke completed cleanly. Narrow git commits were intentionally deferred because the workspace already contains unrelated user changes. The enabled smoke used `pos=1.0`, `cont=0.5`, and `mass=0.25` after the first pass showed continuity dominating the raw physics total.

---

### Task 1: Persist The Approved Design Inputs

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_physics_regularization_design.md`
- Modify: `docs/index.md`

- [ ] **Step 1: Re-read the CNS design/source docs before editing**

Run: `sed -n '1,220p' docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
Expected: existing CNS adapter contract and benchmark boundaries are visible.

- [ ] **Step 2: Save the approved design document**

Write the agreed design to `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_physics_regularization_design.md`.

- [ ] **Step 3: Index the new design doc**

Add a short entry in `docs/index.md` so the design is discoverable from the documentation hub.

- [ ] **Step 4: Sanity-check the new markdown files**

Run: `python - <<'PY'\nfrom pathlib import Path\nfor p in [Path('docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_physics_regularization_design.md'), Path('docs/index.md')]:\n    assert p.exists(), p\nprint('ok')\nPY`
Expected: `ok`

- [ ] **Step 5: Commit the design-doc-only change if the workspace allows a narrow partial commit**

Run:
```bash
git add docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_physics_regularization_design.md docs/index.md
git commit -m "docs: add CNS physics regularization design"
```
Expected: a narrow docs-only commit, or a documented note in the final summary if a partial commit is intentionally deferred because of workspace state.

### Task 2: Extend CNS Metadata With Physical Spacing And Boundary Info

**Files:**
- Modify: `scripts/studies/pdebench_image128/data.py`
- Test: `tests/studies/test_pdebench_cfd_cns_data.py`

- [ ] **Step 1: Write a failing metadata test**

Add a test that expects `inspect_cfd_cns_hdf5(...)` to report:
- `dx`
- `dy`
- `dt`
- `boundary_condition == "periodic"`
- `eta`
- `zeta`

- [ ] **Step 2: Run the new test to confirm failure**

Run: `pytest -q tests/studies/test_pdebench_cfd_cns_data.py -k spacing`
Expected: FAIL because the metadata keys are not yet present.

- [ ] **Step 3: Implement metadata extraction**

Update `inspect_cfd_cns_hdf5(...)` to read:
- `x-coordinate`
- `y-coordinate`
- `t-coordinate`
- root attrs `eta`, `zeta`

Derive:
- `dx` from consecutive x-centers
- `dy` from consecutive y-centers
- `dt` from consecutive saved times
- `boundary_condition = "periodic"`

- [ ] **Step 4: Re-run the focused data test**

Run: `pytest -q tests/studies/test_pdebench_cfd_cns_data.py`
Expected: PASS

- [ ] **Step 5: Commit the metadata step**

```bash
git add scripts/studies/pdebench_image128/data.py tests/studies/test_pdebench_cfd_cns_data.py
git commit -m "feat: expose CNS spacing metadata"
```

### Task 3: Add Reusable Physics-Loss Helpers And CNS Backend

**Files:**
- Create: `scripts/studies/pdebench_image128/physics_losses.py`
- Test: `tests/studies/test_pdebench_physics_losses.py`

- [ ] **Step 1: Write failing unit tests for the new module**

Add tests covering:
- periodic central difference on a constant tensor gives zero
- positivity loss is zero for positive `rho/p` and positive for negative entries
- continuity residual is near zero for a constant stationary field
- global mass penalty is zero when mass matches and positive when density is shifted

- [ ] **Step 2: Run the new unit tests to confirm failure**

Run: `pytest -q tests/studies/test_pdebench_physics_losses.py`
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the reusable framework**

Create `scripts/studies/pdebench_image128/physics_losses.py` with:
- small config/result dataclasses or plain structured payloads
- `build_physics_regularizer(task_id, metadata, state_stats, config)`
- explicit fail-closed behavior for unsupported tasks
- periodic derivative helpers using `torch.roll`
- CNS backend implementing denormalized positivity, continuity, and global-mass terms

- [ ] **Step 4: Re-run the physics-loss unit tests**

Run: `pytest -q tests/studies/test_pdebench_physics_losses.py`
Expected: PASS

- [ ] **Step 5: Commit the framework**

```bash
git add scripts/studies/pdebench_image128/physics_losses.py tests/studies/test_pdebench_physics_losses.py
git commit -m "feat: add reusable CNS physics loss framework"
```

### Task 4: Wire Physics Regularization Into The CNS Runner

**Files:**
- Modify: `scripts/studies/pdebench_image128/cfd_cns.py`
- Modify: `scripts/studies/run_pdebench_image128_suite.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Add runner tests for config plumbing**

Write tests that cover:
- physics regularization defaults to off
- physics config is accepted for `2d_cfd_cns`
- unsupported task requests fail closed
- metrics/provenance payload includes the new fields when enabled

- [ ] **Step 2: Run the targeted runner tests to confirm failure**

Run: `pytest -q tests/studies/test_pdebench_image128_runner.py -k physics`
Expected: FAIL because the CLI and payload fields are not yet wired.

- [ ] **Step 3: Add CLI/config plumbing**

Implement a narrow config surface, for example:
- `--physics-regularization`
- `--physics-loss-weights`

Make sure default behavior remains unchanged.

- [ ] **Step 4: Integrate the regularizer into the shared CNS training loop**

Inside `_run_profile(...)`:
- build the regularizer once
- compute `pred_norm = model(x)`
- compute supervised loss and physics loss separately
- optimize on `supervised + physics_total`
- log supervised and per-term regularization values each epoch
- persist provenance fields in the per-profile metrics payload

- [ ] **Step 5: Re-run the targeted runner tests**

Run: `pytest -q tests/studies/test_pdebench_image128_runner.py -k physics`
Expected: PASS

- [ ] **Step 6: Commit the runner wiring**

```bash
git add scripts/studies/pdebench_image128/cfd_cns.py scripts/studies/run_pdebench_image128_suite.py tests/studies/test_pdebench_image128_runner.py
git commit -m "feat: wire physics regularization into CNS runner"
```

### Task 5: Verify Off-Path Compatibility And Update Durable Docs

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Optional: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`

- [ ] **Step 1: Run the full focused CNS test selector**

Run:
```bash
pytest -q \
  tests/studies/test_pdebench_cfd_cns_data.py \
  tests/studies/test_pdebench_cfd_cns_metrics.py \
  tests/studies/test_pdebench_physics_losses.py \
  tests/studies/test_pdebench_image128_runner.py
```
Expected: all selected tests pass.

- [ ] **Step 2: Run compile verification**

Run: `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
Expected: exit `0`

- [ ] **Step 3: Update the CNS summary doc**

Record:
- design/implementation scope
- default-off behavior
- enabled term bundle
- key provenance fields
- verification evidence

- [ ] **Step 4: Update any required index/ledger entries**

Only add narrow references that point to the new design/summary and do not churn unrelated roadmap text.

- [ ] **Step 5: Commit the verification/doc step**

```bash
git add docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md docs/index.md state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json
git commit -m "docs: record CNS physics regularization support"
```

### Task 6: Run A Small Physics-Regularized CNS Smoke And Inspect Artifacts

**Files:**
- Write: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/<new-run-root>/...`

- [ ] **Step 1: Launch a bounded readiness smoke with physics regularization enabled**

Run:
```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-physics-smoke \
  --profiles hybrid_resnet_base \
  --history-len 2 \
  --epochs 1 \
  --batch-size 4 \
  --max-train-trajectories 8 \
  --max-val-trajectories 2 \
  --max-test-trajectories 2 \
  --max-windows-per-trajectory 2 \
  --device cuda \
  --num-workers 0 \
  --physics-regularization on \
  --physics-loss-weights pos=1.0,cont=1.0,mass=1.0
```
Expected: exit `0` and a fresh artifact root with per-profile metrics JSON.

- [ ] **Step 2: Inspect the emitted metrics payload**

Check that the metrics JSON contains:
- `physics_regularization_enabled`
- `physics_loss_terms`
- `physics_loss_weights`
- per-epoch or final regularization summaries

- [ ] **Step 3: Inspect the extended metadata**

Check that `hdf5_metadata.json` or equivalent manifest now records:
- `dx`
- `dy`
- `dt`
- `boundary_condition`
- `eta`
- `zeta`

- [ ] **Step 4: Record the smoke outcome in the CNS summary**

Add the run root and whether the enabled term bundle completed cleanly.

- [ ] **Step 5: Commit the smoke evidence references if the summary changed**

```bash
git add docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md
git commit -m "docs: record CNS physics regularization smoke"
```

### Task 7: Final Review And Handoff

**Files:**
- Modify only if needed after review

- [ ] **Step 1: Re-run the final focused verification**

Run:
```bash
pytest -q \
  tests/studies/test_pdebench_cfd_cns_data.py \
  tests/studies/test_pdebench_cfd_cns_metrics.py \
  tests/studies/test_pdebench_physics_losses.py \
  tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```
Expected: all green.

- [ ] **Step 2: Review the changed files for scope discipline**

Run: `git diff -- scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py tests/studies docs/plans/NEURIPS-HYBRID-RESNET-2026 docs/index.md`
Expected: only the intended framework, runner, tests, and docs changed.

- [ ] **Step 3: Summarize the delivered contract**

Prepare final notes covering:
- what was implemented
- what is still intentionally unsupported
- verification evidence
- artifact paths for the enabled smoke run

- [ ] **Step 4: Inline execution selected**

The user explicitly requested planning followed by execution in this session, so execute tasks inline rather than asking for a separate execution mode choice.
