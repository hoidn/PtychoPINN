# Known-Probe Object-Domain HIO/ER Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create a worktree for this initiative.

**Goal:** Add a reviewer-facing known-probe, object-domain HIO/ER baseline for the non-ML single-shot CDI benchmark.

**Architecture:** Keep `scripts/reconstruction/hio_cdi_benchmark.py` as the study entry point and add a second solver path whose unknown is the object patch `O`, not the exit wave `psi`. The current PyNX CDI path remains available and clearly labeled as support-constrained exit-wave CDI; the new path owns the fixed-probe projection `O -> P * O -> FFT -> detector-amplitude projection -> least-squares projection back to O`.

**Tech Stack:** Python, NumPy, pytest, existing Table 2 same-split data-bundle machinery, existing `ptycho.workflows.grid_lines_workflow.stitch_predictions`, and existing `ptycho.evaluation.eval_reconstruction`.

---

## Initiative

- ID: `non-ml-single-shot-cdi-known-probe-hio-er`
- Status: pending
- Source design: `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- Prior active solver plan: `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-pynx-replacement-plan.md`
- Current implementation file: `scripts/reconstruction/hio_cdi_benchmark.py`
- Current primary tests: `tests/scripts/test_hio_cdi_benchmark.py`
- Paper checklist: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- Reviewer scope: Reviewer 3 asked for a standard non-ML single-shot phase retrieval comparator, naming ER/HIO with strict support constraints or ADMM.

## Compliance Matrix

- [ ] **Reviewer contract:** The new main candidate row must stay single-frame/single-shot and must not use multi-position ptychographic overlap, position refinement, Tike, PtyChi, or PyNX ptychography.
- [ ] **Known-probe contract:** The probe is fixed and used inside the iterative forward model, not only for support construction and post-hoc division.
- [ ] **Data contract:** Reuse the same frozen/same-split bundle mechanism already used by the non-ML benchmark; do not compare against historical Table 2 values as same-data unless a frozen artifact branch proves exact identity.
- [ ] **Metric contract:** Use the existing direct-stitch and `eval_reconstruction(...)` contract unless a separate approved plan resolves the paper-side alignment/subsample notes differently.
- [ ] **Ambiguity policy:** Select restarts by ground-truth-free known-probe Fourier-amplitude residual only. Do not use amplitude SSIM, PSNR, phase metrics, shift/twin/orientation search, or visual agreement for the main row.
- [ ] **Support policy:** Keep the primary `support_threshold=0.05` and sensitivity thresholds `[0.01, 0.05, 0.10]` pre-registered. Threshold choice must not be metric-selected.
- [ ] **Finding ID:** `GRIDLINES-PROBE-PIPELINE-001` - preserve normalized probe-transform provenance such as `smooth:0.5|pad:64`.
- [ ] **Finding ID:** `STITCH-GRIDSIZE-001` - keep the gridsize-1-safe `stitch_predictions(...)` path.
- [ ] **Finding ID:** `NORMALIZATION-001` - keep detector amplitude, object scale, and evaluation normalization separate.
- [ ] **Project policy:** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- [ ] **Project policy:** Use PATH `python`; use tmux and tracked PIDs for long-running benchmark commands.

## File Structure

- Modify: `scripts/reconstruction/hio_cdi_benchmark.py`
  - Keep:
    - PyNX CDI adapter and manifests for the existing exit-wave baseline.
    - support construction, same-split data-bundle loading, stitching, metrics, and artifact writing.
    - `RestartResult` / `RestartRun` or add parallel object-domain result dataclasses only if the existing shape becomes misleading.
  - Add:
    - `KNOWN_PROBE_SOLVER = "known_probe_object_hio_er"`
    - `known_probe_forward_amplitude(obj, probe)`
    - `project_known_probe_fourier_magnitude(obj, probe, target_magnitude, epsilon_ratio=...)`
    - `known_probe_fourier_residual(obj, probe, target_magnitude)`
    - `known_probe_er_cleanup(previous_obj, probe, target_magnitude, object_support, ...)`
    - `known_probe_hio_update(previous_obj, probe, target_magnitude, object_support, beta, ...)`
    - `run_known_probe_restarts(...)`
  - Modify:
    - CLI parsing to support `--solver {pynx_cdi_hio_er,known_probe_object_hio_er}`.
    - `write_solver_manifest(...)` to record both candidates and the selected solver.
    - benchmark dispatch so `--solver known_probe_object_hio_er` runs the object-domain loop and writes row labels that include `known_probe`.

- Modify: `tests/scripts/test_hio_cdi_benchmark.py`
  - Add focused object-domain unit tests beside existing exit-wave/PyNX tests.
  - Keep existing PyNX tests intact; only adjust manifest assertions to account for an explicit selected solver argument.

- Do not modify stable core physics/model files.

## Context Priming

Read before edits:

- `docs/index.md`
- `docs/findings.md`
- `docs/TESTING_GUIDE.md`
- `docs/development/TEST_SUITE_INDEX.md`
- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-execution-plan.md`
- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-pynx-replacement-plan.md`
- `scripts/reconstruction/hio_cdi_benchmark.py`
- `tests/scripts/test_hio_cdi_benchmark.py`

## Proposed Object-Domain Projection

Use the same normalized FFT convention as `forward_amplitude(...)`:

```python
def known_probe_forward_amplitude(obj: np.ndarray, probe: np.ndarray) -> np.ndarray:
    obj_array = _validate_2d_complex("obj", obj)
    probe_array = _validate_2d_complex("probe", probe)
    if obj_array.shape != probe_array.shape:
        raise ValueError("obj and probe shapes must match")
    exit_wave = probe_array * obj_array
    norm = math.sqrt(float(exit_wave.size))
    return np.abs(np.fft.fftshift(np.fft.fft2(exit_wave)) / norm)
```

Project the detector amplitude through the fixed probe and back to object space:

```python
def project_known_probe_fourier_magnitude(
    obj: np.ndarray,
    probe: np.ndarray,
    target_magnitude: np.ndarray,
    epsilon_ratio: float = DEFAULT_EPSILON_RATIO,
) -> np.ndarray:
    obj_array = _validate_2d_complex("obj", obj)
    probe_array = _validate_2d_complex("probe", probe)
    target = _validate_target_magnitude(target_magnitude, obj_array.shape)
    if probe_array.shape != obj_array.shape:
        raise ValueError("obj and probe shapes must match")

    norm = math.sqrt(float(obj_array.size))
    exit_wave = probe_array * obj_array
    current_shifted = np.fft.fftshift(np.fft.fft2(exit_wave)) / norm
    projected_shifted = target * np.exp(1j * np.angle(current_shifted))
    projected_exit_wave = np.fft.ifft2(np.fft.ifftshift(projected_shifted * norm))

    max_amp = float(np.abs(probe_array).max(initial=0.0))
    if max_amp <= 0:
        raise ValueError("zero-amplitude probe cannot be used")
    denom = np.square(np.abs(probe_array)) + (float(epsilon_ratio) * max_amp) ** 2
    return obj_array + (np.conj(probe_array) / denom) * (projected_exit_wave - exit_wave)
```

This is a least-squares projection back to `O` under the fixed diagonal probe. It is the key difference from the current exit-wave path.

## Task 1: Add Object-Domain Projection Tests

**Files:**
- Modify: `tests/scripts/test_hio_cdi_benchmark.py`

- [ ] **Step 1: Write a failing forward-model test**

Add:

```python
def test_known_probe_forward_amplitude_uses_probe_inside_forward_model():
    from scripts.reconstruction.hio_cdi_benchmark import known_probe_forward_amplitude

    obj = _toy_complex()
    probe = _toy_probe()
    expected = np.abs(np.fft.fftshift(np.fft.fft2(probe * obj)) / 8.0)

    assert np.allclose(known_probe_forward_amplitude(obj, probe), expected)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py::test_known_probe_forward_amplitude_uses_probe_inside_forward_model -q
```

Expected: fail because `known_probe_forward_amplitude` does not exist.

- [ ] **Step 3: Write a failing amplitude-projection test**

Add:

```python
def test_project_known_probe_fourier_magnitude_matches_target_without_ground_truth():
    from scripts.reconstruction.hio_cdi_benchmark import (
        known_probe_forward_amplitude,
        project_known_probe_fourier_magnitude,
    )

    probe = _toy_probe()
    obj = _toy_complex()
    target = known_probe_forward_amplitude(obj * np.exp(0.2j), probe)

    projected = project_known_probe_fourier_magnitude(obj, probe, target)

    assert np.allclose(known_probe_forward_amplitude(projected, probe), target, atol=1e-5)
```

- [ ] **Step 4: Run the failing projection test**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py::test_project_known_probe_fourier_magnitude_matches_target_without_ground_truth -q
```

Expected: fail because `project_known_probe_fourier_magnitude` does not exist.

- [ ] **Step 5: Commit only if requested**

Do not commit by default. If the user asks to commit after this task:

```bash
git add tests/scripts/test_hio_cdi_benchmark.py
git commit -m "test: cover known-probe object-domain projection"
```

## Task 2: Implement the Known-Probe Projection Helpers

**Files:**
- Modify: `scripts/reconstruction/hio_cdi_benchmark.py`
- Test: `tests/scripts/test_hio_cdi_benchmark.py`

- [ ] **Step 1: Add helper functions**

Implement `known_probe_forward_amplitude(...)`, `project_known_probe_fourier_magnitude(...)`, and `known_probe_fourier_residual(...)` near the existing Fourier helper functions. Reuse `_validate_2d_complex(...)`, `_validate_target_magnitude(...)`, `DEFAULT_EPSILON_RATIO`, and the normalized FFT convention from `forward_amplitude(...)`.

- [ ] **Step 2: Run projection tests**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py -k "known_probe_forward or project_known_probe" -q
```

Expected: pass.

- [ ] **Step 3: Run adjacent Fourier tests**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py -k "forward_amplitude or project_fourier or known_probe" -q
```

Expected: pass, preserving existing exit-wave behavior.

## Task 3: Add Object-Domain ER/HIO Update Tests

**Files:**
- Modify: `tests/scripts/test_hio_cdi_benchmark.py`

- [ ] **Step 1: Write failing ER/HIO update test**

Add:

```python
def test_known_probe_hio_and_er_update_object_not_exit_wave():
    from scripts.reconstruction.hio_cdi_benchmark import (
        known_probe_er_cleanup,
        known_probe_hio_update,
        project_known_probe_fourier_magnitude,
    )

    previous = _toy_complex()
    probe = _toy_probe()
    target = np.ones((8, 8), dtype=np.float32)
    support = np.zeros((8, 8), dtype=bool)
    support[2:6, 2:6] = True

    projected = project_known_probe_fourier_magnitude(previous, probe, target)
    hio = known_probe_hio_update(previous, probe, target, support, beta=0.9)
    er = known_probe_er_cleanup(previous, probe, target, support)

    assert np.allclose(hio[support], projected[support])
    assert np.allclose(hio[~support], previous[~support] - 0.9 * projected[~support])
    assert np.allclose(er[support], projected[support])
    assert np.all(er[~support] == 0)
```

- [ ] **Step 2: Run the failing update test**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py::test_known_probe_hio_and_er_update_object_not_exit_wave -q
```

Expected: fail because the update functions do not exist.

## Task 4: Implement Object-Domain ER/HIO Updates

**Files:**
- Modify: `scripts/reconstruction/hio_cdi_benchmark.py`
- Test: `tests/scripts/test_hio_cdi_benchmark.py`

- [ ] **Step 1: Add update functions**

Implement:

```python
def known_probe_er_cleanup(previous_obj, probe, target_magnitude, object_support, epsilon_ratio=DEFAULT_EPSILON_RATIO):
    projected = project_known_probe_fourier_magnitude(previous_obj, probe, target_magnitude, epsilon_ratio)
    support_mask = np.asarray(object_support, dtype=bool)
    cleaned = np.zeros_like(projected)
    cleaned[support_mask] = projected[support_mask]
    return cleaned

def known_probe_hio_update(previous_obj, probe, target_magnitude, object_support, beta=0.9, epsilon_ratio=DEFAULT_EPSILON_RATIO):
    previous_array = _validate_2d_complex("previous_obj", previous_obj)
    projected = project_known_probe_fourier_magnitude(previous_array, probe, target_magnitude, epsilon_ratio)
    support_mask = np.asarray(object_support, dtype=bool)
    if support_mask.shape != previous_array.shape:
        raise ValueError("object support shape must match previous_obj")
    updated = np.empty_like(projected)
    updated[support_mask] = projected[support_mask]
    updated[~support_mask] = previous_array[~support_mask] - float(beta) * projected[~support_mask]
    return updated
```

- [ ] **Step 2: Run update tests**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py -k "known_probe and (hio or er or project)" -q
```

Expected: pass.

## Task 5: Add Known-Probe Restart Loop Tests

**Files:**
- Modify: `tests/scripts/test_hio_cdi_benchmark.py`

- [ ] **Step 1: Write failing restart test**

Add:

```python
def test_run_known_probe_restarts_selects_by_known_probe_residual():
    from scripts.reconstruction.hio_cdi_benchmark import (
        known_probe_forward_amplitude,
        run_known_probe_restarts,
    )

    probe = _toy_probe()
    true_obj = _toy_complex()
    target = known_probe_forward_amplitude(true_obj, probe)
    support = np.abs(probe) >= 0.05 * np.abs(probe).max()

    run = run_known_probe_restarts(
        target,
        probe,
        support,
        seeds=[11, 12],
        beta=0.9,
        hio_iters=2,
        er_iters=1,
        residual_period=1,
        condition_id="gs1_custom",
        patch_index=0,
    )

    assert run.selected.seed in {run.restarts[0].seed, run.restarts[1].seed}
    assert all(restart.residual_curve for restart in run.restarts)
    assert run.selected.psi.shape == probe.shape
```

Note: reuse `RestartResult.psi` for the selected object if that avoids a larger dataclass refactor; if that name becomes too misleading, add `ObjectRestartResult` in this task and update the test accordingly.

- [ ] **Step 2: Run the failing restart test**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py::test_run_known_probe_restarts_selects_by_known_probe_residual -q
```

Expected: fail because `run_known_probe_restarts` does not exist.

## Task 6: Implement Known-Probe Restart Loop

**Files:**
- Modify: `scripts/reconstruction/hio_cdi_benchmark.py`
- Test: `tests/scripts/test_hio_cdi_benchmark.py`

- [ ] **Step 1: Add deterministic initialization**

Add `_initial_object_from_target(...)` or reuse `_initial_psi(...)` by dividing the initialized exit wave by `P_safe` once. Prefer a direct object initializer:

```python
def _initial_object(probe, target_magnitude, seed, epsilon_ratio=DEFAULT_EPSILON_RATIO):
    initial_exit_wave = _initial_psi(target_magnitude, seed)
    probe_array = _validate_2d_complex("probe", probe)
    max_amp = float(np.abs(probe_array).max(initial=0.0))
    epsilon = float(epsilon_ratio) * max_amp
    denom = np.square(np.abs(probe_array)) + epsilon**2
    return (np.conj(probe_array) / denom) * initial_exit_wave
```

- [ ] **Step 2: Add `run_known_probe_restarts(...)`**

Use the same patch-specific seed derivation as `run_restarts(...)`; run `known_probe_hio_update(...)` for `hio_iters`, then `known_probe_er_cleanup(...)` for `er_iters`; sample residual every `residual_period`; select the lowest final `known_probe_fourier_residual(...)`, tie-breaking by lower seed.

- [ ] **Step 3: Run restart tests**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py -k "known_probe and restart" -q
```

Expected: pass.

## Task 7: Add CLI Solver Routing and Manifest Tests

**Files:**
- Modify: `tests/scripts/test_hio_cdi_benchmark.py`
- Modify: `scripts/reconstruction/hio_cdi_benchmark.py`

- [ ] **Step 1: Add failing parse/manifest tests**

Add tests that assert:

- `parse_args(...)` accepts `--solver known_probe_object_hio_er`.
- `parse_args(...)` defaults to the existing `pynx_cdi_hio_er` solver if no solver is passed, preserving current behavior.
- `write_solver_manifest(..., selected_solver="known_probe_object_hio_er")` marks the known-probe solver as accepted and records that it is repo-local and fixed-probe/object-domain.
- artifact context embeds `selected_solver == "known_probe_object_hio_er"` when selected.

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py -k "solver and known_probe" -q
```

Expected: fail before implementation.

- [ ] **Step 2: Implement CLI and manifest routing**

Add:

```python
EXIT_WAVE_SOLVER = "pynx_cdi_hio_er"
KNOWN_PROBE_SOLVER = "known_probe_object_hio_er"
SELECTED_SOLVER = EXIT_WAVE_SOLVER
```

Add `--solver` choices for those two values. Pass `args.solver` into `write_solver_manifest(...)` and benchmark dispatch.

- [ ] **Step 3: Route benchmark execution**

In `run_smoke_benchmark(...)`, dispatch:

```python
if args.solver == KNOWN_PROBE_SOLVER:
    restart_run = run_known_probe_restarts(...)
    reconstructed = restart_run.selected.psi.astype(np.complex64)
else:
    restart_run = run_restarts(...)
    reconstructed = recover_object_patch(...)
```

Use a row label such as:

```python
row_label = f"{condition_label}_{args.solver}_support_{_threshold_token(args.primary_support_threshold)}"
```

Preserve all residual and metric artifacts.

- [ ] **Step 4: Run routing tests**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py -k "known_probe or solver" -q
```

Expected: pass.

## Task 8: Run Smoke Benchmark on Reused Same-Split Bundle

**Files:**
- No source changes unless the smoke run exposes a bug.
- Artifacts: `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/<run_id>/`

- [ ] **Step 1: Locate a valid same-split data bundle manifest**

Use the most recent successful same-split bundle if it exists, for example:

```bash
find .artifacts/revision_studies/non_ml_single_shot_cdi_benchmark -name data_bundle_manifest.json -print | sort
```

- [ ] **Step 2: Run bounded smoke with known-probe solver**

Use `tmux` for this if expected runtime exceeds a minute. Example command:

```bash
RUN_ID=known_probe_smoke_$(date -u +%Y%m%dT%H%M%SZ)
PTYCHO_DISABLE_MEMOIZE=1 python scripts/reconstruction/hio_cdi_benchmark.py \
  --output-root .artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/${RUN_ID} \
  --run-id ${RUN_ID} \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-source custom \
  --probe-scale-mode pad_preserve \
  --probe-smoothing-sigma 0.5 \
  --support-thresholds 0.01 0.05 0.10 \
  --primary-support-threshold 0.05 \
  --restart-seeds 2026041201 2026041202 2026041203 \
  --beta 0.9 \
  --hio-iters 20 \
  --er-iters 5 \
  --residual-period 5 \
  --max-test-frames 4 \
  --data-identity-branch same-split-rerun \
  --metric-contract-mode direct-stitch \
  --reuse-data-bundle-manifest <path/to/data_bundle_manifest.json> \
  --solver known_probe_object_hio_er \
  --smoke
```

Expected: command exits `0`, manifest exists, metrics JSON has `eval_status` `ok` or a clear actionable failure, and residual JSON records selected restarts without ground-truth selection.

- [ ] **Step 3: Archive smoke evidence in the plan note**

Append the run root, metrics path, and any blocker to this plan under "Execution Notes" or in a follow-up report under `.artifacts/work/revision-studies/`.

## Task 9: Run Targeted Verification

**Files:**
- Source/test files touched above.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py -q
```

Expected: pass.

- [ ] **Step 2: Run compile check**

Run:

```bash
python -m py_compile scripts/reconstruction/hio_cdi_benchmark.py
```

Expected: pass with no output.

- [ ] **Step 3: Run collection for the changed test module**

Run:

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py --collect-only -q
```

Expected: collection succeeds and includes the new known-probe tests.

## Task 10: Decide Reporting Status

**Files:**
- Modify only after evidence is accepted:
  - `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
  - paper data/table/manuscript files as directed by the active revision workflow

- [ ] **Step 1: Compare the known-probe row against the PyNX exit-wave row**

Use the metrics and residual payloads to decide whether the known-probe row is reviewer-facing, diagnostic-only, or blocked.

- [ ] **Step 2: Apply the reporting rule**

Use:

- reviewer-facing if the run uses the same-split/frozen data contract, uses no ground-truth selection, and has credible finite metrics;
- diagnostic-only if it still fails due to CDI ambiguities but the forward residual is informative;
- blocked/pivot if the object-domain algorithm cannot be made numerically coherent without adding oracle priors.

- [ ] **Step 3: Update paper-facing checklist only after the decision**

Do not update paper claims until the metrics and artifact contracts are reviewed.

## Verification Commands

```bash
python -m pytest tests/scripts/test_hio_cdi_benchmark.py -k "known_probe" -q
python -m pytest tests/scripts/test_hio_cdi_benchmark.py -q
python -m pytest tests/scripts/test_hio_cdi_benchmark.py --collect-only -q
python -m py_compile scripts/reconstruction/hio_cdi_benchmark.py
```

## Completion Criteria

- [x] `known_probe_object_hio_er` is available as an explicit solver mode and does not replace or blur the existing `pynx_cdi_hio_er` row.
- [x] The known-probe solver uses `probe * object` inside each Fourier projection.
- [x] Restart selection uses known-probe Fourier residual only, with no ground-truth metric selection.
- [x] Output manifests and row labels distinguish known-probe object-domain HIO/ER from PyNX exit-wave CDI.
- [x] A bounded smoke run against the same-split bundle succeeds or produces an actionable blocker.
- [x] Focused tests and compile checks pass.

## Artifacts Index

- Plan: `docs/plans/revision-studies/non-ml-single-shot-cdi-known-probe-hio-er-plan.md`
- Benchmark artifact root: `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/`
- Existing PyNX replacement report: `.artifacts/work/revision-studies/non-ml-single-shot-cdi-benchmark-execution-report.md`
- Paper checklist: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`

## Execution Notes

- 2026-04-13: Implemented the known-probe object-domain solver mode in `scripts/reconstruction/hio_cdi_benchmark.py` and added focused tests in `tests/scripts/test_hio_cdi_benchmark.py`.
- Verification:
  - `python -m pytest tests/scripts/test_hio_cdi_benchmark.py -k "known_probe" -q` -> 7 passed, 37 deselected.
  - `python -m py_compile scripts/reconstruction/hio_cdi_benchmark.py` -> passed.
  - `python -m pytest tests/scripts/test_hio_cdi_benchmark.py --collect-only -q` -> 44 tests collected.
  - `python -m pytest tests/scripts/test_hio_cdi_benchmark.py -q` -> 44 passed.
- Smoke run:
  - First tmux attempt, `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/known_probe_smoke_20260413T072325Z`, failed before support/data reuse because tmux started in the base conda environment and `tensorflow` was unavailable.
  - Successful tmux rerun used `ptycho311`, PID `307111`, and exited `0`.
  - Output root: `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/known_probe_smoke_20260413T072350Z`.
  - Command log: `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/known_probe_smoke_20260413T072350Z/command.log`.
  - Metrics path: `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/known_probe_smoke_20260413T072350Z/metrics_gs1_custom_known_probe_object_hio_er_support_0p05.json`.
  - Residual path: `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/known_probe_smoke_20260413T072350Z/residuals_gs1_custom_known_probe_object_hio_er_support_0p05.json`.
  - Recon path: `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/known_probe_smoke_20260413T072350Z/recons/gs1_custom_known_probe_object_hio_er_support_0p05/recon.npz`.
  - Smoke selected solver: `known_probe_object_hio_er`; solver manifest marks this candidate accepted and marks `pynx_cdi_hio_er` not accepted for the smoke row.
  - Smoke residual payload recorded 2 patches. First patch selected final residual was `0.15846437254189952` with selected curve `[2.2679662709276475e-08, 0.46271156801017627, 0.4627799090285935, 0.4454860556741642, 0.43677658476509834, 0.15846437254189952]`.
  - Metrics JSON has `eval_status: failed` with `ValueError('stitched reconstruction spatial shape (10, 10) does not match stored YY_ground_truth shape (330, 330); run without --max-test-frames for reviewer-facing metrics')`. This is an expected bounded-smoke limitation, not a solver crash. The result is diagnostic only until a full-grid run, or a separately approved metric contract, produces reviewer-facing metrics.
