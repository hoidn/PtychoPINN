# Replace Study-Local HIO/ER With PyNX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create a worktree for this initiative.

**Goal:** Replace the production study-local HIO/ER implementation in the non-ML single-shot CDI benchmark with a PyNX-backed CDI solver, while preserving the existing Table 2 data, metric, support, restart, and artifact contracts.

**Architecture:** Keep `scripts/reconstruction/hio_cdi_benchmark.py` as the revision-study entry point, but move phase retrieval behind a narrow PyNX adapter. The adapter owns PyNX import/preflight, array convention conversion, HIO/ER operator execution, processing-unit cleanup, residual sampling, and unavailable-dependency errors; the surrounding script continues to own data identity, metric contracts, support construction, patch recovery, stitching, and manifest writing.

**Tech Stack:** Python, NumPy, pytest, PyNX `pynx.cdi.CDI`, `HIO`, `ER`, and `FreePU`, existing PtychoPINN grid-lines workflow helpers, and existing `ptycho.evaluation.eval_reconstruction`.

---

## Initiative

- ID: `non-ml-single-shot-cdi-benchmark-pynx-replacement`
- Status: implemented for the PyNX adapter path; later known-probe diagnostic work coexists as an explicit secondary solver mode
- Supersedes: `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-execution-plan.md` solver implementation sections that select `study_local_hio_er`
- Source design: `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- Primary implementation file: `scripts/reconstruction/hio_cdi_benchmark.py`
- Primary tests: `tests/scripts/test_hio_cdi_benchmark.py`
- External reference: PyNX CDI docs, `https://pynx.esrf.fr/en/latest/modules/cdi/index.html`
- External install reference: PyNX install docs, `https://pynx.esrf.fr/en/latest/install.html`

## Compliance Matrix

- [ ] **User directive:** The reviewer-facing HIO/ER production path uses PyNX, not the custom `hio_update` / `er_cleanup` loop.
- [ ] **Design contract:** Keep support threshold policy, restart seeds, final Fourier-amplitude residual selection, and direct support-anchored main-row ambiguity policy unchanged.
- [ ] **PyNX contract:** PyNX `CDI` consumes observed intensity, not amplitude; square the stored normalized amplitude before passing it to PyNX.
- [ ] **PyNX contract:** PyNX CDI arrays are documented as centered at `(0,0)` to avoid `fftshift`; explicitly convert the repo's centered diffraction/support arrays to and from the PyNX convention.
- [ ] **PyNX contract:** PyNX can treat special observed-intensity values as masks/free pixels. Preflight must prove the normalized Table 2 intensity array is not being misinterpreted, or apply a recorded scale factor and undo it on the returned exit wave before object recovery.
- [ ] **Artifact contract:** `solver_manifest.json` records PyNX as the selected solver, installed version, import path, operator names, API preflight result, and any intensity scale/convention transforms.
- [ ] **Artifact contract:** The metrics payload still records `hio_hyperparameters`, `support_policy`, `ambiguity_policy`, `metric_ground_truth`, residual JSON path, and selected restart information.
- [ ] **Finding ID:** `NORMALIZATION-001` - keep physics, statistical, and display/evaluation normalization separate.
- [ ] **Finding ID:** `STITCH-GRIDSIZE-001` - keep using the gridsize-1-safe `stitch_predictions(...)` path.
- [ ] **Project policy:** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- [ ] **Project policy:** Use PATH `python`; use tmux and tracked PIDs for long-running PyNX benchmark runs.

## File Structure

- Modify: `scripts/reconstruction/hio_cdi_benchmark.py`
  - Keep:
    - `RestartResult`, `RestartRun`
    - support construction
    - `_frame_amplitude`
    - object-patch recovery
    - self-consistency diagnostics
    - artifact/manifest writers
    - `run_smoke_benchmark(...)` outer loop
  - Add:
    - `PynxUnavailableError`
    - `PynxCdiAdapter`
    - `check_pynx_available()`
    - `to_pynx_intensity(...)`
    - `to_pynx_realspace(...)`
    - `from_pynx_realspace(...)`
    - `run_pynx_restart(...)`
  - Replace:
    - `run_restarts(...)` body so it calls `PynxCdiAdapter` for each restart.
    - `SELECTED_SOLVER = "study_local_hio_er"` with `SELECTED_SOLVER = "pynx_cdi_hio_er"`.
    - `write_solver_manifest(...)` selected-solver and candidate records so PyNX is accepted when import/preflight succeeds.
  - Remove or demote to test-only/private diagnostics:
    - `project_fourier_magnitude`
    - `hio_update`
    - `er_cleanup`
    - `_initial_psi`
    - any tests that assert the custom HIO/ER recurrence as production behavior.
  - Preserve separately named known-probe object-domain helpers only when selected by `--solver known_probe_object_hio_er`; these are not the old `study_local_hio_er` production recurrence and must remain labeled as a secondary candidate/diagnostic row.

- Modify: `tests/scripts/test_hio_cdi_benchmark.py`
  - Keep support, metric-contract, data-identity, same-split, and manifest tests.
  - Replace custom recurrence tests with PyNX adapter boundary tests using a fake `pynx.cdi` module.
  - Add real PyNX integration tests guarded by `pytest.importorskip("pynx.cdi")`.

- Do not modify stable core physics/model files.

## Context Priming

Read before edits:

- `docs/index.md`
- `docs/findings.md`
- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-execution-plan.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `scripts/reconstruction/hio_cdi_benchmark.py`
- `tests/scripts/test_hio_cdi_benchmark.py`
- PyNX CDI API docs: `https://pynx.esrf.fr/en/latest/modules/cdi/index.html`
- PyNX install docs: `https://pynx.esrf.fr/en/latest/install.html`

Relevant PyNX facts to preserve in code comments/manifests:

- `pynx.cdi.CDI` is the CDI reconstruction class.
- The constructor takes observed diffraction intensity, support, and optional initial object.
- PyNX CDI arrays are documented as centered at `(0,0)` to avoid `fftshift`.
- `get_obj(shift=True)` returns object data shifted with the center in the array center.
- `HIO(beta=..., nb_cycle=...)` and `ER(nb_cycle=...)` are PyNX CDI operators.
- `FreePU()` retrieves latest object/support data from GPU/OpenCL memory and frees the processing unit.

## Task 1: Write PyNX Adapter Boundary Tests

**Files:**
- Modify: `tests/scripts/test_hio_cdi_benchmark.py`

- [ ] **Step 1: Add a fake PyNX module fixture**

Add a fixture that inserts fake `pynx` and `pynx.cdi` modules into `sys.modules`. The fake `CDI` should record `iobs`, `support`, and `obj`; fake `HIO` and `ER` should record cycle counts and return the CDI object from `__mul__`; fake `FreePU` should record that cleanup happened.

- [ ] **Step 2: Add a failing test for intensity and convention conversion**

The test should assert that a centered amplitude frame is converted to PyNX observed intensity by squaring and `np.fft.ifftshift(...)`, and that support/initial object are passed in PyNX real-space convention.

Run:

```bash
pytest tests/scripts/test_hio_cdi_benchmark.py -k "pynx and convention" -vv
```

Expected before implementation: fail because the adapter functions do not exist.

- [ ] **Step 3: Add a failing test for HIO then ER operator use**

The test should call `run_restarts(...)` with `hio_iters=20`, `er_iters=10`, and `residual_period=10`, then assert that the fake PyNX operators were called in HIO then ER order with the requested cycle counts.

Run:

```bash
pytest tests/scripts/test_hio_cdi_benchmark.py -k "pynx and operators" -vv
```

Expected before implementation: fail because `run_restarts(...)` still uses the custom recurrence.

- [ ] **Step 4: Add a failing test for missing PyNX**

The test should clear fake PyNX modules and assert that `check_pynx_available()` returns a structured unavailable result, and that benchmark execution raises an actionable `PynxUnavailableError` rather than silently falling back to the custom solver.

Run:

```bash
pytest tests/scripts/test_hio_cdi_benchmark.py -k "pynx and unavailable" -vv
```

Expected before implementation: fail because no PyNX availability boundary exists.

## Task 2: Implement the PyNX Adapter

**Files:**
- Modify: `scripts/reconstruction/hio_cdi_benchmark.py`

- [ ] **Step 1: Add availability and conversion helpers**

Implement:

```python
class PynxUnavailableError(RuntimeError):
    pass

def check_pynx_available() -> dict[str, object]:
    try:
        import importlib.metadata
        from pynx.cdi import CDI, ER, FreePU, HIO
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    try:
        version = importlib.metadata.version("pynx")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "available": True,
        "version": version,
        "api": {
            "CDI": f"{CDI.__module__}.{CDI.__name__}",
            "HIO": f"{HIO.__module__}.{HIO.__name__}",
            "ER": f"{ER.__module__}.{ER.__name__}",
            "FreePU": f"{FreePU.__module__}.{FreePU.__name__}",
        },
    }
```

Implement `to_pynx_intensity(target_magnitude, scale_factor=1.0)` as:

```python
target = _validate_target_magnitude(target_magnitude, target_magnitude.shape)
iobs = np.fft.ifftshift(np.square(target) * float(scale_factor)).astype(np.float32)
```

Implement real-space conversion helpers with explicit names so reviewer/debug reports can state which convention was used:

```python
def to_pynx_realspace(array):
    return np.fft.ifftshift(np.asarray(array), axes=(-2, -1))

def from_pynx_realspace(array):
    return np.fft.fftshift(np.asarray(array), axes=(-2, -1))
```

- [ ] **Step 2: Add the PyNX adapter class**

Implement a small class so tests can inject fake PyNX operators without importing PyNX at module import time:

```python
class PynxCdiAdapter:
    def __init__(self, *, intensity_scale_factor: float = 1.0):
        self.intensity_scale_factor = float(intensity_scale_factor)

    def _imports(self):
        try:
            from pynx.cdi import CDI, ER, FreePU, HIO
        except Exception as exc:
            raise PynxUnavailableError(
                "PyNX is required for the non-ML CDI benchmark; install and verify pynx.cdi before running HIO/ER metrics"
            ) from exc
        return CDI, HIO, ER, FreePU
```

- [ ] **Step 3: Replace per-restart phase retrieval**

Implement `run_pynx_restart(...)` to:

1. derive the patch-specific seed exactly as the old `run_restarts(...)` did;
2. build initial object from the stored amplitude and deterministic random phase in repo-centered convention;
3. convert `iobs`, support, and initial object to PyNX convention;
4. instantiate `CDI(iobs=..., support=..., obj=..., mask=np.zeros_like(iobs, dtype=np.int8))`;
5. run HIO and ER in chunks of `residual_period` so residual curves stay comparable;
6. retrieve the object using `get_obj(shift=True)` when available, or `from_pynx_realspace(cdi.get_obj())` if needed after preflight proves that convention;
7. divide the returned exit wave by `sqrt(intensity_scale_factor)` if a non-1 scale factor was applied to avoid PyNX free-pixel sentinel behavior;
8. apply `FreePU() * cdi` in `finally` when the operator is available.

- [ ] **Step 4: Keep residual diagnostics, not custom solver math**

Keep `fourier_residual(...)` as a diagnostic against the repo's normalized amplitude convention. Remove production use of custom `hio_update(...)` and `er_cleanup(...)`.

Run:

```bash
pytest tests/scripts/test_hio_cdi_benchmark.py -k "pynx or restart or residual" -vv
```

Expected: the new fake-PyNX tests pass; unrelated custom recurrence tests should be removed or rewritten.

## Task 3: Update Solver Manifest and CLI Preflight

**Files:**
- Modify: `scripts/reconstruction/hio_cdi_benchmark.py`
- Modify: `tests/scripts/test_hio_cdi_benchmark.py`

- [ ] **Step 1: Update `SELECTED_SOLVER`**

Set:

```python
SELECTED_SOLVER = "pynx_cdi_hio_er"
```

- [ ] **Step 2: Update `write_solver_manifest(...)` tests first**

Add assertions that:

- `selected_solver == "pynx_cdi_hio_er"` when PyNX is available;
- PyNX candidate has `accepted: true`;
- study-local solver candidate has `accepted: false` and reason `superseded_by_pynx`;
- the manifest includes `pynx_preflight`, `array_convention`, `intensity_input`, and `processing_unit_cleanup` fields.

Run:

```bash
pytest tests/scripts/test_hio_cdi_benchmark.py::test_solver_manifest_has_a2_provenance_fields -vv
```

- [ ] **Step 3: Make the CLI fail fast when PyNX is unavailable**

Before metrics are run, require `check_pynx_available()["available"] is True`. If unavailable:

- write preflight, solver, data-identity, runtime, invocation, and manifest artifacts when possible;
- set a clear blocked status in the manifest;
- exit nonzero with an actionable error.

Do not silently run the custom HIO/ER path.

Verification:

```bash
pytest tests/scripts/test_hio_cdi_benchmark.py -k "solver_manifest or preflight or unavailable" -vv
```

## Task 4: Add Real PyNX Smoke Integration

**Files:**
- Modify: `tests/scripts/test_hio_cdi_benchmark.py`

- [ ] **Step 1: Add an import-skipped integration test**

Add a test guarded by:

```python
pynx_cdi = pytest.importorskip("pynx.cdi")
```

Use an `8x8` or `16x16` synthetic patch with a non-full support and a deterministic seed. Run a tiny `HIO(nb_cycle=2)` then `ER(nb_cycle=2)` path through `run_restarts(...)`, then assert:

- returned object shape matches target shape;
- values are finite;
- residual curve is present;
- selected restart metadata is recorded;
- `FreePU` cleanup does not leave the result inaccessible.

Run:

```bash
pytest tests/scripts/test_hio_cdi_benchmark.py -k "pynx and integration" -vv
```

Expected in the current environment: skip if PyNX is not installed. Expected in the PyNX environment: pass.

- [ ] **Step 2: Add an explicit environment preflight command**

After installing PyNX in the execution environment, run:

```bash
python - <<'PY'
from pynx.cdi import CDI, ER, FreePU, HIO
import importlib.metadata
print("pynx", importlib.metadata.version("pynx"))
print(CDI, HIO, ER, FreePU)
PY
```

Expected: prints the installed PyNX version and operator classes.

## Task 5: Run a Fresh PyNX Benchmark Pass

**Files:**
- Modify only if required by tests: `scripts/reconstruction/hio_cdi_benchmark.py`
- Artifact output: `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/<new_pynx_run_id>/`

- [ ] **Step 1: Run preflight-only in tmux**

Use a new output root; do not overwrite the custom-solver run:

```bash
python scripts/reconstruction/hio_cdi_benchmark.py \
  --output-root .artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/pynx_preflight_<timestamp> \
  --run-id pynx_preflight_<timestamp> \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-source custom \
  --probe-scale-mode pad_preserve \
  --probe-smoothing-sigma 0.5 \
  --support-thresholds 0.05 \
  --primary-support-threshold 0.05 \
  --restart-seeds 2026041201 2026041202 2026041203 \
  --data-identity-branch same-split-rerun \
  --data-generation-control loader-compatible \
  --metric-contract-mode direct-stitch \
  --preflight-only
```

Expected: exit code `0` only if PyNX is importable and the manifest says PyNX is selected.

- [ ] **Step 2: Run a tiny smoke metric**

Use `--max-test-frames` small enough for fast verification, and label it as smoke-only.

Expected:

- metrics JSON exists;
- residual JSON exists;
- solver manifest selected solver is PyNX;
- metrics payload records PyNX preflight and array convention;
- no production path reports `study_local_hio_er`.

- [ ] **Step 3: Run the primary full row only after the smoke passes**

Use the same primary support threshold and restart seeds as the custom run, but a fresh output root:

```bash
python scripts/reconstruction/hio_cdi_benchmark.py \
  --output-root .artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/full_gs1_custom_pynx_primary_<timestamp> \
  --run-id full_gs1_custom_pynx_primary_<timestamp> \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-source custom \
  --probe-scale-mode pad_preserve \
  --probe-smoothing-sigma 0.5 \
  --support-thresholds 0.05 \
  --primary-support-threshold 0.05 \
  --restart-seeds 2026041201 2026041202 2026041203 \
  --beta 0.9 \
  --hio-iters 1000 \
  --er-iters 200 \
  --data-identity-branch same-split-rerun \
  --data-generation-control loader-compatible \
  --metric-contract-mode direct-stitch
```

Expected: exit code `0`, PyNX-selected solver manifest, metrics JSON, residual JSON, recon NPZ, and no custom-solver provenance in the primary row.

## Task 6: Retire or Quarantine Custom Solver Evidence

**Files:**
- Modify: `scripts/reconstruction/hio_cdi_benchmark.py`
- Modify: `tests/scripts/test_hio_cdi_benchmark.py`
- Optional docs update: `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-execution-plan.md`

- [ ] **Step 1: Remove production references to `study_local_hio_er`**

Search:

```bash
rg -n "\bstudy_local_hio_er\b|\bhio_update\b|\ber_cleanup\b|\bproject_fourier_magnitude\b|\b_initial_psi\b" scripts/reconstruction/hio_cdi_benchmark.py tests/scripts/test_hio_cdi_benchmark.py
```

Expected: no old production solver path remains. Any residual `study_local_hio_er` references must be explicitly historical/diagnostic manifest context and not selectable by the CLI; separately named `known_probe_*` helpers are allowed only under `--solver known_probe_object_hio_er`.

- [ ] **Step 2: Annotate old custom-solver run outputs as superseded**

Do not delete existing artifacts. In the execution report or plan notes, mark the custom-solver metrics as superseded by the PyNX replacement run and unsuitable for reviewer-facing claims.

- [ ] **Step 3: Update the paper checklist only after PyNX evidence exists**

If PyNX run succeeds and review accepts it, update `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`. If PyNX cannot be installed or cannot run correctly, record a pivot note instead of checking the item off.

## Verification Commands

Targeted tests:

```bash
pytest tests/scripts/test_hio_cdi_benchmark.py -vv
```

Import and static checks:

```bash
python -m py_compile scripts/reconstruction/hio_cdi_benchmark.py
python - <<'PY'
from scripts.reconstruction import hio_cdi_benchmark as hio
print(hio.SELECTED_SOLVER)
print(hio.check_pynx_available())
PY
```

PyNX environment check:

```bash
python - <<'PY'
from pynx.cdi import CDI, ER, FreePU, HIO
import importlib.metadata
print("pynx", importlib.metadata.version("pynx"))
print(CDI, HIO, ER, FreePU)
PY
```

Repository hygiene check:

```bash
git diff --check
rg -n "\bstudy_local_hio_er\b|\bhio_update\b|\ber_cleanup\b|\bproject_fourier_magnitude\b|\b_initial_psi\b" scripts/reconstruction/hio_cdi_benchmark.py tests/scripts/test_hio_cdi_benchmark.py
```

## Completion Criteria

- [x] The default production HIO/ER benchmark path uses PyNX operators, not the old `study_local_hio_er` custom update math.
- [x] PyNX unavailable state fails fast with manifest evidence and no custom fallback for the PyNX-selected path.
- [x] Unit tests cover PyNX adapter boundaries with a fake PyNX module.
- [x] Real PyNX integration test exists and skips when PyNX is absent.
- [x] Solver manifest selects `pynx_cdi_hio_er` by default for metric runs.
- [x] Residual curves, restart selection, support policy, ambiguity policy, and metric-ground-truth artifacts remain present.
- [x] A fresh PyNX output root exists for the PyNX run; old custom-solver outputs were not overwritten.
- [x] Reviewer-facing reports do not cite the old custom-solver SSIM/PSNR as the final non-ML CDI comparator.

## Notes

- Earlier local preflight on 2026-04-13 failed with `ModuleNotFoundError` for `pynx`/`pynx.cdi`.
- Later on 2026-04-13, PyNX was installed in the active environment and the PyNX adapter path became runnable. The live implementation imports `CDI`, `HIO`, `ER`, and `FreePU` via `check_pynx_available()`, converts repo-normalized amplitudes to PyNX intensity with `to_pynx_intensity(..., intensity_scale_factor=1.0)`, retrieves objects with `get_obj(shift=True)`, and calls `FreePU` after each restart when available.
- The PyNX run rooted at `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/full_gs1_custom_pynx_20260413T050728Z` is the current full-run PyNX evidence path.
- Later known-probe work restored `known_probe_object_hio_er` as an explicit secondary solver mode. That does not undo this PyNX replacement plan: the default `SELECTED_SOLVER` remains `pynx_cdi_hio_er`, while known-probe outputs must carry their own solver label and interpretation.
