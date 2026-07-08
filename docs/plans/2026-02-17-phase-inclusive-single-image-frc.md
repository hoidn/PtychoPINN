# Phase-Inclusive Single-Image FRC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

> **Historical note (2026-04-13):** Superseded by `docs/plans/2026-04-13-single-image-frc-removal.md`. The no-reference single-image FRC feature and `single_frc50` / `single_frc1over7` API are removed from tracked code; do not execute this plan as current guidance.

**Goal:** Add no-ground-truth single-image FRC metrics for both amplitude and phase (phase via support-weighted phasor), supporting both spatial interleaved splitting and binomial splitting, and report both 0.5 and 1/7 thresholds across grid-lines study outputs.

**Architecture:** Keep existing ground-truth metrics unchanged, and add a separate single-image FRC path in `ptycho/evaluation.py` that operates on the reconstructed image alone. Implement two split backends: (1) 2x2 diagonal interleaved spatial split and (2) Poisson/binomial count split with deterministic RNG control. Reuse the existing FRC backend for curve/cutoff extraction, expose split-mode selection as explicit API/CLI knobs, and integrate metric keys into study runners/wrappers and downstream table/aggregation consumers using the existing metric-pair schema (`metric -> [amp, phase]`) to minimize debt and avoid format divergence.

**Tech Stack:** Python 3.11, NumPy, `ptycho.FRC.fourier_ring_corr`, argparse, pytest

---

### Task 1: Define Single-Image FRC Contract with Failing Unit Tests (RED)

**Files:**
- Create: `tests/test_evaluation_single_image_frc.py`
- Modify: `ptycho/evaluation.py`
- Test: `tests/test_evaluation_single_image_frc.py`

**Step 1: Write the failing tests**

Add tests for contract and algorithm behavior:

```python
def test_single_image_frc_returns_both_threshold_pairs_for_amp_and_phase():
    obj = make_complex_object(seed=3, size=128)
    metrics = single_image_frc_metrics(obj, phase_align_method="plane")
    assert "single_frc50" in metrics
    assert "single_frc1over7" in metrics
    assert len(metrics["single_frc50"]) == 2
    assert len(metrics["single_frc1over7"]) == 2
```

```python
def test_single_image_frc_uses_support_weighted_phase_phasor():
    obj = make_complex_object_with_low_amp_background()
    out = single_image_frc_metrics(obj, support_amp_floor_ratio=0.05)
    assert np.isfinite(out["single_frc50"][1])
```

```python
def test_single_image_frc_phase_is_stable_to_added_plane_ramp():
    obj = make_complex_object(seed=5, size=128)
    ramped = apply_phase_plane(obj, ax=0.02, by=-0.03)
    a = single_image_frc_metrics(obj, phase_align_method="plane")
    b = single_image_frc_metrics(ramped, phase_align_method="plane")
    assert abs(a["single_frc1over7"][1] - b["single_frc1over7"][1]) <= 1
```

```python
def test_single_image_frc_supports_spatial_and_binomial_modes():
    obj = make_complex_object(seed=11, size=128)
    spatial = single_image_frc_metrics(obj, split_mode="spatial")
    binom = single_image_frc_metrics(obj, split_mode="binomial", rng_seed=11)
    assert "single_frc50" in spatial
    assert "single_frc50" in binom
```

```python
def test_single_image_frc_handles_odd_image_sizes_by_center_crop():
    obj = make_complex_object(seed=7, size=129)
    out = single_image_frc_metrics(obj)
    assert "single_frc50" in out
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation_single_image_frc.py -v`
Expected: FAIL (helper functions not implemented).

**Step 3: Commit RED tests**

```bash
git add tests/test_evaluation_single_image_frc.py
git commit -m "test(evaluation): add red tests for phase-inclusive single-image frc contract"
```

---

### Task 2: Implement Single-Image FRC Core Helpers in Evaluation Module (GREEN)

**Files:**
- Modify: `ptycho/evaluation.py`
- Test: `tests/test_evaluation_single_image_frc.py`

**Step 1: Implement minimal helper API**

Implement (module-private unless explicitly exported):

```python
def _center_crop_even_square(arr: np.ndarray) -> np.ndarray:
    ...

def _split_diagonal_interleaved(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # A = 0.5*(00 + 11), B = 0.5*(01 + 10)
    ...

def _first_below_threshold(curve: np.ndarray, threshold: float) -> float:
    ...

def _support_weighted_phase_phasor(
    phase_aligned: np.ndarray,
    amp: np.ndarray,
    support_amp_floor_ratio: float,
) -> np.ndarray:
    # return support_mask * exp(1j * phase)
    ...

def single_image_frc_metrics(
    stitched_obj: np.ndarray,
    *,
    split_mode: str = "spatial",
    rng_seed: int | None = None,
    phase_align_method: str = "plane",
    support_amp_floor_ratio: float = 0.05,
    frc_sigma: float = 0.0,
) -> dict[str, tuple[float, float]]:
    ...
```

Implementation requirements:
- Reuse current phase alignment policy (`plane` / `mean`) from `eval_reconstruction`.
- Compute amplitude single-image FRC on `abs(stitched_obj)`.
- Compute phase single-image FRC on support-weighted phasor representation.
- Support both split methods:
  - `split_mode="spatial"`: 2x2 diagonal interleaved split.
  - `split_mode="binomial"`: generate two half-count images from Poisson-like counts via binomial thinning and map consistently into amplitude/phase branches.
- Make binomial path deterministic when `rng_seed` is provided.
- Report both thresholds in pair format:
  - `single_frc50: (amp_cutoff, phase_cutoff)`
  - `single_frc1over7: (amp_cutoff, phase_cutoff)`
- Use the same FRC engine (`ptycho.FRC.fourier_ring_corr.FSC`) after interleaved splitting.
- If valid support is absent for phase, return `np.nan` for phase cutoffs.

**Step 2: Run tests to verify pass**

Run: `pytest tests/test_evaluation_single_image_frc.py -v`
Expected: PASS.

**Step 3: Commit GREEN implementation**

```bash
git add ptycho/evaluation.py tests/test_evaluation_single_image_frc.py
git commit -m "feat(evaluation): add phase-inclusive single-image frc with 0.5 and 1/7 thresholds"
```

---

### Task 3: Integrate Single-Image FRC into Existing Evaluation Orchestrator

**Files:**
- Modify: `ptycho/evaluation.py`
- Test: `tests/test_evaluation_single_image_frc.py`

**Step 1: Write failing integration test(s)**

Add tests verifying `eval_reconstruction` behavior:

```python
def test_eval_reconstruction_emits_single_image_frc_when_enabled():
    pred, gt = make_prediction_and_gt()
    out = eval_reconstruction(pred, gt, label="pinn", single_image_frc=True)
    assert "single_frc50" in out
    assert "single_frc1over7" in out
```

```python
def test_eval_reconstruction_default_does_not_break_legacy_keys():
    pred, gt = make_prediction_and_gt()
    out = eval_reconstruction(pred, gt, label="pinn")
    for k in ("mae", "mse", "psnr", "ssim", "ms_ssim", "frc50", "frc"):
        assert k in out
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_evaluation_single_image_frc.py -k "eval_reconstruction" -v`
Expected: FAIL until integration is implemented.

**Step 3: Implement minimal integration**

In `eval_reconstruction(...)`:
- Add kwarg `single_image_frc: bool = False` (backward-compatible default).
- Add kwargs passthrough for split config:
  - `single_image_frc_split_mode: str = "spatial"`
  - `single_image_frc_rng_seed: int | None = None`
- When enabled, call `single_image_frc_metrics(...)` on prediction path and merge keys into output dict.
- Keep all existing legacy metric keys and semantics unchanged.

**Step 4: Re-run tests**

Run: `pytest tests/test_evaluation_single_image_frc.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho/evaluation.py tests/test_evaluation_single_image_frc.py
git commit -m "feat(evaluation): wire optional single-image frc into eval_reconstruction"
```

---

### Task 4: Wire Metric Through Grid-Lines Study Runners/Wrappers

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write failing tests**

Add pass-through tests:

```python
def test_torch_runner_passes_single_image_frc_flag_to_eval(monkeypatch, tmp_path):
    ...
    assert captured["single_image_frc"] is True
```

```python
def test_compare_wrapper_models_reuse_path_includes_single_image_frc(monkeypatch, tmp_path):
    ...
    assert captured_eval_kwargs["single_image_frc"] is True
```

```python
def test_grid_lines_workflow_enables_single_image_frc_in_metrics(monkeypatch, tmp_path):
    ...
```

**Step 2: Run failing selectors**

Run:
- `pytest tests/torch/test_grid_lines_torch_runner.py -k "single_image_frc" -v`
- `pytest tests/test_grid_lines_compare_wrapper.py -k "single_image_frc" -v`

Expected: FAIL.

**Step 3: Implement minimal wiring**

- `TorchRunnerConfig` add `single_image_frc: bool = True`.
- `TorchRunnerConfig` add:
  - `single_image_frc_split_mode: str = "spatial"`
  - `single_image_frc_rng_seed: int | None = None`
- Add CLI toggles in torch runner:
  - `--single-image-frc` (default true)
  - `--no-single-image-frc`
- Add CLI split controls:
  - `--single-image-frc-split-mode {spatial,binomial}`
  - `--single-image-frc-rng-seed INT`
- Pass `single_image_frc=cfg.single_image_frc` in `compute_metrics`/`eval_reconstruction` path.
- In compare wrapper evaluation path (`evaluate_selected_models`), pass `single_image_frc=True`.
- In TF grid-lines workflow calls to `eval_reconstruction`, pass `single_image_frc=True` so all model families produce comparable metric payloads.

**Step 4: Re-run tests**

Run:
- `pytest tests/torch/test_grid_lines_torch_runner.py -k "single_image_frc" -v`
- `pytest tests/test_grid_lines_compare_wrapper.py -k "single_image_frc" -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py scripts/studies/grid_lines_compare_wrapper.py ptycho/workflows/grid_lines_workflow.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(studies): wire single-image frc through torch and compare workflows"
```

---

### Task 5: Extend Metrics Tables and Aggregation Schema

**Files:**
- Modify: `scripts/studies/metrics_tables.py`
- Modify: `scripts/studies/aggregate_and_plot_results.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Modify: `tests/studies/test_aggregate_nan_msssim.py`
- Create: `tests/studies/test_aggregate_single_frc_metrics.py`

**Step 1: Write failing tests**

Add tests for schema acceptance and output visibility:

```python
def test_metrics_table_includes_single_frc_columns(tmp_path):
    metrics = {
        "pinn": {"single_frc50": [40, 31], "single_frc1over7": [58, 46], ...},
        "baseline": {...},
    }
    ...
    assert "Single-FRC@0.5" in table_text
    assert "Single-FRC@1/7" in table_text
```

```python
def test_aggregate_loader_accepts_single_frc_columns(tmp_path):
    # comparison_metrics.csv with single_frc50_amp/phase and single_frc1over7_amp/phase
    df = load_comparison_csv(csv_path)
    assert "single_frc50_phase" in df.columns
```

```python
def test_aggregate_cli_metric_choices_include_single_frc_variants():
    ...
```

**Step 2: Run failing selectors**

Run:
- `pytest tests/studies/test_aggregate_single_frc_metrics.py -v`
- `pytest tests/test_grid_lines_compare_wrapper.py -k "metrics_table" -v`

Expected: FAIL.

**Step 3: Implement minimal schema updates**

In `metrics_tables.py`:
- Extend `METRICS` with:
  - `("single_frc50", "Single-FRC@0.5")`
  - `("single_frc1over7", "Single-FRC@1/7")`
- Reuse pair extraction and best-model logic as-is.

In `aggregate_and_plot_results.py`:
- Add expected columns:
  - `single_frc50_phase`, `single_frc50_amp`
  - `single_frc1over7_phase`, `single_frc1over7_amp`
- Add parser `--metric` choices:
  - `single_frc50`, `single_frc1over7`
- Add y-axis labels for the two new metrics.

**Step 4: Re-run tests**

Run:
- `pytest tests/studies/test_aggregate_single_frc_metrics.py -v`
- `pytest tests/test_grid_lines_compare_wrapper.py -k "metrics_table" -v`
- `pytest tests/studies/test_aggregate_nan_msssim.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/metrics_tables.py scripts/studies/aggregate_and_plot_results.py tests/test_grid_lines_compare_wrapper.py tests/studies/test_aggregate_nan_msssim.py tests/studies/test_aggregate_single_frc_metrics.py
git commit -m "feat(studies): expose single-image frc metrics in tables and aggregation"
```

---

### Task 6: Document Metric Semantics and CLI Controls

**Files:**
- Modify: `scripts/studies/README.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify: `tests/test_docs_ptychovit_workflow.py`

**Step 1: Write/update doc tests (RED)**

Add assertions for:
- `single_frc50` and `single_frc1over7` mention in study docs.
- phase method wording: `support-weighted phasor`.
- torch runner flags: `--single-image-frc` and `--no-single-image-frc`.

**Step 2: Run doc tests to verify failure**

Run: `pytest tests/test_docs_ptychovit_workflow.py -v`
Expected: FAIL.

**Step 3: Implement doc updates**

Document:
- Single-image FRC is no-reference and relative (not absolute physical resolution by default).
- Phase branch uses support-weighted phasor, not raw wrapped phase.
- Both thresholds reported: `0.5` and `1/7`.
- Example commands showing default behavior and explicit disable flag.

**Step 4: Re-run doc tests**

Run: `pytest tests/test_docs_ptychovit_workflow.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/README.md docs/workflows/pytorch.md docs/COMMANDS_REFERENCE.md tests/test_docs_ptychovit_workflow.py
git commit -m "docs(studies): document phase-inclusive single-image frc metrics and torch controls"
```

---

### Task 7: Add Sanity Validation Against GT Metrics (Consistency + Monotonicity) and Compare Split Modes

**Files:**
- Create: `tests/studies/test_single_image_frc_sanity.py`
- Modify: `tests/test_evaluation_single_image_frc.py`
- Modify: `docs/workflows/pytorch.md`

**Step 1: Write failing sanity tests (RED)**

Add trend-based sanity tests:

```python
def test_single_image_frc_amp_tracks_gt_frc_under_controlled_blur_levels():
    # Create one GT object and degraded predictions with increasing blur sigma.
    # Run eval_reconstruction(..., single_image_frc=True) per level.
    # Assert GT frc50 amp is non-increasing with degradation.
    # Assert single_frc1over7 amp is non-increasing (allow <=1-bin tolerance).
```

```python
def test_single_image_frc_phase_support_phasor_tracks_gt_phase_frc_under_phase_noise():
    # Add increasing wrapped phase perturbation to prediction while keeping support fixed.
    # Compute metrics with phase_align_method='plane' and single_image_frc=True.
    # Assert GT frc50 phase and single_frc1over7 phase both degrade monotonically
    # (with small tolerance for numerical ties).
```

```python
def test_single_image_frc_amp_and_gt_frc_have_positive_rank_correlation():
    # Across degradation levels, compute Spearman rank correlation.
    # Assert rho > 0 for amp branch (recommend threshold >= 0.7).
```

```python
def test_binomial_and_spatial_single_image_frc_show_consistent_ordering():
    # Across controlled degradation levels, compute both split modes.
    # Assert each mode is monotonic (with tolerance) and rank correlation
    # with GT FRC is positive.
```

```python
def test_binomial_vs_spatial_variance_over_repeated_seeds():
    # Repeat controlled perturbation at multiple RNG seeds.
    # Compare per-mode variance of single_frc cutoffs; report and assert finite.
```

**Step 2: Run failing selector**

Run: `pytest tests/studies/test_single_image_frc_sanity.py -v`
Expected: FAIL (tests absent and/or behavior not yet constrained).

**Step 3: Implement minimal fixes/guardrails**

If monotonicity tests expose instability:
- tighten support-mask handling for phase phasor branch,
- enforce deterministic even-square crop/split path,
- smooth FRC curves only via existing `frc_sigma` policy (no new ad-hoc filters),
- document expected tolerance and non-goals (relative trend metric, not absolute resolution).
- compare mode behavior and codify default policy in docs:
  - keep `spatial` as default unless binomial demonstrates strictly better stability
  - preserve opt-in binomial mode for advanced studies

**Step 4: Re-run tests**

Run:
- `pytest tests/studies/test_single_image_frc_sanity.py -v`
- `pytest tests/test_evaluation_single_image_frc.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/studies/test_single_image_frc_sanity.py tests/test_evaluation_single_image_frc.py ptycho/evaluation.py docs/workflows/pytorch.md
git commit -m "test(studies): add gt-consistency and monotonicity sanity checks for single-image frc"
```

---

### Task 8: Verification Bundle + Split-Mode Comparison Report

**Files:**
- Optional Create: `.artifacts/studies/single_image_frc_phase_inclusive/README.md`

**Step 1: Run targeted verification selectors**

```bash
pytest tests/test_evaluation_single_image_frc.py -v
pytest tests/studies/test_single_image_frc_sanity.py -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "single_image_frc" -v
pytest tests/test_grid_lines_compare_wrapper.py -k "single_image_frc or metrics_table" -v
pytest tests/studies/test_aggregate_single_frc_metrics.py -v
pytest tests/studies/test_aggregate_nan_msssim.py -v
pytest tests/test_docs_ptychovit_workflow.py -v
```

Expected: PASS.

**Step 2: Smoke run one grid-lines torch study**

```bash
python scripts/studies/grid_lines_torch_runner.py \
  --train-npz outputs/.../datasets/N64/gs1/train.npz \
  --test-npz outputs/.../datasets/N64/gs1/test.npz \
  --output-dir outputs/smoke_single_image_frc \
  --architecture hybrid \
  --epochs 1 \
  --single-image-frc
```

Expected in run metrics JSON:
- `single_frc50`
- `single_frc1over7`

**Step 3: Run comparison sweep (both modes)**

```bash
python scripts/studies/grid_lines_torch_runner.py \
  --train-npz outputs/.../datasets/N64/gs1/train.npz \
  --test-npz outputs/.../datasets/N64/gs1/test.npz \
  --output-dir outputs/smoke_single_image_frc_binomial \
  --architecture hybrid \
  --epochs 1 \
  --single-image-frc \
  --single-image-frc-split-mode binomial \
  --single-image-frc-rng-seed 123
```

Expected: both runs complete; compare `single_frc50` / `single_frc1over7` for stability/trend consistency.

**Step 4: Record artifact evidence**

Store command + output paths in:
- `.artifacts/studies/single_image_frc_phase_inclusive/README.md`

**Step 5: Commit evidence (optional policy-dependent)**

```bash
git add .artifacts/studies/single_image_frc_phase_inclusive/README.md
git commit -m "chore(studies): add verification evidence for phase-inclusive single-image frc"
```

---

### Task 9: Visual Validation (MPL PNGs) + `README.md`

**Files:**
- Create: `scripts/studies/analyze_single_image_frc_alignment.py`
- Create: `../frc/README.md`
- Create (generated): `../frc/plots/*.png`

**Goal:**
Produce visual, seed-aggregated evidence that binomial single-image FRC behaves as intended and aligns with SSIM (and GT FRC), with explicit contrast against spatial mode.

**Step 1: Implement analysis script for repeatable sweeps**

Script contract:
- Inputs:
  - `--n-seeds` (default: 20)
  - `--n-levels` (default: 13)
  - `--level-max` (default: 2.0 blur sigma)
  - `--output-dir` (default: `../frc`)
- Compute per-seed/per-level metrics via `eval_reconstruction(..., single_image_frc=True)` for:
  - `split_mode=spatial`
  - `split_mode=binomial`
- Persist compact raw table CSV/JSON for traceability.

**Step 2: Generate matplotlib visual checks (PNG)**

Required plots:
1. `trend_single_frc50_amp_vs_blur.png`
   - x: degradation level
   - y: `single_frc50_amp`
   - curves: spatial/binomial (mean ± std over seeds)
2. `trend_ssim_amp_vs_blur.png`
   - x: degradation level
   - y: `ssim_amp` (mean ± std)
3. `scatter_single_frc50_amp_vs_ssim_amp.png`
   - points pooled across seeds/levels
   - separate color per split mode
   - include fitted line and reported Spearman rho
4. `rho_ci_bar_amp.png`
   - bars: mean Spearman rho between `single_frc50_amp` and
     - `ssim_amp`
     - `frc50_amp`
   - error bars: bootstrap 95% CI
5. `phase_stability_overview.png` (optional but recommended)
   - summarize weaker/noisier phase behavior to avoid over-claiming.

**Step 3: Write `README.md`**

`README.md` must include:
- Exact commands used (copy/paste runnable).
- Data generation/sweep settings (`n_seeds`, `n_levels`, blur schedule, RNG policy).
- Embedded/linked PNGs with one-line interpretation per figure.
- Numerical table:
  - mean rho + 95% CI for binomial/spatial vs `ssim_amp`
  - mean rho + 95% CI for binomial/spatial vs `frc50_amp`
- Conclusion statement:
  - whether binomial satisfies alignment expectations,
  - whether spatial is anti-aligned on blur sweeps,
  - explicit caveat for phase branch.

**Step 4: Acceptance criteria**

For the blur-based sanity sweep:
- Binomial amplitude branch:
  - monotonic direction matches SSIM trend (both decreasing with blur),
  - Spearman rho(`single_frc50_amp`, `ssim_amp`) > 0 with CI not crossing 0.
- Spatial amplitude branch:
  - if anti-aligned, document as known behavior (not silent failure).
- No claims of absolute resolution; relative trend-only interpretation.

**Step 5: Verification + commit**

Run:
```bash
python scripts/studies/analyze_single_image_frc_alignment.py \
  --n-seeds 20 \
  --n-levels 13 \
  --level-max 2.0 \
  --output-dir ../frc
```

Verify files exist:
- `../frc/README.md`
- `../frc/plots/trend_single_frc50_amp_vs_blur.png`
- `../frc/plots/scatter_single_frc50_amp_vs_ssim_amp.png`
- `../frc/plots/rho_ci_bar_amp.png`

Commit:
```bash
git add scripts/studies/analyze_single_image_frc_alignment.py
git commit -m "analysis(studies): add visual frc-vs-ssim alignment report with ci-backed binomial comparison"
```

---

## Rollout Notes

- This plan keeps legacy GT metrics untouched and introduces single-image FRC as additive keys only.
- Pair-format metric keys (`single_frc50`, `single_frc1over7`) preserve compatibility with current metric table and CSV flattening conventions (`*_amp`, `*_phase`).
- Phase is intentionally computed via support-weighted phasor to reduce wrap/support artifacts in no-reference mode.
- Split-mode support is additive and explicit; default remains `spatial` unless comparison evidence supports changing defaults.
