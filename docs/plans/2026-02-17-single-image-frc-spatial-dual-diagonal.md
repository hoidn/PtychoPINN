# Single-Image FRC Spatial Dual-Diagonal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current problematic spatial single-image FRC split with a robust dual-diagonal strided method (no averaging), while preserving reproducibility via an explicit legacy mode and keeping binomial as the default reported mode.

**Architecture:** Implement the full spatial dual-diagonal logic in `../frc/frc/single_image_frc.py` (the external package that now owns single-image FRC). Keep three explicit split modes: `binomial`, `spatial_dual` (new robust mode), and `spatial_legacy` (old interpolated split for reproducibility only). Route `spatial` to `spatial_dual` to avoid future confusion. Use curve-level dual-pair averaging with ring-count weighting to reduce directional bias.

**Tech Stack:** Python 3.11, NumPy, SciPy, pytest, existing study script (`scripts/studies/analyze_single_image_frc_alignment.py`)

---

### Task 1: Lock in the Expected Spatial Behavior with RED Tests

**Files:**
- Modify: `tests/test_evaluation_single_image_frc.py`
- Modify: `tests/studies/test_single_image_frc_sanity.py`
- Test: `tests/test_evaluation_single_image_frc.py`
- Test: `tests/studies/test_single_image_frc_sanity.py`

**Step 1: Write the failing mode-contract test**

Add test asserting supported modes and alias behavior:

```python
def test_external_single_image_frc_supports_spatial_dual_and_legacy_modes():
    from frc.single_image_frc import single_image_frc_metrics
    obj = _make_complex_object(seed=123, size=128)
    m_dual = single_image_frc_metrics(_as_hw1(obj), offset=4, split_mode="spatial_dual")
    m_alias = single_image_frc_metrics(_as_hw1(obj), offset=4, split_mode="spatial")
    m_legacy = single_image_frc_metrics(_as_hw1(obj), offset=4, split_mode="spatial_legacy")
    assert "single_frc50" in m_dual and "single_frc50" in m_alias and "single_frc50" in m_legacy
```

**Step 2: Write the failing blur-direction test for spatial_dual**

Add test requiring spatial dual mode to track GT direction (not opposite):

```python
def test_single_image_frc_amp_rank_correlation_spatial_dual_is_positive_vs_gt():
    gt_vals, spatial_dual_vals = _collect_frc_series("spatial_dual")
    rho, _ = spearmanr(gt_vals, spatial_dual_vals)
    assert np.isfinite(rho)
    assert float(rho) > 0.5
```

**Step 3: Run tests to verify failure**

Run: `pytest tests/test_evaluation_single_image_frc.py tests/studies/test_single_image_frc_sanity.py -k "spatial_dual or spatial_legacy" -v`
Expected: FAIL (modes and behavior not implemented yet).

**Step 4: Commit RED tests**

```bash
git add tests/test_evaluation_single_image_frc.py tests/studies/test_single_image_frc_sanity.py
git commit -m "test(frc): add red tests for spatial dual-diagonal split modes"
```

---

### Task 2: Implement Dual-Diagonal Strided Spatial Split in External Package

**Files:**
- Modify: `../frc/frc/single_image_frc.py`
- Modify: `../frc/frc/__init__.py`
- Test: `tests/test_evaluation_single_image_frc.py`

**Step 1: Add split primitives without smoothing**

Implement in `../frc/frc/single_image_frc.py`:

```python
def split_diagonal_strided_main(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # (00) vs (11)


def split_diagonal_strided_anti(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # (01) vs (10)
```

No interpolation/averaging; direct subsampling only.

**Step 2: Add dual-pair curve path**

Implement helper:

```python
def _single_image_frc_curve_spatial_dual(canvas: np.ndarray, frc_sigma: float) -> np.ndarray:
    # compute curve_main and curve_anti
    # combine with ring-count weighting
```

Use same signed-shell-correlation method already used by current single-image FRC.

**Step 3: Add explicit split mode routing**

In `single_image_frc_curve(...)`, support:
- `split_mode="binomial"`
- `split_mode="spatial_dual"` (new default spatial behavior)
- `split_mode="spatial"` as alias to `spatial_dual`
- `split_mode="spatial_legacy"` to preserve old interpolated behavior (existing `split_diagonal_interleaved`)

**Step 4: Export new public APIs**

Update `../frc/frc/__init__.py` exports for any new split helpers used by tests/docs.

**Step 5: Run focused tests to verify pass**

Run: `pytest tests/test_evaluation_single_image_frc.py -k "spatial_dual or spatial_legacy" -v`
Expected: PASS.

**Step 6: Commit implementation**

```bash
git add ../frc/frc/single_image_frc.py ../frc/frc/__init__.py tests/test_evaluation_single_image_frc.py
git commit -m "feat(frc): add dual-diagonal strided spatial split with explicit legacy mode"
```

---

### Task 3: Wire Through Evaluation Wrapper and Keep Compatibility

**Files:**
- Modify: `ptycho/evaluation.py`
- Test: `tests/test_evaluation_single_image_frc.py`

**Step 1: Add failing wrapper-forwarding test**

```python
def test_eval_reconstruction_accepts_spatial_dual_single_image_mode():
    from ptycho.evaluation import eval_reconstruction
    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=201, size=128))
    pred = _as_hw1(_make_complex_object(seed=202, size=128))
    out = eval_reconstruction(pred, gt, single_image_frc=True, single_image_frc_split_mode="spatial_dual")
    assert "single_frc50" in out
```

**Step 2: Run test to verify failure**

Run: `pytest tests/test_evaluation_single_image_frc.py -k "spatial_dual_single_image_mode" -v`
Expected: FAIL until forwarding/validation accepts mode.

**Step 3: Implement minimal forwarding/validation**

Ensure `ptycho/evaluation.py` wrapper passes split modes directly to external package and preserves existing `single_image_frc_split_mode="spatial"` compatibility.

**Step 4: Run wrapper tests to verify pass**

Run: `pytest tests/test_evaluation_single_image_frc.py -k "single_image_frc" -v`
Expected: PASS.

**Step 5: Commit wrapper changes**

```bash
git add ptycho/evaluation.py tests/test_evaluation_single_image_frc.py
git commit -m "feat(evaluation): support spatial_dual split mode via external frc package"
```

---

### Task 4: Validate Study-Level Metric Direction and Regenerate Artifacts

**Files:**
- Modify: `scripts/studies/analyze_single_image_frc_alignment.py`
- Modify: `../frc/README.md` (if regenerated with `--write-readme`)
- Test: `tests/studies/test_single_image_frc_sanity.py`

**Step 1: Write failing study sanity test for spatial_dual**

Add/adjust study test to require positive alignment vs GT for `spatial_dual`:

```python
def test_single_image_frc_amp_rank_correlation_spatial_dual_positive_vs_gt():
    gt, dual = _collect_frc_series("spatial_dual")
    rho, _ = spearmanr(gt, dual)
    assert float(rho) > 0.5
```

**Step 2: Run test to verify failure**

Run: `pytest tests/studies/test_single_image_frc_sanity.py -k "spatial_dual" -v`
Expected: FAIL until script/paths fully use new mode.

**Step 3: Update analysis script mode usage**

In `scripts/studies/analyze_single_image_frc_alignment.py`:
- keep reported production comparisons binomial-focused (as now)
- if any spatial mode diagnostics remain, use `spatial_dual` (not legacy interpolated)

**Step 4: Re-run tests and regenerate artifacts**

Run:
- `pytest tests/studies/test_single_image_frc_sanity.py -v`
- `python scripts/studies/analyze_single_image_frc_alignment.py --output-dir ../frc --level-min 0.0 --level-max 2.0 --n-levels 13`

Expected:
- tests PASS
- plots and CSV regenerate with no spatial-interpolated artifacts.

**Step 5: Conditionally update README with spatial-mode data (IFF working)**

Gate condition (must all be true):
- `spatial_dual` test selector passes
- `spearmanr(gt, spatial_dual) > 0.5` on the study sanity sweep
- no ceiling-dominated pathological trend in the selected spatial diagnostic plot

If gate condition passes:
- add/refresh a spatial-diagnostics subsection in `../frc/README.md` with the regenerated spatial plot(s) and one-line interpretation

If gate condition fails:
- do not add spatial-diagnostics data to `../frc/README.md`
- add a brief note in commit message body (or follow-up plan note) that spatial mode is still diagnostic-only and withheld from README summary

**Step 6: Commit study updates**

```bash
git add tests/studies/test_single_image_frc_sanity.py scripts/studies/analyze_single_image_frc_alignment.py ../frc/raw_metrics.csv ../frc/summary.json ../frc/plots
# add ../frc/README.md only if --write-readme was used
git commit -m "test(studies): validate spatial_dual single-image frc behavior and refresh artifacts"
```

---

### Task 5: Documentation and Deprecation Notes

**Files:**
- Modify: `../frc/README.md`
- Modify: `docs/plans/2026-02-17-single-image-frc-spatial-dual-diagonal.md` (mark completed if executed)

**Step 1: Add mode semantics section**

Document clearly:
- `binomial`: default recommended mode
- `spatial_dual`: diagnostic spatial no-averaging mode
- `spatial_legacy`: retained only for reproducibility, not recommended

**Step 2: Add mathematical note for dual-diagonal averaging**

Include concise formula for combined curve:

`FRC_dual(r) = (w1(r) * FRC_main(r) + w2(r) * FRC_anti(r)) / (w1(r) + w2(r))`

where `w1, w2` are ring-sample counts.

**Step 3: Verify docs and links**

Run: `rg -n "spatial_legacy|spatial_dual|binomial" ../frc/README.md scripts/studies/analyze_single_image_frc_alignment.py`
Expected: mode semantics are explicit and consistent.

**Step 4: Commit docs**

```bash
git add ../frc/README.md docs/plans/2026-02-17-single-image-frc-spatial-dual-diagonal.md
git commit -m "docs(frc): document spatial_dual method and legacy spatial deprecation"
```

---

Plan complete and saved to `docs/plans/2026-02-17-single-image-frc-spatial-dual-diagonal.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
