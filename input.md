# Ralph Input — DEBUG-SIM-LINES-DOSE-001 REGRESSION FIX (REG-2)

**Summary:** Fix `calculate_intensity_scale()` in `ptycho/train_pinn.py` to restore the 3-priority order for intensity scale computation. Tests currently fail 6/7.

**Focus:** DEBUG-SIM-LINES-DOSE-001 — REGRESSION RECOVERY (REG-2)

**Branch:** paper

**Mapped tests:**
- `pytest tests/test_train_pinn.py::TestIntensityScale -v` (7 tests, currently 6 FAIL)
- `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

**Artifacts:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T234600Z/`

---

## Context — Current Broken State

The current `calculate_intensity_scale()` at `ptycho/train_pinn.py:165-180` only uses the closed-form fallback:

```python
def calculate_intensity_scale(ptycho_data_container: PtychoDataContainer) -> float:
    import tensorflow as tf
    import numpy as np
    from . import params as p
    def count_photons(obj):
        pcount = np.mean(tf.math.reduce_sum(obj**2, (1, 2)))
        return pcount

    def scale_nphotons(X):
        # TODO assumes X is already normalized. this should be enforced
        return tf.math.sqrt(p.get('nphotons')) / (p.get('N') / 2)

    # Calculate the intensity scale using the adapted scale_nphotons function
    intensity_scale = scale_nphotons(ptycho_data_container.X).numpy()

    return intensity_scale
```

**Problems:**
1. Only uses closed-form fallback `sqrt(nphotons)/(N/2)`, ignoring dataset statistics
2. Accesses `.X` property which forces tensor materialization (breaks lazy loading)
3. Ignores `dataset_intensity_stats` attribute even when present
4. Ignores `_X_np` attribute (CPU-safe path)

---

## Do Now — Implement REG-2 Fix

### Implement: `ptycho/train_pinn.py::calculate_intensity_scale`

**Replace lines 165-180** with this implementation that restores the 3-priority order per `specs/spec-ptycho-core.md §Normalization Invariants`:

```python
def calculate_intensity_scale(ptycho_data_container: PtychoDataContainer) -> float:
    """
    Calculate intensity scale per specs/spec-ptycho-core.md §Normalization Invariants.

    Formula: s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])

    Priority order:
    1. dataset_intensity_stats (if present) — pre-computed from raw diffraction, preferred
    2. _X_np NumPy reduction — lazy-container safe, CPU-only
    3. Closed-form fallback sqrt(nphotons)/(N/2) — assumes L2-normalized data

    Returns:
        float: Intensity scale factor for the dataset.
    """
    import numpy as np
    from . import params as p

    nphotons = p.get('nphotons')
    N = p.get('N')

    # Priority 1: Use pre-computed stats from raw diffraction (before normalization)
    if hasattr(ptycho_data_container, 'dataset_intensity_stats'):
        stats = ptycho_data_container.dataset_intensity_stats
        if stats is not None and stats.get('batch_mean_sum_intensity', 0) > 1e-12:
            return float(np.sqrt(nphotons / stats['batch_mean_sum_intensity']))

    # Priority 2: Compute from NumPy backing (lazy-container safe, no .X access)
    if hasattr(ptycho_data_container, '_X_np') and ptycho_data_container._X_np is not None:
        X_np = ptycho_data_container._X_np.astype(np.float64)  # float64 for precision
        # Handle both rank-3 (B, H, W) and rank-4 (B, H, W, C) tensors
        spatial_axes = tuple(range(1, X_np.ndim))  # (1,2) or (1,2,3)
        sum_intensity = np.sum(X_np ** 2, axis=spatial_axes)
        batch_mean = float(np.mean(sum_intensity))
        if batch_mean > 1e-12:
            return float(np.sqrt(nphotons / batch_mean))

    # Priority 3: Closed-form fallback (assumes L2-normalized to (N/2)²)
    return float(np.sqrt(nphotons) / (N / 2))
```

### Verification Commands

```bash
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
ARTIFACTS=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T234600Z
mkdir -p "$ARTIFACTS"/logs

# 1. Collect tests first (should show 7 tests)
pytest --collect-only tests/test_train_pinn.py::TestIntensityScale -q \
  2>&1 | tee "$ARTIFACTS"/logs/pytest_train_pinn_collect.log

# 2. Run intensity scale tests (MUST pass 7/7 after fix)
pytest tests/test_train_pinn.py::TestIntensityScale -v \
  2>&1 | tee "$ARTIFACTS"/logs/pytest_train_pinn.log

# 3. CLI smoke test (import health check)
pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
  2>&1 | tee "$ARTIFACTS"/logs/pytest_cli_smoke.log
```

---

## Pitfalls To Avoid

1. **Do NOT access `.X` property** — it forces tensor materialization and defeats lazy loading (PINN-CHUNKED-001)
2. **Use float64 for intermediate reduction** — prevents numerical underflow on small values
3. **Handle both rank-3 and rank-4 tensors** — test `test_rank3_tensor_handling` expects this
4. **Return Python float** — not numpy scalar (JSON serialization)
5. **Check for None and zero** — `dataset_intensity_stats` may be None or have near-zero values
6. **Do NOT import tensorflow** — the NumPy path is CPU-only and should stay that way
7. **Keep the dead code (`count_photons`, `scale_nphotons` helpers) removed** — they're unused
8. **Archive all logs** in the artifacts directory

---

## If Blocked

- If tests still fail after applying the fix, check:
  1. Are you editing the right function? Look for `def calculate_intensity_scale` around line 165
  2. Is the priority order correct? Stats → NumPy → Fallback
  3. Did you keep the `.X` access? Remove it.
- Add blocking notes to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T234600Z/blockers.md`

---

## Findings Applied (Mandatory)

- **PINN-CHUNKED-001:** Lazy container design requires avoiding `.X` access; use `_X_np` instead
- **specs/spec-ptycho-core.md §Normalization Invariants:** `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])` is the preferred formula
- **CONFIG-001:** Callers must sync `params.cfg` before calling; this function reads `nphotons` and `N` from params

---

## Pointers

- `ptycho/train_pinn.py:165-180` — Current broken implementation to replace
- `tests/test_train_pinn.py:260-360` — Test class `TestIntensityScale` with 7 tests
- `specs/spec-ptycho-core.md §Normalization Invariants` — Authoritative formula
- `docs/findings.md` — PINN-CHUNKED-001 lazy loading requirements

---

## Next Up (after REG-2)

1. **REG-3:** Restore `_update_max_position_jitter_from_offsets()` to `ptycho/workflows/components.py`
2. **REG-4:** Fix `align_for_evaluation_with_registration()` missing from cropping.py
3. **REG-1:** Add `dataset_intensity_stats` parameter to `PtychoDataContainer.__init__`

---

## Success Criteria

- [ ] `pytest tests/test_train_pinn.py::TestIntensityScale -v` → 7/7 PASSED
- [ ] `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` → PASSED
- [ ] All logs archived in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T234600Z/logs/`
- [ ] Commit with message: `DEBUG-SIM-LINES-DOSE-001 REG-2: restore calculate_intensity_scale 3-priority order (tests: TestIntensityScale)`
