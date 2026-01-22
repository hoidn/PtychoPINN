# Ralph Input — DEBUG-SIM-LINES-DOSE-001 Regression Recovery (URGENT)

**Summary:** Fix critical regressions that removed Phase D4f dataset_intensity_stats handling and Phase C canvas jitter guard. The codebase is broken and cannot produce valid telemetry until these are restored.

**Focus:** DEBUG-SIM-LINES-DOSE-001 — REGRESSION RECOVERY (REG-1 through REG-4)

**Branch:** paper

**Mapped tests:**
- `pytest tests/test_train_pinn.py::TestIntensityScale -v` (REG-2)
- `pytest tests/test_workflow_components.py::TestCreatePtychoDataContainer::test_updates_max_position_jitter -v` (REG-3)
- `python -c "from scripts.studies.sim_lines_4x.evaluate_metrics import *"` (REG-4)
- `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (overall health)

**Artifacts:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T233342Z/`

---

## Context — Identified Regressions

A reviewer audit found that prior fixes were removed from the codebase:

| ID | File | Issue |
|----|------|-------|
| REG-1 | `ptycho/loader.py` | `PtychoDataContainer.__init__` no longer accepts `dataset_intensity_stats`; callers raise `TypeError` |
| REG-2 | `ptycho/train_pinn.py` | `calculate_intensity_scale()` reverted to closed-form only (`sqrt(nphotons)/(N/2)`); dataset-derived code removed |
| REG-3 | `ptycho/workflows/components.py` | `_update_max_position_jitter_from_offsets()` deleted; canvas no longer expands for large grouped offsets |
| REG-4 | `ptycho/image/cropping.py` | `align_for_evaluation_with_registration()` deleted but `evaluate_metrics.py` still imports it |

Loss-weight change (REG-5) is a plan/scope decision — do NOT address in this loop.

---

## Do Now — Fix REG-2 (calculate_intensity_scale)

The most urgent fix is REG-2 because all D4f telemetry depends on the dataset-derived scale being computed correctly.

### Implement: `ptycho/train_pinn.py::calculate_intensity_scale`

Restore the dataset-derived priority order per `specs/spec-ptycho-core.md §Normalization Invariants`:

1. **Priority 1 — `dataset_intensity_stats`:** If `ptycho_data_container` has a `dataset_intensity_stats` dict with `batch_mean_sum_intensity > 1e-12`, compute `sqrt(nphotons / batch_mean_sum_intensity)`.
2. **Priority 2 — `_X_np` NumPy reduction:** If priority 1 unavailable, compute stats from `ptycho_data_container._X_np` using NumPy (CPU-only, no tensor cache population).
3. **Priority 3 — Closed-form fallback:** Only when both above are unavailable, use `sqrt(nphotons) / (N/2)`.

The current implementation ONLY does priority 3, which violates the spec.

### Tests

```bash
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
ARTIFACTS=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T233342Z
mkdir -p "$ARTIFACTS"/logs

# Collect tests first
pytest --collect-only tests/test_train_pinn.py::TestIntensityScale -q \
  2>&1 | tee "$ARTIFACTS"/logs/pytest_train_pinn_collect.log

# Run intensity scale tests
pytest tests/test_train_pinn.py::TestIntensityScale -v \
  2>&1 | tee "$ARTIFACTS"/logs/pytest_train_pinn.log

# Overall CLI smoke
pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
  2>&1 | tee "$ARTIFACTS"/logs/pytest_cli_smoke.log
```

---

## How-To Map — Reference Implementation

The prior working implementation looked like this (D4c/D4d from 2026-01-21):

```python
def calculate_intensity_scale(ptycho_data_container: PtychoDataContainer) -> float:
    """
    Calculate intensity scale per specs/spec-ptycho-core.md §Normalization Invariants.

    Priority:
    1. dataset_intensity_stats (if present) — preferred CPU path
    2. _X_np NumPy reduction — lazy-container safe
    3. Closed-form fallback sqrt(nphotons)/(N/2)
    """
    import numpy as np
    from . import params as p

    nphotons = p.get('nphotons')
    N = p.get('N')

    # Priority 1: Use pre-computed stats if available
    if hasattr(ptycho_data_container, 'dataset_intensity_stats'):
        stats = ptycho_data_container.dataset_intensity_stats
        if stats is not None and stats.get('batch_mean_sum_intensity', 0) > 1e-12:
            return float(np.sqrt(nphotons / stats['batch_mean_sum_intensity']))

    # Priority 2: Compute from NumPy backing (lazy-container safe)
    if hasattr(ptycho_data_container, '_X_np') and ptycho_data_container._X_np is not None:
        X_np = ptycho_data_container._X_np
        # Sum over spatial dims, mean over batch
        batch_mean = float(np.mean(np.sum(X_np ** 2, axis=(1, 2))))
        if batch_mean > 1e-12:
            return float(np.sqrt(nphotons / batch_mean))

    # Priority 3: Closed-form fallback
    return float(np.sqrt(nphotons) / (N / 2))
```

---

## Pitfalls To Avoid

1. Do **not** access `ptycho_data_container.X` — it forces tensor materialization and defeats lazy loading.
2. Ensure the function returns `float` (not tensor/array) for downstream JSON serialization.
3. Keep dead code (`count_photons` helper) removed — the reviewer noted it's still present.
4. Test both the NumPy and closed-form paths in `TestIntensityScale`.
5. Do **not** change loss weights or touch `ptycho/model.py`.
6. Guard `hasattr()` checks to avoid AttributeError on containers from different code paths.
7. Maintain dtype consistency (use float64 for reduction then cast to float).
8. Capture test logs in artifacts directory.
9. If tests fail with missing `dataset_intensity_stats` attribute, that's REG-1 — note it but still fix REG-2.
10. Check that existing tests still collect (>0) before running.

---

## If Blocked

- If `test_uses_dataset_stats` fails because `dataset_intensity_stats` attribute is missing from container (REG-1 not yet fixed), skip that test with `pytest.mark.xfail(reason="REG-1 not fixed")` and document in logs.
- If the test module doesn't collect any tests, check that `tests/test_train_pinn.py` exists and has `TestIntensityScale` class.
- Add blocking notes to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T233342Z/blockers.md`.

---

## Findings Applied (Mandatory)

- **PINN-CHUNKED-001:** The lazy-container design requires `_X_np` access for CPU-only stats. The fix must NOT access `.X` property.
- **CONFIG-001:** `calculate_intensity_scale()` reads `params.cfg` — ensure bridging is called beforehand in callers (already done in prior work).
- **specs/spec-ptycho-core.md §Normalization Invariants:** Dataset-derived mode `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])` is preferred over closed-form.

---

## Pointers

- `ptycho/train_pinn.py:165-180` — Current broken implementation (lines may vary).
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` — REGRESSION RECOVERY section with full checklist.
- `specs/spec-ptycho-core.md §Normalization Invariants` — Authoritative formula definitions.
- `docs/findings.md` — PINN-CHUNKED-001 regression note.
- `tests/test_train_pinn.py::TestIntensityScale` — Existing test class (may need restoration if tests were removed).

---

## Next Up (after REG-2)

1. **REG-3:** Restore `_update_max_position_jitter_from_offsets()` to `ptycho/workflows/components.py`.
2. **REG-4:** Restore or update `align_for_evaluation_with_registration()` in cropping.py / evaluate_metrics.py.
3. **REG-1:** Restore `dataset_intensity_stats` parameter to `PtychoDataContainer.__init__` and update callers.

---

## Doc Sync Plan (Conditional)

After code passes:
- Update `docs/findings.md` PINN-CHUNKED-001 status to "Regression fixed" with timestamp.
- Update `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` REGRESSION RECOVERY checklist to mark REG-2-FIX complete.
