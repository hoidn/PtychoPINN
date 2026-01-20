Summary: Phase D4b ROOT CAUSE IDENTIFIED — `train_pinn.py:calculate_intensity_scale()` uses fallback instead of dataset-derived scale. Prepare D4c fix proposal for supervisor approval.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D4c fix preparation (requires core module approval)
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/

ROOT CAUSE SUMMARY:
- `ptycho/train_pinn.py:calculate_intensity_scale()` (lines 165-180) uses closed-form fallback `sqrt(nphotons)/(N/2)` instead of dataset-derived scale
- The function receives `ptycho_data_container.X` but ignores it — dead code at lines 173-175 with unimplemented TODO
- Dataset-derived scale=577.74 vs Fallback scale=988.21 (ratio=0.585) — a 1.7× mismatch
- Per `specs/spec-ptycho-core.md §Normalization Invariants` lines 87-89: dataset-derived mode is preferred

PROPOSED FIX (D4c):
Modify `ptycho/train_pinn.py:calculate_intensity_scale()` to compute actual dataset-derived scale:

```python
def calculate_intensity_scale(ptycho_data_container: PtychoDataContainer) -> float:
    import tensorflow as tf
    import numpy as np
    from . import params as p

    # Dataset-derived mode (preferred) per specs/spec-ptycho-core.md §Normalization Invariants
    # s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])
    X = ptycho_data_container.X
    mean_photons = np.mean(np.sum(X.numpy()**2, axis=(1, 2)))
    intensity_scale = np.sqrt(p.get('nphotons') / mean_photons)

    return float(intensity_scale)
```

APPROVAL REQUIRED:
- This change modifies `ptycho/train_pinn.py` which is a core module
- Per CLAUDE.md directive #6: "Treat core physics/model code as stable. Do not modify [...] unless the active plan explicitly authorizes it."
- The implementation plan (D4c) documents the approval requirement

Do Now (Supervisor scope — NOT implementation):
1. Review the root cause analysis documented in:
   - docs/fix_plan.md (2026-01-20T234500Z entry)
   - plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md (D4b checklist)
   - plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md (latest turn summary)
2. Decide whether to:
   a) APPROVE D4c fix in `ptycho/train_pinn.py` (core module modification)
   b) REQUEST additional evidence (e.g., run scenarios with manually patched scale)
   c) DEFER to a separate initiative for core module changes
3. If approving, update input.md with explicit authorization for D4c implementation

Evidence Supporting Fix:
- D4a telemetry in plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/
- Dataset scale=577.74 vs Fallback=988.21 (ratio=0.585)
- E_batch[Σ|Ψ|²]=2995.97 vs assumed (N/2)²=1024 — 2.9× discrepancy
- diffsim.py:scale_nphotons() correctly implements dataset-derived scale (reference implementation)

Findings Applied (Mandatory):
- CONFIG-001 — Always sync `params.cfg` before legacy modules touch grouped data.
- SIM-LINES-CONFIG-001 — Maintain the plan-local CONFIG-001 bridge.
- NORMALIZATION-001 — Dataset vs fallback scales are physics normalization; cite spec.
- H-SCALE-MISMATCH — CONFIRMED: `train_pinn.py:calculate_intensity_scale()` uses fallback instead of dataset-derived scale.

Pointers:
- ptycho/train_pinn.py:165-180 — Current `calculate_intensity_scale()` implementation (broken)
- ptycho/diffsim.py:68-77 — Reference `scale_nphotons()` implementation (correct)
- specs/spec-ptycho-core.md:87-89 — Normative dataset-derived mode definition
- docs/fix_plan.md:309-323 — D4b root cause analysis entry
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:374-378 — D4c task definition
