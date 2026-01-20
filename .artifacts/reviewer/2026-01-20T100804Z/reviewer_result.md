# Reviewer Report — 2026-01-20T100804Z

## Integration Test Result: PASS

**Test command:**
```bash
RUN_TS=2026-01-20T100804Z RUN_LONG_INTEGRATION=1 \
  INTEGRATION_OUTPUT_DIR=.artifacts/integration_manual_1000_512/2026-01-20T100804Z/output \
  pytest tests/test_integration_manual_1000_512.py -v
```

**Result:** 1 passed in 97.59s (first attempt)

**Output location:** `.artifacts/integration_manual_1000_512/2026-01-20T100804Z/output`

---

## Review Window

- **Baseline commit:** `cf86219a` (SUPERVISOR AUTO iter=00420)
- **Current commit:** `e7f205d8` (DEBUG-SIM-LINES-DOSE-001 B0: add hypothesis enumeration phase)
- **Iterations reviewed:** 420–424

---

## Changes Since Last Review

### Code Changes (.py)
| File | Change Type | Summary |
|------|-------------|---------|
| `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_dose_stage.py` | Added | Compatibility runner for legacy dose_experiments |
| `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/tfa_stub/**` | Added | TensorFlow Addons stubs for legacy imports |
| `scripts/studies/sim_lines_4x/pipeline.py` | Modified | Added CONFIG-001 bridging calls |
| `scripts/studies/sim_lines_4x/run_gs*.py` | Modified | Minor updates |

### Documentation Changes (.md)
| File | Change Type | Summary |
|------|-------------|---------|
| `docs/GRIDSIZE_N_GROUPS_GUIDE.md` | Modified | Fixed broken links (DOC-HYGIENE-20260120) |
| `docs/fix_plan.md` | Modified | Added attempt history entries |
| `prompts/arch_writer.md` | Modified | Updated spec anchor references |
| `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` | Modified | Added Phase B0 hypothesis enumeration |
| `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md` | Modified | Updated turn summaries |

---

## Link Validation

### Fixed Links
- `docs/GRIDSIZE_N_GROUPS_GUIDE.md`: Updated to point to `docs/CONFIGURATION.md`, `../specs/data_contracts.md`, `docs/COMMANDS_REFERENCE.md`

### Remaining Issues
- `prompts/arch_writer.md` references `#pipeline-normative` and `#data-formats-normative` anchors that **do not exist** in the target spec files. These anchors were introduced as "fixes" but the targets were never created.

**Action needed:** Either create the missing anchors in `specs/spec-ptycho-workflow.md` and `specs/spec-ptycho-interfaces.md`, or revert to anchors that exist.

---

## Plan Quality Assessment: DEBUG-SIM-LINES-DOSE-001

### Findings (Critical)

1. **B2 Test Scope Overstated:** The original B2 conclusion claimed "probe scaling is no longer a suspect" but the test only proved legacy vs sim_lines code paths are equivalent. It did NOT test whether the dramatically different normalization characteristics between ideal (norm=0.0098) and custom (norm=1.1) probes cause downstream issues.

2. **Missing Hypothesis Enumeration:** The plan jumped into differential experiments (Phase B) without systematically enumerating root causes. This led to testing hypotheses in an ad-hoc order without clear rationale.

3. **Critical Context Missed:** The fact that ideal probe **worked in dose_experiments** but fails locally was not captured as a key constraint. This reframes the problem from "ideal probe numeric instability" to "ideal probe handling regression."

### Actions Taken

Added **Phase B0 — Hypothesis Enumeration** with:
- Ranked hypothesis table (10 candidates with test status)
- Critical context about dose_experiments success
- Gap analysis of B2's limited scope
- Decision tree for next experiments
- Isolation test (B0f: run gs1_custom) as next step

**Commit:** `e7f205d8`

### Current State

| Scenario | Status | Notes |
|----------|--------|-------|
| gs2_ideal | Healthy | Fixed by CONFIG-001 bridging |
| gs1_ideal | NaN at epoch 3 | Root cause unknown |

### Next Steps (per B0 decision tree)

1. Run `gs1_custom` (gridsize=1 + custom probe) to isolate failure variable
2. If gs1_custom works → investigate ideal probe regression (H-PROBE-IDEAL-REGRESSION)
3. If gs1_custom fails → investigate gridsize=1 numeric paths (H-GRIDSIZE-NUMERIC)

---

## Tech Debt Assessment

### Increased
- `prompts/arch_writer.md` now has broken anchor references (introduced by DOC-HYGIENE-20260120 "fix")
- A1b (dose_experiments ground truth) remains blocked due to Keras 3.x incompatibility

### Decreased
- Phase B0 improves debugging methodology
- B2 conclusion clarified to prevent future misinterpretation

### Net Assessment: Neutral to slightly improved
The hypothesis enumeration framework is valuable but the broken anchor references need cleanup.

---

## Spec/Implementation Consistency

No divergences detected in the changes reviewed. CONFIG-001 bridging was correctly implemented in both `pipeline.py` and `run_phase_c2_scenario.py`.

---

## docs/index.md Verification

The index references `specs/` as the authoritative spec root. No deprecated references to `docs/spec*` or `docs/architecture/` found in the changed files.

---

## Summary

| Category | Status |
|----------|--------|
| Integration test | PASS |
| Link validation | 1 issue (broken anchors in arch_writer.md) |
| Plan quality | Improved (B0 added) |
| Implementation | Correct (CONFIG-001 bridging) |
| Tech debt | Neutral |
| Agent progress | Partial — gs2 fixed, gs1 unresolved |

**Overall:** The agent is making progress but was stuck on an overstated conclusion (B2). The B0 framework should help unstick the investigation by providing a clear isolation test path.
