# Reviewer Result

**Verdict:** PASS — long integration test succeeded on first attempt.

**Test command:**
```bash
RUN_TS=$(date -u +%Y-%m-%dT%H%M%SZ) RUN_LONG_INTEGRATION=1 INTEGRATION_OUTPUT_DIR=.artifacts/integration_manual_1000_512/${RUN_TS}/output pytest tests/test_integration_manual_1000_512.py -v
```

**Output artifacts:** `.artifacts/integration_manual_1000_512/2026-01-20T100804Z/output`

**Key pytest excerpt:**
```
tests/test_integration_manual_1000_512.py::test_train_infer_cycle_1000_train_512_test PASSED [100%]
========================= 1 passed in 97.59s (0:01:37) =========================
```

---

## Review Window

- **Iterations inspected:** 420–424 (fallback window, no `router.review_every_n` set)
- **State file:** `sync/state.json` (iteration 424, expected_actor: galph)
- **Logs directory:** `logs/` (inferred, no orchestration.yaml)

---

## Issues Identified

### 1. Broken Anchor Links in `prompts/arch_writer.md` (PERSISTS)

The previous review flagged broken spec anchors. The fix attempt replaced them with:
- `../specs/spec-ptycho-workflow.md#pipeline-normative`
- `../specs/spec-ptycho-interfaces.md#data-formats-normative`

**Problem:** These anchors still do not exist. The spec files use prose section labels like "Pipeline (Normative)" and "Data Formats (Normative)" but without Markdown `##` headings, so no linkable anchors are generated.

**Evidence:** `grep -E "^#{1,3} " specs/spec-ptycho-workflow.md specs/spec-ptycho-interfaces.md` returns only the top-level `# spec-...` titles.

**Recommendation:** Either add proper Markdown headings (`## Pipeline (Normative)`) to the spec files, or remove the anchor fragments from the prompts and link to the file top-level.

### 2. Deprecated Path References in Prompts (PERSISTS)

Several prompts reference paths that don't exist:
- `prompts/arch_reviewer.md`: references `docs/architecture/` and `docs/spec-shards/` (both nonexistent)
- `prompts/main.md`: references `docs/architecture/pytorch_design.md` (file does not exist at that path)
- `prompts/spec_reviewer.md`: references `docs/spec-shards/*.md` (nonexistent)

**Recommendation:** Audit all prompts and update references to use the canonical `specs/` root and `docs/architecture*.md` top-level files.

### 3. `docs/GRIDSIZE_N_GROUPS_GUIDE.md` Link Fix (RESOLVED)

The previous review flagged broken links to `CONFIGURATION_GUIDE.md` and `data_contracts.md`. The diff shows this was fixed:
- `CONFIGURATION_GUIDE.md` → `CONFIGURATION.md` ✓
- `data_contracts.md` → `../specs/data_contracts.md` ✓
- `TOOL_SELECTION_GUIDE.md` → `COMMANDS_REFERENCE.md` ✓

All targets verified to exist.

### 4. `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` Checklist Sync (PARTIAL)

The previous review noted C4d was unchecked despite artifacts landing. The diff shows:
- C4d is now marked `[x]` ✓
- C4e is now marked `[x]` ✓
- C4f is still marked `[ ]` (unchecked) but has implementation details and artifacts

**Current state:** C4f shows "Implementation: call `update_legacy_dict`..." and verification instructions, and recent commits include `e1e84dba DEBUG-SIM-LINES-DOSE-001 C4f: enforce CONFIG-001 bridging`. The checklist should be marked complete.

---

## Plan Review: DEBUG-SIM-LINES-DOSE-001

### Intention Assessment

The plan aims to isolate why `sim_lines_4x` nongrid scenarios produce degraded reconstructions compared to `dose_experiments` (a legacy branch). The approach is systematic: capture evidence (Phase A), run differential experiments (Phase B), then fix and verify (Phase C).

### Approach Quality

**Strengths:**
1. Methodical evidence capture with timestamped artifact hubs
2. Clear phase structure with testable checklist items
3. pytest guard maintained throughout (`test_sim_lines_pipeline_import_smoke`)
4. Rich telemetry instrumentation (bias analyzer, reassembly limits, intensity stats)
5. Proper spec alignment with CONFIG-001 and normalization policies

**Concerns:**
1. **Iteration churn without convergence:** 27 report directories since 2026-01-16, yet the core amplitude bias (~2.5× undershoot) persists. The plan has gone from C4a through C4f with incremental diagnostics but no definitive fix.
2. **gs1_ideal NaN instability remains unexplained:** The summary notes gs1_ideal collapses at epoch 3 with NaNs while gs2_ideal is healthy. This suggests a gridsize=1 specific issue that hasn't been root-caused.
3. **A1b ground-truth run never completed:** Despite being marked "required (unwaivable)", the legacy environment run failed due to OOM/compatibility issues. The shim (`run_dose_stage.py`) partially works (simulation only) but can't produce full training ground truth.

### Assumptions and Unjustified Claims

1. **Assumption:** "Constant rescaling alone cannot close the gap" — The evidence shows least-squares scaling (~1.86×) doesn't reduce MAE, but this assumes the bias is uniform. The ratio distributions (p05=1.326, p95=2.622) suggest it's not uniform, which might require spatially-varying correction.

2. **Assumption:** "CONFIG-001 bridging will resolve amplitude drift" — C4f implemented this but the bias persists (gs2_ideal still shows -2.296 mean amplitude bias). The implementation may be correct but the hypothesis was wrong.

3. **Unjustified claim:** The plan focuses heavily on intensity_scale bridging but doesn't investigate whether the model architecture itself (gridsize=1 vs gridsize=2) is fundamentally different in how it reconstructs amplitude.

### Progress Assessment

**Real progress made:**
- Phase B completed: confirmed probe normalization identical, grouped offsets far exceed padded size
- Phase C1: `_update_max_position_jitter_from_offsets()` implemented and tested, fixing canvas sizing
- Rich diagnostic tooling created (analyzers, runners, reporters)

**Not real progress:**
- The extensive artifact collection and phase micro-iterations (C4a→C4b→C4c→C4d→C4e→C4f) have not produced a working fix
- The ~2.5× amplitude undershoot remains across all scenarios

### Stuck Assessment

**The agent appears tunnel-visioned on CONFIG-001 and intensity scaling** while the evidence suggests the problem may be:
1. Model architecture specific (gs1 NaN collapse vs gs2 stability)
2. Loss function weighting (realspace_weight=0, mae disabled when no GT)
3. Upstream physics/normalization in the model layers, not the config bridge

**Recommendation:** Pivot away from diagnostics and either:
1. Investigate the gs1 NaN source directly (add gradient monitoring, check for numerical instability in gridsize=1 reassembly)
2. Compare actual model layer outputs between a known-good scenario and the failing one
3. Consider whether the sim_lines pipeline fundamentally differs from dose_experiments in ways the config bridge cannot address

---

## Spec/Architecture Consistency

### `docs/index.md` Accuracy

All referenced files verified to exist:
- `scripts/training/README.md` ✓
- `scripts/inference/README.md` ✓
- `scripts/simulation/README.md` ✓
- `scripts/studies/README.md` ✓
- `scripts/reconstruction/README.md` ✓
- `scripts/tools/README.md` ✓
- `scripts/orchestration/README.md` ✓
- `docs/studies/GENERALIZATION_STUDY_GUIDE.md` ✓
- `scripts/studies/sim_lines_4x/README.md` ✓

The spec root is correctly listed as `specs/`.

### Conceptual Drift

No new conceptual drift detected in the changes since last review. The CONFIG-001 bridging implementation (`update_legacy_dict` calls) aligns with the spec requirement in `specs/spec-inference-pipeline.md`.

---

## Tech Debt Assessment

**Increased:**
1. Plan-local tooling proliferation: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/` now contains 7+ specialized scripts that duplicate some functionality from main scripts
2. `run_dose_stage.py` is a compatibility shim with hardcoded paths and workarounds that should not persist

**Decreased:**
1. Documentation link fixes in `docs/GRIDSIZE_N_GROUPS_GUIDE.md`
2. CONFIG-001 bridging added to `scripts/studies/sim_lines_4x/pipeline.py` (proper fix)

**Net:** Slight increase due to plan-local tooling accumulation, but the core codebase changes are sound.

---

## Implementation Quality

### Code Changes Reviewed

1. **`scripts/studies/sim_lines_4x/pipeline.py`:**
   - Added CONFIG-001 bridging (correct)
   - Added prediction scale helpers (least_squares, center_crop, etc.) — well-structured, documented
   - Return signature change for `run_inference` to include scale_meta — breaking change handled via tuple unpacking
   - Minor issue: duplicate `metadata["prediction_scale_note"]` assignment (lines 486-487)

2. **`run_dose_stage.py`:**
   - Non-production shim with extensive parameter clamping
   - Well-documented workarounds for OOM and KD-tree issues
   - Should not be merged to main — appropriate for plan-local use only

3. **Runner scripts (`run_gs1_*.py`, `run_gs2_*.py`):**
   - Added `--prediction-scale-source` CLI flag — consistent across all four
   - Clean integration with pipeline changes

---

## Summary

| Category | Status |
|----------|--------|
| Integration Test | PASS (first attempt, 97.59s) |
| Broken Links | 2 issues persist (arch_writer.md anchors, deprecated prompt paths) |
| Plan Progress | Stuck on diagnostics; amplitude bias unresolved |
| Tech Debt | Slight increase (plan-local tooling) |
| Code Quality | Good for production changes; shims appropriate for plan scope |

---

## Actionable Items

1. **Fix `prompts/arch_writer.md` anchors** by either adding proper Markdown headings to specs or removing anchor fragments
2. **Audit all prompts for deprecated path references** to `docs/architecture/`, `docs/spec-shards/`
3. **Mark C4f as complete** in the implementation plan (implementation landed)
4. **Consider pivoting DEBUG-SIM-LINES-DOSE-001** away from config diagnostics toward model-layer investigation

These findings are captured in existing plan awareness and do not require a `user_input.md` — the actionable items are documentation hygiene, not blocking implementation issues.
