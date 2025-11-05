### Turn Summary
Implemented aggregate metrics computation in summarize_phase_g_outputs (mean & best MS-SSIM, mean MAE per model across all jobs).
Added aggregate_metrics field to JSON output and ## Aggregate Metrics section to Markdown; test_summarize_phase_g_outputs now validates both JSON structure and Markdown formatting.
All targeted tests GREEN (0.87-0.89s); full suite 416 passed/17 skipped/1 pre-existing fail in 369.29s.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T010500Z/phase_g_dense_full_execution/ (green/collect logs, pytest_full.log)

---

## Implementation Details

**Problem Statement:**
Per `input.md` Do Now, Phase G orchestrator needed aggregate metric support to compute cross-job statistics (mean & best MS-SSIM amplitude/phase, mean MAE amplitude/phase) and emit them in both JSON (`aggregate_metrics`) and Markdown (`## Aggregate Metrics`) outputs.

**SPEC Lines Implemented:**
From `input.md:13-16`:
- Implement aggregate metrics in `summarize_phase_g_outputs` preserving existing per-job data with deterministic ordering
- Extend Markdown writer with `## Aggregate Metrics` section summarizing mean/best MS-SSIM + mean MAE per model
- Update test to validate both JSON (`aggregate_metrics`) and Markdown section contents

**Files Modified:**
1. `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:326-489`
   - Added `model_metrics_collector` dict to accumulate ms_ssim/mae values across jobs (lines 334-336)
   - Enhanced metric extraction loop to collect amplitude/phase values for ms_ssim and mae (lines 381-388)
   - Computed aggregate metrics with sorted model/metric keys for deterministic output (lines 404-433)
   - Added `aggregate_metrics` field to summary_data JSON (line 433)
   - Implemented Markdown `## Aggregate Metrics` section with per-model MS-SSIM/MAE tables (lines 452-489)

2. `tests/study/test_phase_g_dense_orchestrator.py:209-281`
   - Added JSON validation for `aggregate_metrics` field (lines 209-252)
   - Validated PtychoPINN aggregates: mean_ms_ssim_amp=0.9475, best=0.95, mean_mae_amp=0.026 (lines 217-233)
   - Validated Baseline aggregates: mean_ms_ssim_amp=0.9275, best=0.93, mean_mae_amp=0.031 (lines 239-252)
   - Added Markdown validation for `## Aggregate Metrics` section and table structure (lines 267-281)

**Test Results:**
- Targeted selectors: All GREEN (0.87-0.89s)
  - `test_summarize_phase_g_outputs`: PASSED 0.87s
  - `test_validate_phase_c_metadata_accepts_valid_metadata`: PASSED 0.89s
  - `test_run_phase_g_dense_collect_only_generates_commands`: PASSED 0.85s
- Selector inventory: 10 tests collected (no change from baseline)
- Full suite: 416 passed/17 skipped/1 pre-existing fail in 369.29s (0:06:09)

**Acceptance Criteria Met:**
- ✅ Aggregate metrics computed per model (mean & best MS-SSIM, mean MAE)
- ✅ JSON output includes `aggregate_metrics` field with deterministic structure
- ✅ Markdown includes `## Aggregate Metrics` section with formatted tables
- ✅ Test validates both JSON and Markdown aggregate content
- ✅ Deterministic ordering via `sorted()` on model/metric keys
- ✅ Float formatting: JSON preserves raw floats, Markdown uses 3-decimal precision

**Findings Applied:**
- TYPE-PATH-001: Path normalization via `Path(hub).resolve()`
- CONFIG-001: No params.cfg interaction (orchestrator remains stateless helper)
- DATA-001: Preserves existing NPZ contract (only reads CSV outputs)

**Next Steps:**
Per `input.md:82`, when dense evidence is green, mirror aggregate logic for sparse view to complete Phase G coverage. However, nucleus complete per Ralph principle—ship aggregate feature implementation rather than expanding to full pipeline execution in this loop.
