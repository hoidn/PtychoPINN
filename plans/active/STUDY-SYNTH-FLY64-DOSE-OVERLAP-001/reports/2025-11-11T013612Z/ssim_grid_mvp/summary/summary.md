# SSIM Grid MVP Implementation Summary

## Acceptance Focus
**AT-PREVIEW-PHASE-001**: Preview guard enforces phase-only content (rejects "amplitude" keyword)  
**AT-STUDY-001**: MS-SSIM formatted to ±0.000 (3 decimals), MAE to ±0.000000 (6 decimals)  
**AT-TYPE-PATH-001**: Emit POSIX-relative paths in metadata

**Module scope**: Study helpers / Phase G analysis automation

## SPEC/ADR Alignment

From `docs/findings.md:24` (PREVIEW-PHASE-001):
> The dense Phase G preview artifact (`analysis/metrics_delta_highlights_preview.txt`) must contain **only** the four phase deltas (MS-SSIM/MAE vs Baseline/PtyChi) with explicit ± signs; any `amplitude` text or extra tokens indicates corruption and must fail validation.

From `docs/findings.md:16` (STUDY-001):
> On fly64 experiments the baseline model outperformed PtychoPINN by ~6–10 dB, contradicting expectations and motivating architecture review. Report MS-SSIM/MAE deltas with explicit ± signs and phase emphasis.

From `docs/findings.md:21` (TYPE-PATH-001):
> PyTorch workflows failed with AttributeError/TypeError when string paths from TrainingConfig were passed to functions expecting Path objects. Prevention: Normalize path fields via Path() or apply runtime coercion at module boundaries. Emit POSIX-relative paths in artifacts.

## Implementation Summary

### Helper: `ssim_grid.py` (Tier-2 CLI)
**Location**: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/ssim_grid.py:1-217`

**Key functions**:
- `validate_preview_phase_only()` (lines 34-60): Enforces PREVIEW-PHASE-001 by scanning preview file for "amplitude" keyword
- `format_delta()` (lines 63-79): Formats numeric deltas with explicit ± signs and configurable precision
- `generate_ssim_grid_summary()` (lines 82-177): Orchestrates preview validation → JSON loading → markdown table generation

**Exit codes**:
- 0: Success
- 1: Preview guard failure (amplitude contamination)
- 2: Missing/invalid input files
- 3: Other errors

**Precision formatting**:
- MS-SSIM: 3 decimal places (±0.000)
- MAE: 6 decimal places (±0.000000)

### Test: `test_ssim_grid.py`
**Location**: `tests/study/test_ssim_grid.py:1-144`

**Test structure** (TDD smoke test):
1. **RED phase** (lines 68-95): Creates preview with "amplitude" keyword, asserts helper exits with non-zero code
2. **GREEN phase** (lines 97-143): Creates phase-only preview, asserts helper succeeds and emits correctly formatted markdown

**Validations**:
- Helper reads `metrics_delta_summary.json` structure correctly
- Preview guard rejects amplitude contamination (PREVIEW-PHASE-001)
- Output contains MS-SSIM ±0.000 and MAE ±0.000000 formatted deltas (STUDY-001)
- Output uses POSIX-relative paths (TYPE-PATH-001)
- Helper executes via subprocess (proves CLI contract)

## Artifacts Generated

### Test Execution Logs
- `green/pytest_ssim_grid_smoke.log`: Full pytest run (1 test collected, PASSED)
- `red/helper_preview_guard_failure.log`: Demonstrates preview guard rejection with amplitude contamination
- `collect/pytest_collect_ssim_grid.log`: Proves selector `tests/study/test_ssim_grid.py::test_smoke_ssim_grid` collects 1 test

### Sample Outputs
- `analysis/metrics_delta_summary.json`: Sample input JSON with vs_Baseline and vs_PtyChi deltas
- `analysis/metrics_delta_highlights_preview_GREEN.txt`: Valid phase-only preview (4 lines)
- `analysis/metrics_delta_highlights_preview_RED.txt`: Invalid preview with "amplitude" (demonstrates guard)
- `analysis/ssim_grid_summary.md`: Generated markdown table output with phase-only deltas

## Test Results

**Selector**: `pytest tests/study/test_ssim_grid.py::test_smoke_ssim_grid -vv`

**Collection**: 1 test collected (verified via `pytest --collect-only`)

**Execution**: PASSED (0.90s)

**Coverage**:
- ✅ Helper scaffold exists and is executable
- ✅ RED phase: Preview guard correctly rejects amplitude contamination (exit code 1)
- ✅ GREEN phase: Helper succeeds with phase-only preview (exit code 0)
- ✅ Output markdown contains correctly formatted deltas (±0.000 / ±0.000000)
- ✅ No amplitude data in output (phase-only enforcement)
- ✅ POSIX-relative paths in metadata

## Search Evidence

Existing helpers in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/`:
- `verify_dense_pipeline_artifacts.py:309-329`: Reference implementation of `validate_metrics_delta_highlights()`
- Template pattern: argparse + docstring + exit codes + Path types

No duplication detected; `ssim_grid.py` is purpose-built for Phase G markdown table generation.

## Next Steps

1. Run full orchestrator (`bin/run_phase_g_dense.py`) to generate real Phase G artifacts
2. Invoke `ssim_grid.py` on orchestrator output to publish MS-SSIM/MAE delta tables
3. Update `docs/TESTING_GUIDE.md` §2 (Phase G Delta Metrics Persistence) to reference `ssim_grid.py`
4. Update `docs/development/TEST_SUITE_INDEX.md` (Phase G section) with new test selector

## Findings Applied

- **PREVIEW-PHASE-001**: Enforced via `validate_preview_phase_only()` with line-by-line scanning
- **STUDY-001**: Formatted via `format_delta()` with precision parameter (3 for MS-SSIM, 6 for MAE)
- **TYPE-PATH-001**: Helper uses `Path()` types throughout; emits POSIX-style relative paths in markdown metadata
- **TEST-CLI-001**: Test uses subprocess.run() to prove CLI contract; captures both RED and GREEN exit codes

## Compliance Checklist

- [x] Acceptance & module scope declared
- [x] SPEC/ADR quotes present
- [x] Search-first evidence captured
- [x] Helper implements real behavior (no placeholders)
- [x] Test authored (pytest native, not unittest.TestCase)
- [x] RED phase captured (amplitude guard rejection)
- [x] GREEN phase captured (phase-only success)
- [x] Collection verified (1 test collects)
- [x] Full test suite running (in progress)
- [x] Artifacts organized under hub reports directory
- [x] Sample outputs archived
- [x] Exit criteria met: helper executes via pytest and CLI, preview guard enforced

