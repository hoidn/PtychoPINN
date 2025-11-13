# PtychoPINN Testing Guide

This guide describes how to run tests and validate study workflows.

## Running the Full Test Suite

Preferred (pytest)
- The project standard is to run tests via pytest selectors:
  ```bash
  pytest tests/ -q
  ```

Legacy unittest modules
- A few legacy suites still use `unittest` discovery. Run them explicitly when needed:
  ```bash
  python -m unittest tests.image.test_cropping -v
  python -m unittest tests.image.test_registration -v
  ```

Notes
- Always run from the repo root so imports and relative paths resolve correctly.

### Global Integration Marker

- Use the `integration` marker to run project‑level end‑to‑end integration smoke tests without hardcoding paths:
  - Run marker: `pytest -v -m integration`
  - Repository‑specific aliases may exist (e.g., `tf_integration` in this repo). The canonical NodeID for the TensorFlow workflow is: `tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle`.
- The integration test also runs during the comprehensive full‑suite gate; the marker is useful for a fast, early signal after implementation changes to production code.

## Test Types

The PtychoPINN test suite includes two main categories of tests:

### Unit Tests

Fast, focused tests that validate individual components and functions in isolation.

- **Location:** `tests/test_*.py` files
- **Purpose:** Verify specific functionality of individual modules
- **Examples:**
  - `test_cli_args.py` - Tests command-line argument parsing
  - `test_misc.py` - Tests utility functions
  - `test_model_manager.py` - Tests model management functionality
  - `test_tf_helper.py` - Tests TensorFlow helper functions
- **Execution time:** Typically < 1 second per test
- **Dependencies:** Minimal, often using mocks or small test fixtures

### Integration Tests

Comprehensive tests that validate end-to-end workflows and interactions between components.

- **Location:** `tests/test_integration_*.py` files
- **Purpose:** Ensure complete workflows function correctly across process boundaries
- **Key test:** `test_integration_workflow.py`
  - Validates the complete train → save → load → infer cycle
  - Runs training and inference as separate subprocess calls
  - Verifies model persistence and restoration across processes
  - This is the ultimate check for the model persistence layer
- **Execution time:** Can take several seconds to minutes
- **Dependencies:** Requires actual data files and complete environment setup

## The Critical Integration Test

### test_integration_workflow.py

This test is particularly important as it validates the entire machine learning workflow:

1. **Training Phase:** Trains a model using a subprocess call to `scripts/training/train.py`
2. **Save Phase:** Verifies that the model artifact (`.h5.zip`) is correctly saved
3. **Load Phase:** Loads the saved model in a new process via `scripts/inference/inference.py`
4. **Inference Phase:** Runs inference on test data and generates output visualizations
5. **Validation:** Checks that reconstruction images are created and have valid content

This test ensures that:
- Model serialization and deserialization work correctly
- Training and inference scripts can be used independently
- The saved model format is compatible across different execution contexts
- The complete user workflow functions as expected

### Study Tests

Specialized tests for the synthetic fly64 dose/overlap study.

- **Location:** `tests/study/test_dose_overlap_*.py`
- **Purpose:** Validate dataset generation, overlap metrics (explicit `s_img`/`n_groups`), and study orchestration per `specs/overlap_metrics.md` and `docs/GRIDSIZE_N_GROUPS_GUIDE.md`. Dense/sparse labels and spacing/packing acceptance gates are deprecated for Phase D in this study.
- **Examples:**
  - `test_dose_overlap_design.py` - Tests study design configuration and validation
  - `test_dose_overlap_generation.py` - Tests Phase C dataset generation pipeline
  - `test_dose_overlap_overlap.py` - Tests Phase D overlap view filtering and metrics
  - `test_dose_overlap_dataset_contract.py` - Tests DATA-001 contract enforcement
  - `test_dose_overlap_training.py` - Tests Phase E training job matrix enumeration and run helper
- **Key selectors:**
  ```bash
  # Run all study tests
  pytest tests/study/ -v

  # Phase D overlap metrics (unit + integration)
  pytest tests/study/test_dose_overlap_overlap.py -v
  pytest tests/study/test_dose_overlap_overlap.py::test_overlap_metrics_bundle -vv

  # Run Phase E training tests
  pytest tests/study/test_dose_overlap_training.py -v

  # Run specific training job builder test
  pytest tests/study/test_dose_overlap_training.py::test_build_training_jobs_matrix -vv

  # Run run_training_job helper tests
  pytest tests/study/test_dose_overlap_training.py -k run_training_job -vv

  # Run CLI tests (Phase E4-E5)
  pytest tests/study/test_dose_overlap_training.py -k training_cli -vv
  pytest tests/study/test_dose_overlap_training.py::test_training_cli_filters_jobs -vv
  pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv

  # Run skip summary persistence test (Phase E5)
  pytest tests/study/test_dose_overlap_training.py::test_training_cli_manifest_and_bridging -vv

  # Collect all training tests
  pytest tests/study/test_dose_overlap_training.py --collect-only -vv
  ```

**Phase E5 Skip Summary Evidence:**

The training CLI now persists skip metadata to a standalone `skip_summary.json` file when views are absent due to Phase D filtering. The `test_training_cli_manifest_and_bridging` test validates:

- Standalone `skip_summary.json` file exists under `--artifact-root`
- Manifest includes `skip_summary_path` field with relative path
- Skip summary schema matches: `{timestamp, skipped_views: [{dose, view, reason}], skipped_count}`
- Content consistency between standalone skip summary and manifest inline fields

**Deterministic CLI command with dry-run for skip summary demonstration:**

```bash
python -m studies.fly64_dose_overlap.training \
  --phase-c-root tmp/phase_c_training_evidence \
  --phase-d-root tmp/phase_d_training_evidence \
  --artifact-root tmp/training_artifacts \
  --dose 1000 \
  --dry-run
```

The dry-run mode creates artifact directory structure, writes manifest and skip summary, but skips actual training. Phase E5 evidence captured at: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T170500Z/phase_e_training_e5_real_run_baseline/`
- **Execution time:** Typically < 5 seconds per test (lightweight synthetic data)
- **Dependencies:** NumPy, study design configuration, no external datasets required

**Phase F Pty-Chi LSQML Baseline Reconstruction:**

The reconstruction CLI orchestrates pty-chi LSQML baseline runs across the dose/view matrix. The `test_dose_overlap_reconstruction.py` test module validates:

- Job enumeration (18 jobs: 3 doses × 2 views × 3 splits)
- Subprocess dispatch with CLI argument handoff (`--input-npz`, `--output-dir`, `--algorithm`, `--num-epochs`, `--n-images`)
- Dry-run filtering and artifact emission (manifest, skip summary)
- Live execution with per-job logging and execution telemetry

**Key selectors:**

```bash
# Run all Phase F reconstruction tests
pytest tests/study/test_dose_overlap_reconstruction.py -v

# Run manifest construction test
pytest tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest -vv

# Run subprocess dispatch test
pytest tests/study/test_dose_overlap_reconstruction.py::test_run_ptychi_job_invokes_script -vv

# Run dry-run filtering test
pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_filters_dry_run -vv

# Run live execution test with logging
pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv

# Run all ptychi-tagged tests
pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv

# Collect all reconstruction tests
pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv
```

**Deterministic CLI command with dense/test baseline:**

```bash
AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction \
  --phase-c-root tmp/phase_c_f2_cli \
  --phase-d-root tmp/phase_d_f2_cli \
  --artifact-root tmp/reconstruction_artifacts \
  --dose 1000 \
  --view dense \
  --split test \
  --allow-missing-phase-d
```

**Dry-run command:**

```bash
python -m studies.fly64_dose_overlap.reconstruction \
  --phase-c-root tmp/phase_c_f2_cli \
  --phase-d-root tmp/phase_d_f2_cli \
  --artifact-root tmp/reconstruction_artifacts \
  --dose 1000 \
  --view dense \
  --dry-run
```

The dry-run mode enumerates jobs, writes manifest and skip summary, but skips subprocess execution. Phase F dense/test evidence captured at: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/`

- **Execution time:** Typically < 10 seconds per test (subprocess mocking); real LSQML runs take 100+ epochs (varies by dataset)
- **Dependencies:** NumPy, Phase C/D datasets, `scripts/reconstruction/ptychi_reconstruct_tike.py`

**Phase G Orchestrator Metrics Summary:**

Tests for the Phase C→G dense execution orchestrator and metrics aggregation helper.

- **Files:**
  - `test_phase_g_dense_orchestrator.py` - Tests `summarize_phase_g_outputs()` helper, `validate_phase_c_metadata()` guard, and `prepare_hub()` helper in `bin/run_phase_g_dense.py`
  - `test_phase_g_dense_metrics_report.py` - Tests `report_phase_g_dense_metrics.py` CLI helper for aggregate metrics reporting with delta computation
- **Purpose:** Validate orchestrator metrics summary extraction from Phase G comparison results, Phase C metadata integrity enforcement, hub preparation with stale output detection, and aggregate reporting with baseline deltas
- **Key selectors:**
  ```bash
  # Run Phase G orchestrator summary helper test
  pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv

  # Run failure mode tests (missing manifest, n_failed > 0, missing CSV)
  pytest tests/study/test_phase_g_dense_orchestrator.py -k fails -vv

  # Run Phase C metadata guard tests (requires _metadata, requires transpose_rename_convert, accepts valid)
  pytest tests/study/test_phase_g_dense_orchestrator.py -k metadata -vv

  # Run specific metadata guard tests
  pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv
  pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_canonical_transform -vv
  pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_accepts_valid_metadata -vv

  # Run hub preparation tests (stale detection + clobber mode)
  pytest tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_detects_stale_outputs -vv
  pytest tests/study/test_phase_g_dense_orchestrator.py::test_prepare_hub_clobbers_previous_outputs -vv

  # Run metrics reporting helper tests
  pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics -vv
  pytest tests/study/test_phase_g_dense_metrics_report.py::test_report_phase_g_dense_metrics_missing_model_fails -vv

  # Collect all orchestrator tests (10 tests)
  pytest tests/study/test_phase_g_dense_orchestrator.py --collect-only -vv

  # Collect all reporting helper tests (2 tests)
  pytest tests/study/test_phase_g_dense_metrics_report.py --collect-only -vv
  ```

**Hub Preparation Helper:**

The `prepare_hub()` helper detects and manages stale Phase C outputs to prevent accidental overwrites. The helper:

- Normalizes hub path via TYPE-PATH-001 (Path.resolve())
- Detects existing Phase C outputs under `{hub}/data/phase_c/`
- Default behavior (clobber=False): raises RuntimeError with actionable `--clobber` guidance
- Clobber mode (clobber=True): archives stale outputs to timestamped `{hub}/archive/phase_c_<timestamp>/` directory
- Read-only by default to prevent accidental data loss
- Preserves evidence via archiving instead of deletion

Tests validate two behaviors: stale detection with read-only error (no file deletions), and clobber mode with clean hub state (all .npz files removed/archived).

**Phase C Metadata Guard:**

The `validate_phase_c_metadata()` guard enforces Phase C dataset integrity requirements before downstream Phase D/E/F/G processing. The guard validates:

- Phase C NPZ outputs contain required `_metadata` field (DATA-001 contract)
- Metadata includes `transpose_rename_convert` transformation in `data_transformations` list (canonical format enforcement)
- Read-only validation (no mutations or deletions of Phase C outputs)
- TYPE-PATH-001 compliance (Path normalization)
- Actionable RuntimeError messages with missing field details

The guard ensures Phase C outputs have undergone format canonicalization via `scripts/tools/transpose_rename_convert_tool.py`, preventing silent format mismatches in downstream phases. Tests validate three behaviors: missing metadata (baseline), missing canonical transformation (new requirement), and valid metadata acceptance.

**Phase G Orchestrator Summary Helper:**

The `summarize_phase_g_outputs()` helper validates Phase G execution manifest, extracts metrics from per-job `comparison_metrics.csv` files (MS-SSIM, MAE, computation time), and emits JSON + Markdown summaries to `{hub}/analysis/`. The test validates:

- Manifest parsing from `{hub}/analysis/comparison_manifest.json`
- Fail-fast on `n_failed > 0` or missing CSV files
- Metrics extraction (amplitude/phase/value columns) from tidy CSV format
- Deterministic JSON output (`metrics_summary.json`) with aggregate metrics (mean/best per model)
- Formatted Markdown tables (`metrics_summary.md`) by model/job with aggregate section

**Phase G Metrics Reporting Helper:**

The `report_phase_g_dense_metrics.py` CLI helper parses `metrics_summary.json` (output from `summarize_phase_g_outputs`) and generates aggregate reports with delta tables comparing PtychoPINN against Baseline and PtyChi. The helper:

- Loads and validates `metrics_summary.json` structure (fails on missing `aggregate_metrics`)
- Requires presence of all three models (PtychoPINN, Baseline, PtyChi) for delta computation
- Computes deltas: PtychoPINN - Baseline and PtychoPINN - PtyChi for MS-SSIM and MAE (amplitude/phase)
- Formats output with 3-decimal precision and signed deltas (+/- prefix)
- Emits report to stdout + optional Markdown file via `--output` flag
- Optionally generates concise highlights summary via `--highlights` flag (`aggregate_highlights.txt`)
- Exit codes: 0 (success), 1 (required model missing), 2 (invalid input/IO error)

Tests validate two behaviors: successful reporting with all models present (verifies aggregate tables, delta sections, 3-decimal formatting, stdout/Markdown outputs, optional highlights file), and graceful failure with actionable error message when required models are missing.

**Usage:**
```bash
# Generate report from metrics_summary.json with highlights
python scripts/study/report_dense_metrics.py \
  --metrics plans/active/.../reports/<timestamp>/phase_g_dense_execution/analysis/metrics_summary.json \
  --output plans/active/.../reports/<timestamp>/phase_g_dense_execution/analysis/aggregate_report.md \
  --highlights plans/active/.../reports/<timestamp>/phase_g_dense_execution/analysis/aggregate_highlights.txt
```

**Phase G Dense Metrics Analysis Script:**

The `analyze_dense_metrics.py` script generates a concise Markdown digest summarizing Phase G comparison results, combining metrics summaries with highlights for documentation and reporting. The script:

- Loads `metrics_summary.json` (output from `summarize_phase_g_outputs()`)
- Loads `aggregate_highlights.txt` (output from `report_phase_g_dense_metrics.py`)
- Emits success banner when `n_failed == 0`: **✓ ALL COMPARISONS SUCCESSFUL ✓**
- Emits failure banner when `n_failed > 0`: **⚠️ FAILURES PRESENT ⚠️** (stderr + digest)
- Embeds highlights content in code block
- Generates Key Deltas section with MS-SSIM and MAE comparison tables
- Writes digest to specified output file and prints to stdout
- Exit codes: 0 (all jobs succeeded), 1 (failures present or missing files), 2 (invalid format)

Tests validate two behaviors: success digest (exit 0, success banner present, no failure warning) and failure detection (exit 1, failure banner in stderr and digest, digest file still written).

**Usage:**
```bash
# Generate digest from Phase G outputs
python scripts/study/analyze_dense_metrics.py \
  --metrics plans/active/.../reports/<timestamp>/analysis/metrics_summary.json \
  --highlights plans/active/.../reports/<timestamp>/analysis/aggregate_highlights.txt \
  --output plans/active/.../reports/<timestamp>/analysis/metrics_digest.md
```

**Phase G Delta Metrics Persistence:**

The orchestrator automatically persists computed MS-SSIM/MAE deltas to two complementary artifacts after the digest generation step:

1. **JSON artifact** (`analysis/metrics_delta_summary.json`):
   - Contains raw numeric delta values (not formatted strings) for programmatic consumption
   - Includes provenance metadata (`generated_at`, `source_metrics`) for traceability
   - Structure: `{"generated_at": "...", "source_metrics": "...", "deltas": {"vs_Baseline": {...}, "vs_PtyChi": {...}}}`
   - Each metric includes `amplitude` and `phase` fields with numeric delta values (null if source data missing)
   - Referenced in orchestrator success banner for traceability

2. **Highlights text artifact** (`analysis/metrics_delta_highlights.txt`):
   - Contains formatted delta lines (4 lines total) for quick human review
   - Format: `MS-SSIM Δ (PtychoPINN - Baseline)  : amplitude +0.020  phase +0.020`
   - Signed 3-decimal formatting (+/-0.000) for all deltas
   - Order: MS-SSIM vs Baseline, MS-SSIM vs PtyChi, MAE vs Baseline, MAE vs PtyChi
   - Referenced in orchestrator success banner for traceability
   - Useful for quick delta validation without parsing JSON

3. **Preview text artifact** (`analysis/metrics_delta_highlights_preview.txt`):
   - Phase-only subset of highlights for quick sanity-checks (4 lines total)
   - Format: `MS-SSIM Δ (PtychoPINN - Baseline): +0.020` (single value, no "amplitude" keyword)
   - MUST NOT contain "amplitude" keyword (enforced by PREVIEW-PHASE-001 guard)
   - Same signed 3-decimal formatting (±0.000) and ordering as full highlights
   - Used by orchestrator to emit preview banner to stdout without file operations
   - Validated by `verify_dense_pipeline_artifacts.py` with `preview_phase_only` check

4. **SSIM Grid Summary artifact** (`analysis/ssim_grid_summary.md`):
   - Phase-only MS-SSIM/MAE delta table generated by `bin/ssim_grid.py` helper
   - Enforces PREVIEW-PHASE-001 compliance: validates preview file contains no "amplitude" keyword
   - Format: MS-SSIM with ±0.000 precision (3 decimals), MAE with ±0.000000 precision (6 decimals)
   - Contains metadata line: `Preview source: analysis/metrics_delta_highlights_preview.txt (phase-only: ✓)`
   - Logged to `cli/ssim_grid_cli.log` by the orchestrator
   - Required by `verify_dense_pipeline_artifacts.py` for complete pipeline validation
   - Exit codes: 0 (success), 1 (preview guard failure with amplitude contamination), 2 (missing/invalid inputs), 3 (other errors)

**Provenance metadata fields (JSON only):**
- `generated_at`: ISO8601 UTC timestamp (YYYY-MM-DDTHH:MM:SSZ) when deltas were computed
- `source_metrics`: Relative POSIX path from hub to source metrics_summary.json (TYPE-PATH-001 compliance)

**Example structure:**
```json
{
  "generated_at": "2025-11-09T13:05:00Z",
  "source_metrics": "analysis/metrics_summary.json",
  "deltas": {
    "vs_Baseline": {
      "ms_ssim": {"amplitude": 0.020, "phase": 0.020},
      "mae": {"amplitude": -0.005, "phase": -0.005}
    },
    "vs_PtyChi": {
      "ms_ssim": {"amplitude": 0.010, "phase": 0.010},
      "mae": {"amplitude": -0.002, "phase": -0.002}
    }
  }
}
```

**Phase G Full Pipeline Orchestrator:**

```bash
# Dry-run mode (prints planned commands without execution)
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
python scripts/study/run_dense_pipeline.py \
  --output-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/dense_execution \
  --dose 1000 \
  --view dense \
  --splits train test \
  --collect-only

# Real execution mode (remove --collect-only, add --clobber to archive stale outputs)
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
python scripts/study/run_dense_pipeline.py \
  --output-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/dense_execution \
  --dose 1000 \
  --view dense \
  --splits train test \
  --clobber

# Real execution (Phase C→G pipeline with metrics summary)
python scripts/study/run_dense_pipeline.py \
  --output-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/dense_execution \
  --dose 1000 \
  --view dense \
  --splits train test
```

### Phase D — Overlap Metrics (CLI)

Use the dedicated CLI to generate Phase D overlap metrics per spec (no spacing/packing gates; explicit `s_img` and `n_groups`).

```bash
python -m studies.fly64_dose_overlap.overlap \
  --phase-c-root data/phase_c/dose_1000 \
  --output-root tmp/phase_d_overlap/dose_1000 \
  --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/phase_d_overlap_metrics \
  --gridsize 2 \
  --s-img 0.8 \
  --n-groups 512 \
  --neighbor-count 6 \
  --probe-diameter-px 38.4 \
  --rng-seed-subsample 456
```

Selectors
- Unit + integration: `pytest tests/study/test_dose_overlap_overlap.py -v`
- Bundle schema: `pytest tests/study/test_dose_overlap_overlap.py::test_overlap_metrics_bundle -vv`

The orchestrator runs 8 sequential commands (Phase C generation → D overlap → E training baseline/dense → F reconstruction train/test → G comparison train/test), captures per-phase CLI logs under `{HUB}/cli/`, calls `summarize_phase_g_outputs()` after successful pipeline completion, invokes the reporting helper to generate aggregate Markdown + highlights, prints a highlights preview to stdout for quick sanity-checks of MS-SSIM/MAE deltas, then invokes `analyze_dense_metrics.py` to generate the final `metrics_digest.md` combining summary + highlights. Emits blocker log on any command failure.

- **Execution time:** Typically < 2s per test (mocked helper); real pipeline ~2-4 hours (8 phases with TensorFlow training + Pty-chi LSQML reconstruction)
- **Dependencies:** TensorFlow, PyTorch (for Pty-chi), Phase C base NPZ, `scripts/compare_models.py`

## Running Specific Tests

To run a specific test file:
```bash
python -m unittest tests.test_model_manager
```

To run a specific test class:
```bash
python -m unittest tests.test_integration_workflow.TestFullWorkflow
```

To run a specific test method:
```bash
python -m unittest tests.test_integration_workflow.TestFullWorkflow.test_train_save_load_infer_cycle
```

### PyTorch Backend Tests

PyTorch-specific tests are located in `tests/torch/` and use native pytest style. To run them:

```bash
# Run all PyTorch tests
pytest tests/torch/ -vv

# Run API deprecation tests (validates legacy API warning messaging)
pytest tests/torch/test_api_deprecation.py -vv

# Run specific test
pytest tests/torch/test_api_deprecation.py::TestLegacyAPIDeprecation::test_example_train_import_emits_deprecation_warning -vv
```

**Note:** PyTorch tests require `torch>=2.2` installed. Tests are automatically skipped in TensorFlow-only CI environments via directory-based pytest collection rules.

## How to Add New Tests

When contributing new tests to the project, follow these guidelines:

### 1. File Placement

Place all new test files in the top-level `tests/` directory or its subdirectories:
- Unit tests: `tests/test_<module_name>.py`
- Integration tests: `tests/test_integration_<workflow_name>.py`
- Specialized tests: Can be organized in subdirectories like `tests/studies/` or `tests/image/`

### 2. Naming Convention

Follow the `test_*.py` naming convention for all test files to ensure automatic discovery.

### 3. Test Structure

```python
import unittest
import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

class TestYourFeature(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_specific_functionality(self):
        """Test description."""
        # Arrange
        # Act
        # Assert
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
```

### 4. Best Practices

- Write descriptive test names that explain what is being tested
- Use setUp() and tearDown() for test fixtures
- Clean up temporary files and directories in tearDown()
- Keep docstrings concise - one line describing the test's purpose
- Avoid verbose comments - let test names and assertions document intent
- Use appropriate assertion methods (assertEqual, assertTrue, assertRaises, etc.)
- Keep unit tests fast and focused
- Mock external dependencies when appropriate
- For integration tests, use temporary directories for outputs

### 5. Test-Driven Development (TDD) Methodology

This project strongly encourages a Test-Driven Development (TDD) approach for all new features and bug fixes. The methodology follows a "Red-Green-Refactor" cycle:

1.  **RED - Write a Failing Test:** Before writing any implementation code, write a fine-grained, specific test that captures the desired functionality or reproduces the bug. Run the test and watch it fail. This proves the test is working and that the feature/fix is not already present.
2.  **GREEN - Write Minimal Code:** Write the simplest, most direct code possible to make the test pass. Do not worry about elegance or optimization at this stage. The goal is simply to get a passing test.
3.  **REFACTOR - Clean Up:** With a passing test as a safety net, refactor the implementation code to improve its design, readability, and efficiency. Re-run the test frequently to ensure it remains green.

**Case Study:** For a detailed, real-world example of how this TDD process was used to fix a critical architectural bug in the baseline model, see the case study in the **<doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref>**.

By following this TDD process, we ensure that all new code is inherently testable, correct by design, and protected against future regressions.

### 6. Documentation Style

Tests should be self-documenting through:
- Clear test method names (e.g., `test_scaling_preserves_physics`)
- Concise docstrings (one line preferred)
- Meaningful assertion messages that explain failures
- Avoid lengthy explanatory comments - the code should be clear

Example:
```python
def test_both_arrays_scaled_identically(self):
    """Regression test for X and Y_I scaling symmetry."""
    X, Y_I, _, scale = illuminate_and_diffract(...)
    ratio = np.mean(X) / np.mean(Y_I)
    self.assertLess(ratio, 10.0, 
        f"Scaling mismatch: ratio {ratio:.1f} indicates missing scaling")
```

## Test Coverage

While we don't enforce strict coverage metrics, aim to test:
- Happy path scenarios
- Edge cases and boundary conditions
- Error handling and exceptions
- Critical workflows and data pipelines

## Regression Testing for Feature Development

### Establishing Test Baseline

Before starting any significant feature development, establish a baseline of the current test suite status:

1. **Run the full test suite and save results:**
   ```bash
   $PYTHON_BIN -m unittest discover tests/ -v > test_baseline_$(date +%Y%m%d).txt 2>&1
   ```

2. **Document pre-existing failures:**
   Create a `TEST_BASELINE.md` file in your feature branch:
   ```markdown
   # Test Baseline for [Feature Name]
   
   **Date:** [YYYY-MM-DD]
   **Total Tests:** 172
   **Passing:** [number]
   **Failing:** [number]
   
   ## Pre-existing Failures
   - `test_image_registration.py::test_apply_shift_and_crop_basic` - Known issue #XXX
   - [other pre-existing failures]
   
   ## Critical Tests to Monitor
   - `test_integration_workflow.py::test_train_save_load_infer_cycle`
   - `test_coordinate_grouping.py::test_backward_compatibility`
   - [other critical tests for your feature]
   ```

### During Development

Run targeted test suites after each significant change:

```bash
# Quick check of critical tests
python -m unittest tests.test_integration_workflow -v

# Check specific subsystem
python -m unittest tests.test_coordinate_grouping -v

# Full regression check at milestones
python -m unittest discover tests/ -v
```

### Validation Criteria

Before considering your feature complete:
- No new test failures introduced (same pass/fail count as baseline)
- All previously passing tests continue to pass
- New features have comprehensive test coverage
- Performance tests show no significant regression

## Handling Pre-existing Test Failures

### Decision Framework

When encountering pre-existing test failures during feature development:

1. **Assess relevance:** Does the failing test interact with your feature area?
2. **Document decision:** Record your approach in the test baseline document
3. **Take action:**
   - **Fix if related:** If the failure affects your feature, fix it first
   - **Work around if unrelated:** Document the workaround and continue
   - **Flag for later:** Create an issue for unrelated failures that should be fixed

### Example Test Status Tracking

Maintain a test status log throughout development:

```markdown
## Test Status Log

### Phase 1: Core Infrastructure (2025-08-28)
- Baseline: 140/172 passing
- After components.py changes: 140/172 passing ✓
- After config changes: 140/172 passing ✓

### Phase 2: Training Integration (2025-08-29)
- After CLI updates: 140/172 passing ✓
- New subsampling tests: 5/5 passing ✓
```

## Testing New CLI Parameters

When adding new command-line parameters (like `--n-subsample`):

### Required Test Coverage

1. **Parameter parsing:** Test that the parameter is correctly parsed
2. **Default behavior:** Verify backward compatibility when parameter not specified
3. **Edge cases:** Test boundary conditions (e.g., n_subsample > dataset size)
4. **Integration:** Test parameter interaction with existing options
5. **Help text:** Ensure help message is clear and accurate

### Example Test Pattern

```python
class TestSubsamplingParameter(unittest.TestCase):
    def test_parameter_parsing(self):
        """Test --n-subsample parameter is correctly parsed."""
        args = parse_args(['--n-subsample', '1000'])
        self.assertEqual(args.n_subsample, 1000)
    
    def test_backward_compatibility(self):
        """Test behavior when --n-subsample not specified."""
        args = parse_args([])
        self.assertIsNone(args.n_subsample)
    
    def test_edge_case_oversample(self):
        """Test n_subsample > dataset size handling."""
        # Test implementation
```

## Backward Compatibility Testing

### Guidelines for Ensuring No Breaking Changes

1. **Test existing workflows:** Run complete workflows with old parameter sets
2. **Verify saved model compatibility:** Test loading models saved before changes
3. **Check configuration files:** Ensure old YAML configs still work
4. **Validate output format:** Confirm output files maintain expected structure

### Backward Compatibility Checklist

- [ ] Existing CLI commands work without new parameters
- [ ] Old configuration files load successfully
- [ ] Saved models can be loaded and used for inference
- [ ] Output file formats unchanged
- [ ] No performance regression for existing workflows

## Continuous Integration

Tests are automatically run on pull requests. Ensure all tests pass before merging changes.

## Troubleshooting

### Common Issues

1. **Import errors:** Ensure you're running tests from the project root
2. **Missing dependencies:** Install test requirements with `pip install -e .[test]` (if available)
3. **Data file not found:** Some tests require the example dataset in `ptycho/datasets/`
4. **Slow tests:** Use `-v` flag for verbose output to identify slow tests

### Getting Help

If you encounter issues with tests:
1. Check test output for specific error messages
2. Ensure your environment matches project requirements
3. Consult the test file's docstrings for specific requirements
4. Open an issue on GitHub if problems persist

## Related Documentation

- [README.md](../README.md) - Project overview and quick start
- [scripts/README.md](../scripts/README.md) - Information about training and inference scripts
- [tests/README_PERSISTENCE_TESTS.md](../tests/README_PERSISTENCE_TESTS.md) - Specific documentation about persistence tests
