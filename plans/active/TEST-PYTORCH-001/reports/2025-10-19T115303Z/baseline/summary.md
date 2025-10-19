# TEST-PYTORCH-001 Phase A Baseline Summary

**Date:** 2025-10-19
**Loop Scope:** Evidence-only baseline assessment (no production code changes)
**Artifact Hub:** `plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/baseline/`

---

## Executive Summary

**Finding:** PyTorch integration test is already **FULLY FUNCTIONAL** and GREEN.

- **Baseline Outcome:** ✅ PASSED in 32.54s (27% of 120s budget)
- **Blocker Count:** 0
- **Immediate Action Required:** None — test is CI-ready as-is
- **Recommended Next Steps:**
  1. Modernize test to pytest style (Phase C)
  2. Update stale docstrings and charter assumptions
  3. Optional: Create minimal fixture for faster iteration

---

## Environment Prerequisites

### Python Environment
- **Python version:** 3.11.13 (conda environment: ptycho311)
- **PyTorch version:** 2.8.0+cu128
- **CUDA available:** Yes (version 12.8)
- **Test execution mode:** CPU-only (via `CUDA_VISIBLE_DEVICES=""`)

### Dependencies Verified
- ✅ PyTorch >= 2.2 (POLICY-001 requirement satisfied)
- ✅ Lightning (training orchestration)
- ✅ pytest 8.4.1 (test runner)
- ✅ Dataset file accessible (35 MB NPZ at `datasets/Run1084_recon3_postPC_shrunk_3.npz`)

### Required Environment Configuration
```bash
# Force CPU execution (critical for CI reproducibility)
export CUDA_VISIBLE_DEVICES=""

# Run integration test
pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv
```

**No additional environment variables required** — MLflow suppression handled by `--disable_mlflow` CLI flag.

---

## Runtime Performance

### Baseline Execution Metrics
- **Total Runtime:** 32.54 seconds
- **Target Budget:** ≤120 seconds (per charter acceptance criteria)
- **Budget Utilization:** 27.1%
- **Verdict:** ✅ Well under budget; no optimization urgently needed

### Runtime Breakdown (estimated from subprocess workflow)
- **Training Phase:** ~20-25s (2 epochs, 64 groups, batch_size=4)
- **Checkpoint Save:** <1s (Lightning automatic checkpoint)
- **Inference Phase:** ~5-8s (32 groups, single forward pass + stitching)
- **Artifact I/O:** ~2s (PNG generation)

### Performance Observations
- **No GPU required** — CPU execution completes in acceptable time
- **Deterministic behavior** — Test uses fixed seeds (implicitly via Lightning deterministic=True)
- **Temp directory cleanup** — No persistent artifacts leak between runs

---

## Dataset Analysis

### Current Fixture
- **Path:** `datasets/Run1084_recon3_postPC_shrunk_3.npz`
- **Size:** 35 MB (on-disk)
- **Format:** NPZ archive (NumPy compressed)
- **Known Format Issue:** Legacy (H,W,N) transpose (FORMAT-001), auto-corrected by dataloader heuristic

### Training Sampling Parameters
- **`n_images`:** 64 (groups sampled for training)
- **`gridsize`:** 1 (no neighbor grouping)
- **`batch_size`:** 4
- **Effective training batches:** 64 / 4 = 16 batches per epoch

### Inference Sampling Parameters
- **`n_images`:** 32 (reduced from training set for faster inference)

### Fixture Minimization Assessment
**Current Status:** Optional (not blocking)

**Rationale:**
- 32.54s runtime is already 73% faster than budget
- Smaller fixture would only save ~10-15s
- Current dataset exercises realistic edge cases (legacy format, oversampling)

**Recommendation:** Defer fixture minimization to Phase B **only if**:
1. Developer iteration speed becomes a bottleneck
2. CI resources constrain parallel test execution
3. New parity tests require faster turnaround

---

## CLI Parameter Coverage

### Training Command Parameters (10 flags)
All parameters documented in `specs/ptychodus_api_spec.md` §5.2:
- ✅ `--train_data_file` → TrainingConfig.train_data_file
- ✅ `--test_data_file` → TrainingConfig.test_data_file
- ✅ `--output_dir` → TrainingConfig.output_dir
- ✅ `--max_epochs` → TrainingConfig.nepochs
- ✅ `--n_images` → TrainingConfig.n_groups (legacy alias)
- ✅ `--gridsize` → ModelConfig.gridsize
- ✅ `--batch_size` → TrainingConfig.batch_size
- ✅ `--device` → PyTorchExecutionConfig (to be formalized per ADR-003)
- ✅ `--disable_mlflow` → TrainingConfig (MLflow suppression)

### Inference Command Parameters (5 flags)
All parameters align with `InferenceConfig` fields (§5.3):
- ✅ `--model_path` → InferenceConfig.model_path
- ✅ `--test_data` → InferenceConfig.test_data_file
- ✅ `--output_dir` → InferenceConfig.output_dir
- ✅ `--n_images` → InferenceConfig.n_groups
- ✅ `--device` → Execution config

**Finding:** All CLI parameters are spec-compliant with no undocumented experimental flags.

---

## Gaps Highlighted for Phase B–D

### Phase B (Fixture Minimization) — OPTIONAL
- Current fixture is already under budget
- Defer unless developer iteration speed becomes critical

### Phase C (Test Modernization) — RECOMMENDED
1. **Migrate to pytest style** (CLAUDE.md §4.3 compliance)
   - Replace `unittest.TestCase` with pytest fixtures
   - Use `@pytest.fixture` for temp directory management
   - Replace `self.assert*` with native `assert` statements
2. **Update stale docstrings**
   - Remove "EXPECTED TO FAIL" annotations (lines 41-42, 65)
   - Document actual GREEN behavior since INTEGRATE-PYTORCH-001 Attempt #40
3. **Add explicit seeding assertions** (if reproducibility critical)
   - Validate that deterministic=True propagates to all RNG sources
   - Compare reconstruction checksums across runs

### Phase D (Documentation & CI) — REQUIRED
1. **Update charter** (`plans/pytorch_integration_test_plan.md`)
   - Mark open questions resolved
   - Document that PyTorch integration is now functional
2. **Update test suite index** (`docs/development/TEST_SUITE_INDEX.md`)
   - Add entry for `test_integration_workflow_torch.py`
   - Document pytest selector and runtime expectations
3. **CI integration guidance**
   - Confirm pytest marker strategy (`@pytest.mark.integration`)
   - Document CUDA_VISIBLE_DEVICES="" requirement for CI runners

---

## Blockers & Risks

### Confirmed Blockers: **ZERO**
All INTEGRATE-PYTORCH-001 blockers resolved in Attempts #10–#41:
- ✅ Lightning checkpoint loading (Attempt #34)
- ✅ Dtype enforcement (Attempt #37)
- ✅ Decoder shape parity (Attempt #40)

### Low-Severity Risks
1. **Test framework inconsistency** — unittest vs pytest mixing could cause future parametrization issues
2. **Stale documentation** — Charter and test docstrings contradict actual behavior
3. **Missing regression markers** — No explicit pytest marker for integration tier

**Mitigation:** Address in Phase C modernization loop.

---

## Artifacts Generated

1. **`pytest_integration_current.log`** (17 KB)
   - Full pytest output with -vv verbosity
   - Training subprocess stdout/stderr
   - Inference subprocess stdout/stderr
   - Final assertion results
2. **`inventory.md`** (this document's companion)
   - Detailed test coverage assessment
   - Parameter audit
   - Blocker catalog
3. **`summary.md`** (this document)
   - Environment prerequisites
   - Runtime performance analysis
   - Next-phase guidance

---

## Decision Points for Next Loop

### Proceed to Phase B (Fixture Minimization)?
**Recommendation:** ❌ SKIP — current runtime is acceptable

**Justification:**
- 32.54s << 120s budget (27% utilization)
- Fixture minimization would only save ~10-15s
- Developer iteration speed not currently a bottleneck

### Proceed to Phase C (Test Modernization)?
**Recommendation:** ✅ YES — highest value

**Benefits:**
1. Align with project testing standards (CLAUDE.md §4.3)
2. Enable future pytest parametrization for parity tests
3. Improve maintainability and readability
4. Update stale docstrings to reflect GREEN status

### Proceed to Phase D (Documentation & CI)?
**Recommendation:** ✅ YES — after Phase C completion

**Required updates:**
1. Test suite index entry
2. Charter revision (mark open questions resolved)
3. CI marker strategy documentation

---

## Phase A Exit Criteria: SATISFIED

### A1: Inventory existing coverage ✅
- See `inventory.md` for comprehensive assessment
- All test methods catalogued with implementation style analysis

### A2: Validate fixture + CLI readiness ✅
- Baseline run captured in `pytest_integration_current.log`
- Runtime: 32.54s (under budget)
- Return code: 0 (success)

### A3: Capture prerequisites checklist ✅
- Environment requirements documented above
- Dataset path confirmed (35 MB NPZ)
- Runtime budget validated (27% of target)

---

**Next Recommended Action:** Proceed to Phase C (Test Modernization) to migrate to pytest style and update stale docstrings, then Phase D (Documentation) to update charter and test suite index.

**Alternative Path:** If supervisor prioritizes CI integration first, Phase D can precede Phase C since test is already functional.
