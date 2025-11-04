**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001

## 1. Framework & Style
- Use pytest-style tests and lightweight validation scripts (no unittest mix-in) to keep parity with project guidance.
- Tests focus on data contracts, selection invariants, and execution proofs; heavy training is validated via logs/CSVs captured as artifacts.

## 2. CI/CD & Optional Dependencies
- Pty-chi is optional and may be unavailable in CI; mark pty-chi validation as “evidence via artifact” (manual run) with justification in summary.md.
- Torch is required in project policy but not exercised directly in this TensorFlow-based study; ensure tests skip gracefully if torch-related paths are unneeded.

## 3. Test Tiers
- Unit: dataset contract checks (keys/dtypes), dose normalization ranges, spacing calculations.
- Integration: end-to-end dataset creation + split + grouping + filtering (no training), ensuring artifacts produced.
- Smoke: minimal training invocation (short epochs) to verify plumbing, not performance.

## 4. Execution Proof
- For any test-related task, collect:
  - pytest log at `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/<timestamp>/pytest.log`
  - results summary at `.../summary.md` with pass/fail counts and any skips justified
  - evidence CSVs from compare_models (MS-SSIM etc.)
- Acceptable SKIP reasons: pty-chi unavailable; GPU-specific long tests; long benchmarks.

## 5. Mock/Stub Strategy
- For pty-chi steps in CI, rely on recorded outputs (aligned NPZs) when available; otherwise skip with justification.
- For simulator inputs, use small synthetic objects/probes when sanity checking structure.

## 6. Specific Checks

### Phase A — Design Constants (COMPLETE)
**Test module:** `tests/study/test_dose_overlap_design.py`
**Status:** 3/3 PASSED

Validates:
- Dose list [1e3, 1e4, 1e5]
- Gridsizes {1, 2} with neighbor_count=7
- K ≥ C constraint (K=7 ≥ C=4 for gridsize=2)
- Overlap views: dense=0.7, sparse=0.2
- Derived spacing thresholds: dense ≈ 38.4px, sparse ≈ 102.4px
- Spacing formula S = (1 − f_group) × N with N=128
- Train/test split axis='y'
- RNG seeds: simulation=42, grouping=123, subsampling=456
- MS-SSIM config: sigma=1.0, emphasize_phase=True

**Execution proof:**
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T021500Z/green/pytest_green.log`
- All tests PASSED; no SKIPs

### Phase B onwards — Pending
1) Data contracts (specs/data_contracts.md)
   - Required keys present (`diffraction` amplitude float32; `objectGuess`, `probeGuess`, `xcoords`, `ycoords`)
   - `Y` present in final synthetic sets (complex64); squeeze to 3D
2) Dose sanity per dataset
   - Confirm expected scaling behavior across doses (qualitative/log statistics)
3) Group-level overlap filtering
   - Verify computed group centers and enforce minimum spacing S
   - Report acceptance rate and spacing histogram
4) Train/test separation
   - Validate y-axis split with non-overlapping regions
5) Comparison outputs
   - Confirm CSV present with MS-SSIM (phase and amplitude), plots saved, aligned NPZs exist

## 7. PASS Criteria
- All contract checks pass for generated datasets
- Filtering invariants satisfied; logs record spacing stats
- Comparison CSVs exist with non-empty records for each condition
- No unexpected SKIPs, and any expected SKIPs are justified in summary.md

