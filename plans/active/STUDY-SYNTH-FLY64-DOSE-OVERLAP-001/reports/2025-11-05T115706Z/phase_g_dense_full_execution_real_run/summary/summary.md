# Phase G Dense Pipeline Execution Summary — 2025-11-05T115706Z

## Status: RUNNING - Pipeline Launched (Updated 2025-11-05T124729Z)

Pipeline launched at 2025-11-05T12:47:29 UTC with PID 2278335. Phase C (Dataset Generation) is in progress. Full 8-stage pipeline expected to complete in 2-4 hours.

## Execution Timeline

### Pipeline Launch (RUNNING)
- **Launch Time**: 2025-11-05T12:47:29 UTC
- **Background PID**: 2278335
- **Command**:
  ```bash
  export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  HUB=/home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run
  nohup python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
    --hub "$HUB" \
    --dose 1000 \
    --view dense \
    --splits train test \
    --clobber \
    > "$HUB"/cli/run_phase_g_dense_full_2025-11-05T123500Z.log 2>&1 &
  ```
- **Log**: `cli/run_phase_g_dense_full_2025-11-05T123500Z.log`

### Phase C (Dataset Generation)
- **Status**: 🔄 IN PROGRESS
- **Started**: 2025-11-05T12:47:30 UTC
- **GPU**: NVIDIA GeForce RTX 3090 (22259 MB, Compute Capability 8.6)
- **TensorFlow**: Initialized with CUDA malloc Async allocator, XLA compilation active, cuDNN 91002
- **Log**: `cli/phase_c_generation.log`
- **Hub Preparation**: Stale Phase C outputs archived to `archive/phase_c_20251105T124729Z`

### Phase D-G (Overlap, Training, Reconstruction, Comparison)
- **Status**: ⏳ PENDING
- **Expected**: Will execute after Phase C completes (Phases D, E_baseline, E_dense, F_train, F_test, G_train, G_test + analysis helpers)

## Provenance Checks

### CONFIG-001 (Legacy Bridge)
- **Status**: ✓ SATISFIED
- **Evidence**: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` exported before orchestrator launch

### DATA-001 (Metadata Compliance)
- **Status**: ⏳ PENDING Phase C completion
- **Note**: Validator will check all Phase C outputs after generation

### TYPE-PATH-001 (Path Normalization)
- **Status**: ✓ SATISFIED
- **Evidence**: Absolute hub path used: `/home/ollie/Documents/PtychoPINN2/plans/active/...`
- **Hub Preparation**: Successfully prepared, previous Phase C archived correctly

## Technical Notes

### Pipeline Configuration
- **Dose**: 1000
- **View**: dense (K=7, gridsize=2)
- **Splits**: train, test (y-axis split, ~50% each)
- **Total Stages**: 8 commands
- **Expected Runtime**: 2-4 hours total
  - Phase C: 5-15 minutes (simulation + canonicalization + patching + splitting)
  - Phase D: 1-2 minutes (overlap metadata)
  - Phase E: 1-3 hours (baseline + dense training, most time-intensive)
  - Phase F: 30-60 minutes (reconstruction train + test)
  - Phase G: 10-20 minutes (comparison train + test + analysis helpers)

### Monitoring Commands

**Check progress:**
```bash
tail -f /home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_full_2025-11-05T123500Z.log
```

**Verify running:**
```bash
ps aux | grep 2278335
```

**Check stage:**
```bash
grep -E "\[[0-9]/8\]" /home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_full_2025-11-05T123500Z.log | tail -1
```

## Next Actions (Upon Completion)

When PID 2278335 exits and `[8/8]` appears in the orchestrator log:

1. **Verify Completion**:
   - `grep -F "[8/8]" "$HUB"/cli/run_phase_g_dense_full_2025-11-05T123500Z.log`
   - Check exit code was 0

2. **Validate Artifacts** (all 7 expected files):
   - `analysis/metrics_summary.json`
   - `analysis/metrics_delta_summary.json`
   - `analysis/metrics_delta_highlights.txt`
   - `analysis/metrics_delta_highlights_preview.txt`
   - `analysis/metrics_digest.md`
   - `analysis/aggregate_report.md`
   - `analysis/aggregate_highlights.txt`

3. **Run Consistency Check**:
   ```bash
   python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py \
     --hub "$PWD/$HUB" | tee "$HUB"/analysis/highlights_consistency_check.log
   ```

4. **Refresh Metrics Digest** (if needed):
   ```bash
   python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py \
     --metrics "$PWD/$HUB"/analysis/metrics_summary.json \
     --highlights "$PWD/$HUB"/analysis/aggregate_highlights.txt \
     --output "$PWD/$HUB"/analysis/metrics_digest.md \
     | tee "$HUB"/analysis/metrics_digest_refresh.log
   ```

5. **Generate Artifact Inventory**:
   ```bash
   find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory.txt
   ```

6. **Extract MS-SSIM/MAE Deltas**:
   - Parse `analysis/metrics_delta_summary.json`
   - Document PtychoPINN - Baseline and PtychoPINN - PtyChi deltas
   - Include in summary.md and docs/fix_plan.md

7. **Run Validation Test**:
   ```bash
   pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv \
     | tee "$HUB"/green/pytest_orchestrator_dense_exec_post_run.log
   ```

8. **Update Documentation**:
   - Update this summary.md with runtime metrics and MS-SSIM/MAE results
   - Update docs/fix_plan.md Attempts History with completion evidence

## Findings Applied

- **POLICY-001**: PyTorch baseline assets (pty-chi) remain untouched; TensorFlow used for PINN training
- **CONFIG-001**: AUTHORITATIVE_CMDS_DOC exported before all helper invocations
- **DATA-001**: Phase C generation includes metadata validation; all NPZs will have `_metadata` field
- **TYPE-PATH-001**: Absolute hub path used to avoid `/plans` permission errors
- **OVERSAMPLING-001**: Dense view configuration (K=7, gridsize=2) preserved
- **STUDY-001**: MS-SSIM/MAE deltas will be computed for PtychoPINN vs Baseline and PtyChi

---

**Last Updated**: 2025-11-05T124729Z (Ralph i=244)
**Status**: RUNNING - Pipeline launched successfully, Phase C in progress
**Background PID**: 2278335

**Launch Evidence**: See `summary/pipeline_launch_evidence.md` for detailed launch status and monitoring commands.
