# Phase G Dense Pipeline Launch Evidence

**Launch Timestamp:** 2025-11-05T12:47:29Z  
**Background PID:** 2278335  
**Hub:** /home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run

## Launch Status

✓ Orchestrator launched successfully with TYPE-PATH-001 compliant absolute hub path  
✓ CONFIG-001 guard active (AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md)  
✓ Hub prepared with --clobber (stale Phase C archived to archive/phase_c_20251105T124729Z)  
✓ Phase C dataset generation in progress  
✓ GPU detected: NVIDIA GeForce RTX 3090 (22259 MB)  
✓ TensorFlow/XLA backend initialized  
✓ cuDNN 91002 loaded  

## Pipeline Configuration

- **Dose:** 1000  
- **View:** dense  
- **Splits:** train, test  
- **Total Stages:** 8 (C/D/E_baseline/E_dense/F_train/F_test/G_train/G_test + analysis helpers)  
- **Expected Runtime:** 2-4 hours  

## Monitoring

**Check progress:**
```bash
tail -f /home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_full_2025-11-05T123500Z.log
```

**Verify running:**
```bash
ps aux | grep 2278335
```

## Next Actions

Upon completion (when PID 2278335 exits):
1. Verify `[8/8]` appears in orchestrator log
2. Check all Phase D-G artifacts exist
3. Run highlights consistency check
4. Generate metrics digest
5. Extract MS-SSIM/MAE deltas
6. Update summary.md and fix_plan.md with final evidence
