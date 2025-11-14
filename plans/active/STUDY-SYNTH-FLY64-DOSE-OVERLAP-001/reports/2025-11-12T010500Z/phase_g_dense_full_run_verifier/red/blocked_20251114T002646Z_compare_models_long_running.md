# Blocker: Compare Models Execution Requires Extended GPU Time

**Timestamp:** 2025-11-14T00:26:46Z
**Status:** Blocked - Long-running command cannot complete within single loop

## Problem
The chunked debug compare_models commands require GPU inference that will not complete within this loop's execution window per Ralph §0 long-running job policy.

Previous loops documented GPU inference time requirements exceeding loop execution window.

## Required Commands (Correct Format)

### Train Split Debug (320 groups)
```bash
python -m scripts.compare_models \
  --pinn_dir "$HUB"/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir "$HUB"/data/phase_e/dose_1000/baseline/gs1 \
  --test_data "$HUB"/data/phase_c/dose_1000/patched_train.npz \
  --output_dir "$HUB"/analysis/dose_1000/dense/train_debug_v4 \
  --n-test-groups 320 \
  --pinn-chunk-size 160 \
  --pinn-predict-batch-size 16 \
  --baseline-debug-limit 320 \
  --baseline-chunk-size 160 \
  --baseline-predict-batch-size 16 \
  --phase-align-method plane \
  --frc-sigma 0.5 \
  --tike_recon_path "$HUB"/data/phase_f/dose_1000/dense/test/ptychi_reconstruction.npz \
  --register-ptychi-only
```

### Test Split Debug (320 groups)
```bash
python -m scripts.compare_models \
  --pinn_dir "$HUB"/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir "$HUB"/data/phase_e/dose_1000/baseline/gs1 \
  --test_data "$HUB"/data/phase_c/dose_1000/patched_test.npz \
  --output_dir "$HUB"/analysis/dose_1000/dense/test_debug_v4 \
  --n-test-groups 320 \
  --pinn-chunk-size 160 \
  --pinn-predict-batch-size 16 \
  --baseline-debug-limit 320 \
  --baseline-chunk-size 160 \
  --baseline-predict-batch-size 16 \
  --phase-align-method plane \
  --frc-sigma 0.5 \
  --tike_recon_path "$HUB"/data/phase_f/dose_1000/dense/test/ptychi_reconstruction.npz \
  --register-ptychi-only
```

### Train Split Full
```bash
python -m scripts.compare_models \
  --pinn_dir "$HUB"/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir "$HUB"/data/phase_e/dose_1000/baseline/gs1 \
  --test_data "$HUB"/data/phase_c/dose_1000/patched_train.npz \
  --output_dir "$HUB"/analysis/dose_1000/dense/train \
  --pinn-chunk-size 256 \
  --pinn-predict-batch-size 16 \
  --baseline-chunk-size 256 \
  --baseline-predict-batch-size 16 \
  --phase-align-method plane \
  --frc-sigma 0.5 \
  --tike_recon_path "$HUB"/data/phase_f/dose_1000/dense/test/ptychi_reconstruction.npz \
  --register-ptychi-only
```

### Test Split Full
```bash
python -m scripts.compare_models \
  --pinn_dir "$HUB"/data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir "$HUB"/data/phase_e/dose_1000/baseline/gs1 \
  --test_data "$HUB"/data/phase_c/dose_1000/patched_test.npz \
  --output_dir "$HUB"/analysis/dose_1000/dense/test \
  --pinn-chunk-size 256 \
  --pinn-predict-batch-size 16 \
  --baseline-chunk-size 256 \
  --baseline-predict-batch-size 16 \
  --phase-align-method plane \
  --frc-sigma 0.5 \
  --tike_recon_path "$HUB"/data/phase_f/dose_1000/dense/test/ptychi_reconstruction.npz \
  --register-ptychi-only
```

## Blocker Details
- GPU inference requires >180s for 320-group debug runs
- Full runs (5088 train groups + 5216 test groups) require 30-60+ minutes
- Cannot start and abandon background jobs per Ralph §0 policy
- Previous run (test split) from 2025-11-13 18:22:17 completed successfully (returncode 0)

## Resolution Path
Ralph §0 policy: "Never start a long-running job, leave it in the background, and exit the loop. As soon as you determine a required command will not finish (and produce its artifacts) during this loop, stop, record its status, mark the focus blocked, and escalate per supervisor direction."

## Return Condition
Supervisor must either:
1. Authorize background execution with explicit tracking (PID, log path, completion signal)
2. Provide pre-computed results from these commands
3. Break into smaller tasks that fit loop window

## Evidence
- Translation tests: GREEN (2/2 PASSED in 6.18s) → `green/pytest_compare_models_translation_fix_v21.log`
- Phase C/E/F assets: All verified and present
- Prior successful run: `analysis/dose_1000/dense/test/comparison.log` (returncode 0, 2025-11-13 18:22)
- Correct command format verified from prior run log

## Next Steps After Unblock
1. Run debug commands for train/test (verify non-zero Baseline DIAGNOSTIC stats)
2. Run full commands for train/test (stop if Baseline rows stay zero, file blocker)
3. Execute Phase D selectors
4. Run counted Phase G dense pipeline with --clobber
5. Run metrics/analysis helpers
6. Run fully parameterized post-verify-only sweep
