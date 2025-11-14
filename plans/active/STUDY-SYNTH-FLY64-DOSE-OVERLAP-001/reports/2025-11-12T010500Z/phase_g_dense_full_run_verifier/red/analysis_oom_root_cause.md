# OOM Root Cause Analysis

## Finding
The OOM in test-split compare_models is **NOT** in Baseline inference (which is successfully chunked via BASELINE-CHUNKED-001).

## Actual Root Cause
`PtychoDataContainer.__init__` in `ptycho/loader.py:141` calls `combine_complex(Y_I, Y_phi)` which tries to allocate GPU memory for the ENTIRE test dataset (5216 groups) at once.

Stack trace:
```
scripts/compare_models.py:988 → create_ptycho_data_container(test_data_raw, final_config)
ptycho/loader.py:341 → PtychoDataContainer(X, Y_I, Y_phi, ...)
ptycho/loader.py:141 → self.Y = combine_complex(Y_I, Y_phi)  
ptycho/tf_helper.py:336 → tf.cast(amp, tf.complex64)  ← OOM HERE
```

## Why Baseline Chunking Doesn't Help
Baseline chunking (lines 1077-1140) only chunks the Baseline model inference. The PINN model still tries to load the full dataset into `test_container` BEFORE we even get to Baseline inference.

## Solution
Implement PINN inference chunking:
1. Add `--pinn-chunk-size` and `--pinn-predict-batch-size` flags
2. Chunk the test dataset before creating containers
3. Process PINN inference in chunks similar to Baseline
4. Reassemble patches after chunked inference

## Evidence
- Train split (5088 groups): SUCCESS - fits in memory
- Test split (5216 groups): OOM at Cast operation in combine_complex  
- Debug run (320 groups): SUCCESS - smaller dataset fits
- Blocker: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/red/blocked_20251113T220800Z_test_full_oom_despite_chunking.md
