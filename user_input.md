# User Input

## Priority Override

**Resume DEBUG-SIM-LINES-DOSE-001 Phase D — Amplitude Bias Investigation**

The initiative was prematurely marked as complete. "Training without NaN" is NOT a valid success condition — the exit criterion requires "recon success" = actual working reconstructions matching dose_experiments behavior.

### Current State
- NaN crashes fixed (CONFIG-001 bridging applied in C4f)
- **Amplitude bias (~3-6x undershoot) remains** — this is the actual problem
- Reconstructions do NOT match dose_experiments quality

### Phase D Goals
1. Identify why reconstructed amplitude undershoots ground truth by 3-6x
2. Compare sim_lines_4x output quality against dose_experiments baseline
3. Apply fix to achieve reconstruction parity (not just "no crashes")

### Hypotheses to Investigate
- **H-LOSS-WEIGHT**: Loss function weighting differs from dose_experiments
- **H-NORMALIZATION**: Intensity normalization pipeline introduces bias
- **H-TRAINING-PARAMS**: Hyperparameters (lr, epochs, batch size) insufficient
- **H-ARCHITECTURE**: Model architecture mismatch vs legacy

### Key Evidence
- All scenarios show amplitude bias: gs1_ideal, gs1_custom, gs2_ideal, gs2_custom
- Least-squares scaling factors (~1.7-2.0) cannot fully explain the gap
- Both probe types fail equally after CONFIG-001 fix (so not probe-specific)

### Artifacts Hub
`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T200000Z/`

### Do Now
Start Phase D investigation. First step: compare loss function configuration between sim_lines_4x and dose_experiments to identify any weighting differences.
