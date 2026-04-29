# Shared-Blocks10 1024-Cap Longer-Convergence Backlog Plan

## Goal

Run a longer-budget PDEBench CNS `1024 / 128 / 128` capped study for
`spectral_resnet_bottleneck_shared_blocks10` to measure the more fully
converged eval metrics for the variant whose 40-epoch train curve was still
falling sharply.

## Motivation

The completed 1024cap finalist run ended with lower train MSE for
`shared_blocks10` than the base row, and the last 10 epochs dropped by `57.2%`.
The completed 2048cap follow-up then showed aggregate regression for
`shared_blocks10`, making it important to separate a true scaling effect from a
possibly under-converged 1024cap reference.

## Required Contract

- Task: PDEBench `2d_cfd_cns`.
- Cap: `1024 / 128 / 128`.
- History contract: `concat u[t-2:t] -> u[t]`.
- Training loss: MSE on normalized CNS fields.
- Model: `spectral_resnet_bottleneck_shared_blocks10`.
- Shell: preserve the completed 1024cap finalist shell exactly, including
  `base_model="spectral_resnet_bottleneck_net"`, `hidden_channels=32`,
  `fno_modes=12`, `fno_blocks=4`, `hybrid_downsample_steps=2`,
  `hybrid_resnet_blocks=6`, skip-add, pixelshuffle, shared bottleneck gate, and
  `spectral_bottleneck_blocks=10`.
- Scope: capped decision-support only.

## Work Items

1. Inspect the completed 1024cap finalist artifact root and freeze the exact
   run contract needed for a longer shared-blocks10 rerun.
2. Choose and record a longer epoch budget before launch. The default target is
   enough epochs to observe whether train loss materially plateaus after the
   epoch-40 value; do not silently change optimizer, scheduler, data cap, or
   architecture to make the row easier.
3. Launch the longer 1024cap shared-blocks10 run under the same CNS runner and
   retain invocation, split manifest, metrics, stdout, and run-root provenance.
4. Compare the longer row against the completed 40-epoch 1024cap finalist row
   and the 2048cap scaling summary.
5. Write a durable summary that reports train-loss trajectory, final held-out
   eval metrics, and whether the aggregate-vs-frequency interpretation changed.

## Acceptance Criteria

- The artifact root contains metrics and provenance for the longer
  `shared_blocks10` 1024cap run.
- The final summary states the epoch budget, final train MSE, held-out
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, and
  `fRMSE_high`.
- The summary explicitly answers whether longer convergence improves
  `shared_blocks10` enough to challenge the current base-row aggregate
  preference.
- Any validation-loss claim is backed by an actual validation-loss series;
  otherwise, report held-out eval metrics rather than calling them val loss.
