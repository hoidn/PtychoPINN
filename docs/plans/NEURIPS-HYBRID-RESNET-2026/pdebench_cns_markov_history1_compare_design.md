# PDEBench CNS Markov History-1 Equal-Footing Compare Design

## Design Metadata

- ID: `NEURIPS-HYBRID-RESNET-2026-pdebench-cns-markov-history1`
- Title: Markov `history_len=1` Equal-Footing Compare For PDEBench CNS
- Status: draft
- Date: `2026-04-21`
- Scope: evaluate whether the local capped CNS results improve when the
  supervised contract uses only the most recent state, `u[t-1] -> u[t]`, rather
  than the current default `u[t-2:t] -> u[t]`

## Problem

The current local CNS comparisons use `history_len=2`, which means each model
predicts the next state from the two most recent states concatenated along the
channel axis.

The FFNO literature the user referenced reports that using the Markov property,
that is predicting the next state from only the previous state, can improve
results. The repo already supports `history_len=1`, but it does not yet have a
same-slice controlled compare for that lower-context contract.

The missing evidence is:

- does `spectral_resnet_bottleneck_base` improve under `history_len=1`?
- does the ranking against `hybrid_resnet_cns`, `fno_base`, and `unet_strong`
  change under the same lower-context contract?

## Decision Summary

Run a focused `history_len=1` ablation on the existing capped CNS slice and keep
everything else fixed.

This means:

- same dataset file
- same trajectory split sizes
- same `max_windows_per_trajectory`
- same `mse` loss
- same batch size
- same `10`-epoch and `40`-epoch budgets
- same metric family
- only the input-history contract changes

This is intentionally narrower than a full autoregressive rollout study. The
first job is to isolate whether the lower-context Markov-style one-step training
setup changes one-step performance.

## Equal-Footing Contract

Use the existing capped CNS evaluation surface, changing only `history_len`.

Shared contract:

- task: `2d_cfd_cns`
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split: `512 / 64 / 64` trajectories
- `max_windows_per_trajectory=8`
- batch size: `4`
- training loss: `mse`
- metrics:
  - `err_RMSE`
  - `err_nRMSE`
  - `relative_l2`
  - `fRMSE_low`
  - `fRMSE_mid`
  - `fRMSE_high`
- epoch budgets:
  - `10`
  - `40`

Only changed variable:

- old contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
- new contract: `history_len=1`, `concat u[t-1:t] -> u[t]`

For CNS this changes the supervised unit from:

- current default: `input=(8,128,128)`, `target=(4,128,128)`
- Markov compare: `input=(4,128,128)`, `target=(4,128,128)`

## Rows To Compare

Required rows:

- `spectral_resnet_bottleneck_base`
- `hybrid_resnet_cns`
- `fno_base`
- `unet_strong`

Why all four:

- rerunning only spectral would answer whether spectral itself likes
  `history_len=1`
- but it would not answer whether Markov-style training changes the ranking
  under equal footing

The existing `history_len=2` capped runs remain the comparison anchor.

## Why This Is Small And Safe

The repo already supports `history_len=1` in the CNS loader and model builder:

- loader supports any `history_len >= 1`
- model builders already infer `in_channels` from the sampled tensor

So this design does **not** require:

- a new loader
- architecture surgery for Hybrid-spectral
- channel-plumbing changes

The work is experimental orchestration and reporting, not a model rewrite.

## Out Of Scope

- paper-faithful FFNO reproduction
- rollout evaluation as the primary question
- changing optimizer or scheduler to match an external paper
- changing the capped dataset slice
- promoting any Markov-style row into benchmark defaults before evidence exists

## Optional Later Extension

If `history_len=1` shows a clear gain, a later follow-up study may add:

- autoregressive rollout evaluation
- direct comparison against the authors' actual FFNO implementation under the
  same `history_len=1` contract

That follow-up should be a separate plan because rollout introduces a second
variable beyond the one-step data contract.

## Acceptance Criteria

1. A `10`-epoch capped CNS comparison exists for the four required rows at
   `history_len=1`.
2. A `40`-epoch capped CNS comparison exists for the same four rows.
3. The results are documented against the current `history_len=2` anchor.
4. The summary states clearly whether the Markov-style setup helps
   `spectral_resnet_bottleneck_base` and whether the row ranking changes.
5. Any conclusion remains labeled as capped-readiness evidence, not
   benchmark-complete CNS evidence.

## Suggested Checks

- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
