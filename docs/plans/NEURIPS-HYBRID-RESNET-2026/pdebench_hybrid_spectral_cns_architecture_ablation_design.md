# PDEBench CNS Hybrid-Spectral Architecture Ablation Design

## Design Metadata

- ID: `NEURIPS-HYBRID-RESNET-2026-pdebench-hybrid-spectral-cns-arch-ablation`
- Title: CNS Hybrid-Spectral Architecture Ablation
- Status: draft
- Date: `2026-04-22`
- Scope: define a controlled PDEBench `2d_cfd_cns` architecture ablation for
  the Hybrid-spectral family only, keeping CDI/ptycho out of scope and keeping
  the current CNS shell fixed so bottleneck conclusions stay attributable

## Problem

The repo already has several CNS architecture findings:

- the canonical CNS Hybrid shell now uses `skip=add`
- the canonical CNS Hybrid upsampler is `pixelshuffle`
- spectral weight sharing has already been compared
- spectral depth has already been probed on the shared row

Those studies were useful, but they do not yet form one clean architecture
ablation for the **Hybrid-spectral family**.

The missing design question is:

- which architectural choices inside the CNS Hybrid-spectral family are actually
  carrying the current result once the shell is fixed?

This should **not** be answered with one mixed study across CNS and CDI/ptycho.
The domains differ too much in data semantics, loss priorities, artifact modes,
and metric interpretation.

## Decision Summary

Limit this ablation family to **PDEBench CNS** and to the
**Hybrid-spectral family only**.

That means:

- keep the current CNS shell fixed
- do not reopen skip-connection or upsampler promotion as first-class axes in
  this item
- ablate only the spectral-family internals under that fixed shell
- keep CDI/ptycho as a separate future architecture-ablation design

This design treats skip connections and upsampling mode as already-settled
**shell controls** for CNS, not as uncontrolled confounders to mix back into
every spectral comparison.

## Fixed CNS Contract

Keep the current capped CNS comparison surface fixed:

- task: `2d_cfd_cns`
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split: `512 / 64 / 64` trajectories
- `max_windows_per_trajectory=8`
- history contract: `history_len=2`
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

Do not mix in:

- `history_len=1` Markov ablation
- physics regularization
- higher-mode `32/32` spectral runs
- external author baselines

Those remain separate ablation families.

## Fixed Shell Boundary

For this architecture ablation, the CNS shell is fixed to the current promoted
local default:

- `hybrid_skip_connections=True`
- `hybrid_skip_style="add"`
- `hybrid_upsampler="pixelshuffle"`
- `hybrid_downsample_steps=2`
- `hybrid_resnet_blocks=6`
- `hidden_channels=32`
- `fno_blocks=4`

Why fix the shell:

- skip connections and upsampler choice already showed material effect in local
  CNS studies
- reopening them inside the same ablation would confound the bottleneck result
- the current question is Hybrid-spectral architecture, not shell re-selection

## In-Scope Architecture Axes

This ablation should cover only the following spectral-family axes:

1. **spectral weight sharing**
   - shared
   - non-shared

2. **spectral bottleneck depth**
   - `spectral_bottleneck_blocks=6`
   - `spectral_bottleneck_blocks=8`
   - `spectral_bottleneck_blocks=10`

3. **optional later axis: spectral mode count**
   - current base `12/12`
   - later paired higher-mode row

The mode-count axis is explicitly secondary because it already has a separate
queued backlog item and should not be collapsed into the first architectural
ablation tranche.

## Primary Rows

The primary CNS Hybrid-spectral rows for this design are:

- `spectral_resnet_bottleneck_base`
- `spectral_resnet_bottleneck_noshare`
- shared-shell deeper spectral rows derived from the base profile

This is **not** a local-vs-spectral family comparison.

Why:

- local Hybrid already belongs to a different bottleneck family
- mixing local-vs-spectral with sharing/depth would broaden the question too far
- the cleanest first result is whether the spectral-family internals matter once
  the shell is fixed

If a later study wants to compare local Hybrid vs Hybrid-spectral again, it
should do so as a separate family-compare item under the same fixed shell.

## Study Structure

### Stage 1: Sharing Under Fixed Shell

Purpose:

- confirm whether the earlier shared-vs-non-shared signal still holds under the
  canonical shell and current reporting contract

Rows:

- `spectral_resnet_bottleneck_base`
- `spectral_resnet_bottleneck_noshare`

Budgets:

- `10` epochs
- `40` epochs

### Stage 2: Depth Under Fixed Shared Shell

Purpose:

- determine whether deeper spectral mixing changes the held-out asymptote or
  only slows optimization

Rows:

- shared depth `6`
- shared depth `8`
- shared depth `10`

Budgets:

- `40` epochs minimum
- larger budget allowed if `40` epochs clearly under-converges deeper rows

### Stage 3: Confirmation On Larger Cap

Purpose:

- test whether the favored spectral row still holds up when the train/test gap
  is reduced by more data

Contract:

- `1024 / 128 / 128` trajectories
- `8` windows per trajectory
- `40` epochs

This stage should be run only for finalists from stages 1 and 2.

## Decision Rules

Promotion should prioritize:

1. `relative_l2`
2. `err_nRMSE`
3. `fRMSE_high`
4. train/test gap on the larger cap

Reason:

- `relative_l2` and `err_nRMSE` remain the current primary CNS summary metrics
- `fRMSE_high` is needed to avoid rewarding rows that win only by smoothing
- the larger-cap train/test gap keeps the decision from being anchored only to a
  small capped slice

## Output Contract

Each completed row should emit the standard local CNS artifacts:

- `comparison_summary.json`
- `metrics_<profile>.json`
- train-split eval
- held-out test eval
- `comparison_<profile>_sample0.png`
- `comparison_<profile>_sample0.npz`
- comparison galleries when multiple rows belong to the same run root

The durable study summary should state clearly:

- which shell was fixed
- which spectral axis changed
- whether the result is capped-readiness evidence only
- whether any row is being promoted as the current best local spectral variant

## Out Of Scope

- CDI/ptycho architectural ablation
- shell ablation for CNS skip/upsampler choice
- local Hybrid vs Hybrid-spectral family compare
- Markov `history_len=1`
- physics regularization
- external-author FFNO or GNOT
- broad Cartesian sweeps

## Deferred CDI/Ptycho Companion

Create a separate future design for:

- CDI/Ptycho Hybrid-spectral Architecture Ablation

Reason:

- CDI/ptycho has different input/output semantics
- its losses and metrics are not interchangeable with CNS
- artifact behavior differs enough that shell and bottleneck choices should be
  evaluated independently

This CNS design should not make claims about what the CDI/ptycho architecture
should be.

## Acceptance Criteria

1. The design is implemented as a CNS-only Hybrid-spectral architecture
   ablation, not a mixed CNS/CDI item.
2. All rows in the ablation share the fixed CNS shell:
   skip-add plus pixelshuffle.
3. The first executed tranche covers spectral sharing and spectral depth, not a
   broad mixed sweep.
4. Any later higher-mode study remains explicitly separate or clearly labeled as
   a follow-on axis.
5. The summary states whether a current best local spectral CNS row has been
   identified under the fixed shell.

## Suggested Checks

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
