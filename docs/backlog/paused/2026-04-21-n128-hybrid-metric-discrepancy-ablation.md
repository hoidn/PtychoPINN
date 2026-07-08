# Backlog: N128 Hybrid ResNet Metric Discrepancy Ablation

**Created:** 2026-04-21
**Status:** Paused
**Priority:** High
**Related:** `docs/model_baselines.md`, `docs/studies/index.md`, `tests/torch/test_grid_lines_hybrid_resnet_integration.py`, `scripts/studies/grid_lines_torch_runner.py`, `.artifacts/NEURIPS-HYBRID-RESNET-2026/hybrid_l2_match_ablation_20260421T203354Z/`, `.artifacts/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_n128_padex_e40_no_l2_20260421T211500Z/`, `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/e40_compare_20260421T190448Z/`
**Impacts:** Hybrid ResNet baseline governance, grid-lines study reproducibility, interpretation of historical N=128 lines evidence

## Summary

Recent `N=128` Hybrid ResNet runs produced materially different stitched metrics under contracts that looked superficially similar.

The best recovered replay under the old strong contract ended at:

- amp / phase MAE: `0.0269 / 0.0720`
- amp / phase SSIM: `0.9881 / 0.9947`

Run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_n128_padex_e40_no_l2_20260421T211500Z/`

The earlier worse 40-epoch row ended at:

- amp / phase MAE: `0.2012 / 0.1016`
- amp / phase SSIM: `0.9206 / 0.9618`

Run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-spectral-resnet-bottleneck-n128-integration-prototype/e40_compare_20260421T190448Z/hybrid_resnet/`

This discrepancy is too large to leave as folklore. The repo needs a proper ablation that turns the “we think it was mostly `torch_mae_pred_l2_match_target` plus dataset-contract drift” explanation into durable evidence.

## Why This Matters

Right now the project baseline docs still point to a Hybrid ResNet recipe with:

- `torch_mae_pred_l2_match_target=on`

But the strongest recovered `N=128` lines result used:

- `torch_mae_pred_l2_match_target=off`
- regenerated `nimgs_test=2`
- `probe_scale_mode=pad_extrapolate`
- custom Run1084 probe
- `set_phi=True`

That means the repo currently has three different notions of “baseline” in play:

1. the recommended baseline in `docs/model_baselines.md`,
2. the integration-test command contract,
3. the recovered strong historical decision-support contract.

Until those are explicitly compared, future agents will keep reconstructing the explanation from artifacts instead of reading one durable source of truth.

## Required Ablation Questions

The ablation should answer these questions in order:

1. How much of the metric drop is caused by `torch_mae_pred_l2_match_target=on` by itself?
2. How much is caused by test-set / dataset-contract drift (`nimgs_test=1` vs `2`)?
3. How much is caused by probe preprocessing drift (`pad_preserve` / cached integration contract vs `pad_extrapolate` recovered contract)?
4. Are the stitched metrics or the patch-level validation curves the more sensitive signal for these contract changes?
5. Does the recovered strong contract justify a docs-baseline update, or is it only a study-local override?

## Proposed Ablation Matrix

Use `hybrid_resnet` only, fixed architecture, fixed seed `3`, fixed 40-epoch budget.

Start from the recovered strong run and vary one factor at a time:

### Axis A: Loss Contract

- `torch_mae_pred_l2_match_target=off`
- `torch_mae_pred_l2_match_target=on`

### Axis B: Dataset Split Contract

- recovered contract: `nimgs_train=2`, `nimgs_test=2`
- cached integration contract: `nimgs_train=2`, `nimgs_test=1`

### Axis C: Probe Contract

- recovered contract: custom probe, `probe_scale_mode=pad_extrapolate`, `probe_smoothing_sigma=0.5`
- current `lines_256`-style preserved-padding family only if explicitly needed as a contrast; do not mix this into the first pass unless the first two axes leave ambiguity

### Fixed Controls

- `N=128`
- `gridsize=1`
- `set_phi=True`
- `nphotons=1e9`
- no probe mask
- `fno_modes=12`
- `fno_width=32`
- `fno_blocks=4`
- `hybrid_resnet_blocks=6`
- scheduler: `ReduceLROnPlateau`
- learning rate: `2e-4`
- plateau factor / patience / threshold / min lr unchanged from the recovered replay

## Minimum Execution Order

If resumed, run in this order:

1. recovered strong contract, rerun once as a fresh control
2. same command with only `torch_mae_pred_l2_match_target=on`
3. same command with only `nimgs_test=1`
4. same command with both changed
5. only if needed, a probe-contract contrast run

This keeps attribution clean and prevents a full factorial explosion before the likely primary driver is settled.

## Required Outputs

The resumed work must produce:

- one concise comparison table for all rows,
- final stitched metrics: MAE, MSE, PSNR, SSIM, MS-SSIM, FRC50, FRC `1/7`,
- final train / val losses,
- a short note stating which variable accounts for most of the gap,
- an explicit recommendation on whether `docs/model_baselines.md` should change.

If the answer is “baseline doc should change,” that must happen in a follow-on doc update rather than being left implicit in the summary.

## Scope Boundaries

In scope:

- Hybrid ResNet only,
- `N=128` lines only,
- loss / dataset / probe contract attribution,
- baseline-governance recommendation.

Out of scope:

- architecture changes,
- spectral bottleneck variants,
- `N=256` runs,
- probe-mask studies,
- PDEBench comparisons.

## Resume Conditions

Resume this backlog if at least one of these becomes true:

1. the NeurIPS CDI anchor work needs a durable explanation for why the better replay differs from recent weaker rows;
2. a reviewer or maintainer asks whether the project baseline should still keep `torch_mae_pred_l2_match_target=on`;
3. a future `N=256` or PDE transfer run needs the true best-known lines contract rather than the convenience wrapper defaults.

## Acceptance Criteria If Resumed

1. The ablation isolates `torch_mae_pred_l2_match_target` from dataset-contract changes.
2. At least one row exactly matches the recovered strong replay contract.
3. The summary names the dominant cause of the discrepancy, not just a list of suspects.
4. The outcome explicitly says whether the project baseline, the integration-test contract, or only the study-local contract should change.

## Suggested Next Step

Do not execute this ad hoc. When resumed, write a narrow execution plan that pins the control row to the recovered replay contract and treats every other row as a single-variable perturbation from that control.
