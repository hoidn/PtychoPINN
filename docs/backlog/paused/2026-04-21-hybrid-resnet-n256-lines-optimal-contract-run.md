# Backlog: Hybrid ResNet N256 Lines Run Under Recovered Optimal Contract

**Created:** 2026-04-21
**Status:** Paused
**Priority:** Medium
**Related:** `docs/studies/lines_256_dataset.md`, `docs/studies/index.md`, `docs/model_baselines.md`, `scripts/studies/run_lines_256_arch_experiment.py`, `scripts/studies/grid_lines_torch_runner.py`, `.artifacts/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_n128_padex_e40_no_l2_20260421T211500Z/`
**Impacts:** `256x256` CDI decision-support evidence, Hybrid ResNet scaling evidence, lines_256 contract clarity

## Summary

The repo currently has a convenient `lines_256` study path, but its baked-in contract is not the same as the strongest `N=128` Hybrid ResNet contract we just recovered.

Current `lines_256` convenience path:

- canonical dataset family: `pad_preserve`
- `nimgs_test=1`
- wrapper default: `epochs=20`
- wrapper default: `torch_mae_pred_l2_match_target=on`

Recovered stronger `N=128` contract:

- custom Run1084 probe
- `probe_scale_mode=pad_extrapolate`
- `probe_smoothing_sigma=0.5`
- `set_phi=True`
- `nimgs_train=2`, `nimgs_test=2`
- `epochs=40`
- `torch_mae_pred_l2_match_target=off`
- `ReduceLROnPlateau`, `2e-4`, floor `1e-4`
- no probe mask
- `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `hybrid_resnet_blocks=6`

If the project wants a meaningful `N=256` Hybrid ResNet lines run as scaling evidence, it should not silently inherit the current convenience-wrapper contract. It should be executed as a deliberate direct-run or purpose-built wrapper run under the recovered stronger contract.

## Why This Matters

Right now “run `lines_256`” and “run the best-known Hybrid ResNet lines recipe at `256x256`” are not the same instruction.

That mismatch will keep causing confusion unless the project explicitly queues the higher-value run:

- same architecture family,
- same known-good training recipe,
- scaled to `N=256`,
- with a clearly stated dataset/probe contract.

This is especially important because the NeurIPS campaign treats `256x256` as optional higher-mode scaling context rather than the primary anchor. If we do run it, it should answer the intended question cleanly.

## Required Contract For The Queued Run

Unless a later design changes it explicitly, the intended `N=256` row should use:

- architecture: `hybrid_resnet`
- `N=256`
- `gridsize=1`
- `set_phi=True`
- custom Run1084 probe
- `probe_scale_mode=pad_extrapolate`
- `probe_smoothing_sigma=0.5`
- `nimgs_train=2`
- `nimgs_test=2`
- `nphotons=1e9`
- `seed=3`
- `epochs=40`
- optimizer `adam`
- learning rate `2e-4`
- scheduler `ReduceLROnPlateau`
- plateau factor `0.5`
- plateau patience `2`
- plateau threshold `0.0`
- plateau min lr `1e-4`
- loss mode `mae`
- `torch_mae_pred_l2_match_target=off`
- `probe_mask=off`
- `fno_modes=12`
- `fno_width=32`
- `fno_blocks=4`
- `hybrid_resnet_blocks=6`
- `hybrid_downsample_steps=2`
- `hybrid_downsample_op=stride_conv`

## Important Contract Warning

This queued run is intentionally **not** the same as the current `lines_256` convenience study contract documented in `docs/studies/lines_256_dataset.md`.

That means one of two things must happen when this backlog is resumed:

1. either regenerate a dedicated `N=256` dataset under the recovered-style contract, or
2. document clearly that the run is a direct one-off scaling study outside the current `lines_256` convenience family.

Do not quietly treat a `pad_preserve`, `nimgs_test=1`, `l2_match=on`, 20-epoch wrapper run as equivalent evidence.

## Expected Deliverable

The resumed work should produce one durable decision-support run with:

- exact invocation capture,
- exact dataset metadata,
- final stitched metrics,
- visuals,
- concise comparison against the recovered strong `N=128` run,
- explicit statement that this is `N=256` scaling context, not the primary paper anchor.

## Scope Boundaries

In scope:

- one explicit `N=256` Hybrid ResNet lines run under the recovered stronger contract,
- dataset / probe contract reconciliation if needed,
- direct comparison against the recovered `N=128` replay.

Out of scope:

- controller-loop architecture search,
- `lines_256` keep/discard automation,
- alternate architectures,
- PDEBench evidence,
- changing the primary `N=128` anchor plan.

## Resume Conditions

Resume this backlog only if at least one of these becomes true:

1. the NeurIPS campaign wants fresh `256x256` scaling evidence rather than relying on older incomplete historical rows;
2. the `N=128` discrepancy ablation confirms the recovered strong contract should be treated as the real best-known recipe;
3. a future manuscript draft needs one clean `N=256` Hybrid ResNet scaling row with modern provenance.

## Acceptance Criteria If Resumed

1. The run uses the recovered strong training contract rather than the current convenience-wrapper defaults.
2. The dataset/probe contract is recorded explicitly and does not rely on folklore.
3. The final summary states whether the `N=256` result supports or weakens the scaling hypothesis relative to the recovered `N=128` run.
4. The output is labeled decision-support or paper-facing evidence explicitly; it must not sit in an ambiguous middle ground.

## Suggested Next Step

Do not route this through the existing `lines_256` convenience wrapper unchanged. When resumed, write a dedicated execution plan that first resolves whether the correct path is a regenerated `N=256` dataset under the recovered contract or a clearly-labeled one-off direct-run study.
