---
priority: 38
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/execution_plan.md
check_commands:
  - pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet_encoder or hybrid_encoder"
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_encoder"
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The current Hybrid ResNet encoder hard-codes equal additive spectral/local branch fusion; this item tests whether learned branch balance or encoder update scaling improves the CDI anchor family.
  - This is a narrow architecture ablation, not a replacement for the completed Lines128 paper table or the U-NO append-only comparator lane.
  - The first comparison should reuse the fixed N=128 grid-lines contract and avoid simultaneous probe, loss, bottleneck, or decoder changes.
---

# Backlog: Hybrid ResNet Encoder Fusion Variants

**Created:** 2026-04-21
**Status:** Active
**Priority:** Medium
**Related:** `ptycho_torch/generators/hybrid_resnet.py`, `ptycho_torch/generators/resnet_components.py`, `tests/torch/test_grid_lines_hybrid_resnet_integration.py`, `docs/studies/index.md`, `.artifacts/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_n128_padex_e40_no_l2_20260421T211500Z/`
**Impacts:** Hybrid ResNet encoder stability, `N=128` grid-lines follow-on ablations, future PDE/CNS encoder experiments

## Summary

The current `HybridResnetEncoderBlock` still uses the simplest possible fusion rule:

- compute a spectral branch,
- compute a local `3x3` conv branch,
- sum the two branches,
- apply `GELU`,
- add the result back to the identity residual:

`x_next = x + GELU(spectral(x) + conv(x))`

That is a clean baseline, but it leaves the encoder without any explicit outer update scaling or branch-level gating. Elsewhere in the repo, the model family already uses gates and layerscales successfully:

- `ResnetBlock` / `ResnetBottleneck` use shared residual gating,
- Hybrid encoder-decoder skip fusion supports `gated_add`,
- the newer spectral bottleneck variants use explicit spectral gates.

So the encoder branch-fusion path is now the clearest remaining place to test whether a more controlled update rule improves stability or final reconstruction quality.

## Why This Is Worth Revisiting

The replayed old-contract `N=128` grid-lines Hybrid ResNet run remains strong:

- amp / phase SSIM: `0.9881 / 0.9947`
- amp / phase MAE: `0.0269 / 0.0720`

That makes the current encoder block a credible baseline, but not necessarily the local optimum. During review of the encoder design, three follow-on variants emerged as more plausible than changing the residual path from identity `x` to a projected skip `W x`:

1. add an outer encoder-update layerscale,
2. add separate spectral and local branch gates,
3. add lightweight normalization around branch fusion.

These are better-motivated than a learned skip projection because the input/output shapes already match and the current optimization question is about update magnitude and branch balance, not skip-path remapping.

## Proposed Variant Set

### 1. Encoder Layerscale Variant

Keep the current branch structure and identity residual, but scale the full update:

`x_next = x + alpha * GELU(spectral(x) + conv(x))`

Where:

- `alpha` is a learned scalar,
- initialize conservatively, for example `0.1`,
- the implementation plan should either test per-block scalars only or test both shared-across-encoder-blocks and per-block scalar forms; shared-only is not sufficient.

Why this is promising:

- it is the smallest change from the current encoder,
- it matches the successful bottleneck residual-gate pattern already used in the repo,
- it directly tests whether the encoder update is simply too aggressive.

### 2. Branch-Gated Fusion Variant

Keep the identity residual path, but gate the spectral and local branches separately:

`x_next = x + GELU(g_s * spectral(x) + g_c * conv(x))`

Minimum viable form:

- learned scalar `g_s` for the spectral branch,
- learned scalar `g_c` for the local branch,
- initialize both small but nonzero, for example `0.1`.
- the implementation plan should either test per-block scalar gates only or test both shared-across-encoder-blocks and per-block scalar gate forms; shared-only is not sufficient.

Why this is promising:

- it lets the model tune the spectral/local balance instead of hard-coding equal pre-activation weight,
- it is directly aligned with the architectural question the model is supposed to answer,
- it is more interpretable than adding a projected skip.

This is the highest-priority follow-on variant.

### 3. Normalized Fusion Variant

Add a lightweight normalization step around the fused update, for example:

- prenorm on the fused branch signal before `GELU`, or
- post-branch norm before the residual add.

Candidate forms:

- `x_next = x + GELU(norm(spectral(x) + conv(x)))`
- `x_next = x + norm(GELU(spectral(x) + conv(x)))`

This should stay narrow:

- use a channelwise norm suitable for the current PyTorch image stack,
- do not combine this with unrelated architecture changes in the first pass.

Why this is only third priority:

- it adds more interpretation burden than simple scaling or gating,
- normalization placement can interact with branch balance in less transparent ways.

## Recommended Experiment Order

If this backlog is resumed, run the variants in this order:

1. `encoder_layerscale`
2. `encoder_branch_gates`
3. `encoder_branch_gates + encoder_layerscale`
4. optional normalized-fusion follow-up only if one of the first two is promising

That order keeps the first tranche easy to interpret and prevents normalization from obscuring the simpler questions.
For the learned-scalar variants, the implementation plan should explicitly
decide whether the first scored pass uses per-block scalars only or compares
shared and per-block scalars. It should treat shared versus per-block scalar
placement as an architecture axis rather than an incidental implementation
detail, and it should not satisfy this item with shared-only scalar gates.

## Naming Guidance

Unlike the spectral bottleneck family, these should remain inside the `hybrid_resnet` family because they are encoder-block ablations, not a distinct shell.

Use explicit profile suffixes such as:

- `hybrid_resnet_encoder_layerscale`
- `hybrid_resnet_encoder_branch_gated`
- `hybrid_resnet_encoder_branch_gated_layerscale`
- `hybrid_resnet_encoder_fusion_norm`

Do not overload existing baseline names or use ambiguous labels like `hybrid_resnet_v2`.

## Scope Boundaries

In scope when resumed:

- encoder-block fusion only,
- same residual identity path,
- same bottleneck / upsampler shell,
- same dataset contract for the comparison tranche,
- same loss contract as the chosen baseline command.

Out of scope for this item:

- projected residual skips `W x`,
- bottleneck replacements,
- PDEBench model-family renaming,
- simultaneous probe-mask or loss-contract changes,
- broad training-recipe retuning before architecture-isolation runs.

## Suggested First Execution Target

Use the recovered strong `N=128` grid-lines contract as the first target:

- `N=128`
- `gridsize=1`
- `set_phi=True`
- custom Run1084 probe
- `probe_scale_mode=pad_extrapolate`
- `nimgs_train=2`
- `nimgs_test=2`
- `torch_mae_pred_l2_match_target=off`
- `40` epochs

Reason:

- this is the cleanest current local reference point for Hybrid ResNet quality,
- the replay artifact is fresh and reproducible in the active checkout,
- it avoids mixing encoder-design questions with the PDEBench adaptation work.

## Resume Conditions

Resume this backlog only if at least one of these becomes true:

1. the NeurIPS Hybrid ResNet campaign needs one more bounded `N=128` architecture follow-on after the current anchor replay work;
2. a reviewer or local analysis specifically questions whether the encoder branch fusion is undercontrolled compared with the bottleneck and skip paths;
3. PDE/CNS work suggests encoder update stability is the next higher-value change after current bottleneck variants.

## Acceptance Criteria If Resumed

1. The resumed plan tests at least the first two variants: encoder layerscale and branch-gated fusion.
2. The implementation keeps the current identity residual path and documents that choice explicitly.
3. Tests cover shape preservation and config/build behavior for every new encoder profile.
4. The first scored comparison uses one fixed dataset and one fixed loss contract; architecture must be the only intended change.
5. The resulting summary states clearly whether any variant beats the replayed `N=128` Hybrid baseline on final stitched metrics rather than only training loss.

## Suggested Next Step

Do not execute this now. Revisit it only after the current CDI anchor and PDEBench priority work stops being the critical path, then write a narrow implementation plan that targets the recovered `N=128` grid-lines contract first.
