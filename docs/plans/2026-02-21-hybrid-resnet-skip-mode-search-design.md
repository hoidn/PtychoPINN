# Hybrid ResNet Skip + Mode Search Design

**Date:** 2026-02-21  
**Companion Plan:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`

## 1. Purpose

Define the normative design for:
- optional skip connections in `hybrid_resnet`,
- staged hyperparameter and architecture search,
- promotion criteria between search stages.

This document constrains implementation choices so the sweep remains reproducible, bounded, and comparable across runs.

## 2. Scope

In scope:
- new model knobs and their semantics,
- architectural invariants and compatibility requirements,
- staged search strategy (A through E),
- ranking and promotion policy,
- run artifact contract.

Out of scope:
- changing core physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`),
- replacing objective functions beyond existing toggles,
- unconstrained Cartesian HPO over all axes simultaneously.

## 3. Architectural Invariants

1. Default parity:
- Default config must preserve current behavior (`hybrid_skip_connections=False`, downsample schedule/operator defaults unchanged).

2. Output contract:
- Generator output shape and dtype contracts stay unchanged (`real_imag` and `amp_phase` paths).

3. Backward compatibility:
- Existing scripts/tests without new flags must run unchanged.

4. Explicit confounder control during sweeps:
- `probe_mask` and MAE normalization toggles are fixed per sweep stage and recorded in manifest.

## 4. New Knobs and Semantics

## 4.1 Skip Controls

- `hybrid_skip_connections: bool = False`
  - Enables encoder-to-decoder lateral fusion.
- `hybrid_skip_style: {"add","concat","gated_add"} = "add"`
  - `add`: projected additive skip.
  - `concat`: channel concat followed by projection.
  - `gated_add`: additive skip with learnable gate initialized near identity-safe value.

## 4.2 Downsampling Controls

- `hybrid_downsample_steps: int = 2`
  - `1` means `N -> N/2`.
  - `2` means `N -> N/2 -> N/4`.
- `hybrid_downsample_op: {"stride_conv","avgpool_conv","blurpool_conv"} = "stride_conv"`
  - Selects operator family used at each downsample step.

## 4.3 Capacity/Depth Controls

- `fno_modes` (existing)
- `fno_width` (existing)
- `fno_blocks` (existing)
- `max_hidden_channels` (existing)
- `resnet_width` (existing, divisible-by-4 guard)
- `hybrid_resnet_blocks` (new, default `6`)

## 5. Staged Search Strategy

One structural axis (or tightly coupled pair) changes per stage. No full Cartesian over all axes.

## 5.1 Stage A (Baseline Search)

Axes:
- `fno_modes x hybrid_skip_connections x fno_width`

Policy:
- full grid on `N=128`,
- promote top-K to `N=256`.

## 5.2 Stage B (Axis 1)

Axis:
- `fno_blocks`

Policy:
- vary `fno_blocks` with Stage-A-best settings fixed,
- `N=128` full stage budget, top-K to `N=256`.

## 5.3 Stage C (Axis 2)

Sub-stage C1:
- `hybrid_downsample_steps`

Sub-stage C2:
- lock best C1 schedule, vary `hybrid_downsample_op`.

## 5.4 Stage D (Axes 3 + 4)

Sub-stage D1:
- capacity axis (`max_hidden_channels` or `resnet_width`)

Sub-stage D2:
- lock best D1, vary `hybrid_resnet_blocks`.

## 5.5 Stage E (Axis 5)

Axis:
- `hybrid_skip_style`

Policy:
- evaluate skip style variants on best Stage-D configuration.

## 6. Ranking and Promotion Policy

Primary ranking (lexicographic):
1. lower amplitude MAE,
2. lower amplitude MSE,
3. higher amplitude SSIM.

Guardrail:
- reject candidates with phase SSIM drop > 0.03 relative to stage baseline.

Tie-breakers:
1. lower parameter count,
2. lower inference time.

Stop conditions:
- two consecutive stages with <1% relative improvement on primary metric, or
- all new-stage candidates regress on both amplitude MAE and MSE at `N=256`.

## 7. Run Artifact Contract

Each stage writes:
- per-run invocation and stdout/stderr logs,
- stage manifest with fixed confounder settings and knob values,
- `summary.csv` and `summary.md` including:
  - stage id,
  - all active knobs,
  - key metrics (amp/phase MAE, MSE, SSIM),
  - model params,
  - inference time,
  - promotion decisions.

## 8. Validation and Test Expectations

For each new knob:
- parser/CLI tests,
- config bridge propagation tests,
- generator forward-shape tests,
- invalid-value rejection tests.

For staged runner:
- matrix cardinality tests,
- guardrail tests (disallow multi-structural-axis explosion),
- top-K promotion determinism tests.

## 9. Risks and Mitigations

Risk: combinatorial blow-up.  
Mitigation: strict stage gating and run-budget caps.

Risk: metric instability from confounders.  
Mitigation: fix probe-mask and MAE-normalization toggles per stage.

Risk: shape/runtime regressions when changing downsampling/skip styles.  
Mitigation: enforce output-contract tests before any stage run.

