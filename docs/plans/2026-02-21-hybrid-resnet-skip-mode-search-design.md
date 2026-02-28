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
- dataset-profile parameterization and provenance requirements,
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

3b. Bridge scope guard:
- Torch-only sweep knobs must not expand `params.cfg` surface unless a cross-backend requirement is explicitly documented and approved.
- `hybrid_skip_connections` is explicitly Torch-only and must not be added to canonical `ModelConfig` or config-bridge mappings.
- `hybrid_skip_connections` must flow through Torch runner/execution path only and remain outside `update_legacy_dict(...)` / `params.cfg` contracts.
- Stage C-E knobs introduced by this initiative (`hybrid_downsample_steps`, `hybrid_downsample_op`, `hybrid_encoder_conv_hidden_scale`, `hybrid_encoder_spectral_hidden_scale`, `hybrid_resnet_blocks`, `hybrid_skip_style`) are Torch-only by default; no bridge/spec expansion in this plan.

4. Explicit confounder control during sweeps:
- `probe_mask` and MAE normalization toggles are fixed per sweep stage and recorded in manifest.
- confounder fields are mandatory persisted keys in manifest + summary rows (`probe_mask_enabled`, `torch_mae_pred_l2_match_target`).

5. Dataset profile explicitness:
- Every run must resolve a named dataset profile.
- No implicit “integration-style” dataset generation is allowed without a profile id and frozen recipe.
- Profile ids are orchestration-layer aliases, not replacements for path-based leaf CLI contracts.

## 4. New Knobs and Semantics

## 4.1 Skip Controls

- `hybrid_skip_connections: bool = False`
  - Enables encoder-to-decoder lateral fusion.
  - Scope: Torch-only runner/execution knob (not a TensorFlow/canonical config-bridge field).
- `hybrid_skip_style: {"add","concat","gated_add"} = "add"`
  - `add`: projected additive skip.
  - `concat`: channel concat followed by projection.
  - `gated_add`: additive skip with learnable gate `g` in `y = decoder + g * skip_proj`, initialized to `0.0` (identity-safe because the skip path starts neutral).
  - Scope: Torch-only runner/execution/model knob in this initiative.

## 4.2 Downsampling Controls

- `hybrid_downsample_steps: int = 2`
  - `1` means `N -> N/2`.
  - `2` means `N -> N/2 -> N/4`.
- `hybrid_downsample_op: {"stride_conv","avgpool_conv","blurpool_conv"} = "stride_conv"`
  - Selects operator family used at each downsample step.
  - Scope: Torch-only runner/execution/model knob in this initiative.

Topology contract:
- Skip fusion point selection must be derived from encoder/decoder stage metadata (resolution/stage mapping), not hard-coded to a fixed number of downsample steps.
- `hybrid_downsample_steps=2` must preserve current behavior as the compatibility baseline.

## 4.3 Capacity/Depth Controls

- `fno_modes` (existing)
- `fno_width` (existing)
- `fno_blocks` (existing)
- `hybrid_encoder_conv_hidden_scale` (new, default `1.0`)
  - Multiplicative scale for encoder local-convolution branch internal width.
  - Effective per-block width is resolved from stage width using deterministic mapping:
    `resolved_width = max(1, round(stage_channels * scale))`.
  - `1.0` preserves current behavior.
- `hybrid_encoder_spectral_hidden_scale` (new, default `1.0`)
  - Multiplicative scale for encoder spectral branch internal width.
  - Effective per-block width is resolved from stage width using deterministic mapping:
    `resolved_width = max(1, round(stage_channels * scale))`.
  - `1.0` preserves current behavior.
- `max_hidden_channels` (existing)
- `resnet_width` (existing, divisible-by-4 guard)
- `hybrid_resnet_blocks` (new, default `6`)
  - Scope: Torch-only runner/execution/model knob in this initiative.

Encoder-branch decoupling contract:
- additive fusion remains shape-safe at fixed stage width (`x + GELU(spectral + conv)`),
- spectral and conv branches may use independent internal widths,
- each branch must project back to stage width before additive merge,
- run artifacts MUST persist both configured scale values and resolved per-block widths for provenance/auditability,
- generator output shape/dtype contracts remain unchanged.

## 4.4 Dataset Profile Controls

- `dataset_profiles_n128` (list of profile ids; default: `integration_grid_lines_n128_v1`)
- `dataset_profiles_n256` (list of profile ids; default: `cameraman256_halfsplit_v1`)

Resolution rules:
- The sweep/runbook resolves each profile into concrete dataset inputs.
- Leaf execution remains path-first:
  - wrapper path: `--dataset-source` and `--train-data/--test-data` when required.
  - torch-runner path: `--train-npz/--test-npz`.
- Explicit caller-supplied paths override profile defaults.

Profile semantics:
- `integration_grid_lines_n128_v1`:
  - generated from fixed recipe:
    - `N=128`, `gridsize=1`
    - `probe_npz=datasets/Run1084_recon3_postPC_shrunk_3.npz`
    - `nimgs_train=2`, `nimgs_test=1`, `nphotons=1e9`
    - `probe_source=custom`, `probe_smoothing_sigma=0.5`, `probe_scale_mode=pad_extrapolate`, `set_phi=True`
  - deterministic dataset seed is required and recorded.
- `fly001_external_n128_top_bottom_v1`:
  - external NPZ split profile for N=128 (train/test paths supplied by caller).
- `custom_npz_pair_n128`:
  - caller-provided `train.npz` and `test.npz`.
- `cameraman256_halfsplit_v1`:
  - requires caller-provided `cameraman_dp` and `cameraman_para` HDF5 paths
  - train from `prepare_hybrid_dataset(..., half=\"top\")`
  - test from `prepare_hybrid_dataset(..., half=\"bottom\")` using returned `train_npz`.
- `custom_npz_pair_n256`:
  - caller-provided `train.npz` and `test.npz`.

## 5. Staged Search Strategy

One structural axis (or tightly coupled pair) changes per stage. No full Cartesian over all axes.

Stage-isolation contract:
- For stages `B-E` at `N=128`, each stage/sub-stage sweep must inherit exactly one upstream anchor configuration from the prior stage (`is_stage_anchor=true` or a single-row source summary).
- Non-active knobs are copied from that single anchor row and must remain scalar for the stage run matrix.
- Top-K candidate sets from prior stages are used for `N=256` confirmation/rerank, not for `N=128` stage fan-out.
- Multi-anchor fan-out is allowed only in explicitly labeled diagnostic runs that do not drive canonical stage promotion state.
- Canonical initialization rule: Stage-B `N=128` MUST start from a Stage-A control anchor row that matches true `hybrid_resnet` defaults (`modes=12`, `skip=off`, `width=32`, `fno_blocks=4`, `downsample_schedule=2`, `downsample_op=stride_conv`, `encoder_conv_hidden=none`, `encoder_spectral_hidden=none`, `max_hidden=none`, `resnet_width=none`, `resnet_blocks=6`, `skip_style=add`).
- Stage-A promoted winners remain valid for Stage-A `N=256` promotion/rerank and optional diagnostics, but they are not the canonical Stage-B `N=128` anchor source.

Stage identifier contract:
- `stage_id` values are `A|B|C|D|E`.
- `substage_id` values are:
  - `none` for stages `A`, `B`, `E`,
  - `C1|C2` for stage `C`,
  - `D1|D2|D3|D4` for stage `D`.
- Manifests and summaries must persist both `stage_id` and `substage_id`.
- Retention and promotion provenance must be keyed at sub-stage granularity where applicable.

Epoch-floor contract:
- All stage runs (`A-E`) MUST use at least `10` training epochs per run (`--epochs-n128 >= 10`, `--epochs-n256 >= 10`) unless an approved exception is recorded in the execution log with rationale.
- Outputs generated below this floor are non-canonical and MUST NOT be used as promotion sources.
- If non-canonical outputs were consumed downstream, rerun from the earliest violating stage and regenerate downstream promotion artifacts.

## 5.1 Stage A (Baseline Search)

Axes:
- `fno_modes x hybrid_skip_connections x fno_width`

Policy:
- full grid on `N=128`,
- promote top-K feasible Pareto-ranked candidates to `N=256`.
- emit a single Stage-A control-anchor summary row for downstream Stage-B `N=128` sweeps; this row MUST match the true-default tuple above.
- run on one or more `dataset_profiles_n128` and aggregate rankings across profiles.
- keep broad exploration single-seed (`seed=3` default), then apply the Section-6 boundary seed-robustness rule before promotion.

## 5.2 Stage B (Axis 1)

Axis:
- `fno_blocks`

Policy:
- vary `fno_blocks` with the Stage-A control-anchor configuration fixed,
- `N=128` full stage budget, top-K feasible Pareto-ranked candidates to `N=256`.
- evaluate with selected dataset profile sets for N=128/N=256.
- apply the Section-6 boundary seed-robustness rule on the `N=128` source summary before promotion to `N=256`.

## 5.3 Stage C (Axis 2)

Sub-stage C1:
- `hybrid_downsample_steps`
- vary this axis with one Stage-B anchor configuration fixed.

Sub-stage C2:
- lock best C1 schedule, vary `hybrid_downsample_op`.
- apply the Section-6 boundary seed-robustness rule on each promotion source summary before any `N=256` run.

## 5.4 Stage D (Axes 3 + 4 + 4b)

Sub-stage D1:
- encoder conv-branch capacity axis (`hybrid_encoder_conv_hidden_scale`) with spectral branch scale fixed to default (`1.0`).
- canonical sweep values for this initiative: `0.5,1,2`.
- source configuration comes from one upstream anchor row.

Sub-stage D2:
- lock best D1, vary encoder spectral-branch capacity axis (`hybrid_encoder_spectral_hidden_scale`).
- canonical sweep values for this initiative: `0.5,1,2`.

Sub-stage D3:
- lock best D2, vary global capacity axis (`max_hidden_channels` or `resnet_width`).

Sub-stage D4:
- lock best D3, vary `hybrid_resnet_blocks`.
- apply the Section-6 boundary seed-robustness rule on each promotion source summary before any `N=256` run.

## 5.5 Stage E (Axis 5)

Axis:
- `hybrid_skip_style`

Policy:
- evaluate skip style variants on best Stage-D configuration.
- apply the Section-6 boundary seed-robustness rule on the Stage-D source summary before promotion.
- Stage-E `N=128` sweep inherits one Stage-D anchor configuration.

## 6. Ranking and Promotion Policy

Promotion feasibility filters (hard-required before ranking):
- phase guardrail: reject candidates with `phase_ssim_drop_vs_baseline > max_phase_ssim_drop`.
- `max_phase_ssim_drop` defaults to `0.03` and is configurable via `--max-phase-ssim-drop`.
- phase baseline is resolved from stage baseline provenance (promotion source summary for downstream stages; Stage-A baseline for stage A) and persisted in summary fields.
- parameter cap: `model_params <= 300_000_000`.
- train-time cap:
  - `N=128`: `train_wall_time_sec <= 2700`.
  - `N=256`: `train_wall_time_sec <= 9000`.
- inference SLA (constraint, not objective):
  - `N=128`: `inference_time_s <= 60`.
  - `N=256`: `inference_time_s <= 240`.

Primary promotion objective (Pareto):
- use non-dominated sorting with directional objectives:
  - maximize amplitude SSIM (`amp_ssim`),
  - minimize `train_wall_time_sec`.
- implementation may realize this as direct maximize support for `amp_ssim` or as
  minimization of `amp_ssim_loss = 1 - amp_ssim`, but emitted summaries must include
  `amp_ssim` explicitly.
- `inference_time_s` is enforced via feasibility filter above, not optimized directly.
- amplitude MAE/MSE remain required diagnostics in summaries and baseline-comparison artifacts.

Multi-profile aggregation:
- compute per-profile Pareto ranks first (`pareto_rank_profile`, lower is better),
- compute macro rank as median Pareto rank across profiles,
- tie-break by higher mean amplitude SSIM across profiles, then lower params.

Seed policy:
- Broad sweeps remain single-seed (default `seed=3`) for throughput.
- Before every promotion event (for example `N=128 -> N=256`), run a boundary rerank with seeds `11` and `17`.
- Boundary candidate set is `top-K + next 2` from the feasible Pareto-ranked source summary.
- If fewer than `K+2` eligible candidates exist, rerun all eligible candidates.
- Promotion uses median candidate Pareto rank across seeds `{3,11,17}`.
- Seed-tie break uses higher mean amplitude SSIM across seeds; if still tied, keep standard tie-breakers (params, then lower inference time within SLA).

Tie-breakers:
1. higher amplitude SSIM,
2. lower parameter count,
3. lower inference time (within SLA).

Promotion source API:
- `summary.csv` is a versioned stage-state API.
- Every summary row includes `summary_schema_version`.
- Promotion loader must validate version + required columns + non-empty feasible candidate set before scheduling runs.
- For stages `B-E` `N=128` sweeps, promotion-source loader must resolve exactly one anchor row before matrix expansion.
- Seed-rerank collation must emit both:
  - full robustness summary (for top-K `N=256` promotion),
  - single-row anchor summary (for downstream `N=128` stage isolation).

Pause-and-diagnose conditions (not immediate abandonment):
- Trigger a pause when either condition is met:
  - two consecutive stages show `<1%` median relative improvement on the primary metric **and** the seed-rerank confidence interval overlaps zero across seeds `{3,11,17}`,
  - all new-stage candidates regress on amplitude SSIM at `N=256` **and** the same directional regression appears in the `N=128` robustness summary.
- Before final stop on an axis, run one bounded rescue mini-sweep on that axis and re-evaluate promotion gates.
- If rescue still fails, pause expansion on that axis and carry at least one hedge candidate into the next stage for low-budget monitoring.

## 7. Run Artifact Contract

Each stage writes:
- per-run invocation and stdout/stderr logs,
- stage manifest with fixed confounder settings and knob values,
- dataset profile ids and resolved input provenance:
  - train/test file paths,
  - emitted leaf CLI dataset args (`--train-data/--test-data` or `--train-npz/--test-npz`),
  - hashes (NPZ and/or source HDF5 where applicable),
- `summary.csv` and `summary.md` including:
  - stage id,
  - substage id,
  - dataset profile id,
  - all active knobs,
  - key metrics (amp/phase MAE, MSE, SSIM) including explicit `amp_ssim`,
  - phase guardrail fields (`phase_ssim_drop_vs_baseline`, `max_phase_ssim_drop`, `phase_guardrail_pass`),
  - model params,
  - `train_wall_time_sec`,
  - inference time,
  - feasibility columns (`is_feasible`, violated constraints),
  - Pareto columns (`pareto_rank_profile`, `pareto_rank_macro`, seed-robust Pareto rank fields),
  - stage-isolation fields (`is_stage_anchor`, `anchor_source_summary`),
  - promotion decisions.
- `promotion/stage_anchor_summary.csv` containing exactly one canonical anchor row for downstream `N=128` stage/sub-stage sweeps.

Retention/cleanup policy (normative):
- Default mode is prune-after-run with retention tiers.
- Cleanup executes after per-run metrics/manifest row persistence.
- Cleanup scope is restricted to the run output subtree (never external input paths).
- Between consecutive run invocations, delete repo-root `memoized_data/` before launching the next run to prevent cross-run cache contamination.
- Canonical command: `rm -rf memoized_data/` (safe no-op when absent).
- Retain one full-artifact anchor run per `(stage_id, substage_id, N, dataset_profile)` tuple (`retention_tier=full_anchor`) for forensic/debug use.
- Prune-heavy policy applies to subsequent successful runs in the same tuple (`retention_tier=pruned`).
- Prune heavy intermediates by default (large dataset/recon NPZs, checkpoints, transient caches/log blobs).
- Preserve reproducibility-critical artifacts:
  - `invocation.json`, `invocation.sh`
  - per-run metrics payload required for ranking/promotion
  - stage `sweep_manifest.json`, `summary.csv`, `summary.md`
  - curated comparison PNG evidence.
- Emit `cleanup_report.json` for each run with deleted paths, reclaimed bytes, and `retention_tier`.
- Operational severity (strong advisory): treat heavy-artifact retention as a serious plan misalignment even when workflow-level enforcement is soft.
- Runs MUST aggressively prune heavyweight training byproducts after metrics are captured.
- Runs MUST NOT retain checkpoints, per-epoch logs, or large transient intermediates in pruned tiers unless explicitly justified.
- Any retention exception MUST be explicit, justified, and time-bounded in the execution log.
- Reviewers SHOULD return `REVISE` when avoidable heavy artifacts remain without explicit approved exception.
- Disk pressure or growth trends SHOULD be treated as blocking operational risk and called out in review artifacts.

## 8. Validation and Test Expectations

For each new knob:
- parser/CLI tests,
- propagation tests along the intended path:
  - Torch-only knobs (including `hybrid_skip_connections`): runner/execution/workflow propagation tests plus explicit no-bridge assertions.
  - cross-backend knobs (when explicitly approved): config-bridge propagation tests.
- branch-decoupling tests for `hybrid_encoder_conv_hidden_scale` and `hybrid_encoder_spectral_hidden_scale` (independent branch-width effects with additive shape invariance),
- generator forward-shape tests,
- invalid-value rejection tests.

For staged runner:
- matrix cardinality tests,
- guardrail tests (disallow multi-structural-axis explosion),
- stage/sub-stage identity tests (including retention-key granularity),
- phase-guardrail threshold propagation/enforcement tests,
- feasibility filtering tests (train-time caps, inference SLA, params cap),
- Pareto front construction and promotion determinism tests.

## 9. Risks and Mitigations

Risk: combinatorial blow-up.  
Mitigation: strict stage gating and run-budget caps.

Risk: metric instability from confounders.  
Mitigation: fix probe-mask and MAE-normalization toggles per stage.

Risk: shape/runtime regressions when changing downsampling/skip styles.  
Mitigation: enforce output-contract tests before any stage run.
