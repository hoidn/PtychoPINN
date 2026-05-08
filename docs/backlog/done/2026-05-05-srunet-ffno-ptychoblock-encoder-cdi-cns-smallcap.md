---
priority: 1
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
        Path("docs/backlog/done/2026-05-04-cdi-lines128-srunet-branch-objective-ablation.md"),
        Path("docs/backlog/done/2026-05-04-cns-matched-condition-table-refresh.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md"),
        Path("ptycho_torch/generators/ffno.py"),
        Path("ptycho_torch/generators/fno.py"),
        Path("ptycho_torch/generators/ffno_bottleneck.py"),
        Path("ptycho_torch/generators/hybrid_resnet.py"),
        Path("scripts/studies/grid_lines_torch_runner.py"),
        Path("scripts/studies/run_pdebench_image128_suite.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing SRU-Net FFNO->PtychoBlock encoder inputs: {missing}")
    print("SRU-Net FFNO->PtychoBlock encoder inputs present")
    PY
  - pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"
  - pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
  - 2026-05-04-cdi-lines128-srunet-branch-objective-ablation
  - 2026-05-04-cns-matched-condition-table-refresh
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The completed Lines128 branch ablation showed that the SRU-Net spectral-only encoder is strong, but it did not test an FFNO-style encoder inside the otherwise unchanged SRU-Net shell.
  - The completed FFNO bottleneck bridge changed the SRU-Net body rather than the encoder, so it does not answer whether an FFNO-style encoder helps.
  - This item is a bounded cross-pillar mechanism probe: one new architecture variant, evaluated on CDI Lines128 and small-cap CNS without rerunning existing baseline rows.
  - User scope update on 2026-05-06: for the FFNO-encoder versus standard SRU-Net mechanism comparison, the fresh corrected rows should run for `20` epochs rather than `40`. Compare against reused baseline lineage at the same epoch where epoch-level histories are available, and keep final claims bounded as mechanism evidence rather than headline table replacement.
  - User priority on 2026-05-05: run this before the remaining active WaveBench backlog items.
---

# Reactivated Status

Reactivated on 2026-05-06 at operator request. The prior interruption remains
diagnostic context only: partial in-flight artifacts from the interrupted
20-epoch CDI attempt do not satisfy this item's completion gate unless this
reactivated pass audits them against the fixed row contract and records an
explicit promotion decision.

# Backlog Item: SRU-Net FFNO-To-PtychoBlock Encoder Ablation

## Objective

- Implement and evaluate a narrow SRU-Net encoder variant that replaces the
  current SRU-Net encoder with:
  - a shared-weight 24-layer `FactorizedFfnoBlock` stack at the input/lifted
    resolution; then
  - two `PtychoBlock`-style encoder stages paired with the existing SRU-Net
    downsampling schedule.
- Measure whether that encoder substitution improves or weakens SRU-Net on:
  - the fixed `lines128` CDI benchmark; and
  - the small-cap PDEBench `2d_cfd_cns` decision-support benchmark.

## Scope

- Implement one explicit architecture/profile family, for example
  `hybrid_resnet_ffno_ptychoblock_encoder`, with a clearer equivalent name
  allowed if the implementation plan justifies it.
- Preserve the SRU-Net shell outside the encoder:
  - same lifter/input transform policy as the baseline SRU-Net row unless a
    shape adapter is strictly required and documented;
  - same two-step downsampling depth and downsample operator as the baseline
    SRU-Net contract;
  - same bottleneck family, bottleneck width/depth, decoder family, skip
    connection structure, residual scaling policy, output mode, loss,
    scheduler, seed policy, visual sample policy, and metric schema.
- Treat `PtychoBlock` as a shape-preserving local-plus-spectral update. Pair
  each of the two `PtychoBlock` stages with the existing SRU-Net downsample
  layer rather than inventing a new downsampling primitive.
- Use the same local reusable FFNO stack helper as the end-to-end CDI
  `FfnoGeneratorModule`, factored out as a shared module such as
  `FactorizedFfnoStack` and built from `FactorizedFfnoBlock` /
  `FactorizedSpectralConv2d`. Wire that stack as the pre-downsample encoder
  stack rather than as a replacement for the SRU-Net bottleneck or as a
  post-downsample bottleneck substitute.
- Keep the FFNO encoder stack fixed and explicit for the first row:
  - `ffno_encoder_blocks` must be `24`, matching the depth of the authored CNS
    FFNO baseline family rather than a lightweight two-block proxy;
  - `ffno_encoder_share_weights` must be `true`;
  - default modes should match the current SRU-Net/FFNO Lines128 convention
    unless the fixed CNS shape requires a documented adjustment;
  - weight sharing, gate initialization, normalization, and MLP ratio must be
    recorded in row config and manifest outputs;
  - exclude the end-to-end FFNO generator's local residual refiners from this
    encoder ablation;
  - do not tune these hyperparameters inside this item.
- Launch only the new rows:
  - `pinn_hybrid_resnet_ffno_ptychoblock_encoder` or equivalent on the fixed
    Lines128 CDI contract;
  - a matching small-cap CNS profile under the same local CNS training recipe
    and metric family used by the matched-condition CNS evidence lane.
- Reuse completed baseline rows by lineage:
  - Lines128: `pinn_hybrid_resnet`, `pinn_hybrid_resnet_encoder_spectral_only`,
    historical proxy `pinn_ffno`, and `pinn_hybrid_resnet_ffno_bottleneck`
    where available. Label `pinn_ffno` as `FFNO-local proxy` unless a
    corrected no-refiner row is explicitly substituted by lineage;
  - CNS: `spectral_resnet_bottleneck_base` / SRU-Net*, `author_ffno_cns_base`,
    `fno_base`, and `unet_strong` under the selected matched-condition capped
    authority.
- Do not rerun completed baselines just to assemble the comparison.

## CNS Small-Cap Contract

- Use the official `2d_cfd_cns` PDEBench dataset and the existing task-local
  CNS runner.
- Use the completed matched-condition CNS headline lane
  `h5_512_64_64_40ep`, with `history_len=5`, as the split/history/loss
  authority, but run the new FFNO-encoder mechanism row for `20` epochs.
- Use the small capped split family `512 / 64 / 64` for train/val/test
  trajectories, `20` epochs, batch size `4`, Adam at `2e-4`, and the existing
  CNS MSE training recipe.
- Treat the `20`-epoch CNS row as the intended mechanism-probe result for this
  item, not as a smoke-only run. Do not insert it into the `40`-epoch CNS
  headline table; compare it against same-contract lineage curves where
  available and label any comparison to `40`-epoch endpoints explicitly.

## Required Interpretation

- Frame this as an encoder mechanism ablation, not a new default SRU-Net family.
- The causal question is whether replacing the current SRU-Net encoder with a
  `shared-weight 24-layer FactorizedFfnoBlock stack -> 2x(PtychoBlock + downsample)` encoder improves the same downstream
  SRU-Net body/decoder.
- Do not conflate this with the completed FFNO bottleneck bridge. That row
  changed the body/bottleneck; this item changes only the encoder.
- Do not conflate this with the completed spectral-only encoder ablation. That
  row removed the local branch from the current encoder; this item introduces a
  different FFNO-first encoder stack before the two downsample stages.
- Keep CDI and CNS conclusions separate. CNS rows remain capped
  decision-support unless a later roadmap decision explicitly reopens
  full-training CNS evidence.
- If the variant helps one benchmark and hurts the other, report that as a
  domain-dependent mechanism result rather than averaging the metrics into one
  scalar ranking.

## Outputs

- Row-local invocation/config/history/metrics/reconstruction artifacts for the
  Lines128 CDI row.
- Row-local invocation/config/history/metrics/field-visual artifacts for the
  CNS small-cap row.
- A concise durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` only as
    bounded mechanism evidence, not as a headline table replacement;
  - `docs/studies/index.md`;
  - any model-variant or ablation index that already covers SRU-Net mechanism
    studies.

## Notes For Reviewer

- Reject implementations that change decoder skip wiring, bottleneck family,
  residual scaling, loss, probe/data contract, schedule, or metric definitions
  while claiming to isolate the encoder.
- Reject plans that tune FFNO encoder depth, modes, gates, or sharing after
  seeing the first metrics. Hyperparameter search requires a separate backlog
  item.
- Reject implementations that use a two-block FFNO proxy for the headline row;
  two-block artifacts may be archived only as misconfigured diagnostic context,
  not as the intended FFNO-encoder result.
- Reject CNS summaries that mix caps or history lengths inside one model-ranking
  table. If epoch budgets differ from headline rows, label the new `20`-epoch
  row as a mechanism-probe row and compare to same-epoch lineage values where
  available.
- Reject CDI summaries that rerun completed baseline rows or overwrite the
  completed Lines128 benchmark bundle.
- Require explicit manifest fields for the encoder recipe:
  `encoder_variant`, `ptychoblock_stage_count`, `downsample_steps`,
  `downsample_op`, `ffno_encoder_blocks`, `ffno_encoder_modes`,
  `ffno_encoder_share_weights`, `ffno_encoder_gate_init`,
  `ffno_encoder_norm`, and `ffno_encoder_mlp_ratio`.
