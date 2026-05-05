---
priority: 37
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-branch-objective-ablation/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
        Path("docs/backlog/done/2026-04-29-cdi-lines128-supervised-equivalent-rows.md"),
        Path("docs/backlog/done/2026-04-30-cdi-lines128-uno-table-extension.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md"),
        Path("scripts/studies/grid_lines_compare_wrapper.py"),
        Path("scripts/studies/grid_lines_torch_runner.py"),
        Path("ptycho_torch/generators/hybrid_resnet.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing SRU-Net branch/objective ablation inputs: {missing}")
    print("SRU-Net branch/objective ablation inputs present")
    PY
  - pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or hybrid_encoder"
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or hybrid_encoder or supervised"
  - pytest -q tests/test_grid_lines_compare_wrapper.py
  - python -m compileall -q scripts/studies ptycho_torch
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
  - 2026-04-29-cdi-lines128-supervised-equivalent-rows
  - 2026-04-30-cdi-lines128-uno-table-extension
  - 2026-04-21-hybrid-resnet-encoder-fusion-variants
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - This is the highest-value missing CDI architecture-mechanism ablation for the SRU-Net story.
  - It directly tests whether SRU-Net needs both local spatial features and global spectral coupling in the encoder.
  - It also adds the missing supervised SRU-Net objective-control row so the architecture effect can be separated from the CDI physics-consistency loss.
  - This is append-only Phase 3 CDI evidence; it must not rerun completed benchmark rows or rewrite the base Lines128 authority.
---

# Backlog Item: Lines128 SRU-Net Branch And Objective Ablation

## Objective

- Produce a same-contract `lines128` CDI ablation that tests the two most
  important remaining SRU-Net story controls:
  - whether the SRU-Net encoder needs both local spatial features and global
    spectral mixing;
  - whether SRU-Net's CDI advantage persists as an architecture effect when
    compared against a supervised SRU-Net row rather than only the
    physics-consistency-trained SRU-Net row.

## Scope

- Consume the completed `lines128` CDI paper benchmark as the baseline
  authority.
- Reuse the existing `pinn_hybrid_resnet` / `SRU-Net + PINN` row by lineage;
  do not rerun it unless a deterministic audit proves the existing row is
  unusable under the current contract.
- Launch only missing rows under the fixed `lines128` contract:
  - `pinn_hybrid_resnet_encoder_conv_only`: disables or zeroes the spectral
    encoder branch while preserving the local `3x3` spatial branch, identity
    residual, bottleneck, decoder, loss, and training contract;
  - `pinn_hybrid_resnet_encoder_spectral_only`: disables or zeroes the local
    spatial encoder branch while preserving the spectral branch, identity
    residual, bottleneck, decoder, loss, and training contract;
  - `supervised_hybrid_resnet` or an equivalent explicit row id labelled
    `SRU-Net + supervised`: uses the same SRU-Net architecture body as the
    `pinn_hybrid_resnet` row, but trains under the supervised CDI objective
    used by the existing supervised comparator rows.
- Include an optional `neither/linearized` encoder-control row only if it can be
  implemented as a clean same-contract branch disablement and does not force
  risky model surgery or broaden the run budget.
- Keep fixed across every row: dataset, split, probe preprocessing, seed,
  epoch budget, scheduler, output mode, bottleneck, decoder, skip setting,
  metric schema, fixed visual sample IDs, and shared visual scales.

## Required Interpretation

- Frame the branch ablation around local spatial features versus global
  spectral coupling. Do not imply the local branch alone reconstructs the
  object.
- Treat encoder branch disablement as a mechanistic ablation, not a new
  default model family.
- Treat the supervised SRU-Net row as an objective-control row. It answers
  whether the SRU-Net architecture remains strong without the CDI
  physics-consistency loss; it does not replace the PINN-trained CDI headline
  row.
- Compare against existing FFNO, FNO, U-NO, CNN/U-Net-class, and spectral
  bottleneck rows by reference only. Do not rerun completed rows just to assemble
  the new table.

## Outputs

- A new append-only artifact root under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-branch-objective-ablation/`
- Row-local invocation/config/history/metrics/reconstruction artifacts for
  each fresh row.
- A concise durable summary under:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`
- Updates to the evidence matrix, model variant index, ablation index, and
  study index so the result is discoverable from the normal evidence entry
  points.
- If manuscript/table artifacts are refreshed, they must preserve the completed
  base `lines128` bundle by lineage and clearly label this as append-only
  architecture/objective-control evidence.

## Notes For Reviewer

- Reject plans that turn this into a broad Hybrid ResNet hyperparameter sweep.
- Reject plans that conflate this branch-disable ablation with the already
  completed encoder branch-gating / LayerScale ablation. Gating tests learned
  branch balance; this item tests branch necessity by removing one branch at a
  time.
- Reject implementations that change both branch availability and decoder,
  bottleneck, skip policy, loss, probe, or schedule in the same row.
- If a branch cannot be disabled cleanly, emit a precise implementation blocker
  and keep any completed supervised SRU-Net objective-control row as a separate
  append-only result rather than weakening the fixed contract.
