---
priority: 39
plan_path: docs/backlog/active/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation.md
check_commands:
  - pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or resnet_decoder_block or skip_style"
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_skip or hybrid_resnet_blocks or resnet_width"
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing skip/residual ablation summary: {missing}")
    print("skip/residual ablation summary present")
    PY
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - This is a same-contract Lines128 CDI Hybrid ResNet ablation, not a new headline benchmark row family.
  - It should isolate decoder skip connections and bottleneck residual scaling from probe, loss, dataset, scheduler, encoder-fusion, and comparator-model changes.
  - Results must be appended and cross-referenced with existing relevant ablation evidence rather than rewriting completed paper bundles.
---

# Backlog Item: Lines128 Hybrid ResNet Skip And Residual Ablation

## Objective

- Quantify how decoder skip connections and bottleneck residual scaling affect
  `hybrid_resnet` performance on the fixed `N=128` lines CDI contract, then
  publish an append-only summary that cross-references existing relevant
  ablation evidence.

## Scope

- Consume the completed Lines128 CDI paper benchmark execution as the baseline
  contract authority.
- Use the same dataset, split, probe, seed policy, loss, scheduler, training
  procedure, metrics, and visual contract as the completed Lines128
  `pinn_hybrid_resnet` row unless the implementation plan records an explicit
  reviewed blocker.
- Run only Hybrid ResNet variants needed to isolate these axes:
  - baseline reference: current completed `pinn_hybrid_resnet` row
    (`hybrid_skip_connections=false`, bottleneck residual scaling enabled);
  - skip-enabled rows: at minimum `hybrid_skip_connections=true` with
    `hybrid_skip_style=add`; include `concat` and `gated_add` only if the plan
    can keep the run budget bounded and same-contract;
  - residual-scaling rows: compare the current learned bottleneck residual gate
    against a disabled or fixed-equivalent residual-scaling control. The plan
    must state the exact implementation route, such as a config knob, a narrow
    profile, or a controlled patch, and must preserve checkpoint/provenance
    clarity.
- If both axes are feasible in one item, include the interaction row that
  combines the best skip setting with the residual-scaling control. If not,
  explain why the interaction is deferred.
- Emit metrics and visuals using the same table/figure schema as the completed
  Lines128 benchmark wherever practical.

## Append-Only Evidence Requirements

- Do not rewrite or rerun the completed six-row Lines128 paper bundle.
- Create a new dated ablation artifact root and a durable summary at:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md`.
- Cross-reference, without mutating, these existing evidence surfaces when
  relevant:
  - completed Lines128 complete-paper benchmark summary and artifact root;
  - the older Hybrid ResNet skip/mode search design and any discoverable scored
    outputs from that lane;
  - CNS skip-add evidence, explicitly labeled as PDEBench/CNS context rather
    than CDI evidence;
  - the active encoder-fusion backlog item, explicitly labeled as a separate
    future encoder-branch ablation.
- The summary must distinguish:
  - same-contract Lines128 CDI findings from non-CDI context;
  - fresh rows from reused baseline rows;
  - paper-facing implications from decision-support-only architecture context.

## Notes for Reviewer

- Do not let this item become a broad Hybrid ResNet hyperparameter sweep.
- Do not mix this with encoder branch gates or encoder LayerScale; those belong
  to the separate encoder-fusion backlog item.
- Do not treat CNS skip-add improvement as evidence that skip connections help
  CDI; it is only a prior/context link.
- If residual scaling cannot be disabled cleanly without risky model surgery,
  the correct outcome is a narrow blocked/deferred note plus the skip-only
  ablation, not an ad hoc untracked code mutation.
