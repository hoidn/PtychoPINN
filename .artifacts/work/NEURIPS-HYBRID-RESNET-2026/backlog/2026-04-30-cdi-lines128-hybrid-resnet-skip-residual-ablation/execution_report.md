# Execution Report

## Completed In This Pass

- Reran the three mandatory fresh ablation rows under direct row-local training roots: `pinn_hybrid_resnet_skip_add`, `pinn_hybrid_resnet_residual_fixed`, and `pinn_hybrid_resnet_skip_add_residual_fixed`.
- Rebuilt the append-only comparison bundle from the direct fresh-row outputs and archived the rerun and collation logs under `verification/`.
- Removed the stale recovery-manifest references from the active summary/report surfaces and corrected the skip-add changed-factor wording in the evidence matrix.

## Completed Current-Scope Work

- Resolved the blocking implementation-review issue: each fresh row now has direct `invocation.json` provenance pointing at `training_runs/<row_id>`, plus fresh run/recon artifacts generated from those row-local roots.
- Re-ran the required deterministic selectors and supporting gates (`test_fno_generators`, `test_grid_lines_torch_runner`, study helper, summary presence, integration, and `compileall`) and archived the passing logs under the ablation root.
- Preserved scope and authority boundaries: the completed six-row CDI benchmark remains unchanged, the reused `pinn_hybrid_resnet` baseline stays promoted from the authoritative source root, and optional `pinn_hybrid_resnet_skip_gated_add` remains deferred.

## Follow-Up Work

- Optional only: `pinn_hybrid_resnet_skip_gated_add` remains the bounded next row if a later approved plan reopens this ablation family.

## Residual Risks

- Scientific interpretation is unchanged: this remains same-contract, two-test-image, decision-support CDI evidence rather than promoted paper-grade headline evidence.
- The fixed residual-scale read remains narrow to this Hybrid ResNet shell and frozen `lines128` contract; broader transfer claims are still unsupported.
