# Phase 0 Evidence Inventory Execution Report

## Completed In This Pass

- Fixed the high-severity implementation-review blocker in `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md`: the Phase 3 launch template is now a wrapper-level `grid_lines_compare_wrapper.py` command that parses for the named entrypoint.
- Documented the boundary for omitted child-runner fields. Old standard: pass runner-only flags through the wrapper command. New standard: use the wrapper-supported flags, rely on the current `TorchRunnerConfig` defaults for Hybrid ResNet structural fields and `probe_mask=off`, and verify those defaults from emitted child invocation/config artifacts in Phase 3.
- Hardened `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-0-evidence-inventory/execution_plan.md` with an extracted-command parse check for the documented regeneration command.
- Fixed the medium-severity raw-artifact contract defect in `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_baseline_candidates.json` by replacing the non-standard `NaN` FRC phase value with JSON `null`.
- Updated the execution-plan raw-artifact gates to use strict JSON parsing so future `NaN`, `Infinity`, or `-Infinity` values fail before Phase 3 consumes the inventory.
- Updated `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md` to record the parseable wrapper command and strict-JSON baseline artifact behavior.

## Completed Current-Scope Work

- The CDI anchor regeneration handoff is executable at the wrapper boundary without launching training or changing model/runbook behavior.
- The raw baseline inventory is now consumable by strict JSON parsers; unavailable numeric values are represented as `null`.
- Current Phase 0 approval gates now cover both implementation-review findings: wrapper command parseability and strict JSON artifact parsing.
- No later roadmap work was implemented: no PDE benchmark was selected, no PDE/CDI training was launched, no N=256 scaling run was promoted, and no `/home/ollie/Documents/neurips/` artifact was created.

## Verification

- `inventory docs present`
- `phase 0 structural gates passed`
- `regeneration plan gate passed`
- `inventory gate language present`
- `docs index references evidence inventory`
- `plan_path pointer intact`
- `execution_report_path pointer intact`
- Strict JSON parsing passed for all raw Phase 0 inventory JSON artifacts.
- `grid_lines_compare_wrapper.parse_args` accepts the documented wrapper-level regeneration command.

## Follow-Up Work

- Phase 1 must score and select a primary and fallback PDE benchmark.
- Phase 2 must run the selected PDE benchmark or pivot to the fallback.
- Phase 3 must regenerate the `128x128` CDI Hybrid ResNet anchor, verify emitted child invocation/config artifacts for the default Hybrid ResNet fields, and rerun or recover protocol-compatible CDI baselines.
- Phase 4 may evaluate N=256 scaling only after the CDI/PDE core evidence is secure.
- Phase 5 must assemble `/home/ollie/Documents/neurips/` evidence artifacts only after the evidence exists.

## Residual Risks

- No paper-grade `128x128` `pinn_hybrid_resnet` anchor was recovered; Phase 3 must run the regeneration before paper claims.
- The wrapper command does not expose every child-runner Hybrid ResNet structural flag; Phase 3 must verify those fields from the child artifacts or add wrapper support under a separately reviewed plan if explicit overrides are needed.
- Historical CDI and baseline metrics remain decision-support only until provenance or same-protocol reruns exist.
- PDE candidates were inventoried from primary sources but not installed, downloaded, or smoked; Phase 1 must reject any dataset that exceeds the local disk/GPU budget.
- Local root disk had about 31 GB free during this pass, so full benchmark downloads are likely infeasible without external storage or a smaller shard.
