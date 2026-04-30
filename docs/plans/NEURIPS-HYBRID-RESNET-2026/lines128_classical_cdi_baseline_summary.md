# NeurIPS Lines128 Classical CDI Baseline Summary

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-classical-baseline-feasibility`
- State: `not_protocol_compatible`
- Claim boundary: `lines128_classical_optional_extension`
- Authoritative incompatibility root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/`

## Completed In This Pass

- froze the accepted `lines128` contract and the preserved six-row CDI bundle
  as the prerequisite authority for any optional classical extension
- audited both currently exposed classical solver branches in
  `scripts/reconstruction/hio_cdi_benchmark.py` against the accepted
  `lines128` paper-bundle contract
- recorded the incompatibility closeout in durable checked-in and
  machine-readable artifacts:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_classical_cdi_execution_authority.md`,
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/execution/classical_baseline_execution_manifest.json`,
  and
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/execution/protocol_compatibility_audit.md`
- preserved the existing six-row `paper_complete` Lines128 CDI root unchanged
  and did not launch a classical run because the audit resolved the item before
  any expensive execution step

## Solver Outcome

- `pynx_cdi_hio_er`: `not_protocol_compatible`
  - remains hard-wired to the older Table-2 `N=64` lane
  - emits a fresh-rerun exploratory metric contract rather than the frozen
    `lines128` paper-bundle contract
  - does not emit the required `lines128` extension-bundle tables/manifests
- `known_probe_object_hio_er`: `not_protocol_compatible`
  - inherits the same `N=64` / Table-2 contract mismatch
  - remains a repo-local object-domain diagnostic branch rather than a reviewed
    same-contract classical paper row
  - does not emit the required `lines128` extension-bundle tables/manifests

## Claim Boundary

- the preserved six-row root remains the headline CDI authority:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- this item adds no same-contract classical row to that bundle
- any future classical `lines128` row would need a reviewed broader plan that
  intentionally replaces the current Table-2-specific classical study path with
  an `N=128` bundle-aware implementation

## Verification

- required context check log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/verification/required_context_check.log`
- compile gate log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/verification/compileall_classical_lines128.log`
- focused contract-surface pytest log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/verification/pytest_classical_contract_surfaces.log`
- authority consistency validation log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/verification/authority_consistency_check.log`

## Remaining Caveats

- this is a truthful incompatibility closeout, not a blocked item
- the audit only covers the currently exposed classical branches in
  `scripts/reconstruction/hio_cdi_benchmark.py`; it does not reject all
  possible future classical `lines128` approaches
- any future attempt must start from the frozen `N=128` bundle contract and add
  the full row-schema / provenance / collation surfaces up front rather than
  extending the current Table-2 study path incrementally
