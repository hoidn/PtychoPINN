# Execution Report

## Completed In This Pass

- Audited the current checkout against the consumed design, execution plan,
  predecessor FFNO packaging summary, and both depth-24 FFNO authorities.
- Confirmed the plan outputs already exist in the repo-local paper-refresh
  surface: the final summary, canonical/versioned FFNO asset families, updated
  evidence indexes, the 16-command checks ledger, and the archived verification
  logs under `verification/`.
- Normalized this execution report to the required section contract:
  `Completed In This Pass`, `Completed Plan Tasks`,
  `Remaining Required Plan Tasks`, `Verification`, and `Residual Risks`.
- No real blocker was found, so no blocked progress report was written.
- Left unrelated worktree changes and workflow-owned backlog queue files
  untouched.

## Completed Plan Tasks

- Task 1: Audited the corrected four-block no-refiner pair and both depth-24
  companions, confirmed the locked same-depth rule, and kept
  `four_block_no_refiner` as the final manuscript-facing FFNO pair.
  Promotion stayed rejected because the supervised depth-24 companion is mixed
  while the depth-24 family is materially more expensive.
- Task 2: Parameterized the paper-refresh generators around an explicit final
  FFNO pair/output stem and propagated active-row FFNO provenance through the
  canonical machine-consumed JSON surfaces:
  `cdi_lines128_metrics_extended.json`,
  `model_config_by_benchmark.json`, and
  `paper_efficiency_table.json`.
- Task 3: Regenerated the final repo-local CDI FFNO asset families from the
  chosen four-block no-refiner pair, including canonical compatibility outputs
  and versioned `ffno_final_depth4pair` copies.
- Task 4: Refreshed the durable discovery/evidence surfaces so the promotion
  decision and preserved lineage are consistent across
  `evidence_matrix.md`, `paper_evidence_index.md`,
  `paper_evidence_manifest.json`, `model_variant_index.json`,
  `ablation_index.json`, and `docs/studies/index.md`.
- Task 5: Wrote the durable summary authority at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_final_paper_refresh_summary.md`.
- Non-FFNO `lines128` rows were reused strictly by lineage and were not rerun.

## Remaining Required Plan Tasks

- None within the authority of this backlog item.
- The recurring state-side republication of the 3-command default checks ledger
  remains a workflow-tooling follow-up outside this item's implementation
  authority; it is not a remaining required task for the approved plan.

## Verification

- Archived passing pytest evidence already present under
  `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/`:
  - `pytest_preflight.log`: `11 passed, 44 deselected`
  - `pytest_postfix.log`: `12 passed, 44 deselected`
  - `pytest_model_config_preflight.log`: `7 passed`
  - `pytest_model_config_postfix.log`: `8 passed`
  - `pytest_efficiency_preflight.log`: `9 passed`
  - `pytest_efficiency_postfix.log`: `11 passed`
  - `pytest_collect.log`: `75 tests collected`
- Archived compile checks already present:
  - `compileall_preflight.log`: exit `0`
  - `compileall_postfix.log`: exit `0`
- Archived deterministic refresh already present:
  - `paper_results_refresh_ffno_final_depth4pair.log`: exit `0`
- Archived JSON validation logs already present:
  - `json_paper_evidence_manifest.log`
  - `json_model_variant_index.log`
  - `json_ablation_index.log`
  - `json_cdi_lines128_metrics_extended.log`
  - `json_model_config_by_benchmark.log`
  - `json_paper_efficiency_table.log`
- Machine-readable checks ledger:
  `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json`
  records `command_count: 16`, `failed_count: 0`, `status: "PASS"`.
- Comparison standard consumed from the depth studies:
  - config and invocation fields: exact equality on the locked-contract fields
  - dataset payloads: exact equality for every NPZ key after dropping only
    `_metadata.creation_info.timestamp`

## Residual Risks

- The depth-24 FFNO family remains append-only decision-support evidence, not
  the canonical paper-local FFNO pair.
- The workflow-side `check_commands.json` still enumerates only the 3-command
  default skeleton, so future workflow republication can overwrite the published
  16-command ledger again until the workflow tooling is fixed.
- Manuscript prose and `/home/ollie/Documents/neurips/` remain out of scope for
  this item; the repo-local paper assets are the only refreshed outputs here.
