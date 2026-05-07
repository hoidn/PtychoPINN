# Execution Report

## Completed In This Pass

- Promoted the corrected paired BRDT `40`-epoch bundle at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/`
  to the additive-secondary claim boundary `paper_evidence_brdt_additive`.
- Wrote the durable summary
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_corrected_ffno_40ep_rerun_summary.md`
  and updated the BRDT evidence/package/index surfaces so they agree on the
  backlog item, artifact root, and final claim boundary.
- Rebuilt bundle metadata without retraining; the corrected bundle gate now
  passes with no failed checks and truthful runtime provenance.
- Refreshed repo-local BRDT packaging surfaces so the manuscript-local metrics,
  efficiency, and model-configuration tables consume the corrected paired
  bundle instead of the historical FFNO-local-refiner proxy.

## Completed Plan Tasks

- Task 1: corrected `40`-epoch runner surface and tests were completed earlier
  in this pass; the current summary/report/doc work kept that code unchanged.
- Task 2: dry-run bundle proof was completed earlier in this pass and already
  showed the corrected backlog identity, corrected FFNO lineage root, and the
  paired `hybrid_resnet` + `ffno` roster.
- Task 3: the live corrected `40`-epoch bundle was completed earlier in this
  pass under tmux with both neural rows rerun in one immutable root and
  runtime provenance captured from the training run.
- Task 4: recomputed the gate through `--rebuild-meta-only`, wrote the new
  corrected `40`-epoch summary, and recorded the truthful final gate result
  (`promotion_status=passed`, `claim_boundary=paper_evidence_brdt_additive`).
- Task 5: refreshed `paper_evidence_index.md`,
  `paper_evidence_manifest.json`, `model_variant_index.json`,
  `ablation_index.json`, `paper_evidence_package_design.md`, `docs/index.md`,
  and the dependent repo-local BRDT packaging tables/figure.
- Task 6: archived verification logs, reran the required deterministic checks,
  and revalidated the promoted gate outputs.

## Remaining Required Plan Tasks

- None.

## Verification

- Same-contract comparison standard: exact equality for the locked BRDT
  contract surfaces recorded by the machine-written manifests and gate lineage
  checks; no relaxed `atol`/`rtol` tolerance was used for the plan’s contract
  audit surfaces.
- `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/verification/input_presence.log`
  records the required input-presence check and ends with
  `corrected BRDT 40-epoch inputs present`.
- `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/verification/pytest.log`
  records `131 passed in 370.61s (0:06:10)` for
  `pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py`.
- `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/verification/compileall.log`
  records the required compile check and ends with `compileall_ok`.
- `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/verification/meta_rebuild.log`
  records a successful `--rebuild-meta-only` run after the surface updates.
- Post-rebuild bundle checks now agree on the final boundary:
  `paper_evidence_gate.json`, `preflight_manifest.json`, `metrics.json`,
  `combined_metrics.json`, and `metric_schema.json` all carry
  `claim_boundary=paper_evidence_brdt_additive`.
- `paper_evidence_gate.json` now records `promotion_status=passed`,
  `failed_gate_checks=[]`, and every provenance check as `true`, including
  `evidence_surfaces_prepared`, `git_provenance`, `host_provenance`,
  `same_contract_lineage`, and `sample_255_visual_bundle`.

## Residual Risks

- Both neural rows remain materially improving at stop, so the correct read is
  bounded additive-secondary evidence rather than a full-convergence or
  same-protocol full-training BRDT competitiveness claim.
- The sample-`255` classical/model-based Born comparator is inherited by
  lineage from the frozen baseline bundle rather than regenerated in this pass.
- The historical `2026-05-05-brdt-supervised-born-40ep-paper-evidence` bundle
  remains preserved for provenance, but its FFNO row is proxy-only context and
  is superseded for paired pure-FFNO interpretation by the corrected `2026-05-06`
  authorities.
