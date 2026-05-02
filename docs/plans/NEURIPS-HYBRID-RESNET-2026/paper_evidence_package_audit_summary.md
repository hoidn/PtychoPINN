# NeurIPS Hybrid ResNet Paper Evidence Package Audit Summary

## Status Vocabulary

- `paper_grade`: Evidence satisfies the current paper-facing provenance contract and may support headline claims.
- `full_training`: Evidence proves a same-protocol full-training benchmark lane rather than a capped substitute.
- `capped_decision_support`: Evidence is coherent and manuscript-usable only with explicit bounded capped wording.
- `decision_support`: Evidence is useful for local comparison or continuity context but is not a current headline authority.
- `blocked`: A required row or claim cannot be promoted under the present authority set.
- `not_protocol_compatible`: A result exists but cannot be used as the same-contract production answer for this paper lane.

## Current Authorities

- CDI headline authority: `paper_grade` under `complete_lines128_cdi_benchmark` from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`.
- CDI bundle status: `paper_complete` with selected comparator `fno_vanilla` and fixed seed `3`.
- CNS headline authority: `capped_decision_support` under `bounded_capped_decision_support_only` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle`.
- CNS bundle status: `paper_complete` reflects table/figure assembly completeness only; it does not upgrade the pillar beyond `capped_decision_support`.
- CNS 2048cap companion bundle: `capped_decision_support` under `bounded_capped_decision_support_only` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap` (same-contract `2048 / 256 / 256`, `history_len=2`, 40 epochs). Published alongside the 512cap bundle as a wider-cap companion view; does not relabel any row `paper_grade` or `full_training`.
- No outputs from this item target `/home/ollie/Documents/neurips/`; all emitted paths stay repo-local.

## Emitted Outputs

- Manifest path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- Summary path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`
- Validation payload: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/audit_validation.json`
Verification logs:
- `required_inputs_check_log`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/verification/required_inputs_check.log`
- `pytest_log`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/verification/pytest_paper_evidence_audit.log`
- `audit_output_validation_log`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/verification/audit_output_validation.log`
- `audit_direct_entrypoint_log`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/verification/audit_direct_entrypoint.log`

## Draftable Now

- The CDI pillar is draftable now as the current paper-grade anchor because the complete six-row Lines128 bundle is the headline authority.
- The CNS pillar is draftable only with bounded capped wording because every current headline row remains `capped_decision_support`.
- Cross-pillar manuscript language is draftable only if it keeps the asymmetry explicit: paper-grade CDI anchor plus bounded capped CNS support.
- full-training CNS competitiveness claims remain blocked.

- `results_cdi_lines128`: `draftable_now` with supporting status `paper_grade`.
- `results_pdebench_cns`: `draftable_with_bounded_wording` with supporting status `capped_decision_support`.
- `results_cross_pillar_takeaway`: `draftable_with_asymmetric_evidence` with supporting status `mixed`.
- `results_cns_full_training_competitiveness`: `placeholder_only` with supporting status `blocked`.
- `results_inverse_wave_candidate_lane`: `placeholder_only` with supporting status `blocked`.

## Placeholder-Only Or Blocked Claims

- `same_protocol_full_training_cns_competitiveness`: The current CNS paper authority is capped decision-support only.
- `history_len3_locked_cns_headline_lane`: The locked CNS headline contract remains history_len=2 with authored FFNO present only there.
- `minimum_subset_is_current_cdi_headline_authority`: The complete six-row Lines128 bundle supersedes the earlier minimum subset as the CDI headline authority.
- `paper_facing_neurips_bundle_emitted`: This item is repo-local only and must not populate /home/ollie/Documents/neurips/ yet.

## Adjacent Context

### CDI Continuity Context

- `cdi_lines128_minimum_subset`: `decision_support` under `minimum_draftable_cdi_subset` from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`.
- `cdi_lines128_ffno_prerequisite_pair`: `decision_support` under `lines128_ffno_vs_hybrid_prerequisite_pair` from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`.

### CNS Continuity Context

- `hybrid_resnet_cns`: continuity/support only under the same capped contract.
- `history_len=3 pilots`: `not_protocol_compatible` because history_len diverges from the selected history_len=2 headline lane and the authored FFNO row was not completed under that alternate temporal contract by the cutoff
- `history_len=1 pilots`: `not_protocol_compatible` because lower-context Markov ablation is contract-divergent temporal context only, not part of the locked headline table
- `gnot`: `not_protocol_compatible` because protocol-divergent environment and recipe lane; not part of the required same-contract headline roster
- `ffno_bottleneck_base`: `not_protocol_compatible` because repo-local FFNO proxy row is explicitly not an authored FFNO substitute for the locked paper row bundle
- `ffno_bottleneck_localconv_base`: `not_protocol_compatible` because repo-local FFNO proxy with local branch is explicitly adjacent context only and cannot replace authored FFNO in the locked roster
