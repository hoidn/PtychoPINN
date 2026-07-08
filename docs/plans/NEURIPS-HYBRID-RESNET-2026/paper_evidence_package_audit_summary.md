# NeurIPS Hybrid ResNet Paper Evidence Package Audit Summary

## Status Vocabulary

- `paper_grade`: Evidence satisfies the current paper-facing provenance contract and may support headline claims.
- `full_training`: Evidence proves a same-protocol full-training benchmark lane rather than a capped substitute.
- `capped_decision_support`: Evidence is coherent and manuscript-usable only with explicit bounded capped wording.
- `decision_support`: Evidence is useful for local comparison or continuity context but is not a current headline authority.
- `paper_approved_secondary`: Evidence is approved additive secondary manuscript context with its own claim boundary; it does not replace the primary CDI/CNS pillars.
- `blocked`: A required row or claim cannot be promoted under the present authority set.
- `not_protocol_compatible`: A result exists but cannot be used as the same-contract production answer for this paper lane.

## Current Authorities

- CDI headline authority: `paper_grade` under `complete_lines128_cdi_benchmark` from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`.
- CDI bundle status: `paper_complete` with selected comparator `fno_vanilla` and fixed seed `3`.
- CNS headline authority: `capped_decision_support` under `bounded_capped_decision_support_only` from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-matched-condition-table-refresh` (manuscript headline role `matched_condition_history_len_5_512_64_64_40ep`, manuscript summary `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`).
- CNS bundle status: `paper_complete` reflects table/figure assembly completeness only; it does not upgrade the pillar beyond `capped_decision_support`.
- CNS larger-cap capped context preserved for provenance: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap` (summary `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md`) under the same capped claim boundary; it is no longer the current manuscript headline.
- Historical CNS fallback bundle preserved for provenance: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle` under the same capped claim boundary; it is no longer the current discoverability target.
- BRDT additive secondary authority: `paper_approved_secondary` under `paper_evidence_brdt_additive` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence` (summary `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_40ep_paper_evidence_summary.md`, lane `brdt_sinogram_input_40ep`). Headline learned rows: `ffno, sru_net`.
- BRDT historical lineage preserved for provenance only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence`, `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun`. These remain valid only for the older `born_init_image` contract and must not source `paper_evidence_brdt_additive` claims.
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
- BRDT context is draftable as additive secondary evidence only, bounded by `paper_evidence_brdt_additive`; it does not replace CDI `lines128` or PDEBench CNS.
- full-training CNS competitiveness claims remain blocked.

- `results_cdi_lines128`: `draftable_now` with supporting status `paper_grade`.
- `results_pdebench_cns`: `draftable_with_bounded_wording` with supporting status `capped_decision_support`.
- `results_cross_pillar_takeaway`: `draftable_with_asymmetric_evidence` with supporting status `mixed`.
- `results_cns_full_training_competitiveness`: `placeholder_only` with supporting status `blocked`.
- `results_inverse_wave_candidate_lane`: `placeholder_only` with supporting status `blocked`.
- `results_brdt_additive_secondary`: `draftable_as_additive_secondary_only` with supporting status `paper_approved_secondary`.

## Placeholder-Only Or Blocked Claims

- `same_protocol_full_training_cns_competitiveness`: The current CNS paper authority is capped decision-support only.
- `history_len3_locked_cns_headline_lane`: The locked CNS headline contract remains history_len=2 with authored FFNO present only there.
- `minimum_subset_is_current_cdi_headline_authority`: The complete six-row Lines128 bundle supersedes the earlier minimum subset as the CDI headline authority.
- `paper_facing_neurips_bundle_emitted`: This item is repo-local only and must not populate /home/ollie/Documents/neurips/ yet.
- `brdt_replaces_cdi_or_cns_pillar`: BRDT remains additive secondary context only; it does not replace the required CDI or CNS pillars.
- `same_protocol_full_training_brdt_competitiveness`: The BRDT lane is bounded by paper_evidence_brdt_additive and does not authorize same-protocol full-training competitiveness claims.

## Adjacent Context

### CDI Continuity Context

- `cdi_lines128_minimum_subset`: `decision_support` under `minimum_draftable_cdi_subset` from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`.
- `cdi_lines128_ffno_prerequisite_pair`: `decision_support` under `lines128_ffno_vs_hybrid_prerequisite_pair` from `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`.

### CNS Continuity Context

- `hybrid_resnet_cns`: continuity/support only under the same capped contract.


### Additive Secondary BRDT Context

- Authority: `brdt_sinogram_input_40ep` under `paper_evidence_brdt_additive` from `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence` (summary `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_40ep_paper_evidence_summary.md`).
- Headline learned rows: `ffno`, `sru_net`; non-learned reference rows: `classical_born_backprop`.
- Historical `born_init_image` lineage preserved for provenance only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence`, `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun`.
- BRDT remains additive secondary evidence and does not promote past `paper_approved_secondary`.
