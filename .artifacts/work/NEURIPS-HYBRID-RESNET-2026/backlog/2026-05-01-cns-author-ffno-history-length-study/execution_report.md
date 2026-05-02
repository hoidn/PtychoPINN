# Execution Report - CNS Authored FFNO History-Length Study

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-05-01-cns-author-ffno-history-length-study`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-01-cns-author-ffno-history-length-study/execution_plan.md`
- Summary authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md`
- Implementation state: COMPLETED

## Completed In This Pass

- Fixed the remaining review-reported schema gap in
  `write_history_delta_compare(...)` so the `history_len=3` compare
  payload now emits top-level `claim_scope` metadata, and the generated
  CSV carries the same claim-boundary column as the `history_len=4/5`
  multi-reference compare family.
- Added a focused regression assertion in
  `tests/studies/test_pdebench_image128_runner.py` that fails if the
  history-delta compare payload drops `claim_scope` again.
- Regenerated
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/compare_40ep_history3_against_history2.json`
  and `.csv` with the repaired schema while preserving the existing
  metric values and provenance.
- Archived the missing blocking Task 1 verification under the
  authoritative backlog-item artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/verification/history2_anchor_artifact_check.json`
  records the frozen `history_len=2` anchor file check, and the current
  pytest / compileall logs were copied into the same `verification/`
  directory.
- Updated the durable summary
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md`
  so its verification section points at the authoritative artifact-root
  verification archive and accurately describes the repaired compare
  sidecar claim-boundary label.

## Completed Current-Scope Work

- Task 1 is now actually closed: the frozen authored-FFNO
  `history_len=2` anchor is both verified and archived under the
  backlog-item artifact root required by the plan.
- Task 2 through Task 7 remain complete from the prior pass: the audit,
  `history_len=3/4/5` inspect and pilot runs, gate decisions, compare
  payloads, durable summary, and discoverability updates still stand.
- No current-scope implementation, contract, or approval-gating review
  work remains for this backlog item after the review fixes in this
  pass.

## Follow-Up Work

- Normalize the history-delta and multi-reference compare schemas more
  broadly so every compare sidecar carries the same claim-boundary
  fields at the same nested levels, not just the top-level
  `claim_scope` now required for approval.

## Verification

Required deterministic checks rerun in this pass:

- `pytest -q tests/studies/test_pdebench_image128_models.py -k 'author_ffno'`
  — 4 passed, 41 deselected.
- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  — 62 passed.
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  — exit 0, no output.
- `pytest -q tests/studies/test_pdebench_image128_runner.py -k 'history3_cross_run_compare_records_increase_direction_and_dynamic_labels'`
  — 1 passed, 55 deselected.

Verification artifacts:

- Work-log archive:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-01-cns-author-ffno-history-length-study/verification/`
- Authoritative backlog-item archive:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/verification/`
- Blocking Task 1 frozen-anchor record:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/verification/history2_anchor_artifact_check.json`
  — `PASS`, all required anchor files present.

Required contract proofs:

- `history3-inspect-20260502T045749Z`:
  `sample_contract = concat u[t-3:t] -> u[t]`, `history_len=3`,
  emitted windows `4096 / 512 / 512`, `max_windows_per_trajectory=8`,
  field order `[density, Vx, Vy, pressure]`.
- `history4-inspect-20260502T062027Z`:
  `sample_contract = concat u[t-4:t] -> u[t]`, `history_len=4`,
  emitted windows `4096 / 512 / 512`, same field order.
- `history5-inspect-20260502T074436Z`:
  `sample_contract = concat u[t-5:t] -> u[t]`, `history_len=5`,
  emitted windows `4096 / 512 / 512`, same field order.

Tracked completion proofs (each `exit_code = 0`):

- `launch-history3-pilot-40ep-20260502T045955Z` (4860 s)
- `launch-history4-pilot-40ep-20260502T062100Z` (4981 s)
- `launch-history5-pilot-40ep-20260502T074500Z` (5100 s)

Comparison standard:

- the per-history-length contract delta is restricted to
  `delta_kind = history_len_only`. Numerical comparisons are direct
  metric differences (`fresh - reference`); the `history4` and
  `history5` gates require strict improvement (delta < 0) on
  `err_nRMSE`, `err_RMSE`, `relative_l2`, and use a floating-point
  tolerance `+1e-6` for the no-regression check on `fRMSE_high`.

Final per-row metrics at `40` epochs:

| history_len | err_nRMSE | err_RMSE | relative_l2 | fRMSE_low | fRMSE_mid | fRMSE_high | runtime_sec | params |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.0281477310 | 0.6802443266 | 0.0281477310 | 1.6124732494 | 0.0759288296 | 0.1210141182 | 4725.51 | 1,073,672 |
| 3 | 0.0230038911 | 0.5554690361 | 0.0230038911 | 1.3145936728 | 0.0578523874 | 0.1176725253 | 4850.00 | 1,073,928 |
| 4 | 0.0193971880 | 0.4684740603 | 0.0193971880 | 1.1064407825 | 0.0476019271 | 0.1138561592 | 4970.44 | 1,074,184 |
| 5 | 0.0197584499 | 0.4772877395 | 0.0197584499 | 1.1303862333 | 0.0420697667 | 0.1018067747 | 5103.25 | 1,074,440 |

Discoverability validation:

- `model_variant_index.json` parses, `model_variants` count `31`.
- `ablation_index.json` parses, `ablation_families` count `16`; the
  `cns_history_length` family now lists this backlog item as a
  completed item, the new summary authority, and the new artifact root.
- The locked headline `history_len=2` authored-FFNO row reused by the
  current CNS paper bundle (`pdebench_cns_paper_2048cap_extension_summary.md`,
  `pdebench_cns_paper_table_figure_bundle_summary.md`,
  `paper_evidence_index.md`, `evidence_matrix.md`) was not edited.

## Residual Risks

- These rows remain `adjacent_capped_context_only` decision-support
  evidence. They should not be cited as headline benchmark performance
  for authored FFNO, and they should not be promoted into the locked
  `history_len=2` CNS paper table or any later `2048 / 256 / 256`
  authority lane without a roadmap-level decision.
- The cross-history compare sidecars do not render PNG galleries
  because the per-step saved targets differ across `history_len`; the
  compare JSON/CSV remain authoritative for the numerical read.
- Within-row gallery PNGs from each pilot run root exist
  (`comparison_author_ffno_cns_base_sample0.png`) but are per-run
  artifacts only.
