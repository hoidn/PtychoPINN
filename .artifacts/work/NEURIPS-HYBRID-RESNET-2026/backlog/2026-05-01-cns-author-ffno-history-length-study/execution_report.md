# Execution Report - CNS Authored FFNO History-Length Study

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-05-01-cns-author-ffno-history-length-study`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-01-cns-author-ffno-history-length-study/execution_plan.md`
- Summary authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md`
- Implementation state: COMPLETED

## Completed In This Pass

- Fixed the remaining compare-sidecar contract gaps in
  [reporting.py](/home/ollie/Documents/PtychoPINN/scripts/studies/pdebench_image128/reporting.py):
  `_load_run_record(...)` now preserves `peak_cuda_memory_bytes`,
  `write_history_delta_compare(...)` now records explicit per-profile
  `metric_deltas` plus fresh/reference row payloads, and both compare
  writers now expose runtime and peak-memory fields in their CSV output.
- Added regression coverage in
  [test_pdebench_image128_runner.py](/home/ollie/Documents/PtychoPINN/tests/studies/test_pdebench_image128_runner.py)
  for the repaired history-delta schema and the peak-memory propagation
  used by the multi-reference compare family.
- Regenerated the authoritative compare sidecars under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/`:
  `compare_40ep_history3_against_history2.{json,csv}`,
  `compare_40ep_history4_against_history2_history3.{json,csv}`, and
  `compare_40ep_history5_against_history2_history3_history4.{json,csv}`
  now include the repaired machine-readable fields while preserving the
  existing run metrics and gate decisions.
- Updated the durable summary
  [pdebench_author_ffno_history_length_summary.md](/home/ollie/Documents/PtychoPINN/docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md)
  so the verification notes explicitly call out the per-profile history
  deltas and runtime/peak-memory coverage now present in the sidecars.
- Archived fresh verification logs for this pass under both:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-01-cns-author-ffno-history-length-study/verification/`
  and
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/verification/`.

## Completed Current-Scope Work

- Task 4 now matches the governing plan: the `history_len=3` compare
  payload records the fixed `history_len_only` contract delta, absolute
  metrics, explicit metric deltas against the frozen `history_len=2`
  authored-FFNO anchor, split/window counts, parameter count, runtime,
  and `peak_cuda_memory_bytes`.
- The related `history_len=4` and `history_len=5` multi-reference
  compare sidecars now expose the same peak-memory field in their
  machine-readable outputs, closing the review-reported gap across the
  compare family without changing any study conclusions.
- All current-scope plan work remains complete after this repair: the
  frozen authored-FFNO anchor audit, `history_len=3/4/5` inspect and
  pilot runs, gate decisions, durable summary, and discoverability
  updates still stand, and no approval-gating review items remain open.

## Follow-Up Work

- Normalize the history-delta and multi-reference compare schemas more
  broadly so every compare sidecar carries the same claim-boundary
  fields at the same nested levels and with the same delta-row naming;
  this is cleanup, not a blocker for the current backlog item.

## Verification

Required deterministic checks rerun in this pass:

- `pytest -q tests/studies/test_pdebench_image128_models.py -k 'author_ffno'`
  — 4 passed, 41 deselected.
- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  — 62 passed.
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  — exit 0, no output.
- `pytest -q tests/studies/test_pdebench_image128_runner.py -k 'history3_cross_run_compare_records_increase_direction_and_dynamic_labels or same_profile_multi_reference_history_compare_writes_dual_anchor_payload'`
  — 2 passed, 54 deselected.

Verification artifacts:

- Work-log archive:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-01-cns-author-ffno-history-length-study/verification/`
- Authoritative backlog-item archive:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/verification/`
- Blocking Task 1 frozen-anchor record:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/verification/history2_anchor_artifact_check.json`
  — `PASS`, all required anchor files present.
- Current pass logs:
  `pytest_models_author_ffno_compare_contract_fix.log`,
  `pytest_runner_data_metrics_compare_contract_fix.log`,
  `compileall_compare_contract_fix.log`, and
  `pytest_history_compare_contract_fix.log` in both verification
  archives.

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
- the regenerated sidecars now expose those metric deltas directly for
  `history_len=3 vs history_len=2`, and all three compare families now
  carry runtime plus `peak_cuda_memory_bytes` in machine-readable JSON
  and CSV outputs.

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
