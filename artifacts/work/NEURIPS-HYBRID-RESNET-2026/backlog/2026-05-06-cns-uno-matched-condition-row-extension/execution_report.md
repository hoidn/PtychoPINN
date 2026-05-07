# Execution Report

## Completed In This Pass

- Strengthened the matched-condition U-NO loader so it now refuses run roots
  that violate task identity, dataset identity, field order, or normalization
  lineage. Added `task_id` and `field_order` to `CNS_H5_FIXED_CONTRACT` and
  extended `load_cns_matched_condition_uno_row()` to require
  `dataset_manifest.json` and `normalization_stats_state.json` and to validate
  `invocation.parsed_args.task_id`, `dataset_manifest.task_id`,
  `dataset_manifest.field_order`, `normalization_stats_state.field_order`, and
  the normalization-stats `history_len`. Added the matching cross-checks to
  `_build_cns_matched_condition_plus_uno_decision()`.
- Captured the new normalization/dataset lineage artifacts in the published
  U-NO row payload so packaged provenance now exposes
  `source_artifacts.dataset_manifest_json` and
  `source_artifacts.normalization_stats_state_json`.
- Regenerated the paper-local plus-U-NO bundle so the published JSON/CSV/TeX
  carry the new `task_id`/`field_order` provenance fields and so the appended
  row's `source_run_root` is now repo-relative, matching the four inherited
  headline rows (resolves implementation-review follow-up #1 on mixed
  absolute/relative serialization).
- Added focused regression tests for the strengthened loader covering the
  happy path, mismatched invocation `task_id`, mismatched
  `dataset_manifest.task_id`, mismatched `dataset_manifest.field_order`,
  mismatched `normalization_stats_state.field_order`, and missing
  `dataset_manifest.json` / `normalization_stats_state.json`.

## Completed Current-Scope Work

- Implementation-review High issue: blocking contract violation in
  `paper_results_refresh.py` at the previously identified ranges. The loader
  and the downstream plus-U-NO decision builder now enforce the approved
  "same task / same normalization contract" rule and reject silently
  mismatched run roots.
- Implementation-review follow-up #1 (path serialization consistency between
  inherited headline rows and the appended U-NO row) addressed by
  regenerating with a repo-relative run root.

## Follow-Up Work

- Implementation-review follow-up #2: persist an explicit tracked-run exit
  proof artifact alongside future tmux-launched long CNS runs. The current
  artifact set remains usable for this item; this is a forward-looking
  provenance hardening task, not in current scope.

## Residual Risks

- The strengthened validator only constrains task identity, dataset identity,
  field order, and normalization lineage at the run-root level. It does not
  re-derive numerical normalization statistics or recompute dataset MD5s, so
  a deliberately-tampered `normalization_stats_state.json` with consistent
  schema fields could still pass. This is acceptable for the bounded capped
  decision-support claim boundary but should be tightened if the row is ever
  promoted to a stronger evidence tier.
- All required deterministic checks from the plan and the
  `cns_matched_condition`/`plus_uno`/loader tranche of
  `tests/studies/test_paper_results_refresh.py` pass after the change.

## Verification

Comparison standard: exact same-contract equality on `task_id=2d_cfd_cns`,
`field_order=[density, Vx, Vy, pressure]`, `history_len=5`,
`split_counts=512/64/64`, `max_windows_per_trajectory=8`, `epochs=40`,
`batch_size=4`, and `training_loss=mse`. No `atol`/`rtol` numerical parity
gate was specified for this backlog item.

- `python - <<'PY' ...` deterministic input-presence gate from the execution
  plan: passed.
- `pytest -q tests/studies/test_pdebench_image128_models.py -k "uno or profile or cns"`:
  passed (`40 passed, 8 deselected`).
- `pytest -q tests/studies/test_pdebench_image128_runner.py -k "matched_condition or pdebench_cns or uno"`:
  passed (`1 passed, 56 deselected`).
- `pytest -q tests/studies/test_paper_results_refresh.py -k "cns_matched_condition"`:
  passed (`14 passed, 34 deselected`), including the six new loader-rejection
  regressions.
- `python -m compileall -q ptycho_torch scripts/studies`: passed.
- Re-loaded the existing run root at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/runs/cns_uno_h5_512cap_40ep_20260507T024412Z`
  through `load_cns_matched_condition_uno_row()` to confirm the strengthened
  validator still accepts the legitimate run.
