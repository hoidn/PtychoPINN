# Execution Report

## Completed In This Pass

- Closed the implementation-review High issue on the matched-condition U-NO
  packaging path. The plus-U-NO acceptance path now enforces dataset (split)
  identity end-to-end:
  - Added `_compute_split_identity()` in
    `scripts/studies/paper_results_refresh.py` that derives a deterministic
    SHA-256 over the canonical JSON of `split_manifest.splits` plus `seed` and
    `source_file.path`. The hash captures the exact per-split trajectory IDs
    used during training, not just the split counts.
  - Extended `load_cns_matched_condition_uno_row()` to fail closed if
    `split_manifest.json` is missing the `splits` block, and to expose the
    computed `dataset_identity` (`splits_hash`, `seed`, `source_file_path`)
    on the loaded U-NO row.
  - Added `_load_headline_dataset_identity()` and wired it into
    `_build_cns_matched_condition_plus_uno_decision()` so the decision builder
    now reads each headline run root's `split_manifest.json`, requires all
    four headlines to agree on `splits_hash`, and rejects any U-NO row whose
    `splits_hash` differs from the headline lineage.
- Propagated the verified `dataset_identity` into every published artifact
  the bundle emits: the `plus_uno_decision.json`, the paper-local
  `pdebench_cns_matched_condition_metrics_plus_uno.{json,csv,tex}`, the
  item-local `cns_paper_table_rows_plus_uno.{json,csv,tex}`, the
  `plus_uno_lineage.json`, and the `plus_uno_row_manifest.json`. Provenance
  for the appended row now exposes `splits_hash`, `seed`, and
  `source_file_path` directly under `dataset_identity` at the bundle level
  and on the U-NO row itself.
- Regenerated the published plus-U-NO bundle from the legitimate run root
  `.artifacts/.../runs/cns_uno_h5_512cap_40ep_20260507T024412Z`. The new
  paper-local JSON now records
  `splits_hash=7f124af99c5f5982451b73b78109097782aed4bf881f93f2ab12a091dc515a80`,
  `seed=20260420`, and the canonical CNS HDF5 path; the headline lineage and
  the U-NO row both produce that same hash.
- Strengthened the regression test suite in
  `tests/studies/test_paper_results_refresh.py`:
  - Updated `_write_valid_cns_uno_run_root` so the loader fixture emits a
    realistic `split_manifest.json` with explicit `splits`, `seed`, and
    `source_file.path` blocks.
  - Added `test_load_cns_matched_condition_uno_row_accepts_matched_contract`
    coverage that the loader now exposes a 64-character `splits_hash` plus
    `seed` and `source_file_path` on the returned row.
  - Added `test_load_cns_matched_condition_uno_row_rejects_missing_splits`
    so the missing-`splits` guard is exercised.
  - Rewrote
    `test_write_cns_matched_condition_plus_uno_assets_emits_five_row_append_only_payload`
    to materialize four headline run roots with matching split manifests under
    `tmp_path`, exercise the new dataset-identity check end-to-end, and assert
    that `dataset_identity.splits_hash` flows into the table payload, lineage,
    and row manifest.
  - Added
    `test_write_cns_matched_condition_plus_uno_assets_rejects_uno_dataset_identity_mismatch`
    and
    `test_write_cns_matched_condition_plus_uno_assets_rejects_disagreeing_headline_splits`
    so a U-NO row trained on a different split slice and a heterogeneous
    headline lineage both raise instead of silently publishing.

## Completed Current-Scope Work

- Implementation-review High issue: the loader and the downstream plus-U-NO
  decision builder no longer silently accept a `512 / 64 / 64` U-NO run on a
  different trajectory slice. The fairness invariant from the approved
  fixed-contract plan ("reuse the existing CNS dataset/split/normalization
  contract exactly. Do not generate new splits") is now enforced as code
  rather than as an unverified narrative claim, and the published paper assets
  now serialize `dataset_identity` alongside the existing fixed-contract
  fields so future review can audit it directly.
- All required deterministic checks from the plan and the
  `cns_matched_condition`/`plus_uno`/loader tranche of
  `tests/studies/test_paper_results_refresh.py` pass after the change,
  including three new dataset-identity regressions.

## Follow-Up Work

- Implementation-review follow-up #2: persist an explicit tracked-run exit
  proof artifact alongside future tmux-launched long CNS runs. The current
  artifact set remains usable for this item; this is a forward-looking
  provenance hardening task, not in current scope.

## Residual Risks

- The dataset-identity hash captures the per-split trajectory IDs, the seed,
  and the source HDF5 path, but it does not recompute the source HDF5's
  SHA-256 or re-derive numerical normalization statistics. A deliberately
  tampered `split_manifest.json` whose `splits` block matches the headline
  lineage byte-for-byte would still pass. This is acceptable for the bounded
  capped decision-support claim boundary but should be tightened if the row
  is ever promoted to a stronger evidence tier.
- The headline split manifests must remain present at their recorded
  `source_run_root` paths for future regenerations; the plus-U-NO build now
  fails closed if any headline run root is missing `split_manifest.json`.
  The four current headline run roots all have that file; future cleanup of
  `.artifacts/` would have to preserve them.

## Verification

Comparison standard: exact same-contract equality on `task_id=2d_cfd_cns`,
`field_order=[density, Vx, Vy, pressure]`, `history_len=5`,
`split_counts=512/64/64`, `max_windows_per_trajectory=8`, `epochs=40`,
`batch_size=4`, `training_loss=mse`, plus exact equality on the
`splits_hash` derived from the per-split trajectory IDs, `seed`, and
`source_file.path` against the matched-condition headline lineage. No
`atol`/`rtol` numerical parity gate was specified for this backlog item.

- `python - <<'PY' ...` deterministic input-presence gate from the execution
  plan: passed.
- `pytest -q tests/studies/test_pdebench_image128_models.py -k "uno or profile or cns"`:
  passed (`40 passed, 8 deselected`).
- `pytest -q tests/studies/test_pdebench_image128_runner.py -k "matched_condition or pdebench_cns or uno"`:
  passed (`1 passed, 56 deselected`).
- `pytest -q tests/studies/test_paper_results_refresh.py -k "cns_matched_condition"`:
  passed (`17 passed, 34 deselected`), including the three new dataset-identity
  regressions and the rewritten plus-U-NO publishing test.
- `python -m compileall -q ptycho_torch scripts/studies`: passed.
- Re-published the plus-U-NO bundle with the strengthened code path. The
  resulting `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.json`
  now exposes `dataset_identity.splits_hash=7f124af99c5f5982451b73b78109097782aed4bf881f93f2ab12a091dc515a80`,
  `seed=20260420`, and the canonical CNS Train.hdf5 path, and the appended
  U-NO row carries the matching `dataset_identity` block.
