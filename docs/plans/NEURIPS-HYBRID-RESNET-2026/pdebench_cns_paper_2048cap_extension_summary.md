# PDEBench CNS Paper 2048cap Extension Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-cns-paper-2048cap-row-extension`
- Date: `2026-05-02`
- Outcome: `same_contract_2048_bundle_complete`
- Bundle root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/`
- Locked-rows manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/cns_paper_locked_rows_2048cap.json`
- Contract authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- Prior 512cap bundle (still durable):
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/`

The capped decision-support claim boundary is unchanged. This extension adds a
parallel `2048 / 256 / 256` lane built under exactly the same fairness contract
as the existing `512 / 64 / 64` bundle. The 512cap bundle remains intact and
authoritative; the 2048cap bundle is published alongside it as a wider-cap
companion view, **not** as `paper_grade` or `full_training` evidence.

## Same-Cap Contract

- split counts: `train=2048`, `val=256`, `test=256`
- emitted window counts: `train=16384`, `val=2048`, `test=2048`
- history lane: `history_len=2`
- epoch budget: `40`
- window cap: `max_windows_per_trajectory=8`
- training loss: `mse`
- batch size: `4`
- training recipe: task-local mse override; Adam `lr=2e-4`;
  `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`;
  `batch_size=4`; `epochs=40`
- normalization: train-only per-field stats fit on the 2048 training
  trajectories, reused across all history slots and target channels, with
  evaluation reported in denormalized target space
- claim boundary: `bounded_capped_decision_support_only`

## Headline Roster (All 4 Recovered Under The 2048 Contract)

| row_id | source | err_nRMSE | parameter_count |
| --- | --- | ---: | ---: |
| `spectral_resnet_bottleneck_base` | reused 2048cap scaling run | 0.04217 | 8,391,814 |
| `fno_base` | item-local rerun (`rerun_candidates/fno-2048cap-40ep`) | 0.05072 | 357,860 |
| `unet_strong` | helper-tracked replacement rerun (`rerun_candidates/unet_strong-2048cap-40ep`) | 0.64315 | 7,764,580 |
| `author_ffno_cns_base` | item-local rerun (`rerun_candidates/author_ffno_cns-2048cap-40ep`) | 0.02631 | 1,073,672 |

Continuity row roster is empty for the 2048cap bundle.

Reused run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z`.

## Audit Outcome

- `2048_same_cap_audit.json.audit_outcome = upgrade_ready`
- `missing_or_incompatible_rows = []`
- audit artifacts:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/2048_same_cap_audit.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/2048_same_cap_audit.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/2048_rerun_manifest.json`

The initial pre-rerun audit reported `fallback_to_512_required` with three
missing headline rows (`fno_base`, `unet_strong`, `author_ffno_cns_base`). After
the same-contract reruns, the item-root audit and the in-bundle audit copy
under `bundle_2048cap/` both report `upgrade_ready`, and the rerun manifest is
empty.

## Bundle Outputs

- table rows (JSON / CSV / TeX):
  - `bundle_2048cap/cns_paper_table_rows.json`
  - `bundle_2048cap/cns_paper_table_rows.csv`
  - `bundle_2048cap/cns_paper_table_rows.tex`
- figure manifest: `bundle_2048cap/figure_manifest.json`
- fixed-sample manifest: `bundle_2048cap/fixed_sample_manifest.json`
- shared scales: `bundle_2048cap/shared_field_scales.json`,
  `bundle_2048cap/shared_error_scales.json`
- source arrays: `bundle_2048cap/figure_sources/sample000/<row_id>.npz`
- rendered figures: `bundle_2048cap/figures/sample000/<field>__<row_id>__{prediction,abs_error}.png`
  and `bundle_2048cap/figures/sample000/<field>__target.png`
- bundle validation: `bundle_2048cap/bundle_validation.json`

`bundle_validation.json` confirms:

- `headline_contract_consistent = true`
- `mixed_cap_headline_table = false`
- `all_rows_capped_decision_support = true`
- `no_paper_grade_or_full_training_labels = true`
- `table_and_visual_row_rosters_agree = true`
- `figure_entries_match_visual_bundle = true`
- `sample_manifest_matches_figure_manifest = true`

## Replacement Rerun Proof

The earlier `unet_strong` 2048cap evidence with reconstructed PID/exit-code
artifacts was archived under `rerun_candidates/` and is no longer the
authoritative row source. The canonical `unet_strong` row now comes from a
fresh helper-launched rerun at the original path
`rerun_candidates/unet_strong-2048cap-40ep`, with verification captured in:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/verification/unet_rerun_replacement_verification.json`

That proof records the helper-tracked PID `1845354`, exit code `0`, the locked
2048 contract, and the final metrics for the replacement run.

## Authority Status

- The existing `512 / 64 / 64` bundle remains the durable paper bundle and
  the authoritative locked-rows manifest pointer.
- The 2048cap bundle is published as a wider-cap companion under the same
  capped decision-support claim boundary.
- No row in either bundle is relabelled `paper_grade` or `full_training`.
- The full-training benchmark gate is still unmet for every model in either
  bundle.

## Verification

- `pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py`
  -> 58 passed (61.93s). Log:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/verification/pytest.log`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  -> exit 0. Log:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/verification/compileall.log`
- Helper verification: `verification/unet_rerun_replacement_verification.json`
  records the replacement `unet_strong` rerun with PID `1845354` and exit code `0`.

## Residual Risks And Open Items

- Hardware accelerator label is `artifact_missing_precise_accelerator` for all
  rows because the run roots do not record a precise accelerator string. This
  matches the prior 512cap bundle and does not change the claim boundary.
- The 2048cap companion bundle remains under the
  `bounded_capped_decision_support_only` claim boundary. Full-training
  benchmark gates remain unmet for every model in either bundle.
