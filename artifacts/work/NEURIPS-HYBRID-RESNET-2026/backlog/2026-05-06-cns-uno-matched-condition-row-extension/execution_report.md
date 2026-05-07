# Execution Report

## Completed In This Pass

- Added the manual-only CNS U-NO profile `neuralop_uno_cns_base`, the
  task-local adapter `scripts/studies/pdebench_image128/uno_adapter.py`, and
  the model-builder hook needed to instantiate the external
  `neuralop.models.UNO` body under the PDEBench CNS tensor contract.
- Added focused regressions covering U-NO profile selection, external-source
  provenance capture, equal-footing training-recipe wiring, and derived
  plus-U-NO table emission.
- Launched exactly one same-contract CNS U-NO row under the locked matched h5
  contract. The first launch attempt failed immediately because the output root
  had been precreated; the rerun completed under the same contract and produced
  fresh metrics/provenance artifacts at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/runs/cns_uno_h5_512cap_40ep_20260507T024412Z`.
- Published the derived five-row plus-U-NO bundle at the item root and
  paper-local `tables/pdebench_cns_matched_condition_metrics_plus_uno.{json,csv,tex}`.
- Wrote the durable summary and updated the evidence matrix, paper evidence
  index, model variant index, and studies index so all discoverability surfaces
  use the same current-authority wording.

## Completed Plan Tasks

- Task 1 completed: froze the matched `history_len=5`, `512 / 64 / 64`,
  `40`-epoch, batch-`4`, MSE contract and kept the existing four-row table as
  the binding baseline.
- Task 2 completed: added the CNS U-NO profile/builder surface and emitted
  external-source provenance plus task-local adaptation metadata.
- Task 3 completed: ran exactly one U-NO row under the locked contract and
  captured fresh metrics, manifests, and comparison artifacts.
- Task 4 completed: emitted the derived five-row plus-U-NO JSON/CSV/TeX bundle
  plus item-local lineage payloads without overwriting the current four-row
  headline authority files.
- Task 5 completed: wrote the new summary authority and updated the durable
  human-readable and machine-readable discovery surfaces.

## Remaining Required Plan Tasks

None. The approved implementation scope is complete, and the manuscript-facing
recommendation is explicit: keep the current four-row matched-condition CNS
headline table unchanged and treat the five-row plus-U-NO bundle as adjacent
append-only context only.

## Verification

Comparison standard: exact same-contract equality on
`history_len=5`, `split_counts=512/64/64`, `max_windows_per_trajectory=8`,
`epochs=40`, `batch_size=4`, and `training_loss=mse`. No `atol`/`rtol`
numerical parity gate was specified for this backlog item.

- `python - <<'PY' ...` deterministic input-presence gate from the execution
  plan: passed.
- `pytest -q tests/studies/test_pdebench_image128_models.py -k "uno or profile or cns"`:
  passed (`40 passed, 8 deselected`).
- `pytest -q tests/studies/test_pdebench_image128_runner.py -k "matched_condition or pdebench_cns or uno"`:
  passed (`1 passed, 56 deselected`).
- `pytest -q tests/studies/test_paper_results_refresh.py -k "cns_matched_condition or plus_uno"`:
  passed (`7 passed, 34 deselected`).
- `python -m compileall -q ptycho_torch scripts/studies`: passed.
- Derived-asset consistency check across
  `pdebench_cns_matched_condition_metrics_plus_uno.{json,csv,tex}` and the
  item-local lineage payloads: passed.

## Residual Risks

- The successful tmux-managed run produced the full fresh artifact set, but no
  separate shell-exit proof file was persisted outside the launcher log. Fresh
  metrics/manifests and the completed comparison bundle indicate success, so
  this remains a minor provenance caveat rather than a blocker.
