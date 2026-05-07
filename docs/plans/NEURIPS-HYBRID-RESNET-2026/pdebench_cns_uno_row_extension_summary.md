# PDEBench CNS U-NO Row Extension Summary

- Date: `2026-05-06`
- Backlog item: `2026-05-06-cns-uno-matched-condition-row-extension`
- State: `completed`
- Claim boundary: `bounded_capped_decision_support_only`
- Item root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/`
- Successful run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/runs/cns_uno_h5_512cap_40ep_20260507T024412Z`
- Governing plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/execution_plan.md`

## Objective

Append one same-contract NeuralOperator U-NO comparator row to the current
PDEBench CNS matched-condition bundle, then publish a derived five-row
plus-U-NO table by lineage without rerunning the four existing headline rows.

## Fixed Contract

Comparison standard: exact same-contract equality for the locked matched h5
lane.

- task: `2d_cfd_cns`
- spatial size: `128x128`
- `history_len=5`
- split counts: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- `epochs=40`
- `batch_size=4`
- training loss: `mse`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`

## U-NO Contract

The executed row is `neuralop_uno_cns_base`, a narrow PDEBench task-local
adapter around external `neuralop.models.UNO`:

- external distribution: `neuraloperator`
- external module: `neuralop.models.UNO`
- external version: `2.0.0`
- input/output layout: `B,C,H,W -> B,C,H,W`
- task-local config:
  - `hidden_channels=32`
  - `lifting_channels=128`
  - `projection_channels=128`
  - `n_layers=4`
  - `uno_out_channels=[32, 64, 64, 32]`
  - `uno_n_modes=[[12, 12], [12, 12], [12, 12], [12, 12]]`
  - `uno_scalings=[[1, 1], [0.5, 0.5], [1, 1], [2, 2]]`
  - `positional_embedding="grid"`
  - `channel_mlp_skip="linear"`
- parameter count: `1,260,548`

Profile provenance is recorded in:
`.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/runs/cns_uno_h5_512cap_40ep_20260507T024412Z/model_profile_neuralop_uno_cns_base.json`

## Completed In This Pass

- Added the manual-only CNS profile `neuralop_uno_cns_base` and a narrow
  task-local U-NO adapter under
  `scripts/studies/pdebench_image128/uno_adapter.py`.
- Wired the PDEBench image-suite builder to construct the U-NO row through the
  external NeuralOperator body while emitting external-source provenance in the
  row manifest.
- Extended `scripts/studies/paper_results_refresh.py` so it can:
  - validate one fresh U-NO run root against the locked matched h5 contract
  - emit item-local `plus_uno_decision.json`,
    `cns_paper_table_rows_plus_uno.{json,csv,tex}`,
    `plus_uno_lineage.json`, and `plus_uno_row_manifest.json`
  - emit paper-local
    `tables/pdebench_cns_matched_condition_metrics_plus_uno.{json,csv,tex}`
- Launched exactly one same-contract U-NO row under tmux in `ptycho311`.
  The first launcher attempt failed immediately because the output root had
  been precreated and the runner correctly refused to write into a non-empty
  directory. The run was relaunched without changing the contract and the
  successful root above produced fresh metrics, manifests, and comparison
  artifacts.

## Result And Manuscript Recommendation

The derived five-row bundle preserves the current four-row matched-condition
headline table by lineage and appends one U-NO comparator row. The manuscript
recommendation is unchanged: keep
`tables/pdebench_cns_matched_condition_metrics.tex` as the headline table and
use `tables/pdebench_cns_matched_condition_metrics_plus_uno.{json,csv,tex}` as
adjacent append-only context only.

Same-contract result snapshot:

| Row | `relative_l2` | `fRMSE_low` | `fRMSE_mid` | `fRMSE_high` |
|---|---:|---:|---:|---:|
| `author_ffno_cns_base` | 0.019758 | 1.130386 | 0.042070 | 0.101807 |
| `spectral_resnet_bottleneck_base` | 0.033069 | 1.854178 | 0.171350 | 0.262218 |
| `fno_base` | 0.038425 | 2.133659 | 0.122383 | 0.432856 |
| `neuralop_uno_cns_base` | 0.038466 | 2.189750 | 0.078673 | 0.260623 |
| `unet_strong` | 0.538623 | 30.988291 | 0.644871 | 1.742789 |

Interpretation:

- U-NO is effectively tied with same-contract FNO on aggregate error
  (`0.038466` vs `0.038425`).
- U-NO improves over FNO on `fRMSE_mid` and `fRMSE_high`.
- U-NO is slightly better than the SRU-Net row on `fRMSE_high` but still worse
  on aggregate error and low-band error.
- The row does not displace authored FFNO as the strongest same-contract read,
  and it does not justify replacing the current four-row matched-condition
  headline table.

## Artifact Inventory

Item-local row and derived assets:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/runs/cns_uno_h5_512cap_40ep_20260507T024412Z/metrics_neuralop_uno_cns_base.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/runs/cns_uno_h5_512cap_40ep_20260507T024412Z/model_profile_neuralop_uno_cns_base.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/runs/cns_uno_h5_512cap_40ep_20260507T024412Z/comparison_summary.{json,csv}`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/cns_paper_table_rows_plus_uno.{json,csv,tex}`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/plus_uno_decision.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/plus_uno_lineage.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/plus_uno_row_manifest.json`

Paper-local derived assets:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.csv`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.tex`

## Verification

- `python - <<'PY' ...` deterministic input-presence gate from the execution
  plan
- `pytest -q tests/studies/test_pdebench_image128_models.py -k "uno or profile or cns"`
- `pytest -q tests/studies/test_pdebench_image128_runner.py -k "matched_condition or pdebench_cns or uno"`
- `pytest -q tests/studies/test_paper_results_refresh.py -k "cns_matched_condition or plus_uno"`
- `python -m compileall -q ptycho_torch scripts/studies`

## Residual Risks

- The successful tmux-managed run produced the full fresh artifact set, but no
  separate shell-exit proof file was persisted outside the launcher log.
  Treat this as a minor provenance caveat only; it is not a contract blocker.
