# PDEBench CNS Authored FFNO History-Length Compare Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-05-01-cns-author-ffno-history-length-study`
- Date: `2026-05-02`
- Status: implementation complete; frozen authored-FFNO `history_len=2`
  reference manifest, `history_len=3/4/5` inspect proofs, fresh
  `40`-epoch authored-FFNO pilots for `history_len=3`, `history_len=4`,
  and `history_len=5`, anchored multi-history compare sidecars, and the
  explicit `history4` and `history5` gate decisions are all recorded
- Governing plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-01-cns-author-ffno-history-length-study/execution_plan.md`
- Artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/`

This summary records `adjacent_capped_context_only` decision-support
evidence for the within-model authored-FFNO temporal-context question.
It does not reopen the locked CNS paper lane, does not modify the
`512 / 64 / 64`, `history_len=2` authored-FFNO row used by the locked
headline table, does not promote any later `2048 / 256 / 256` follow-up
lane into current authority by implication, and does not create
`/home/ollie/Documents/neurips/` outputs.

## Fixed Compare Contract

The fresh longer-context authored-FFNO rows and the frozen
`history_len=2` authored-FFNO reference kept the capped local CNS
surface fixed everywhere except temporal context:

- task: `2d_cfd_cns`
- profile: `author_ffno_cns_base`
  (manuscript label `Author FFNO`)
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- emitted windows for every compared lane:
  `4096 / 512 / 512`
- batch size: `4`
- training loss: `mse`
- optimizer: `Adam`, `lr=2e-4`
- scheduler:
  `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- normalization: train-only per-field state stats
- target horizon: one-step
- field order: `density`, `Vx`, `Vy`, `pressure`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`
- claim scope:
  `adjacent_capped_context_only`

Derived history contracts recorded in this pass:

- `history_len=2`: `input_channels=8`, `concat u[t-2:t] -> u[t]`
- `history_len=3`: `input_channels=12`, `concat u[t-3:t] -> u[t]`
- `history_len=4`: `input_channels=16`, `concat u[t-4:t] -> u[t]`
- `history_len=5`: `input_channels=20`, `concat u[t-5:t] -> u[t]`

The emitted capped split counts stayed fixed at `4096 / 512 / 512` for
every lane, so each longer-history claim remains a row-local comparison
on one bounded contract family rather than a new paper bundle. The
authored FFNO body (`FNOFactorized2DBlock`, `n_layers=24`, `width=64`,
`modes=16`, `share_weight=True`, weight-norm feedforward, position
features) and the local output-head adapter were not modified; only
`in_channels` widened to absorb the longer history input.

## Recorded Artifacts

Frozen authored-FFNO reference manifest:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history2_reference_runs.json`

Inspect proofs:

- `history_len=3`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history3-inspect-20260502T045749Z`
- `history_len=4`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history4-inspect-20260502T062027Z`
- `history_len=5`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history5-inspect-20260502T074436Z`

Fresh authored-FFNO `40`-epoch pilots:

- `history_len=3`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history3-pilot-40ep-20260502T045955Z`
- `history_len=4`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history4-pilot-40ep-20260502T062100Z`
- `history_len=5`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history5-pilot-40ep-20260502T074500Z`

Tracked launch-completion proofs (each with `exit_code = 0`):

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/launch-history3-pilot-40ep-20260502T045955Z`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/launch-history4-pilot-40ep-20260502T062100Z`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/launch-history5-pilot-40ep-20260502T074500Z`

Gate records:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history4_gate_decision.json`
  (decision: `open`)
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history5_gate_decision.json`
  (decision: `open`)

Cross-history compare sidecars:

- `compare_40ep_history3_against_history2.json` and `.csv`
- `compare_40ep_history4_against_history2_history3.json` and `.csv`
- `compare_40ep_history5_against_history2_history3_history4.json` and `.csv`

## Results

Fresh authored-FFNO row metrics by history length at the matched
`40`-epoch budget:

- `history_len=2` (frozen anchor):
  `err_nRMSE = relative_l2 = 0.0281477310`,
  `err_RMSE = 0.6802443266`,
  `fRMSE_low = 1.6124732494`,
  `fRMSE_mid = 0.0759288296`,
  `fRMSE_high = 0.1210141182`,
  `parameter_count = 1,073,672`,
  `runtime_sec = 4725.5117`,
  `peak_cuda_memory_bytes = 3,410,884,096`
- `history_len=3`:
  `err_nRMSE = relative_l2 = 0.0230038911`,
  `err_RMSE = 0.5554690361`,
  `fRMSE_low = 1.3145936728`,
  `fRMSE_mid = 0.0578523874`,
  `fRMSE_high = 0.1176725253`,
  `parameter_count = 1,073,928`,
  `runtime_sec = 4850.0043`,
  `peak_cuda_memory_bytes = 3,412,985,344`
- `history_len=4`:
  `err_nRMSE = relative_l2 = 0.0193971880`,
  `err_RMSE = 0.4684740603`,
  `fRMSE_low = 1.1064407825`,
  `fRMSE_mid = 0.0476019271`,
  `fRMSE_high = 0.1138561592`,
  `parameter_count = 1,074,184`,
  `runtime_sec = 4970.4401`,
  `peak_cuda_memory_bytes = 3,415,086,592`
- `history_len=5`:
  `err_nRMSE = relative_l2 = 0.0197584499`,
  `err_RMSE = 0.4772877395`,
  `fRMSE_low = 1.1303862333`,
  `fRMSE_mid = 0.0420697667`,
  `fRMSE_high = 0.1018067747`,
  `parameter_count = 1,074,440`,
  `runtime_sec = 5103.2501`,
  `peak_cuda_memory_bytes = 3,417,187,840`

Directional answer:

- `history_len=3` improved every recorded metric versus the frozen
  `history_len=2` authored-FFNO anchor:
  - `err_nRMSE -0.0051438399`
  - `err_RMSE -0.1247752905`
  - `relative_l2 -0.0051438399`
  - `fRMSE_low -0.2978795767`
  - `fRMSE_mid -0.0180764422`
  - `fRMSE_high -0.0033416029`
  - the `history4` gate opened on this row.
- `history_len=4` improved every recorded metric versus the freshest
  prior authored-FFNO `history_len=3` row:
  - `err_nRMSE -0.0036067031`
  - `err_RMSE -0.0869949758`
  - `relative_l2 -0.0036067031`
  - `fRMSE_low -0.2081528902`
  - `fRMSE_mid -0.0102504604`
  - `fRMSE_high -0.0038163662`
  - the `history5` gate opened on this row.
- `history_len=5` did not strictly continue the aggregate-error trend
  versus the freshest prior `history_len=4` row:
  - `err_nRMSE +0.0003612619`
  - `err_RMSE +0.0088136792`
  - `relative_l2 +0.0003612619`
  - `fRMSE_low +0.0239454508`
  - `fRMSE_mid -0.0055321604`
  - `fRMSE_high -0.0120493844`
  - so `history_len=5` traded a small aggregate-error and `fRMSE_low`
    regression for further `fRMSE_mid` and `fRMSE_high` gains.

## Interpretation

This backlog item answers the within-model authored-FFNO temporal-context
question on the fixed capped CNS contract:

- longer temporal context cleanly improved authored FFNO at the
  `40`-epoch budget through `history_len=4`. Every aggregate and
  per-band metric improved versus the frozen `history_len=2` headline
  anchor and versus the fresh `history_len=3` row.
- `history_len=5` slightly worsened aggregate error and `fRMSE_low`
  versus `history_len=4` but improved `fRMSE_mid` and `fRMSE_high`.
- the best authored-FFNO row by aggregate error on this slice is
  `history_len=4`; the best authored-FFNO row by `fRMSE_high` is
  `history_len=5`.
- raw eligible windows shrank with longer history
  (`history_len=2/3/4/5` raw windows-per-trajectory =
  `19 / 18 / 17 / 16`), but the emitted capped split counts stayed
  fixed at `4096 / 512 / 512`, so longer-history rows are not training
  on a smaller capped budget.

The bounded takeaway is therefore:

- for the authored FFNO row only, longer temporal context through
  `history_len=4` is unambiguously beneficial on this capped slice at
  `40` epochs. `history_len=5` is a frequency-band trade rather than a
  free improvement.
- this evidence remains `adjacent_capped_context_only` and does not
  reopen the locked `history_len=2` CNS paper lane. Any later headline
  authority change must come from a roadmap-level decision rather than
  this within-model temporal-context ablation.

## Verification

Required deterministic checks (run from `/home/ollie/Documents/PtychoPINN`,
logs archived under
`.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-01-cns-author-ffno-history-length-study/verification/`
and copied into the authoritative backlog-item artifact root at
`.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/verification/`):

```bash
pytest -q tests/studies/test_pdebench_image128_models.py -k 'author_ffno'
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Required contract proofs:

- `history_len=3` inspect root showed
  `sample_contract = concat u[t-3:t] -> u[t]`,
  emitted window counts `4096 / 512 / 512`, and unchanged
  trajectory caps and field order.
- `history_len=4` inspect root showed
  `sample_contract = concat u[t-4:t] -> u[t]`,
  emitted window counts `4096 / 512 / 512`, and unchanged
  trajectory caps and field order.
- `history_len=5` inspect root showed
  `sample_contract = concat u[t-5:t] -> u[t]`,
  emitted window counts `4096 / 512 / 512`, and unchanged
  trajectory caps and field order.

Artifact validation performed in this pass:

- `verification/history2_anchor_artifact_check.json` records the
  blocking Task 1 frozen-anchor check under the authoritative backlog-
  item artifact root and confirms the reused `history_len=2` authored-
  FFNO anchor still contains `invocation.json`, `dataset_manifest.json`,
  `split_manifest.json`, `comparison_summary.json`,
  `model_profile_author_ffno_cns_base.json`, and
  `metrics_author_ffno_cns_base.json`.
- the three fresh `40`-epoch pilot run roots each contain
  `invocation.json`, `dataset_manifest.json`, `split_manifest.json`,
  `normalization_stats_state.json`, `comparison_summary.json`,
  `model_profile_author_ffno_cns_base.json`, and
  `metrics_author_ffno_cns_base.json`.
- each tracked launch dir recorded `exit_code = 0`.
- `history4_gate_decision.json` was written before any `history_len=4`
  launch and `history5_gate_decision.json` was written before any
  `history_len=5` launch.
- the three compare sidecars are present, all carry the
  `delta_kind = history_len_only` allowed-delta declaration and a
  `claim_scope = adjacent_capped_context_only` claim-boundary label,
  the `history_len=3` sidecar now records explicit per-profile metric
  deltas against the frozen `history_len=2` anchor, and all JSON / CSV
  pairs now surface runtime plus `peak_cuda_memory_bytes` for the fresh
  and reference rows.

The locked headline `history_len=2` authored-FFNO row reused by the
current CNS paper bundle was not edited or replaced. Manuscript-facing
authorities (`pdebench_cns_paper_2048cap_extension_summary.md`,
`pdebench_cns_paper_table_figure_bundle_summary.md`,
`paper_evidence_index.md`, `evidence_matrix.md`) continue to point at
the `history_len=2` capped row family for the headline CNS table.
