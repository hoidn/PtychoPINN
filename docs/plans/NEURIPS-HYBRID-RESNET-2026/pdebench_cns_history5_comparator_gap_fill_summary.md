# PDEBench CNS History-Len-5 Comparator Gap Fill Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-05-04-cns-history5-comparator-gap-fill`
- Date: `2026-05-04`
- Status: implementation complete; frozen `history_len=2` and completed
  `history_len=5` reference manifest, `history_len=5` inspect proof for
  `fno_base` and `unet_strong`, fresh `40`-epoch capped pilot for both
  missing profiles, history-delta compare against the frozen same-profile
  `history_len=2` anchors, same-history cross-run compare against the
  completed authored-FFNO and spectral `history_len=5` authorities, and
  durable index synchronization are all recorded
- Governing plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-history5-comparator-gap-fill/execution_plan.md`
- Artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/`
- Claim scope: `adjacent_capped_context_only`

This summary records `adjacent_capped_context_only` decision-support
evidence for the all-`history_len=5` CNS comparator question on the fixed
capped contract. It does not reopen the locked CNS paper lane, does not
modify the active `history_len=2`, `2048 / 256 / 256` capped paper
authority, and does not create `/home/ollie/Documents/neurips/` outputs.

## Fixed Compare Contract

The fresh `history_len=5` comparator rows reuse the same capped local CNS
contract already used by the completed authored-FFNO and spectral
`history_len=5` authorities:

- task: `2d_cfd_cns`
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
- profile and manuscript labels:
  `fno_base` is manuscript label `FNO`,
  `unet_strong` is manuscript label `U-Net`,
  `author_ffno_cns_base` is manuscript label `Author FFNO`,
  `spectral_resnet_bottleneck_base` is manuscript label `SRU-Net*`

Only the `history_len` axis differs in the history-delta compare:

- frozen `history_len=2` reference: `input_channels=8`,
  `concat u[t-2:t] -> u[t]`,
  raw `windows_per_trajectory=19`, raw `available_windows=190000`
- fresh `history_len=5` row: `input_channels=20`,
  `concat u[t-5:t] -> u[t]`,
  raw `windows_per_trajectory=16`, raw `available_windows=160000`

Emitted capped split counts stayed fixed at `4096 / 512 / 512` for every
lane, so the `history_len=5` rows are not training on a smaller capped
budget.

## Reference Roots Consumed

Frozen `history_len=2`, `40`-epoch capped reference root for both
missing profiles (locked capped CNS paper lane):

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

Completed `history_len=5`, `40`-epoch capped authority roots for the
same-history cross-run compare:

- `author_ffno_cns_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/history5-pilot-40ep-20260502T074500Z`
- `spectral_resnet_bottleneck_base`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history5-pilot-40ep-20260501T101147Z`

## Recorded Artifacts

Frozen reference manifest:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/history2_history5_reference_runs.json`

Inspect proof for the missing profiles under the target contract:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/history5-inspect-20260504T214431Z`

Fresh `history_len=5`, `40`-epoch capped run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/history5-gap-fill-40ep-20260504T214614Z`

Tracked launch-completion proof (`exit_code = 0`, recorded `pid`,
captured `pane.log`, `run.sh`):

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/launch-history5-gap-fill-40ep-20260504T214614Z`

Compare sidecars:

- history-delta compare against frozen same-profile `history_len=2`
  anchors:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/compare_40ep_history5_against_history2.json`
  and `.csv`
- same-history cross-run compare against completed `history_len=5`
  authorities:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/compare_40ep_against_existing.json`
  and `.csv`

Same-history cross-run gallery (`history_len=5` only):

- `compare_40ep_sample0.png`
- `compare_40ep_sample0_error.png`

The history-delta compare did not produce a paired sample gallery
because the `history_len=2` frozen run and the fresh `history_len=5` run
emit different sample0 windows. Per plan, missing PNG/NPZ comparison
galleries are supporting-only packaging and do not affect completion.

## Results

Fresh `history_len=5`, `40`-epoch capped row metrics:

- `fno_base` (manuscript label `FNO`):
  `err_nRMSE = relative_l2 = 0.0384249315`,
  `err_RMSE = 0.9281977415`,
  `fRMSE_low = 2.1336586475`,
  `fRMSE_mid = 0.1223827526`,
  `fRMSE_high = 0.4328559041`,
  `parameter_count = 358,628`,
  `runtime_sec = 1129.7886884`,
  `peak_cuda_memory_bytes = 285,889,024`
- `unet_strong` (manuscript label `U-Net`):
  `err_nRMSE = relative_l2 = 0.5386230350`,
  `err_RMSE = 13.0110502243`,
  `fRMSE_low = 30.9882907867`,
  `fRMSE_mid = 0.6448712349`,
  `fRMSE_high = 1.7427886724`,
  `parameter_count = 7,768,036`,
  `runtime_sec = 1070.9980743`,
  `peak_cuda_memory_bytes = 318,223,872`

Frozen `history_len=2`, `40`-epoch capped same-profile reference rows
(unchanged):

- `fno_base`:
  `err_nRMSE = 0.0740992129`,
  `err_RMSE = 1.7907506227`,
  `fRMSE_low = 4.1671009064`,
  `fRMSE_mid = 0.2390728593`,
  `fRMSE_high = 0.6717720628`,
  `parameter_count = 357,860`,
  `runtime_sec = 1137.9493690`
- `unet_strong`:
  `err_nRMSE = 0.6757976413`,
  `err_RMSE = 16.3319549561`,
  `fRMSE_low = 38.9795150757`,
  `fRMSE_mid = 0.5071899891`,
  `fRMSE_high = 1.3326253891`,
  `parameter_count = 7,764,580`,
  `runtime_sec = 1366.6471713`

History-delta direction (`history_len=5` minus `history_len=2`):

- `fno_base`: every recorded metric improves at `history_len=5`:
  - `err_nRMSE -0.0356742814`
  - `err_RMSE -0.8625528812`
  - `relative_l2 -0.0356742814`
  - `fRMSE_low -2.0334422588`
  - `fRMSE_mid -0.1166901067`
  - `fRMSE_high -0.2389161587`
- `unet_strong`: aggregate-error and `fRMSE_low` improve, `fRMSE_mid`
  and `fRMSE_high` regress:
  - `err_nRMSE -0.1371746063`
  - `err_RMSE -3.3209047318`
  - `relative_l2 -0.1371746063`
  - `fRMSE_low -7.9912242889`
  - `fRMSE_mid +0.1376812458`
  - `fRMSE_high +0.4101632833`

Same-history cross-run ranking on the completed `history_len=5`,
`40`-epoch capped slice:

- by `err_nRMSE` (best first):
  `author_ffno_cns_base` (`0.0197584`),
  `spectral_resnet_bottleneck_base` (`0.0330694`),
  `fno_base` (`0.0384249`),
  `unet_strong` (`0.5386230`)
- by `fRMSE_high` (best first):
  `author_ffno_cns_base` (`0.1018068`),
  `spectral_resnet_bottleneck_base` (`0.2622178`),
  `fno_base` (`0.4328559`),
  `unet_strong` (`1.7427887`)

## Interpretation

This backlog item closes the missing all-`history_len=5` CNS comparator
gap on the fixed capped local CNS contract:

- `fno_base` is unambiguously better at `history_len=5` than at the
  frozen `history_len=2` capped anchor, with every aggregate and
  per-band metric improving by a substantial margin.
- `unet_strong` is mixed at `history_len=5` versus the frozen
  `history_len=2` capped anchor: aggregate error and low-frequency
  band improve, but the mid- and high-frequency bands regress. On
  this capped slice, `unet_strong` remains last by aggregate error and
  by `fRMSE_high`.
- with both rows now available, the same-history cross-run ranking on
  this capped slice is `author_ffno_cns_base` first, then
  `spectral_resnet_bottleneck_base`, then `fno_base`, then
  `unet_strong`. The all-`history_len=5` capped comparator surface no
  longer carries the missing-FNO/missing-U-Net gap.

The bounded takeaway is therefore:

- the all-`history_len=5` CNS comparator gap is closed for the fixed
  capped `512 / 64 / 64`, `40`-epoch contract.
- the freshly produced rows are `adjacent_capped_context_only`
  decision-support evidence and explicitly do not reopen the locked
  `history_len=2`, `2048 / 256 / 256` capped CNS paper authority
  recorded in
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md`
  and
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`.
- any later headline authority change for the CNS paper bundle must
  come from a roadmap-level decision, not from this row-local
  history-length comparator.

## Verification

Required deterministic checks (run from `/home/ollie/Documents/PtychoPINN`,
logs archived under
`.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/verification/`
and copied into
`.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-history5-comparator-gap-fill/verification/`):

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Both checks were rerun in this implementation pass and pass cleanly
(`107 passed in 67.60s`, `compileall` silent success). Targeted
follow-up:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'reference_run_manifest or cross_run_compare or history' -v
```

passed `15 / 15`.

Required contract proofs:

- `history5-inspect-20260504T214431Z` showed
  `sample_contract = concat u[t-5:t] -> u[t]`,
  `field_order = [density, Vx, Vy, pressure]`,
  `data_file = 2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`,
  `split_counts = {train: 512, val: 64, test: 64}`,
  `window_counts = {train: 4096, val: 512, test: 512}`,
  `max_windows_per_trajectory = 8`, and `history_len = 5` for both
  `fno_base` and `unet_strong`.
- the fresh `40`-epoch run root contains `invocation.json`,
  `dataset_manifest.json`, `split_manifest.json`,
  `normalization_stats_state.json`, `comparison_summary.json`,
  `model_profile_fno_base.json`, `model_profile_unet_strong.json`,
  `metrics_fno_base.json`, and `metrics_unet_strong.json`.
- the tracked launch dir recorded `exit_code = 0`, captured `pid`,
  `run.sh`, and `pane.log`.
- the history-delta compare carries
  `delta_kind = history_len_only`,
  `reference_history_len = 2`, `fresh_history_len = 5`,
  `reference_input_channels = 8`, `fresh_input_channels = 20`,
  `target_channels = 4`, and `evidence_scope =
  capped_decision_support_only`.
- the same-history cross-run compare carries
  `contract.history_len = 5`, fresh profiles `fno_base` and
  `unet_strong`, and required reference rows `author_ffno_cns_base`
  and `spectral_resnet_bottleneck_base`.

The locked headline `history_len=2` capped CNS paper-authority bundle
was not edited or replaced. Manuscript-facing authorities
(`pdebench_cns_paper_contract_decision.md`,
`pdebench_cns_paper_2048cap_extension_summary.md`,
`paper_evidence_index.md`, `evidence_matrix.md`) continue to point at
the `history_len=2` capped row family for the headline CNS table.
