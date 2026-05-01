# PDEBench CNS Spectral History Length 4+ Compare Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-cns-spectral-history-len4plus-compare`
- Date: `2026-05-01`
- Status: implementation complete; frozen spectral anchors, `history_len=4` and `history_len=5` inspect proofs, fresh spectral-only `10`/`40`-epoch pilots, multi-anchor compare sidecars, and the explicit `history5` gate decision are all recorded
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-spectral-history-len4plus-compare/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/`

This summary records capped decision-support evidence only. It does not reopen
the locked CNS paper lane, does not create `/home/ollie/Documents/neurips/`
outputs, and does not relabel any longer-history result as the headline paper
contract.

## Fixed Compare Contract

The fresh longer-context rows and frozen references kept the capped local CNS
surface fixed everywhere except temporal context:

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
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`
- profile and manuscript mapping:
  repo row `spectral_resnet_bottleneck_base` == manuscript label `SRU-Net*`
- claim scope:
  `adjacent_capped_context_only`

Derived history contracts recorded in this pass:

- `history_len=2`:
  `input_channels=8`, raw `windows_per_trajectory=19`,
  raw `available_windows=190000`
- `history_len=3`:
  `input_channels=12`, raw `windows_per_trajectory=18`,
  raw `available_windows=180000`
- `history_len=4`:
  `input_channels=16`, raw `windows_per_trajectory=17`,
  raw `available_windows=170000`
- `history_len=5`:
  `input_channels=20`, raw `windows_per_trajectory=16`,
  raw `available_windows=160000`

The emitted capped split counts stayed fixed at `4096 / 512 / 512` for every
lane, so the longer-history claims remain row-local comparisons on one bounded
contract family rather than a new paper bundle.

## Recorded Artifacts

Frozen spectral-anchor manifest:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history2_history3_spectral_reference_manifest.json`

Inspect proofs:

- `history_len=4`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history4-inspect-20260501T093023Z`
- `history_len=5`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history5-inspect-20260501T093023Z`

Fresh spectral-only pilot runs:

- `history_len=4`, `10ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history4-pilot-10ep-20260501T093239Z`
- `history_len=4`, `40ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history4-pilot-40ep-20260501T094209Z`
- `history_len=5`, `10ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history5-pilot-10ep-20260501T100435Z`
- `history_len=5`, `40ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history5-pilot-40ep-20260501T101147Z`

Tracked run-completion proof:

- `history_len=4`, `10ep` exit code:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/launch-history4-pilot-10ep-20260501T093239Z/exit_code = 0`
- `history_len=4`, `40ep` exit code:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/launch-history4-pilot-40ep-20260501T094209Z/exit_code = 0`
- `history_len=5`, `10ep` exit code:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/launch-history5-pilot-10ep-20260501T100435Z/exit_code = 0`
- `history_len=5`, `40ep` exit code:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/launch-history5-pilot-40ep-20260501T101147Z/exit_code = 0`

Gate record:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history5_gate_decision.json`

Cross-history compare sidecars:

- `history_len=4`, `10ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/compare_10ep_history4_against_history2_history3.json`
  and `.csv`
- `history_len=4`, `40ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/compare_40ep_history4_against_history2_history3.json`
  and `.csv`
- `history_len=5`, `10ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/compare_10ep_history5_against_history2_history3_history4.json`
  and `.csv`
- `history_len=5`, `40ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/compare_40ep_history5_against_history2_history3_history4.json`
  and `.csv`

## Results

### `10` Epochs

Spectral row metrics by history length:

- `history_len=2`:
  `err_nRMSE=0.0869938582`, `err_RMSE=2.1023747921`,
  `fRMSE_high=0.6955373287`, `runtime_sec=426.6635`
- `history_len=3`:
  `err_nRMSE=0.1407901347`, `err_RMSE=3.3996233940`,
  `fRMSE_high=0.6477258205`, `runtime_sec=324.2352`
- `history_len=4`:
  `err_nRMSE=0.0640729368`, `err_RMSE=1.5474669933`,
  `fRMSE_high=0.7701910734`, `runtime_sec=352.7080`
- `history_len=5`:
  `err_nRMSE=0.0939789638`, `err_RMSE=2.2701683044`,
  `fRMSE_high=0.5622383952`, `runtime_sec=383.8927`

Directional answer:

- `history_len=4` improved aggregate error versus both frozen anchors at
  `10` epochs:
  - versus `history_len=2`:
    `err_nRMSE -0.0229209214`, `err_RMSE -0.5549077988`
  - versus `history_len=3`:
    `err_nRMSE -0.0767171979`, `err_RMSE -1.8521564007`
- the first regressed metric for `history_len=4` at `10` epochs was
  `fRMSE_high`:
  - versus `history_len=2`: `+0.0746537447`
  - versus `history_len=3`: `+0.1224652529`
- `history_len=5` did not continue the short-budget aggregate improvement:
  - versus `history_len=4`:
    `err_nRMSE +0.0299060270`, `err_RMSE +0.7227013111`
  - versus `history_len=2`:
    `err_nRMSE +0.0069851056`, `err_RMSE +0.1677935123`
- `history_len=5` still improved the high-band diagnostic at `10` epochs:
  - versus `history_len=4`: `fRMSE_high -0.2079526782`
  - versus `history_len=2`: `fRMSE_high -0.1332989335`

### `40` Epochs

Spectral row metrics by history length:

- `history_len=2`:
  `err_nRMSE=0.0615620054`, `err_RMSE=1.4877649546`,
  `fRMSE_high=0.4349334538`, `runtime_sec=1861.6252`
- `history_len=3`:
  `err_nRMSE=0.0455205254`, `err_RMSE=1.0991724730`,
  `fRMSE_high=0.3467437923`, `runtime_sec=1207.1542`
- `history_len=4`:
  `err_nRMSE=0.0343294032`, `err_RMSE=0.8291117549`,
  `fRMSE_high=0.2862498760`, `runtime_sec=1312.4204`
- `history_len=5`:
  `err_nRMSE=0.0330694355`, `err_RMSE=0.7988296747`,
  `fRMSE_high=0.2622178197`, `runtime_sec=1436.7038`

Directional answer:

- `history_len=4` improved cleanly versus the fresh `history_len=3` anchor:
  - `err_nRMSE 0.0455205254 -> 0.0343294032`
  - `err_RMSE 1.0991724730 -> 0.8291117549`
  - `fRMSE_high 0.3467437923 -> 0.2862498760`
- `history_len=5` continued that `40`-epoch improvement trend on the headline
  gate metrics:
  - versus `history_len=4`:
    `err_nRMSE -0.0012599677`, `err_RMSE -0.0302820802`,
    `fRMSE_high -0.0240320563`
  - versus `history_len=3`:
    `err_nRMSE -0.0124510899`, `err_RMSE -0.3003427982`,
    `fRMSE_high -0.0845259726`
- the first regressed metric for `history_len=5` at `40` epochs versus
  `history_len=4` was `fRMSE_mid`:
  `0.1678307652 -> 0.1713495851` (`+0.0035188198`)

## Interpretation

This backlog item answers the reopened spectral-only question narrowly:

- on the fixed capped CNS contract, longer temporal context keeps helping the
  spectral row once the run is trained to `40` epochs
- `history_len=4` was a clear improvement over the frozen `history_len=2` and
  fresh `history_len=3` anchors at `40` epochs
- `history_len=5` slightly improved the `40`-epoch spectral row again on
  `err_nRMSE`, `err_RMSE`, and `fRMSE_high`, but it also introduced a small
  `fRMSE_mid` regression and remained slower than the `history_len=4` row
- the short-budget story is not monotone:
  `history_len=4` improved aggregate error at `10` epochs but worsened
  `fRMSE_high`, and `history_len=5` then regressed aggregate error versus
  `history_len=4`

The bounded takeaway is therefore:

- for the spectral row only, longer context through `history_len=5` appears
  worthwhile on this capped slice once the row is trained long enough
- the benefit is not stable at `10` epochs and is not clean across every
  frequency band even at `40` epochs
- this remains `adjacent capped context only` and does not change the locked
  `history_len=2` CNS paper lane

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode inspect --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history4-inspect-20260501T093023Z --history-len 4 --max-train-trajectories 512 --max-val-trajectories 64 --max-test-trajectories 64 --max-windows-per-trajectory 8
python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode inspect --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/history5-inspect-20260501T093023Z --history-len 5 --max-train-trajectories 512 --max-val-trajectories 64 --max-test-trajectories 64 --max-windows-per-trajectory 8
```

Artifact validation performed in this pass:

- required inspect artifacts confirmed for `history_len=4` and `history_len=5`
- fresh pilot roots confirmed for all four runs with `invocation.json`,
  `dataset_manifest.json`, `split_manifest.json`,
  `model_profile_spectral_resnet_bottleneck_base.json`,
  `metrics_spectral_resnet_bottleneck_base.json`, and tracked `exit_code=0`
- `history5_gate_decision.json` recorded before any `history_len=5` launch
- all four compare sidecars confirmed present and labeled with
  `claim_scope=adjacent_capped_context_only`

`paper_evidence_index.md` now includes this backlog item as a
`decision_support` CNS ablation row so manuscript-facing navigation can find
the completed longer-history follow-up without implying any change to the
locked `history_len=2` paper contract.
