# CNS Same-Contract Audit

## Purpose

Normalize the existing PDEBench `2d_cfd_cns` evidence into explicit contract
lanes before selecting a paper-facing CNS contract. This audit now carries the
required per-row provenance contract instead of leaving key fields only at the
lane level.

## Deterministic Check Gate

- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  - result: `47 passed in 53.21s`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/verification/pytest_cns_contract_20260429.log`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  - result: exit code `0`
  - log:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/verification/compileall_cns_contract_20260429.log`

## Lane A: Coherent Same-Contract `history_len=2` Headline Lane

Fixed contract:

- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split counts: `512 / 64 / 64` trajectories
- window counts: `4096 / 512 / 512`
- history contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
- `max_windows_per_trajectory=8`
- normalization contract:
  train-only per-field normalization fit on the training trajectories, reused
  across history slots and target channels; evaluation reported in
  denormalized target space
- training loss: `mse`
- optimizer/scheduler contract:
  `Adam(lr=2e-4)` with `ReduceLROnPlateau(factor=0.5, patience=2,
  threshold=0.0, min_lr=1e-5)`
- batch size: `4`
- epoch budget: `40`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`

### `spectral_resnet_bottleneck_base`

- evidence status: `same_contract_reusable_headline`
- headline role: `headline_required`
- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- environment:
  `conda_env=ptycho311`, `python_executable=/home/ollie/miniconda3/envs/ptycho311/bin/python3.11`,
  `pythonpath=""`, `device=cuda`
- dataset path:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split counts: `512 / 64 / 64`
- window counts: `4096 / 512 / 512`
- history/sample contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
- `max_windows_per_trajectory=8`
- model profile:
  `spectral_resnet_bottleneck_base` / `PadCropWrapper` /
  `spectral_resnet_bottleneck_net` /
  `profile_evidence_scope=benchmark_candidate`
- training loss: `mse`
- optimizer/scheduler:
  `Adam(lr=2e-4)` / `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- epoch budget: `40`
- metrics:
  `err_RMSE=1.4877649546`, `err_nRMSE=0.0615620054`,
  `relative_l2=0.0615620054`, `fRMSE_low=3.4756414890`,
  `fRMSE_mid=0.2800448835`, `fRMSE_high=0.4349334538`
- parameter count: `8,186,726`
- runtime: `1861.6252s`

### `hybrid_resnet_cns`

- evidence status: `same_contract_reusable_continuity`
- headline role: `continuity_support_only`
- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
- environment:
  `conda_env=ptycho311`, `python_executable=/home/ollie/miniconda3/envs/ptycho311/bin/python3.11`,
  `pythonpath=/home/ollie/Documents/agent-orchestration`, `device=cuda`
- dataset path:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split counts: `512 / 64 / 64`
- window counts: `4096 / 512 / 512`
- history/sample contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
- `max_windows_per_trajectory=8`
- model profile:
  `hybrid_resnet_cns` / `PadCropWrapper` / `hybrid_resnet` /
  `profile_evidence_scope=benchmark_candidate`
- training loss: `mse`
- optimizer/scheduler:
  `Adam(lr=2e-4)` / `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- epoch budget: `40`
- metrics:
  `err_RMSE=1.5567935705`, `err_nRMSE=0.0644183308`,
  `relative_l2=0.0644183308`, `fRMSE_low=3.6567487717`,
  `fRMSE_mid=0.2804315388`, `fRMSE_high=0.3683068156`
- parameter count: `7,998,597`
- runtime: `886.2512s`

### `fno_base`

- evidence status: `same_contract_reusable_headline`
- headline role: `headline_required`
- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- environment:
  `conda_env=ptycho311`, `python_executable=/home/ollie/miniconda3/envs/ptycho311/bin/python3.11`,
  `pythonpath=""`, `device=cuda`
- dataset path:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split counts: `512 / 64 / 64`
- window counts: `4096 / 512 / 512`
- history/sample contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
- `max_windows_per_trajectory=8`
- model profile:
  `fno_base` / `FNO` / `fno` / `profile_evidence_scope=benchmark_candidate`
- training loss: `mse`
- optimizer/scheduler:
  `Adam(lr=2e-4)` / `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- epoch budget: `40`
- metrics:
  `err_RMSE=1.7907506227`, `err_nRMSE=0.0740992129`,
  `relative_l2=0.0740992129`, `fRMSE_low=4.1671009064`,
  `fRMSE_mid=0.2390728593`, `fRMSE_high=0.6717720628`
- parameter count: `357,860`
- runtime: `1137.9494s`

### `unet_strong`

- evidence status: `same_contract_reusable_headline`
- headline role: `headline_required`
- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- environment:
  `conda_env=ptycho311`, `python_executable=/home/ollie/miniconda3/envs/ptycho311/bin/python3.11`,
  `pythonpath=""`, `device=cuda`
- dataset path:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split counts: `512 / 64 / 64`
- window counts: `4096 / 512 / 512`
- history/sample contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
- `max_windows_per_trajectory=8`
- model profile:
  `unet_strong` / `PadCropWrapper` / `unet_strong` /
  `profile_evidence_scope=benchmark_candidate`
- training loss: `mse`
- optimizer/scheduler:
  `Adam(lr=2e-4)` / `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- epoch budget: `40`
- metrics:
  `err_RMSE=16.3319549561`, `err_nRMSE=0.6757976413`,
  `relative_l2=0.6757976413`, `fRMSE_low=38.9795150757`,
  `fRMSE_mid=0.5071899891`, `fRMSE_high=1.3326253891`
- parameter count: `7,764,580`
- runtime: `1366.6472s`

### `author_ffno_cns_base`

- evidence status: `same_contract_reusable_headline`
- headline role: `headline_required`
- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
- environment:
  `conda_env=ptycho311`, `python_executable=/home/ollie/miniconda3/envs/ptycho311/bin/python`,
  `python_version=3.11.13`, `torch_version=2.9.1+cu128`,
  `einops_version=0.8.1`, `pythonpath=""`, `device=cuda`
- dataset path:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split counts: `512 / 64 / 64`
- window counts: `4096 / 512 / 512`
- history/sample contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
- `max_windows_per_trajectory=8`
- model profile:
  `author_ffno_cns_base` / `AuthorFfnoCnsModel` / `author_ffno_cns_net` /
  `profile_evidence_scope=readiness-only`
- training loss: `mse`
- optimizer/scheduler:
  `Adam(lr=2e-4)` / `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- epoch budget: `40`
- metrics:
  `err_RMSE=0.6802443266`, `err_nRMSE=0.0281477310`,
  `relative_l2=0.0281477310`, `fRMSE_low=1.6124732494`,
  `fRMSE_mid=0.0759288296`, `fRMSE_high=0.1210141182`
- parameter count: `1,073,672`
- runtime: `4725.5117s`

Directional read:

- This is the only lane that already contains the exact paper-facing row
  families under one contract, including authored FFNO.
- Within the local Hybrid-family rows, `spectral_resnet_bottleneck_base`
  slightly beats `hybrid_resnet_cns` on aggregate denormalized error, while
  `hybrid_resnet_cns` keeps a lower `fRMSE_high`.
- The authored FFNO row is fully available under the same local contract and
  is materially stronger than the local FNO/U-Net rows on the capped slice.

## Lane B: Coherent Same-Contract `history_len=3` Local-Only Lane

Fixed contract delta:

- same dataset, split counts, window counts, batch size, loss, optimizer,
  scheduler, and metric family as Lane A
- only allowed contract delta:
  `history_len=3`, `concat u[t-3:t] -> u[t]`
- comparison standard from the emitted sidecar:
  `Only history_len and its derived sample/input-channel contract may differ.`

### `spectral_resnet_bottleneck_base`

- evidence status: `adjacent_same_contract_context_completed`
- headline role: `headline_required`
- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z`
- environment:
  `conda_env=ptycho311`, `python_executable=/home/ollie/miniconda3/envs/ptycho311/bin/python3.11`,
  `pythonpath=/home/ollie/Documents/agent-orchestration`, `device=cuda`
- split counts: `512 / 64 / 64`
- window counts: `4096 / 512 / 512`
- history/sample contract: `history_len=3`, `concat u[t-3:t] -> u[t]`
- `max_windows_per_trajectory=8`
- model profile:
  `spectral_resnet_bottleneck_base` / `PadCropWrapper` /
  `spectral_resnet_bottleneck_net` / `profile_evidence_scope=manual-only`
- training loss: `mse`
- optimizer/scheduler:
  `Adam(lr=2e-4)` / `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- epoch budget: `40`
- metrics:
  `err_RMSE=1.0991724730`, `err_nRMSE=0.0455205254`,
  `relative_l2=0.0455205254`, `fRMSE_low=2.5599651337`,
  `fRMSE_mid=0.2156804800`, `fRMSE_high=0.3467437923`
- parameter count: `8,392,966`
- runtime: `1207.1542s`

### `hybrid_resnet_cns`

- evidence status: `adjacent_same_contract_context_completed`
- headline role: `continuity_support_only`
- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z`
- environment:
  `conda_env=ptycho311`, `python_executable=/home/ollie/miniconda3/envs/ptycho311/bin/python3.11`,
  `pythonpath=/home/ollie/Documents/agent-orchestration`, `device=cuda`
- split counts: `512 / 64 / 64`
- window counts: `4096 / 512 / 512`
- history/sample contract: `history_len=3`, `concat u[t-3:t] -> u[t]`
- `max_windows_per_trajectory=8`
- model profile:
  `hybrid_resnet_cns` / `PadCropWrapper` / `hybrid_resnet` /
  `profile_evidence_scope=benchmark_candidate`
- training loss: `mse`
- optimizer/scheduler:
  `Adam(lr=2e-4)` / `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- epoch budget: `40`
- metrics:
  `err_RMSE=1.3001306057`, `err_nRMSE=0.0538428985`,
  `relative_l2=0.0538428985`, `fRMSE_low=3.0537090302`,
  `fRMSE_mid=0.2251202166`, `fRMSE_high=0.3200356364`
- parameter count: `7,999,749`
- runtime: `1002.8943s`

### `fno_base`

- evidence status: `adjacent_same_contract_context_completed`
- headline role: `headline_required`
- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z`
- environment:
  `conda_env=ptycho311`, `python_executable=/home/ollie/miniconda3/envs/ptycho311/bin/python3.11`,
  `pythonpath=/home/ollie/Documents/agent-orchestration`, `device=cuda`
- split counts: `512 / 64 / 64`
- window counts: `4096 / 512 / 512`
- history/sample contract: `history_len=3`, `concat u[t-3:t] -> u[t]`
- `max_windows_per_trajectory=8`
- model profile:
  `fno_base` / `FNO` / `fno` / `profile_evidence_scope=benchmark_candidate`
- training loss: `mse`
- optimizer/scheduler:
  `Adam(lr=2e-4)` / `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- epoch budget: `40`
- metrics:
  `err_RMSE=1.3697336912`, `err_nRMSE=0.0567254014`,
  `relative_l2=0.0567254014`, `fRMSE_low=3.1596221924`,
  `fRMSE_mid=0.1725350171`, `fRMSE_high=0.6104770303`
- parameter count: `358,116`
- runtime: `912.3334s`

### `unet_strong`

- evidence status: `adjacent_same_contract_context_completed`
- headline role: `headline_required`
- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z`
- environment:
  `conda_env=ptycho311`, `python_executable=/home/ollie/miniconda3/envs/ptycho311/bin/python3.11`,
  `pythonpath=/home/ollie/Documents/agent-orchestration`, `device=cuda`
- split counts: `512 / 64 / 64`
- window counts: `4096 / 512 / 512`
- history/sample contract: `history_len=3`, `concat u[t-3:t] -> u[t]`
- `max_windows_per_trajectory=8`
- model profile:
  `unet_strong` / `PadCropWrapper` / `unet_strong` /
  `profile_evidence_scope=benchmark_candidate`
- training loss: `mse`
- optimizer/scheduler:
  `Adam(lr=2e-4)` / `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- epoch budget: `40`
- metrics:
  `err_RMSE=16.3513793945`, `err_nRMSE=0.6771671176`,
  `relative_l2=0.6771671176`, `fRMSE_low=39.0328674316`,
  `fRMSE_mid=0.5551229715`, `fRMSE_high=1.1730086803`
- parameter count: `7,765,732`
- runtime: `847.3525s`

### `author_ffno_cns_base`

- evidence status: `blocked_missing_same_contract_authored_ffno`
- headline role: `headline_required_if_lane_selected`
- required same-contract row: authored FFNO under the Lane B contract
  (`history_len=3`, `512 / 64 / 64`, `4096 / 512 / 512`,
  `max_windows_per_trajectory=8`, `mse`, `Adam(lr=2e-4)`,
  `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`,
  `40` epochs, batch size `4`)
- blocker:
  no same-contract authored FFNO row exists for `history_len=3` at the paper
  cutoff
- accepted Lane A author row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
- downstream implication:
  selecting Lane B would require a fresh same-contract authored FFNO run or an
  explicit row-level `blocked` / `not_protocol_compatible` outcome in the
  later row-lock item

Directional read:

- This lane is promising local decision-support evidence.
- It is not complete for the paper table because there is no authored FFNO row
  under the same `history_len=3` contract.
- The `10`-epoch and `40`-epoch spectral signals disagree strongly enough that
  the optional `history_len=4` gate stayed closed.

## Contract-Divergent Context

- `history_len=1` Markov rows:
  contract-divergent temporal context only; authority summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- GNOT rows:
  contract-divergent environment and recipe context only; authority summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`
  because the working row depends on `ptycho311_2`,
  `relative_l2`, `AdamW`, `OneCycleLR`, `lr=1e-3`, and `weight_decay=5e-5`
- repo-local FFNO proxy rows:
  `ffno_bottleneck_base` and `ffno_bottleneck_localconv_base` remain bounded
  local FFNO-family context only and are not authored-FFNO substitutes;
  authority summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`

## Audit Conclusion

Recommended paper-headline contract:

- `bounded_capped_decision_support`
- freeze Lane A (`history_len=2`, `40` epochs, `512 / 64 / 64`, `8` windows)
- use `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`, and
  `author_ffno_cns_base` as the locked headline rows
- keep `hybrid_resnet_cns` as an audited continuity/support row
- treat Lane B (`history_len=3`) as adjacent capped context only because the
  same-contract authored FFNO row is missing

## Comparison Standards Recorded

- headline-lane selection standard:
  same dataset, same split counts, same `history_len`, same loss, same
  optimizer/scheduler recipe, same metric family, same capped/full-training
  status
- cross-history only-delta standard:
  `Only history_len and its derived sample/input-channel contract may differ.`
- cross-history gallery alignment standard:
  `np.allclose(..., atol=1e-6, rtol=1e-6)`
