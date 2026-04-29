# CNS Same-Contract Audit

## Purpose

Normalize the existing PDEBench `2d_cfd_cns` evidence into explicit contract
lanes before selecting a paper-facing CNS contract.

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
- `max_windows_per_trajectory=8`
- history contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
- windows: `4096 / 512 / 512`
- loss: `mse`
- batch size: `4`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`

Required same-contract rows already completed at `40` epochs:

| Row | `relative_l2` | `fRMSE_low` | `fRMSE_mid` | `fRMSE_high` | Params | Runtime sec | Run root |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `spectral_resnet_bottleneck_base` | `0.0615620054` | `3.4756414890` | `0.2800448835` | `0.4349334538` | `8186726` | `1861.6252` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep` |
| `hybrid_resnet_cns` | `0.0644183308` | `3.6567487717` | `0.2804315388` | `0.3683068156` | `7793509` | `886.2512` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z` |
| `fno_base` | `0.0740992129` | `4.1671009064` | `0.2390728593` | `0.6717720628` | not emitted in summary row | `1137.9494` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse` |
| `unet_strong` | `0.6757976413` | `38.9795150757` | `0.5071899891` | `1.3326253891` | not emitted in summary row | `1366.6472` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse` |
| `author_ffno_cns_base` | `0.0281477310` | `1.6124732494` | `0.0759288296` | `0.1210141182` | `1073672` | `4725.5117` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z` |

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

- same dataset, split counts, batch size, loss, and metric family as Lane A
- only allowed contract delta:
  `history_len=3`, `concat u[t-3:t] -> u[t]`
- comparison standard from the emitted sidecar:
  `Only history_len and its derived sample/input-channel contract may differ.`

Completed local rows at `40` epochs:

| Row | `relative_l2` | `fRMSE_low` | `fRMSE_mid` | `fRMSE_high` | Params | Runtime sec | Run root |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `spectral_resnet_bottleneck_base` | `0.0455205254` | `2.5599651337` | `0.2156804800` | `0.3467437923` | `8392966` | `1207.1542` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z` |
| `hybrid_resnet_cns` | `0.0538428985` | `3.0551939011` | `0.2016548216` | `0.3200356364` | emitted in sidecar | emitted in sidecar | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z` |
| `fno_base` | `0.0567254014` | `3.2984776497` | `0.2230665535` | `0.6104770303` | emitted in sidecar | emitted in sidecar | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z` |
| `unet_strong` | `0.6771671176` | `39.1342582703` | `0.4880806208` | `1.1730086803` | emitted in sidecar | emitted in sidecar | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z` |

Directional read:

- This lane is promising local decision-support evidence.
- It is not complete for the paper table because there is no authored FFNO row
  under the same `history_len=3` contract.
- The `10`-epoch and `40`-epoch spectral signals disagree strongly enough that
  the optional `history_len=4` gate stayed closed.

## Contract-Divergent Context Rows

These rows are informative but not headline-eligible under one fixed CNS paper
table without widening the contract:

- GNOT rows:
  - different host environment: `ptycho311_2`
  - different paper-default recipe: `relative_l2`, `AdamW`, `OneCycleLR`
- `history_len=1` Markov rows:
  - coherent local lane, but different temporal contract
- repo-local FFNO proxy rows:
  - `ffno_bottleneck_base`
  - `ffno_bottleneck_localconv_base`
  - useful for architecture context, not authored-FFNO substitution

## Audit Conclusion

Recommended paper-headline contract:

- `bounded_capped_decision_support`
- freeze Lane A (`history_len=2`, `40` epochs, `512 / 64 / 64`, `8` windows)
- use `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`, and
  `author_ffno_cns_base` as the locked headline rows
- keep `hybrid_resnet_cns` as an audited continuity/support row
- treat Lane B (`history_len=3`) as adjacent capped context only

## Comparison Standards Recorded

- lane-selection standard:
  same dataset, same split counts, same `history_len`, same loss, same metric
  family, same capped/full-training status
- cross-history gallery alignment standard from the emitted sidecars:
  `np.allclose(..., atol=1e-6, rtol=1e-6)`
