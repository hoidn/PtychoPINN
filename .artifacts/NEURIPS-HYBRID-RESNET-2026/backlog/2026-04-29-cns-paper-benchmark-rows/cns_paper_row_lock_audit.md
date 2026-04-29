# CNS Paper Row Lock Audit

## Contract Authority

- Authority doc: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- Selected contract: `bounded_capped_decision_support`
- Locked history lane: `history_len=2`, `40` epochs, `512 / 64 / 64` trajectories, `max_windows_per_trajectory=8`, emitted windows `4096 / 512 / 512`
- Metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`

## Accepted Rows
| Row | Role | Status | Run Mode | relative_l2 | fRMSE_high | Params | Runtime (s) | Contract Parity |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `spectral_resnet_bottleneck_base` | `headline` | `capped_decision_support` | `readiness` | 0.0615620054 | 0.4349334538 | 8186726 | 1861.63 | pass |
| `fno_base` | `headline` | `capped_decision_support` | `readiness` | 0.0740992129 | 0.6717720628 | 357860 | 1137.95 | pass |
| `unet_strong` | `headline` | `capped_decision_support` | `readiness` | 0.6757976413 | 1.3326253891 | 7764580 | 1366.65 | pass |
| `author_ffno_cns_base` | `headline` | `capped_decision_support` | `readiness` | 0.0281477310 | 0.1210141182 | 1073672 | 4725.51 | pass |
| `hybrid_resnet_cns` | `continuity` | `capped_decision_support` | `pilot` | 0.0644183308 | 0.3683068156 | 7998597 | 886.25 | pass |

## Row Findings
### `spectral_resnet_bottleneck_base`
- Outcome: `capped_decision_support`
- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- Dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- Split counts: `512 / 64 / 64` trajectories; emitted windows `4096 / 512 / 512`
- History lane: `history_len=2`, `max_windows_per_trajectory=8`
- Training recipe: `mse`, `Adam`, `lr=2e-4`, `ReduceLROnPlateau`, `batch_size=4`, `epochs=40`
- Metrics: `relative_l2=0.0615620054`, `err_nRMSE=0.0615620054`, `fRMSE_high=0.4349334538`
- Assets: sample NPZ `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep/comparison_spectral_resnet_bottleneck_base_sample0.npz`, sample PNG `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep/comparison_spectral_resnet_bottleneck_base_sample0.png`
- Parity check: all selected-lane contract fields matched the approved history-2 capped contract.
- Provenance gap note: no standalone repo git SHA, dirty-state marker, run log, or exit-code artifact is present in the reused run root; treat these rows as bounded capped decision-support evidence only, not paper-grade provenance-complete rows.

### `fno_base`
- Outcome: `capped_decision_support`
- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- Dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- Split counts: `512 / 64 / 64` trajectories; emitted windows `4096 / 512 / 512`
- History lane: `history_len=2`, `max_windows_per_trajectory=8`
- Training recipe: `mse`, `Adam`, `lr=2e-4`, `ReduceLROnPlateau`, `batch_size=4`, `epochs=40`
- Metrics: `relative_l2=0.0740992129`, `err_nRMSE=0.0740992129`, `fRMSE_high=0.6717720628`
- Assets: sample NPZ `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse/comparison_fno_base_sample0.npz`, sample PNG `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse/comparison_fno_base_sample0.png`
- Parity check: all selected-lane contract fields matched the approved history-2 capped contract.
- Provenance gap note: no standalone repo git SHA, dirty-state marker, run log, or exit-code artifact is present in the reused run root; treat these rows as bounded capped decision-support evidence only, not paper-grade provenance-complete rows.

### `unet_strong`
- Outcome: `capped_decision_support`
- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- Dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- Split counts: `512 / 64 / 64` trajectories; emitted windows `4096 / 512 / 512`
- History lane: `history_len=2`, `max_windows_per_trajectory=8`
- Training recipe: `mse`, `Adam`, `lr=2e-4`, `ReduceLROnPlateau`, `batch_size=4`, `epochs=40`
- Metrics: `relative_l2=0.6757976413`, `err_nRMSE=0.6757976413`, `fRMSE_high=1.3326253891`
- Assets: sample NPZ `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse/comparison_unet_strong_sample0.npz`, sample PNG `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse/comparison_unet_strong_sample0.png`
- Parity check: all selected-lane contract fields matched the approved history-2 capped contract.
- Provenance gap note: no standalone repo git SHA, dirty-state marker, run log, or exit-code artifact is present in the reused run root; treat these rows as bounded capped decision-support evidence only, not paper-grade provenance-complete rows.

### `author_ffno_cns_base`
- Outcome: `capped_decision_support`
- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
- Dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- Split counts: `512 / 64 / 64` trajectories; emitted windows `4096 / 512 / 512`
- History lane: `history_len=2`, `max_windows_per_trajectory=8`
- Training recipe: `mse`, `Adam`, `lr=2e-4`, `ReduceLROnPlateau`, `batch_size=4`, `epochs=40`
- Metrics: `relative_l2=0.0281477310`, `err_nRMSE=0.0281477310`, `fRMSE_high=0.1210141182`
- Assets: sample NPZ `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z/comparison_author_ffno_cns_base_sample0.npz`, sample PNG `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z/comparison_author_ffno_cns_base_sample0.png`
- Parity check: all selected-lane contract fields matched the approved history-2 capped contract.
- Provenance gap note: no standalone repo git SHA, dirty-state marker, run log, or exit-code artifact is present in the reused run root; treat these rows as bounded capped decision-support evidence only, not paper-grade provenance-complete rows.

### `hybrid_resnet_cns`
- Outcome: `capped_decision_support`
- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
- Dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- Split counts: `512 / 64 / 64` trajectories; emitted windows `4096 / 512 / 512`
- History lane: `history_len=2`, `max_windows_per_trajectory=8`
- Training recipe: `mse`, `Adam`, `lr=2e-4`, `ReduceLROnPlateau`, `batch_size=4`, `epochs=40`
- Metrics: `relative_l2=0.0644183308`, `err_nRMSE=0.0644183308`, `fRMSE_high=0.3683068156`
- Assets: sample NPZ `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z/comparison_hybrid_resnet_cns_sample0.npz`, sample PNG `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z/comparison_hybrid_resnet_cns_sample0.png`
- Parity check: all selected-lane contract fields matched the approved history-2 capped contract.
- Provenance gap note: no standalone repo git SHA, dirty-state marker, run log, or exit-code artifact is present in the reused run root; treat these rows as bounded capped decision-support evidence only, not paper-grade provenance-complete rows.

## Excluded Adjacent Context
- `history_len=3 pilots`: history_len diverges from the selected history_len=2 headline lane and the authored FFNO row was not completed under that alternate temporal contract by the cutoff
- `history_len=1 pilots`: lower-context Markov ablation is contract-divergent temporal context only, not part of the locked headline table
- `gnot`: protocol-divergent environment and recipe lane; not part of the required same-contract headline roster
- `ffno_bottleneck_base`: repo-local FFNO proxy row is explicitly not an authored FFNO substitute for the locked paper row bundle
- `ffno_bottleneck_localconv_base`: repo-local FFNO proxy with local branch is explicitly adjacent context only and cannot replace authored FFNO in the locked roster

## Row-Lock Read
- Accepted headline rows: `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`, `author_ffno_cns_base`.
- Accepted continuity row: `hybrid_resnet_cns`.
- All accepted rows remain reusable for the selected bounded capped contract in this pass; no code patch or rerun was required to complete the lock manifest.
- The resulting lock is intentionally limited: it is suitable for downstream table/figure assembly and contract-preserving comparison, but not for relabeling the reused rows as `paper_grade` or `full_training`.
