# NeurIPS Hybrid ResNet Evidence Matrix

Status: draft  
Last updated: 2026-05-09 (added the CDI `lines128` coordinate-grid conditioning ablation rows and refreshed the CDI conditioning discovery surfaces)

This is the human-facing master matrix for NeurIPS/SRU-Net evidence. It points
to all relevant completed backlog outputs, keyed by dataset, model/config
surface, training mode, and ablation family. Detailed metrics and validation
logs remain owned by the linked summary authorities.

Machine-readable companions:

- `model_variant_index.json`
- `ablation_index.json`

## Current Claim Authorities

| Scope | Current authority | Boundary |
|---|---|---|
| CDI `lines128` complete table | `lines128_paper_benchmark_summary.md` | `paper_grade`, six-row complete CDI bundle |
| CDI `lines128` final paper-local FFNO refresh | `cdi_lines128_ffno_depth24_final_paper_refresh_summary.md` | current manuscript-facing CDI tables, figures, model-config, efficiency, and discovery assets; four-block same-depth no-refiner pair retained as canonical; claim boundary `complete_lines128_cdi_benchmark_plus_uno_extension_with_final_four_block_no_refiner_ffno_pair` |
| CDI `lines128` pure-FFNO corrected prerequisite row | `cdi_lines128_ffno_no_refiner_row_rerun_summary.md` | corrected source-row evidence for the active paper-local FFNO refresh; claim boundary `lines128_ffno_vs_hybrid_prerequisite_pair` |
| CDI `lines128` probe-channel conditioning ablation | `cdi_lines128_probe_channel_conditioning_ablation_summary.md` | append-only input-conditioning evidence only; claim boundary `cdi_lines128_probe_channel_conditioning_ablation_only` |
| CDI `lines128` coordinate-grid conditioning ablation | `cdi_lines128_coordinate_grid_conditioning_ablation_summary.md` | append-only input-conditioning evidence only; claim boundary `cdi_lines128_coordinate_grid_conditioning_ablation_only` |
| CDI `lines128` pure-FFNO corrected objective-control pair | `cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md` | corrected source-row evidence for the active paper-local FFNO objective table; claim boundary `lines128_ffno_objective_control_corrected_pair` |
| CDI `lines128` pure-FFNO depth ablation | `cdi_lines128_ffno_depth24_ablation_summary.md` | append-only depth-only FFNO evidence comparing corrected no-refiner `fno_blocks=4` against fresh `fno_blocks=24`; claim boundary `cdi_ffno_depth_ablation_only` |
| CDI `lines128` supervised pure-FFNO depth companion | `cdi_lines128_supervised_ffno_depth24_no_refiner_summary.md` | append-only supervised depth-only FFNO evidence comparing corrected no-refiner `fno_blocks=4` against fresh `fno_blocks=24`; claim boundary `cdi_supervised_ffno_depth_companion_only` |
| CDI U-NO extension readiness | `lines128_uno_preflight_summary.md` | feasibility-only external UNO environment/API authority before generator integration |
| CDI U-NO table extension | `lines128_uno_table_extension_summary.md` | append-only `paper_grade` eight-row extended bundle; claim boundary `complete_lines128_cdi_benchmark_plus_uno_extension` |
| CDI SRU-Net branch / objective ablation | `lines128_srunet_branch_objective_ablation_summary.md` | append-only `decision_support_append_only`; mechanistic encoder branch removal + supervised SRU-Net objective control |
| CDI SRU-Net ConvNeXt-bottleneck ablation | `lines128_srunet_convnext_bottleneck_ablation_summary.md` | append-only `decision_support_append_only`; bottleneck block family swap (ResNet → ConvNeXt-style) on the locked `lines128` CDI contract |
| PDEBench CNS matched-condition table | `pdebench_cns_matched_condition_table_refresh_summary.md` | bounded capped decision-support only, matched `history_len=5`, `512 / 64 / 64`, `40` epochs headline |
| PDEBench CNS checkpoint+rollout lineage refresh | `cns_rollout_checkpoint_refresh_summary.md` | manuscript-adjacent checkpoint/rollout packaging only; fresh same-contract checkpoints and `16`-frame density rollouts for `author_ffno_cns_base`, `spectral_resnet_bottleneck_base`, and `fno_base`; headline four-row table unchanged |
| PDEBench CNS matched-condition plus-U-NO derivative | `pdebench_cns_uno_row_extension_summary.md` | bounded capped decision-support only, same-contract `neuralop_uno_cns_base` append-only comparator plus `pdebench_cns_matched_condition_metrics_plus_uno.{json,csv,tex}`; headline four-row table unchanged |
| PDEBench CNS larger-cap context | `pdebench_cns_paper_2048cap_extension_summary.md` | bounded capped decision-support only, preserved larger-cap `2048 / 256 / 256` context |
| PDEBench CNS 512cap fallback provenance | `pdebench_cns_paper_table_figure_bundle_summary.md` | bounded capped decision-support only, preserved historical fallback |
| SRU-Net FFNO-to-PtychoBlock encoder cross-pillar probe | `srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md` | corrected `24`-block `20`-epoch CDI/CNS mechanism probes completed; historical `2`-block CDI/CNS roots remain superseded lineage only |
| SRU-Net encoder-order comparison | `srunet_encoder_order_ffno_vs_ptychoblock_summary.md` | three-row CDI and capped CNS encoder-order comparison; reversed `PtychoBlock -> FFNO` order beats the corrected `FFNO -> PtychoBlock` companion on both pillars, but regular SRU-Net remains the better overall anchor |
| PDEBench Darcy static-operator full-training benchmark | `pdebench_darcy_static_operator_summary.md` | `benchmark_performance` for `hybrid_resnet_base`, `fno_base`, `unet_strong` under the locked `8000/1000/1000` split, relative-L2 loss, `50` epochs |
| BRDT sinogram-input adapter contract | `brdt_sinogram_input_adapter_contract.md` | feasibility-only readiness authority for the learned sinogram-input lane; `ffno` and `sru_net` consume measured complex sinograms directly while the Born inverse remains non-learned reference only |
| BRDT corrected pure-FFNO 20-epoch rerun | `brdt_corrected_ffno_row_rerun_summary.md` | pure-BRDT-FFNO `20`-epoch authority under the locked `2048 / 256 / 256` supervised+Born contract; claim boundary `decision_support_append_only` |
| BRDT sinogram-input 40-epoch secondary bundle | `brdt_sinogram_input_40ep_paper_evidence_summary.md` | current additive-secondary BRDT authority for the learned `sinogram` input contract; `sru_net` and `ffno` consume measured complex sinograms directly while the Born inverse remains non-learned reference only |
| BRDT SRU-Net coordgrid diagnostic | `brdt_srunet_sinogram_coordinate_conditioning_ablation_summary.md` | append-only representational diagnostic over the learned `sinogram` input contract; appends deterministic object-grid `x/y` channels after bilinear resize and compares only against the unconditioned `sru_net` lineage authority |
| WaveBench native inverse-source references | `wavebench_native_baseline_summary.md` | candidate-lane external references only; full test-split native U-Net checkpoint evaluation plus official native FNO retraining/eval on the locked `thick_lines_gaussian_lens` split; not manuscript evidence and not a shared-encoder comparison |
| WaveBench shared-encoder supervised benchmark | `wavebench_shared_encoder_supervised_summary.md` | candidate-lane repo-local architecture comparison only; one fixed shared anisotropic measurement encoder + five repo-owned body wrappers (`cnn`, `hybrid_resnet`, `spectral_resnet_bottleneck_net`, `fno`, `ffno`) under one locked split, loss, optimizer, scheduler, seed, and reporting schema; `C=32` minimum row set plus `C=64` latent-width sensitivity; not manuscript evidence and explicitly separate from native WaveBench rows |
| Cross-pillar efficiency table | `paper_efficiency_table_summary.md` | grouped parameter/runtime/throughput context for Synthetic CDI, PDEBench CNS, and current additive-secondary BRDT (sinogram-input `ffno` and `sru_net` from `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` under `paper_evidence_brdt_additive`) |
| Cross-pillar claim audit | `paper_evidence_package_audit_summary.md` | preserves CDI/CNS claim asymmetry |
| CDI natural-patch fixed-probe dataset (`natural_patches128_fixedprobe_v1`) | `cdi_natural_patch_fixedprobe_dataset_summary.md` | dataset prerequisite only; not benchmark evidence; does not replace `lines128` table |
| CDI natural-patch expanded-object benchmark (`natural_patches128_fixedprobe_v1`) | `cdi_natural_patch_expanded_benchmark_summary.md` | single-seed expanded-object CDI bundle currently `benchmark_incomplete`; original tmux launcher exited `1`, recollated bundle preserves the original execution commit and surfaces every required row as `recovered_non_authoritative` (four torch rows still report `failed/1` invocations); not paper-grade until a clean from-scratch tmux launcher exits `0` end-to-end; does not replace `lines128` |
| Completed backlog outcome map | `paper_evidence_index.md` | first-stop durable outcome index |

## Manuscript Incorporation Map

Current manuscript draft:
`docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

| Evidence source | Manuscript target |
|---|---|
| CDI no-refiner FFNO paper refresh | `tab:cdi_lines128_pinn`, `tab:cdi_lines128_objective_controls`, `fig:cdi_main_qualitative`; generated assets: `tables/cdi_lines128_pinn_metrics.tex`, `tables/cdi_lines128_objective_comparison.tex`, `tables/cdi_lines128_metrics_extended.{csv,json,tex}`, `figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet.png`, `figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet_per_panel_scaled.png`, `tables/model_config_by_benchmark.{json,csv,tex}`, `tables/paper_efficiency_table.{json,csv,tex}` |
| Corrected FFNO source reruns | supply the active `FFNO + PINN` and `FFNO + supervised` rows consumed by the paper-local refresh while preserving the historical local-refiner rows as lineage-only context |
| CNS matched-condition refresh (`history_len=5`, `512 / 64 / 64`, `40` epochs) | `tab:cns_bundle` (input: `tables/pdebench_cns_matched_condition_metrics.tex`); `fig:cns_sample_predictions` retained as adjacent context only |
| CNS checkpoint+rollout lineage refresh (`history_len=5`, `512 / 64 / 64`, `40` epochs) | adjacent context only; generated assets: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/runs/*`, `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-08-cns-rollout-checkpoint-refresh/rollouts/*`, and `cns_rollout_checkpoint_refresh_summary.md`. Keep `tab:cns_bundle` unchanged; use these assets only for checkpoint lineage and rollout visuals. |
| CNS matched-condition plus-U-NO derivative (`history_len=5`, `512 / 64 / 64`, `40` epochs) | adjacent context only; generated assets: `tables/pdebench_cns_matched_condition_metrics_plus_uno.{json,csv,tex}`. Keep `tab:cns_bundle` unchanged and use the five-row derivative only when explicit same-contract U-NO context is useful. |
| BRDT corrected pure-FFNO 20-epoch rerun (`pure_brdt_ffno_20ep_authority`) | corrected secondary BRDT FFNO authority — artifact inputs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/metrics.{json,csv}`, `combined_metrics.{json,csv}`, `combined_manifest.json`, `rows/ffno/row_summary.json`, `rows/ffno/model_profile.json`, `same_contract_audit.{json,md}`. The row is same-contract with the four-row preflight, rejects `cnn_blocks`, and records `parameter_count=27394`. |
| BRDT sinogram-input 40-epoch secondary bundle (`paper_evidence_brdt_additive`) | current additive secondary BRDT context — artifact inputs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/metrics.{json,csv}`, `combined_metrics.{json,csv}`, `convergence_audit.{json,csv}`, `paper_evidence_gate.json`, `visuals/sample_0255_compare_q.png`, `visuals/sample_0255_error_q.png`, `figures/source_arrays/sample_0255_*`. This is the current `sru_net` + corrected pure-FFNO authority for the learned `sinogram` input contract. It remains additive secondary evidence only and does not replace CDI `lines128` or PDEBench CNS. |
| BRDT SRU-Net coordgrid diagnostic (`decision_support_append_only_coordgrid_diagnostic`) | append-only BRDT diagnostic context — artifact inputs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation/combined_metrics.{json,csv}`, `comparison_summary.md`, `convergence_audit.{json,csv}`, `rows/sru_net_coordgrid/{row_summary.json,adapter_contract.json,model_profile.json}`, `visuals/sample_0255_compare_q.png`, `visuals/sample_0255_error_q.png`, `figures/source_arrays/sample_0255_*`. This row keeps the same sinogram-input contract but appends deterministic object-grid `x/y` channels after bilinear resize. It regresses materially versus the unconditioned `sru_net` authority and is not promotable. |
| BRDT 40-epoch born-image-input bundle (`historical_brdt_40ep_proxy_context`) | historical secondary transfer/efficiency context — artifact inputs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/metrics.{json,csv}`, `combined_metrics.{json,csv}`, `convergence_audit.{json,csv}`, `paper_evidence_gate.json`, `visuals/sample_0255_compare_q.png`, `visuals/sample_0255_error_q.png`, `figures/source_arrays/sample_0255_*`. The FFNO row is a local-refiner proxy; manuscript use must label it as such and use `2026-05-06-brdt-corrected-ffno-row-rerun` for pure-FFNO reads. The provenance gate caveat remains explicit; this historical `born_init_image` context does not replace CDI `lines128` or PDEBench CNS. |
| Paper efficiency table | grouped parameter/runtime/throughput context for Synthetic CDI, PDEBench CNS, and current additive-secondary BRDT (sinogram-input `ffno` and `sru_net` from `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` under `paper_evidence_brdt_additive`); generated assets: `tables/paper_efficiency_table.{json,csv,tex}` and `paper_efficiency_table_summary.md` |
| Hybrid skip/residual ablation | `tab:cdi_skip_residual_ablation` |

The manuscript CNS headline table is now matched-condition under the
`history_len=5`, `512 / 64 / 64`, `40`-epoch capped contract via
`2026-05-04-cns-matched-condition-table-refresh`. The capped CNS claim
boundary remains `bounded_capped_decision_support_only`; the
larger-cap `history_len=2`, `2048 / 256 / 256` bundle is preserved as
bounded larger-cap context only.

## CDI Lines128 Model Matrix

Fixed contract: synthetic grid-lines `N=128`, `gridsize=1`, `seed=3`,
`40` epochs, custom Run1084 probe, `pad_extrapolate`, train/test images `2/2`.

| Row | Architecture | Training | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Source |
|---|---|---:|---:|---:|---:|---:|---|
| `baseline` | `cnn` | supervised | 0.402672 | 0.786194 | 0.277214 | 0.496353 | complete CDI table |
| `pinn` | `cnn` | PINN | 0.123171 | 0.786322 | 0.732568 | 0.263815 | complete CDI table |
| `pinn_hybrid_resnet` | `hybrid_resnet` | PINN | 0.026939 | 0.072063 | 0.988114 | 0.994740 | complete CDI table |
| `pinn_hybrid_resnet_probe_channels` | `hybrid_resnet` | PINN | 0.028848 | 0.131669 | 0.987296 | 0.961703 | append-only probe-conditioned input ablation (`input_conditioning_mode=probe_real_imag`, `learned_input_channels=3`) |
| `pinn_hybrid_resnet_grid_channels` | `hybrid_resnet` | PINN | 0.025132 | 3.014876 | 0.989999 | 0.186289 | append-only coordinate-grid-conditioned input ablation (`input_conditioning_mode=coordinate_grid`, `learned_input_channels=3`) |
| `pinn_fno_vanilla` | `fno_vanilla` | PINN | 0.124816 | 0.143540 | 0.740936 | 0.933464 | complete CDI table |
| `pinn_spectral_resnet_bottleneck_net` | `spectral_resnet_bottleneck_net` | PINN | 0.024944 | 0.092881 | 0.989855 | 0.972219 | complete CDI table |
| `pinn_ffno` | `ffno` | PINN | 0.082043 | 0.137965 | 0.890305 | 0.959644 | corrected pure-FFNO active paper row (`fno_cnn_blocks=0`); current paper-local tables/figures consume this lineage |
| `pinn_ffno_probe_channels` | `ffno` | PINN | 0.073340 | 3.154161 | 0.907495 | 0.012315 | append-only probe-conditioned pure-FFNO ablation (`input_conditioning_mode=probe_real_imag`, `learned_input_channels=3`, `fno_cnn_blocks=0`) |
| `pinn_ffno_grid_channels` | `ffno` | PINN | 0.069789 | 1.682742 | 0.914300 | 0.384336 | append-only coordinate-grid-conditioned pure-FFNO ablation (`input_conditioning_mode=coordinate_grid`, `learned_input_channels=3`, `fno_cnn_blocks=0`) |
| `pinn_ffno_depth24` | `ffno` | PINN | 0.056506 | 0.121740 | 0.944487 | 0.974460 | append-only pure-FFNO depth-ablation row (`fno_blocks=24`, `fno_cnn_blocks=0`); not yet promoted into paper-local tables |
| `pinn_ffno` (historical proxy) | `ffno` | PINN | 0.062772 | 0.082839 | 0.934830 | 0.981592 | historical FFNO-local-refiner proxy (`fno_cnn_blocks=2`); preserved for lineage only |
| `supervised_ffno` | `ffno` | supervised | 0.351512 | 0.066118 | 0.265006 | 0.901529 | corrected pure-FFNO objective-control rerun (`fno_cnn_blocks=0`); active comparator to corrected `pinn_ffno` |
| `supervised_ffno_depth24` | `ffno` | supervised | 0.348855 | 0.061699 | 0.258666 | 0.910437 | append-only supervised pure-FFNO depth companion (`fno_blocks=24`, `fno_cnn_blocks=0`); not yet promoted into paper-local tables |
| `supervised_ffno` (historical proxy) | `ffno` | supervised | 0.386413 | 0.046563 | 0.248427 | 0.937179 | historical supervised FFNO-local-refiner proxy (`fno_cnn_blocks=2`); preserved for lineage only |
| `pinn_neuralop_uno` | `neuralop_uno` | PINN | 0.093164 | 0.068291 | 0.827995 | 0.956859 | U-NO table extension (append-only) |
| `supervised_neuralop_uno` | `neuralop_uno` | supervised | 0.320684 | 0.056251 | 0.268940 | 0.910490 | U-NO table extension (append-only) |

CDI artifact roots:

- Complete table:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Minimum table preserved root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- FFNO prerequisite pair:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
- Corrected pure-FFNO prerequisite rerun:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z`
- Probe-channel conditioning ablation:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/`
- Coordinate-grid conditioning ablation:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/`
- Pure-FFNO depth-24 ablation:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z`
- Corrected supervised FFNO no-refiner rerun:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z`
- Supervised pure-FFNO depth-24 companion:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/runs/supervised_ffno_depth24_20260507T192840Z`
- Supervised FFNO extension:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`

## CDI Lines128 Pure-FFNO Depth Ablation

Fixed contract: same corrected no-refiner `lines128` CDI contract as the active
`pinn_ffno` row; append-only `decision_support_append_only` depth study only.

| Row | Changed factor | Nearest anchor | amp_mae | phase_mae | amp_ssim | phase_ssim | Source |
|---|---|---|---:|---:|---:|---:|---|
| `pinn_ffno_depth24` | `fno_blocks: 4 -> 24` with `fno_cnn_blocks=0` held fixed | `pinn_ffno` | 0.056506 | 0.121740 | 0.944487 | 0.974460 | CDI depth-24 ablation |

Depth-ablation artifact root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z`

Current read:

- deeper pure FFNO improves every tracked scalar reconstruction metric versus
  the corrected four-block no-refiner row on this locked single-seed CDI
  contract
- the improvement costs substantially more compute and capacity:
  `124,968 -> 701,628` parameters, `873.742 -> 4,754.923` train seconds, and
  `1.231 -> 8.505` inference seconds
- this row remains append-only depth evidence only; the explicit final refresh
  retained the corrected four-block no-refiner row as the repo-local canonical
  FFNO paper pair
- the probe-channel conditioning follow-up is recorded separately because it
  changes learned input composition rather than spectral depth
- U-NO table extension (claim boundary `complete_lines128_cdi_benchmark_plus_uno_extension`):
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z`
- Generated current manuscript tables and refresh authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_pinn_metrics.tex`
  and
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison.tex`
  plus
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_final_paper_refresh_summary.md`

## CDI Lines128 Supervised Pure-FFNO Depth Companion

Fixed contract: same corrected no-refiner `lines128` CDI contract as the active
`supervised_ffno` row; append-only `decision_support_append_only` depth
companion only.

| Row | Changed factor | Nearest anchor | amp_mae | phase_mae | amp_ssim | phase_ssim | Source |
|---|---|---|---:|---:|---:|---:|---|
| `supervised_ffno_depth24` | `fno_blocks: 4 -> 24` with `fno_cnn_blocks=0` held fixed | `supervised_ffno` | 0.348855 | 0.061699 | 0.258666 | 0.910437 | supervised CDI depth-24 companion |

Depth-companion artifact root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/runs/supervised_ffno_depth24_20260507T192840Z`

Current read:

- versus the corrected four-block supervised no-refiner row, the depth-24
  supervised row modestly improves phase-side metrics and nudges amplitude MAE
  downward, but amplitude SSIM/FRC regress slightly
- the compute cost is still much higher than the corrected depth-4 supervised
  anchor despite the small metric movement:
  `124,966 -> 136,355` parameters, `874.939 -> 4609.254` train seconds, and
  `1.229 -> 6.373` inference seconds
- this row completes the supervised half of the FFNO depth-24 family but does
  not itself justify paper-local promotion over the corrected four-block
  supervised row; the final FFNO paper refresh kept the four-block same-depth
  pair canonical

## CDI Lines128 Probe-Channel Conditioning Ablation

Fixed contract: same authoritative `lines128` CDI contract as the complete
six-row bundle and corrected pure-FFNO rerun; append-only
input-conditioning evidence only.

| Row | Changed factor | Nearest anchor | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Source |
|---|---|---|---:|---:|---:|---:|---|
| `pinn_hybrid_resnet_probe_channels` | append fixed probe real/imag channels to the learned input (`learned_input_channels: 1 -> 3`) | `pinn_hybrid_resnet` | 0.028848 | 0.131669 | 0.987296 | 0.961703 | CDI probe-conditioning ablation |
| `pinn_ffno_probe_channels` | append fixed probe real/imag channels to the learned input (`learned_input_channels: 1 -> 3`) while keeping `fno_cnn_blocks=0` | `pinn_ffno` | 0.073340 | 3.154161 | 0.907495 | 0.012315 | CDI probe-conditioning ablation |

Probe-conditioning artifact root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/`

Current read:

- the conditioned Hybrid row does not improve over the authoritative Hybrid
  baseline and materially regresses the phase-side metrics, including
  `phase_mae +0.059606`, `phase_ssim -0.033037`, and
  `phase_frc50 -105.871029`
- the conditioned pure-FFNO row improves some amplitude-side metrics versus the
  corrected pure-FFNO anchor, but phase quality collapses badly enough that the
  row is not promotable:
  `phase_mae +3.016196`, `phase_ssim -0.947329`,
  `phase_frc50 -10.117289`
- this family is append-only input-conditioning evidence only and does not
  replace the six-row CDI authority or the current paper-local FFNO refresh

## CDI Lines128 Coordinate-Grid Conditioning Ablation

Fixed contract: same authoritative `lines128` CDI contract as the complete
six-row bundle and corrected pure-FFNO rerun; append-only
input-conditioning evidence only.

| Row | Changed factor | Nearest anchor | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Source |
|---|---|---|---:|---:|---:|---:|---|
| `pinn_hybrid_resnet_grid_channels` | append deterministic unit-interval `[y, x]` channels to the learned input (`learned_input_channels: 1 -> 3`) | `pinn_hybrid_resnet` | 0.025132 | 3.014876 | 0.989999 | 0.186289 | CDI coordinate-grid conditioning ablation |
| `pinn_ffno_grid_channels` | append deterministic unit-interval `[y, x]` channels to the learned input (`learned_input_channels: 1 -> 3`) while keeping `fno_cnn_blocks=0` | `pinn_ffno` | 0.069789 | 1.682742 | 0.914300 | 0.384336 | CDI coordinate-grid conditioning ablation |

Coordinate-grid conditioning artifact root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/`

Current read:

- the conditioned Hybrid row marginally improves amplitude metrics versus the
  authoritative Hybrid baseline but catastrophically regresses the phase-side
  metrics: `phase_mae +2.942813`, `phase_ssim -0.808451`,
  `phase_frc50 -57.185184`
- the conditioned pure-FFNO row improves amplitude-side metrics versus the
  corrected pure-FFNO anchor but again loses phase quality at a magnitude that
  rules out paper-local promotion: `phase_mae +1.544778`,
  `phase_ssim -0.575308`, `phase_frc50 -31.454777`
- this family is append-only input-conditioning evidence only and does not
  replace the six-row CDI authority or the current paper-local FFNO refresh
- coordinate-grid conditioning was not combined with probe-channel
  conditioning in the same row

## CDI Lines128 Bridge Study

Fixed contract: same `lines128` CDI contract as the complete six-row bundle;
decision-support only.

| Row | Changed factor | Nearest anchor | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Source |
|---|---|---|---:|---:|---:|---:|---|
| `pinn_spectral_resnet_bottleneck_ds1` | shallower encoder/downsampling | `pinn_spectral_resnet_bottleneck_net` | 0.035460 | 0.064363 | 0.979936 | 0.993934 | CDI bridge study |
| `pinn_spectral_resnet_bottleneck_linear_decoder` | lighter bilinear+`1x1` decoder | `pinn_spectral_resnet_bottleneck_net` | 0.134743 | 0.116926 | 0.723464 | 0.904272 | CDI bridge study |
| `pinn_hybrid_resnet_ffno_bottleneck` | FFNO bottleneck inside Hybrid shell | `pinn_hybrid_resnet` | 0.031176 | 0.087600 | 0.983581 | 0.990966 | CDI bridge study |

Bridge-study artifact root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi`

Current read:

- no fresh bridge row displaced the paper-grade CDI anchors
- the DS1 row exposed a phase-leaning trade but lost amplitude fidelity versus
  the spectral anchor
- the linear-decoder bridge regressed sharply
- the FFNO bottleneck did not improve the Hybrid anchor

## CDI Lines128 Hybrid Skip/Residual Ablation

Fixed contract: same `lines128` CDI contract as the complete six-row bundle;
decision-support only.

| Row | Changed factor | Nearest anchor | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Source |
|---|---|---|---:|---:|---:|---:|---|
| `pinn_hybrid_resnet_skip_add` | enable decoder skip fusion with `add` style | `pinn_hybrid_resnet` | 0.026447 | 0.061022 | 0.988681 | 0.993895 | CDI skip/residual ablation |
| `pinn_hybrid_resnet_residual_fixed` | bottleneck residual scale `learned -> fixed` | `pinn_hybrid_resnet` | 0.024611 | 0.077322 | 0.990003 | 0.994298 | CDI skip/residual ablation |
| `pinn_hybrid_resnet_skip_add_residual_fixed` | skip-add plus fixed residual scale | `pinn_hybrid_resnet` | 0.028890 | 0.063259 | 0.986797 | 0.992850 | CDI skip/residual ablation |

Skip/residual artifact root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation`

Current read:

- skip-add is the clearest phase-oriented decision-support variant; it materially improves phase MAE and phase FRC50 while leaving amplitude quality roughly flat to slightly better
- fixed residual scale is the clearest amplitude-oriented decision-support variant; it improves amplitude MAE and SSIM but worsens phase MAE versus the Hybrid anchor
- the combined skip-add plus fixed-residual row does not beat the simpler single-factor rows
- this family does not replace the paper-grade Hybrid baseline or the six-row CDI headline bundle

## CDI Lines128 SRU-Net FFNO-To-PtychoBlock Encoder Probe

Fixed contract: same `lines128` CDI contract as the complete six-row bundle;
decision-support append-only mechanism evidence only.

| Row | Changed factor | Nearest anchor | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Source |
|---|---|---|---:|---:|---:|---:|---|
| `pinn_hybrid_resnet_ffno_ptychoblock_encoder` | replace the current SRU-Net encoder with `24x shared-weight FFNO -> 2x(PtychoBlock + downsample)` while preserving the rest of the shell | `pinn_hybrid_resnet` | 0.036440 | 0.080485 | 0.978317 | 0.990516 | corrected `24`-block rerun (`20`-epoch mechanism probe) |

Encoder-probe artifact root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/`

Current read:

- versus `pinn_hybrid_resnet`, the corrected row regresses every tracked
  aggregate metric, including `amp_mae +0.0095`, `phase_mae +0.0084`,
  `amp_ssim -0.0098`, `phase_ssim -0.0042`, `amp_frc50 -55.52`, and
  `phase_frc50 -27.12`
- versus `pinn_hybrid_resnet_encoder_spectral_only`, the corrected row is also
  worse on every tracked aggregate metric, including `amp_mae +0.0112`,
  `phase_mae +0.0190`, `amp_frc50 -57.75`, and `phase_frc50 -56.03`
- versus the historical `pinn_ffno` local-proxy row, the corrected encoder row
  is better across all tracked scalar metrics
- versus `pinn_hybrid_resnet_ffno_bottleneck`, phase MAE and phase PSNR improve
  slightly, but amplitude quality and both FRC50 scores regress sharply
- versus the historical `2`-block same-item root, the corrected `24`-block row
  is worse on every tracked metric
- this row remains append-only mechanism evidence and does not replace the
  paper-grade six-row CDI authority or the append-only U-NO extension

## CDI Lines128 SRU-Net Encoder-Order Comparison

Fixed contract: same `lines128` CDI contract as the complete six-row bundle;
all mechanism rows use the bounded `20`-epoch probe budget.

| Row | Encoder order | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Amp FRC50 | Phase FRC50 | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `pinn_hybrid_resnet` | regular SRU-Net anchor | 0.026939 | 0.072063 | 0.988114 | 0.994740 | 135.464 | 106.801 | six-row CDI authority |
| `pinn_hybrid_resnet_ffno_ptychoblock_encoder` | `FFNO -> PtychoBlock` | 0.036440 | 0.080485 | 0.978317 | 0.990516 | 79.948 | 79.676 | corrected `24`-block companion rerun |
| `pinn_hybrid_resnet_ptychoblock_ffno_encoder` | `PtychoBlock -> FFNO` | 0.037627 | 0.078883 | 0.978619 | 0.990768 | 133.998 | 104.968 | fresh reversed-order rerun |

Current read:

- the reversed-order row remains worse than regular SRU-Net on the main CDI
  scalar metrics, including `amp_mae +0.0107` and `phase_mae +0.0068`
- versus the corrected `FFNO -> PtychoBlock` companion, the reversed-order row
  is mixed on amplitude image-space scalars but improves phase MAE, both SSIMs,
  and both FRC50 scores, especially `amp_frc50 +54.05`
- the strongest directional gain from moving FFNO post-downsample is frequency
  recovery: the reversed-order row lands within `1.47` amp-FRC50 points and
  `1.83` phase-FRC50 points of the regular SRU-Net anchor
- this comparison remains append-only mechanism evidence only and does not
  replace the six-row CDI authority

## CDI Natural-Patch Expanded-Object Benchmark (Recovered Non-Authoritative)

Fixed contract: frozen `natural_patches128_fixedprobe_v1`, `N=128`, single-shot
CDI forward model, `8000 / 1000 / 1000` train/validation/test split, fixed
Run1084 probe lineage `pad_extrapolate:128|smooth:0.5`, fixed seed `3`, and the
locked six-row roster from
`2026-05-04-cdi-natural-patch-expanded-benchmark`. This is a single-seed
expanded-object bundle only and does not replace the synthetic `lines128`
headline authority.

**Status: not paper-grade.** The original tmux launcher exited `1` during
bundle collation. The numbers below are surfaced from the recovered
(non-authoritative) bundle for diagnostic context only; they must not be cited
as authoritative natural-patch evidence until a clean from-scratch tmux
launcher exits `0` end-to-end. The four torch rows still report
`row_invocation_status="failed"` / `row_invocation_exit_code=1` in the bundle.

| Row | Architecture | Training | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Source |
|---|---|---:|---:|---:|---:|---:|---|
| `baseline` | `cnn` | supervised | 0.071577 | 0.395421 | 0.486379 | 0.645555 | natural-patch expanded benchmark (recovered, non-authoritative) |
| `pinn` | `cnn` | PINN | 0.292340 | 1.447159 | 0.244005 | 0.236621 | natural-patch expanded benchmark (recovered, non-authoritative) |
| `pinn_hybrid_resnet` | `hybrid_resnet` | PINN | 0.260947 | 0.437384 | 0.027532 | 0.442481 | natural-patch expanded benchmark (recovered, non-authoritative) |
| `pinn_fno_vanilla` | `fno_vanilla` | PINN | 0.157052 | 0.423742 | 0.042040 | 0.530686 | natural-patch expanded benchmark (recovered, non-authoritative) |
| `pinn_ffno` | `ffno` | PINN | 0.156714 | 0.396144 | 0.059388 | 0.604128 | natural-patch expanded benchmark (recovered, non-authoritative FFNO-local-refiner proxy) |
| `pinn_neuralop_uno` | `neuralop_uno` | PINN | 0.170826 | 0.399654 | 0.051717 | 0.598107 | natural-patch expanded benchmark (recovered, non-authoritative) |

Natural-patch artifact root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/runs/natural-patch-benchmark-20260505T213458Z`

Current read:

- The single-seed natural-patch ranking inverts the synthetic `lines128`
  headline story: the supervised CNN baseline is strongest on MAE and SSIM.
- Learned operator rows still contribute distinct signal: Hybrid ResNet and FNO
  Vanilla lead phase FRC50, and Hybrid ResNet posts the best amplitude FRC50.
- FFNO and U-NO are the closest learned rows to the baseline on phase MAE, but
  neither recovers the baseline's amplitude quality on natural-image-derived
  objects.
- The bundle is usable, but it carries a recovered-invocation caveat because
  the final tables/manifests were rebuilt from completed row artifacts after
  the tracked launcher exited on harness post-processing bugs.

## PDEBench Darcy Static-Operator Model Matrix

Fixed contract: PDEBench `2D_DarcyFlow_beta1.0_Train.hdf5`, native `128x128`,
sample-level deterministic split `8000 / 1000 / 1000` with seed `20260420`,
relative-L2 loss, Adam `lr=2e-4`, `ReduceLROnPlateau(factor=0.5, patience=2,
min_lr=1e-5, threshold=0.0)`, `50` epochs, batch size `8`, single RTX 3090.
Metrics in denormalized target units.

| Profile | Status | err_RMSE | err_nRMSE | relative_l2 | Parameters | Source |
|---|---|---:|---:|---:|---:|---|
| `hybrid_resnet_base` | completed | 0.018767 | 0.085735 | 0.085735 | 7,786,178 | Darcy full-training benchmark |
| `fno_base` | completed | 0.018109 | 0.082727 | 0.082727 | 357,217 | Darcy full-training benchmark |
| `unet_strong` | completed | 0.020415 | 0.093264 | 0.093264 | 7,762,465 | Darcy full-training benchmark |

Same-contract ranking (`relative_l2`, lower is better):
`fno_base < hybrid_resnet_base < unet_strong`.

Darcy artifact root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/full_benchmark_20260504T182832Z`

Literature calibration values recorded in
`.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/full_benchmark_20260504T182832Z/literature_context.json`
are calibration context only and use different (often `2x`-subsampled)
protocols; they are not same-contract reproduction targets for the rows above.

## PDEBench CNS Model Matrix

Fixed manuscript headline contract: PDEBench 2D_CFD CNS `128x128`,
`history_len=5`, `512 / 64 / 64`, `40` epochs, `mse`, capped
decision-support boundary. Source: matched-condition table refresh
(`pdebench_cns_matched_condition_table_refresh_summary.md`).

| Row | Architecture | Training | relative_l2 | fRMSE_high | Role | Source |
|---|---|---|---:|---:|---|---|
| `author_ffno_cns_base` | authored FFNO | supervised | 0.019758 | 0.101807 | headline (matched h5) | CNS matched-condition refresh |
| `spectral_resnet_bottleneck_base` | spectral bottleneck | supervised | 0.033069 | 0.262218 | headline (matched h5) | CNS matched-condition refresh |
| `fno_base` | FNO | supervised | 0.038425 | 0.432856 | headline (matched h5) | CNS matched-condition refresh |
| `neuralop_uno_cns_base` | external NeuralOperator U-NO | supervised | 0.038466 | 0.260623 | adjacent append-only comparator | CNS U-NO row extension |
| `unet_strong` | U-Net | supervised | 0.538623 | 1.742789 | headline (matched h5) | CNS matched-condition refresh |

Larger-cap `history_len=2`, `2048 / 256 / 256` rows from the
`2048cap` extension bundle remain available as bounded larger-cap
context (`pdebench_cns_paper_2048cap_extension_summary.md`):
`author_ffno_cns_base` `relative_l2=0.026314` /
`fRMSE_high=0.067210`,
`spectral_resnet_bottleneck_base` `0.042166` / `0.311760`,
`fno_base` `0.050722` / `0.495405`,
`unet_strong` `0.597567` / `0.709608`. These rows are not
the manuscript headline ranking.

The derived plus-U-NO bundle keeps the four headline rows unchanged by lineage
and appends `neuralop_uno_cns_base` as same-contract context only. U-NO is
effectively tied with FNO on aggregate error, improves on `fRMSE_mid` and
`fRMSE_high` versus FNO, and slightly edges the SRU-Net row on
`fRMSE_high`, but it still trails the SRU-Net row on aggregate and low-band
error. The manuscript recommendation therefore stays append-only.

Best observed capped CNS rows by model family, selected by lowest observed
`relative_l2`/`err_nRMSE` across completed capped evidence:

| Family | Row | Contract | relative_l2 | fRMSE_high | Source |
|---|---|---|---:|---:|---|
| FFNO | `author_ffno_cns_base` | `2048 / 256 / 256`, `history_len=2`, `40` epochs | 0.026314 | 0.067210 | CNS 2048 authority bundle |
| SRU-Net | `spectral_resnet_bottleneck_base` | `512 / 64 / 64`, `history_len=5`, `40` epochs | 0.033069 | 0.262218 | `pdebench_cns_spectral_history_len4plus_compare_summary.md` |
| FNO | `fno_base` | `512 / 64 / 64`, `history_len=5`, `40` epochs | 0.038425 | 0.432856 | `pdebench_cns_history5_comparator_gap_fill_summary.md` |
| U-NO | `neuralop_uno_cns_base` | `512 / 64 / 64`, `history_len=5`, `40` epochs | 0.038466 | 0.260623 | `pdebench_cns_uno_row_extension_summary.md` |
| U-Net | `unet_strong` | `512 / 64 / 64`, `history_len=5`, `40` epochs | 0.538623 | 1.742789 | `pdebench_cns_history5_comparator_gap_fill_summary.md` |

Notes:

- The table above is not a same-contract ranking; it is a best-observed capped
  evidence summary.
- The best observed SRU-Net high-band row remains
  `spectral_resnet_bottleneck_shared_blocks10`, `1024 / 128 / 128`,
  `80` epochs, `relative_l2=0.037568`, `fRMSE_high=0.215245`; see
  `pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md`.
- The same-contract `history_len=5`, `512 / 64 / 64`, `40`-epoch matched
  bundle from `2026-05-04-cns-matched-condition-table-refresh` is the current
  manuscript headline CNS authority.
- The five-row plus-U-NO derivative from
  `2026-05-06-cns-uno-matched-condition-row-extension` preserves that
  headline bundle by lineage and appends one same-contract comparator only.
- The selection above is by aggregate error. The U-Net h5 capped row
  improves aggregate error and `fRMSE_low` versus its `2048`-cap h2
  anchor but regresses `fRMSE_mid` and `fRMSE_high`; the
  `2048 / 256 / 256`, `history_len=2`, `40`-epoch U-Net row remains
  the better high-band U-Net evidence. The FNO h5 capped row improves
  every recorded metric versus its `2048`-cap h2 anchor.
- The larger-cap `2048 / 256 / 256`, `history_len=2`, 40-epoch bundle remains
  bounded larger-cap context only; see
  `pdebench_cns_paper_2048cap_extension_summary.md` (bundle root
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/`).
  No capped CNS bundle is relabelled `paper_grade` or `full_training`.
- `hybrid_resnet_cns` remains continuity/support context from the historical
  `512 / 64 / 64` fallback bundle because the promoted `2048` authority bundle
  intentionally carries only the four headline rows.

CNS current headline bundle root:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-matched-condition-table-refresh/`

CNS plus-U-NO derivative root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/`

CNS larger-cap context bundle root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap`

Historical fallback bundle root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle`

Bundle outputs include:

- `cns_paper_table_rows.json`
- `cns_paper_table_rows.csv`
- `cns_paper_table_rows.tex`
- `cns_paper_table_rows_plus_uno.json`
- `cns_paper_table_rows_plus_uno.csv`
- `cns_paper_table_rows_plus_uno.tex`
- fixed-sample prediction/error figures
- `shared_field_scales.json`
- `fixed_sample_manifest.json`
- `2048_same_cap_audit.json`

CNS matched-condition headline status:

- The manuscript CNS headline ranking is now the matched-condition
  `history_len=5`, `512 / 64 / 64`, `40`-epoch table; see
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`.
- The four headline rows reuse the completed `history_len=5` lane
  evidence from
  `2026-05-04-cns-history5-comparator-gap-fill`,
  `2026-05-01-cns-author-ffno-history-length-study`, and
  `2026-04-29-cns-spectral-history-len4plus-compare`.
- Capped claim boundary unchanged
  (`bounded_capped_decision_support_only`); no row is relabelled
  `paper_grade` or `full_training`.
- The larger-cap `history_len=2`, `2048 / 256 / 256` capped bundle is
  preserved as bounded larger-cap context only and is no longer the
  manuscript headline source.

## PDEBench CNS SRU-Net FFNO-To-PtychoBlock Encoder Probe

Fixed contract: matched-condition PDEBench `2d_cfd_cns` capped lane
`history_len=5`, `512 / 64 / 64`; this backlog item records a bounded capped
`20`-epoch mechanism probe only.

| Row | Architecture | relative_l2 | fRMSE_low | fRMSE_mid | fRMSE_high | Source |
|---|---|---:|---:|---:|---:|---|
| `hybrid_resnet_ffno_ptychoblock_encoder_cns` | `hybrid_resnet_ffno_ptychoblock_encoder` | 0.049338 | 2.731857 | 0.201231 | 0.553375 | corrected `24`-block rerun (`20`-epoch mechanism probe) |

Current read:

- the corrected row is worse than `spectral_resnet_bottleneck_base` on every
  tracked metric, including `relative_l2 +0.0163`, `fRMSE_mid +0.0299`, and
  `fRMSE_high +0.2912`
- it is also worse than `author_ffno_cns_base` and `fno_base` on every tracked
  metric
- it still beats `unet_strong` on every tracked metric
- versus the historical `2`-block same-item root, the corrected `24`-block row
  is worse on every tracked metric
- by aggregate error the row lands between `fno_base` and `unet_strong`
- this row remains bounded capped mechanism evidence only and does not reopen
  full-training claims or alter the matched-condition headline authority

## PDEBench CNS SRU-Net Encoder-Order Comparison

Fixed contract: matched-condition PDEBench `2d_cfd_cns` capped lane
`history_len=5`, `512 / 64 / 64`; both mechanism rows remain bounded `20`-epoch
probes against the existing `40`-epoch headline authority.

| Row | Encoder order | relative_l2 | fRMSE_low | fRMSE_mid | fRMSE_high | Source |
|---|---|---:|---:|---:|---:|---|
| `spectral_resnet_bottleneck_base` | regular SRU-Net anchor | 0.033069 | 1.854178 | 0.171350 | 0.262218 | matched-condition headline authority |
| `hybrid_resnet_ffno_ptychoblock_encoder_cns` | `FFNO -> PtychoBlock` | 0.049338 | 2.731857 | 0.201231 | 0.553375 | corrected `24`-block companion rerun |
| `hybrid_resnet_ptychoblock_ffno_encoder_cns` | `PtychoBlock -> FFNO` | 0.037419 | 2.040936 | 0.215718 | 0.457957 | fresh reversed-order rerun |

Current read:

- the reversed-order row is still worse than the regular SRU-Net anchor on
  every tracked CNS metric, so the order swap does not beat the headline shell
- versus the corrected `FFNO -> PtychoBlock` companion, the reversed-order row
  materially improves `relative_l2`, `fRMSE_low`, and `fRMSE_high`
- the only tracked regression versus the companion is `fRMSE_mid +0.0145`
- operationally, the reversed-order row is also lighter than the companion on
  this capped lane: runtime and peak CUDA memory both drop substantially
- this comparison remains bounded capped mechanism evidence only and does not
  alter the matched-condition headline authority

## BRDT Decision-Support Bundle

The 40-epoch BRDT bundle remains repo-local **decision-support context only**
under claim boundary `decision_support_convergence_followup`. The
paper-evidence gate failed promotion (`failed_gate_checks=["git_provenance",
"host_provenance"]`) after honest reconstruction of the overwritten
`runtime_provenance.json` from `invocation.json`; an additive
paper-evidence promotion would require retraining on a clean repo so the
original runtime provenance is captured at training time. The bundle does
**not** replace CDI `lines128` or PDEBench CNS.

Current decision-support authority root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`

Current decision-support inputs (not manuscript paper evidence):

- `metrics.json`
- `metrics.csv`
- `convergence_audit.json`
- `convergence_audit.csv`
- `paper_evidence_gate.json`
- `visuals/sample_0255_compare_q.png`
- `visuals/sample_0255_error_q.png`
- `visuals/sample_0255_sinogram_residual.png`
- `figures/source_arrays/sample_0255_*`

## Completed Backlog Output Coverage

| Backlog item | Output type | Summary authority | Artifact root / durable output |
|---|---|---|---|
| `2026-04-21-pdebench-author-ffno-equal-footing-cns` | CNS authored FFNO baseline | `pdebench_author_ffno_equal_footing_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/` |
| `2026-04-22-pdebench-gnot-paper-default-cns-compare` | CNS GNOT comparator | `pdebench_gnot_cns_compare_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/` |
| `2026-04-21-pdebench-cns-markov-history1-compare` | CNS history ablation | `pdebench_cns_markov_history1_compare_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/` |
| `2026-04-21-pdebench-cns-spectral-modes32-compare` | CNS spectral modes ablation | `pdebench_spectral_modes32_compare_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/` |
| `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation` | CNS spectral architecture ablation | `pdebench_cns_hybrid_spectral_arch_ablation_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/` |
| `2026-04-27-pdebench-ffno-convolutional-features-cns` | CNS FFNO-family local-conv ablation | `pdebench_ffno_convolutional_features_cns_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/` |
| `2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap` | CNS cap-scaling decision support | `pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/` |
| `2026-04-28-pdebench-cns-spectral-modes24-convergence-compare` | CNS modes24 convergence ablation | `pdebench_spectral_modes24_convergence_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/` |
| `2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence` | CNS deeper shared-block convergence | `pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/` |
| `2026-04-29-pdebench-cns-history-len3plus-compare` | CNS longer-history ablation | `pdebench_cns_history_len3plus_compare_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/` |
| `2026-04-29-cns-spectral-history-len4plus-compare` | CNS spectral longer-history follow-up | `pdebench_cns_spectral_history_len4plus_compare_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/` |
| `2026-05-01-cns-author-ffno-history-length-study` | CNS authored-FFNO longer-history follow-up | `pdebench_author_ffno_history_length_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/` |
| `2026-05-04-cns-history5-comparator-gap-fill` | CNS history-len-5 FNO/U-Net comparator gap fill | `pdebench_cns_history5_comparator_gap_fill_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/` |
| `2026-04-27-cdi-ffno-generator-lines-best-config` | CDI historical FFNO-local-refiner proxy prerequisite pair | `cdi_ffno_generator_lines_best_config_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet` |
| `2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space` | CNS shell-bridge ablation | `pdebench_cns_hybrid_spectral_ffno_parameter_space_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/` |
| `2026-04-29-cdi-lines128-paper-benchmark-harness` | CDI harness readiness | `lines128_paper_benchmark_harness_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight/` |
| `2026-04-29-cns-paper-contract-decision` | CNS contract lock | `pdebench_cns_paper_contract_decision.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_same_contract_audit.json` |
| `2026-04-29-cns-paper-benchmark-rows` | CNS row lock | `pdebench_cns_paper_row_lock_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/` |
| `2026-04-29-cns-paper-table-figure-bundle` | CNS historical fallback bundle | `pdebench_cns_paper_table_figure_bundle_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/` |
| `2026-04-29-cns-paper-2048cap-row-extension` | CNS current 2048cap authority bundle | `pdebench_cns_paper_2048cap_extension_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/` |
| `2026-05-06-cns-uno-matched-condition-row-extension` | same-contract CNS U-NO append-only comparator plus derived five-row table bundle | `pdebench_cns_uno_row_extension_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/` |
| `2026-04-29-cdi-lines128-minimum-paper-table` | CDI minimum table | `lines128_minimum_paper_table_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z` |
| `2026-04-29-cdi-lines128-paper-benchmark-execution` | CDI complete six-row table with historical FFNO-local-refiner proxy row | `lines128_paper_benchmark_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux` |
| `2026-04-29-cdi-lines128-supervised-equivalent-rows` | CDI supervised FFNO-local-refiner proxy extension | `lines128_supervised_equivalent_rows_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z` |
| `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun` | CDI corrected supervised FFNO no-refiner objective-control rerun | `cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z` |
| `2026-04-27-hybrid-spectral-ffno-parameter-space-cdi` | CDI Hybrid/spectral to FFNO bridge study | `cdi_hybrid_spectral_ffno_parameter_space_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi` |
| `2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation` | CDI Hybrid skip/residual same-contract ablation | `lines128_hybrid_resnet_skip_residual_ablation_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation` |
| `2026-04-21-hybrid-resnet-encoder-fusion-variants` | CDI Hybrid encoder-fusion same-contract ablation | `lines128_hybrid_resnet_encoder_fusion_variants_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/` |
| `2026-05-04-cdi-lines128-srunet-branch-objective-ablation` | CDI SRU-Net branch / objective same-contract ablation (`decision_support_append_only`) | `lines128_srunet_branch_objective_ablation_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-branch-objective-ablation/runs/ablation_20260505T010316Z` |
| `2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap` | cross-pillar SRU-Net FFNO-to-PtychoBlock encoder probe | `srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap/` (corrected `24`-block reruns completed; historical `2`-block roots superseded) |
| `2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension` | cross-pillar SRU-Net encoder-order comparison (`FFNO -> PtychoBlock` vs `PtychoBlock -> FFNO`) | `srunet_encoder_order_ffno_vs_ptychoblock_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/` |
| `2026-04-30-cdi-lines128-uno-design-preflight` | CDI U-NO environment/API readiness | `lines128_uno_preflight_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/` |
| `2026-04-29-paper-evidence-package-audit` | cross-pillar evidence audit | `paper_evidence_package_audit_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/` |
| `2026-05-05-paper-efficiency-table` | cross-pillar efficiency packaging | `paper_efficiency_table_summary.md` | `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.{json,csv,tex}` |
| `2026-04-29-brdt-operator-validation` | BRDT candidate-lane operator validation (feasibility, not paper evidence) | `brdt_operator_validation_report.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/` |
| `2026-04-29-brdt-dataset-preflight` | BRDT candidate-lane smoke dataset (feasibility, not paper evidence) | `brdt_dataset_preflight.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/` |
| `2026-04-29-brdt-task-adapters` | BRDT candidate-lane task-local adapters / loss / train-eval surfaces (adapter readiness only, not paper evidence) | `brdt_task_adapters.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/` |
| `2026-05-07-brdt-sinogram-input-adapter-contract` | BRDT learned sinogram-input readiness-only contract; keeps historical `born_init_image` lineage discoverable while moving `ffno` and `sru_net` to measured complex sinogram input | `brdt_sinogram_input_adapter_contract.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/` |
| `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` | current additive-secondary BRDT authority for the learned `sinogram` input contract; reruns `sru_net` and corrected `ffno` for `40` epochs, keeps the model-based Born inverse as non-learned reference only, emits gate/provenance/manifests, and refreshes repo-local manuscript BRDT assets to the new root | `brdt_sinogram_input_40ep_paper_evidence_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/` |
| `2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation` | append-only BRDT coordgrid diagnostic that reuses the immutable sinogram-input authority by lineage, trains only `sru_net_coordgrid`, and records the same-contract regression caused by appending deterministic object-grid `x/y` channels after bilinear resize | `brdt_srunet_sinogram_coordinate_conditioning_ablation_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation/` |
| `2026-04-29-wavebench-native-baseline-reproduction` | WaveBench candidate-lane native reference baselines on locked `time_varying/is/thick_lines_gaussian_lens`; reusable native U-Net checkpoint evaluation plus official native FNO retraining/eval; external-reference context only, not paper evidence | `wavebench_native_baseline_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-native-baseline-reproduction/` |
| `2026-04-29-wavebench-shared-encoder-supervised-benchmark` | WaveBench candidate-lane repo-local shared-encoder supervised benchmark on locked `time_varying/is/thick_lines_gaussian_lens`; one fixed shared anisotropic measurement encoder + five repo-owned body wrappers (`cnn`, `hybrid_resnet`, `spectral_resnet_bottleneck_net`, `fno`, `ffno`) under one locked split, loss, optimizer, scheduler, seed, and reporting schema; `C=32` minimum row set plus `C=64` latent-width sensitivity; not paper evidence and not native-reference equivalent | `wavebench_shared_encoder_supervised_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-shared-encoder-supervised-benchmark/` |
| `2026-04-29-brdt-four-row-preflight` | BRDT bounded four-row decision-support preflight (decision_support_preflight_only, not paper evidence) | `brdt_preflight_summary.md` (owned by `2026-04-29-brdt-preflight-summary-promotion-decision`) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/` |
| `2026-05-04-brdt-physics-only-objective-ablation` | BRDT append-only physics-only objective ablation for the three neural rows (decision_support_append_only, not paper evidence) | `brdt_physics_only_objective_ablation_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-physics-only-objective-ablation/` |
| `2026-05-04-brdt-ffno-row-extension` | BRDT append-only architecture row extension adding a single factorized Fourier operator (FFNO) row to the four-row preflight (decision_support_append_only, not paper evidence) | `brdt_ffno_row_extension_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/` |
| `2026-05-06-brdt-corrected-ffno-row-rerun` | corrected BRDT pure-FFNO rerun under the original `20`-epoch supervised+Born contract; same-contract audit and no-refiner proof included; candidate-lane decision support only | `brdt_corrected_ffno_row_rerun_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/` |
| `2026-05-05-brdt-supervised-born-40ep-paper-evidence` | historical secondary BRDT context; reruns `hybrid_resnet` and the historical `ffno` local-refiner proxy at `40` epochs with scheduler/history provenance, convergence audit, sample-`255` visual/source-array set, parameter counts, and evaluation throughput over the 256-sample test split. Provenance caveats remain recorded, and the FFNO row is superseded for pure-FFNO claims by `2026-05-06-brdt-corrected-ffno-row-rerun`. | `brdt_supervised_born_40ep_paper_evidence_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/` |
| `2026-05-04-pdebench-darcy-full-training-benchmark` | PDEBench Darcy full-training benchmark | `pdebench_darcy_static_operator_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/full_benchmark_20260504T182832Z` |
| `2026-05-04-cdi-natural-patch-fixedprobe-dataset` | CDI natural-patch fixed-probe dataset prerequisite (`natural_patches128_fixedprobe_v1`; not benchmark evidence) | `cdi_natural_patch_fixedprobe_dataset_summary.md` | `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/` |
| `2026-05-04-cdi-natural-patch-expanded-benchmark` | CDI natural-patch expanded-object single-seed benchmark (`natural_patches128_fixedprobe_v1`; bundle currently `benchmark_incomplete` recovered/non-authoritative — original tmux launcher exited `1`, recollated with original execution commit preserved, every required row `recovered_non_authoritative`, four torch rows still report `failed/1` invocations) | `cdi_natural_patch_expanded_benchmark_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/runs/natural-patch-benchmark-20260505T213458Z` |
| `2026-02-26-hybrid-resnet-skip-mode-search-design` | legacy CDI architecture-search design | `docs/studies/index.md#hybrid-resnet-mode-skip-sweep` | `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221` |
| `2026-02-26-hybrid-resnet-skip-mode-search-stage-a-execution` | legacy CDI architecture-search execution | `docs/studies/index.md#hybrid-resnet-mode-skip-sweep` | `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221` |
| `2026-02-26-hybrid-resnet-skip-mode-search-stage-b-execution` | legacy CDI architecture-search execution | `docs/studies/index.md#hybrid-resnet-mode-skip-sweep` | `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221` |
| `2026-02-26-hybrid-resnet-skip-mode-search-stage-c-execution` | legacy CDI architecture-search execution | `docs/studies/index.md#hybrid-resnet-mode-skip-sweep` | `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221` |
| `2026-02-26-hybrid-resnet-skip-mode-search-stage-d-execution` | legacy CDI architecture-search execution | `docs/studies/index.md#hybrid-resnet-mode-skip-sweep` | `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221` |
| `2026-02-26-hybrid-resnet-skip-mode-search-stage-e-execution` | legacy CDI architecture-search execution | `docs/studies/index.md#hybrid-resnet-mode-skip-sweep` | `outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221` |

## Ablation Families

| Family | Completed outputs represented | Current read |
|---|---|---|
| CDI `lines128` headline architecture table | harness, minimum subset, complete table | complete six-row CDI bundle is current paper authority |
| CDI FFNO training mode | supervised FFNO extension | historical local-refiner proxy context only; the old supervised row improved phase MAE but lost badly on amplitude quality |
| CDI FFNO no-refiner objective control | corrected pure-FFNO `pinn_ffno` + `supervised_ffno` pair | under the shared no-refiner FFNO contract, supervised keeps lower phase MAE but is materially worse on amplitude metrics and phase SSIM/FRC than the corrected PINN row |
| CDI Hybrid/spectral to FFNO bridge | hybrid-spectral/FFNO parameter-space | no fresh bridge row beat the existing anchors; DS1 is a phase-leaning trade only |
| CDI Hybrid skip/residual controls | same-contract skip/residual ablation | skip-add helps phase-side metrics without displacing the anchor, fixed residual helps amplitude-side metrics, and the combined row shows no constructive interaction |
| CDI Hybrid branch / objective controls | same-contract branch ablation + supervised objective control | spectral-only edges out conv-only on amplitude under the locked contract and roughly matches the both-branch baseline; conv-only is similar with a small phase-MAE/FRC trade. Supervised SRU-Net keeps the body and contract but shows an amplitude-scale collapse without the PINN physics-consistency loss, so the row is presented as objective-control evidence, not a CDI headline candidate. |
| CDI/CNS SRU-Net FFNO-to-PtychoBlock encoder probe | cross-pillar same-recipe encoder mechanism probe | the encoder swap is not a clean upgrade: on CDI it trades amplitude for better phase-side metrics, while on matched-condition CNS it lands between FNO and U-Net on aggregate error but improves `fRMSE_high` versus FNO. Keep as append-only / bounded capped mechanism evidence only. |
| CDI U-NO extension readiness | U-NO design preflight | external `UNO` imports cleanly in `ptycho311`; later integration must preserve the frozen `uno_out_channels`, nested `uno_n_modes`, `uno_scalings`, and direct real/imag output contract |
| CDI FFNO generator prerequisite pair | FFNO vs Hybrid pair | prerequisite evidence, superseded for table claims by complete bundle |
| CNS paper contract and bundle | contract, row lock, table/figure bundle | bounded capped CNS table only; no full-training SOTA claim |
| CNS authored FFNO | author FFNO equal-footing row | strongest locked capped CNS row |
| CNS GNOT | paper-default GNOT compare | not promoted because aggregate error was poor |
| CNS history length | history1, history3+, spectral history4+, authored-FFNO history3+, and the all-h5 FNO/U-Net comparator gap fill | history1 bad; history3 improved the stronger `40`-epoch rows; spectral history4 and history5 kept helping at `40` epochs; authored-FFNO history3 and history4 each strictly improved every recorded metric over the previous step at `40` epochs and history5 traded a small aggregate-error and `fRMSE_low` regression versus history4 for further `fRMSE_mid` and `fRMSE_high` gains; the h5 gap fill shows `fno_base` improves every metric versus its frozen h2 anchor while `unet_strong` improves aggregate error and `fRMSE_low` but regresses `fRMSE_mid` / `fRMSE_high`. All longer-history rows remain adjacent capped context only and do not change the locked `history_len=2` paper lane |
| CNS spectral modes | modes32 and modes24 compares | no higher-mode default promotion |
| CNS hybrid-spectral architecture | sharing/depth, 2048-cap, shared-blocks10 longer convergence | deeper variants remain decision-support |
| CNS FFNO-family local convolution | local-conv follow-up | improves repo-local FFNO proxy but not stronger than authored FFNO |
| CNS shell bridge | hybrid-spectral/FFNO parameter-space | promotion gate closed |
| Legacy CDI lines256 search | Stage A-E skip/mode search | historical architecture context only |
| BRDT objective ablation | supervised+Born baseline + relative-physics-only neural ablation | append-only candidate-lane decision support: FNO collapse is largely objective-induced; U-Net collapse persists with compressed output dynamic range; Hybrid ResNet's image-space gap widens without the supervised image L1 signal |
| BRDT sinogram-input adapter contract | historical `born_init_image` adapter lineage + dedicated `sinogram` smoke proof for `ffno` and `sru_net` | readiness-only contract split: learned rows now consume measured complex sinograms directly, the Born inverse remains non-learned reference only, and the old `born_init_image` summaries remain honest historical lineage rather than being silently rewritten |
| BRDT FFNO row extension | supervised+Born four-row preflight baseline (by lineage) + appended historical FFNO-local-refiner proxy row | append-only candidate-lane context: the artifact-producing code included post-bottleneck CNN refiners (`parameter_count=36674`), so the recorded metrics are not a pure FFNO-paper-stack result. Current BRDT FFNO code removes the refiners (`parameter_count=27394`) and requires a fresh rerun before pure-FFNO claims. |
| BRDT corrected pure-FFNO rerun | same-contract four-row preflight baseline (by lineage) + corrected no-refiner FFNO row | pure-BRDT-FFNO `20`-epoch authority: same fairness contract as the four-row preflight, `cnn_blocks` rejected, `parameter_count=27394`, and weaker headline metrics than the historical local-refiner proxy. Decision-support only; does not promote BRDT into the required pillars. |
| BRDT 40-epoch secondary evidence | fresh 40-epoch `hybrid_resnet` and historical `ffno` local-refiner-proxy supervised+Born rows plus model-based Born inverse context | Hybrid ResNet remains best in image space (`image_relative_l2_phys=0.2875`). The recorded FFNO-like row is competitive at lower parameter count, but it was generated with the old local-refiner wrapper; preserve the provenance/architecture caveats and use `2026-05-06-brdt-corrected-ffno-row-rerun` for pure-FFNO reads. |
| BRDT SRU-Net coordgrid diagnostic | immutable sinogram-input `sru_net` authority plus one appended `sru_net_coordgrid` row with deterministic object-grid `x/y` channels after bilinear resize | append-only representational diagnostic only: the coordgrid row regresses every required same-contract image and measurement metric versus the unconditioned `sru_net` authority, reduces throughput, and is not promotable. |
| WaveBench native reference baselines | native U-Net checkpoint reuse + official native FNO retraining route on the locked `thick_lines_gaussian_lens` split | candidate-lane external-reference context only: the native U-Net row beats the reproduced native FNO row on all four packaged metrics, but neither row is a fair repo-local architecture comparison and neither promotes WaveBench into manuscript evidence |
| Cross-pillar efficiency table | Synthetic CDI, PDEBench CNS, and approved secondary BRDT rows | grouped packaging output only: compiles parameter counts, provenance-backed runtime fields, hardware labels, and explicit inference throughput where present. It does not rank rows across benchmarks or convert heterogeneous training runtimes into cross-benchmark rate claims. |

## Maintenance

- Add new completed backlog outputs here when a backlog item moves to
  `docs/backlog/done/` and produces metrics, figures, contract decisions,
  harnesses, audits, or durable readiness outputs.
- Add new model rows to `model_variant_index.json`.
- Add new ablation families or append completed items to existing families in
  `ablation_index.json`.
- Keep this matrix descriptive. The actual claim wording belongs in the paper
  evidence audit and manuscript-facing package docs.
