# NeurIPS Hybrid ResNet Evidence Matrix

Status: draft  
Last updated: 2026-04-30

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
| CDI FFNO supervised extension | `lines128_supervised_equivalent_rows_summary.md` | paper-complete extension to existing table |
| CDI U-NO extension readiness | `lines128_uno_preflight_summary.md` | feasibility-only external UNO environment/API authority before generator integration |
| PDEBench CNS table/figures | `pdebench_cns_paper_table_figure_bundle_summary.md` | bounded capped decision-support only |
| Cross-pillar claim audit | `paper_evidence_package_audit_summary.md` | preserves CDI/CNS claim asymmetry |
| Completed backlog outcome map | `paper_evidence_index.md` | first-stop durable outcome index |

## CDI Lines128 Model Matrix

Fixed contract: synthetic grid-lines `N=128`, `gridsize=1`, `seed=3`,
`40` epochs, custom Run1084 probe, `pad_extrapolate`, train/test images `2/2`.

| Row | Architecture | Training | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Source |
|---|---|---:|---:|---:|---:|---:|---|
| `baseline` | `cnn` | supervised | 0.402672 | 0.786194 | 0.277214 | 0.496353 | complete CDI table |
| `pinn` | `cnn` | PINN | 0.123171 | 0.786322 | 0.732568 | 0.263815 | complete CDI table |
| `pinn_hybrid_resnet` | `hybrid_resnet` | PINN | 0.026939 | 0.072063 | 0.988114 | 0.994740 | complete CDI table |
| `pinn_fno_vanilla` | `fno_vanilla` | PINN | 0.124816 | 0.143540 | 0.740936 | 0.933464 | complete CDI table |
| `pinn_spectral_resnet_bottleneck_net` | `spectral_resnet_bottleneck_net` | PINN | 0.024944 | 0.092881 | 0.989855 | 0.972219 | complete CDI table |
| `pinn_ffno` | `ffno` | PINN | 0.062772 | 0.082839 | 0.934830 | 0.981592 | complete CDI table / prerequisite FFNO pair |
| `supervised_ffno` | `ffno` | supervised | 0.386413 | 0.046563 | 0.248427 | 0.937179 | supervised FFNO extension |

CDI artifact roots:

- Complete table:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Minimum table preserved root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- FFNO prerequisite pair:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
- Supervised FFNO extension:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`

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

## PDEBench CNS Model Matrix

Fixed paper bundle contract: PDEBench 2D_CFD CNS `128x128`,
`history_len=2`, `512 / 64 / 64`, `40` epochs, `mse`, capped
decision-support boundary.

| Row | Architecture | Training | relative_l2 | fRMSE_high | Role | Source |
|---|---|---:|---:|---:|---|---|
| `author_ffno_cns_base` | authored FFNO | supervised | 0.028148 | 0.121014 | headline | CNS table bundle |
| `spectral_resnet_bottleneck_base` | spectral bottleneck | supervised | 0.061562 | 0.434933 | headline | CNS table bundle |
| `fno_base` | FNO | supervised | 0.074099 | 0.671772 | headline | CNS table bundle |
| `unet_strong` | U-Net | supervised | 0.675798 | 1.332625 | headline | CNS table bundle |
| `hybrid_resnet_cns` | Hybrid ResNet | supervised | 0.064418 | 0.368307 | continuity | CNS table bundle |

CNS table/figure bundle root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle`

Bundle outputs include:

- `cns_paper_table_rows.json`
- `cns_paper_table_rows.csv`
- `cns_paper_table_rows.tex`
- fixed-sample prediction/error figures
- `shared_field_scales.json`
- `fixed_sample_manifest.json`
- `1024_same_cap_audit.json`

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
| `2026-04-27-cdi-ffno-generator-lines-best-config` | CDI FFNO prerequisite pair | `cdi_ffno_generator_lines_best_config_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet` |
| `2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space` | CNS shell-bridge ablation | `pdebench_cns_hybrid_spectral_ffno_parameter_space_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/` |
| `2026-04-29-cdi-lines128-paper-benchmark-harness` | CDI harness readiness | `lines128_paper_benchmark_harness_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight/` |
| `2026-04-29-cns-paper-contract-decision` | CNS contract lock | `pdebench_cns_paper_contract_decision.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_same_contract_audit.json` |
| `2026-04-29-cns-paper-benchmark-rows` | CNS row lock | `pdebench_cns_paper_row_lock_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/` |
| `2026-04-29-cns-paper-table-figure-bundle` | CNS table/figure bundle | `pdebench_cns_paper_table_figure_bundle_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/` |
| `2026-04-29-cdi-lines128-minimum-paper-table` | CDI minimum table | `lines128_minimum_paper_table_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z` |
| `2026-04-29-cdi-lines128-paper-benchmark-execution` | CDI complete six-row table | `lines128_paper_benchmark_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux` |
| `2026-04-29-cdi-lines128-supervised-equivalent-rows` | CDI supervised FFNO extension | `lines128_supervised_equivalent_rows_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z` |
| `2026-04-27-hybrid-spectral-ffno-parameter-space-cdi` | CDI Hybrid/spectral to FFNO bridge study | `cdi_hybrid_spectral_ffno_parameter_space_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi` |
| `2026-04-30-cdi-lines128-uno-design-preflight` | CDI U-NO environment/API readiness | `lines128_uno_preflight_summary.md` | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/` |
| `2026-04-29-paper-evidence-package-audit` | cross-pillar evidence audit | `paper_evidence_package_audit_summary.md` | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/` |
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
| CDI FFNO training mode | supervised FFNO extension | supervised FFNO improves phase MAE but loses badly on amplitude quality |
| CDI Hybrid/spectral to FFNO bridge | hybrid-spectral/FFNO parameter-space | no fresh bridge row beat the existing anchors; DS1 is a phase-leaning trade only |
| CDI U-NO extension readiness | U-NO design preflight | external `UNO` imports cleanly in `ptycho311`; later integration must preserve the frozen `uno_out_channels`, nested `uno_n_modes`, `uno_scalings`, and direct real/imag output contract |
| CDI FFNO generator prerequisite pair | FFNO vs Hybrid pair | prerequisite evidence, superseded for table claims by complete bundle |
| CNS paper contract and bundle | contract, row lock, table/figure bundle | bounded capped CNS table only; no full-training SOTA claim |
| CNS authored FFNO | author FFNO equal-footing row | strongest locked capped CNS row |
| CNS GNOT | paper-default GNOT compare | not promoted because aggregate error was poor |
| CNS history length | history1 and history3+ compares | history1 bad; history3 adjacent context only |
| CNS spectral modes | modes32 and modes24 compares | no higher-mode default promotion |
| CNS hybrid-spectral architecture | sharing/depth, 2048-cap, shared-blocks10 longer convergence | deeper variants remain decision-support |
| CNS FFNO-family local convolution | local-conv follow-up | improves repo-local FFNO proxy but not stronger than authored FFNO |
| CNS shell bridge | hybrid-spectral/FFNO parameter-space | promotion gate closed |
| Legacy CDI lines256 search | Stage A-E skip/mode search | historical architecture context only |

## Maintenance

- Add new completed backlog outputs here when a backlog item moves to
  `docs/backlog/done/` and produces metrics, figures, contract decisions,
  harnesses, audits, or durable readiness outputs.
- Add new model rows to `model_variant_index.json`.
- Add new ablation families or append completed items to existing families in
  `ablation_index.json`.
- Keep this matrix descriptive. The actual claim wording belongs in the paper
  evidence audit and manuscript-facing package docs.
