# NeurIPS Hybrid ResNet Paper Evidence Index

This is the durable outcome map for completed NeurIPS Hybrid ResNet backlog
items. It is intended to be the first stop after `docs/index.md` or
`docs/studies/index.md` when locating result summaries, artifact roots, and
claim boundaries.

Last synchronized from:

- backlog run state:
  `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/run_state.json`
- study summaries under:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/`
- done backlog queue:
  `docs/backlog/done/`

## Evidence Tiers

- `bounded_capped`: usable for bounded paper-supporting tables or figures only
  with the stated capped protocol. Not a full-training benchmark claim.
- `decision_support`: useful for model-selection, ablation, or routing
  decisions. Not paper-grade headline evidence by itself.
- `feasibility`: readiness, harness, integration, or protocol decision evidence
  that enables later result production.
- `superseded`: preserved for provenance but no longer the preferred source for
  a current result claim.

## Paper-Facing Authorities

| Backlog item | Phase / benchmark | Tier | Protocol / cap | Summary authority | Artifact root | Outcome / downstream use |
|---|---|---|---|---|---|---|
| `2026-04-29-cns-paper-contract-decision` | Phase 2 / PDEBench CNS | `feasibility` | `history_len=2`, `40` epochs, capped CNS lane | [CNS paper contract decision](pdebench_cns_paper_contract_decision.md) | decision document only | Locks the bounded CNS paper lane, normalization, recipe, row roster, authored-FFNO status, and claim boundary. |
| `2026-04-29-cns-paper-benchmark-rows` | Phase 2 / PDEBench CNS | `bounded_capped` | `512 / 64 / 64`, `history_len=2`, `40` epochs | [CNS paper row lock summary](pdebench_cns_paper_row_lock_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/` | Locks headline rows: `author_ffno_cns_base`, `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`; keeps `hybrid_resnet_cns` as continuity/support. |
| `2026-04-29-cns-paper-table-figure-bundle` | Phase 2 / PDEBench CNS | `superseded` | fallback `512 / 64 / 64`, `history_len=2`, `40` epochs | [CNS paper table/figure bundle summary](pdebench_cns_paper_table_figure_bundle_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/` | Historical fallback CNS bundle preserved for provenance. It assembled the first complete same-contract table/figure payload after the 1024 audit failed, but later same-contract 2048 promotion superseded it as the current discoverability target. |
| `2026-04-29-cns-paper-2048cap-row-extension` | Phase 2 / PDEBench CNS | `bounded_capped` | same-contract `2048 / 256 / 256`, `history_len=2`, `40` epochs | [CNS paper 2048cap extension summary](pdebench_cns_paper_2048cap_extension_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/` | Current capped CNS authority. Reuses the 2048cap spectral run, reruns FNO, U-Net strong, and authored FFNO under identical fairness settings, and preserves the 512cap bundle as historical provenance only. |
| `2026-04-29-cdi-lines128-paper-benchmark-harness` | Phase 3 / CDI Lines128 | `feasibility` | readiness-only `lines128`, fixed `seed=3` | [Lines128 paper benchmark harness summary](lines128_paper_benchmark_harness_summary.md) | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight/` | Builds the shared harness and paper-metric schema; supports minimum rows plus spectral and FFNO routing, but does not launch the full paper benchmark. |
| `2026-04-30-cdi-lines128-uno-design-preflight` | Phase 3 / CDI Lines128 U-NO extension | `feasibility` | preflight-only `lines128`, fixed `seed=3`, dummy `B=2` shape probe | [Lines128 U-NO preflight summary](lines128_uno_preflight_summary.md) | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/` | Freezes the external `neuralop.models.UNO` environment/API contract, records the required `uno_out_channels` / `uno_scalings` / nested `uno_n_modes` settings, confirms direct `B x 2 x 128 x 128` output acceptance, and hands off to generator integration without running rows. |
| `2026-04-27-cdi-ffno-generator-lines-best-config` | Phase 3 / CDI Lines128 | `decision_support` | `N=128`, grid-lines, `seed=3`, `40` epochs | [CDI FFNO generator best-config summary](cdi_ffno_generator_lines_best_config_summary.md) | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet/` | Fresh FFNO-vs-Hybrid prerequisite pair. Hybrid ResNet beat FFNO on amp/phase MAE, SSIM, and PSNR; still not the full Lines128 paper table. |
| `2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation` | Phase 3 / CDI Lines128 | `decision_support` | same-contract `lines128`, `seed=3`, `40` epochs | [Lines128 Hybrid skip/residual ablation summary](lines128_hybrid_resnet_skip_residual_ablation_summary.md) | `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation/` | Three fresh same-contract Hybrid ablations. Skip-add is the clearest phase-side trade, fixed residual scale is the clearest amplitude-side trade, and the combined row shows no constructive interaction. No row displaces the paper-grade `pinn_hybrid_resnet` anchor. |
| `2026-04-29-paper-evidence-package-audit` | Phase 2 + Phase 3 / cross-pillar paper package | `feasibility` | repo-local current-state audit over the locked CDI and CNS authorities | [Paper evidence package audit summary](paper_evidence_package_audit_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/` | Records what the manuscript can claim now, preserves CDI `paper_grade` versus CNS `capped_decision_support`, names draftable sections, and keeps blocked claims explicit without reopening the underlying row contracts. |

## Manuscript Incorporation Map

Current manuscript draft:
`docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

| Evidence source | Manuscript target |
|---|---|
| Complete Lines128 CDI bundle | `tab:cdi_lines128_complete`, `fig:cdi_main_qualitative` |
| Supervised FFNO extension | `tab:cdi_lines128_complete` |
| CNS paper table/figure bundle (`2048` authority; `512` historical fallback) | `tab:cns_bundle`, `fig:cns_sample_predictions` |
| Hybrid skip/residual ablation | `tab:cdi_skip_residual_ablation` |

## CNS Ablation And Baseline Evidence

| Backlog item | Phase / benchmark | Tier | Protocol / cap | Summary authority | Artifact root | Outcome / downstream use |
|---|---|---|---|---|---|---|
| `2026-04-21-pdebench-author-ffno-equal-footing-cns` | Phase 2 / PDEBench CNS | `decision_support` | fixed local CNS capped `10`/`40` epoch rows | [Author FFNO equal-footing summary](pdebench_author_ffno_equal_footing_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/` | Integrates the authored `fourierflow` FFNO baseline. The accepted `40`-epoch row later became the strongest locked CNS headline row: `relative_l2=0.0281477310`, `fRMSE_high=0.1210141182`. |
| `2026-04-22-pdebench-gnot-paper-default-cns-compare` | Phase 2 / PDEBench CNS | `decision_support` | fixed local CNS paper-default GNOT follow-up | [GNOT CNS compare summary](pdebench_gnot_cns_compare_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/` | Records GNOT integration and paper-default `40`-epoch follow-up. GNOT had lower high-frequency error but much worse aggregate error than the spectral anchor, so it was not promoted to headline CNS evidence. |
| `2026-04-21-pdebench-cns-markov-history1-compare` | Phase 2 / PDEBench CNS | `decision_support` | `history_len=1` versus frozen `history_len=2` | [Markov history-1 compare summary](pdebench_cns_markov_history1_compare_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/` | `history_len=1` did not help; it worsened spectral, hybrid, and FNO aggregate error at both `10` and `40` epochs, with only U-Net improving while staying last. |
| `2026-04-21-pdebench-cns-spectral-modes32-compare` | Phase 2 / PDEBench CNS | `decision_support` | spectral `12/12` versus `32/32`, capped `10`/`40` epochs | [Spectral modes-32 compare summary](pdebench_spectral_modes32_compare_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/` | Modes-32 improved high-frequency metrics at `40` epochs but lost aggregate error to the shared `12/12` spectral row; no default-profile promotion. |
| `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation` | Phase 2 / PDEBench CNS | `decision_support` | capped `512 / 64 / 64` pilots plus `1024 / 128 / 128` finalist confirmation | [Hybrid-spectral architecture ablation summary](pdebench_cns_hybrid_spectral_arch_ablation_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/` | Shared base recovered the aggregate lead on the `1024` confirmation slice; deeper shared-blocks10 remained a manual high-frequency follow-up, not a default promotion. |
| `2026-04-27-pdebench-ffno-convolutional-features-cns` | Phase 2 / PDEBench CNS | `decision_support` | local-conv FFNO-family CNS follow-up | [FFNO convolutional features CNS summary](pdebench_ffno_convolutional_features_cns_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/` | Local convolution materially improved the repo-local FFNO proxy and beat the local shared-spectral row at `40` epochs, while authored FFNO still had stronger high-frequency performance. |
| `2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap` | Phase 2 / PDEBench CNS | `decision_support` | `2048 / 256 / 256`, `40` epochs | [Hybrid-spectral 2048-cap scaling summary](pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/` | Base spectral remained the better aggregate larger-cap local reference; `2048` did not strengthen the case for deeper shared spectral promotion. |
| `2026-04-28-pdebench-cns-spectral-modes24-convergence-compare` | Phase 2 / PDEBench CNS | `decision_support` | `1024 / 128 / 128`, `80` epochs | [Spectral modes-24 convergence summary](pdebench_spectral_modes24_convergence_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes24-convergence-compare/` | Both `12/12` and `24/24` rows were still materially improving at stop; final metrics favored `12/12`, but the correct read is inconclusive rather than a clean convergence verdict. |
| `2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence` | Phase 2 / PDEBench CNS | `decision_support` | `1024 / 128 / 128`, shared-blocks10, `80` epochs | [Shared-blocks10 1024-cap longer-convergence summary](pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/` | Shows the old `40`-epoch shared-blocks10 row was materially under-converged; the `80`-epoch row improved every tracked metric but was still materially improving at stop, so no default promotion. |
| `2026-04-29-pdebench-cns-history-len3plus-compare` | Phase 2 / PDEBench CNS | `decision_support` | `history_len=3` pilots versus frozen `history_len=2`; `history_len=4` gate | [History length 3+ compare summary](pdebench_cns_history_len3plus_compare_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/` | `history_len=3` was mixed at `10` epochs and helped most controlled `40`-epoch rows, but the optional `history_len=4` gate stayed closed because the spectral row did not improve all gate metrics. |
| `2026-04-29-cns-spectral-history-len4plus-compare` | Phase 2 / PDEBench CNS | `decision_support` | spectral-only `history_len=4` plus conditional `history_len=5`, capped `10`/`40` epochs | [Spectral history length 4+ compare summary](pdebench_cns_spectral_history_len4plus_compare_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-history-len4plus-compare/` | Fresh spectral-only follow-up showed longer context kept improving the `40`-epoch capped row through `history_len=5`, while the `10`-epoch story stayed mixed; the result remains adjacent capped context only and does not change the locked `history_len=2` paper lane. |
| `2026-05-01-cns-author-ffno-history-length-study` | Phase 2 / PDEBench CNS | `decision_support` | authored-FFNO-only `history_len=3/4/5` versus frozen `history_len=2` anchor, `512 / 64 / 64`, `40` epochs | [Author FFNO history-length compare summary](pdebench_author_ffno_history_length_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-history-length-compare/` | Authored-FFNO `history_len=3` and `history_len=4` each strictly improved every recorded metric versus the previous step at `40` epochs; `history_len=5` slightly regressed aggregate error and `fRMSE_low` versus `history_len=4` while improving `fRMSE_mid` and `fRMSE_high`. Adjacent capped context only; does not reopen the locked `history_len=2` CNS paper lane. |
| `2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space` | Phase 2 / PDEBench CNS | `decision_support` | bounded shell-bridge probes | [Hybrid-spectral to FFNO parameter-space summary](pdebench_cns_hybrid_spectral_ffno_parameter_space_summary.md) | `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/` | Downsample and transpose decoder probes worsened promotion keys; the `40`-epoch promotion gate closed cleanly. |

## Maintenance Rules

- Add a row here whenever a NeurIPS backlog item moves to `docs/backlog/done/`
  and produces durable results, decisions, or readiness artifacts.
- Every row must name exactly one summary authority. The summary authority owns
  detailed metrics, validation commands, and artifact lists.
- Artifact roots should be stable directory roots, not transient logs, unless
  the item produced only a decision document.
- Do not relabel capped CNS evidence as `paper_grade` without a later
  full-training or explicitly approved paper-grade contract.
- If a later item supersedes an outcome, keep the older row and change its tier
  to `superseded`, then add the new authority row.
