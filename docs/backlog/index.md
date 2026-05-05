# Backlog Dependency Index

This index records the current dependency relations for the active NeurIPS
backlog queue. It is the human-readable companion to backlog frontmatter such as
`prerequisites` and `related_roadmap_phases`.

Current source of truth:

- strategic order: [steering.md](../steering.md)
- roadmap gates: [2026-04-20-neurips-hybrid-resnet-submission-roadmap.md](../plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md)
- deterministic active gate: [roadmap_gate.json](roadmap_gate.json)
- queue items: [`docs/backlog/active/`](active/), [`docs/backlog/in_progress/`](in_progress/), [`docs/backlog/done/`](done/), and [`docs/backlog/paused/`](paused/)

## Rules

- A backlog item gets a `prerequisites` entry only when another backlog item or
  completed tranche must land first.
- Reuse of existing CNS anchors, shell-lock decisions, or adapter support is
  treated as an already-satisfied study precondition, not as a backlog-to-backlog
  dependency.
- Parallel items on the same roadmap phase are explicitly listed as such so the
  selector does not invent a false serial order.
- `docs/backlog/roadmap_gate.json` is applied before provider selection. The
  current gate admits active NeurIPS evidence work across Phase 2 PDEBench,
  Phase 3 CDI, and `candidate-*` preflights. Relative selection should be
  expressed through backlog `priority` values and steering value rather than
  brittle one-off gate prefixes. Phase 2 PDEBench remains preferred by priority,
  Phase 3 CDI may be selected as useful parallel work, and candidate preflights
  may run concurrently without counting as completed Phase 2 or Phase 3
  evidence. If no active item satisfies the current window and the missing work
  is already authorized by the roadmap, the workflow may draft a new active
  backlog item instead of selecting later-phase work.

## Current Queue Graph

| Item | Status | Roadmap Phase | Dependency Relation | Notes |
|---|---|---|---|---|
| [2026-04-29-pdebench-cns-history-len3plus-compare.md](done/2026-04-29-pdebench-cns-history-len3plus-compare.md) | `done` | `phase-2-pdebench-128x128-image-suite` | Depends on completed Markov history-1 compare | Completed controlled longer-context CNS ablation; remains capped decision-support context, not full-training benchmark evidence. |
| [2026-04-29-cns-spectral-history-len4plus-compare.md](active/2026-04-29-cns-spectral-history-len4plus-compare.md) | `active` | `phase-2-pdebench-128x128-image-suite` | Depends on completed history_len=3 comparison | Tests whether the spectral-bottleneck CNS row, currently labeled `SRU-Net*` in the manuscript, continues improving beyond history length 3 and quantifies the deltas without changing the headline CNS bundle. |
| [2026-05-01-cns-author-ffno-history-length-study.md](active/2026-05-01-cns-author-ffno-history-length-study.md) | `active` | `phase-2-pdebench-128x128-image-suite` | Depends on completed authored-FFNO equal-footing row and completed SRU-Net longer-history context | Tests whether the authored FFNO CNS row benefits from additional temporal context, while keeping longer-history FFNO rows out of the locked same-history CNS headline table unless a later roadmap-level paper-contract decision reopens that lane. |
| [2026-05-04-cns-history5-comparator-gap-fill.md](in_progress/2026-05-04-cns-history5-comparator-gap-fill.md) | `in_progress` | `phase-2-pdebench-128x128-image-suite` | Depends on completed authored-FFNO and spectral h5 studies | Runs missing h5 FNO and U-Net comparator rows so h5 CNS result tables can be assembled without mixing history lengths. |
| [2026-05-04-cns-matched-condition-table-refresh.md](active/2026-05-04-cns-matched-condition-table-refresh.md) | `active` | `phase-2-pdebench-128x128-image-suite` | Depends on h5 comparator gap fill and completed 2048-cap h2 authority | Refreshes the manuscript CNS headline table so it uses one matched condition; best-observed mixed rows remain context only. |
| [2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space.md](done/2026-04-28-pdebench-cns-hybrid-spectral-ffno-parameter-space.md) | `done` | `phase-2-pdebench-128x128-image-suite` | Depends on completed Hybrid-spectral architecture and CNS FFNO-conv follow-ups | Completed Phase 2 CNS-only parameter-space study under the capped decision-support contract. |
| [2026-04-29-cns-paper-contract-decision.md](done/2026-04-29-cns-paper-contract-decision.md) | `done` | `phase-2-pdebench-128x128-image-suite` | No active backlog prerequisite | Completed the bounded capped CNS paper-evidence contract decision. |
| [2026-04-29-cns-paper-benchmark-rows.md](done/2026-04-29-cns-paper-benchmark-rows.md) | `done` | `phase-2-pdebench-128x128-image-suite` | Depends on completed CNS paper contract decision | Locked the bounded CNS row bundle for downstream table/figure assembly. |
| [2026-04-29-cns-paper-table-figure-bundle.md](in_progress/2026-04-29-cns-paper-table-figure-bundle.md) | `in_progress` | `phase-2-pdebench-128x128-image-suite` | Depends on completed CNS paper benchmark rows | Builds the CNS JSON/CSV/TeX table, fixed-sample field visuals, source-array manifest, and claim-boundary summary; it should upgrade the immediate bundle to `1024 / 128 / 128` when same-cap rows can be recovered or run, without mixing caps. |
| [2026-04-27-cdi-ffno-generator-lines-best-config.md](done/2026-04-27-cdi-ffno-generator-lines-best-config.md) | `done` | `phase-3-cdi-anchor-regeneration` | CDI/ptycho generator lane; no active backlog prerequisite | Completed CDI FFNO generator baseline work. It unlocks CDI FFNO-dependent lanes but does not satisfy the Phase 2 PDEBench evidence gate. |
| [2026-04-29-cdi-lines128-paper-benchmark-harness.md](done/2026-04-29-cdi-lines128-paper-benchmark-harness.md) | `done` | `phase-3-cdi-anchor-regeneration` | No active backlog prerequisite | Completed shared Lines128 paper-benchmark wrapper/harness, contract-reconstruction preflight, FNO/seed decision manifest, and metric-schema downgrade gate. |
| [2026-04-29-cdi-lines128-minimum-paper-table.md](done/2026-04-29-cdi-lines128-minimum-paper-table.md) | `done` | `phase-3-cdi-anchor-regeneration` | Depends on completed Lines128 paper benchmark harness | Completed the minimum paper-grade CDI subset and visual bundle for `hybrid_resnet`, paired CDI `cnn` U-Net-class supervised and PINN rows, and selected FNO comparator, with labels aligned to but distinct from CNS `unet_strong`. |
| [2026-04-29-cdi-lines128-paper-benchmark-execution.md](done/2026-04-29-cdi-lines128-paper-benchmark-execution.md) | `done` | `phase-3-cdi-anchor-regeneration` | Depends on completed CDI FFNO generator baseline, completed Lines128 harness, and the completed minimum CDI table | Completed the Lines128 CDI benchmark table by extending the minimum subset with required `spectral_resnet_bottleneck_net` and FFNO rows. |
| [2026-04-29-cdi-lines128-supervised-equivalent-rows.md](done/2026-04-29-cdi-lines128-supervised-equivalent-rows.md) | `done` | `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution | Completed the required supervised FFNO CDI control row as the same-contract training-procedure comparator for the complete FFNO + PINN Lines128 row. |
| [2026-04-30-cdi-lines128-uno-design-preflight.md](done/2026-04-30-cdi-lines128-uno-design-preflight.md) | `done` | `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution and supervised FFNO extension | Completed the `ptycho311` / `neuraloperator==2.0.0` U-NO environment/API preflight and froze `neuralop_uno` constructor defaults. |
| [2026-04-30-cdi-lines128-uno-generator-integration.md](done/2026-04-30-cdi-lines128-uno-generator-integration.md) | `done` | `phase-3-cdi-anchor-regeneration` | Depends on U-NO design preflight | Completed `neuralop_uno` generator integration and proved both PINN and supervised paths use the external NeuralOperator U-NO body. |
| [2026-04-30-cdi-lines128-uno-table-extension.md](done/2026-04-30-cdi-lines128-uno-table-extension.md) | `done` | `phase-3-cdi-anchor-regeneration` | Depends on U-NO generator integration and complete Lines128 CDI benchmark execution | Completed the append-only U-NO table extension without rerunning existing rows. |
| [2026-05-04-cdi-natural-patch-fixedprobe-dataset.md](active/2026-05-04-cdi-natural-patch-fixedprobe-dataset.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution | Creates and locks `natural_patches128_fixedprobe_v1`, a fixed-probe natural-image-patch CDI dataset capped at 10000 object images. |
| [2026-05-04-cdi-natural-patch-expanded-benchmark.md](active/2026-05-04-cdi-natural-patch-expanded-benchmark.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on natural-patch fixed-probe dataset, complete Lines128 CDI benchmark, and U-NO extension | Runs a standalone expanded-object CDI benchmark on the frozen natural-patch dataset without rewriting the Lines128 authority. |
| [2026-05-04-cdi-lines128-multiseed-headline-robustness.md](active/2026-05-04-cdi-lines128-multiseed-headline-robustness.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution and U-NO table extension | Adds training-seed robustness for the headline CDI table by reusing completed seed roots and running only missing seeds under the fixed Lines128 contract. |
| [2026-05-05-paper-efficiency-table.md](active/2026-05-05-paper-efficiency-table.md) | `active` | `phase-2-pdebench-128x128-image-suite`, `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution, U-NO table extension, and matched-condition CNS table refresh | Compiles the repo-local paper efficiency table from existing parameter/runtime evidence, runs only missing lightweight inference probes if needed, and keeps heterogeneous runtime fields caveated by source contract. |
| [2026-05-04-cdi-lines128-srunet-branch-objective-ablation.md](active/2026-05-04-cdi-lines128-srunet-branch-objective-ablation.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution, supervised FFNO extension, U-NO table extension, and completed encoder-fusion variants | Append-only SRU-Net mechanism ablation: runs conv-only and spectral-only encoder-branch rows plus the missing supervised SRU-Net objective-control row, without rerunning completed table rows. |
| [2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation.md](active/2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution and completed skip/residual ablation context | Append-only SRU-Net bottleneck-family ablation: adds only a ConvNeXt-style bottleneck row and compares it with the completed SRU-Net bottleneck row by lineage. |
| [2026-05-05-srunet-ptychoblock-ffno-encoder-cdi-cns-smallcap.md](active/2026-05-05-srunet-ptychoblock-ffno-encoder-cdi-cns-smallcap.md) | `active` | `phase-2-pdebench-128x128-image-suite`, `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution, completed SRU-Net branch/objective ablation, and completed CNS matched-condition table refresh | Cross-pillar append-only SRU-Net encoder mechanism ablation: implements one `2x(PtychoBlock + downsample) -> FFNO` encoder variant while preserving the SRU-Net shell/skip/body settings, then runs only the new CDI Lines128 and CNS small-cap rows. |
| [2026-04-27-hybrid-spectral-ffno-parameter-space-cdi.md](active/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on completed CDI FFNO generator baseline | Phase 3 CDI-only split of the former mixed CNS/CDI parameter-space item. The dependency is now satisfied, so selection is governed by roadmap value rather than prerequisite blocking. |
| [2026-04-21-hybrid-resnet-encoder-fusion-variants.md](active/2026-04-21-hybrid-resnet-encoder-fusion-variants.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution | Narrow Hybrid ResNet encoder-fusion ablation that tests learned encoder update scaling and spectral/local branch gates against the fixed N=128 grid-lines contract, without changing probe, loss, bottleneck, or decoder settings. |
| [2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation.md](active/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation.md) | `active` | `phase-3-cdi-anchor-regeneration` | Depends on complete Lines128 CDI benchmark execution | Same-contract Lines128 Hybrid ResNet ablation isolating decoder skip connections and bottleneck residual scaling, with append-only cross-references to existing skip/mode, CNS skip-add, and encoder-fusion evidence surfaces. |
| [2026-04-29-paper-evidence-package-audit.md](active/2026-04-29-paper-evidence-package-audit.md) | `active` | `phase-2-pdebench-128x128-image-suite`, `phase-3-cdi-anchor-regeneration` | Depends on minimum CDI table and CNS table/figure bundle | Creates the repo-local paper evidence manifest and completeness audit without creating the Phase 5 paper-facing evidence map. |
| [2026-04-29-cns-paper-2048cap-row-extension.md](active/2026-04-29-cns-paper-2048cap-row-extension.md) | `active` | `phase-2-pdebench-128x128-image-suite` | Depends on paper evidence package audit | Later long-running CNS evidence-strengthening pass for same-cap `2048 / 256 / 256` FFNO/FNO/U-Net rows; not a blocker for the current 1024-cap table/figure bundle. |
| [2026-04-29-brdt-operator-validation.md](active/2026-04-29-brdt-operator-validation.md) | `active` | `candidate-brdt-preflight` | No active backlog prerequisite | First BRDT execution item. Implements and validates the differentiable Born operator with independent checks before any dataset or training evidence is trusted. |
| [2026-04-29-brdt-dataset-preflight.md](active/2026-04-29-brdt-dataset-preflight.md) | `active` | `candidate-brdt-preflight` | Depends on BRDT operator validation | Locks physical `q`, train-only normalization, smoke dataset generation, split policy, and dataset manifest. |
| [2026-04-29-brdt-task-adapters.md](active/2026-04-29-brdt-task-adapters.md) | `active` | `candidate-brdt-preflight` | Depends on BRDT dataset preflight | Adds task-local loaders/adapters/training surfaces without registering BRDT as a CDI generator. |
| [2026-04-29-brdt-four-row-preflight.md](active/2026-04-29-brdt-four-row-preflight.md) | `active` | `candidate-brdt-preflight` | Depends on BRDT task adapters | Runs the bounded four-row decision-support preflight: classical Born, U-Net, FNO vanilla, and SRU/Hybrid-family. |
| [2026-05-04-brdt-physics-only-objective-ablation.md](active/2026-05-04-brdt-physics-only-objective-ablation.md) | `active` | `candidate-brdt-preflight` | Depends on completed BRDT four-row preflight | High-priority append-only BRDT diagnostic that reruns U-Net, FNO vanilla, and Hybrid ResNet with only `relative_physics_L2(A(q_pred_phys), observed_sinogram)` active, to test whether sparse-target collapse is objective-induced. |
| [2026-05-04-brdt-ffno-row-extension.md](active/2026-05-04-brdt-ffno-row-extension.md) | `active` | `candidate-brdt-preflight` | Depends on completed BRDT four-row preflight | Adds one append-only FFNO row against the fixed BRDT decision-support contract without rerunning the existing classical, U-Net, FNO vanilla, or Hybrid ResNet rows. |
| [2026-04-29-brdt-preflight-summary-promotion-decision.md](active/2026-04-29-brdt-preflight-summary-promotion-decision.md) | `active` | `candidate-brdt-preflight` | Depends on BRDT four-row preflight | Writes the durable BRDT summary and recommends promotion, deferral, or rejection without adding paper claims. |
| [2026-04-29-brdt-candidate-preflight.md](active/2026-04-29-brdt-candidate-preflight.md) | `active` | `candidate-brdt-preflight` | Depends on all split BRDT preflight items | Umbrella closeout for discoverability only; it prevents the broad BRDT design from being selected as a vague implementation scope. |
| [2026-04-29-wavebench-inverse-source-preflight.md](active/2026-04-29-wavebench-inverse-source-preflight.md) | `active` | `candidate-wavebench-inverse-source-preflight` | No active backlog prerequisite | Reprioritized on 2026-04-30 as the next candidate lane to attempt before remaining optional U-NO table-extension work. It remains candidate work, not a replacement paper pillar. |
| [2026-04-29-wavebench-native-baseline-reproduction.md](active/2026-04-29-wavebench-native-baseline-reproduction.md) | `active` | `wavebench-additional-inverse-wave-extension` | Depends on WaveBench inverse-source preflight | Reproduces/evaluates native WaveBench FNO and U-Net references only if the inverse-source preflight authorizes supervised follow-up. |
| [2026-04-29-wavebench-forward-model-physics-validation.md](active/2026-04-29-wavebench-forward-model-physics-validation.md) | `active` | `wavebench-additional-inverse-wave-extension` | Depends on WaveBench inverse-source preflight | Validates the differentiable inverse-source forward map before any WaveBench row may be called physics-informed. |
| [2026-04-29-wavebench-shared-encoder-supervised-benchmark.md](active/2026-04-29-wavebench-shared-encoder-supervised-benchmark.md) | `active` | `wavebench-additional-inverse-wave-extension` | Depends on WaveBench inverse-source preflight and native baseline status | Runs the first repo-local supervised shared-encoder WaveBench inverse-source comparison if the preflight and native baseline gates pass. |
| [2026-04-29-wavebench-hybrid-physics-rows.md](active/2026-04-29-wavebench-hybrid-physics-rows.md) | `active` | `wavebench-additional-inverse-wave-extension` | Depends on shared-encoder supervised results and forward-model validation | Adds physics-informed or hybrid supervised-plus-physics inverse-source rows only if the forward-model gate passes. |
| [2026-04-29-wavebench-paper-table-figure-bundle.md](active/2026-04-29-wavebench-paper-table-figure-bundle.md) | `active` | `wavebench-additional-inverse-wave-extension` | Depends on supervised WaveBench results and evidence-lane approval | Assembles WaveBench inverse-source tables and figures only after the lane is approved for paper evidence. |
| [2026-04-29-paper-facing-evidence-index.md](paused/2026-04-29-paper-facing-evidence-index.md) | `paused` | `phase-5-paper-facing-evidence-bundle` | Depends on paper evidence package audit | Creates `/home/ollie/Documents/neurips/index.md` and the evidence checklist only after the roadmap reaches Phase 5. |
| [2026-04-29-manuscript-draft-continuity.md](paused/2026-04-29-manuscript-draft-continuity.md) | `paused` | `phase-5-paper-facing-evidence-bundle` | Depends on paper-facing evidence index | Requires future manuscript-drafting tasks to start from `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex` so they reuse the existing introduction, context, methods, benchmark framing, result framing, and associated reconstruction figure instead of reinventing them. |
| [2026-03-13-lines256-experiment-history-summary-input.md](active/2026-03-13-lines256-experiment-history-summary-input.md) | `active` | n/a | Separate non-NeurIPS queue branch | Does not block or unlock the CNS benchmark queue. |

## Implications For Selection

- The immediate roadmap-consistent Phase 2 CNS queue now centers on the
  paper-evidence chain: completed contract decision, completed bounded row lock,
  then the active table/figure bundle. The current bundle may upgrade to a
  same-contract `1024 / 128 / 128` package, but it must not mix caps in the
  headline table.
- The history-length-beyond-2 item is complete. It should remain context for
  the CNS paper-contract decision and later row locking, not a prerequisite for
  CDI work and not full-training benchmark evidence. The new history-length-4+
  spectral follow-up is a focused context-ablation item for the
  `spectral_resnet_bottleneck_base` / `SRU-Net*` row only; it should quantify
  whether improvement continues beyond three frames, not rewrite the current
  same-history CNS headline table.
- The authored-FFNO history-length item is the analogous temporal-context
  ablation for `author_ffno_cns_base`. It should start from the locked
  `history_len=2` authored-FFNO anchor and gate longer histories from observed
  deltas. It must not be used to mix FFNO longer-history rows with
  `history_len=2` FNO/U-Net/SRU-Net rows in a headline model-ranking table
  unless a later roadmap-level paper-contract decision reopens the CNS history
  lane.
- The h5 comparator gap-fill is the prerequisite for the matched-condition CNS
  table refresh. The refresh item should pick one complete headline condition:
  all-row h5 if the full roster is complete under one condition, otherwise the
  locked h2 `2048 / 256 / 256` authority. It should preserve best-observed
  mixed rows as context only, not as the headline model-ranking table.
- The CNS paper-contract decision is the gate that prevents capped
  decision-support rows and full-training benchmark rows from being mixed into
  one claim. It also sets the authored-FFNO inclusion cutoff and claim impact.
  The row-lock and table/figure items depend on that decision.
- The `2048 / 256 / 256` CNS all-row extension is deliberately later than the
  current table/figure bundle and paper evidence audit because the missing
  comparator rows are expected to be long-running. It should not be selected
  merely because the immediate 1024-cap table bundle is active.
- The FFNO-as-CDI-generator lane is complete. It remains separate from CNS and
  does not close any remaining Phase 2 PDEBench evidence requirement.
- The Lines128 paper-quality CDI harness is complete. The minimum
  `hybrid_resnet`/paired-CDI-cnn-U-Net-class/FNO paper subset can proceed, and
  the complete CDI execution item is now blocked only by the minimum CDI table,
  not by the FFNO generator or harness prerequisites.
- The classical CDI item is optional paper-strengthening work. It must return a
  protocol-compatible row or an explicit incompatibility note; it should not
  force a solver into a misleading same-protocol comparison.
- The remaining Lines128 execution items are Phase 3 CDI work. They are allowed
  by the current deterministic gate as CDI-track work, but they do not count as
  Phase 2 PDEBench evidence and should not be used to close the PDEBench
  image-suite gate.
- The supervised Lines128 CDI extension completed the required supervised FFNO
  row downstream of the complete PINN-trained CDI table. The CDI `cnn`
  U-Net-class supervised row belongs to the minimum CDI table next to the
  matching CDI `cnn` PINN row.
- The U-NO Lines128 extension completed as an append-only Phase 3 CDI
  comparator lane. The multi-seed Lines128 robustness item is Phase 3
  evidence-strengthening for the headline CDI table: it should reuse completed
  seed roots by lineage, add only missing seeds under the fixed contract, and
  report per-seed values plus mean/std without changing the object distribution.
  Expanded-object CDI work is now a separate two-item Phase 3 chain: the
  natural-patch fixed-probe dataset item must first freeze
  `natural_patches128_fixedprobe_v1` with at most `10_000` object images,
  source/probe/split manifests, and git-ignored generated arrays; only then may
  the expanded benchmark item run model rows against that frozen dataset. The
  expanded benchmark should create a standalone natural-patch table, not rewrite
  the completed `lines128` table authority.
  The SRU-Net branch/objective ablation is a separate append-only Phase 3 CDI
  mechanism item: it should run only the missing conv-only encoder,
  spectral-only encoder, and supervised SRU-Net rows so the paper can
  distinguish local spatial features, global spectral coupling, and
  physics-consistency training. It must not rerun or rewrite the completed base
  Lines128 or U-NO table bundles, and it does not become required Phase 3
  completion work.
  The paper efficiency table item is a cross-pillar packaging pass that remains
  inside the current repo-local evidence package rather than Phase 5
  paper-facing artifact assembly. It may compile parameter counts and
  provenance-backed training runtime from existing CDI/CNS artifacts, and may
  run only lightweight missing inference probes. It must not rerun training or
  present heterogeneous launch runtimes as a normalized throughput benchmark
  without an explicit common measurement protocol.
- The SRU-Net ConvNeXt bottleneck ablation is a separate append-only Phase 3
  CDI mechanism item. It should add only the ConvNeXt-style bottleneck variant,
  preserve the completed Lines128 contract, and compare against the existing
  SRU-Net bottleneck row by lineage. It must not be merged into the encoder
  branch/objective ablation because that would mix two mechanisms.
- The SRU-Net PtychoBlock-to-FFNO encoder ablation is a bounded cross-pillar
  mechanism item. It should preserve the SRU-Net shell outside the encoder,
  launch only the new Lines128 CDI row and a small-cap CNS row, and compare each
  benchmark against completed lineage rows without mixing CDI metrics with CNS
  capped decision-support metrics. Its priority is intentionally higher than
  the remaining active WaveBench items per 2026-05-05 steering.
- The Hybrid ResNet encoder-fusion variants item is active Phase 3 CDI
  architecture-ablation work. It should test encoder update scaling and
  spectral/local branch gates against the fixed N=128 grid-lines contract, but
  it is not a prerequisite for the completed Lines128 bundle or the U-NO
  append-only extension unless a later reviewed plan explicitly promotes it.
- The Hybrid ResNet skip/residual ablation is a separate append-only Phase 3 CDI
  architecture-context item. It should isolate decoder skip connections and
  bottleneck residual scaling on the fixed Lines128 contract, cross-reference
  prior skip/mode and CNS skip-add evidence without treating those as CDI
  evidence, and avoid rerunning or rewriting completed benchmark bundles.
- The broader Hybrid-spectral-to-FFNO parameter-space study has been split by
  roadmap phase. The CNS half is complete; the CDI half is active with its CDI
  FFNO generator prerequisite satisfied.
- BRDT and WaveBench inverse source are active concurrent `candidate-*`
  preflights. Steering on 2026-04-30 raised WaveBench inverse source ahead of
  remaining optional U-NO table-extension work; this does not promote it into a
  CDI/CNS replacement pillar or authorize paper claims. BRDT is split into operator,
  dataset, adapter, four-row, summary, and umbrella-closeout items so the
  workflow cannot satisfy the candidate lane with a vague single preflight. The
  BRDT physics-only objective ablation is a high-priority append-only candidate
  diagnostic for the completed four-row bundle; it should run only U-Net, FNO
  vanilla, and Hybrid ResNet under the pure relative-physics objective and
  compare against existing rows without overwriting them. The BRDT FFNO row
  extension is an append-only candidate follow-up to the completed four-row
  bundle; it should reference the original bundle and add only the new FFNO row.
  These items do not close CDI, CNS, or paper-evidence-package gates.
  Do not add one-off gate prefixes for either candidate; use priority and
  steering.
- The Markov history-1, modes-32, Hybrid-spectral architecture, FFNO local-conv,
  paper-default GNOT, author-FFNO equal-footing, modes-24 convergence,
  2048-cap scaling, shared-blocks10 longer-convergence, history-length-beyond-2,
  CNS Hybrid-spectral/FFNO parameter-space, CDI FFNO generator, and Lines128
  paper harness lanes are completed. The history-length-4+ spectral follow-up is
  active as a narrow Phase 2 evidence-strengthening item.
- For the independently runnable lanes, the selector should choose between them
  using steering and roadmap value, not a fabricated prerequisite chain.
- The paper-facing evidence-index and manuscript-draft-continuity items are
  intentionally paused because Phase 5 is outside the current deterministic
  gate and writes outside the current evidence-generation workflow. Move them
  only when the roadmap reaches the evidence-bundle and manuscript-drafting
  phase.
- If a future backlog item truly requires another backlog item to land first,
  add that item id to `prerequisites` and update this index in the same patch.
