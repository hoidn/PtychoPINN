# NeurIPS Hybrid ResNet Submission Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a triaged NeurIPS 2026 submission evidence package for the Hybrid ResNet architecture, with `128x128` CDI as the anchor and a required compact native `128x128` PDEBench image-suite contribution.

**Architecture:** Run studies from `/home/ollie/Documents/PtychoPINN/`, regenerate the missing paper-grade `128x128` CDI Hybrid ResNet anchor, execute a scoped PDEBench `128x128` image suite covering SWE, Darcy Flow, and 2D Compressible Navier-Stokes, and eventually write paper-facing artifacts under `/home/ollie/Documents/neurips/` with `index.md` as the top-level evidence map. The `/home/ollie/Documents/neurips/index.md` file is a planned Phase 5 output and is not expected to exist during earlier tranches. The roadmap is deadline- and compute-constrained: NeurIPS full paper deadline is 2026-05-06 AOE, and the assumed compute budget is one RTX 3090 for several days.

**Tech Stack:** Python 3.11 in the `ptycho311` environment, PyTorch/Lightning, existing PtychoPINN grid-lines and Torch study runbooks, optional external PDE benchmark dependencies, Markdown/JSON/CSV/TeX artifact sources.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Title: Hybrid ResNet NeurIPS 2026 Submission Campaign
- Status: pending
- Design/Source: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Experiment root: `/home/ollie/Documents/PtychoPINN/`
- Future artifact root: `/home/ollie/Documents/neurips/` (created/populated during Phase 5; it may be absent before then)

## Compliance Matrix

- [ ] **Repo Guidance:** `CLAUDE.md` - read `docs/index.md`, keep plans under `docs/plans/`, use tmux for long-running commands, use `ptycho311` for long-running PtychoPINN workflows, do not create worktrees.
- [ ] **Baseline Authority:** `docs/model_baselines.md` - inherit the recommended Hybrid ResNet baseline unless a study explicitly overrides it.
- [ ] **Study Discovery:** `docs/studies/index.md` - reuse existing study runbooks and artifact contracts where possible.
- [ ] **Known Findings:** `docs/findings.md` - obey active PyTorch/grid-lines findings, including probe-mask and reassembly contracts.
- [ ] **Evidence Index Maintenance:** Result-producing NeurIPS work must consult and update the durable evidence indexes before it is marked complete.
- [ ] **Artifact Policy:** Do not write manuscript prose or generated paper-facing artifacts until the roadmap reaches the artifact assembly phase.

## Context Priming

Before executing any phase, read:

- `docs/index.md`
- `CLAUDE.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/workflows/pytorch.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/findings.md`
- `docs/litsurvey.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`

For non-ML CDI comparator context, also read:

- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- `scripts/reconstruction/hio_cdi_benchmark.py`

## Evidence Index Maintenance Policy

Any NeurIPS backlog item that trains, evaluates, packages, or audits a model
variant, ablation, hyperparameter study, or paper-facing run must consult the
project evidence indexes before implementation closes:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` for
  paper-facing outputs, claim-boundary decisions, or completed backlog outcomes

Completion requires one of:

- add or update `model_variant_index.json` for any new or changed evaluated
  model row;
- add or update `ablation_index.json` for any ablation, hparam study,
  architecture/config probe, harness, audit, or decision output;
- add or update `evidence_matrix.md` for every durable generated output that a
  later manuscript/table/planning task should discover, including visual
  bundles, figure manifests, figure source arrays, and paper-facing rendered
  panels;
- for paper-facing evidence, add or update `paper_evidence_index.md`, or
  explicitly point to the authoritative paper-evidence summary that already
  owns the row;
- if no index update is applicable, state that explicitly in the execution
  report with the reason.

Backlog plans and implementation reviews should reject result-producing items
that leave these surfaces stale. This policy applies to append-only extensions
as well as fresh runs; promoting existing rows into a new table or bundle still
requires the index lineage to be current.

## Phase 0 - Evidence Inventory and Regeneration Plan

Purpose: know what can be reused before launching expensive runs, and convert the lost `128x128` anchor condition into an explicit regeneration plan.

- [ ] 0.1: Inventory surviving `128x128` Hybrid ResNet grid-lines run directories, including integration-test-like runs, longer-epoch remnants, and any recovered metadata.
- [ ] 0.2: Record that the most relevant paper-grade `128x128` Hybrid ResNet runs are lost unless complete run directories with metrics, configs, seeds, and qualitative outputs are recovered.
- [ ] 0.3: Identify the exact regeneration path for the `128x128` anchor, starting from the current Torch grid-lines integration path with more epochs and the current recommended Hybrid ResNet baseline; use `docs/studies/index.md` and `tests/torch/test_grid_lines_hybrid_resnet_integration.py` as runtime/runbook guidance.
- [ ] 0.4: Inventory local CDI baselines already available or cheaply runnable: CNN, FNO variants, classical CDI/PyNX/HIO/ER where applicable, PtychoViT/PtyChi/Tike only when scientifically appropriate.
- [ ] 0.5: Inventory existing `256x256` CDI runbooks/results, including any mode-count evidence.
- [ ] 0.6: Inventory external PDE/forward-modeling candidates and local environment constraints.
- [ ] 0.7: Write an inventory summary under `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`.
- [ ] 0.8: Write a regeneration note under `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md` if no complete paper-grade anchor run is recovered.

Gate:

- [ ] Inventory distinguishes paper-grade evidence from decision-support-only artifacts.
- [ ] The lost-run condition for the `128x128` Hybrid ResNet anchor is explicitly recorded.
- [ ] Either a complete auditable CDI anchor run is recovered, or a fresh regeneration tranche is scheduled with command/runbook, seed/config, metric contract, and runtime budget derived from the study index and Hybrid ResNet integration-test path where possible.
- [ ] At least three PDE candidates are listed for the neutral selection phase.

Recommended verification:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing required inventory docs: {missing}")
print("inventory docs present")
PY
```

If the fresh anchor must be regenerated, also verify:

```bash
python - <<'PY'
from pathlib import Path
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md")
text = path.read_text() if path.exists() else ""
required_terms = ["128x128", "Hybrid ResNet", "regenerate", "seed", "metric", "runtime"]
missing = [term for term in required_terms if term.lower() not in text.lower()]
if missing:
    raise SystemExit(f"regeneration plan missing required terms: {missing}")
print("regeneration plan contains required fields")
PY
```

## Phase 1 - Required PDE Benchmark Selection

Purpose: reduce the highest-risk required contribution before CDI polish.

- [ ] 1.1: Define a scorecard schema with fields for architectural fit, benchmark maturity, metric clarity, data size, install burden, RTX 3090 feasibility, local baseline feasibility, published SOTA availability, and paper-story fit.
- [ ] 1.2: Evaluate PDEBench/PDEArena-style fluids candidates.
- [ ] 1.3: Evaluate wave-equation candidates.
- [ ] 1.4: Evaluate inverse-scattering or related wave/inverse-problem candidates.
- [ ] 1.5: Choose an initial primary deep benchmark and fallback, then record any approved scope amendment.
- [ ] 1.6: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`.

Gate:

- [ ] Selected benchmark or approved suite has clear metrics and runnable data access.
- [ ] Selected benchmark or approved suite can fit small smoke/data-contract runs on the RTX 3090; smoke runs are readiness evidence only, not benchmark performance evidence.
- [ ] At least two feasible local baselines are identified, or one strong local baseline plus published SOTA is justified.
- [ ] Fallback benchmark has enough information to begin quickly if the selected benchmark or suite fails.

Recommended verification:

```bash
python - <<'PY'
from pathlib import Path
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md")
text = path.read_text() if path.exists() else ""
required_terms = ["primary", "fallback", "RTX 3090", "baseline", "SOTA"]
missing = [term for term in required_terms if term.lower() not in text.lower()]
if missing:
    raise SystemExit(f"selection doc missing expected terms: {missing}")
print("selection doc contains required decision fields")
PY
```

## Phase 2 - PDEBench `128x128` Image-Suite Execution

Purpose: produce the required non-CDI empirical contribution.

- [ ] 2.1: Stage or locate the official PDEBench files for SWE (`swe`), Darcy Flow (`darcy`), and 2D Compressible Navier-Stokes (`2d_cfd_cns`) under an ignored or external data root; record source, size, checksum or size/mtime manifest, license/access notes, and exact HDF5 schemas.
- [ ] 2.2: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md` before code-changing adapter work.
- [ ] 2.3: Generalize the existing `scripts/studies/pdebench_swe/` code into a shared PDEBench image-suite adapter only where reuse is straightforward; keep task-specific schema and metric policies explicit.
- [ ] 2.4: Add focused tests for the shared adapter, SWE migration, Darcy static operator-map loading, 2D CNS separate-field dynamic loading, split manifests, normalization, metrics, CNS frequency-band fRMSE (`fRMSE_low/mid/high`), and result writers.
- [ ] 2.5: Run smoke/data-contract gates for all three tasks. Label every smoke metric as sanity/provenance only; smoke output cannot rank models, trigger performance pivots, or satisfy competitiveness.
- [ ] 2.6: Run any capped pilot/triage passes needed to debug runtime, learning curves, and logging. Label capped/subsampled pilot metrics as decision-support only, not meaningful benchmark-performance evidence.
- [ ] 2.7: Run Hybrid ResNet on the three selected tasks under the agreed metric contracts, the inherited Hybrid ResNet training recipe from `docs/model_baselines.md`, and the full available training-split rule unless an implementation plan records a justified override. Competitiveness runs must not use a one-epoch feasibility budget or a capped/subsampled training set.
- [ ] 2.8: Run FNO and U-Net local baselines on the same full available training splits, normalization, horizons, and metrics for each completed task, or document a task-specific blocker before using any published context.
- [ ] 2.8a: For Darcy Flow beta `1.0`, treat the PDEBench supplement's U-Net and FNO values as calibration context, not as same-protocol reproduction unless the reduced-resolution and split/training protocol also match. The current planning target is U-Net RMSE/nRMSE about `6.4e-3`/`3.3e-2` and FNO RMSE/nRMSE about `1.2e-2`/`6.4e-2`; later HAMLET/OFormer context gives nRMSE about `1.40e-2`/`2.05e-2` under its protocol. A tiny smoke U-Net is not a strong baseline for this task.
- [ ] 2.8b: For 2D CNS, report denormalized nRMSE/RMSE plus `fRMSE_low`, `fRMSE_mid`, and `fRMSE_high` for Hybrid ResNet, FNO, and strong U-Net under the same split/history/normalization. Treat `fRMSE_high` as the required shock/small-scale-structure diagnostic.
- [ ] 2.8c: Before packaging CNS as paper evidence, write a CNS paper-contract decision under `docs/plans/NEURIPS-HYBRID-RESNET-2026/` that chooses either `full_training_paper_benchmark` or `bounded_capped_decision_support`, records the exact split/cap/training contract, and sets the authored-FFNO inclusion cutoff and claim impact.
- [ ] 2.8d: Lock the CNS paper rows under the selected contract: best Hybrid/Hybrid-spectral row, local FNO row, U-Net/CNN-style row, and authored FFNO only if available by the predeclared cutoff under the same local CNS contract. If authored FFNO is unavailable or incompatible, record a row-level `blocked` or `not_protocol_compatible` status and limit claims accordingly.
- [ ] 2.8e: Build the CNS paper table and figure bundle with JSON/CSV/TeX metrics, fixed-sample `density`/`Vx`/`Vy`/`pressure` prediction and error panels, source-array manifests, row-status labels, and explicit full-training versus capped-decision-support claim boundaries. Prefer a same-contract `1024 / 128 / 128` bounded bundle when the full headline roster can be recovered or rerun under one contract; if any required same-cap row is missing, fall back to the earlier complete `512 / 64 / 64` bounded row lock instead of mixing caps in one headline table.
- [ ] 2.8f: Treat same-contract `2048 / 256 / 256` CNS comparator-row promotion as a later evidence-strengthening pass, not a blocker for the current `1024 / 128 / 128` CNS table/figure bundle or paper evidence audit.
- [ ] 2.8g: Run a controlled authored-FFNO CNS history-length ablation only as adjacent temporal-context evidence: start from the locked `author_ffno_cns_base` `history_len=2` row, test `history_len=3` first under the same capped contract, gate `history_len=4/5` from observed deltas, and do not mix longer-history FFNO rows into the locked CNS headline table unless a later roadmap-level paper-contract decision reopens the history lane.
- [ ] 2.9: Run focused ablations tied to the spectral/spatial design, such as spectral capacity reduction (`fno_modes=6`) and local branch reduction (`hybrid_resnet_blocks=2`), after full-training primary profiles complete and within the RTX 3090 budget.
- [ ] 2.10: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_summary.md`, and update or supersede `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md` with suite-level interpretation.

Gate:

- [ ] All three task schemas are confirmed as native `128x128` image-grid tasks or any deviation is explicitly recorded before training.
- [ ] At least two of the three tasks produce same-protocol Hybrid ResNet, FNO, and U-Net benchmark-performance rows; the strongest outcome is all three.
- [ ] Meaningful benchmark-performance rows train on the full available training split for the selected official file after validation/test holdout. If 10,000 samples exist and 2,000 are held out, the benchmark row uses the remaining 8,000 training samples. Capped or subsampled rows are pilot/triage evidence only.
- [ ] Hybrid ResNet is competitive with local baselines on at least one completed task and is not rejected from smoke-only evidence.
- [ ] Hybrid ResNet competitiveness is judged only from runs whose profile and training budget inherit the current baseline recipe with the PDE-specific scheduler floor: `fno_modes=12`, width/hidden channels `32`, `fno_blocks=4`, MAE training loss, Adam `lr=2e-4`, and `ReduceLROnPlateau` with factor `0.5`, patience `2`, min LR no higher than `1e-5` (default `1e-5` for PDE studies), and threshold `0.0`, or from a documented intentional override.
- [ ] For Darcy beta `1.0`, a strong local comparison includes FNO and a non-toy U-Net; if the local FNO or U-Net is far outside published PDEBench calibration values after a full available training-split run, implementation/protocol debugging happens before Hybrid ResNet is interpreted.
- [ ] Smoke-gate metrics, regardless of epoch count, are labeled as data/adapter/runtime sanity only and are not used as benchmark performance, paper-grade competitiveness evidence, or model-ranking evidence.
- [ ] Published SOTA comparisons, if used, are labeled as protocol-dependent and not same-code reproduction.
- [ ] Metrics, configs, seeds, environment, task schemas, data identities, and split manifests are recorded.
- [ ] CNS paper evidence has an explicit contract decision before final row locking; capped rows are never promoted to full-training benchmark claims by prose alone.
- [ ] Authored FFNO is either present under the same CNS contract by the predeclared cutoff or recorded as a row-level blocker with the resulting claim limitation.
- [ ] Authored-FFNO longer-history results, if run, are reported as within-model temporal-context ablation evidence and kept separate from same-history model-ranking tables unless a later roadmap-level paper-contract decision changes the CNS headline contract.
- [ ] If the suite is infeasible or clearly noncompetitive after full-training benchmark runs, choose the next plan scope explicitly: SWE-only salvage, OpenFWI FlatVel-A fallback/extension, or a narrower PDE claim.

Recommended verification:

```bash
pytest tests/studies/test_pdebench_swe_metrics.py tests/studies/test_pdebench_swe_splits_data.py tests/studies/test_pdebench_swe_run_config.py -v
# Add the new PDEBench image-suite test modules to this selector as they are created.
```

For long runs, launch in tmux and activate `ptycho311`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python <pdebench_image_suite_runbook>.py --task swe --smoke
```

## Phase 3 - CDI Anchor Regeneration, Verification, and Packaging

Purpose: regenerate the missing `128x128` Hybrid ResNet CDI anchor, then package the `lines128` CDI evidence under the package-level paper evidence design and detailed `lines128` benchmark authority.

- [ ] 3.1: Finalize the fresh `128x128` Hybrid ResNet regeneration command/runbook from Phase 0.
- [ ] 3.2: Run the regenerated anchor with captured invocation/config/seed/git/runtime provenance.
- [ ] 3.3: Verify or rerun compact baselines under the same metric contract.
- [ ] 3.3a: Build or verify the shared `lines128` paper benchmark harness, including contract reconstruction, selected FNO comparator, seed policy, metric schema, fixed sample IDs, visual scales, and row-level blockers for missing complete-table rows.
- [ ] 3.3b: Produce the minimum draftable CDI subset under one paper-grade contract: `hybrid_resnet`, paired CDI `cnn` U-Net-class supervised and PINN rows, and the selected FNO comparator. Label the CDI `cnn` rows as the CDI-side U-Net/CNN-style local baseline family aligned to CNS `unet_strong`, while recording that they are not identical implementations. This subset can unblock manuscript table shells, but it is not the complete `lines128` benchmark.
- [ ] 3.3c: Complete the `lines128` benchmark table required by `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`: `hybrid_resnet`, paired CDI `cnn` U-Net-class supervised and PINN rows, `spectral_resnet_bottleneck_net`, selected FNO comparator, and FFNO as a CDI/grid-lines generator. Optional classical CDI rows may be added under the same contract.
- [ ] 3.3d: Assess a classical HIO/ER/PyNX-style CDI baseline only if it can obey the same `lines128` data, probe, split, metric, and provenance contract; otherwise record an explicit `not_protocol_compatible` outcome.
- [ ] 3.3e: Add the required supervised FFNO CDI control row only after the minimum local supervised/PINN pair and physics-informed Lines128 table are locked, so the complete FFNO + PINN row has a same-contract training-procedure comparator.
- [ ] 3.3f: After the authoritative six-row `lines128` CDI bundle is locked, optional append-only comparator extensions may run only from a checked-in design/preflight that preserves the immutable base root, launches only the new rows, and records explicit lineage to the authoritative complete-table bundle. The current approved candidate for this lane is the external NeuralOperator U-NO extension defined in `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md`.
- [ ] 3.4: Run only the highest-value CDI ablations: spectral/FNO mode or capacity, local/ResNet capacity, and skip routing if already supported. Do not let ablations replace the complete `lines128` table unless a checked-in design amendment changes the required rows.
- [ ] 3.5: Prepare qualitative figure inputs from regenerated or verified reconstructions.
- [ ] 3.6: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_summary.md`.

Gate:

- [ ] `128x128` Hybrid ResNet shows a clear advantage on primary reconstruction metrics.
- [ ] The anchor result is fresh or explicitly recovered with complete provenance; lost historical runs are not used as paper-grade evidence.
- [ ] Compact baselines are protocol-compatible.
- [ ] The minimum draftable CDI subset is labeled as a subset and not as the complete `lines128` benchmark.
- [ ] The complete `lines128` CDI benchmark includes `hybrid_resnet`, `spectral_resnet_bottleneck_net`, the selected FNO comparator, and FFNO, or records explicit row-level blockers / a checked-in design amendment.
- [ ] The minimum CDI subset includes both supervised and PINN rows for the CDI `cnn` U-Net-class architecture; after the complete Lines128 table is locked, the required supervised FFNO control is run under the same contract. Other supervised counterparts remain non-required unless a checked-in design amendment adds them.
- [ ] Optional post-table comparator extensions remain append-only follow-up evidence: they cannot rewrite the authoritative six-row root, cannot silently change the locked `lines128` contract, and do not become required Phase 3 completion work unless a later checked-in design amendment promotes them.
- [ ] Qualitative outputs match the quantitative claim.
- [ ] Any non-ML CDI baseline row is labeled with its exact prior/solver contract.

Recommended verification:

```bash
# Use the narrow selectors that cover the chosen runbook/metrics path.
pytest tests/torch/test_grid_lines_hybrid_resnet_integration.py -v
```

## Phase 4 - `256x256` CDI Scaling Evidence

Purpose: include secondary scaling evidence only if the core two pillars are secure.

- [ ] 4.1: Identify the best existing `256x256` runbook/result candidate.
- [ ] 4.2: Run a memory/runtime preflight for inherited `N=128` Hybrid ResNet settings at `N=256`.
- [ ] 4.3: Select one or two higher FNO mode variants, such as `fno_modes=16/24` or `24/32`, based on memory preflight.
- [ ] 4.4: Keep non-mode variables fixed where possible: split, epochs, scheduler, width/depth, loss mode, probe/mask policy.
- [ ] 4.5: Run only feasible variants.
- [ ] 4.6: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_n256_scaling_summary.md`.

Gate:

- [ ] Higher FNO modes fit the RTX 3090 budget or are explicitly recorded as infeasible.
- [ ] `256x256` results are labeled as secondary scaling evidence.
- [ ] No unstable or provenance-poor run is promoted to paper evidence.

Recommended verification:

```bash
# Fill in with the exact runbook or metric selector used for the selected N=256 path.
pytest <n256_runbook_or_metrics_tests> -v
```

## Phase 5 - Paper-Facing Evidence Bundle

Purpose: assemble derived artifacts under `/home/ollie/Documents/neurips/` after the evidence is ready. This is the first roadmap phase that is expected to create `/home/ollie/Documents/neurips/index.md`. Do not write manuscript prose in this phase unless separately requested.

- [ ] 5.1: Create `/home/ollie/Documents/neurips/index.md` as the top-level evidence map.
- [ ] 5.1a: Use the paused backlog item `docs/backlog/paused/2026-04-29-paper-facing-evidence-index.md` only after this Phase 5 gate opens; it must remain out of active workflow selection while `docs/backlog/roadmap_gate.json` disallows `phase-5-*`.
- [ ] 5.2: Add links to CDI result tables, PDE result tables, source data, figure manifests, run provenance, metric contracts, baseline definitions, ablation summaries, dataset/split descriptions, claim-boundary summaries, and failed/pivoted experiment notes.
- [ ] 5.3: Generate or copy table source artifacts into `/home/ollie/Documents/neurips/tables/`.
- [ ] 5.4: Generate figure manifests and point them to source reconstructions/plots.
- [ ] 5.5: Record published SOTA comparisons separately from locally rerun baselines.
- [ ] 5.6: Write a short evidence completeness checklist under `/home/ollie/Documents/neurips/evidence_checklist.md`.

Gate:

- [ ] `/home/ollie/Documents/neurips/index.md` references all reviewer-relevant evidence.
- [ ] Tables and figures link back to source runs and metric contracts.
- [ ] The index distinguishes paper-grade, full-training, capped decision-support, blocked, and not-protocol-compatible rows.
- [ ] Published-SOTA caveats are visible from the index.
- [ ] No manuscript prose is added unless the user explicitly asks.

Recommended verification:

```bash
python - <<'PY'
from pathlib import Path
root = Path("/home/ollie/Documents/neurips")
required = [root / "index.md", root / "evidence_checklist.md"]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing evidence files: {missing}")
print("neurips evidence index files present")
PY
```

## Routing State

This roadmap is drained by `workflows/examples/neurips_hybrid_resnet_plan_impl_review.yaml`, which uses a progress ledger and provider selector rather than a fixed tranche manifest. The selector must treat the phases above as coarse ordering and gate boundaries, then choose the next coherent plan scope from the approved design, this roadmap, and the live progress ledger.

The lost `128x128` anchor condition is a binding selector input through this roadmap. Until a complete auditable anchor is recovered or a fresh regeneration plan is written, the selector should keep Phase 0 focused on documenting the loss and scheduling regeneration. Later selector iterations must not promote lost historical `128x128` runs as paper-grade evidence.

The Hybrid ResNet training recipe is also a binding selector input for Phase 2. After a smoke/data-access gate succeeds, the next selected benchmark-execution scope must either use the current recipe from `docs/model_baselines.md` plus the PDE-specific scheduler floor (`min_lr <= 1e-5`) or make an explicit plan-level override before launch. Smoke gates can unblock data access and contract readiness, but no smoke metric table satisfies Phase 2 competitiveness or supports a performance pivot by itself.

The full available training-split rule is a binding selector input for Phase 2. Capped runs may be selected for pilot/triage scopes, but any scope selected to produce meaningful benchmark-performance evidence must train each model on the full available training split for the selected official file after validation/test holdout.

The PDEBench image-suite amendment is now a binding selector input for Phase 2. With Darcy preflight/plan work complete and the official 2D CNS file now checksum-verified, adapter-supported, and capped-compare ready, the next selected PDE scope should come from the live suite plan and progress ledger: either the Darcy full-training benchmark tranche or a bounded CNS follow-up compare/ablation that stays explicitly capped and decision-support-only. OpenFWI FlatVel-A should not be selected as the next performance tranche unless the suite plan records a blocker, the user explicitly re-prioritizes it, or the selector documents why OpenFWI is the least risky replacement contribution.

The paper-evidence package design is a binding selector input for late Phase 2 and Phase 3 packaging. CNS paper work must pass through the contract-decision, row-lock, and table/figure-bundle sequence before claims are drafted. CDI paper work must distinguish the minimum draftable `hybrid_resnet`/paired-CDI-cnn-U-Net-class/FNO subset from the complete `lines128` benchmark table required by the detailed `lines128` design, including `spectral_resnet_bottleneck_net` and FFNO. Candidate inverse-wave preflights, including Born/Rytov diffraction tomography and WaveBench inverse source, may execute concurrently under `candidate-*` roadmap phases; priority values in backlog frontmatter should control when they run relative to CDI/CNS work instead of adding brittle one-off gate prefixes. Steering on 2026-04-30 moved WaveBench inverse source ahead of remaining optional U-NO table-extension work, without promoting it into a CDI/CNS replacement pillar or authorizing paper claims before a later evidence-package amendment. Phase 5 paper-facing index work remains paused until the roadmap gate changes; do not make `phase-5-*` selectable while `docs/backlog/roadmap_gate.json` disallows it.

After the authoritative six-row `lines128` CDI bundle is complete, Phase 3 may also admit optional append-only comparator preflights that preserve the locked base bundle and run no existing rows again. The selected `2026-04-30-cdi-lines128-uno-design-preflight` item falls in this lane: it may verify the `neuraloperator`/`neuralop.models.UNO` environment and freeze the later `neuralop_uno` row contract, but it must not reopen the required six-row roster or silently promote U-NO into Phase 3 completion criteria without a checked-in design amendment. The U-NO table-extension item is now lower priority than WaveBench inverse-source candidate work.

After the 2026-04-21 CNS readiness and capped-comparison updates, the selector may choose either `phase-2-pdebench-darcy-static-operator-benchmark` as the next full-training benchmark scope or a capped CNS comparison scope that reuses the verified `history_len=2` MSE anchor and records any follow-up variant against that anchor. If the selected CNS scope is an equal-footing whole-table history-contract compare, it must rerun the full four-row shell (`spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, and `unet_strong`) rather than probing only a single row. A row-local temporal-context ablation is also allowed when it stays explicitly capped and decision-support-only, keeps the reduced-setting contract fixed, compares only against frozen spectral anchors, reports valid-window/runtime deltas, and does not masquerade as a headline model-ranking table update. These CNS follow-up compares remain benchmark-incomplete until full-training Hybrid ResNet, FNO, and `unet_strong` rows run on the full available training split. The Darcy tranche is defined by `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md` and should implement Darcy static-operator support, strong local U-Net/FNO baselines, and literature-calibrated reporting before any full-suite summary.

Current backlog dependency relations for the Phase 2 CNS queue are tracked in
`docs/backlog/index.md`. As of this roadmap revision, the paper-default GNOT
rerun and the author-FFNO compare are now completed external-baseline lanes on
the same local `2d_cfd_cns` contract; they should not stay in the active queue
or be treated as prerequisites for later CNS follow-ups. The earlier
converged-budget spectral mode-count follow-up (`12/12` versus `24/24`, which
depended on the completed `modes32` compare) is now a completed capped CNS
lane. The shared-blocks10 longer-convergence follow-up at `1024 / 128 / 128`
is now also a completed capped CNS lane after its `80`-epoch rerun closed on
2026-04-29, so it no longer defines the immediate active queue. The
history-length-beyond-2 compare is now also a completed capped CNS lane after
its `2026-04-29` closeout, so it no longer defines the active queue. The
current recovered in-progress Phase 2 CNS evidence-strengthening follow-up is
`2026-04-29-cns-paper-2048cap-row-extension`. It reuses the completed
Hybrid-spectral finalist scaling lane at `2048 / 256 / 256` as the
spectral-family anchor and recovers or reruns same-cap
`author_ffno_cns_base`, `fno_base`, and `unet_strong` rows under the locked
local `history_len=2`, `40`-epoch, `max_windows_per_trajectory=8` contract.
This item must not delay or rewrite the current `1024 / 128 / 128` bounded
table/figure bundle or paper evidence audit, must not mix `2048` rows with
`1024` or `512` rows in one headline table, and may replace the current CNS
bundle only if the full required same-cap roster completes with the same
row-status and provenance checks. If any required same-cap row cannot be
produced honestly, keep the current `1024 / 128 / 128` bounded bundle as the
manuscript-supporting authority when complete, or fall back to the earlier
complete `512 / 64 / 64` locked row set rather than merging caps in one
headline bundle. The spectral-history
`2026-04-29-cns-spectral-history-len4plus-compare` item remains a separate
row-local capped ablation for `spectral_resnet_bottleneck_base` (`SRU-Net*` in
manuscript-facing outputs). It must compare only against the frozen
`history_len=2` and `history_len=3` spectral anchors under the same reduced
training setting, preserve the repo-row to manuscript-label mapping in
machine-readable outputs, gate any `history_len=5` run on aggregate
improvement without an unacceptable `fRMSE_high` regression or on an explicit
pre-run scientific rationale, must not rerun `fno_base`, `unet_strong`,
authored FFNO, or `hybrid_resnet_cns` unless a later roadmap decision expands
the contract, and must not be treated as permission to mix history lengths
inside the CNS headline table.
The converged-budget `modes24` compare remains in progress.
The deterministic backlog gate now admits active NeurIPS evidence work across
Phase 2 PDEBench, Phase 3 CDI, and `candidate-*` preflights. Selection order
should be governed by backlog priority and steering value rather than narrow
gate-prefix churn: remaining Phase 2 PDEBench evidence stays preferred by
priority, Phase 3 CDI may run as useful parallel work, and candidate preflights
may run concurrently when their lower priority is reached. Phase 3 CDI items
remain Phase 3 work and must not count as satisfying Phase 2 PDEBench evidence.
Candidate work remains additional preflight work and must not count as
satisfying Phase 2 PDEBench or Phase 3 CDI evidence. The earlier
`history_len=1` Markov compare, `modes32` compare, Hybrid-spectral architecture
ablation, Hybrid-spectral-scaling `2048cap` follow-up, and
FFNO-with-convolutional-features extension are now completed capped CNS lanes,
not active queue items.
These independent capped CNS studies should not be serialized unless a backlog
item introduces an explicit prerequisite.

## Completion Criteria

- [ ] Phase 0 inventory identifies reusable CDI support evidence, records the lost `128x128` anchor condition, schedules regeneration if needed, and lists candidate PDE benchmarks.
- [ ] Phase 1 selects a primary and fallback PDE benchmark and records the approved PDEBench image-suite amendment.
- [ ] Phase 2 produces credible same-protocol evidence for the PDEBench `128x128` image suite or records an explicit suite-failure/pivot decision.
- [ ] Phase 3 regenerates or recovers with complete provenance, verifies, and packages the `128x128` CDI anchor, with the minimum draftable subset clearly separated from the complete `lines128` benchmark table.
- [ ] Phase 4 either provides clean `256x256` scaling evidence with higher-mode consideration or records an explicit infeasibility decision.
- [ ] Phase 5 creates the `/home/ollie/Documents/neurips/` evidence map and derived artifact index files.

## Artifacts Index

- Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- Paper evidence package design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Lines128 CDI benchmark design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
- Lines128 U-NO extension design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md`
- Backlog dependency index: `docs/backlog/index.md`
- PDEBench image-suite plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Initiative docs root: `docs/plans/NEURIPS-HYBRID-RESNET-2026/`
- Future paper-facing root: `/home/ollie/Documents/neurips/`
