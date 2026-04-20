# NeurIPS Hybrid ResNet Submission Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a triaged NeurIPS 2026 submission evidence package for the Hybrid ResNet architecture, with `128x128` CDI as the anchor and a required compact native `128x128` PDEBench image-suite contribution.

**Architecture:** Run studies from `/home/ollie/Documents/PtychoPINN/`, regenerate the missing paper-grade `128x128` CDI Hybrid ResNet anchor, execute a scoped PDEBench `128x128` image suite covering SWE, Darcy Flow, and 2D diffusion-reaction, and eventually write paper-facing artifacts under `/home/ollie/Documents/neurips/` with `index.md` as the top-level evidence map. The roadmap is deadline- and compute-constrained: NeurIPS full paper deadline is 2026-05-06 AOE, and the assumed compute budget is one RTX 3090 for several days.

**Tech Stack:** Python 3.11 in the `ptycho311` environment, PyTorch/Lightning, existing PtychoPINN grid-lines and Torch study runbooks, optional external PDE benchmark dependencies, Markdown/JSON/CSV/TeX artifact sources.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Title: Hybrid ResNet NeurIPS 2026 Submission Campaign
- Status: pending
- Design/Source: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Experiment root: `/home/ollie/Documents/PtychoPINN/`
- Future artifact root: `/home/ollie/Documents/neurips/`

## Compliance Matrix

- [ ] **Repo Guidance:** `CLAUDE.md` - read `docs/index.md`, keep plans under `docs/plans/`, use tmux for long-running commands, use `ptycho311` for long-running PtychoPINN workflows, do not create worktrees.
- [ ] **Baseline Authority:** `docs/model_baselines.md` - inherit the recommended Hybrid ResNet baseline unless a study explicitly overrides it.
- [ ] **Study Discovery:** `docs/studies/index.md` - reuse existing study runbooks and artifact contracts where possible.
- [ ] **Known Findings:** `docs/findings.md` - obey active PyTorch/grid-lines findings, including probe-mask and reassembly contracts.
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

- [ ] 2.1: Stage or locate the official PDEBench files for SWE (`swe`), Darcy Flow (`darcy`), and 2D diffusion-reaction (`2d_reacdiff`) under an ignored or external data root; record source, size, checksum or size/mtime manifest, license/access notes, and exact HDF5 schemas.
- [ ] 2.2: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md` before code-changing adapter work.
- [ ] 2.3: Generalize the existing `scripts/studies/pdebench_swe/` code into a shared PDEBench image-suite adapter only where reuse is straightforward; keep task-specific schema and metric policies explicit.
- [ ] 2.4: Add focused tests for the shared adapter, SWE migration, Darcy static operator-map loading, 2D diffusion-reaction one-step loading, split manifests, normalization, metrics, and result writers.
- [ ] 2.5: Run smoke/data-contract gates for all three tasks. Label every smoke metric as sanity/provenance only; smoke output cannot rank models, trigger performance pivots, or satisfy competitiveness.
- [ ] 2.6: Run any capped pilot/triage passes needed to debug runtime, learning curves, and logging. Label capped/subsampled pilot metrics as decision-support only, not meaningful benchmark-performance evidence.
- [ ] 2.7: Run Hybrid ResNet on the three selected tasks under the agreed metric contracts, the inherited Hybrid ResNet training recipe from `docs/model_baselines.md`, and the full available training-split rule unless an implementation plan records a justified override. Competitiveness runs must not use a one-epoch feasibility budget or a capped/subsampled training set.
- [ ] 2.8: Run FNO and U-Net local baselines on the same full available training splits, normalization, horizons, and metrics for each completed task, or document a task-specific blocker before using any published context.
- [ ] 2.8a: For Darcy Flow beta `1.0`, treat the PDEBench supplement's U-Net and FNO values as calibration context, not as same-protocol reproduction unless the reduced-resolution and split/training protocol also match. The current planning target is U-Net RMSE/nRMSE about `6.4e-3`/`3.3e-2` and FNO RMSE/nRMSE about `1.2e-2`/`6.4e-2`; later HAMLET/OFormer context gives nRMSE about `1.40e-2`/`2.05e-2` under its protocol. A tiny smoke U-Net is not a strong baseline for this task.
- [ ] 2.9: Run focused ablations tied to the spectral/spatial design, such as spectral capacity reduction (`fno_modes=6`) and local branch reduction (`hybrid_resnet_blocks=2`), after full-training primary profiles complete and within the RTX 3090 budget.
- [ ] 2.10: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_summary.md`, and update or supersede `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md` with suite-level interpretation.

Gate:

- [ ] All three task schemas are confirmed as native `128x128` image-grid tasks or any deviation is explicitly recorded before training.
- [ ] At least two of the three tasks produce same-protocol Hybrid ResNet, FNO, and U-Net benchmark-performance rows; the strongest outcome is all three.
- [ ] Meaningful benchmark-performance rows train on the full available training split for the selected official file after validation/test holdout. If 10,000 samples exist and 2,000 are held out, the benchmark row uses the remaining 8,000 training samples. Capped or subsampled rows are pilot/triage evidence only.
- [ ] Hybrid ResNet is competitive with local baselines on at least one completed task and is not rejected from smoke-only evidence.
- [ ] Hybrid ResNet competitiveness is judged only from runs whose profile and training budget inherit the current baseline recipe: `fno_modes=12`, width/hidden channels `32`, `fno_blocks=4`, MAE training loss, Adam `lr=2e-4`, and `ReduceLROnPlateau` with factor `0.5`, patience `2`, min LR `1e-4`, and threshold `0.0`, or from a documented intentional override.
- [ ] For Darcy beta `1.0`, a strong local comparison includes FNO and a non-toy U-Net; if the local FNO or U-Net is far outside published PDEBench calibration values after a full available training-split run, implementation/protocol debugging happens before Hybrid ResNet is interpreted.
- [ ] Smoke-gate metrics, regardless of epoch count, are labeled as data/adapter/runtime sanity only and are not used as benchmark performance, paper-grade competitiveness evidence, or model-ranking evidence.
- [ ] Published SOTA comparisons, if used, are labeled as protocol-dependent and not same-code reproduction.
- [ ] Metrics, configs, seeds, environment, task schemas, data identities, and split manifests are recorded.
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

Purpose: regenerate the missing `128x128` Hybrid ResNet CDI anchor and turn it into paper-grade evidence.

- [ ] 3.1: Finalize the fresh `128x128` Hybrid ResNet regeneration command/runbook from Phase 0.
- [ ] 3.2: Run the regenerated anchor with captured invocation/config/seed/git/runtime provenance.
- [ ] 3.3: Verify or rerun compact baselines under the same metric contract.
- [ ] 3.4: Run only the highest-value CDI ablations: spectral/FNO mode or capacity, local/ResNet capacity, and skip routing if already supported.
- [ ] 3.5: Prepare qualitative figure inputs from regenerated or verified reconstructions.
- [ ] 3.6: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_summary.md`.

Gate:

- [ ] `128x128` Hybrid ResNet shows a clear advantage on primary reconstruction metrics.
- [ ] The anchor result is fresh or explicitly recovered with complete provenance; lost historical runs are not used as paper-grade evidence.
- [ ] Compact baselines are protocol-compatible.
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

Purpose: assemble derived artifacts under `/home/ollie/Documents/neurips/` after the evidence is ready. Do not write manuscript prose in this phase unless separately requested.

- [ ] 5.1: Create `/home/ollie/Documents/neurips/index.md` as the top-level evidence map.
- [ ] 5.2: Add links to CDI result tables, PDE result tables, source data, figure manifests, run provenance, metric contracts, baseline definitions, ablation summaries, dataset/split descriptions, and failed/pivoted experiment notes.
- [ ] 5.3: Generate or copy table source artifacts into `/home/ollie/Documents/neurips/tables/`.
- [ ] 5.4: Generate figure manifests and point them to source reconstructions/plots.
- [ ] 5.5: Record published SOTA comparisons separately from locally rerun baselines.
- [ ] 5.6: Write a short evidence completeness checklist under `/home/ollie/Documents/neurips/evidence_checklist.md`.

Gate:

- [ ] `/home/ollie/Documents/neurips/index.md` references all reviewer-relevant evidence.
- [ ] Tables and figures link back to source runs and metric contracts.
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

The Hybrid ResNet training recipe is also a binding selector input for Phase 2. After a smoke/data-access gate succeeds, the next selected benchmark-execution scope must either use the current recipe from `docs/model_baselines.md` or make an explicit plan-level override before launch. Smoke gates can unblock data access and contract readiness, but no smoke metric table satisfies Phase 2 competitiveness or supports a performance pivot by itself.

The full available training-split rule is a binding selector input for Phase 2. Capped runs may be selected for pilot/triage scopes, but any scope selected to produce meaningful benchmark-performance evidence must train each model on the full available training split for the selected official file after validation/test holdout.

The PDEBench image-suite amendment is now a binding selector input for Phase 2. The next selected PDE plan should decide a coherent tranche scope from the suite plan and live progress ledger, starting with Darcy and 2D diffusion-reaction data/schema preflight plus shared adapter design before any expensive three-task training campaign. OpenFWI FlatVel-A should not be selected as the next performance tranche unless the suite plan records a blocker, the user explicitly re-prioritizes it, or the selector documents why OpenFWI is the least risky replacement contribution.

After the 2026-04-20 Darcy staging update, the selector may choose `phase-2-pdebench-darcy-static-operator-benchmark` as the next coherent Phase 2 scope because SWE and Darcy are preflight-ready while 2D diffusion-reaction is still data-blocked. That tranche is defined by `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md` and should implement Darcy static-operator support, strong local U-Net/FNO baselines, and literature-calibrated reporting before any full-suite summary.

## Completion Criteria

- [ ] Phase 0 inventory identifies reusable CDI support evidence, records the lost `128x128` anchor condition, schedules regeneration if needed, and lists candidate PDE benchmarks.
- [ ] Phase 1 selects a primary and fallback PDE benchmark and records the approved PDEBench image-suite amendment.
- [ ] Phase 2 produces credible same-protocol evidence for the PDEBench `128x128` image suite or records an explicit suite-failure/pivot decision.
- [ ] Phase 3 regenerates or recovers with complete provenance, verifies, and packages the `128x128` CDI anchor.
- [ ] Phase 4 either provides clean `256x256` scaling evidence with higher-mode consideration or records an explicit infeasibility decision.
- [ ] Phase 5 creates the `/home/ollie/Documents/neurips/` evidence map and derived artifact index files.

## Artifacts Index

- Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- PDEBench image-suite plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Initiative docs root: `docs/plans/NEURIPS-HYBRID-RESNET-2026/`
- Future paper-facing root: `/home/ollie/Documents/neurips/`
