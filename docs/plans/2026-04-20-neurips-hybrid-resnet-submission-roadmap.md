# NeurIPS Hybrid ResNet Submission Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a triaged NeurIPS 2026 submission evidence package for the Hybrid ResNet architecture, with `128x128` CDI as the anchor and one required deep PDE/forward-modeling benchmark.

**Architecture:** Run studies from `/home/ollie/Documents/PtychoPINN/`, regenerate the missing paper-grade `128x128` CDI Hybrid ResNet anchor, select one deep PDE benchmark through a neutral screen, and eventually write paper-facing artifacts under `/home/ollie/Documents/neurips/` with `index.md` as the top-level evidence map. The roadmap is deadline- and compute-constrained: NeurIPS full paper deadline is 2026-05-06 AOE, and the assumed compute budget is one RTX 3090 for several days.

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
- [ ] 1.5: Choose one primary deep benchmark and one fallback.
- [ ] 1.6: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`.

Gate:

- [ ] Primary benchmark has clear metrics and runnable data access.
- [ ] Primary benchmark can fit a small smoke run on the RTX 3090.
- [ ] At least two feasible local baselines are identified, or one strong local baseline plus published SOTA is justified.
- [ ] Fallback benchmark has enough information to begin quickly if the primary fails.

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

## Phase 2 - Deep PDE Benchmark Execution

Purpose: produce the required non-CDI empirical contribution.

- [ ] 2.1: Add or configure the selected external benchmark dependency/data access in a documented, minimal way.
- [ ] 2.2: Create a benchmark adapter or runbook only if existing tooling cannot run the task directly.
- [ ] 2.3: Add focused tests for any new adapter, data contract, metric parser, or result writer.
- [ ] 2.4: Run a smoke test for data loading, one tiny training/evaluation pass, and metric writing.
- [ ] 2.5: Run Hybrid ResNet on the selected benchmark under the agreed metric contract.
- [ ] 2.6: Run at least two feasible local baselines, or document why only one local baseline is feasible and pair it with published SOTA.
- [ ] 2.7: Run one or two focused ablations tied to spectral/spatial design, such as spectral capacity reduction or local branch reduction.
- [ ] 2.8: Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`.

Gate:

- [ ] Hybrid ResNet is competitive with local baselines.
- [ ] Published SOTA comparisons, if used, are labeled as protocol-dependent and not same-code reproduction.
- [ ] Metrics, configs, seeds, and environment are recorded.
- [ ] If Hybrid ResNet is not competitive, pivot to the fallback benchmark before spending CDI polish time.

Recommended verification:

```bash
# Exact commands depend on the selected benchmark and must be filled in after Phase 1.
pytest <selected_adapter_or_metric_tests> -v
```

For long runs, launch in tmux and activate `ptycho311`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python <selected_benchmark_runbook>.py --smoke
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

## Completion Criteria

- [ ] Phase 0 inventory identifies reusable CDI support evidence, records the lost `128x128` anchor condition, schedules regeneration if needed, and lists candidate PDE benchmarks.
- [ ] Phase 1 selects a primary and fallback PDE benchmark.
- [ ] Phase 2 produces a credible deep PDE benchmark result or executes the fallback decision.
- [ ] Phase 3 regenerates or recovers with complete provenance, verifies, and packages the `128x128` CDI anchor.
- [ ] Phase 4 either provides clean `256x256` scaling evidence with higher-mode consideration or records an explicit infeasibility decision.
- [ ] Phase 5 creates the `/home/ollie/Documents/neurips/` evidence map and derived artifact index files.

## Artifacts Index

- Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- Initiative docs root: `docs/plans/NEURIPS-HYBRID-RESNET-2026/`
- Future paper-facing root: `/home/ollie/Documents/neurips/`
