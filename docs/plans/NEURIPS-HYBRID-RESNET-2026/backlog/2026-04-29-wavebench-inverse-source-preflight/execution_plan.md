# WaveBench Inverse Source Preflight Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. This item authorizes only WaveBench preflight inspection, narrow loader/checkpoint/forward-model smoke checks, and durable decision artifacts. Do not launch the full WaveBench benchmark, do not amend the NeurIPS roadmap, do not create worktrees, and do not alter the CDI `lines128` or PDEBench CNS paper lanes here. If any command becomes long-running, keep it under implementation ownership until terminal success or recoverable failure handling completes; use tmux plus the `ptycho311` environment when appropriate.

**Goal:** Decide whether WaveBench inverse source reconstruction is ready for a follow-up supervised plan, ready for a supervised-plus-physics plan, needs an external data/checkpoint decision, or should be rejected as unsuitable for the current manuscript.

**Architecture:** Split the work into three units: external benchmark contract audit, local compatibility evidence, and durable decision packaging. The implementation should gather only the minimum repo/data/runtime evidence needed to lock a first runnable inverse-source variant, prove or reject local physics-loop fidelity, and hand off explicit follow-up backlog directions without spending training budget intended for the required CDI and CNS pillars.

**Tech Stack:** PATH `python`, Markdown/JSON artifacts, optional external WaveBench repo checkout, PyTorch/Lightning/FFCV only if needed for loader or checkpoint smoke, optional tmux-managed long-running staging or smoke commands in `ptycho311`.

---

## Selected Objective

- Implement backlog item `2026-04-29-wavebench-inverse-source-preflight`.
- Determine whether WaveBench inverse source reconstruction is a practical optional inverse-wave evidence lane for the manuscript.
- Separate supervised readiness from physics-informed readiness.
- Produce a durable preflight summary at `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md`.

## Scope

- Consume the binding design input at `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_benchmark_design.md`.
- Inspect the upstream WaveBench repo surface, dataset availability, inverse-source variants, loader contracts, native baseline checkpoints, and forward-model implementation surfaces.
- Select the first runnable inverse-source variant and record exact observed-data, target, and optional wavespeed tensor contracts.
- Validate whether a differentiable local forward model can reproduce WaveBench measurements from ground-truth `q_0` closely enough to authorize a later physics-informed plan.
- Emit one final readiness state:
  - `ready_for_supervised_plan`
  - `ready_for_supervised_and_physics_plan`
  - `needs_dataset_or_checkpoint_decision`
  - `not_suitable_for_current_manuscript`

## Explicit Non-Goals

- Do not run full WaveBench training, benchmark sweeps, or paper-table generation.
- Do not treat WaveBench as Phase 2 PDEBench work.
- Do not amend the roadmap or promote WaveBench into manuscript claims from this item alone.
- Do not alter CDI `lines128`, PDEBench CNS, or optional U-NO table-extension outputs.
- Do not describe WaveBench inverse source as geology, full waveform inversion, or material-property inversion.
- Do not accept any physics-informed readiness claim unless `F_c(q_0) ~= y` is demonstrated on ground-truth examples for the selected dataset variant.

## Steering, Roadmap, and Design Constraints

- The roadmap's required pillars remain CDI `lines128` and PDEBench `2d_cfd_cns`; WaveBench is only an optional candidate lane at this stage.
- Steering keeps WaveBench candidate work on equal footing with BRDT as optional inverse-wave exploration, but as of 2026-04-30 it should be attempted before any new optional U-NO rows are added.
- Preserve equal-footing comparison standards and the design's fairness boundaries: native WaveBench FNO/U-Net rows stay separate from internal shared-encoder comparison rows.
- The preflight must not silently expand into later backlog items such as native baseline reproduction, shared-encoder benchmark execution, forward-model integration, or paper bundle assembly.
- Any dataset access, checkpoint reuse, or solver reproduction blocker must be recorded explicitly instead of being papered over with approximate language.

## Prerequisite Status

- Backlog frontmatter lists no formal prerequisites.
- Progress ledger shows the initiative already completed Phase 0 evidence inventory, Phase 1 PDE benchmark selection, and early PDE/OpenFWI smoke work; no WaveBench tranche is completed yet.
- Because the required CDI/CNS pillars remain incomplete, this item must stay low-expense and decision-oriented. Minimal staging and smoke checks are allowed; benchmark training is not.
- Before any dataset staging larger than a minimal inspection subset, checkpoint-load smoke, or forward-model smoke, the implementation must first pass the deterministic input-presence check and record the selected variant plus access path.

## Execution Rules

- Diagnose/fix/rerun normal import, path, environment, or test-harness failures before considering the item blocked.
- Reserve `BLOCKED` for missing external data access, unavailable credentials, unresolved upstream dependency incompatibility after a documented narrow fix attempt, unavailable hardware, roadmap conflict, or a forward-model validity gap that remains unrecoverable after a narrow reproduction attempt.
- Keep scratch downloads or temporary extracts under `tmp/` or an external ignored location, then remove repo-local scratch before completion.
- If helper code is needed, keep it minimal and preflight-specific; do not build the full benchmark stack in this item.

## Implementation Architecture

1. **Benchmark Contract Audit**
   - Owns upstream repo/document inspection, dataset/source discovery, inverse-source variant inventory, dependency risks, and native baseline surface discovery.
2. **Local Compatibility Evidence**
   - Owns minimal dataset staging, tensor-shape/normalization inspection, optional checkpoint-load smoke, and the forward-model reproduction check that gates physics-informed readiness.
3. **Durable Decision Packaging**
   - Owns the checked-in summary, machine-readable metadata, explicit final readiness status, follow-up backlog routing, and any required discoverability updates.

## File and Artifact Targets

Create or update:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/preflight_metadata.json`
- `docs/index.md` to index the new durable preflight summary once it exists

Optional, only if needed for reusable inspection:

- `scripts/studies/wavebench/` for small preflight helpers
- `tests/studies/` for narrow selectors covering any new helper code
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/loader_smoke/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/checkpoint_smoke/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/forward_model_check/`

Leave unchanged unless the work uncovers a concrete contract mismatch that must be documented separately:

- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`

## Mandatory Durable Output Contract

Before any later WaveBench training can be authorized, the checked-in summary
and machine-readable metadata together must cover every design-required
preflight field. Treat these as required outputs of this item, not as optional
notes that a later planner must rediscover.

- Provenance and access fields:
  - exact WaveBench repository URL and revision or release;
  - exact dataset DOI or source URL, file names, local staging path, and
    license/access notes;
  - dataset checksum when practical, otherwise a size/mtime manifest;
- Variant and split fields:
  - selected inverse-source variant and selected split;
  - train/validation/test sample counts for the selected dataset file or a
    precise blocker explaining why they could not be read directly;
- Tensor-contract fields:
  - observed-data `y`, target `q_0`, and wavespeed `c(x)` tensor shapes when
    exposed;
  - dtype and normalization conventions for each exposed tensor;
- Native-baseline fields:
  - native FNO and native U-Net availability status;
  - checkpoint identifiers or the exact reason they are unavailable;
  - exact checkpoint/evaluation load path if reuse is possible;
- Physics-loop readiness fields:
  - whether WaveBench generation or solver code is available for
    reproduction;
  - grid, time step, boundary condition, fixed-wavespeed, receiver, and
    initial-condition metadata for the selected variant;
  - normalization applied to `y` and to `F_c(q_0)`;
  - reproduction-check sample count;
  - waveform MAE, RMSE, relative L2, and normalized residual
    `||F_c(q_0)-y||_2 / ||y||_2`;
  - accepted thresholds used for the go/no-go decision, including any approved
    threshold override;
- Final decision fields:
  - one final status token from
    `ready_for_supervised_plan`,
    `ready_for_supervised_and_physics_plan`,
    `needs_dataset_or_checkpoint_decision`,
    `not_suitable_for_current_manuscript`;
  - explicit recommended follow-up backlog directions and any row-level
    blockers.

The summary should be readable on its own, while `preflight_metadata.json`
should mirror the same decisions with stable keys so later plan/execution
phases do not have to reread raw WaveBench sources to recover the contract.

## Execution Tranches

### Tranche 1: Audit The External WaveBench Contract

**Purpose:** Lock the exact upstream benchmark surface before touching local data or making readiness claims.

- [ ] Record the authoritative upstream WaveBench repo URL and revision or release used for preflight.
- [ ] Inspect README, notebooks, package modules, configs, and linked papers for inverse-source coverage.
- [ ] Enumerate inverse-source dataset variants, split semantics, wavespeed conventions, and any OOD or fixed-variant distinctions.
- [ ] Record dataset DOI or source URLs, exact file names, license/access notes, checksum or size/mtime expectations, and whether access requires manual credentials.
- [ ] Record runtime/setup risks: package versions, CUDA assumptions, FFCV requirements, Lightning expectations, checkpoint-loading assumptions, and whether a dedicated environment appears necessary.
- [ ] Record where native WaveBench FNO and U-Net baselines live conceptually: checkpoints, configs, notebooks, or evaluation scripts.

**Verification for Tranche 1**

- [ ] `preflight_metadata.json` contains repo identity, inverse-source variant names, dataset source details, exact file names, access/license notes, checksum-or-size/mtime provenance, and setup-risk notes.
- [ ] The summary draft or working notes explicitly state whether local setup looks straightforward, dedicated-environment-bound, or blocked by external access.

### Tranche 2: Select The First Runnable Variant And Lock Tensor Contracts

**Purpose:** Identify the first concrete supervised target without over-staging data.

- [ ] Stage or access only the smallest dataset slice needed to inspect one train example and one validation/test example for a candidate inverse-source variant.
- [ ] Load samples through the intended loader path; if loader execution is blocked, document the blocker and inspect the on-disk schema directly.
- [ ] Record the selected split and train/validation/test sample counts for the chosen dataset file before any checkpoint or forward-model smoke runs proceed.
- [ ] Record exact tensor shapes, dtype, value ranges, and normalization conventions for:
  - observed boundary-time measurements `y`
  - target initial pressure/source field `q_0`
  - wavespeed field `c(x)` if exposed
  - receiver or boundary-location metadata if exposed
- [ ] Decide the first runnable variant for later supervised benchmarking.
- [ ] Record whether `q_0` is already `128 x 128`, requires resizing, or requires an explicit target-resolution follow-up decision.

**Verification for Tranche 2**

- [ ] `preflight_metadata.json` includes machine-readable selected-variant, selected-split, train/validation/test counts, tensor-shape, dtype, normalization, and staging-path fields.
- [ ] `wavebench_inverse_source_preflight_summary.md` names the selected variant and split, records train/validation/test counts, or explicitly records why that selection/count extraction is blocked.
- [ ] No larger data staging or further smoke work proceeds until these tensor-contract fields are populated.

### Tranche 3: Verify Native Baseline Reuse Feasibility

**Purpose:** Decide whether WaveBench's own baselines can anchor later equal-footing reference rows.

- [ ] Locate native FNO and U-Net checkpoints, if available, and record whether they cover the selected variant.
- [ ] Record the exact evaluation or load path, such as `LitModel.load_from_checkpoint` or the current equivalent.
- [ ] If the local environment and checkpoint files permit it, run the smallest checkpoint-load or evaluation smoke needed to prove compatibility.
- [ ] If checkpoint reuse is not possible, record the shortest credible native-baseline reproduction path from WaveBench configs or notebooks.
- [ ] Classify each native baseline as `checkpoint_reusable`, `retrain_required`, or `not_available`.
- [ ] Record concrete checkpoint identifiers, paths, or artifact references when a baseline is marked reusable; if not reusable, record the exact missing identifier or incompatibility.

**Verification for Tranche 3**

- [ ] The summary and metadata agree on FNO/U-Net baseline availability status and checkpoint identifiers or missing-identifier reasons for the selected variant.
- [ ] Any smoke output is archived under the preflight artifact root.
- [ ] If helper code was added, its narrowest test or import selector passes before the checkpoint conclusion is considered final.

### Tranche 4: Check Shared-Encoder And Physics-Loop Feasibility

**Purpose:** Separate supervised follow-up readiness from physics-informed follow-up readiness.

- [ ] Confirm whether boundary-time data can be represented as a stable 2D measurement image `y(t, b)` for shared-encoder rows.
- [ ] Record which in-repo model families can consume the proposed latent/input contract and what adapter or output-head work is missing for later supervised rows.
- [ ] Decide whether the first follow-up should target `C=32`, `C=64`, or both, without launching training.
- [ ] Locate WaveBench forward-model or data-generation code for the selected inverse-source variant, and classify it as differentiable in PyTorch, portable with narrow work, offline-only, or unavailable.
- [ ] Record the exact wave-equation convention, grid/time-step metadata, boundary conditions, receiver layout, wavespeed convention, and initial-condition convention used by the selected variant.
- [ ] If forward-model access is feasible, run the smallest reproduction check needed to test `F_c(q_0) ~= y` on ground-truth examples. Default target: at least 16 held-out examples unless data access only permits a smaller smoke.
- [ ] Record the normalization applied to observed `y` and reproduced `F_c(q_0)` before computing comparison metrics.
- [ ] Record waveform MAE, RMSE, relative L2, and normalized residual `||F_c(q_0)-y||_2 / ||y||_2` for the checked examples.
- [ ] Compare the reproduction residuals against the design's default validity gate:
  - median relative residual `<= 0.02`
  - no more than 5 percent of checked examples above `0.05`
- [ ] If the design thresholds are not appropriate for the actual data scale, document an alternate threshold proposal before making any physics-ready claim.
- [ ] Classify physics readiness as one of:
  - `exact_physics_loop_ready`
  - `approximate_physics_regularization_possible`
  - `physics_loop_deferred`
  - `physics_loop_not_recommended`

**Verification for Tranche 4**

- [ ] Physics-informed readiness is marked ready only if the reproduction check ran, recorded waveform MAE/RMSE/relative L2/normalized residual plus sample count and normalization details, and satisfied the accepted threshold set.
- [ ] If reproduction fails or cannot run, the summary explicitly defers or rejects physics-informed claims while preserving any supervised-only recommendation.
- [ ] No training or roadmap-amendment recommendation is written before this gate is resolved.

### Tranche 5: Package The Durable Decision And Follow-Up Routing

**Purpose:** Close the backlog item with a self-contained readiness decision and discoverable artifacts.

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md`.
- [ ] Write `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/preflight_metadata.json`.
- [ ] Ensure both artifacts record the same:
  - selected variant or blocking reason
  - selected split plus train/validation/test sample counts
  - dataset file identity, provenance, and local staging path
  - observed/target tensor contracts
  - dtype and normalization conventions
  - native baseline availability
  - checkpoint identifiers or exact missing-identifier reasons
  - forward-model availability
  - forward-model metric bundle: sample count, waveform MAE, RMSE, relative L2, normalized residual, and accepted thresholds
  - physics readiness classification
  - final backlog-item status
- [ ] Include text-only recommended follow-up backlog directions for:
  - supervised shared-encoder benchmark
  - native baseline reproduction
  - forward-model validation / physics-loss integration
  - optional paper bundle assembly
- [ ] State explicitly that roadmap amendment is out of scope for this item; any promotion of WaveBench into manuscript authority is a later follow-up conditioned on the preflight result.
- [ ] Update `docs/index.md` so the new durable preflight summary is discoverable.
- [ ] Consult the initiative evidence-index policy and either update the applicable durable index surface or state in the summary why no model/evidence index beyond `docs/index.md` changed because this item produced no benchmark row or paper-facing artifact.

**Verification for Tranche 5**

- [ ] Summary and metadata agree on the final status token, selected variant/split decision, and the design-required provenance, split-count, checkpoint, and forward-model metric fields.
- [ ] The summary contains enough context that a later implementation phase can proceed without rereading the raw backlog item.
- [ ] Any unchanged evidence-index surface is explicitly justified rather than silently skipped.

## Required Deterministic Checks

Run the backlog item's declared check command at minimum:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_benchmark_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/execution_plan.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing WaveBench preflight inputs: {missing}")
print("wavebench preflight inputs present")
PY
```

Run this stronger final consistency check before closing the item:

```bash
python - <<'PY'
import json
from pathlib import Path

summary = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md")
metadata = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/preflight_metadata.json")
index_path = Path("docs/index.md")
statuses = {
    "ready_for_supervised_plan",
    "ready_for_supervised_and_physics_plan",
    "needs_dataset_or_checkpoint_decision",
    "not_suitable_for_current_manuscript",
}

if not summary.exists():
    raise SystemExit(f"missing summary: {summary}")
if not metadata.exists():
    raise SystemExit(f"missing metadata: {metadata}")

text = summary.read_text()
hits = [status for status in statuses if status in text]
if len(hits) != 1:
    raise SystemExit(f"expected exactly one final status in summary, found {hits}")

data = json.loads(metadata.read_text())
metadata_status = data.get("final_status")
if metadata_status != hits[0]:
    raise SystemExit(
        f"summary status {hits[0]!r} does not match metadata final_status {metadata_status!r}"
    )

required_metadata_fields = [
    "repo_url",
    "repo_revision",
    "dataset_source",
    "dataset_files",
    "dataset_access",
    "dataset_provenance",
    "staging_path",
    "selected_variant",
    "selected_split",
    "sample_counts",
    "tensor_contracts",
    "native_baselines",
    "forward_model",
]
missing_fields = [field for field in required_metadata_fields if field not in data]
if missing_fields:
    raise SystemExit(f"metadata missing required fields: {missing_fields}")

sample_counts = data["sample_counts"]
for key in ("train", "validation", "test"):
    if key not in sample_counts:
        raise SystemExit(f"sample_counts missing {key!r}")

native_baselines = data["native_baselines"]
for key in ("fno", "unet"):
    if key not in native_baselines:
        raise SystemExit(f"native_baselines missing {key!r}")
    baseline_entry = native_baselines[key]
    if "status" not in baseline_entry:
        raise SystemExit(f"native_baselines[{key!r}] missing 'status'")
    if "checkpoint_identifier" not in baseline_entry and "blocker_reason" not in baseline_entry:
        raise SystemExit(
            f"native_baselines[{key!r}] must include checkpoint_identifier or blocker_reason"
        )

forward_model = data["forward_model"]
for key in (
    "availability",
    "sample_count",
    "normalization",
    "metrics",
    "accepted_thresholds",
    "physics_readiness",
):
    if key not in forward_model:
        raise SystemExit(f"forward_model missing {key!r}")

metrics = forward_model["metrics"]
for key in ("waveform_mae", "waveform_rmse", "relative_l2", "normalized_residual"):
    if key not in metrics:
        raise SystemExit(f"forward_model.metrics missing {key!r}")

if "wavebench_inverse_source_preflight_summary.md" not in index_path.read_text():
    raise SystemExit("docs/index.md does not reference the WaveBench preflight summary")

text_lower = text.lower()
summary_required_terms = [
    "selected split",
    "train/validation/test",
    "checkpoint",
    "waveform mae",
    "rmse",
    "relative l2",
    "normalized residual",
]
missing_terms = [term for term in summary_required_terms if term not in text_lower]
if missing_terms:
    raise SystemExit(f"summary missing required terms: {missing_terms}")

print("wavebench preflight summary, metadata, and index references are consistent")
PY
```

If helper code or tests were added, run the narrowest relevant selector and archive the log under the preflight artifact root. If a forward-model reproduction script was added, its verification must stay narrow and must complete before any claim of physics readiness is made.
