# Phase 0 Evidence Inventory Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the Phase 0 evidence inventory for the NeurIPS Hybrid ResNet campaign, identifying reusable CDI evidence, baseline evidence, N=256 scaling evidence, and at least three PDE/forward-modeling candidates, and write the required CDI anchor regeneration note if no paper-grade `128x128` anchor is recovered.

**Architecture:** Keep Phase 0 as an inventory and provenance tranche. Reuse valid carry-forward raw inventory artifacts before rescanning, separate paper-grade evidence from decision-support-only artifacts, store bulky/raw inventory under a git-ignored artifact root, write the durable Phase 0 summary at `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`, and write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md` when the N=128 anchor remains unrecovered.

**Tech Stack:** Python 3.11 via PATH `python`, Markdown, JSON, existing grid-lines/Torch study artifacts, existing study runbooks, optional primary-source web research for PDE candidate metadata, no new production dependencies.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Title: Phase 0 Evidence Inventory
- Status: pending
- Spec/Source:
  - Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
  - Roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - Tranche context: `state/NEURIPS-HYBRID-RESNET-2026/items/phase-0-evidence-inventory/tranche-context.md`
- Plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-0-evidence-inventory/execution_plan.md`

## Compliance Matrix

- [ ] **Roadmap Phase Order:** Execute only Roadmap Phase 0. Do not select a primary/fallback PDE benchmark, run Phase 2 PDE training, polish CDI evidence, run N=256 scaling variants, or assemble `/home/ollie/Documents/neurips/` artifacts.
- [ ] **Phase 0 Gate:** The inventory must distinguish paper-grade evidence from decision-support-only artifacts, explicitly record the lost-run condition for the `128x128` Hybrid ResNet anchor, either recover a complete auditable CDI anchor or write a fresh regeneration note, and list at least three PDE candidates for Phase 1.
- [ ] **Reuse-First CDI Anchor:** Existing `128x128` grid-lines Hybrid ResNet runs are anchor candidates, but no CDI claim may be marked paper-grade without path, invocation/config, commit or commit-gap, seed, epoch count, scheduler, dataset split, metrics, and qualitative-output provenance.
- [ ] **Neutral PDE Screen Boundary:** Phase 0 inventories PDEBench/PDEArena-style fluids or waves plus inverse-scattering or related wave/inverse-problem candidates. It must not rank them into primary/fallback decisions.
- [ ] **N=256 Boundary:** Existing `256x256` runbooks/results and mode-count evidence are inventoried only as secondary scaling context. Do not promote N=256 evidence or run higher-mode variants in this tranche.
- [ ] **Artifact Hygiene:** Keep raw scans and machine-generated inventories under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/` or another ignored artifact root. Link concise summaries from tracked docs.
- [ ] **Stable Modules:** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- [ ] **No Worktrees:** Use the current checkout only.
- [ ] **PyTorch Policy:** Follow `POLICY-001`. PyTorch is mandatory; inventory tasks that inspect PyTorch run artifacts must honor `docs/workflows/pytorch.md` and the active findings in `docs/findings.md`.
- [ ] **Interpreter Policy:** Use PATH `python` in commands. Do not introduce repository-specific interpreter wrappers.
- [ ] **Long-Run Guardrail:** Long-running commands are not expected. If an unexpected long scan or validation is needed, use tmux with `ptycho311`, track the exact launched PID, and do not duplicate a run writing the same `--output-root`.

## Spec Alignment

- **Normative Roadmap Phase:** Phase 0 - Evidence Inventory.
- **Required durable outputs:**
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md` if no complete paper-grade `128x128` anchor is recovered. Carry-forward raw inventory currently reports no paper-grade or current `pinn_hybrid_resnet` CDI anchor, so this output is expected unless execution overturns that finding with complete provenance.
- **Allowed auxiliary output:** git-ignored raw inventory files under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/`.
- **Explicit non-goals:** no PDE benchmark selection, no full PDE training, no full CDI reruns, no expensive ablations, no manuscript prose, no generated paper-facing artifacts under `/home/ollie/Documents/neurips/`, no broad architecture sweep, no worktrees, no stable core model-module edits.

## Documents Read For This Plan Draft

- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-0-evidence-inventory/tranche-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/iterations/0/selected-paths.json`
- `docs/index.md`
- `docs/findings.md`
- `docs/plans/templates/implementation_plan.md`
- `docs/templates/` listing
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/workflows/pytorch.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/litsurvey.md`
- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- `workflows/examples/neurips_hybrid_resnet_plan_impl_review.yaml`
- `workflows/library/roadmap_seeded_plan_phase.yaml`
- `workflows/library/prompts/roadmap_seeded_plan_phase/draft_plan.md`
- `workflows/library/prompts/roadmap_seeded_plan_phase/review_plan.md`

## Implementation Architecture

Phase 0 is not a single implementation unit. It has multiple evidence boundaries whose outputs feed later roadmap phases, so broad shared-work tasks would make review and Phase 1 handoff unsafe.

### Unit 1: Preflight and Inventory Workspace

- **Owns:** current-checkout preflight, ignored raw artifact root, validation/reuse of carry-forward raw artifacts, `plan_path` pointer preservation, and explicit record of consumed docs.
- **Stable interfaces/artifacts:**
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
  - `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-0-evidence-inventory/plan-phase/plan_path.txt`
- **Must not own:** scientific classification, benchmark selection, or run execution.
- **Focused verification:** check raw artifact root is ignored, existing raw JSON parses when reused, and `plan_path.txt` still contains only the plan path.

### Unit 2: Local CDI N=128 Anchor Inventory

- **Owns:** discovery and provenance classification for existing `128x128` grid-lines Hybrid ResNet candidate runs.
- **Stable data structure:** records keyed by run path with fields `resolution`, `model_id`, `architecture`, `output_root`, `metrics_path`, `wrapper_invocation`, `runner_invocation`, `git_commit`, `seed`, `epochs`, `scheduler`, `dataset_split`, `config_summary`, `metric_summary`, `qualitative_outputs`, `freshness`, `evidence_grade`, and `blocking_gaps`.
- **Must not own:** baseline comparisons outside the same artifact context, full rerun execution, or CDI claim wording.
- **Dependency direction:** consumes Unit 1 workspace and existing `outputs/`, `.artifacts/`, and `artifacts/` files; produces raw records for Unit 6 regeneration-note routing and Unit 7 summary.
- **Focused verification:** every candidate marked paper-grade must have metrics, invocation/config, seed, commit or commit-gap, dataset/split, and qualitative-output provenance; if none qualify, the Unit 6 regeneration note must be required.

### Unit 3: Local CDI Baseline Inventory

- **Owns:** inventory of available or cheaply runnable CDI baselines, including CNN, FNO variants, classical CDI/PyNX/HIO/ER where applicable, and PtychoViT/PtyChi/Tike only when scientifically appropriate.
- **Stable data structure:** baseline records keyed by method and output path with `method_family`, `scientific_role`, `same_protocol_status`, `metrics_path`, `provenance_paths`, `required_caveats`, `evidence_grade`, and `blocking_gaps`.
- **Must not own:** deciding final paper rows or declaring protocol-compatible comparisons unless provenance supports it.
- **Dependency direction:** consumes local artifacts and the non-ML CDI benchmark design context; feeds Unit 7 summary and later Phase 3.
- **Focused verification:** baselines labeled "protocol-compatible" must name the comparator contract and source run path.

### Unit 4: N=256 Scaling Inventory

- **Owns:** discovery of existing N=256 runbooks/results, including `lines_256` docs, NERSC/cameraman N=256 runbooks, mode-count evidence, and memory/runtime notes.
- **Stable data structure:** N=256 records with `runbook_or_result_path`, `resolution`, `dataset_profile`, `mode_count`, `fixed_variables`, `metrics_path`, `runtime_or_memory_note`, `evidence_grade`, and `blocking_gaps`.
- **Must not own:** higher-mode selection or promotion of N=256 evidence to paper headline.
- **Dependency direction:** consumes `docs/studies/index.md`, `docs/studies/lines_256_*`, runbooks under `scripts/studies/runbooks/`, and local output artifacts; feeds later Phase 4 only.
- **Focused verification:** every N=256 entry must be labeled "secondary scaling context" or "not usable" in the summary.

### Unit 5: PDE Candidate and Environment Inventory

- **Owns:** neutral listing of at least three PDE/forward-modeling candidates and local environment constraints needed for Phase 1.
- **Stable data structure:** candidate records with `candidate_name`, `candidate_family`, `primary_source`, `task_type`, `architectural_fit_notes`, `benchmark_maturity_notes`, `metric_clarity`, `data_size_or_access`, `install_burden`, `rtx_3090_feasibility_notes`, `local_baseline_options`, `published_sota_pointer`, `license_or_access_notes`, `phase1_questions`, and `not_a_selection=true`.
- **Must not own:** primary/fallback selection, local installation, dataset download, training, or published-SOTA comparison claims.
- **Dependency direction:** consumes design criteria and primary benchmark sources; feeds Phase 1 scorecard.
- **Focused verification:** raw candidate records must contain every Unit 5 field, summary must contain at least three candidates, and summary must state that Phase 1 chooses primary/fallback.

### Unit 6: CDI Anchor Regeneration Note

- **Owns:** the conditional but currently expected `cdi_anchor_regeneration_plan.md` output that converts the lost `128x128` anchor condition into an executable later Phase 3 plan.
- **Stable interfaces/artifacts:**
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md`
  - raw `cdi_128_hybrid_candidates.json` gate statement as input evidence
  - existing Torch/grid-lines runbooks and current Hybrid ResNet baseline as command sources
  - runtime/runbook guidance from `docs/studies/index.md` and `tests/torch/test_grid_lines_hybrid_resnet_integration.py`
- **Must not own:** launching the regeneration run, running compact baselines or ablations, selecting paper table rows, or changing model/runbook behavior.
- **Dependency direction:** consumes Unit 2 classification plus `docs/model_baselines.md`, `docs/workflows/pytorch.md`, `docs/studies/index.md`, `tests/torch/test_grid_lines_hybrid_resnet_integration.py`, and existing grid-lines/Torch runbooks; feeds later Roadmap Phase 3.
- **Compatibility boundaries:** command examples must use PATH `python`; any future long run must use tmux with `ptycho311`, exact PID tracking, and unique output roots. Core physics/model files remain out of scope.
- **Focused verification:** note contains `128x128`, `Hybrid ResNet`, `regenerate`, `seed`, `metric`, `runtime`, a command/runbook source, config/provenance capture, qualitative output plan, and a later-phase boundary that says no run is launched in Phase 0.

### Unit 7: Durable Summary, Discoverability, and Gate Verification

- **Owns:** the tracked `evidence_inventory.md` summary, `docs/index.md` discoverability entries, gate status, and verification commands.
- **Stable interfaces/artifacts:**
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md` link when the note is created
  - `docs/index.md` entries for the evidence inventory and regeneration note, because these are durable project knowledge
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/*.json` links from the summary when useful
- **Must not own:** raw bulky logs, paper-facing `/home/ollie/Documents/neurips/` files, or later roadmap summaries.
- **Dependency direction:** consumes Units 2-6 outputs and writes final tracked docs.
- **Focused verification:** roadmap existence check plus structural checks for required sections, gate labels, lost-anchor/regeneration statement, and at least three PDE candidates.

## Context Priming Before Edits

Re-read these before executing the plan:

- `docs/index.md`
- `docs/findings.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/workflows/pytorch.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-0-evidence-inventory/tranche-context.md`

Required findings and policies to carry into classification:

- `POLICY-001`: PyTorch is mandatory.
- `CONFIG-001`: `update_legacy_dict(params.cfg, config)` is mandatory before legacy modules in PyTorch workflows.
- `PROBE-MASK-DEFAULT-001`: Hybrid ResNet recommended baseline keeps probe masking off unless a run explicitly records otherwise.
- `GRIDLINES-OBJECT-BIG-001` and `GRIDLINES-PROBE-BIG-001`: grid-lines Torch runner parity depends on `object_big=False` and `probe_big=False`.
- `REASSEMBLY-M-CONTRACT-001`: position reassembly has a single trim owner.
- `FORWARD-SIG-001` and `OUTPUT-COMPLEX-001`: FNO/Hybrid forward/output contracts affect interpretation of runner artifacts.
- `FNO-DEPTH-001`, `FNO-DEPTH-002`, and `STABLE-CRASH-DEPTH-001`: memory and stability notes matter when recording N=256 and higher-mode feasibility context.

## Phases

### Phase A: Preflight and Raw Inventory Workspace

**Files:**
- Modify: none
- Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/`

- [ ] A1: Verify the selected plan pointer and current checkout.

Run:

```bash
pwd
git status --short
sed -n '1p' state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-0-evidence-inventory/plan-phase/plan_path.txt
```

Expected: working directory is `/home/ollie/Documents/PtychoPINN`; unrelated dirty files are noted and not reverted; pointer prints `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-0-evidence-inventory/execution_plan.md`.

- [ ] A2: Create the ignored raw inventory root.

Run:

```bash
RAW_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory"
mkdir -p "${RAW_ROOT}"
git check-ignore -v "${RAW_ROOT}/raw_inventory_probe.json"
```

Expected: `git check-ignore` reports an ignore rule for `.artifacts/`.

- [ ] A3: Validate and reuse carry-forward raw inventory artifacts when present.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory")
known = [
    "environment_probe.json",
    "local_metrics_index.json",
    "cdi_128_hybrid_candidates.json",
    "cdi_baseline_candidates.json",
    "cdi_n256_candidates.json",
]
for name in known:
    path = root / name
    if path.exists():
        json.loads(path.read_text(encoding="utf-8"))
        print(f"reuse candidate parses: {path}")
    else:
        print(f"missing; regenerate if needed: {path}")
PY
```

Expected: existing carry-forward JSON parses. Reuse parsed artifacts if they match the current scope; regenerate only missing, stale, or structurally invalid files.

- [ ] A4: Capture non-expensive environment constraints for later notes if the carry-forward probe is missing or stale.

Run:

```bash
python - <<'PY' > .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/environment_probe.json
import importlib.util
import json
import platform
import shutil
import subprocess

payload = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages_available": {
        name: importlib.util.find_spec(name) is not None
        for name in ["torch", "lightning", "neuralop", "pdearena", "pynx"]
    },
}
try:
    import torch
    payload["torch"] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    payload["torch_error"] = repr(exc)
if shutil.which("nvidia-smi"):
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    payload["nvidia_smi"] = {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
```

Expected: command exits `0` and writes parseable JSON. If CUDA or `nvidia-smi` is unavailable, record that as an environment constraint, not as a Phase 0 failure.

### Phase B: Local CDI N=128 Hybrid ResNet Anchor Inventory

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/local_metrics_index.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_128_hybrid_candidates.json`
- Later modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`

- [ ] B1: Reuse or generate a local metrics and invocation index without interpreting evidence yet.

If `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/local_metrics_index.json` already exists, parses, and points at the current checkout's local artifact roots, reuse it. Otherwise run:

Run:

```bash
python - <<'PY' > .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/local_metrics_index.json
from pathlib import Path
import json

roots = [Path("outputs"), Path(".artifacts"), Path("artifacts")]
records = []
for root in roots:
    if not root.exists():
        continue
    for metrics_path in sorted(root.rglob("metrics.json")):
        run_root = metrics_path.parent
        wrapper_root = run_root.parent.parent if run_root.name.startswith("pinn_") and run_root.parent.name == "runs" else run_root
        record = {
            "metrics_path": metrics_path.as_posix(),
            "run_root": run_root.as_posix(),
            "wrapper_root": wrapper_root.as_posix(),
            "invocation_json_candidates": [
                p.as_posix()
                for p in [
                    wrapper_root / "invocation.json",
                    run_root / "invocation.json",
                    metrics_path.parent / "invocation.json",
                ]
                if p.exists()
            ],
            "invocation_sh_candidates": [
                p.as_posix()
                for p in [
                    wrapper_root / "invocation.sh",
                    run_root / "invocation.sh",
                    metrics_path.parent / "invocation.sh",
                ]
                if p.exists()
            ],
        }
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            record["metric_keys"] = sorted(payload.keys()) if isinstance(payload, dict) else []
            record["metric_payload_type"] = type(payload).__name__
        except Exception as exc:
            record["metrics_error"] = repr(exc)
        records.append(record)
print(json.dumps({"records": records}, indent=2, sort_keys=True))
PY
```

Expected: JSON contains local metrics candidates from `outputs/`, `.artifacts/`, and `artifacts/`.

- [ ] B2: Reuse or hand-classify plausible `128x128` Hybrid ResNet anchor candidates.

If the carry-forward `cdi_128_hybrid_candidates.json` parses, keep its existing records as the starting point and verify whether its gate statement is still valid. Use `local_metrics_index.json`, `docs/model_baselines.md`, and invocation artifacts to create or update `cdi_128_hybrid_candidates.json` with the Unit 2 data structure. Seed roots to inspect include, but are not limited to:

- `outputs/grid_lines_gs1_n128_e50_phi_all`
- `outputs/grid_lines_gs1_n128_e50_phi_all_rerun1`
- `outputs/grid_lines_gs1_n128_e20_phi_all`
- `outputs/grid_lines_gs1_n128_e20_phi_cnn_hybrid`
- `outputs/grid_lines_gs1_n128_tf1_torch50_neuralop_clip1_hybrid`
- `outputs/grid_lines_gs1_n128_tf1_torch50_neuralop_clip0_hybrid_log1p`

Classification rules:

- `paper-grade`: all required provenance exists and the run matches or explicitly documents deviations from the recommended Hybrid ResNet baseline.
- `decision-support`: useful metrics exist but one or more paper-grade provenance fields are missing.
- `not-usable`: missing metrics, corrupt payloads, incompatible protocol, or known contract violation.
- `unknown`: candidate needs manual inspection and must not satisfy the gate.

- [ ] B3: Record the CDI anchor gate statement and regeneration-note requirement.

In the draft inventory, state either:

- at least one plausible CDI anchor run path and its remaining gaps, or
- "No complete paper-grade CDI anchor identified; write `cdi_anchor_regeneration_plan.md` in Phase F and schedule a fresh regeneration run for Roadmap Phase 3" with the exact missing evidence that forces the rerun.

Do not launch the rerun in Phase 0.

### Phase C: Local CDI Baseline and Non-ML Comparator Inventory

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_baseline_candidates.json`
- Later modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`

- [ ] C1: Inventory neural local baselines from the local metrics index.

If the carry-forward `cdi_baseline_candidates.json` parses, reuse it as the starting point and refresh only entries made stale by newly discovered metrics or provenance.

Use `local_metrics_index.json` to locate CNN/FNO/Hybrid/Stability variants and classify their protocol relationship to the CDI anchor candidates. Required baseline families to check:

- CNN or TensorFlow `pinn` baselines.
- Torch `pinn_fno`, `pinn_hybrid`, `pinn_stable_hybrid`, and `pinn_fno_vanilla` variants when present.
- Any wrapper-level `metrics_by_model.json` that records multiple model entries.

- [ ] C2: Inventory classical or non-ML CDI evidence without turning it into a NeurIPS paper row.

Inspect these sources and record only applicable, provenance-backed entries:

- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- `scripts/reconstruction/hio_cdi_benchmark.py`
- `.artifacts/sim_lines_4x_metrics_2026-01-27/`
- `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/`
- `artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/`

Classification must label whether a non-ML row is same-protocol, historical context, exploratory, or not scientifically appropriate for the NeurIPS CDI anchor.

- [ ] C3: Record baseline gaps and cheap-runnable candidates.

In `cdi_baseline_candidates.json`, include a `cheap_rerun_possible` boolean and `required_later_phase` field. Do not run any baseline in Phase 0.

### Phase D: Existing N=256 CDI Scaling Inventory

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_n256_candidates.json`
- Later modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`

- [ ] D1: Inventory N=256 runbooks and known contracts.

If the carry-forward `cdi_n256_candidates.json` parses, reuse it as the starting point and refresh only if the cited runbooks or local output roots have changed materially.

Read and cite relevant N=256 sources from:

- `docs/studies/index.md`
- `docs/studies/lines_256_dataset.md`
- `docs/studies/lines_256_arch_improvement_loop.md`
- `docs/studies/lines_256_controller_loop.md`
- `scripts/studies/run_lines_256_arch_experiment.py`
- `scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n256.py`
- `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py`

- [ ] D2: Inventory existing N=256 outputs and mode-count hints.

Use local metrics and state files to record available evidence. Include mode counts only when they are visible in invocation/config/manifest artifacts, not inferred from directory names alone.

- [ ] D3: Record N=256 as secondary scaling context.

Every N=256 entry in the summary must state that it is secondary scaling evidence and that higher-mode selection remains a Phase 4 question.

### Phase E: PDE Candidate and Local Constraint Inventory

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/pde_candidate_inventory.json`
- Later modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`

- [ ] E1: Define the Phase 0 PDE candidate record schema.

Create a JSON file with at least three candidates. Use the Unit 5 fields and include `not_a_selection: true` on every record.

- [ ] E2: Inventory candidate families neutrally.

Include at least one candidate from each applicable design bucket where primary-source information can be found:

- PDEBench/PDEArena-style fluids or operator-learning benchmark.
- Wave-equation or wave-propagation benchmark.
- Inverse-scattering or related wave/inverse-problem benchmark.

Use official benchmark docs, papers, or repository metadata as sources. Record source URLs and access dates if internet research is used. Do not install dependencies, download datasets, or run training.

- [ ] E3: Record local environment constraints for Phase 1.

Use `environment_probe.json` and local disk/GPU facts to record install burden and RTX 3090 feasibility notes. If a package is not installed, mark it as "not installed locally" rather than adding it.

- [ ] E4: Add Phase 1 handoff questions.

For each candidate, record the exact questions Phase 1 must answer before primary/fallback selection, such as metric protocol, dataset size, baseline availability, smoke-run budget, and published-SOTA caveat requirements.

### Phase F: Write the CDI Anchor Regeneration Note When Required

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md`
- Consume/link: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_128_hybrid_candidates.json`

- [ ] F1: Decide whether the regeneration note is required from the N=128 anchor classification.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

path = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_128_hybrid_candidates.json")
payload = json.loads(path.read_text(encoding="utf-8"))
records = payload.get("records", [])
paper_grade = [r for r in records if str(r.get("evidence_grade", "")).lower() == "paper-grade"]
if paper_grade:
    print(f"paper-grade anchor recovered: {len(paper_grade)}")
else:
    print("no paper-grade anchor recovered; regeneration note required")
PY
```

Expected for current carry-forward context: `no paper-grade anchor recovered; regeneration note required`.

- [ ] F2: Draft `cdi_anchor_regeneration_plan.md` if no paper-grade anchor is recovered.

Use this structure unless a clearer equivalent is needed:

```markdown
# NeurIPS Hybrid ResNet CDI Anchor Regeneration Plan

## Scope and Trigger
## Lost-Run Evidence
## Regeneration Command / Runbook Source
## Baseline Configuration
## Dataset and Split Identity
## Seed, Config, and Provenance Capture
## Metric Contract
## Qualitative Output Plan
## Runtime and Resource Budget
## Freshness and Output-Root Guardrails
## Later-Phase Boundary
## Verification
```

Minimum content requirements:

- State that the `128x128` grid-lines Hybrid ResNet anchor is not recovered as paper-grade evidence unless execution found complete contrary evidence.
- Name the current recommended Hybrid ResNet baseline from `docs/model_baselines.md`, including `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `hybrid_resnet_blocks=6`, `probe_mask=off`, MAE loss, and `ReduceLROnPlateau` schedule unless a later Phase 3 plan explicitly overrides them.
- Identify the existing Torch/grid-lines path to start from, such as `scripts/studies/grid_lines_compare_wrapper.py` routing to `scripts/studies/grid_lines_torch_runner.py`, and any known wrapper/runbook source used to derive the command. Use `docs/studies/index.md` and `tests/torch/test_grid_lines_hybrid_resnet_integration.py` as runtime guidance before estimating the future Phase 3 command duration, dataset generation steps, or required preflight.
- Specify seed/config/provenance capture: invocation artifacts, git commit, environment/GPU, dataset path and split identity, model config, seed, metrics path, qualitative outputs, and stdout/stderr log path.
- Include a runtime budget estimate or an explicit "Phase 3 must preflight runtime before launch" placeholder if exact duration cannot be known in Phase 0.
- State that Phase 0 does not launch the regeneration, baselines, ablations, or paper-facing artifact assembly.

- [ ] F3: Verify the regeneration note when it is required.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json
import re
import shlex

from scripts.studies.grid_lines_compare_wrapper import parse_args

raw = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/cdi_128_hybrid_candidates.json")
records = json.loads(raw.read_text(encoding="utf-8")).get("records", [])
paper_grade = [r for r in records if str(r.get("evidence_grade", "")).lower() == "paper-grade"]
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md")
if not paper_grade:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required_terms = ["128x128", "Hybrid ResNet", "regenerate", "seed", "metric", "runtime"]
    missing = [term for term in required_terms if term.lower() not in text.lower()]
    if missing:
        raise SystemExit(f"regeneration plan missing required terms: {missing}")
    command_blocks = re.findall(r"```bash\n(.*?)\n```", text, flags=re.S)
    wrapper_commands = [block for block in command_blocks if "grid_lines_compare_wrapper.py" in block]
    if len(wrapper_commands) != 1:
        raise SystemExit(f"expected one wrapper command block, found {len(wrapper_commands)}")
    command = re.sub(r"\\\s*\n\s*", " ", wrapper_commands[0])
    argv = shlex.split(command)
    expected_prefix = ["python", "scripts/studies/grid_lines_compare_wrapper.py"]
    if argv[:2] != expected_prefix:
        raise SystemExit(f"unexpected wrapper command prefix: {argv[:2]}")
    parse_args(argv[2:])
print("regeneration-note gate satisfied")
PY
```

Expected: `regeneration-note gate satisfied`.

### Phase G: Write Durable Evidence Inventory and Discoverability Update

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
- Modify: `docs/index.md`
- Consume/link: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/*.json`

- [ ] G1: Draft `evidence_inventory.md` with the required structure.

Use this structure unless a clearer equivalent is needed:

```markdown
# NeurIPS Hybrid ResNet Phase 0 Evidence Inventory

## Scope and Sources
## Gate Status
## CDI N=128 Anchor Candidates
## CDI Baseline Inventory
## N=256 Secondary Scaling Inventory
## PDE Candidate Inventory for Phase 1
## Environment Constraints
## Raw Artifact Links
## CDI Anchor Regeneration Note
## Carry-Forward Notes
## Non-Goals Confirmed
```

Minimum content requirements:

- Gate status explicitly says pass, blocked, or pass-with-gaps for each Phase 0 gate.
- CDI N=128 section distinguishes paper-grade, decision-support-only, not-usable, and unknown candidates.
- CDI N=128 section links `cdi_anchor_regeneration_plan.md` when no complete paper-grade anchor is recovered.
- Baseline section records local baselines and any non-ML/classical CDI caveats.
- N=256 section labels all entries secondary.
- PDE section contains at least three candidates and states that Phase 1 selects primary/fallback.
- Raw artifact links point to ignored `.artifacts/...` files only when useful; do not paste bulky JSON into tracked docs.

- [ ] G2: Update `docs/index.md` for discoverability.

Add concise entries for `plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md` and, when created, `plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md` near the existing NeurIPS design and roadmap entries. Do not rewrite unrelated index content.

- [ ] G3: Verify Markdown references and artifact boundaries.

Run:

```bash
python - <<'PY'
from pathlib import Path

inventory = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md")
text = inventory.read_text(encoding="utf-8")
required_terms = [
    "CDI N=128",
    "paper-grade",
    "decision-support",
    "N=256",
    "secondary",
    "PDE",
    "Phase 1",
    "not select",
]
missing = [term for term in required_terms if term.lower() not in text.lower()]
if missing:
    raise SystemExit(f"inventory missing required terms: {missing}")
print("inventory required terms present")
PY
```

Expected: command exits `0`.

### Phase H: Gate Verification and Handoff

**Files:**
- Verify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
- Verify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md` when required
- Verify: `docs/index.md`
- Verify: raw JSON under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/`

- [ ] H1: Run the roadmap-required artifact existence check.

Run:

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

Expected: `inventory docs present`.

- [ ] H2: Run structural gate checks.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory")
inventory = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md")
text = inventory.read_text(encoding="utf-8").lower()

def load_strict_json(path: Path):
    def reject_constant(value):
        raise ValueError(f"{path}: non-strict JSON constant {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)

for name in [
    "environment_probe.json",
    "local_metrics_index.json",
    "cdi_128_hybrid_candidates.json",
    "cdi_baseline_candidates.json",
    "cdi_n256_candidates.json",
    "pde_candidate_inventory.json",
]:
    path = root / name
    if not path.exists():
        raise SystemExit(f"missing raw inventory artifact: {path}")
    load_strict_json(path)

pde = load_strict_json(root / "pde_candidate_inventory.json")
candidates = pde.get("candidates", pde if isinstance(pde, list) else [])
if len(candidates) < 3:
    raise SystemExit(f"expected at least 3 PDE candidates, found {len(candidates)}")
required_candidate_fields = {
    "candidate_name",
    "candidate_family",
    "primary_source",
    "task_type",
    "architectural_fit_notes",
    "benchmark_maturity_notes",
    "metric_clarity",
    "data_size_or_access",
    "install_burden",
    "rtx_3090_feasibility_notes",
    "local_baseline_options",
    "published_sota_pointer",
    "license_or_access_notes",
    "phase1_questions",
    "not_a_selection",
}
for idx, candidate in enumerate(candidates):
    missing = sorted(required_candidate_fields - set(candidate))
    if missing:
        raise SystemExit(f"PDE candidate {idx} missing Unit 5 fields: {missing}")
    if candidate.get("not_a_selection") is not True:
        raise SystemExit(f"PDE candidate {idx} must set not_a_selection=true")
if "primary" in text and "phase 1" not in text:
    raise SystemExit("inventory appears to mention primary without Phase 1 boundary")
print("phase 0 structural gates passed")
PY
```

Expected: `phase 0 structural gates passed`.

- [ ] H3: Run the roadmap-required regeneration-note check when no complete anchor is recovered.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json
import re
import shlex

from scripts.studies.grid_lines_compare_wrapper import parse_args

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory")
candidates = json.loads((root / "cdi_128_hybrid_candidates.json").read_text(encoding="utf-8"))
records = candidates.get("records", [])
paper_grade = [r for r in records if str(r.get("evidence_grade", "")).lower() == "paper-grade"]
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md")
if not paper_grade:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required_terms = ["128x128", "Hybrid ResNet", "regenerate", "seed", "metric", "runtime"]
    missing = [term for term in required_terms if term.lower() not in text.lower()]
    if missing:
        raise SystemExit(f"regeneration plan missing required terms: {missing}")
    command_blocks = re.findall(r"```bash\n(.*?)\n```", text, flags=re.S)
    wrapper_commands = [block for block in command_blocks if "grid_lines_compare_wrapper.py" in block]
    if len(wrapper_commands) != 1:
        raise SystemExit(f"expected one wrapper command block, found {len(wrapper_commands)}")
    command = re.sub(r"\\\s*\n\s*", " ", wrapper_commands[0])
    argv = shlex.split(command)
    expected_prefix = ["python", "scripts/studies/grid_lines_compare_wrapper.py"]
    if argv[:2] != expected_prefix:
        raise SystemExit(f"unexpected wrapper command prefix: {argv[:2]}")
    parse_args(argv[2:])
print("regeneration plan gate passed")
PY
```

Expected: `regeneration plan gate passed`.

- [ ] H4: Verify discoverability entry.

Run:

```bash
python - <<'PY'
from pathlib import Path
idx = Path("docs/index.md").read_text(encoding="utf-8")
target = "plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md"
if target not in idx:
    raise SystemExit(f"docs/index.md missing {target}")
regen = "plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md"
if Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md").exists() and regen not in idx:
    raise SystemExit(f"docs/index.md missing {regen}")
print("docs index references evidence inventory")
PY
```

Expected: `docs index references evidence inventory`.

- [ ] H5: Confirm `plan_path.txt` remains pointer-only.

Run:

```bash
python - <<'PY'
from pathlib import Path
path = Path("state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-0-evidence-inventory/plan-phase/plan_path.txt")
lines = path.read_text(encoding="utf-8").splitlines()
expected = "docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-0-evidence-inventory/execution_plan.md"
if lines != [expected]:
    raise SystemExit(f"plan_path.txt must contain only {expected!r}; got {lines!r}")
print("plan_path pointer intact")
PY
```

Expected: `plan_path pointer intact`.

## Workflow Compatibility Contract

When this plan is executed by the roadmap tranche-drain workflow:

- Backlog/workflow-selected plan path is `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-0-evidence-inventory/execution_plan.md`.
- Execution unit is this full Phase 0 tranche plan.
- The implementation phase must produce `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`.
- The implementation phase must produce `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md` if no complete paper-grade `128x128` anchor is recovered.
- The implementation phase may produce ignored raw inventory artifacts under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/`.
- Completion requires both verification commands passing and the Phase 0 gate statements present in the inventory summary.

## Compatibility and Migration Boundaries

- No migrations are authorized.
- Do not alter data contracts, model APIs, Torch runner behavior, or stable core model/physics modules.
- Do not introduce a reusable inventory helper script unless a follow-up plan adds tests for its schema and CLI behavior.
- Do not modify generated paper-facing assets or `/home/ollie/Documents/neurips/`.
- `docs/index.md` may be modified only to add discoverability for the new inventory document and the conditional CDI anchor regeneration note.

## Verification Commands

Run these after implementation:

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

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory")
def load_strict_json(path: Path):
    def reject_constant(value):
        raise ValueError(f"{path}: non-strict JSON constant {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)

for name in [
    "environment_probe.json",
    "local_metrics_index.json",
    "cdi_128_hybrid_candidates.json",
    "cdi_baseline_candidates.json",
    "cdi_n256_candidates.json",
    "pde_candidate_inventory.json",
]:
    path = root / name
    if not path.exists():
        raise SystemExit(f"missing raw inventory artifact: {path}")
    load_strict_json(path)
pde = load_strict_json(root / "pde_candidate_inventory.json")
candidates = pde.get("candidates", pde if isinstance(pde, list) else [])
required_candidate_fields = {
    "candidate_name",
    "candidate_family",
    "primary_source",
    "task_type",
    "architectural_fit_notes",
    "benchmark_maturity_notes",
    "metric_clarity",
    "data_size_or_access",
    "install_burden",
    "rtx_3090_feasibility_notes",
    "local_baseline_options",
    "published_sota_pointer",
    "license_or_access_notes",
    "phase1_questions",
    "not_a_selection",
}
for idx, candidate in enumerate(candidates):
    missing = sorted(required_candidate_fields - set(candidate))
    if missing:
        raise SystemExit(f"PDE candidate {idx} missing Unit 5 fields: {missing}")
    if candidate.get("not_a_selection") is not True:
        raise SystemExit(f"PDE candidate {idx} must set not_a_selection=true")
print("raw inventory strict JSON parses and Unit 5 PDE schema passes")
PY
```

```bash
python - <<'PY'
from pathlib import Path
text = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md").read_text(encoding="utf-8").lower()
required = ["paper-grade", "decision-support", "cdi n=128", "n=256", "secondary", "pde", "phase 1"]
missing = [term for term in required if term not in text]
if missing:
    raise SystemExit(f"inventory missing required gate language: {missing}")
print("inventory gate language present")
PY
```

```bash
python - <<'PY'
from pathlib import Path
import json
import re
import shlex

from scripts.studies.grid_lines_compare_wrapper import parse_args

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory")
raw = root / "cdi_128_hybrid_candidates.json"
if raw.exists():
    records = json.loads(raw.read_text(encoding="utf-8")).get("records", [])
    paper_grade = [r for r in records if str(r.get("evidence_grade", "")).lower() == "paper-grade"]
else:
    paper_grade = []
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md")
if not paper_grade:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required_terms = ["128x128", "Hybrid ResNet", "regenerate", "seed", "metric", "runtime"]
    missing = [term for term in required_terms if term.lower() not in text.lower()]
    if missing:
        raise SystemExit(f"regeneration plan missing required terms: {missing}")
    command_blocks = re.findall(r"```bash\n(.*?)\n```", text, flags=re.S)
    wrapper_commands = [block for block in command_blocks if "grid_lines_compare_wrapper.py" in block]
    if len(wrapper_commands) != 1:
        raise SystemExit(f"expected one wrapper command block, found {len(wrapper_commands)}")
    command = re.sub(r"\\\s*\n\s*", " ", wrapper_commands[0])
    argv = shlex.split(command)
    expected_prefix = ["python", "scripts/studies/grid_lines_compare_wrapper.py"]
    if argv[:2] != expected_prefix:
        raise SystemExit(f"unexpected wrapper command prefix: {argv[:2]}")
    parse_args(argv[2:])
print("regeneration-note requirement satisfied")
PY
```

```bash
python - <<'PY'
from pathlib import Path
idx = Path("docs/index.md").read_text(encoding="utf-8")
target = "plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md"
if target not in idx:
    raise SystemExit(f"docs/index.md missing {target}")
regen = "plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md"
if Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md").exists() and regen not in idx:
    raise SystemExit(f"docs/index.md missing {regen}")
print("docs index references evidence inventory")
PY
```

## Completion Criteria

- [ ] `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md` exists and contains the Phase 0 gate status.
- [ ] The inventory distinguishes paper-grade evidence from decision-support-only, not-usable, and unknown artifacts.
- [ ] At least one plausible CDI anchor run is identified, or a fresh regeneration run is explicitly scheduled for a later tranche.
- [ ] If no complete paper-grade `128x128` anchor is recovered, `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md` exists and records command/runbook, seed/config/provenance capture, metric contract, qualitative output plan, and runtime budget.
- [ ] Local CDI baselines are inventoried with protocol caveats and no unsupported paper-row claims.
- [ ] Existing N=256 evidence is inventoried and labeled as secondary scaling context.
- [ ] At least three PDE/forward-modeling candidates are listed for Phase 1 without selecting primary/fallback.
- [ ] Raw JSON inventory artifacts exist under an ignored artifact root and parse successfully under strict JSON parsing.
- [ ] `docs/index.md` references the new evidence inventory document and, when created, the CDI anchor regeneration note.
- [ ] No paper-facing artifacts under `/home/ollie/Documents/neurips/` are created.
- [ ] No expensive reruns, full training jobs, broad architecture sweeps, worktrees, or stable core-module edits occur.

## Artifacts Index

- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-0-evidence-inventory/execution_plan.md`
- Durable inventory summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
- CDI anchor regeneration note: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md`
- Raw inventory root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/`
- Workflow execution report target: `artifacts/work/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory-execution-report.md`
- Plan review report target: `artifacts/review/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory-plan-review.json`
- Implementation review report target: `artifacts/review/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory-implementation-review.md`
