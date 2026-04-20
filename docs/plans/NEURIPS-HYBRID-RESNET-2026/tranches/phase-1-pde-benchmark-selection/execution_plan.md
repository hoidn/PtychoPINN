# Phase 1 PDE Benchmark Selection Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the Roadmap Phase 1 PDE/forward-modeling benchmark selection document for the NeurIPS Hybrid ResNet campaign, naming exactly one primary deep benchmark and one fallback from the neutral Phase 0 candidates.

**Architecture:** Keep this tranche as a selection and handoff boundary. Use Phase 0 candidate inventory plus primary benchmark sources to define a scorecard, collect lightweight feasibility evidence, score PDEBench fluids, PDEArena Maxwell-3D, and OpenFWI 2D acoustic FWI, then write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md` with the primary/fallback decision and Phase 2 handoff. Do not implement adapters, install production dependencies, download full datasets, run PDE training, polish CDI evidence, or create `/home/ollie/Documents/neurips/` artifacts.

**Tech Stack:** Python 3.11 via PATH `python`, Markdown, JSON, primary benchmark sources, existing Phase 0 raw inventory, non-invasive environment checks, optional small source probes, no new production dependencies.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Title: Phase 1 Required PDE Benchmark Selection
- Status: pending
- Spec/Source:
  - Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
  - Roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - Tranche context: `state/NEURIPS-HYBRID-RESNET-2026/items/phase-1-pde-benchmark-selection/tranche-context.md`
- Plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-1-pde-benchmark-selection/execution_plan.md`

## Compliance Matrix

- [ ] **Roadmap Phase Order:** Execute only Roadmap Phase 1. Phase 0 is complete; do not start Phase 2 PDE execution, Phase 3 CDI anchor regeneration, Phase 4 `256x256` scaling, or Phase 5 paper-facing evidence assembly.
- [ ] **Required PDE Pillar:** The selection must preserve the design requirement for one deep PDE/forward-modeling benchmark plus one fallback.
- [ ] **Neutral Screen:** Score all three Phase 0 candidates before naming primary/fallback: PDEBench 2D fluids, PDEArena Maxwell-3D, and OpenFWI 2D acoustic FWI.
- [ ] **Selection Gate:** The primary must have clear metrics, runnable data access, small-smoke feasibility on the local RTX 3090, feasible local baselines, and a defensible spectral/local Hybrid ResNet paper story. If no candidate satisfies this, mark Phase 1 blocked and escalate instead of forcing a weak selection.
- [ ] **Published SOTA Boundary:** Published SOTA may be used only with explicit protocol caveats. The selection must still identify at least two feasible local baselines, or justify one strong local baseline plus published SOTA.
- [ ] **External Dependency Boundary:** External benchmark dependencies and datasets are permitted only as documented Phase 2 candidates. Do not add production dependencies, run package installs that modify the repo, or download full datasets in this tranche.
- [ ] **Artifact Hygiene:** Store raw scorecards and source notes under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/`. Keep tracked docs concise and human-readable.
- [ ] **Discoverability:** Because this tranche creates a durable selection document that controls later roadmap work, update `docs/index.md` with a concise entry for `pde_benchmark_selection.md`.
- [ ] **Stable Modules:** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- [ ] **No Worktrees:** Use the current checkout only.
- [ ] **Interpreter Policy:** Use PATH `python` in commands. Do not introduce repository-specific interpreter wrappers.
- [ ] **Long-Run Guardrail:** Long-running commands are not expected. If a bounded source probe unexpectedly becomes long-running, stop and replace it with documented source metadata; do not launch duplicate runs or broad polling loops.

## Spec Alignment

- **Normative roadmap phase:** Phase 1 - Required PDE Benchmark Selection.
- **Required durable output:** `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`.
- **Expected raw/supporting outputs:**
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/pdebench_fluids.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/pdearena_maxwell3d.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/openfwi_2d_acoustic_fwi.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/environment_recheck.json`
- **Roadmap verification command:** the structural check in the Phase 1 gate must pass and print `selection doc contains required decision fields`.
- **Explicit non-goals:** no PDE adapter, metric parser, result writer, production dependency addition, full dataset download, PDE training, local baseline training, Phase 2 ablation, CDI anchor regeneration, CDI baseline rerun, `256x256` scaling run, manuscript prose, `/home/ollie/Documents/neurips/` artifact, worktree, or stable core model-module edit.

## Documents Read For This Plan Draft

- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-1-pde-benchmark-selection/tranche-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-1-pde-benchmark-selection/plan-phase/plan_path.txt`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/index.md`
- `docs/templates/` listing
- `docs/plans/templates/implementation_plan.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/findings.md`
- `docs/model_baselines.md`
- `docs/workflows/pytorch.md`
- `docs/studies/index.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/litsurvey.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-0-evidence-inventory/execution_plan.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/pde_candidate_inventory.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/environment_probe.json`

## Implementation Architecture

This tranche needs an Implementation Architecture section because it crosses external benchmark sources, local feasibility constraints, raw ignored artifacts, durable tracked docs, and the Phase 2 execution handoff. A single "write selection doc" task would hide source ownership, scoring ownership, and handoff contracts that later phases depend on.

### Unit 1: Scope, Gate, and Workspace Preflight

- **Owns:** current checkout preflight, Phase 0 completion check, absence/presence check for the Phase 1 durable output, ignored artifact root creation, and preservation of the `plan_path.txt` contract.
- **Stable interfaces/artifacts:**
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  - `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-1-pde-benchmark-selection/plan-phase/plan_path.txt`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/`
- **Must not own:** candidate scoring, primary/fallback selection, dependency installation, or benchmark execution.
- **Dependency direction:** provides validated inputs and workspace for all later units.
- **Focused tests:** `git check-ignore` for the raw artifact root, JSON parse checks for progress/candidate inventory, and exact `plan_path.txt` content check.

### Unit 2: Scorecard Schema and Decision Rubric

- **Owns:** the scorecard fields and rubric used consistently across all candidates.
- **Stable data structure:** `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json` with:
  - `schema_version`
  - `rubric`
  - `candidates[]`
  - `decision`
  - `phase2_handoff`
- **Required candidate fields:** `candidate_id`, `candidate_name`, `family`, `task_type`, `architectural_fit`, `benchmark_maturity`, `metric_clarity`, `data_access`, `data_size`, `install_burden`, `license_access`, `rtx_3090_feasibility`, `local_baseline_feasibility`, `published_sota_availability`, `paper_story_fit`, `risks`, `required_phase2_preflight`, `recommended_status`, and `source_notes_path`.
- **Decision rule:** use gate-first selection, not a hidden weighted total. A candidate can become primary only if metric clarity, data access, RTX 3090 smoke feasibility, and baseline feasibility are all acceptable or explicitly bounded. Use score totals only as a secondary explanation.
- **Must not own:** source collection, candidate-specific interpretation, or final prose beyond schema definitions.
- **Dependency direction:** consumes Unit 1 inputs and defines the contract that Units 3-5 must fill.
- **Focused tests:** JSON structural validation ensures all required candidate fields exist for exactly the three Phase 0 candidates.

### Unit 3: Primary Source and Feasibility Notes

- **Owns:** concise source notes and non-invasive feasibility checks for each candidate.
- **Stable artifacts:**
  - `source_notes/pdebench_fluids.md`
  - `source_notes/pdearena_maxwell3d.md`
  - `source_notes/openfwi_2d_acoustic_fwi.md`
  - `environment_recheck.json`
- **Source boundaries:** use primary sources where possible: official papers, repositories, documentation, dataset pages, and license files. Avoid secondary summaries unless a primary source is unavailable, and label any inference from sources.
- **Must not own:** installing `pdearena`, cloning full repositories, downloading full datasets, running training, or deciding primary/fallback.
- **Dependency direction:** consumes Phase 0 candidate inventory and provides evidence for Unit 4 scoring.
- **Focused tests:** each source note must include source URLs, access date, metric/split evidence, data/license evidence, local feasibility implications, local baseline options, published-SOTA caveats, and open risks.

### Unit 4: Candidate Evaluation and Scorecard Population

- **Owns:** applying the Unit 2 schema to all three candidates using Unit 3 source notes and Phase 0 environment constraints.
- **Stable artifact:** populated `pde_scorecard.json`.
- **Must not own:** tracked Markdown selection prose or `docs/index.md` discoverability.
- **Dependency direction:** consumes Units 2-3 and feeds Unit 5.
- **Compatibility boundaries:** do not retroactively treat Phase 0 inventory as a selection; do not reject candidates for CDI-only reasons; do not promote a benchmark if Phase 2 would require broad architecture changes beyond the roadmap.
- **Focused tests:** parse JSON and assert exactly one tentative `primary` and one tentative `fallback`, unless `decision.status` is `blocked`.

### Unit 5: Durable Selection Document and Phase 2 Handoff

- **Owns:** tracked `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`.
- **Required sections:** scope, documents/sources used, scorecard schema, candidate scorecard, primary decision, fallback decision, rejection rationale for non-selected candidates, metric contract, data access plan, baseline plan, published-SOTA caveats, Phase 2 handoff, pivot/blocked conditions, and non-goals.
- **Must not own:** raw source-note bulk, new code, benchmark adapters, or paper-facing evidence artifacts.
- **Dependency direction:** consumes Unit 4 scorecard and writes the durable Phase 1 gate output.
- **Focused tests:** roadmap structural verification plus checks for the three candidate names, primary/fallback labels, local baselines, metric contract, data access, license/access notes, SOTA caveats, and Phase 2 smoke-run handoff.

### Unit 6: Discoverability and Final Verification

- **Owns:** `docs/index.md` entry for the new selection document, final structural validation, and output-contract preservation.
- **Stable interfaces/artifacts:**
  - `docs/index.md`
  - `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-1-pde-benchmark-selection/plan-phase/plan_path.txt`
- **Must not own:** selection content beyond linking/describing it.
- **Dependency direction:** runs after Unit 5 so the index can describe the final durable document.
- **Focused tests:** index contains a concise `pde_benchmark_selection.md` entry; `plan_path.txt` contains only the execution plan path; no `/home/ollie/Documents/neurips/` outputs were created by this tranche.

## Compatibility, Migration, and Boundary Notes

- No migrations are planned.
- No production dependency changes are planned. If a benchmark appears to require a dependency such as `pdearena`, document it as a Phase 2 setup requirement instead of installing it in Phase 1.
- The Phase 1 selection document must remain compatible with Phase 0 raw candidate inventory and must not rewrite Phase 0 findings.
- The selected primary benchmark must be framed as one deep PDE/forward-modeling benchmark, not a broad benchmark suite.
- Published SOTA comparisons must remain explicitly protocol-dependent unless local reproduction uses the same code/data/protocol.
- Keep `/home/ollie/Documents/neurips/` untouched; it belongs to Roadmap Phase 5.
- Keep CDI anchor regeneration and `256x256` scaling behind later roadmap gates.

## Context Priming Before Edits

Re-read these before executing the plan:

- `docs/index.md`
- `docs/findings.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-1-pde-benchmark-selection/tranche-context.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/pde_candidate_inventory.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/environment_probe.json`

Required findings and policies to keep in scope:

- `POLICY-001`: PyTorch is mandatory.
- `CONFIG-001`: if any PyTorch/legacy workflow is inspected or planned for Phase 2, `update_legacy_dict(params.cfg, config)` remains mandatory before legacy modules.
- `PROBE-MASK-DEFAULT-001`, `GRIDLINES-OBJECT-BIG-001`, `GRIDLINES-PROBE-BIG-001`, `FORWARD-SIG-001`, and `OUTPUT-COMPLEX-001`: CDI Hybrid ResNet run contracts matter for paper-story consistency, but they do not authorize CDI execution in this tranche.
- `FNO-DEPTH-001`, `FNO-DEPTH-002`, and `STABLE-CRASH-DEPTH-001`: use these as cautionary memory/stability context when judging RTX 3090 feasibility for spectral models.

## Phases

### Phase A: Scope and Workspace Preflight

**Files:**
- Read: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Read: `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-1-pde-benchmark-selection/plan-phase/plan_path.txt`
- Read: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/pde_candidate_inventory.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/`

- [ ] A1: Verify the current checkout and selected plan pointer.

Run:

```bash
pwd
git status --short
sed -n '1p' state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-1-pde-benchmark-selection/plan-phase/plan_path.txt
```

Expected: working directory is `/home/ollie/Documents/PtychoPINN`; unrelated dirty files are noted and not reverted; pointer prints `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-1-pde-benchmark-selection/execution_plan.md`.

- [ ] A2: Confirm Phase 0 is complete and Phase 1 is not already satisfied.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

ledger = json.loads(Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json").read_text(encoding="utf-8"))
assert "phase-0-evidence-inventory" in ledger.get("completed_tranches", []), "Phase 0 is not marked complete"
assert "phase-1-pde-benchmark-selection" not in ledger.get("completed_tranches", []), "Phase 1 already marked complete"
assert Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md").exists(), "missing Phase 0 evidence inventory"
assert Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md").exists(), "missing Phase 0 CDI regeneration note"
selection = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md")
print(f"selection_doc_exists={selection.exists()}")
PY
```

Expected: command exits `0`. If `selection_doc_exists=True`, inspect the document and update it only if it is incomplete against this plan; do not duplicate it.

- [ ] A3: Create the ignored Phase 1 artifact workspace.

Run:

```bash
RAW_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection"
mkdir -p "${RAW_ROOT}/source_notes"
git check-ignore -v "${RAW_ROOT}/probe.json"
```

Expected: `git check-ignore` reports an ignore rule for `.artifacts/`.

- [ ] A4: Validate Phase 0 candidate inventory before using it.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

path = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/pde_candidate_inventory.json")
payload = json.loads(path.read_text(encoding="utf-8"))
names = [candidate["candidate_name"] for candidate in payload["candidates"]]
required = {
    "PDEBench 2D incompressible Navier-Stokes or compressible fluid task",
    "PDEArena Maxwell-3D",
    "OpenFWI 2D acoustic full waveform inversion",
}
missing = required.difference(names)
if missing:
    raise SystemExit(f"Phase 0 PDE candidate inventory missing required candidates: {sorted(missing)}")
if any(not candidate.get("not_a_selection") for candidate in payload["candidates"]):
    raise SystemExit("Phase 0 candidate inventory contains a selected candidate; inspect before proceeding")
print("Phase 0 PDE candidate inventory is neutral and complete")
PY
```

Expected: prints `Phase 0 PDE candidate inventory is neutral and complete`.

### Phase B: Define the Scorecard Schema

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json`

- [ ] B1: Draft the scorecard schema and rubric before scoring candidates.

Create `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json` with this top-level shape:

```json
{
  "schema_version": 1,
  "rubric": {
    "scale": "0=unacceptable, 1=high risk, 2=acceptable with caveats, 3=strong",
    "decision_rule": "gate-first; primary requires acceptable metric clarity, runnable data access, RTX 3090 smoke feasibility, and baseline feasibility",
    "required_candidates": [
      "pdebench_fluids",
      "pdearena_maxwell3d",
      "openfwi_2d_acoustic_fwi"
    ]
  },
  "candidates": [],
  "decision": {
    "status": "pending",
    "primary_candidate_id": null,
    "fallback_candidate_id": null,
    "blocked_reason": null
  },
  "phase2_handoff": {}
}
```

- [ ] B2: Validate the empty schema.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

path = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json")
payload = json.loads(path.read_text(encoding="utf-8"))
for key in ["schema_version", "rubric", "candidates", "decision", "phase2_handoff"]:
    assert key in payload, f"missing {key}"
assert payload["decision"]["status"] == "pending"
print("scorecard schema initialized")
PY
```

Expected: prints `scorecard schema initialized`.

### Phase C: Gather Primary Source and Feasibility Notes

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/environment_recheck.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/pdebench_fluids.md`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/pdearena_maxwell3d.md`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/openfwi_2d_acoustic_fwi.md`

- [ ] C1: Recheck local environment constraints without installing anything.

Run:

```bash
python - <<'PY' > .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/environment_recheck.json
from pathlib import Path
import importlib.util
import json
import platform
import shutil
import subprocess

payload = {
    "python": platform.python_version(),
    "packages_available": {
        name: importlib.util.find_spec(name) is not None
        for name in ["torch", "lightning", "neuralop", "pdearena", "pynx", "h5py", "numpy"]
    },
    "disk": {},
    "nvidia_smi": None,
}
usage = shutil.disk_usage(Path.cwd())
payload["disk"] = {
    "cwd": str(Path.cwd()),
    "free_gb": round(usage.free / (1024**3), 2),
    "total_gb": round(usage.total / (1024**3), 2),
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

Expected: parseable JSON with Python, disk, package, Torch, CUDA/GPU evidence. If CUDA is unavailable, record it as a selection risk.

- [ ] C2: Write the PDEBench fluids source note.

Use primary sources from the Phase 0 inventory:

- Paper: `https://arxiv.org/abs/2210.07182`
- Repository: `https://github.com/pdebench/PDEBench`

The note must include:

- source URLs and access date
- exact candidate task narrowed for scoring, or a short list of task variants still under consideration
- metric and split evidence
- dataset access and expected shard/data-size evidence
- install/dependency burden
- license/access terms or unresolved license question
- RTX 3090 and local disk feasibility
- local baseline options, especially FNO/neuralop, U-Net, and a trivial persistence/coarse baseline if protocol-compatible
- published-SOTA or published-baseline rows that may be cited only with protocol caveats
- Phase 2 smoke-run shape: what a bounded data-load/one-epoch check would need to prove

- [ ] C3: Write the PDEArena Maxwell-3D source note.

Use primary sources from the Phase 0 inventory:

- Paper: `https://arxiv.org/abs/2209.15616`
- Repository: `https://github.com/pdearena/pdearena`
- Data docs: `https://pdearena.github.io/pdearena/datadownload/`
- Architecture docs: `https://pdearena.github.io/pdearena/architectures/`

The note must include the same fields as C2, plus an explicit risk assessment for 3D adaptation versus a reduced 2D/channel policy. If the only credible path requires broad 3D model adaptation, mark that as a Phase 2 scope risk rather than smoothing it over.

- [ ] C4: Write the OpenFWI 2D acoustic FWI source note.

Use primary sources from the Phase 0 inventory:

- Paper: `https://arxiv.org/abs/2111.02926`
- Project: `https://openfwi-lanl.github.io/`
- Repository: `https://github.com/lanl/OpenFWI`

The note must include the same fields as C2, plus an explicit small-shard plan because the full collection is not locally feasible. Record whether official InversionNet can plausibly be a local baseline or should remain published-SOTA context only.

- [ ] C5: Validate source-note structure.

Run:

```bash
python - <<'PY'
from pathlib import Path

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes")
notes = [
    root / "pdebench_fluids.md",
    root / "pdearena_maxwell3d.md",
    root / "openfwi_2d_acoustic_fwi.md",
]
required_terms = [
    "Source",
    "Access",
    "Metric",
    "Data",
    "License",
    "RTX 3090",
    "Baseline",
    "SOTA",
    "Phase 2",
]
for path in notes:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    missing = [term for term in required_terms if term.lower() not in text.lower()]
    if missing:
        raise SystemExit(f"{path} missing required source-note terms: {missing}")
print("source notes contain required evidence fields")
PY
```

Expected: prints `source notes contain required evidence fields`.

### Phase D: Score All Candidates and Select Primary/Fallback

**Files:**
- Modify: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json`

- [ ] D1: Populate one scorecard record for each candidate.

For each candidate, fill all required Unit 2 fields. Keep the evidence concise but concrete. Use `score` fields for the major criteria and `notes` fields for justification.

Required candidate IDs:

- `pdebench_fluids`
- `pdearena_maxwell3d`
- `openfwi_2d_acoustic_fwi`

- [ ] D2: Apply the gate-first decision rule.

Set `decision.status` to one of:

- `selected`: exactly one `primary_candidate_id` and one `fallback_candidate_id` are set.
- `blocked`: no candidate meets the primary gate; `blocked_reason` explains the missing decision and the durable selection document must escalate instead of pretending the gate passed.

Do not set the same candidate as both primary and fallback.

- [ ] D3: Fill the Phase 2 handoff fields for the primary and fallback.

The handoff must identify:

- candidate IDs and names
- expected data access path or shard
- smoke-run proof needed before full execution
- local baseline plan
- metric contract
- dependency/install risk
- published-SOTA caveat
- pivot trigger to fallback

- [ ] D4: Validate the populated scorecard.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

path = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json")
payload = json.loads(path.read_text(encoding="utf-8"))
required_ids = {"pdebench_fluids", "pdearena_maxwell3d", "openfwi_2d_acoustic_fwi"}
candidate_ids = {candidate.get("candidate_id") for candidate in payload.get("candidates", [])}
if candidate_ids != required_ids:
    raise SystemExit(f"unexpected candidate IDs: {sorted(candidate_ids)}")
required_fields = [
    "candidate_name",
    "family",
    "task_type",
    "architectural_fit",
    "benchmark_maturity",
    "metric_clarity",
    "data_access",
    "data_size",
    "install_burden",
    "license_access",
    "rtx_3090_feasibility",
    "local_baseline_feasibility",
    "published_sota_availability",
    "paper_story_fit",
    "risks",
    "required_phase2_preflight",
    "recommended_status",
    "source_notes_path",
]
for candidate in payload["candidates"]:
    missing = [field for field in required_fields if field not in candidate]
    if missing:
        raise SystemExit(f"{candidate.get('candidate_id')} missing fields: {missing}")
decision = payload["decision"]
if decision["status"] == "selected":
    primary = decision.get("primary_candidate_id")
    fallback = decision.get("fallback_candidate_id")
    if not primary or not fallback or primary == fallback:
        raise SystemExit(f"invalid selected decision: {decision}")
elif decision["status"] == "blocked":
    if not decision.get("blocked_reason"):
        raise SystemExit("blocked decision requires blocked_reason")
else:
    raise SystemExit(f"decision status must be selected or blocked, got {decision['status']!r}")
print("scorecard decision is structurally valid")
PY
```

Expected: prints `scorecard decision is structurally valid`.

### Phase E: Write the Durable Selection Document

**Files:**
- Create or modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`

- [ ] E1: Draft `pde_benchmark_selection.md` from the scorecard.

Required structure:

```markdown
# NeurIPS Hybrid ResNet PDE Benchmark Selection

## Scope and Gate
## Documents and Sources Used
## Scorecard Schema
## Candidate Scorecard
## Primary Benchmark Decision
## Fallback Benchmark Decision
## Rejected Candidate Rationale
## Metric Contract
## Data Access and License Plan
## Local Baseline Plan
## Published SOTA Caveats
## Phase 2 Handoff
## Pivot and Blocked Conditions
## Non-Goals Confirmed
## Raw Artifact Links
## Verification
```

- [ ] E2: Preserve the roadmap gate language in the selection document.

The document must explicitly state:

- Phase 1 is a selection tranche, not an execution tranche.
- All three Phase 0 candidates were evaluated.
- Phase 0 did not select primary/fallback.
- Phase 2 must start with a smoke/data-load check for the selected primary.
- If the primary fails install, data access, metric, smoke-fit, or competitiveness gates, the roadmap pivots to the named fallback before spending CDI polish time.

- [ ] E3: Include the primary benchmark metric contract.

The metric contract must name:

- prediction/inversion target
- train/validation/test or shard split expectation
- primary metric
- secondary metrics
- normalization or rollout horizon caveats
- source of published numbers if used
- what Phase 2 must write as metrics/provenance

- [ ] E4: Include the local baseline plan.

The baseline plan must name at least two feasible local baselines, or explicitly justify one strong local baseline plus published SOTA. For each baseline, record whether it is:

- expected to run locally in Phase 2
- published-SOTA context only
- rejected as infeasible

- [ ] E5: Include data access, license, and disk/GPU feasibility.

Record:

- intended dataset family/shard
- expected download/generation path
- license/access terms or unresolved license item
- checksum or manifest expectation for Phase 2
- local disk implication
- RTX 3090 smoke feasibility claim and what would disprove it

- [ ] E6: If no candidate satisfies the primary gate, write a blocked selection document.

The blocked document must:

- set the gate status to blocked
- explain which required gate failed for each candidate
- recommend the narrowest human decision needed
- avoid naming a weak primary as if selected
- keep later roadmap phases blocked until the decision is resolved

### Phase F: Discoverability and Verification

**Files:**
- Modify: `docs/index.md`
- Read: `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-1-pde-benchmark-selection/plan-phase/plan_path.txt`

- [ ] F1: Add a concise `docs/index.md` entry for the durable selection document.

Place it near the existing NeurIPS Hybrid ResNet entries. The entry should identify `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md` as the Roadmap Phase 1 benchmark scorecard and primary/fallback decision for the required PDE pillar.

- [ ] F2: Run the roadmap-required structural verification.

Run:

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

Expected: prints `selection doc contains required decision fields`.

- [ ] F3: Run the stricter Phase 1 artifact validation.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

selection = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md")
scorecard = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json")
index = Path("docs/index.md")
plan_path = Path("state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-1-pde-benchmark-selection/plan-phase/plan_path.txt")

payload = json.loads(scorecard.read_text(encoding="utf-8"))
text = selection.read_text(encoding="utf-8")
index_text = index.read_text(encoding="utf-8")

required_terms = [
    "PDEBench",
    "PDEArena",
    "OpenFWI",
    "primary",
    "fallback",
    "metric",
    "data access",
    "license",
    "baseline",
    "SOTA",
    "Phase 2",
    "Non-Goals",
]
missing = [term for term in required_terms if term.lower() not in text.lower()]
if missing:
    raise SystemExit(f"selection doc missing required terms: {missing}")
if "pde_benchmark_selection.md" not in index_text:
    raise SystemExit("docs/index.md missing pde_benchmark_selection.md entry")
if plan_path.read_text(encoding="utf-8").strip() != "docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-1-pde-benchmark-selection/execution_plan.md":
    raise SystemExit("plan_path.txt does not contain the expected plan path")
decision = payload["decision"]
if decision["status"] == "selected" and decision["primary_candidate_id"] == decision["fallback_candidate_id"]:
    raise SystemExit("primary and fallback cannot be the same")
print("Phase 1 selection artifacts are structurally valid")
PY
```

Expected: prints `Phase 1 selection artifacts are structurally valid`.

- [ ] F4: Confirm no out-of-scope paper or execution artifacts were created.

Run:

```bash
python - <<'PY'
from pathlib import Path

forbidden = [
    Path("/home/ollie/Documents/neurips/index.md"),
    Path("/home/ollie/Documents/neurips/evidence_checklist.md"),
]
existing = []
for path in forbidden:
    if path.exists():
        stat = path.stat()
        existing.append({"path": str(path), "mtime": stat.st_mtime, "size": stat.st_size})
print({"paper_facing_paths_to_inspect": existing})
PY
```

Expected: this command reports any existing paper-facing paths for inspection. If they preexisted, record that they were not modified. If they were created by this tranche, revert or escalate because that violates the scope boundary.

## Verification Commands

Run after completing all phases:

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

```bash
python - <<'PY'
from pathlib import Path
import json

scorecard = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json")
payload = json.loads(scorecard.read_text(encoding="utf-8"))
candidate_ids = {candidate.get("candidate_id") for candidate in payload.get("candidates", [])}
expected = {"pdebench_fluids", "pdearena_maxwell3d", "openfwi_2d_acoustic_fwi"}
if candidate_ids != expected:
    raise SystemExit(f"unexpected candidate IDs: {sorted(candidate_ids)}")
decision = payload["decision"]
if decision["status"] not in {"selected", "blocked"}:
    raise SystemExit(f"unexpected decision status: {decision['status']}")
if decision["status"] == "selected" and (not decision.get("primary_candidate_id") or not decision.get("fallback_candidate_id")):
    raise SystemExit("selected decision requires primary and fallback")
print("scorecard structure is valid")
PY
```

```bash
python - <<'PY'
from pathlib import Path

index_text = Path("docs/index.md").read_text(encoding="utf-8")
if "pde_benchmark_selection.md" not in index_text:
    raise SystemExit("docs/index.md missing pde_benchmark_selection.md entry")
plan_path = Path("state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-1-pde-benchmark-selection/plan-phase/plan_path.txt")
expected = "docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-1-pde-benchmark-selection/execution_plan.md"
if plan_path.read_text(encoding="utf-8").strip() != expected:
    raise SystemExit("plan_path.txt does not contain the expected path")
print("discoverability and output pointer are valid")
PY
```

No `pytest` selector is required if execution only writes Markdown and ignored JSON/source notes. If the executor adds or modifies Python code, benchmark adapters, metric parsers, result writers, or CLI behavior despite the non-goals, they must add focused tests and run the narrowest relevant `pytest` selectors before completing the tranche.

## Completion Criteria

- [ ] `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md` exists.
- [ ] The selection document defines the required scorecard fields.
- [ ] The scorecard evaluates PDEBench fluids, PDEArena Maxwell-3D, and OpenFWI 2D acoustic FWI.
- [ ] Exactly one primary and one fallback benchmark are named, or the document is explicitly blocked with a reason.
- [ ] The primary decision, if selected, has clear metrics, runnable data access, RTX 3090 smoke feasibility, baseline plan, and paper-story rationale.
- [ ] Published SOTA caveats are explicit and separated from local baseline plans.
- [ ] Rejection rationale is recorded for non-selected candidates.
- [ ] Phase 2 handoff identifies smoke-run proof, data/access plan, metric contract, baseline plan, and pivot trigger.
- [ ] Raw scorecard/source notes are under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/`.
- [ ] `docs/index.md` links the durable selection document.
- [ ] Verification commands pass.
- [ ] No out-of-scope PDE execution, CDI execution, `256x256` scaling, paper-facing artifact assembly, worktree creation, or stable core model-module edit occurred.

## Artifacts Index

- Execution plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-1-pde-benchmark-selection/execution_plan.md`
- Durable Phase 1 output: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- Raw Phase 1 artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/`
- Phase 0 candidate inventory input: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/pde_candidate_inventory.json`
- Phase 0 durable inventory input: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
