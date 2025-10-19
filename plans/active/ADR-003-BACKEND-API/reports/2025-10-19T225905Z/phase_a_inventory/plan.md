# ADR-003 Phase A Execution Plan — Architecture Carve-Out (2025-10-19T225905Z)

## Context
- **Initiative:** ADR-003-BACKEND-API — Standardize PyTorch backend API
- **Phase:** A — Architecture Carve-Out & Inventory
- **Phase Goal:** Produce authoritative inventories of the current CLI surfaces and backend-specific execution knobs so Phase B factory work can proceed without guesswork.
- **Dependencies & References:**
  - `specs/ptychodus_api_spec.md` §4 — Canonical reconstructor lifecycle contract
  - `docs/workflows/pytorch.md` §§5–12 — Current PyTorch workflow + backend selection rules
  - `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` — CLI wiring decisions & runtime controls
  - `ptycho_torch/train.py` (new + legacy CLI), `ptycho_torch/inference.py` (new CLI + MLflow legacy path)
  - `ptycho_torch/config_params.py` & `ptycho_torch/config_bridge.py` — Current singleton configs and TF bridge mappings
  - `tests/torch/test_config_bridge.py` — Enforced field translations
- **Artifact Discipline:** Store all outputs for this phase under `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/`.
  - Required deliverables this loop: `cli_inventory.md`, `execution_knobs.md`, `overlap_notes.md`, `summary.md`.
  - Capture raw command output in `logs/` subdirectory (create if needed) when helpful (e.g., grep dumps).

---

### Phase A.1 — CLI Flag Inventory
Goal: Enumerate every CLI entry point feeding the PyTorch backend, map each flag to the config field(s) it influences, and identify gaps compared with TensorFlow CLI surfaces.

Prereqs: PyTorch extras installed (`pip install -e .[torch]`); confirm files listed below are unchanged locally before recording line anchors.

Exit Criteria:
1. `cli_inventory.md` contains a table for each CLI (training new, training legacy, inference new, inference MLflow) with columns: `Flag`, `Source (file:line)`, `Target Config Field(s)`, `Notes / TODO`.
2. Inventory explicitly notes required/optional flags, default values, and whether the flag is currently ignored or stubbed.
3. Differences vs TensorFlow CLI called out (missing flags, name divergences, semantics).

| ID | Task Description | State | Guidance |
| --- | --- | --- | --- |
| A1.a | Enumerate CLI definitions | [ ] | Use `rg --no-heading --line-number "add_argument" ptycho_torch/train.py ptycho_torch/inference.py` to collect flag definitions. Capture the raw command output under `logs/a1_cli_flags.txt`. |
| A1.b | Map CLI → config fields | [ ] | For each flag, inspect `cli_main()` (`train.py:366-520`), `run()` in `inference.py`, and downstream usage (e.g., `_infer_probe_size`, `_run_pytorch_workflow`) to document how the value flows into `DataConfig`, `TrainingConfig`, `InferenceConfig`, or Lightning runtime. Record mapping + notes in `cli_inventory.md`. Cite supporting line numbers. |
| A1.c | Compare with TensorFlow CLI | [ ] | Review `scripts/training/train.py` & `scripts/inference/inference.py` (TensorFlow) to list flags absent on PyTorch CLI or with divergent defaults. Summarize the delta in `cli_inventory.md` under a dedicated “Parity Gaps” section. |

---

### Phase A.2 — Backend Execution Knob Catalog
Goal: Identify every PyTorch-specific execution knob (CLI flags, dataclass fields, environment toggles) that is not already represented in the canonical TensorFlow dataclasses so we can design `PyTorchExecutionConfig` and factory overrides.

Exit Criteria:
1. `execution_knobs.md` lists each knob with columns: `Knob Name`, `Current Definition (file:line)`, `Purpose`, `Proposed Home (Factory override vs PyTorchExecutionConfig)`, `Parity Notes`.
2. Coverage includes Lightning trainer parameters (device/strategy/deterministic), MLflow toggles, distributed settings, logging/progress knobs, and any hard-coded defaults in `config_bridge.py`.
3. Document which knobs already have parity coverage in TensorFlow and which require spec updates or ADR clarifications.

| ID | Task Description | State | Guidance |
| --- | --- | --- | --- |
| A2.a | Mine config singletons & bridge overrides | [ ] | Inspect `ptycho_torch/config_params.py` and `ptycho_torch/config_bridge.py` for fields that currently lack canonical counterparts (`device`, `strategy`, `experiment_name`, `probe_scale` divergences, etc.). Record findings with direct quotes or code snippets in `execution_knobs.md`. |
| A2.b | Capture runtime-only knobs | [ ] | Review `_train_with_lightning`, `LightningTrainer` initialization in `ptycho_torch/workflows/components.py`, and CLI command assembly in `train.py` for hidden knobs (e.g., `--disable_mlflow`, deterministic flags). Add them to the catalog. |
| A2.c | Classify knob handling strategy | [ ] | For each entry, decide whether the knob belongs in a new `PyTorchExecutionConfig`, an override dict passed into factories, or requires ADR follow-up. Document rationale and cross-reference spec lines or findings entries (`POLICY-001`, `FORMAT-001`). |

---

### Phase A.3 — Cross-Plan Overlap Audit
Goal: Prevent duplicated effort by confirming which responsibilities remain with INTEGRATE-PYTORCH-001 and which shift to ADR-003 before engineering work starts.

Exit Criteria:
1. `overlap_notes.md` summarises overlaps/gaps between ADR-003 Phase B–E tasks and existing artifacts (`phase_e2_implementation.md`, `phase_e_closeout/closure_summary.md`, TEST-PYTORCH-001 runtime guidance).
2. Outstanding follow-ups (e.g., CI integration hooks, TEST-PYTORCH-001 monitoring) flagged with recommended owner (ADR-003 vs other initiative).
3. Identify any prerequisites (fixtures, parity logs) that Phase B will rely on, citing artifact paths.

| ID | Task Description | State | Guidance |
| --- | --- | --- | --- |
| A3.a | Review upstream plans | [ ] | Skim `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md`, `phase_e_closeout/closure_summary.md`, and `plans/active/TEST-PYTORCH-001/implementation.md` to list items already solved or explicitly deferred to ADR-003. |
| A3.b | Document ownership recommendations | [ ] | In `overlap_notes.md`, create a table with columns `Topic`, `Existing Artifact`, `Owner`, `Notes` indicating whether ADR-003 must absorb the work. Include references to artifact timestamps. |
| A3.c | Flag missing ADR documentation | [ ] | Note that `docs/architecture/adr/ADR-003.md` is currently absent; capture whether it must be authored during Phase B or separately. |

---

## Reporting & Handoff
- Populate `summary.md` (same directory) with bullet points covering:
  - Key discoveries per subtask (new flags, missing parity knobs, overlaps)
  - Follow-up questions or blockers
  - Suggested priorities for Phase B
- Update `plans/active/ADR-003-BACKEND-API/implementation.md` Phase A rows with `[x]`/`[P]` and artifact references once tasks complete.
- Append a `docs/fix_plan.md` attempt entry summarizing the loop with links to `cli_inventory.md`, `execution_knobs.md`, `overlap_notes.md`.
- No code changes or test executions in this phase (analysis only); keep working tree clean.

## Verification Checklist (Supervisor Use)
- [ ] `cli_inventory.md` and `execution_knobs.md` exist with populated tables and citations.
- [ ] `overlap_notes.md` records owner decisions and missing artifacts.
- [ ] `summary.md` documents outcomes + next steps, referencing artifact paths.
- [ ] `plans/active/ADR-003-BACKEND-API/implementation.md` and `docs/fix_plan.md` updated in the same loop.
- [ ] All artifacts stored under `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/`.
