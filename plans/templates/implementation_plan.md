# Implementation Plan Template (Phased)

> Copy this file to `plans/active/<initiative-id>/implementation.md` and customize.

## Initiative
- ID: <initiative-id>
- Title: <short title>
- Owner: <name>
- Spec Owner: <normative spec filename>  <!-- e.g., docs/spec-db-core.md -->
- Status: pending | in_progress | blocked | done | archived

## Goals
- <goal 1>
- <goal 2>

## Phases Overview
- Phase A — <name>: <one-line objective>
- Phase B — <name>: <one-line objective>
- Phase C — <name>: <one-line objective>

## Exit Criteria
1. <criterion 1>
2. <criterion 2>
3. <criterion 3>
4. Test registry synchronized: `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` reflect any new/changed tests; `pytest --collect-only` logs for documented selectors are saved under `plans/active/<initiative-id>/reports/<timestamp>/`. Do not close the initiative if any selector marked "Active" collects 0 tests.

## Compliance Matrix (Mandatory)
> List the specific Spec constraints, Fix-Plan ledger rows, and Findings/Policies this initiative must honor. Missing a relevant entry is a plan defect per ARRP.
- [ ] **Spec Constraint:** <e.g., `spec-db-core.md §5.2 — Variance model definition`>
- [ ] **Fix-Plan Link:** <e.g., `docs/fix_plan.md — Row [PHYSICS-LOSS-001]`>
- [ ] **Finding/Policy ID:** <e.g., `CONFIG-001`, `POLICY-001 (PyTorch Optional)`>

## Spec Alignment
- **Normative Spec:** [path to spec file]
- **Key Clauses:** [list of specific requirements this plan satisfies]

## Architecture / Interfaces (optional)
- **Key Data Types / Protocols:**  
  e.g., `User`, `OrderService`, `PaymentGateway` in a web app, or `Model`, `Trainer`, `MetricSink` in an ML pipeline. You can sketch these in a tiny IDL-style block if helpful (e.g., `types: User { id: UUID; email: string }`).
- **Boundary Definitions:**  
  Briefly describe the main seams between components (layers, services, processes, or subsystems), e.g., `[Client] -> [API] -> [Service] -> [DB]`.
- **Sequence Sketch (Happy Path):**  
  Short textual outline of the primary request/response or job execution path, e.g., `Client -> API: POST /orders -> Service -> DB -> Client`.
- **Data-Flow Notes:**  
  Note what data moves where (shape, format, rate) and across which boundaries (in‑process, network, devices, storage), e.g., `[Raw events] -> [Validate] -> [Transform] -> [Warehouse]`.

> One-shot example (replace with your own):
> - types: `User { id: UUID; email: string }`, `Order { id: UUID; user_id: UUID; total_cents: int }`
> - boundaries: `[Browser] -> [HTTP API] -> [OrderService] -> [Postgres]`
> - sequence: `Browser -> API: POST /orders -> OrderService -> DB -> API -> Browser`
> - data-flow: JSON request (~2 KB) → row in `orders` table → JSON response with `order_id`, or for a training step: batch tensor `[B, C, H, W]` on GPU → forward pass → loss scalar → gradient tensors → optimizer step

## Context Priming (read before edits)
- Primary docs/specs to re-read: <list explicit files + sections>
- Required findings/case law: <docs/findings.md IDs + summary>
- Related telemetry/attempts: <links to relevant artifacts or plan history>
- Data dependencies to verify: <summarize the external inputs (datasets, configs, HKL/sigma assets, etc.) this initiative relies on; reference `docs/data_dependency_manifest.md` entries and note any additions needed>

## Phase A — <name>
### Checklist
- [ ] A0: **Nucleus / Test-first gate:** <minimal probe or selector to validate assumptions before implementation>
- [ ] A1: <task> (owner, expected artifacts)
- [ ] A2: <task>
- [ ] A3: <task>

### Dependency Analysis (Required for Refactors)
- **Touched Modules:** [list]
- **Circular Import Risks:** [analysis]
- **State Migration:** [how state moves from old to new]

### Notes & Risks
- <risk 1>

## Phase B — <name>
### Checklist
- [ ] B1: <task>
- [ ] B2: <task>

### Notes & Risks
- <risk 2>

## Phase C — <name>
### Checklist
- [ ] C1: <task>
- [ ] C2: <task>

### Notes & Risks
- <risk 3>

## Artifacts Index
- Reports root: `plans/active/<initiative-id>/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`

