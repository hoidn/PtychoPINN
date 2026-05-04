# BRDT Preflight Summary And Promotion Decision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. This item authorizes only summary writing, recommendation packaging, and discoverability updates for the completed BRDT preflight. Do not create worktrees, do not relaunch BRDT dataset generation or the four-row preflight, do not amend the NeurIPS roadmap here, and do not promote any BRDT row directly into manuscript tables or figures from this item. Keep any normal verification command under implementation ownership until it exits successfully or reaches a documented recoverable failure; if a source artifact is missing or corrupt after a narrow diagnosis, stop and surface that upstream blocker instead of widening scope.

**Goal:** Turn the completed BRDT candidate-lane preflight into one durable, self-contained decision artifact that recommends either `promote_to_evidence_amendment_plan`, `defer_after_preflight`, or `reject_for_current_manuscript`, without widening BRDT beyond its current optional-candidate status.

**Architecture:** Treat the work as a documentation-and-decision pass over already-produced BRDT evidence. First lock the prerequisite authorities and the exact machine-readable four-row bundle that this item is allowed to summarize. Then derive the recommendation from the bundle contract, row statuses, metrics, visuals, provenance, and blocker details, write `brdt_preflight_summary.md`, and update the durable evidence index so later backlog selection or BRDT umbrella-closeout work can consume one summary authority instead of reconstructing the lane from execution reports.

**Tech Stack:** PATH `python`, Markdown, JSON artifacts under `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`, existing BRDT summaries under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.

---

## Selected Objective

- Implement backlog item `2026-04-29-brdt-preflight-summary-promotion-decision`.
- Consume the completed BRDT four-row preflight outputs and the locked operator/dataset/adapter authorities.
- Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md`.
- Emit exactly one recommendation token:
  - `promote_to_evidence_amendment_plan`
  - `defer_after_preflight`
  - `reject_for_current_manuscript`

## Scope

- Summarize the completed BRDT candidate-lane preflight only.
- Cover the fields required by the selected backlog item:
  - operator validation result;
  - dataset and normalization validity;
  - row roster and metrics;
  - visual bundle and source-array availability;
  - dependency and environment issues;
  - known limitations and claim boundaries.
- Update the durable evidence index so the BRDT four-row row now points at the checked-in summary authority rather than a placeholder note.

## Explicit Non-Goals

- Do not promote BRDT rows into manuscript tables, figures, or `/home/ollie/Documents/neurips/` outputs in this item.
- Do not rerun BRDT operator validation, dataset generation, task-adapter sanity runs, or the four-row preflight unless a source artifact is missing and a narrowly scoped recovery is explicitly required by the user later.
- Do not amend `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`, `docs/steering.md`, or the BRDT candidate-lane design from this item.
- Do not add BRDT rows, new budgets, wider baselines, Rytov mode, limited-angle stress tests, or physics-only lanes.
- Do not hide weak results behind vague wording. If operator/data validity is acceptable but the lane is not ready for manuscript spend, prefer a narrow deferral over silent optimism.

## Steering, Roadmap, and Prerequisite Constraints

- Steering keeps CDI `lines128` and PDEBench CNS as the required manuscript pillars; BRDT remains additional candidate work only.
- The roadmap and BRDT candidate-lane design both require a reviewed promotion decision before BRDT can become a later evidence-amendment candidate.
- Progress-ledger status matters here: only Phase 0, Phase 1, and early Phase 2 smoke tranches are completed in `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`; BRDT is not elevated to a required roadmap pillar and therefore this item must stay low-expense, summary-only, and claim-boundary-tight.
- Immediate prerequisite chain for this item is already satisfied by completed backlog work:
  - `2026-04-29-brdt-operator-validation`
  - `2026-04-29-brdt-dataset-preflight`
  - `2026-04-29-brdt-task-adapters`
  - `2026-04-29-brdt-four-row-preflight`
- Treat the four-row bundle's machine-readable artifacts as authoritative for metrics and row statuses. Use the four-row execution report only for supporting provenance about seed plumbing, rerun history, and the documented ODTbrain narrow-fix attempts.

## Implementation Architecture

1. **Prerequisite Audit**
   - Owns authority locking across the BRDT operator, dataset, adapter, and four-row preflight artifacts.
2. **Decision Extraction**
   - Owns evidence reading from the current machine-readable four-row bundle and the recommendation rubric.
3. **Durable Packaging**
   - Owns the checked-in summary, the paper-evidence-index update, and the deterministic agreement checks between summary and index.

## File and Artifact Targets

### Mandatory Contract Outputs

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - update the `2026-04-29-brdt-four-row-preflight` row so the summary authority points to `brdt_preflight_summary.md`
  - preserve the lane as `decision_support` / candidate-only unless the actual summary evidence justifies a different tier within current authority, which is unlikely

### Mandatory Source Inputs To Consume

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metric_schema.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/visual_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/rows/*/row_summary.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/rows/*/invocation.json`

### Preferred Packaging

- Verification logs under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-preflight-summary-promotion-decision/verification/`
- `docs/index.md` update only if implementation determines this checked-in BRDT summary should be directly discoverable from the top-level docs hub rather than only through `paper_evidence_index.md`

## Recommendation Rule

Apply the recommendation only after reading the current machine-readable bundle. Do not infer it from stale prose tables.

- Choose `promote_to_evidence_amendment_plan` only if:
  - the operator, dataset, and adapter prerequisites remain valid and internally consistent;
  - the four-row preflight bundle is complete and reproducible enough to support a later scoped amendment plan;
  - the lane shows enough promise or differentiating value to justify explicit later work on exact rows, budgets, artifacts, and claim boundaries;
  - the summary still states that this item itself does not authorize manuscript-table promotion.
- Choose `defer_after_preflight` if:
  - the operator/data/model path is scientifically valid enough to preserve,
  - but current row outcomes, missing dependencies, incomplete comparators, or current roadmap priorities do not justify immediate manuscript-evidence follow-up;
  - this is the preferred outcome when the lane is real but not strong or complete enough to spend current budget.
- Choose `reject_for_current_manuscript` only if:
  - the preflight reveals a deeper scientific, fairness, provenance, or competitiveness problem that makes later manuscript follow-up unjustified within the current campaign;
  - or the remaining blocker sits outside current authority and materially undercuts the lane rather than merely delaying it.

If the summary recommends promotion, it must also say that a separate roadmap/evidence amendment plan is required to name exact rows, budgets, artifacts, and claim boundaries. No direct manuscript authorization is allowed here.

## Execution Tranches

### Tranche 1: Lock The Authoritative BRDT Inputs

**Purpose:** Ensure the summary is derived from the same locked BRDT contract already approved by upstream backlog items.

- [ ] Read the three checked-in prerequisite summaries and the BRDT candidate-lane design only as context for claim boundaries, not as substitutes for current bundle metrics.
- [ ] Confirm the four-row artifact root exists and contains the current `preflight_manifest.json`, `metrics.json`, `metric_schema.json`, `visual_manifest.json`, and decision-support dataset manifest.
- [ ] Confirm the current summary item is consuming the capped decision-support BRDT dataset (`brdt128_decision_support_preflight`) rather than the earlier smoke dataset.
- [ ] Record any source-artifact gap as an upstream blocker only after one narrow path/availability diagnosis; do not relaunch preflight work from this item.

**Verification for Tranche 1**

- [ ] **Blocking:** confirm all prerequisite docs and four-row machine-readable artifacts exist before drafting the summary.
- [ ] **Supporting:** compare the four-row execution report against the machine-readable bundle only to capture rerun/provenance context such as the fixed seed, duplicate-writer guard, and the ODTbrain narrow-fix attempts.

### Tranche 2: Extract The Evidence And Decide The Outcome

**Purpose:** Derive the recommendation from authoritative row data, not from prior narrative summaries.

- [ ] Read `preflight_manifest.json` and lock:
  - `claim_boundary`
  - `next_backlog_item`
  - training contract, including the seeded rerun contract if present
  - row fingerprints, resumed rows, and classical narrow-fix attempts
- [ ] Read `metrics.json` and extract the exact current row roster, row statuses, image-space physical-`q` metrics, measurement-space metrics, parameter counts, and runtime metadata.
- [ ] Read `visual_manifest.json` and confirm the bundle includes the expected compare/error/residual visuals plus source-array references.
- [ ] Read the per-row `row_summary.json` and `invocation.json` files for any details not lifted into the top-level metrics or manifest, especially blocker reason and device/runtime provenance.
- [ ] Summarize the dependency/environment issues explicitly, including whether ODTbrain remains unavailable and whether that blocker is row-local or lane-wide.
- [ ] Decide the final recommendation token using the rule above and write down the follow-up implication:
  - separate evidence-amendment plan required;
  - preserve as deferred candidate evidence only;
  - or close the lane for the current manuscript.

**Verification for Tranche 2**

- [ ] **Blocking:** derive all quoted row metrics and statuses from `metrics.json`; do not copy a stale table from the execution report when the report contains historical rerun snapshots.
- [ ] **Blocking:** confirm the recommendation is consistent with the recorded `decision_support_preflight_only` claim boundary and does not silently treat the bundle as paper-grade evidence.
- [ ] **Supporting:** if the bundle shows mixed evidence, preserve that ambiguity directly in the recommendation rationale instead of forcing a stronger claim.

### Tranche 3: Write The Durable Summary And Update Discoverability

**Purpose:** Produce one checked-in authority that later planners can trust without reopening the raw backlog chain.

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md` as a self-contained authority with, at minimum, these sections:
  - identity and scope boundary;
  - prerequisite status;
  - operator validation result;
  - dataset and normalization validity;
  - four-row contract and source-artifact root;
  - row roster, metrics, row statuses, and runtime/provenance notes;
  - visual bundle and source-array availability;
  - dependency/environment issues;
  - known limitations and claim boundary;
  - final recommendation token and follow-up routing.
- [ ] Make the claim boundary explicit in the summary:
  - BRDT remains additional candidate work;
  - the current bundle is `decision_support_preflight_only`;
  - this item does not authorize manuscript-table or figure promotion.
- [ ] If promotion is recommended, explicitly require a later roadmap/evidence amendment plan and name the kinds of fields that amendment must lock: exact rows, budgets, artifacts, fairness contract, and claim boundaries.
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` so the `2026-04-29-brdt-four-row-preflight` row points to `[BRDT preflight summary](brdt_preflight_summary.md)` as the summary authority and keeps the artifact root and candidate-lane boundary accurate.
- [ ] Update `docs/index.md` only if the new summary should become a top-level discoverability surface rather than being discovered solely via `paper_evidence_index.md`.

**Verification for Tranche 3**

- [ ] **Blocking:** the summary names exactly one final recommendation token from the approved set.
- [ ] **Blocking:** the summary explicitly states that this item itself does not promote BRDT into manuscript tables or figures.
- [ ] **Supporting:** if `docs/index.md` is not updated, state in the execution report that `paper_evidence_index.md` is the intended discoverability surface for this candidate-lane summary.

### Tranche 4: Run Deterministic Agreement Checks

**Purpose:** Prove the checked-in summary and index surface agree with the backlog contract.

- [ ] Run the backlog-required deterministic presence check.
- [ ] Run a stronger summary/index consistency check that validates:
  - the summary exists;
  - exactly one approved recommendation token appears in the summary;
  - `paper_evidence_index.md` references `brdt_preflight_summary.md`;
  - the BRDT four-row row remains visible in the index.
- [ ] If implementation touched only docs, stop after the deterministic doc checks.
- [ ] If implementation added or modified any reusable helper or validator code under `scripts/studies/born_rytov_dt/` or `tests/studies/`, rerun the narrow BRDT selector and compile check before completion.

**Verification for Tranche 4**

- [ ] **Blocking:** required backlog check command:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing BRDT preflight summary: {missing}")
print("brdt preflight summary present")
PY
```

- [ ] **Blocking:** stronger doc/index agreement check:

```bash
python - <<'PY'
from pathlib import Path

summary = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md")
index = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md")

summary_text = summary.read_text() if summary.exists() else ""
index_text = index.read_text() if index.exists() else ""

tokens = [
    "promote_to_evidence_amendment_plan",
    "defer_after_preflight",
    "reject_for_current_manuscript",
]
hits = [token for token in tokens if token in summary_text]
if len(hits) != 1:
    raise SystemExit(f"expected exactly one recommendation token, found {hits}")
if "2026-04-29-brdt-four-row-preflight" not in index_text:
    raise SystemExit("paper_evidence_index.md is missing the BRDT four-row row")
if "brdt_preflight_summary.md" not in index_text:
    raise SystemExit("paper_evidence_index.md does not reference brdt_preflight_summary.md")
print("brdt preflight summary and paper evidence index are consistent")
PY
```

- [ ] **Supporting, conditional on code changes:** if helper code changed, rerun:

```bash
pytest -q tests/studies/test_born_rytov_dt_preflight.py
python -m compileall -q scripts/studies/born_rytov_dt
```

## Completion Criteria

- The checked-in BRDT summary exists at `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md`.
- The summary is self-contained enough that later implementation or backlog-selection work does not need to reread the raw backlog item, steering doc, or roadmap to recover BRDT scope, prerequisites, claim boundaries, or follow-up implications.
- Exactly one recommendation token is present and is justified from the current machine-readable four-row bundle.
- `paper_evidence_index.md` now points the BRDT four-row row at the new summary authority.
- The backlog-required presence check passes.
- The stronger summary/index agreement check passes.
- No BRDT training, dataset generation, benchmark rerun, roadmap amendment, or manuscript-table promotion was launched from this item.
