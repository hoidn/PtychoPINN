# CNS Paper Contract Decision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Choose and record the authoritative PDEBench `2d_cfd_cns` paper evidence contract before any CNS claim drafting or row locking, including whether the paper uses full-training benchmark rows or bounded capped decision-support rows, the exact same-contract row set, and the authored-FFNO cutoff/status.

**Architecture:** Treat this as an audit-first Phase 2 decision lane. First normalize the existing CNS evidence into explicit coherent contract lanes, then derive the full-training compute/deadline implications from those audited lanes, then write one durable decision document that names the selected contract, exact row requirements, FFNO cutoff, claim boundaries, and downstream row-lock expectations. Only touch runner/reporting code if a real metadata or parity gap blocks the audit or deterministic checks.

**Tech Stack:** PATH `python`, PyTorch (POLICY-001), existing PDEBench image-suite runner/reporting surfaces under `scripts/studies/pdebench_image128/`, pytest, compileall, Markdown/JSON artifacts under `.artifacts/`, tmux with `ptycho311` only if a narrow rerun becomes unavoidable.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-cns-paper-contract-decision`
- Selection mode: `ACTIVE_SELECTION`
- Plan authority date: `2026-04-29`
- Scope owner: Roadmap Phase 2 CNS paper-contract lane
- Selected-item context: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-cns-paper-contract-decision/selected-item-context.md`
- Recorded plan path source: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-cns-paper-contract-decision/plan-phase/plan_path.txt`
- Previous plan path from selected-item context: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Durable decision target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/`

This document supersedes earlier plan content for this backlog item and is the execution authority for implementation.

## Inputs Read

- `AGENTS.md`
- `docs/index.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/workflows/pytorch.md`
- `docs/findings.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- `docs/backlog/in_progress/2026-04-29-cns-paper-contract-decision.md`
- `docs/backlog/active/2026-04-29-cns-paper-benchmark-rows.md`
- `docs/backlog/active/2026-04-29-paper-evidence-package-audit.md`
- `docs/backlog/index.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-cns-paper-contract-decision/selected-item-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-cns-paper-contract-decision/plan-phase/plan_path.txt`
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision-plan-review.json`

## Selected Backlog Objective

- Decide whether the paper uses:
  - `full_training_paper_benchmark`, or
  - `bounded_capped_decision_support`
- Base that decision on the completed CNS evidence, not on implication or wishful scheduling.
- Audit the strongest current same-contract Hybrid-family, FNO, U-Net/CNN-style, and authored FFNO candidates.
- Estimate the compute required to promote CNS to same-contract full-training rows on the official `2d_cfd_cns` file under the active local contract.
- Set an authored-FFNO inclusion cutoff and claim impact before the downstream row-lock backlog item starts.
- Write the decision as a durable authority document and update downstream backlog items if the chosen contract changes their exact row requirements.

## Scope

- Audit the existing durable CNS summaries and their referenced run roots, focusing on coherent same-contract lanes rather than raw “best metric anywhere” cherry-picking.
- Separate reusable same-contract evidence from adjacent but contract-divergent evidence.
- Quantify what work is already done versus what work would still be required under each contract option.
- Decide the authoritative row-lock contract, including history length, cap/full-training status, required row IDs, authored FFNO cutoff, and claim boundary.
- Freeze the authoritative normalization and training-recipe contract that later CNS row-lock and claim-drafting work must obey, including whether the current CNS lane keeps the task-local `mse` loss override instead of the general design baseline's `mae` recipe.
- Record downstream implications for the active CNS row-lock and paper-evidence-audit backlog items.

## Explicit Non-Goals

- Do not run the downstream CNS paper rows in this item. This item decides the contract; it does not execute the row-lock workload.
- Do not reopen broad architecture search, change the official CNS dataset, or reinterpret unrelated PDEBench tasks.
- Do not silently mix `history_len=2` and `history_len=3` rows in one headline table unless the resulting incompatibility is explicit and accepted as a claim limitation.
- Do not use the repo-local `ffno_bottleneck_base` or `ffno_bottleneck_localconv_base` rows as substitutes for the authored FFNO row.
- Do not make GNOT a required headline row for this item. GNOT remains optional context unless a later approved plan promotes it explicitly.
- Do not create `/home/ollie/Documents/neurips/` artifacts, manuscript prose, or Phase 5 evidence-map outputs.
- Do not treat capped, smoke, or pilot evidence as full-training benchmark evidence by wording alone.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering, Roadmap, and Fairness Constraints

- Steering requires explicit equal-footing comparisons and forbids silently relaxing fairness constraints to make an item easier.
- The roadmap keeps this work inside Phase 2 PDEBench CNS paper-contract packaging. It must not be presented as satisfying Phase 3 CDI work or Phase 5 artifact assembly.
- The approved design and PDEBench image-suite plan require the CNS pillar to preserve metric, split, protocol, and claim boundaries. Smoke or capped evidence cannot satisfy the benchmark gate.
- The approved design's default Hybrid ResNet competitiveness recipe is the `mae` baseline family, but the current CNS implementation summary records a task-local `mse` override aligned to the official PDEBench forward baselines. This item must not leave that override implicit; the durable decision has to state whether the selected paper contract keeps that override and which optimizer/scheduler fields remain binding.
- The selected backlog item explicitly requires an authored-FFNO cutoff and forbids leaving FFNO in an ambiguous “when available” state.
- The decision must preserve the current official CNS dataset and local metric family:
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
  - denormalized evaluation space
- The decision must preserve the current fairness rule for a headline table:
  - same official file
  - same split family
  - same history length
  - same normalization contract
  - same target metric schema
  - same cap/full-training status inside the table
- GNOT evidence is deliberately separate:
  - different environment: `ptycho311_2`
  - different paper-default recipe: `relative_l2`, `AdamW`, `OneCycleLR`
  - therefore it is not a same-contract local row unless the decision document explicitly widens the contract, which this plan forbids

## Prerequisite Status

- Satisfied from current durable state:
  - the official CNS file is staged, checksum-verified, and summarized
  - the CNS runner and reporting surfaces already support the fixed local metric family
  - required capped comparison lanes already exist for meaningful contract selection
  - the active downstream row-lock backlog item explicitly depends on this decision and is waiting for it
- Current coherent reusable lane A:
  - contract: `history_len=2`, `512 / 64 / 64` trajectories, `max_windows_per_trajectory=8`, `mse`, `10` and `40` epochs
  - completed same-contract rows already available:
    - `spectral_resnet_bottleneck_base`
    - `hybrid_resnet_cns`
    - `fno_base`
    - `unet_strong`
    - `author_ffno_cns_base`
    - `ffno_bottleneck_localconv_base` as repo-local FFNO-family context only
- Current coherent reusable lane B:
  - contract: `history_len=3`, `512 / 64 / 64` trajectories, `max_windows_per_trajectory=8`, `mse`, `10` and `40` epochs
  - completed same-contract rows already available:
    - `spectral_resnet_bottleneck_base`
    - `hybrid_resnet_cns`
    - `fno_base`
    - `unet_strong`
  - missing from this lane today:
    - authored FFNO same-contract row
    - repo-local local-conv FFNO same-contract row
- Benchmark gap that remains open regardless of lane:
  - no full-training same-contract CNS benchmark bundle currently exists on the full available training split
  - this is why the contract decision must explicitly weigh compute and deadline impact instead of implying “full training later”
- Adjacent evidence that is informative but not contract-defining for the headline row decision:
  - GNOT paper-default lane
  - `history_len=1` compare
  - modes-32 compare
  - shared-blocks10 larger-cap convergence follow-up

## Implementation Architecture

- **Audit unit:** Build one item-local contract matrix from the authoritative CNS summaries and referenced run roots. This unit owns lane separation, row status, parity fields, and compute-estimate inputs.
- **Decision unit:** Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md` with an explicit selected path, row list, FFNO cutoff/status, compute/deadline rationale, claim limits, and row-lock handoff rules.
- **Sync unit:** Update the downstream backlog items and durable discoverability surfaces so later implementation consumes the chosen CNS contract without rereading raw summaries or inferring hidden assumptions.

## Concrete File and Artifact Targets

### Code and Test Surfaces

- Audit and modify only if the decision cannot be made from current emitted metadata:
  - `scripts/studies/pdebench_image128/reporting.py`
  - `scripts/studies/pdebench_image128/run_config.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/run_pdebench_image128_suite.py`
  - `tests/studies/test_pdebench_image128_runner.py`
  - `tests/studies/test_pdebench_cfd_cns_data.py`
  - `tests/studies/test_pdebench_cfd_cns_metrics.py`

### Durable Docs and State

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- Modify: `docs/backlog/active/2026-04-29-cns-paper-benchmark-rows.md`
- Modify only if the chosen contract changes required CNS manifest/claim wording:
  - `docs/backlog/active/2026-04-29-paper-evidence-package-audit.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify `docs/findings.md` only if implementation discovers a stable reusable policy rather than a paper-local decision

### Required Artifacts

- Create artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/`
- Create verification log root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/verification/`
- Create audit outputs:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_same_contract_audit.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_same_contract_audit.json`
- Create compute-estimate outputs:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_full_training_cost_estimate.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_full_training_cost_estimate.json`

## Execution Guardrails

- The selected backlog item’s required deterministic checks stay mandatory. Any focused selector in this plan is additive, not a replacement.
- No expensive training belongs in this item by default. If implementation concludes a narrow fresh run is absolutely required to unblock the decision, it must first:
  - document why existing artifacts are insufficient
  - get green deterministic checks
  - keep the run inside this item’s explicit scope
- If a test, import, path, or harness failure occurs, diagnose, fix, and rerun before considering the item blocked.
- Reserve `BLOCKED` for missing resources, unavailable hardware, roadmap conflict, user decision required, external dependency outside current authority, or an unrecoverable failure after a documented narrow fix attempt.
- If any narrow rerun becomes unavoidable, use tmux, activate `ptycho311` unless an external baseline explicitly requires another approved environment, track the exact launched PID, avoid duplicate launches into the same output root, and require both exit code `0` and fresh required artifacts before treating the rerun as complete.
- Any downstream full-training work selected by this decision must wait for green deterministic checks and belongs to `2026-04-29-cns-paper-benchmark-rows`, not to this decision item.
- Preserve the pointer contract: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-cns-paper-contract-decision/plan-phase/plan_path.txt` must continue to contain only `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/execution_plan.md`.

## Required Deterministic Checks

The selected backlog item requires these unchanged checks:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Run this focused selector before any narrow rerun or any code change justified by a metadata gap:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'reference_run_manifest or cross_run_compare or author_ffno or gnot' -v
```

Archive passing logs under the active artifact root per `docs/TESTING_GUIDE.md`.

## Task 1: Audit the Existing CNS Evidence Into Coherent Contract Lanes

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_same_contract_audit.md`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_same_contract_audit.json`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`
- Modify code/test surfaces only if the audit exposes a real emitted-metadata gap

- [ ] Run the required deterministic checks and archive the passing logs before trusting any existing contract-emission surfaces.
- [ ] Build a lane-by-lane audit that separates:
  - coherent `history_len=2` same-contract evidence
  - coherent `history_len=3` same-contract evidence
  - optional but contract-divergent GNOT evidence
  - bounded local FFNO-family proxy evidence that is not the authored FFNO row
- [ ] For each candidate row, record:
  - row ID
  - run root
  - environment
  - dataset path
  - split counts
  - `history_len`
  - `max_windows_per_trajectory`
  - normalization contract
  - training loss
  - model profile ID
  - optimizer/scheduler fields needed to restate the paper contract if available
  - epoch budget
  - metrics
  - parameter count
  - runtime
  - evidence status
- [ ] Make the audit explicit about authored-FFNO availability by lane:
  - present and same-contract in lane A
  - absent in lane B unless a new same-contract run is funded downstream
- [ ] If the audit reveals a missing machine-readable field that blocks parity or compute estimation, patch only the narrowest runner/reporting surface needed, rerun the focused selector, then rerun the required backlog-item checks.

**Verification**

- [ ] `cns_same_contract_audit.md` and `.json` exist and separate lane A, lane B, and optional GNOT.
- [ ] The audit explicitly labels whether each row is:
  - same-contract reusable
  - adjacent but contract-divergent
  - blocked / unavailable for the selected lane
- [ ] Archived logs exist for the required pytest and compileall checks.

## Task 2: Estimate Full-Training Cost and Decision Consequences

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_full_training_cost_estimate.md`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/cns_full_training_cost_estimate.json`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
- Audit: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`

- [ ] Define the benchmark target that full training would have to satisfy:
  - official CNS file
  - full available training split after validation/test holdout
  - fixed normalization contract
  - fixed training-recipe contract, including whether CNS keeps the task-local `mse` override or returns to the design baseline recipe
  - fixed denormalized metric family
  - no capped rows silently promoted to benchmark evidence
- [ ] Derive the decision-relevant cost estimate from existing capped runs and emitted window counts rather than launching new long runs inside this item.
- [ ] Estimate, per viable lane, the total work still missing before a same-contract full-training headline table exists, including:
  - required row IDs
  - already completed rows reusable only as decision-support inputs
  - rows that would need fresh full-training execution
  - extra rows that would be needed only because a lane currently lacks authored FFNO
- [ ] Make the history-length consequence explicit:
  - choosing lane A preserves an already completed authored FFNO same-contract row for capped evidence selection
  - choosing lane B creates a same-contract authored-FFNO gap that the downstream row-lock item must either run by cutoff or mark `blocked` / `not_protocol_compatible`
- [ ] Include compute/deadline rationale relative to the surrounding campaign, especially the cost of delaying CNS row locking or CDI work.

**Verification**

- [ ] `cns_full_training_cost_estimate.md` and `.json` exist.
- [ ] The estimate distinguishes “rows already available under the chosen contract” from “rows that must still be run.”
- [ ] The estimate names the full-training versus bounded-capped consequence for authored FFNO explicitly.

## Task 3: Write the Durable CNS Paper Contract Decision

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- Inputs: the Task 1 and Task 2 audit artifacts

- [ ] Write the durable decision document with these required sections:
  - `Context And Authority`
  - `Audited Contract Lanes`
  - `Selected contract:`
  - `Selected history lane:`
  - `Selected normalization contract:`
  - `Selected training recipe contract:`
  - `Compute And Deadline Rationale`
  - `Required CNS Rows`
  - `Authored FFNO cutoff:`
  - `Authored FFNO status:`
  - `Claim Boundary`
  - `Stop / Failure Criteria`
  - `Downstream Handoff`
- [ ] Make the selected path explicit as exactly one of:
  - `full_training_paper_benchmark`
  - `bounded_capped_decision_support`
- [ ] If `full_training_paper_benchmark` is selected, define:
  - exact official dataset / file identity
  - exact split policy
  - exact history length
  - exact normalization contract
  - exact training recipe contract, including loss mode, optimizer, learning rate, scheduler family, scheduler floor/threshold, and whether CNS keeps the task-local `mse` override relative to the design baseline
  - required model profiles
  - epoch/budget policy
  - metric schema
  - provenance requirements
  - failure/abort conditions
  - authored FFNO inclusion cutoff and what happens if it misses the cutoff
- [ ] If `bounded_capped_decision_support` is selected, define:
  - exact official dataset / file identity
  - exact cap and fixed manifest roots
  - exact split policy and history length for the frozen capped lane
  - exact normalization contract
  - exact training recipe contract, including loss mode, optimizer, learning rate, scheduler family, scheduler floor/threshold, and whether CNS keeps the task-local `mse` override relative to the design baseline
  - exact required rows and row statuses
  - missing full-training caveat
  - exact allowed CNS claims and forbidden benchmark claims
  - authored FFNO cutoff/status and whether any additional same-contract row is still required for table coherence
- [ ] Keep GNOT outside the required headline bundle unless the decision explicitly labels it as optional context with caveats.
- [ ] Do not leave any “decide later” ambiguity around history length, normalization, training recipe, FFNO inclusion, or row status labels.

**Verification**

- [ ] Run this document-structure check and archive the log:

```bash
python - <<'PY'
from pathlib import Path
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md")
text = path.read_text() if path.exists() else ""
required = [
    "Selected contract:",
    "Selected history lane:",
    "Selected normalization contract:",
    "Selected training recipe contract:",
    "Compute And Deadline Rationale",
    "Required CNS Rows",
    "Authored FFNO cutoff:",
    "Authored FFNO status:",
    "Claim Boundary",
    "Stop / Failure Criteria",
    "Downstream Handoff",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"decision doc missing required sections: {missing}")
print("decision doc structure looks complete")
PY
```

## Task 4: Sync Downstream Backlog and Durable Discoverability

**Files:**
- Modify: `docs/backlog/active/2026-04-29-cns-paper-benchmark-rows.md`
- Modify only if needed: `docs/backlog/active/2026-04-29-paper-evidence-package-audit.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`

- [ ] Update `2026-04-29-cns-paper-benchmark-rows.md` so it no longer asks implementation to infer the contract. It must name the selected contract, required row IDs/statuses, and the authored-FFNO cutoff outcome from the new decision doc.
- [ ] Update `2026-04-29-cns-paper-benchmark-rows.md` so it also restates the selected history lane, normalization contract, and training-recipe contract that downstream row-lock execution must obey, including the explicit CNS loss-mode choice and any scheduler-floor binding.
- [ ] Update `paper_evidence_package_design.md` so the open CNS-contract decision is closed by reference to the new durable decision doc instead of remaining open-ended, and so the package design stops requiring future readers to recover normalization or recipe rules from older summaries.
- [ ] Update `pdebench_2d_cfd_cns_summary.md`, `docs/index.md`, and `docs/studies/index.md` so the CNS paper-contract decision becomes a discoverable authority surface for the full-vs-capped choice, selected history lane, normalization contract, and training-recipe binding.
- [ ] Update `progress_ledger.json` with a concise durable note that records:
  - selected contract
  - selected history lane
  - selected normalization contract
  - selected training-recipe contract
  - authored FFNO status/cutoff outcome
  - whether CNS remains bounded capped evidence or proceeds to full-training row locking
- [ ] Update `docs/backlog/active/2026-04-29-paper-evidence-package-audit.md` only if the decision changes what the audit item must expect from CNS row statuses or manifest paths.
- [ ] Update `docs/findings.md` only if implementation discovers a reusable policy that should outlive this paper lane. Otherwise keep the result in the summary/decision docs only.

**Verification**

- [ ] Run this downstream-consistency check and archive the log:

```bash
python - <<'PY'
from pathlib import Path
decision = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md")
text = decision.read_text()
required_lines = {}
for prefix in [
    "Selected contract:",
    "Selected history lane:",
    "Selected normalization contract:",
    "Selected training recipe contract:",
]:
    line = next((ln for ln in text.splitlines() if ln.startswith(prefix)), None)
    if line is None:
        raise SystemExit(f"missing {prefix} line")
    required_lines[prefix] = line

row_lock = Path("docs/backlog/active/2026-04-29-cns-paper-benchmark-rows.md")
row_lock_body = row_lock.read_text() if row_lock.exists() else ""
missing = []
if "pdebench_cns_paper_contract_decision.md" not in row_lock_body:
    missing.append(f"{row_lock}: missing decision-doc reference")
for line in required_lines.values():
    if line not in row_lock_body:
        missing.append(f"{row_lock}: missing exact contract line `{line}`")

for path in [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md"),
]:
    body = path.read_text() if path.exists() else ""
    for needle in [
        "pdebench_cns_paper_contract_decision.md",
        required_lines["Selected contract:"],
        required_lines["Selected history lane:"],
    ]:
        if needle not in body:
            missing.append(f"{path}: missing `{needle}`")
    for keyword in ["normalization", "training recipe"]:
        if keyword not in body.lower():
            missing.append(f"{path}: missing {keyword} authority mention")

if missing:
    raise SystemExit("downstream contract sync incomplete:\\n- " + "\\n- ".join(missing))
print("downstream files reference the decision doc and selected contract")
PY
```

## Task 5: Final Verification and Completion Evidence

**Files:**
- Verification log outputs under `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/verification/`
- No additional production files beyond those already named

- [ ] Re-run the required deterministic checks if any code/test surface changed during the audit.
- [ ] If the work stayed doc-only, keep the archived passing required-check logs from Task 1 and archive the Task 3 and Task 4 consistency checks alongside them.
- [ ] Run one final consistency check that extracts:
  - `Selected contract:`
  - `Selected history lane:`
  - `Selected normalization contract:`
  - `Selected training recipe contract:`
  - `Authored FFNO status:`
  from the durable decision doc and verifies those exact labels are reflected consistently in the downstream backlog item and the ledger update.
- [ ] Confirm the `plan_path.txt` pointer still contains only the execution-plan path.

**Verification**

- [ ] The selected backlog item’s required deterministic checks are archived.
- [ ] The durable decision doc exists and passes the structure check.
- [ ] The downstream backlog/design/summary files reference the decision doc and chosen contract.
- [ ] The ledger update records the same contract choice, history lane, normalization contract, training-recipe contract, and authored-FFNO status as the durable decision doc.
- [ ] `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-29-cns-paper-contract-decision/plan-phase/plan_path.txt` still contains only:

```text
docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-contract-decision/execution_plan.md
```
