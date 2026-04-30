# Paper Evidence Package Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task-by-task. Keep this file as the execution authority for the selected backlog item.

**Goal:** Produce a repo-local paper evidence manifest and completeness audit that states exactly what the manuscript can claim now, what remains bounded or decision-support only, and what must stay blocked or placeholder-only.

**Architecture:** Treat the completed CDI `lines128` paper bundle and the locked CNS capped bundle as the only scientific authorities for this item, normalize them into one shared manifest schema, then emit a durable audit summary that preserves their different claim boundaries instead of collapsing them into one generic success label. Keep all outputs under the NeurIPS plan tree and repo-local artifact roots; `/home/ollie/Documents/neurips/` remains out of scope until the later paper-facing bundle phase.

**Tech Stack:** Python 3.11, JSON, Markdown, repo-local artifact inspection, `scripts/studies/paper_provenance.py`, lightweight audit helpers under `scripts/studies/`, pytest.

---

## Selected Backlog Objective

- Create a repo-local paper evidence manifest plus durable audit summary for the NeurIPS Hybrid ResNet campaign.
- The audit must tell downstream manuscript work:
  - which CDI and CNS tables/figure bundles are currently authoritative
  - which rows and pillars are `paper_grade`, `capped_decision_support`, or only `decision_support`
  - which claims are draftable now
  - which claims must remain placeholders or explicitly blocked

## Scope

In scope:

- consume the completed CDI `lines128` paper-benchmark authority and the bounded CNS paper bundle authority
- emit `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- emit `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`
- preserve the CNS pillar exactly as the approved bounded capped `history_len=2` contract from `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- record table rows, figure bundles, source-array roots, metric schemas, provenance roots, row roles, row statuses, and claim boundaries
- state which manuscript sections or result subsections are draftable versus placeholder-only
- update durable index surfaces when the new audit becomes a first-class authority

Out of scope:

- creating or populating `/home/ollie/Documents/neurips/`
- rerunning CDI or CNS experiments
- reopening completed benchmark row rosters, fairness contracts, or selected comparators
- upgrading capped CNS evidence to `full_training` by prose alone
- manuscript prose beyond the repo-local audit summary and draftability matrix
- later roadmap phases or candidate additive lanes

## Explicit Non-Goals

- Do not relabel missing provenance as acceptable merely because metrics exist.
- Do not collapse CDI and CNS into one combined headline status.
- Do not treat the CDI minimum subset or FFNO prerequisite pair as the current headline CDI authority when the complete six-row bundle exists.
- Do not patch benchmark semantics to make the audit easier; if a source authority disagrees or is incomplete, record that disagreement explicitly.
- Do not launch long-running training, benchmark, or recovery jobs in this item. This is an audit-and-publication tranche, not an execution rerun tranche.

## Binding Constraints And Prerequisite Status

Strategic and roadmap constraints:

- `docs/steering.md` requires explicit equal-footing wording and forbids silently relaxing fairness constraints.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` keeps this work inside the repo-local evidence package lane; Phase 5 `/home/ollie/Documents/neurips/` assembly remains closed.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md` and the package design keep the paper two-pillar structure fixed:
  - CDI `lines128` anchor
  - PDEBench CNS generalization evidence
- `selected_item_context.md` is authoritative for this backlog item and explicitly requires:
  - a structured manifest under the NeurIPS plan tree
  - a durable summary with explicit labels for `paper_grade`, `full_training`, `capped_decision_support`, `decision_support`, `blocked`, and `not_protocol_compatible`
  - preservation of the CNS bounded capped contract

Progress-ledger status that matters here:

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` already records Phase 0, Phase 1, and early Phase 2 groundwork as complete.
- The ledger snapshot does not yet encode the later 2026-04-29/2026-04-30 backlog completions this audit depends on, so implementation must treat the checked-in summary authorities below as binding current prerequisites rather than waiting for a ledger rewrite in this item.

Binding prerequisite authorities:

- CDI complete headline authority:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
  - authoritative root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- CDI adjacent context that may still appear in the audit as non-headline support:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_minimum_paper_table_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
- CNS contract and locked bundle authorities:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_table_figure_bundle_summary.md`
  - locked-row JSON:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`
  - bundle root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/`
- Existing durable evidence registry:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

Failure-handling policy for this item:

- Normal import, path, JSON-shape, or test failures must be diagnosed, fixed narrowly, and rerun.
- If the audit finds a missing field or provenance gap in an already accepted scientific authority, record the claim limitation in the manifest and summary rather than silently broadening scope into a rerun.
- Reserve `BLOCKED` only for a missing authoritative artifact, irreconcilable source-authority conflict, external dependency outside current authority, or another unrecoverable issue after a documented narrow fix attempt.

## Implementation Architecture

- **Evidence ingestion unit**
  - Load authoritative CDI and CNS docs plus their machine-readable manifests from existing artifact roots.
  - Prefer JSON payloads as data sources and use Markdown summaries as claim-boundary and root-selection authorities.
- **Normalization and audit unit**
  - Convert heterogeneous pillar artifacts into one shared schema for rows, bundles, statuses, claim boundaries, provenance gaps, and manuscript draftability.
  - Fail closed when source surfaces disagree on root identity, bundle status, or claim boundary.
- **Publication and discoverability unit**
  - Write the final manifest and durable audit summary under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.
  - Synchronize `paper_evidence_index.md` and, if needed, `docs/index.md` so later planning and manuscript work can discover this audit without rereading the backlog item.

## Concrete File And Artifact Targets

Likely code targets:

- Create: `scripts/studies/paper_evidence_audit.py`
- Modify only if reusable helpers are needed: `scripts/studies/paper_provenance.py`

Likely test targets:

- Create: `tests/studies/test_paper_evidence_audit.py`
- Modify if helper behavior changes: `tests/studies/test_paper_provenance.py`

Durable output targets:

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify `docs/index.md` if the new audit summary becomes a top-level discoverable authority

Item-local artifact root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/`

Expected item-local artifacts:

- `audit_inputs.json`
- `audit_validation.json`
- archived pytest logs under `verification/`
- any focused schema/debug outputs needed to prove manifest-summary consistency

## Execution Checklist

### Tranche 1: Freeze Audit Inputs, Status Vocabulary, And Shared Schema

- [ ] Create a machine-readable audit-input manifest at:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/audit_inputs.json`
- [ ] Record the exact authoritative CDI and CNS inputs this item will consume:
  - CDI complete bundle summary and authoritative root
  - CDI minimum-subset and FFNO-pair summaries as adjacent context only
  - CNS contract decision, row-lock summary, locked-row JSON, and table/figure bundle summary
  - existing `paper_evidence_index.md`
- [ ] Define one shared manifest schema covering at minimum:
  - `pillar_id`
  - `artifact_kind`
  - `artifact_id`
  - `row_id`
  - `row_role`
  - `row_status`
  - `claim_boundary`
  - `evidence_tier`
  - `source_summary`
  - `source_root`
  - `metric_schema`
  - `table_artifacts`
  - `figure_artifacts`
  - `source_array_roots`
  - `provenance_gaps`
  - `draftability`
  - `blocked_claims`
- [ ] Freeze the required normalized status vocabulary and meanings in the implementation:
  - `paper_grade`
  - `full_training`
  - `capped_decision_support`
  - `decision_support`
  - `blocked`
  - `not_protocol_compatible`
- [ ] Freeze the source-to-status mapping rules before writing any outputs:
  - the current six-row CDI bundle is the headline CDI authority and must be surfaced as `paper_grade`
  - the CNS headline bundle must remain `capped_decision_support`
  - the CDI minimum subset and FFNO prerequisite pair may only appear as adjacent `decision_support` context, not as current headline authority
  - `full_training` must only be emitted when an authoritative source explicitly proves a full-training paper lane; do not infer it

Verification before moving on:

- [ ] Run the backlog item’s required deterministic check verbatim; no narrower replacement is allowed:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing evidence package inputs: {missing}")
print("evidence package inputs present")
PY
```

- [ ] Add a focused test that fails if a source-status mapping would collapse CDI `paper_grade` and CNS `capped_decision_support` into one undifferentiated outcome.

### Tranche 2: Implement The Cross-Pillar Audit Loader And Consistency Checks

- [ ] Implement `scripts/studies/paper_evidence_audit.py` as the repo-local orchestration entrypoint for this item.
- [ ] Prefer machine-readable artifact sources wherever available:
  - CDI `paper_benchmark_manifest.json`, `metrics.json`, and `model_manifest.json` from the authoritative complete root
  - CNS `cns_paper_locked_rows.json`, `bundle_validation.json`, table JSON/CSV/TeX pointers, figure-manifest pointers, and fixed-sample/scales manifests from the authoritative bundle root
- [ ] Use the Markdown summaries as authority for:
  - which root is authoritative
  - which claim boundary is allowed
  - whether a row is headline versus continuity/support versus adjacent context
- [ ] Fail closed if same-pillar machine-readable surfaces disagree on:
  - accepted root identity
  - roster size
  - top-level status
  - selected comparator
  - claim boundary
- [ ] Keep CNS normalization faithful to the approved contract:
  - headline rows:
    `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`, `author_ffno_cns_base`
  - continuity row:
    `hybrid_resnet_cns`
  - claim boundary:
    bounded capped decision-support only
- [ ] Keep CDI normalization faithful to the approved contract:
  - six accepted rows from the authoritative complete bundle
  - claim boundary:
    `complete_lines128_cdi_benchmark`
  - selected comparator:
    `fno_vanilla`
  - fixed seed:
    `3`
- [ ] If implementation needs reusable path, file-identity, or provenance-gap helpers, extend `scripts/studies/paper_provenance.py` rather than duplicating logic.

Verification for this tranche:

- [ ] Add focused tests covering:
  - successful CDI authority ingestion
  - successful CNS authority ingestion
  - disagreement detection across same-pillar sources
  - explicit preservation of the CNS capped claim boundary
  - rejection of `/home/ollie/Documents/neurips/` output paths for this item
- [ ] Run:

```bash
pytest -q tests/studies/test_paper_evidence_audit.py tests/studies/test_paper_provenance.py
```

### Tranche 3: Emit The Final Manifest And Durable Audit Summary

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`.
- [ ] The manifest must contain, at minimum:
  - authoritative input paths
  - pillar summaries for CDI and CNS
  - headline table and figure bundle authorities
  - per-row registry entries
  - claim-boundary registry
  - provenance-gap registry
  - manuscript draftability matrix
  - blocked or placeholder-only claims
- [ ] Ensure the manifest distinguishes:
  - current headline evidence
  - adjacent decision-support context
  - continuity/support rows
  - blocked or not-protocol-compatible future possibilities
- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`.
- [ ] The summary must explicitly:
  - define the status vocabulary in prose
  - name the authoritative CDI and CNS bundle roots
  - state that the CDI pillar is draftable as the current paper-grade anchor
  - state that the CNS pillar is draftable only with bounded capped wording
  - state that full-training CNS competitiveness claims remain blocked
  - identify which manuscript sections or subsections are ready to draft now and which must remain placeholders
- [ ] Keep the summary self-contained enough that later implementation or manuscript work does not need to reread the backlog item, steering doc, or roadmap to understand claim limits.
- [ ] If any expected source field is absent, emit an explicit gap in the manifest and summary rather than silently dropping the field.

Verification for this tranche:

- [ ] Add a deterministic manifest-validation test that asserts:
  - CDI headline authority is `paper_grade`
  - CNS headline authority is `capped_decision_support`
  - no current pillar is labeled `full_training` unless a source artifact explicitly says so
  - adjacent CDI context is not promoted above the six-row complete bundle
- [ ] Add a summary/manifest sync check so the same authoritative roots, statuses, and claim boundaries appear in both surfaces.
- [ ] Run the audit entrypoint against the real checked-in authorities and write a validation payload to:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/audit_validation.json`

### Tranche 4: Update Durable Index Surfaces

- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` with a new row for this backlog item.
- [ ] The new row should point to:
  - summary authority:
    `paper_evidence_package_audit_summary.md`
  - artifact root:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/`
  - outcome:
    repo-local cross-pillar audit of current claim-ready evidence, draftable sections, and blocked claims
- [ ] Preserve older evidence rows; do not replace them with the audit row.
- [ ] Update `docs/index.md` if this new audit summary is now the preferred discoverability surface for “what can the paper claim right now?”
- [ ] Do not update `/home/ollie/Documents/neurips/`; that remains a later phase.

Verification for this tranche:

- [ ] Confirm the new `paper_evidence_index.md` row names the same summary path and artifact root used by the emitted audit surfaces.
- [ ] If `docs/index.md` is updated, confirm the description preserves the distinction between:
  - the package design
  - the durable evidence index
  - the new current-state audit summary

### Tranche 5: Final Deterministic Closeout

- [ ] Rerun the required backlog check command and archive its stdout under the item-local `verification/` directory.
- [ ] Rerun the focused pytest selector and archive the log:

```bash
pytest -q tests/studies/test_paper_evidence_audit.py tests/studies/test_paper_provenance.py
```

- [ ] Run a final audit-output validation command that checks the real emitted manifest and summary for:
  - required output paths exist
  - the two pillar authorities are present
  - CDI is `paper_grade`
  - CNS is `capped_decision_support`
  - no `/home/ollie/Documents/neurips/` path appears as an emitted output target
- [ ] Record the final manifest path, summary path, and validation-log paths in the summary.

## Completion Criteria

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json` exists and is populated from the checked-in CDI and CNS authorities.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md` exists and is self-contained.
- The audit preserves the current scientific claim boundaries exactly:
  - CDI headline authority is `paper_grade`
  - CNS headline authority is `capped_decision_support`
  - no full-training CNS claim is implied
- The manifest and summary both identify draftable versus placeholder-only manuscript sections.
- `paper_evidence_index.md` is updated to reference the new audit summary without deleting prior evidence rows.
- Required deterministic checks pass and their logs are archived.

## Required Deterministic Checks

Implementation may use additional focused checks during development, but these are mandatory for completion:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing evidence package inputs: {missing}")
print("evidence package inputs present")
PY

pytest -q tests/studies/test_paper_evidence_audit.py tests/studies/test_paper_provenance.py
```
