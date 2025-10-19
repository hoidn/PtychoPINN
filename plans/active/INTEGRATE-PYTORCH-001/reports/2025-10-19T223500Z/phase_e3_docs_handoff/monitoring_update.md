# Phase E3.D3 — Monitoring Cadence & Escalation Triggers Update

**Date:** 2025-10-19
**Initiative:** INTEGRATE-PYTORCH-001 (Phase E3.D3 — TEST-PYTORCH-001 Handoff Package)
**Artifact Hub:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T223500Z/phase_e3_docs_handoff/`
**Source Document:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/handoff_brief.md`

---

## Executive Summary

Extended the TEST-PYTORCH-001 Phase D3 handoff brief with comprehensive monitoring cadence guidance and explicit escalation trigger matrix. Updates address the operational gap identified in Phase E3.A inventory: how to sustain PyTorch integration regression health post-handoff.

**Key Additions:**
1. **Monitoring Frequency Detail** (§2.2): Per-PR, nightly, and weekly validation schedules with runtime budgets
2. **Escalation Triggers Matrix** (§3.4): 12 automated alert conditions with severity levels, notification targets, and response SLAs
3. **Escalation Workflow Updates** (§3.5): Added trigger ID matching and cross-reference to §3.4 table

---

## 1. Monitoring Cadence Additions (§2.2)

### 1.1. Per-PR Pre-Merge Requirements

**Mandatory Selectors:**
- Integration Workflow + Backend Selection Suite as blocking gate
- Budget: ≤2 minutes total (90s integration + 5s backend suite + overhead)
- Scope: All PRs touching `ptycho_torch/`, `tests/torch/`, or backend selection code

**Rationale:** Prevents regression introduction at merge time; aligns with runtime guardrails from `runtime_profile.md` (≤90s CI budget).

### 1.2. Nightly Automated Runs

**Coverage:**
- Full Parity Validation Suite (config bridge, Lightning orchestration, stitching, checkpoint serialization, decoder parity)
- Runtime trend monitoring with 60s warning threshold (1.7× baseline)
- Artifact archival under `plans/active/TEST-PYTORCH-001/reports/<timestamp>/nightly/`

**Rationale:** Detects drift from cumulative changes across multiple PRs; early warning for sustained runtime degradation.

### 1.3. Weekly Deep Validation

**Operations:**
- Full torch test suite: `pytest tests/torch/ -vv`
- Cross-backend checkpoint compatibility validation
- Environment refresh (Python/PyTorch/Lightning version updates + guardrail revalidation)

**Rationale:** Comprehensive regression check with dependency updates; ensures guardrails remain calibrated as ecosystem evolves.

---

## 2. Escalation Trigger Matrix (§3.4)

### 2.1. Trigger Categories

Added 12 distinct trigger conditions across four severity levels:

| Category | Trigger Count | Severity Range |
|:---------|:--------------|:---------------|
| **Runtime Guardrails** | 3 | CRITICAL, WARNING |
| **Test Failures** | 3 | CRITICAL, HIGH |
| **Policy Violations** | 3 | HIGH, MEDIUM |
| **Parity Regressions** | 1 | HIGH |
| **Artifact Integrity** | 2 | HIGH, MEDIUM |

### 2.2. Critical Triggers (Immediate Response Required)

| ID | Condition | Response SLA | Target |
|:---|:----------|:-------------|:-------|
| **RT-001** | Runtime >90s | <4 hours | TEST-PYTORCH-001 owner |
| **FAIL-001** | Integration test FAILED | <2 hours | Both initiatives |
| **FAIL-002** | Backend selection suite FAILED | <2 hours | INTEGRATE-PYTORCH-001 owner |

**Source:** Runtime thresholds from `runtime_profile.md` §3.1 (90s = 2.5× baseline, 60s = 1.7× baseline).

### 2.3. High Priority Triggers

| ID | Condition | Response SLA | Target |
|:---|:----------|:-------------|:-------|
| **FAIL-003** | Checkpoint loading TypeError | <4 hours | INTEGRATE-PYTORCH-001 owner |
| **POLICY-001** | PyTorch ImportError (local dev) | Immediate | Developer + TEST-PYTORCH-001 owner |
| **CONFIG-001** | Shape mismatch (gridsize sync) | <4 hours | INTEGRATE-PYTORCH-001 owner |
| **PARITY-001** | Decoder shape mismatch | <4 hours | INTEGRATE-PYTORCH-001 owner |
| **ARTIF-001** | Missing checkpoint hyperparameters | <4 hours | INTEGRATE-PYTORCH-001 owner |

**Rationale:** These conditions indicate fundamental contract violations (POLICY-001, CONFIG-001) or regressions in Phase D fixes (FAIL-003, PARITY-001, ARTIF-001).

### 2.4. Medium Priority Triggers

| ID | Condition | Response SLA | Target |
|:---|:----------|:-------------|:-------|
| **FORMAT-001** | NPZ transpose IndexError | <24 hours | INTEGRATE-PYTORCH-001 owner |
| **ARTIF-002** | Reconstruction PNGs <1KB or missing | <24 hours | INTEGRATE-PYTORCH-001 owner |

**Rationale:** Known edge cases with existing mitigations (FORMAT-001 auto-transpose heuristic); artifact integrity issues that don't block core workflow.

### 2.5. Automated Alert Logic

Provided pseudo-code (§3.4) for CI monitoring hook demonstrating:
- Runtime threshold branching (>90s CRITICAL, >60s WARNING, <20s WARNING)
- Error message pattern matching for typed failures (checkpoint/shape/generic)
- Owner routing per severity and failure type

**Reference:** `specs/ptychodus_api_spec.md` §4.8 (backend routing), `docs/findings.md` (POLICY-001, CONFIG-001, FORMAT-001).

---

## 3. Escalation Workflow Enhancements (§3.5)

### 3.1. Added Trigger ID Matching

**Change:** Step 3 ("Document Failure") now requires matching the failure against escalation trigger IDs from §3.4 table.

**Benefit:** Standardizes failure classification; enables automated alert routing and SLA tracking.

### 3.2. Expanded Issue Filing

**Change:** Step 4 ("File Issue") now specifies:
- Append to `docs/fix_plan.md` Attempts history for owning initiative per §3.3 ownership matrix **AND** §3.4 trigger target
- Include trigger ID in escalation artifact directory naming/metadata

**Benefit:** Cross-references ownership matrix (component-based) with trigger matrix (condition-based) for unambiguous responsibility assignment.

### 3.3. New Authority Reference

**Change:** Step 5 ("Reference Authorities") now includes cross-reference to "This document §3.4" for escalation trigger definitions.

**Benefit:** Closes documentation loop; ensures escalation workflow always references canonical trigger list.

---

## 4. Cross-References & Alignment

### 4.1. Runtime Profile Integration

**Source:** `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md`

**Thresholds Used:**
- 35.92s baseline (mean from Phase C/D runs)
- 60s warning threshold (1.7× baseline)
- 90s CI budget (2.5× baseline)
- 20s minimum (incomplete execution indicator)

**Alignment:** All runtime triggers (RT-001, RT-002, RT-003) directly reference §3.1 guardrails from runtime profile.

### 4.2. Specification Contracts

**Backend Selection:** `specs/ptychodus_api_spec.md` §4.8 — Backend routing guarantees, fail-fast behavior
**Data Contracts:** `specs/data_contracts.md` §1 — NPZ format requirements (diffraction=amplitude, float32)

**Triggers Enforcing Specs:**
- **FAIL-002** → Spec §4.8 backend routing violations
- **FORMAT-001** → Data contract §1 legacy format handling

### 4.3. Policy Findings

**POLICY-001:** `docs/findings.md#POLICY-001` — PyTorch >=2.2 mandatory
**CONFIG-001:** `docs/findings.md#CONFIG-001` — `update_legacy_dict` requirement
**FORMAT-001:** `docs/findings.md#FORMAT-001` — NPZ auto-transpose guard

**Triggers Enforcing Policies:**
- **POLICY-001 Trigger** → Missing PyTorch installation (local dev)
- **CONFIG-001 Trigger** → Shape mismatch from params.cfg desync
- **FORMAT-001 Trigger** → NPZ transpose IndexError

---

## 5. Artifact Summary

| Artifact | Size | Purpose |
|:---------|:-----|:--------|
| `handoff_brief.md` (updated) | ~25 KB (+3.5 KB) | Extended with §2.2 monitoring cadence detail, §3.4 escalation trigger matrix, §3.5 workflow updates |
| `monitoring_update.md` (this file) | 6.5 KB | Summary of Phase E3.D3 updates with rationale and cross-references |

**Storage Location:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T223500Z/phase_e3_docs_handoff/`

**Handoff Brief Diff:**
- **§2.2 Monitoring Cadence:** +22 lines (per-PR, nightly, weekly guidance)
- **§3.4 Escalation Triggers:** +40 lines (12 triggers table + automated alert pseudo-code)
- **§3.5 Escalation Workflow:** +3 lines (trigger ID matching, cross-reference)

---

## 6. Exit Criteria Validation

| Task | Status | Evidence |
|:-----|:-------|:---------|
| **D3.A: Monitoring Cadence** | ✅ | §2.2 extended with per-PR/nightly/weekly schedules + runtime budgets |
| **D3.B: Escalation Triggers** | ✅ | §3.4 new section with 12 triggers, severity levels, targets, SLAs |
| **D3.C: Workflow Integration** | ✅ | §3.5 updated with trigger ID matching + authority cross-reference |

**Phase E3.D3 Status:** **COMPLETE** — All D3 tasks executed per `phase_e3_docs_plan.md` guidance.

---

## 7. Next Steps (Post-Handoff)

### 7.1. Immediate Actions (TEST-PYTORCH-001 Owner)

1. **Implement CI Monitoring Hook:**
   - Translate §3.4 pseudo-code alert logic into CI configuration (GitHub Actions/Jenkins/etc.)
   - Configure notification targets per ownership matrix (§3.3) + trigger targets (§3.4)

2. **Establish Baseline Artifact Archive:**
   - Create `plans/active/TEST-PYTORCH-001/reports/<timestamp>/nightly/` directory structure
   - Configure nightly cron job for Parity Validation Suite

3. **Validate Escalation Workflow:**
   - Test-trigger RT-002 warning by intentionally degrading runtime (e.g., larger dataset)
   - Verify alert routing, artifact capture, and docs/fix_plan.md update procedure

### 7.2. Governance Review (INTEGRATE-PYTORCH-001 Owner)

1. **Update Phase E3 Plan:**
   - Mark D3.A/D3.B rows `[x]` in `phase_e3_docs_plan.md`
   - Append Attempt to `docs/fix_plan.md` [INTEGRATE-PYTORCH-001-STUBS] history

2. **Close Phase E3:**
   - Confirm all E3.A (gap assessment), E3.B (docs updates), E3.C (spec sync), E3.D (handoff) tasks complete
   - Author Phase E3 closure summary with artifact inventory

3. **Propose INTEGRATE-PYTORCH-001 Closure:**
   - Draft closure recommendation citing Phase D2 parity achievement + Phase E3 handoff completion
   - Submit for governance sign-off

---

## 8. Open Questions & Recommendations

### 8.1. CI Environment Configuration

**Question:** Which CI platform will host the automated monitoring hook (GitHub Actions, Jenkins, custom)?

**Recommendation:** Start with GitHub Actions workflow triggered on:
- PR open/sync events (for per-PR gate)
- Schedule cron (for nightly/weekly runs)

**Sample Workflow Snippet:**
```yaml
name: PyTorch Integration Monitoring
on:
  pull_request:
    paths:
      - 'ptycho_torch/**'
      - 'tests/torch/**'
  schedule:
    - cron: '0 2 * * *'  # Nightly at 2 AM UTC

jobs:
  integration-test:
    runs-on: ubuntu-latest
    timeout-minutes: 5  # 90s test + overhead
    env:
      CUDA_VISIBLE_DEVICES: ""
    steps:
      - uses: actions/checkout@v3
      - name: Run Integration Test
        run: pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
      - name: Check Runtime Threshold
        run: |
          if [ "$RUNTIME" -gt 90 ]; then
            echo "::error::RT-001 CRITICAL - Runtime exceeded 90s"
            exit 1
          fi
```

### 8.2. Alert Notification Targets

**Question:** How should alerts be delivered (Slack, email, GitHub Issues)?

**Recommendation:** Multi-channel strategy:
- **CRITICAL (RT-001, FAIL-001, FAIL-002):** Slack + email + auto-create GitHub Issue
- **HIGH (all other HIGH triggers):** Slack + email
- **WARNING/MEDIUM:** Slack only (aggregated daily digest)

### 8.3. Historical Trend Tracking

**Question:** Should runtime trends be visualized (e.g., dashboard, plots)?

**Recommendation:** Archive pytest JSON reports (`--json-report` flag) from nightly runs; visualize with simple script:
```bash
# Nightly run command
pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv \
  --json-report --json-report-file=plans/active/TEST-PYTORCH-001/reports/$(date +%Y-%m-%dT%H%M%SZ)/nightly/pytest_report.json
```

---

## References

### Normative Sources
- `specs/ptychodus_api_spec.md` §4.8 — Backend Selection & Dispatch
- `specs/data_contracts.md` §1 — NPZ format requirements
- `docs/findings.md` — POLICY-001, CONFIG-001, FORMAT-001

### Evidence & Guidance
- `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md` — Runtime guardrails authority
- `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` — Phase E checklist
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md` — Phase E3 task breakdown

### Updated Documents
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/handoff_brief.md` — Monitoring + escalation guidance

---

**Document Status:** FINAL — Phase E3.D3 complete, ready for governance review.
**Next Artifact:** Phase E3 closure summary (to be authored by supervisor).
