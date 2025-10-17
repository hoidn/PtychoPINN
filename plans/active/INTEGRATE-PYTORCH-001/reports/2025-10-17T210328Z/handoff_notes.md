# Phase F4.3 — Initiative Handoff Notes

## Executive Summary

This document captures the required downstream actions following the torch-required transition (INTEGRATE-PYTORCH-001 Phase F1-F3). It serves as the handoff packet for initiative owners, CI maintainers, and release managers to execute without re-reading Phase F implementation history.

**Document Date:** 2025-10-17T210328Z
**Initiative:** INTEGRATE-PYTORCH-001 Phase F4.3
**Supersedes:** None (new handoff artifact)
**References:**
- Governance decision: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md`
- Phase F plan: `plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md`
- Test plan context: `plans/pytorch_integration_test_plan.md`
- Policy anchor: `docs/findings.md#POLICY-001`

---

## 1. Initiative Owner Matrix

This section identifies downstream initiatives impacted by the torch-required baseline and their required actions.

| Initiative/Owner | Required Action | Blocking Dependencies | Priority | References |
|:-----------------|:----------------|:---------------------|:---------|:-----------|
| **TEST-PYTORCH-001** | Author subprocess integration tests for PyTorch training/inference workflows (`tests/test_pytorch_integration.py`) | PyTorch installation in CI environment; fixture dataset creation | **P0** | `plans/pytorch_integration_test_plan.md:1-64` |
| **CI/CD Maintainers** | Add `torch>=2.2` to CI runner provisioning; validate import before pytest; document runner matrix | Access to CI workflow YAML files; ~500MB additional image size budget | **P0** | Governance §4.2 (`governance_decision.md:113-118`), §7.1 |
| **Ptychodus Integration** | Clone Ptychodus repository; test PyTorch backend activation paths; validate fail-fast error messages when torch unavailable | None (ready to execute) | **P1** | `specs/ptychodus_api_spec.md:140-162` (§4.2) |
| **Release Management** | Increment version to v2.0.0; draft CHANGELOG entry per governance §5.4; apply git tag; publish release notes | Phase F4 documentation completion | **P1** | Governance §5.4 (`governance_decision.md:222-250`) |
| **Documentation Team** | (Complete — Phase F4.1+F4.2) All developer-facing docs updated | N/A | **DONE** | `reports/2025-10-17T203640Z/doc_updates.md`, `reports/2025-10-17T205413Z/spec_sync.md` |

---

## 2. CI/CD Environment Updates

### 2.1 Required Changes

**Goal:** Ensure CI runners can execute `tests/torch/` suite without collection failures.

**Configuration Updates:**

1. **Runner Provisioning (add to CI workflow YAML):**
   ```yaml
   # Example GitHub Actions snippet
   - name: Install PyTorch
     run: |
       pip install torch>=2.2 torchvision --index-url https://download.pytorch.org/whl/cpu
       python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
   ```

2. **Validation Step (pre-test gate):**
   ```bash
   # Verify torch availability before pytest
   python -c "import torch; print(f'Torch version: {torch.__version__}')" || {
     echo "ERROR: PyTorch not available but required for torch tests";
     exit 1;
   }
   ```

3. **Runner Matrix Documentation:**
   Create CI configuration documentation specifying:
   - **PyTorch-required runners:** Execute full test suite including `tests/torch/`
   - **TensorFlow-only runners:** (Optional) Execute `tests/` excluding `tests/torch/` via directory-based skip (existing conftest logic)
   - **Minimum specs:** 4GB RAM recommended (per governance §2.3)
   - **Expected overhead:** ~30s provisioning time, ~500MB image size increase

### 2.2 Fallback Strategy

**If GPU runners unavailable:**
- Force CPU execution: `export CUDA_VISIBLE_DEVICES=""` before pytest
- PyTorch CPU wheels sufficient for test coverage (no CUDA requirement)
- Backend selection tests validate GPU unavailability handling

**Validation Command (see §3 below):**
```bash
pytest tests/torch/test_backend_selection.py -k pytorch_unavailable_raises_error -vv
```

### 2.3 Configuration File Targets

**Action Required:** Locate and update the following CI configuration files:
- GitHub Actions: `.github/workflows/*.yml` (if present)
- GitLab CI: `.gitlab-ci.yml` (if present)
- Other: Document CI platform and config paths in this section once identified

**Status:** Configuration file inventory pending (no CI YAML files found in repository root during Phase F surveys; may be managed externally).

---

## 3. Ongoing Verification Cadence

### 3.1 Authoritative Test Commands

Execute the following pytest selectors to validate torch-required baseline integrity:

**1. Collection Health Check:**
```bash
pytest --collect-only tests/torch/ -q
```
**Expected outcome:** Lists 66+ test items without `ImportError` or collection failures.

**2. Backend Selection Validation:**
```bash
pytest tests/torch/test_backend_selection.py -k pytorch_unavailable_raises_error -vv
```
**Expected outcome:** `PASSED` — confirms fail-fast behavior when torch import fails.

**3. Configuration Bridge Parity:**
```bash
pytest tests/torch/test_config_bridge.py -k parity -vv
```
**Expected outcome:** All parameterized parity tests `PASSED` — validates PyTorch→TensorFlow config translation.

**4. Full Torch Suite Regression:**
```bash
pytest tests/torch/ -v
```
**Expected outcome:** 66+ passed, 3 skipped (expected: `test_tf_helper.py` module-level skips for unimplemented helpers), 1 xfailed.

**5. Full Project Regression:**
```bash
pytest tests/ -v
```
**Expected outcome:** 207+ passed, 13 skipped, 1 xfailed (no new failures vs Phase F3.4 baseline).

### 3.2 Cadence Schedule

| Check Type | Frequency | Trigger | Owner |
|:-----------|:----------|:--------|:------|
| **Collection health** | Every commit to `tests/torch/` | Pre-merge CI gate | CI/CD Maintainers |
| **Backend selection** | Nightly or on torch stack changes | Scheduled CI job | TEST-PYTORCH-001 |
| **Config bridge parity** | On config schema changes | Manual validation | INTEGRATE-PYTORCH-001 |
| **Full torch suite** | Every pull request | Pre-merge CI gate | CI/CD Maintainers |
| **Full project regression** | Every pull request | Pre-merge CI gate | CI/CD Maintainers |

### 3.3 Log Archive Locations

Store pytest execution logs under:
```
plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/verification_<test_type>.log
```

Example:
```
plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T220000Z/verification_torch_suite.log
```

Cross-reference log paths in `docs/fix_plan.md` Attempts history for traceability.

---

## 4. Ledger and Plan Synchronization Steps

### 4.1 Plan Cross-Reference Updates

**Phase F completion requires:**

1. **Mark `phase_f_torch_mandatory.md` F4 row complete:**
   - File: `plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:60`
   - Change: `| F4.3 | Coordinate initiative handoffs | [ ] | ...` → `| F4.3 | Coordinate initiative handoffs | [x] | ✅ 2025-10-17 — Handoff notes created at reports/2025-10-17T210328Z/handoff_notes.md; owner matrix, CI updates, and verification cadence documented. Next: TEST-PYTORCH-001 activation + v2.0.0 release prep. |`

2. **Log `docs/fix_plan.md` Attempt:**
   - Append new attempt entry to INTEGRATE-PYTORCH-001 history
   - Format: `* [2025-10-17] Attempt #<N> — Phase F4.3 Handoff: <summary>`
   - Include artifact path: `reports/2025-10-17T210328Z/handoff_notes.md`
   - Note residual risks: TEST-PYTORCH-001 activation, CI configuration file identification, v2.0.0 release coordination

3. **Update `phase_f4_doc_sync.md` checklist:**
   - File: `plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md:46-48`
   - Mark F4.3.A, F4.3.B, F4.3.C tasks `[x]` with completion timestamps

### 4.2 Version Bump Trigger

**Action:** Increment to **v2.0.0** upon Phase F4 exit criteria satisfaction.

**Rationale:** Torch-optional → torch-required is a breaking change per semantic versioning (governance §5.4).

**CHANGELOG Entry Template:**
```markdown
## [2.0.0] - 2025-10-17

### BREAKING CHANGES
- **PyTorch is now required for `ptycho_torch/` backend.** Install with `pip install torch>=2.2`.
- Removed torch-optional execution guards from `ptycho_torch/` modules.
- `tests/torch/` test suite now assumes PyTorch availability (tests fail instead of skip when unavailable).
- Updated CLAUDE.md directive: torch-optional requirement superseded by torch-required policy (POLICY-001).

### Migration Guide
- **Developers:** Add `pip install torch>=2.2` to your environment setup.
- **CI/CD:** Update runners to include PyTorch in test environment (see `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T210328Z/handoff_notes.md`).
- **Ptychodus Users:** PyTorch is mandatory when `backend='pytorch'`; use `backend='tensorflow'` as fallback for constrained environments.

### Technical Details
- Implementation: INTEGRATE-PYTORCH-001 Phase F1-F4
- Governance: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md`
- Policy anchor: `docs/findings.md#POLICY-001`
```

**Git Tag Application:**
```bash
git tag -a v2.0.0 -m "Release 2.0.0: Torch-required transition (INTEGRATE-PYTORCH-001 Phase F)"
git push origin v2.0.0
```

**Release Notes:** Create GitHub/GitLab release referencing POLICY-001 and linking to governance decision artifact.

---

## 5. Residual Risks and Open Issues

### 5.1 Identified Risks

| Risk ID | Description | Impact | Mitigation | Owner |
|:--------|:------------|:-------|:-----------|:------|
| **R-H1** | CI configuration files not located in repository | CI updates blocked | Document external CI platform; coordinate with CI maintainers to apply torch installation steps | CI/CD Maintainers |
| **R-H2** | TEST-PYTORCH-001 starts before torch-required baseline stable | Test suite assumes torch-optional patterns | Gate TEST-PYTORCH-001 activation on Phase F4 completion; reference this handoff document in test plan | TEST-PYTORCH-001 Lead |
| **R-H3** | Ptychodus repository not cloned/tested locally | Integration validation incomplete | Clone Ptychodus, execute backend selection workflows, validate error messages | Ptychodus Integration Team |
| **R-H4** | v2.0.0 release coordination with external stakeholders delayed | Breaking change not communicated | Publish CHANGELOG and release notes promptly after Phase F4; notify user mailing list/forum if available | Release Management |

### 5.2 Follow-Up Items (docs/fix_plan.md TODO candidates)

1. **[TEST-PYTORCH-001] Activation Gate:** Create new docs/fix_plan.md item blocking TEST-PYTORCH-001 on Phase F4 completion. Reference this handoff document for CI prerequisites.

2. **[CI-CONFIG-001] Configuration File Inventory:** Identify CI platform and YAML file locations; apply torch installation steps per §2.1; validate with collection health check.

3. **[PTYCHODUS-INT-001] Backend Validation:** Clone Ptychodus repository; test `backend='pytorch'` selection; confirm fail-fast error messages align with spec §4.2; document results.

4. **[RELEASE-001] v2.0.0 Coordination:** Draft CHANGELOG per §4.2 template; apply git tag; publish GitHub/GitLab release; notify stakeholders per governance §7.2.

---

## 6. Handoff Verification Checklist

Use this checklist to confirm handoff completeness before closing Phase F4.3:

- [x] Owner matrix created with initiative names, actions, dependencies, and references (§1)
- [x] CI update instructions documented with provisioning, validation, and fallback strategies (§2)
- [x] Verification cadence defined with 5+ authoritative pytest selectors and expected outcomes (§3)
- [x] Ledger synchronization steps specified for `phase_f_torch_mandatory.md`, `docs/fix_plan.md`, and `phase_f4_doc_sync.md` (§4.1)
- [x] Version bump trigger and CHANGELOG template provided per governance §5.4 (§4.2)
- [x] Residual risks logged with mitigation strategies and follow-up TODO candidates (§5)
- [ ] Handoff artifact path (`reports/2025-10-17T210328Z/handoff_notes.md`) recorded in `docs/fix_plan.md` Attempt entry
- [ ] `phase_f4_doc_sync.md` F4.3 table updated with completion timestamps
- [ ] `phase_f_torch_mandatory.md` F4 row marked `[x]` with artifact reference

---

## 7. References

### 7.1 Phase F Artifacts

- **F1.1 Directive Conflict:** `reports/2025-10-17T184624Z/directive_conflict.md`
- **F1.2 Governance Decision:** `reports/2025-10-17T184624Z/governance_decision.md`
- **F1.3 Guidance Updates:** `reports/2025-10-17T184624Z/guidance_updates.md`
- **F2.1 Torch-Optional Inventory:** `reports/2025-10-17T192500Z/torch_optional_inventory.md`
- **F2.2 Test Skip Audit:** `reports/2025-10-17T192500Z/test_skip_audit.md`
- **F2.3 Migration Plan:** `reports/2025-10-17T192500Z/migration_plan.md`
- **F3.1 Dependency Update:** `reports/2025-10-17T193400Z/dependency_update.md`
- **F3.2 Guard Removal:** `reports/2025-10-17T193753Z/guard_removal_summary.md`
- **F3.3 Skip Rewrite:** `reports/2025-10-17T195624Z/skip_rewrite_summary.md`
- **F3.4 Regression Verification:** `reports/2025-10-17T201922Z/regression_summary.md`
- **F4.1 Documentation Updates:** `reports/2025-10-17T203640Z/doc_updates.md`
- **F4.2 Spec Synchronization:** `reports/2025-10-17T205413Z/spec_sync.md`
- **F4.3 Handoff Notes:** `reports/2025-10-17T210328Z/handoff_notes.md` (this document)

### 7.2 Normative Documents

- **Configuration API Spec:** `specs/ptychodus_api_spec.md`
- **Data Contracts:** `specs/data_contracts.md`
- **Knowledge Base:** `docs/findings.md#POLICY-001`
- **Testing Guide:** `docs/TESTING_GUIDE.md`
- **PyTorch Workflow:** `docs/workflows/pytorch.md`
- **Agent Directives:** `CLAUDE.md:57-59` (updated)

### 7.3 Plan Documents

- **Phase F Plan:** `plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md`
- **Phase F4 Doc Sync:** `plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md`
- **Phase F4 Handoff Guidance:** `plans/active/INTEGRATE-PYTORCH-001/phase_f4_handoff.md`
- **Test Plan Context:** `plans/pytorch_integration_test_plan.md`

---

**Handoff Summary:**
This document provides executable guidance for TEST-PYTORCH-001 activation, CI torch installation, Ptychodus integration validation, and v2.0.0 release coordination. All downstream owners have clear action items with dependencies, priorities, and reference anchors. Verification cadence defined with 5 authoritative pytest commands. Residual risks logged with mitigation strategies. Phase F4.3 ready for completion upon ledger synchronization (§4.1 checklist items).
