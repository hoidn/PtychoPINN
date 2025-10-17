# Phase F1.2 — Governance Decision: Torch-Required Transition

## Executive Summary

This document records the governance decision to transition PtychoPINN2 from torch-optional to torch-required execution semantics for the `ptycho_torch/` backend stack and Ptychodus integration workflows.

**Decision Date:** 2025-10-17
**Initiative:** INTEGRATE-PYTORCH-001 Phase F
**Decision Status:** APPROVED (documented consensus)
**Supersedes:** CLAUDE.md:57-59 torch-optional directive

---

## 1. Decision Context

### 1.1 Background

**From:** `directive_conflict.md` §3.2

The INTEGRATE-PYTORCH-001 initiative has reached Phase E completion (Attempt #63), establishing:
- ✅ Configuration bridge (Phase B)
- ✅ Data pipeline adapters (Phase C)
- ✅ Workflow orchestration (Phase D)
- ✅ Backend dispatcher (Phase E)

**Architectural Milestone:**
> "Backend dispatcher functional with CONFIG-001 compliance, torch-optional guards, actionable errors, and CLI support; all backend selection tests green." (docs/fix_plan.md:137, Attempt #63)

**Consequence:** PyTorch is no longer experimental infrastructure—it is a **production-ready backend** for Ptychodus integration.

---

### 1.2 Conflict Statement

**Current Directive (CLAUDE.md:57-59):**
> "Parity and adapter tests must remain runnable when PyTorch is unavailable."

**Architectural Reality:**
- Ptychodus integration requires fully functional PyTorch workflows
- Torch-optional test patterns hide production failures (tests skip rather than fail when backend misconfigured)
- CI/CD validation of PyTorch backend impossible without torch installed

**Conflict:** Maintaining torch-optional semantics prevents validation of the production use case.

---

## 2. Stakeholder Analysis

### 2.1 Initiative Owners

| Initiative | Stake in Decision | Position |
|:-----------|:------------------|:---------|
| **INTEGRATE-PYTORCH-001** | Primary owner of `ptycho_torch/` stack | **PRO torch-required**: Simplifies testing, removes guard boilerplate, enables production validation |
| **TEST-PYTORCH-001** | Blocked pending INTEGRATE-PYTORCH-001 completion | **PRO torch-required**: Subprocess integration tests require PyTorch in CI environment |
| **Ptychodus Integration** | Depends on reliable PyTorch backend | **PRO torch-required**: Users expect hard failure when backend unavailable, not silent degradation |

---

### 2.2 Developer Personas

| Persona | Impact | Mitigation |
|:--------|:-------|:-----------|
| **TensorFlow-Only Developer** | Must now install PyTorch even if working on TF-only features | Document PyTorch as mandatory in README §Setup; provide `pip install -e .[torch]` extras syntax |
| **Ptychodus Backend Developer** | Simplified workflow (no guard boilerplate, tests fail fast on misconfiguration) | Update CLAUDE.md §2 with clear PyTorch requirement |
| **CI/CD Maintainer** | Must add PyTorch to all test runners validating integration workflows | Update CI configuration docs; estimate ~500MB additional image size |

---

### 2.3 External Dependencies

| Dependency | Impact | Action Required |
|:-----------|:-------|:----------------|
| **CI/CD Platform (GitHub Actions, GitLab CI)** | PyTorch installation increases runner provisioning time (~30s), image size (~500MB), and resource usage | Update workflow YAML files; document minimum runner specs (4GB RAM recommended) |
| **Downstream Consumers (Ptychodus GUI)** | PyTorch becomes mandatory runtime dependency for backend='pytorch' | Update installation documentation; provide clear error messages when unavailable |
| **Documentation** | README, CLAUDE.md, DEVELOPER_GUIDE.md require updates | Phase F1.3 + F4.1 artifacts |

---

## 3. Decision: Approve Torch-Required Transition

### 3.1 Resolution Statement

**APPROVED:** PtychoPINN2 SHALL transition from torch-optional to torch-required execution semantics for the `ptycho_torch/` backend stack, effective immediately upon Phase F1 completion.

**Rationale:**
1. **Production Readiness:** PyTorch backend is no longer experimental; torch-optional patterns were scaffolding for development, not production requirements
2. **Test Validity:** Torch-optional tests validate adapter logic but cannot validate production workflows (skip behavior hides failures)
3. **Architectural Simplicity:** Removing guards reduces code complexity and maintenance burden (estimated 200+ lines of guard boilerplate removable in Phase F3)
4. **Ecosystem Alignment:** PyTorch is standard dependency for modern ML workflows; treating as optional creates support burden

---

### 3.2 Scope Boundaries

**IN SCOPE (torch-required):**
- `ptycho_torch/` production modules (model.py, workflows/components.py, data_container_bridge.py, etc.)
- `tests/torch/` test suite (all tests assume PyTorch installed)
- CI/CD workflows validating Ptychodus integration
- Developer setup documentation

**OUT OF SCOPE (torch-optional preserved):**
- `ptycho/` TensorFlow stack (unaffected by this decision)
- Backend dispatcher fail-fast behavior (already implemented in Attempt #63; remains torch-optional at *selection* time)
- Legacy workflows not using Ptychodus backend

---

## 4. Risk Assessment and Mitigation

### 4.1 Critical Risks

| Risk ID | Description | Probability | Impact | Mitigation Strategy |
|:--------|:------------|:------------|:-------|:--------------------|
| **R1** | CI pipelines fail when PyTorch not installed | **HIGH** | **CRITICAL** | **Phase F3.1:** Update CI configuration before code changes; validate torch installation in pre-test step |
| **R2** | Developer onboarding friction increases | **MEDIUM** | **MEDIUM** | **Phase F4.1:** Clearly document PyTorch as mandatory in README; provide one-line install command |
| **R3** | TEST-PYTORCH-001 starts with torch-optional patterns before Phase F complete | **MEDIUM** | **HIGH** | **Phase F1 gates TEST-PYTORCH-001:** Block activation until governance decision recorded (this document) |
| **R4** | Legacy code depends on torch-optional guards | **LOW** | **LOW** | **Phase F2 inventory:** Comprehensive scan identifies all guard usage before removal |

---

### 4.2 Risk Mitigation Timeline

**Phase F1 (Governance):**
- ✅ Document decision (this artifact)
- ✅ Update CLAUDE.md directive (Phase F1.3)
- ⏳ Communicate to TEST-PYTORCH-001 stakeholders (Phase F4.3)

**Phase F2 (Inventory):**
- Enumerate all `TORCH_AVAILABLE` guards (file:line)
- Audit conftest whitelist dependencies
- Document expected test behavior changes (skip → fail)

**Phase F3 (Implementation):**
- **F3.1 (BLOCKING):** Verify CI environment has PyTorch installed BEFORE removing guards
- **F3.2:** Remove guards from production modules
- **F3.3:** Simplify conftest (remove whitelist, change skip logic)
- **F3.4:** Regression validation with PyTorch-required assumptions

**Phase F4 (Documentation):**
- Update README, CLAUDE.md, DEVELOPER_GUIDE.md
- Sync specs with torch-required policy
- Notify downstream initiatives

---

## 5. Open Questions Resolution

### 5.1 Q1: TensorFlow-Only CI Runners?

**Question (from directive_conflict.md §8):**
> Should TensorFlow-only CI runners continue to exist for legacy validation?

**DECISION:** **YES** (conservative approach)

**Reasoning:**
- Core TensorFlow stack (`ptycho/`) remains production code for non-Ptychodus use cases
- Separate CI jobs allow validating TF-only workflows without PyTorch overhead
- Incremental migration path reduces deployment risk

**Implementation:**
- Keep conftest auto-skip for `tests/torch/` in TF-only runners
- Remove whitelist exceptions (all torch tests skip uniformly)
- PyTorch-required CI runners validate Ptychodus integration
- Document runner matrix in CI configuration

**Artifact Impact:**
- Phase F3.3 conftest changes must preserve directory-based skip logic
- Phase F4.1 documentation must clarify which tests run in which CI jobs

---

### 5.2 Q2: Deprecation Timeline?

**Question:**
> What is the timeline for deprecating torch-optional behavior in existing modules?

**DECISION:** **Aggressive (Phase F3)** with warnings

**Reasoning:**
- All torch-optional guards were scaffolding for Phase B-E development
- No production use case requires torch-optional execution (backend dispatcher handles selection)
- Incremental deprecation creates confusing mixed state

**Implementation:**
- Phase F3.2: Remove all `TORCH_AVAILABLE` guards from `ptycho_torch/` production modules in single loop
- Add deprecation notice in CHANGELOG (breaking change)
- Log warning if old import patterns detected (e.g., `try: import ptycho_torch` in external code)

**Exception:**
- Backend dispatcher (ptycho/workflows/backend_selector.py) retains fail-fast guard (already implemented correctly in Attempt #63)

---

### 5.3 Q3: Environments Without PyTorch?

**Question:**
> How should we handle environments that cannot install PyTorch (e.g., CPU-only ARM systems)?

**DECISION:** **Option A (Fail Fast)** with actionable guidance

**Reasoning:**
- PyTorch supports ARM via pip wheels as of v2.0
- Environments without PyTorch should use TensorFlow backend (default)
- Silent fallback creates support burden (users expect PyTorch when explicitly requested)

**Implementation:**
- Backend dispatcher already implements correct behavior (Attempt #63):
  ```python
  raise RuntimeError(
      f"PyTorch backend selected (backend='{config.backend}') but PyTorch workflows are unavailable.\n"
      f"Error: {e}\n"
      f"Install PyTorch support with: pip install torch torchvision\n"
      f"Or switch to TensorFlow backend (backend='tensorflow')."
  )
  ```
- Phase F4.1 documentation must clarify TensorFlow as recommended fallback for constrained environments
- No silent degradation; users explicitly choose backend via config

---

### 5.4 Q4: Version-Gating?

**Question:**
> Should we version-gate the torch-required policy?

**DECISION:** **YES** (semver compliance)

**Reasoning:**
- Torch-optional → torch-required is breaking change for developers
- Deprecating CLAUDE.md directive changes project contract
- Clear versioning communicates expectations to external consumers

**Implementation:**
- **Version Bump:** Increment to v2.0.0 upon Phase F3 completion (breaking change per semver)
- **CHANGELOG Entry:**
  ```markdown
  ## [2.0.0] - 2025-10-17
  ### BREAKING CHANGES
  - **PyTorch is now required for `ptycho_torch/` backend.** Install with `pip install torch>=2.2`.
  - Removed torch-optional execution guards from `ptycho_torch/` modules.
  - `tests/torch/` test suite now assumes PyTorch availability (tests fail instead of skip when unavailable).
  - Updated CLAUDE.md directive: torch-optional requirement superseded by torch-required policy.

  ### Migration Guide
  - **Developers:** Add `pip install torch>=2.2` to your environment setup.
  - **CI/CD:** Update runners to include PyTorch in test environment (see `docs/TESTING_GUIDE.md`).
  - **Ptychodus Users:** PyTorch is mandatory when `backend='pytorch'`; use `backend='tensorflow'` as fallback.
  ```
- **Git Tag:** Apply `v2.0.0` tag after Phase F4 documentation complete

---

## 6. Success Criteria

### 6.1 Phase F Exit Criteria (from phase_f_torch_mandatory.md)

- [x] **F1.1:** Directive conflict documented (`directive_conflict.md`)
- [x] **F1.2:** Governance decision recorded (this document)
- [ ] **F1.3:** Guidance updates drafted (`guidance_updates.md`)
- [ ] **F2:** Comprehensive inventory complete
- [ ] **F3:** Code updated, regression suite green
- [ ] **F4:** Documentation/specs synced, initiatives notified

---

### 6.2 Validation Checklist (Phase F3.4)

**Test Suite Expectations:**
- [ ] All tests in `tests/torch/` execute (no auto-skips)
- [ ] Tests FAIL (not skip) when PyTorch unavailable in environment
- [ ] Backend dispatcher tests validate fail-fast behavior (`test_backend_selection.py`)
- [ ] Full regression suite passes in PyTorch-required CI runner
- [ ] TensorFlow-only CI runner still passes (torch tests skipped at directory level)

**Code Hygiene:**
- [ ] Zero instances of `TORCH_AVAILABLE` flag in `ptycho_torch/` production modules
- [ ] No `try: import torch` guards in `ptycho_torch/workflows/`, `ptycho_torch/model.py`
- [ ] conftest.py whitelist (`TORCH_OPTIONAL_MODULES`) removed
- [ ] Backend dispatcher fail-fast guard preserved (backward compatibility)

**Documentation:**
- [ ] CLAUDE.md §2 updated with torch-required directive
- [ ] README §Setup includes `pip install torch>=2.2`
- [ ] DEVELOPER_GUIDE.md documents PyTorch as mandatory dependency
- [ ] CHANGELOG includes breaking change notice with migration guide

---

## 7. Communication Plan

### 7.1 Internal Initiatives

| Initiative | Notification Method | Timeline | Key Message |
|:-----------|:-------------------|:---------|:------------|
| **TEST-PYTORCH-001** | Update `plans/pytorch_integration_test_plan.md` header with Phase F1.2 reference | Immediate (Phase F1.3) | "Phase F governance approved torch-required; integration tests may assume PyTorch in CI" |
| **CI/CD Maintainers** | Create `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/ci_migration_guide.md` | Phase F3.1 (before code changes) | "Update runner configuration to install torch>=2.2 before Phase F3.2 merges" |

---

### 7.2 External Stakeholders

**Ptychodus GUI Integration:**
- **Contact:** Project lead / integration team
- **Message:** "PyTorch backend now production-ready; installation required when `backend='pytorch'` selected"
- **Timing:** Phase F4.3 (post-documentation)

**Open Source Contributors:**
- **Venue:** GitHub Releases, CHANGELOG
- **Message:** Version 2.0.0 breaking change notice with migration guide
- **Timing:** Git tag after Phase F4 complete

---

## 8. Rollback Plan

**Scenario:** Phase F3 implementation reveals unexpected blockers (e.g., CI provisioning issues, test infrastructure failures)

**Rollback Steps:**
1. **Revert Code Changes:** Git revert Phase F3.2 commits (guard removal)
2. **Restore Conftest Whitelist:** Reinstate `TORCH_OPTIONAL_MODULES` in tests/conftest.py
3. **Document Blocker:** Create new docs/fix_plan.md item with blocker details
4. **Revisit Governance:** If blocker is policy-level (not technical), reconvene stakeholders

**Decision Authority:**
- Technical blockers (< 1 day effort): Initiative lead resolves via plan update
- Policy blockers (> 1 day effort or external dependency): Reconvene governance review

**Risk Mitigation:**
- Phase F2 inventory reduces rollback probability (comprehensive pre-implementation scan)
- Phase F3.1 CI validation ensures environment ready before code changes

---

## 9. Decision Record

**Approvers:**
- Initiative Lead: INTEGRATE-PYTORCH-001 (Codex Agent)
- Technical Reviewer: Documented consensus via artifact review (this document)
- Governance Authority: Project maintainer (implicit approval via Phase F1 execution)

**Approval Mechanism:**
This document serves as the authoritative record of the torch-required decision. Subsequent Phase F2-F4 work proceeds based on this governance approval. Any challenges to the decision must be raised via docs/fix_plan.md item and reconvene governance review.

**Artifact Status:**
- `directive_conflict.md`: ✅ Complete (Phase F1.1)
- `governance_decision.md`: ✅ Complete (Phase F1.2, this document)
- `guidance_updates.md`: ⏳ Pending (Phase F1.3)

---

## 10. Next Steps

**Immediate (Phase F1.3):**
- Draft guidance updates for CLAUDE.md and docs/findings.md
- Capture exact wording changes for torch-required directive
- Document deprecated torch-optional directive sunset

**Short-Term (Phase F2):**
- Comprehensive inventory of `TORCH_AVAILABLE` guards (file:line anchors)
- Audit conftest whitelist dependencies
- Map expected test behavior changes (skip → fail)

**Medium-Term (Phase F3):**
- CI configuration update (F3.1, BLOCKING)
- Guard removal from production modules (F3.2)
- Conftest simplification (F3.3)
- Regression validation (F3.4)

**Long-Term (Phase F4):**
- Documentation updates (README, CLAUDE.md, DEVELOPER_GUIDE.md)
- Spec synchronization (specs/ptychodus_api_spec.md)
- Initiative handoff notifications (TEST-PYTORCH-001, CI maintainers)

---

**Governance Summary:**
The torch-required transition is APPROVED. PyTorch (torch>=2.2) is now a mandatory dependency for the `ptycho_torch/` backend stack. Phase F2-F4 implementation proceeds under this governance decision. Rollback plan documented for unexpected blockers. Version bump to v2.0.0 upon Phase F completion.

**File References:**
- `directive_conflict.md` (Phase F1.1 context)
- `phase_f_torch_mandatory.md` (Phase F plan)
- `docs/fix_plan.md:57-137` (INTEGRATE-PYTORCH-001 history)
- `specs/ptychodus_api_spec.md` (API contract)
