# Phase E3 — Documentation Gap Inventory Summary

**Date:** 2025-10-19
**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** E3.A (Gap Assessment & Scope Confirmation)
**Tasks Completed:** A.A1, A.A2, A.A3
**Mode:** Docs-only inventory loop (no file edits)
**Attempt:** docs/fix_plan.md [INTEGRATE-PYTORCH-001-STUBS] Attempt #13

---

## Executive Summary

Completed comprehensive documentation gap inventory for Phase E3 (Backend Selection & Handoff). Surveyed developer docs, specs, and TEST-PYTORCH-001 coordination requirements to identify deltas between current PyTorch backend capabilities (end-to-end operational as of Phase D2) and documentation/spec guidance.

**Key Finding:** PyTorch backend is production-ready but lacks **backend selection API specification** (§4.8 in ptychodus_api_spec.md), which is **BLOCKING** Phase E completion.

---

## Tasks Completed

### ✅ A.A1 — Catalogue Documentation Gaps

**Files Surveyed:**
- `docs/workflows/pytorch.md` (343 lines)
- `README.md` (84 lines)
- `docs/architecture.md` (161 lines)
- `CLAUDE.md` (127 lines)

**Method:**
- `rg -n "backend" docs/workflows/pytorch.md README.md docs/architecture.md`
- `rg -n "NotImplementedError" docs/workflows/pytorch.md docs/architecture.md`
- `rg -n "TensorFlow" docs/workflows/pytorch.md README.md docs/architecture.md`
- Manual read of Phase D2 updates (Attempt #13, Attempt #41)

**Findings:**
1. **`docs/workflows/pytorch.md`:** ✅ Largely accurate (no NotImplementedError warnings, Phase D2 updates complete), but **missing backend selection guidance** (how Ptychodus chooses TensorFlow vs PyTorch)
2. **`README.md`:** ✅ POLICY-001 compliant, but no mention of dual-backend architecture
3. **`docs/architecture.md`:** Component diagram shows only TensorFlow stack, lacks PyTorch parallel structure note
4. **`CLAUDE.md`:** ✅ POLICY-001 documented, but missing CONFIG-001 reminder for PyTorch workflows

**Severity:** **1 HIGH** (missing backend selection API in pytorch.md §14), 3 LOW/MEDIUM enhancements

**Evidence:** See `phase_e3_docs_inventory.md` §A.A1 (lines 15-139)

---

### ✅ A.A2 — Audit Spec & Findings Coverage

**Files Surveyed:**
- `specs/ptychodus_api_spec.md` (311 lines)
- `docs/findings.md` (19 lines, 2 active policies)

**Method:**
- `rg -n "backend|PyTorch|TensorFlow" specs/ptychodus_api_spec.md` (captured 30 lines)
- Manual review of §1 (Overview), §2.2 (PyTorch Config Adapters), §4 (Reconstructor Contract)
- Cross-check POLICY-001 and FORMAT-001 coverage

**Findings:**
1. **`specs/ptychodus_api_spec.md`:**
   - ✅ §1: POLICY-001 banner present (line 14)
   - ✅ §2.2: PyTorch config adapters fully documented (lines 75-81)
   - ❌ **§4.8 MISSING:** No backend selection/dispatch behavior specification
     - Gap: How does `PtychoPINNTrainableReconstructor` choose backend?
     - Gap: Static selection at instantiation vs dynamic fallback?
     - Gap: Fail-fast behavior when PyTorch unavailable?
     - Gap: Cross-backend checkpoint compatibility?
   - **SEVERITY: HIGH BLOCKING** — Ptychodus integration requires this contract

2. **`docs/findings.md`:**
   - ✅ POLICY-001 (PyTorch mandatory) documented
   - ✅ FORMAT-001 (NPZ auto-transpose) documented
   - 🔄 POLICY-002 placeholder recommended for backend selection (after §4.8 implementation)

**Recommendation:** Author §4.8 spec amendment as **design review artifact** before implementation.

**Evidence:** See `phase_e3_docs_inventory.md` §A.A2 (lines 141-203)

---

### ✅ A.A3 — Map Downstream Handoff Needs

**Files Surveyed:**
- `plans/active/TEST-PYTORCH-001/implementation.md` (82 lines)
- `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md`
- `plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/baseline/summary.md`

**Method:**
- Read TEST-PYTORCH-001 Phase D status (A/C/D1/D2 complete, D3 pending)
- Identify coordination gaps between INTEGRATE-PYTORCH-001 and TEST-PYTORCH-001
- Review runtime profile + baseline for CI requirements

**Findings:**

**4 Handoff Items Required:**

1. **CI Execution Guidance (BLOCKING Phase D3):**
   - Pytest markers (`@pytest.mark.integration`, `@pytest.mark.slow`)
   - Skip policy for TensorFlow-only CI runners
   - Timeout (120s recommended, 3.3× 35.92s baseline)
   - Retry policy (1 retry on timeout)
   - ✅ Environment requirements already documented (runtime_profile.md)

2. **Parity Validation Selectors:**
   - 6 critical selectors identified (config_bridge, Lightning, stitching, checkpoint, decoder)
   - Run frequency: After changes to `ptycho_torch/model.py`, `workflows/components.py`, `config_bridge.py`
   - TensorFlow baseline selector for comparison

3. **Dataset & Fixture Coordination:**
   - Current: Canonical dataset (35MB, 1087 patterns, 35.92s runtime)
   - Decision: Approved per TEST-PYTORCH-001 baseline assessment
   - Alternative: Phase B minimal fixture (deferred, <5s runtime) if needed

4. **Ownership Matrix:**
   - `ptycho_torch/` code: INTEGRATE-PYTORCH-001 alumni
   - Integration tests: TEST-PYTORCH-001 alumni
   - CI config: DevOps / Project Lead
   - Documentation: INTEGRATE-PYTORCH-001 alumni
   - Escalation path defined

**Recommendation:** Create `phase_e3_handoff.md` consolidating all 4 items for TEST-PYTORCH-001 Phase D3.

**Evidence:** See `phase_e3_docs_inventory.md` §A.A3 (lines 205-322)

---

## Checklist Status

Phase E3.A tasks from `phase_e3_docs_plan.md`:

| ID | Task | State | Notes |
|----|------|-------|-------|
| A1 | Catalogue documentation gaps | **[x]** | 4 files surveyed, 1 HIGH + 3 LOW/MEDIUM gaps identified |
| A2 | Audit spec & findings coverage | **[x]** | §4.8 backend selection spec MISSING (BLOCKING) |
| A3 | Map downstream handoff needs | **[x]** | 4 handoff items documented for TEST-PYTORCH-001 D3 |

**Exit Criteria Satisfied:**
- ✅ Inventory document capturing deltas (`phase_e3_docs_inventory.md`, 322 lines)
- ✅ Target file anchors documented (e.g., pytorch.md:334, ptychodus_api_spec.md:222)
- ✅ Ownership recommendations provided (handoff matrix in A3)

---

## Artifacts Generated

1. **`phase_e3_docs_inventory.md`** (322 lines) — Comprehensive gap analysis
2. **`summary.md`** (this file) — Loop-level executive summary

**Artifact Location:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/`

---

## Key Recommendations

### Immediate (Phase E3.B)
1. **HIGH:** Add `docs/workflows/pytorch.md` §14 "Backend Selection in Ptychodus Integration"
2. **HIGH BLOCKING:** Draft `specs/ptychodus_api_spec.md` §4.8 "Backend Selection and Dispatch" for design review
3. **MEDIUM:** Add CLAUDE.md CONFIG-001 PyTorch reminder (§4.1)

### Deferred (Phase E3.D or later)
1. Create `phase_e3_handoff.md` for TEST-PYTORCH-001 Phase D3
2. Update README.md with dual-backend architecture mention (nice-to-have)
3. Add architecture.md PyTorch backend note (low priority)
4. Add findings.md POLICY-002 after §4.8 design (medium priority)

---

## Next Actions

**For Phase E3.B (Documentation Updates):**
- Implement A1.1 (pytorch.md §14) — straightforward doc addition once §4.8 design settled
- **BLOCKER:** Design review required for §4.8 (static vs dynamic backend selection)

**For Phase E3.D (Handoff Package):**
- Author `phase_e3_handoff.md` consolidating CI/parity/fixture/ownership guidance
- Update TEST-PYTORCH-001 plan Phase D3 row with handoff link

---

## References

**Evidence Inputs:**
- Phase D2 parity: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T201500Z/phase_d2_completion/parity_update.md`
- TEST-PYTORCH-001 baseline: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/baseline/summary.md`
- Runtime profile: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md`

**Plan Documents:**
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_plan.md`
