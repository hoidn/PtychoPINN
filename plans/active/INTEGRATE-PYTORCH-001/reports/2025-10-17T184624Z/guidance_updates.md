# Phase F1.3 — Guidance Updates: Torch-Required Directive Wording

## Executive Summary

This document provides the exact redline edits required to update authoritative guidance documents (CLAUDE.md, docs/findings.md) to reflect the torch-required policy approved in Phase F1.2.

**Date:** 2025-10-17
**Initiative:** INTEGRATE-PYTORCH-001 Phase F
**Governance Reference:** `governance_decision.md` (Phase F1.2)
**Status:** DRAFT (pending supervisor confirmation before applying)

---

## 1. CLAUDE.md Updates

### 1.1 Section 2 — Core Project Directives

**File:** `CLAUDE.md:57-59`
**Action:** REPLACE existing torch-optional directive with torch-required directive

#### OLD TEXT (LINES 57-59):
```xml
<directive level="critical" purpose="Keep PyTorch parity tests torch-optional">
  Parity and adapter tests must remain runnable when PyTorch is unavailable. Use documented shims or skip rules in `tests/conftest.py`, avoid hard `import torch` statements in modules that only need type aliases, and capture any fallback behavior in the loop artifacts.
</directive>
```

#### NEW TEXT (REPLACEMENT):
```xml
<directive level="critical" purpose="PyTorch is required for Ptychodus backend">
  PyTorch (torch>=2.2) is a **mandatory dependency** for the `ptycho_torch/` backend stack and Ptychodus integration workflows. Production modules in `ptycho_torch/` MUST use unconditional `import torch` statements. Test infrastructure in `tests/torch/` MUST assume PyTorch availability and FAIL (not skip) when PyTorch is unavailable. CI/CD pipelines validating Ptychodus integration MUST install PyTorch.

  **Rationale:** PyTorch backend is production-ready as of INTEGRATE-PYTORCH-001 Phase E completion. Torch-optional patterns were development scaffolding and have been retired. See `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md` for policy context.

  **TensorFlow Stack:** The legacy `ptycho/` TensorFlow stack remains the default backend and does NOT require PyTorch. Backend selection is controlled via `config.backend` field (see `ptycho/workflows/backend_selector.py`).
</directive>
```

**Justification:**
- Explicit version requirement (torch>=2.2) per governance decision §5.2
- Clear scope boundary (ptycho_torch/ vs ptycho/)
- References governance artifact for policy context
- Documents backend selection mechanism

---

### 1.2 Section 5 — Key Commands

**File:** `CLAUDE.md:110-117`
**Action:** ADD PyTorch verification commands

#### INSERT AFTER LINE 116 (after "# Run a quick, small training job..."):
```bash
# Verify PyTorch installation (required for ptycho_torch backend)
python3 -c "import torch; print(f'PyTorch {torch.__version__} available')"

# Test PyTorch backend selection
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_groups 512 --backend pytorch --output_dir verification_torch_run
```

**Justification:**
- Provides immediate verification command for developers
- Documents backend selection CLI flag
- Parallel structure to existing TensorFlow verification

---

### 1.3 New Section — PyTorch Backend Requirements (Optional Enhancement)

**File:** `CLAUDE.md`
**Action:** ADD new subsection after §5 (Key Commands), before §6 (if exists) or at end

#### NEW SECTION:
```markdown
## 6. PyTorch Backend Requirements

### Installation

The PyTorch backend (`ptycho_torch/`) is required for Ptychodus GUI integration and optional for CLI workflows.

```bash
# Install PyTorch (CPU-only)
pip install torch>=2.2 torchvision

# Install PyTorch (CUDA support, Linux/Windows)
pip install torch>=2.2 torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python3 -c "import torch; print(f'PyTorch {torch.__version__} available')"
```

### Backend Selection

Choose backend via CLI flag or configuration file:

```bash
# Explicit backend selection (CLI)
ptycho_train --train_data_file <path> --backend pytorch

# Configuration file (YAML)
backend: pytorch
train_data_file: <path>
...
```

**Default Behavior:** TensorFlow backend used when `backend` not specified (backward compatibility).

### Troubleshooting

**Error: "PyTorch backend selected but PyTorch workflows are unavailable"**
- **Cause:** PyTorch not installed or import failed
- **Solution:** Run `pip install torch>=2.2 torchvision` or switch to `backend: tensorflow`

**Error: "No module named 'torch'"**
- **Cause:** PyTorch not in environment
- **Solution:** Verify with `python3 -c "import torch"` and reinstall if needed

### CI/CD Configuration

CI runners validating Ptychodus integration MUST install PyTorch:

```yaml
# GitHub Actions example
- name: Install PyTorch
  run: pip install torch>=2.2 torchvision --index-url https://download.pytorch.org/whl/cpu
```

TensorFlow-only CI runners may skip `tests/torch/` via directory-level auto-skip (see `tests/conftest.py`).
```

**Justification:**
- Centralizes PyTorch setup guidance
- Documents backend selection for new users
- Provides CI configuration examples
- Clarifies TensorFlow vs PyTorch scope

---

## 2. docs/findings.md Updates

### 2.1 New Finding — Torch-Required Policy

**File:** `docs/findings.md`
**Action:** ADD new finding entry at end of table

#### NEW ENTRY (after last row):
```markdown
| POLICY-001 | 2025-10-17 | pytorch, torch-required, backend, policy | PyTorch (torch>=2.2) transitioned from optional to mandatory dependency for `ptycho_torch/` backend stack; torch-optional directive (CLAUDE.md:57-59) superseded; production modules use unconditional imports; tests fail (not skip) when unavailable. | [Link](plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md) | Active |
```

**Table Fields:**
- **Finding ID:** POLICY-001 (new category for governance decisions)
- **Date:** 2025-10-17 (governance approval date)
- **Keywords:** pytorch, torch-required, backend, policy
- **Synopsis:** Concise summary of policy shift
- **Evidence Pointer:** Governance decision artifact
- **Status:** Active (current policy)

---

### 2.2 Update Existing Finding — CONFIG-001

**File:** `docs/findings.md` (row with Finding ID `CONFIG-001`)
**Action:** UPDATE synopsis to reference torch-required policy

#### CURRENT TEXT (CONFIG-001 Synopsis):
```
`update_legacy_dict(params.cfg, config)` must run before any legacy module executes; missing this broke gridsize sync and legacy interop.
```

#### NEW TEXT (REPLACEMENT):
```
`update_legacy_dict(params.cfg, config)` must run before any legacy module executes; missing this broke gridsize sync and legacy interop. PyTorch backend enforces this via `ptycho/workflows/backend_selector.py` entry points (torch-required policy).
```

**Justification:**
- Links CONFIG-001 finding to torch-required implementation
- Documents backend dispatcher as enforcement mechanism
- Preserves original finding while adding context

---

## 3. File-Level Summary

### 3.1 Files Requiring Updates

| File | Lines Affected | Action Type | Priority |
|:-----|:--------------|:------------|:---------|
| `CLAUDE.md` | 57-59 | REPLACE directive | **CRITICAL** (blocks Phase F2) |
| `CLAUDE.md` | After 116 | INSERT verification commands | **HIGH** (developer UX) |
| `CLAUDE.md` | End of file | INSERT new section (optional) | **MEDIUM** (enhancement) |
| `docs/findings.md` | Last row | INSERT POLICY-001 entry | **HIGH** (traceability) |
| `docs/findings.md` | CONFIG-001 row | UPDATE synopsis | **MEDIUM** (context linkage) |

---

### 3.2 Edit Verification Checklist

**Before Applying Edits:**
- [ ] Confirm governance_decision.md approved (Phase F1.2 complete)
- [ ] Verify no concurrent edits to CLAUDE.md/findings.md (check git status)
- [ ] Review redlines for typos / XML syntax errors

**After Applying Edits:**
- [ ] Run XML syntax validator on CLAUDE.md directives
- [ ] Verify all file:line references point to correct artifacts
- [ ] Check hyperlinks in findings.md Evidence Pointer column
- [ ] Run `rg "torch-optional" CLAUDE.md` (expect zero results)
- [ ] Run `rg "POLICY-001" docs/findings.md` (expect 1 result)

---

## 4. Implementation Sequence (for Next Loop)

**Recommended Order:**
1. **Step 1:** Apply CLAUDE.md:57-59 REPLACE (critical directive update)
2. **Step 2:** Apply docs/findings.md POLICY-001 INSERT (governance record)
3. **Step 3:** Apply CLAUDE.md:116 INSERT (verification commands)
4. **Step 4:** Apply docs/findings.md CONFIG-001 UPDATE (context linkage)
5. **Step 5 (Optional):** Apply CLAUDE.md new section (enhancement)

**Validation Between Steps:**
- After Step 1: Run `rg "torch-optional" CLAUDE.md` → expect zero results
- After Step 2: Run `rg "POLICY-001" docs/findings.md` → expect 1 result
- After Step 4: Run full test suite to ensure no doc-driven test failures

---

## 5. Rollback Instructions

**Scenario:** Edits applied but Phase F2 reveals blocking issues requiring policy reversal

**Rollback Steps:**
1. **Revert CLAUDE.md:** `git checkout HEAD -- CLAUDE.md`
2. **Revert docs/findings.md:** `git checkout HEAD -- docs/findings.md`
3. **Document Blocker:** Create docs/fix_plan.md item with rollback reason
4. **Archive Artifacts:** Move Phase F1 artifacts to `plans/archive/INTEGRATE-PYTORCH-001/phase_f_rollback_<timestamp>/`

**Decision Authority:**
- Initiative lead may rollback if Phase F2 inventory reveals > 5 days effort to complete (per governance_decision.md §8)
- Rollback must be documented in docs/fix_plan.md Attempts History with rationale

---

## 6. Cross-Reference Table

| Artifact | Purpose | References |
|:---------|:--------|:-----------|
| `directive_conflict.md` | Documents torch-optional vs torch-required conflict | CLAUDE.md:57-59, tests/conftest.py:42, docs/fix_plan.md:80-137 |
| `governance_decision.md` | Records approval to transition to torch-required | directive_conflict.md, specs/ptychodus_api_spec.md |
| `guidance_updates.md` (this doc) | Provides exact redline edits for CLAUDE.md/findings.md | governance_decision.md, CLAUDE.md, docs/findings.md |

---

## 7. Open Questions (for Next Loop)

**Q1:** Should we add a deprecation notice in the old torch-optional directive before removing it?
- **Option A:** Replace directive immediately (per this document)
- **Option B:** Add deprecation notice, keep old directive for 1 loop, then remove
- **Recommendation:** Option A (aggressive, aligns with governance_decision.md §5.2)

**Q2:** Should new PyTorch Backend Requirements section (§1.3) be in CLAUDE.md or separate doc?
- **Option A:** In CLAUDE.md (keeps setup instructions centralized)
- **Option B:** Separate `docs/PYTORCH_BACKEND_GUIDE.md` (avoids CLAUDE.md bloat)
- **Recommendation:** Option A for Phase F1, refactor to separate doc in Phase F4 if CLAUDE.md exceeds 200 lines

**Q3:** Do we need a BREAKING CHANGES section in CLAUDE.md?
- **Consideration:** v2.0.0 release includes breaking change (torch-required policy)
- **Proposal:** Add `## Breaking Changes in v2.0.0` section after §6 documenting:
  - Torch-required policy
  - Removed torch-optional guards
  - Updated conftest behavior
  - Migration guide for developers
- **Decision:** Defer to Phase F4.1 (documentation phase) to avoid premature CLAUDE.md changes

---

## 8. Artifact Handoff

**Next Loop (Phase F2 Execution):**
- **Actor:** Ralph (engineer agent)
- **Instructions:** Apply edits from §1-§2 in sequence per §4
- **Validation:** Run verification checklist after each edit
- **Output:** Updated CLAUDE.md and docs/findings.md with torch-required policy

**Blocking Dependencies:**
- Phase F1.2 governance_decision.md MUST be complete (approved) before applying edits
- No concurrent work on CLAUDE.md/findings.md (check git status for uncommitted changes)

---

## 9. Summary

**Phase F1.3 Deliverable:**
This document provides exact redline edits to transition CLAUDE.md and docs/findings.md from torch-optional to torch-required policy. Edits are scoped, sequenced, and validated per governance approval. Next loop applies edits and proceeds to Phase F2 inventory.

**Key Changes:**
1. CLAUDE.md:57-59 — REPLACE torch-optional directive with torch-required directive
2. CLAUDE.md:116 — INSERT PyTorch verification commands
3. CLAUDE.md (optional) — INSERT new PyTorch Backend Requirements section
4. docs/findings.md — INSERT POLICY-001 finding entry
5. docs/findings.md — UPDATE CONFIG-001 synopsis with torch-required context

**File References:**
- CLAUDE.md:57-59, 110-117 (edit targets)
- docs/findings.md (POLICY-001, CONFIG-001 entries)
- governance_decision.md (approval authority)
- directive_conflict.md (policy context)

**Status:** DRAFT redlines ready for next loop execution
