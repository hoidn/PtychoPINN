# Phase F4.1 Documentation Updates — Torch-Required Policy Migration

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** F4.1 — Developer-Facing Documentation Updates
**Date:** 2025-10-17
**Purpose:** Migrate all developer-facing documentation from torch-optional to torch-required baseline per Phase F governance approval (recorded in `reports/2025-10-17T184624Z/governance_decision.md`).

---

## Inventory Summary (Task F4.1.A)

### Search Results: `rg "torch-optional" docs/ README.md`

**Finding:** Zero matches in active developer-facing documents. All occurrences of "torch-optional" are in historical records (`docs/fix_plan.md` attempt histories).

**Broader Pattern Search:** `rg -i "pytorch.*optional|torch.*optional"`
- `CLAUDE.md:57` — Critical directive requiring torch-optional parity tests (BLOCKER)
- `docs/workflows/pytorch.md:4` — Optional MLflow reference (non-blocking)
- `docs/fix_plan.md` — Multiple historical references (no action required)

**Additional Searches:**
- `README.md` — No PyTorch prerequisite documentation; installation section generic
- `docs/workflows/pytorch.md:17-22` — Prerequisites section lacks explicit torch-required framing

---

## Documents Requiring Updates

### 1. CLAUDE.md (CRITICAL)

**Location:** `/CLAUDE.md:57-59`
**Current Text (§2, torch-optional directive):**
```markdown
<directive level="critical" purpose="Keep PyTorch parity tests torch-optional">
  Parity and adapter tests must remain runnable when PyTorch is unavailable. Use documented shims or skip rules in `tests/conftest.py`, avoid hard `import torch` statements in modules that only need type aliases, and capture any fallback behavior in the loop artifacts.
</directive>
```

**Required Change:**
- Remove entire directive block (lines 57-59)
- Replace with torch-required guidance referencing Phase F migration and governance decision
- Update directive to reflect new skip behavior (directory-based skipping in CI only)

**Rationale:**
- Conflicts with Phase F1 governance approval (documented in `reports/2025-10-17T184624Z/governance_decision.md`)
- Phase F3.1 added `torch>=2.2` to `setup.py` install_requires
- Phase F3.2-F3.3 removed all `TORCH_AVAILABLE` guards and whitelist mechanisms
- New behavior: all `tests/torch/` tests require PyTorch installed; skip in CI via conftest directory check

---

### 2. docs/workflows/pytorch.md (MODERATE)

**Location:** `/docs/workflows/pytorch.md:17-22` (Prerequisites section)

**Current Text:**
```markdown
## 2. Prerequisites

- PyTorch >= 2.2 with CUDA support (if training on GPU).
- `lightning`, `mlflow`, and `tensordict` installed.
- Input NPZ files generated according to the project data contract.
- Optional: running MLflow tracking server (set `MLFLOW_TRACKING_URI` if using a remote instance).
```

**Required Changes:**
- Emphasize PyTorch as **required** (not optional)
- Cross-reference `setup.py` dependency specification
- Add installation command snippet
- Clarify "optional" applies only to MLflow tracking server, not PyTorch itself

**Rationale:**
- Prerequisites section ambiguous about PyTorch status
- Should match governance decision and `setup.py` specification
- Readers may interpret PyTorch as optional given current phrasing

---

### 3. README.md (MINOR)

**Location:** `/README.md:23-28` (Installation section)

**Current Text:**
```markdown
## Installation
`conda create -n ptycho python=3.10`

`conda activate ptycho`

`pip install .`
```

**Required Changes:**
- Add note that PyTorch is now a required dependency (automatically installed)
- Optionally: document manual PyTorch installation for GPU/CUDA customization

**Rationale:**
- Users should know PyTorch will be installed by `pip install .`
- GPU users may want to pre-install specific CUDA versions
- Transparency about large dependency additions

---

## Cross-References and Validation

### Documents NOT Requiring Updates

- `docs/findings.md` — Will be updated in Phase F4.2.B with new POLICY-001 finding
- `specs/ptychodus_api_spec.md` — Spec sync addressed in Phase F4.2.A
- `docs/TESTING_GUIDE.md` — No torch-optional references found
- `docs/DEVELOPER_GUIDE.md` — No torch-optional guidance

### Historical Records (No Action Required)

- `docs/fix_plan.md` — Historical attempt logs preserve torch-optional context for audit trail
- `plans/active/INTEGRATE-PYTORCH-001/reports/*` — Phase-specific artifacts document transition

---

## Next Steps

1. **F4.1.B** — Apply CLAUDE.md directive update with diff rationale
2. **F4.1.C** — Refresh pytorch.md prerequisites and README installation notes
3. **F4.1 Exit** — Capture all diffs and anchors in this document's "Edits Applied" section below

---

## Edits Applied

### 1. CLAUDE.md Update ✅

**File:** `/CLAUDE.md`
**Lines Modified:** 57-59 (3 lines removed, 3 lines added)
**Diff Summary:**
- **Removed:** Old directive "Keep PyTorch parity tests torch-optional" (3 lines)
  - Referenced torch-optional shims, conftest.py skip rules, fallback behavior
- **Added:** New directive "Enforce PyTorch Requirement" (3 lines)
  - Documents PyTorch as required dependency (`torch>=2.2` in setup.py)
  - References Phase F governance decision and migration artifacts
  - Explains directory-based pytest skipping in CI vs local development behavior

**Key Changes:**
- Directive level: `critical` (unchanged)
- Purpose: Changed from "Keep...torch-optional" → "Enforce PyTorch Requirement"
- Content: Complete rewrite referencing Phase F1 governance decision, Phase F3 implementation, and artifact locations

**Anchor:** `<directive level="critical" purpose="Enforce PyTorch Requirement">` (line 57)

**Rationale:** Directive directly conflicted with Phase F governance approval removing torch-optional execution paths. New text aligns with completed Phase F3 migration removing all `TORCH_AVAILABLE` guards and enforcing unconditional torch imports.

---

### 2. docs/workflows/pytorch.md Update ✅

**File:** `/docs/workflows/pytorch.md`
**Lines Modified:** 17-22 (5 lines replaced with 5 lines, expanded content)
**Diff Summary:**
- **Enhanced:** Prerequisites section with explicit torch-required framing
- **Added:** Bold emphasis on "PyTorch >= 2.2 (REQUIRED)"
- **Added:** Automatic installation note via setup.py
- **Added:** Link to PyTorch installation guide for GPU/CUDA customization
- **Added:** Cross-reference to specs/data_contracts.md for NPZ files
- **Clarified:** "Optional" now clearly applies only to MLflow tracking server, not PyTorch

**Key Changes:**
- Line 19: Added bold "**PyTorch >= 2.2 (REQUIRED)**" with automatic installation explanation
- Line 20: Noted lightning/mlflow/tensordict as automatic dependencies
- Line 21: Added doc-ref tag for data contracts
- Line 22: Clarified optional MLflow tracking server with environment variable note

**Anchor:** `## 2. Prerequisites` (line 17)

**Rationale:** Prerequisites section was ambiguous about PyTorch's required status. New text matches governance decision and setup.py specification while providing actionable installation guidance.

---

### 3. README.md Update ✅

**File:** `/README.md`
**Lines Modified:** 28-30 (added 3 new lines after installation commands)
**Diff Summary:**
- **Added:** Note explaining PyTorch >= 2.2 automatic installation
- **Added:** GPU/CUDA customization guidance with link to official PyTorch installation guide

**Key Changes:**
- Line 30: Bold "**Note:**" explaining PyTorch is required and auto-installed
- Line 30 (cont): Link to https://pytorch.org/get-started/locally/ for manual PyTorch install
- Guidance for users needing specific CUDA versions to pre-install PyTorch

**Anchor:** `## Installation` (line 23)

**Rationale:** Users should be transparently informed that `pip install .` now includes large PyTorch dependency. GPU users benefit from guidance on manual pre-installation for CUDA version control.

---

## Verification

### Files Changed
1. `/CLAUDE.md` — 3 lines modified (directive rewrite)
2. `/docs/workflows/pytorch.md` — 5 lines enhanced (prerequisites expanded)
3. `/README.md` — 3 lines added (installation note)

**Total:** 3 files, 11 lines changed

### Consistency Checks
- ✅ All references point to `torch>=2.2` (matches setup.py specification)
- ✅ PyTorch framed as "required" consistently across all 3 documents
- ✅ Phase F governance artifacts referenced in CLAUDE.md
- ✅ No lingering torch-optional language in updated sections
- ✅ Cross-references use proper doc-ref XML tags

### No Changes Required
- `docs/findings.md` — POLICY-001 finding will be added in Phase F4.2.B
- `specs/ptychodus_api_spec.md` — Spec sync in Phase F4.2.A
- `docs/fix_plan.md` — Historical records preserved for audit trail

---

*Document Status: Complete*
*Last Updated: 2025-10-17*
*Phase: F4.1.A+B+C Complete*
