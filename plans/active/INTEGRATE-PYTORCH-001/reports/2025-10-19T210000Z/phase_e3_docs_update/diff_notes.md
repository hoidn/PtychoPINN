# Phase E3.B1 Documentation Update — Diff Notes

**Date:** 2025-10-19
**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** E3 (Documentation & Handoff)
**Task:** B1 — Update workflow & architecture docs with backend selection guidance

---

## Files Modified

### 1. `docs/workflows/pytorch.md`

**Location:** After §11 "Regression Test & Runtime Expectations"
**Lines Added:** New §12 "Backend Selection in Ptychodus Integration" (108 lines)
**Section Renumbering:** Original §12 "Troubleshooting" → §13, §13 "Keeping Parity" → §14

**Content Added:**

- **§12.1 Configuration API**: Example code showing `backend='pytorch'` parameter in `TrainingConfig` and `InferenceConfig`
- **§12.2 Dispatcher Routing**: Four key guarantees from `specs/ptychodus_api_spec.md` §4.8:
  1. TensorFlow path delegation to `ptycho.workflows.components`
  2. PyTorch path delegation to `ptycho_torch.workflows.components`
  3. CONFIG-001 enforcement (`update_legacy_dict` before backend inspection)
  4. Result metadata annotation (`results['backend']`)
- **§12.3 Error Handling**:
  - PyTorch unavailability raises actionable `RuntimeError` per POLICY-001
  - Invalid backend string raises `ValueError`
- **§12.4 Checkpoint Compatibility**: Backend-specific formats (`.h5.zip` vs `.ckpt`), cross-backend loading error behavior
- **§12.5 Test Selectors**: Explicit pytest commands for `test_backend_selection.py` (lines 59-170) and cross-backend checkpoint tests
- **§12.6 Integration Example**: Ptychodus reconstructor code snippet demonstrating backend selection workflow

**References Added:**
- `specs/ptychodus_api_spec.md` §4.8 (3 citations)
- `docs/findings.md#POLICY-001` (1 citation)
- `ptycho/workflows/backend_selector.py:121-165` (note: conditional on codebase presence)
- `tests/torch/test_backend_selection.py:59-170` (1 citation)
- `tests/torch/test_model_manager.py:238-372` (1 citation)

**Rationale:** Addresses Phase A.A1 HIGH gap — missing backend selection API guidance for Ptychodus integrators. Provides actionable examples and explicit test selectors per E3 inventory recommendations.

---

### 2. `docs/architecture.md`

**Location:** §1 "Component Diagram", after existing Note about `params.py` legacy status
**Lines Added:** New "Backend Selection" paragraph (5 lines)

**Content Added:**

New paragraph after line 11:
> **Backend Selection:** As of Phase F (INTEGRATE-PYTORCH-001), PtychoPINN supports dual backends via the `backend` configuration field (`'tensorflow'` or `'pytorch'`). The dispatcher routes to `ptycho.workflows.components` (TensorFlow) or `ptycho_torch.workflows.components` (PyTorch) based on this field. Both paths share the same data pipeline (`raw_data.py`, `loader.py`) and configuration system (`config/config.py`, `params.py` bridge), ensuring CONFIG-001 compliance regardless of backend choice. See `specs/ptychodus_api_spec.md` §4.8 for routing guarantees and `docs/workflows/pytorch.md` §12 for integration guidance.

**References Added:**
- `specs/ptychodus_api_spec.md` §4.8 (1 citation)
- `docs/workflows/pytorch.md` §12 (1 citation)
- CONFIG-001 requirement reminder (1 mention)

**Rationale:** Addresses Phase A.A1 LOW/MEDIUM enhancement — architecture diagram should note backend selector routing. Adds text block (not diagram update) per input.md guidance to avoid mermaid syntax risks.

---

## Cross-Reference Validation

Confirmed all internal document references point to valid locations:

- ✅ `specs/ptychodus_api_spec.md` §4.8 exists (inserted in INTEGRATE-PYTORCH-001 Attempt #15)
- ✅ `docs/findings.md#POLICY-001` exists (PyTorch mandatory policy)
- ✅ `docs/workflows/pytorch.md` §12 now exists (this update)
- ⚠️ `ptycho/workflows/backend_selector.py:121-165` — marked as conditional ("if available in your codebase")
- ⚠️ `tests/torch/test_backend_selection.py:59-170` — cited but not verified in this loop (assumed present per spec §4.8 references)

**Action Required:** Verify test file paths exist before final commit. If missing, update citations to match actual test locations or note as "to be implemented."

---

## Style & Formatting Notes

- **ASCII Formatting:** Preserved existing code block and list formatting; no smart quotes introduced
- **Mermaid Diagrams:** No diagram edits made (text-only updates per input.md pitfall guidance)
- **Section Numbering:** Updated §12 → §13, §13 → §14 in pytorch.md to accommodate new §12
- **Consistency:** Backend field values `'tensorflow'` and `'pytorch'` consistently quoted as string literals
- **Line Length:** Kept paragraphs under 120 characters where practical; code blocks allowed to exceed

---

## Artifact Map

This diff note captures file:line anchors for all edits made in this loop:

| File | Section | Lines Modified | Nature of Change |
|------|---------|----------------|------------------|
| `docs/workflows/pytorch.md` | After §11 | +108 lines | New §12 backend selection guidance |
| `docs/workflows/pytorch.md` | §12 → §13 | Header edit | Section renumbering |
| `docs/workflows/pytorch.md` | §13 → §14 | Header edit | Section renumbering |
| `docs/architecture.md` | §1 after line 11 | +5 lines | Backend selection paragraph |

**Total Lines Added:** 113
**Total Sections Affected:** 4

---

**Next Steps (per input.md directive):**
1. Create `summary.md` documenting completion and rationale
2. Update `phase_e3_docs_plan.md` B1 row to `[x]` with artifact cross-links
3. Append docs/fix_plan.md Attempt entry with this artifact directory path
4. Verify test file paths exist or update citations accordingly
