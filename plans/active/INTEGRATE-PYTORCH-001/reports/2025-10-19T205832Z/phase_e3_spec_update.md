# Phase E3.C1/C2 — Backend Selection Spec Update (INTEGRATE-PYTORCH-001)

**Date:** 2025-10-19
**Attempt:** #15 (docs/fix_plan.md)
**Mode:** Docs
**Tasks Completed:** C1 (spec update), C2 (findings review)

## Summary

Successfully updated `specs/ptychodus_api_spec.md` with §4.8 "Backend Selection & Dispatch" specification based on approved draft from `phase_e3_spec_patch.md`. Reviewed `docs/findings.md` for policy alignment and determined no new POLICY-002 entry required.

## Task C1: Update specs/ptychodus_api_spec.md

### Edit Applied
- **File:** `specs/ptychodus_api_spec.md`
- **Location:** Inserted new §4.8 after §4.7 (lines 224-235)
- **Change Type:** New section addition
- **Content Source:** `phase_e3_spec_patch.md` §Proposed Specification Additions

### Spec Changes Summary
Added comprehensive backend selection requirements covering:
1. **Configuration Field** — `backend: Literal['tensorflow', 'pytorch']` with `'tensorflow'` default
2. **CONFIG-001 Compliance** — Mandatory `update_legacy_dict()` call before backend inspection
3. **Routing Guarantees** — TensorFlow vs PyTorch workflow delegation per backend literal
4. **Torch Unavailability** — Actionable `RuntimeError` when PyTorch missing (POLICY-001 enforcement)
5. **Result Metadata** — `results['backend']` annotation requirement
6. **Persistence Parity** — Backend-specific archive formats with clear cross-backend error messaging
7. **Validation Errors** — `ValueError` for unsupported backend literals
8. **Inference Symmetry** — `load_inference_bundle_with_backend()` mirrors training guarantees

### References Preserved
- Inline code references: `ptycho/workflows/backend_selector.py`, `tests/torch/test_backend_selection.py:59-170`, `tests/torch/test_model_manager.py:238-372`
- Policy citations: POLICY-001 (PyTorch fail-fast)
- Config conventions: CONFIG-001 (params.cfg sync)

### Verification
- Table of contents numbering stable (§5 Configuration Field Reference now follows §4.8)
- Markdown formatting consistent with existing sections
- ASCII characters only (no formatting issues)
- Cross-references use inline code formatting

## Task C2: Review docs/findings.md for Policy Need

### Analysis
Searched `docs/findings.md` for existing backend selection policies:
```bash
rg "backend.*selection|POLICY-002" docs/findings.md
# Result: No matches found
```

### Decision: POLICY-002 NOT REQUIRED

**Rationale:**
- Backend selection behavior is **normative specification** (§4.8 in ptychodus_api_spec.md), not policy
- Existing policies already cover enforcement:
  - **POLICY-001** (PyTorch mandatory, fail-fast on missing torch)
  - **CONFIG-001** (params.cfg initialization requirement)
- §4.8 spec text provides comprehensive contract for implementers
- No new discovery/convention/recurring issue to document in findings

**Documented:** Added C2 completion note to `phase_e3_spec_patch.md` Plan Integration section.

## Exit Criteria Validation

✅ **C1 Complete:**
- `specs/ptychodus_api_spec.md` updated with §4.8 backend selection text
- Cross-references to implementation (`backend_selector.py`) and tests (`test_backend_selection.py`) preserved
- Spec language mirrors approved draft from `phase_e3_spec_patch.md`

✅ **C2 Complete:**
- Reviewed `docs/findings.md` for policy alignment
- Determined POLICY-002 not required (behavior covered by normative spec + existing POLICY-001/CONFIG-001)
- Documented decision in `phase_e3_spec_patch.md`

## Artifacts

| File Path | Purpose | Size |
|-----------|---------|------|
| `specs/ptychodus_api_spec.md` | Updated spec with §4.8 | 311 lines → 323 lines (+12) |
| `phase_e3_spec_patch.md` | Draft source + completion notes | Updated Plan Integration section |
| `phase_e3_spec_update.md` | This summary document | — |

## Next Steps

Per `phase_e3_docs_plan.md`:
1. **Phase E3.B** — Update developer-facing docs (`docs/workflows/pytorch.md`, `docs/architecture.md`, `CLAUDE.md`, `README.md`) with backend selection guidance
2. **Phase E3.D** — Author `phase_e3_handoff.md` for TEST-PYTORCH-001 Phase D3 coordination
3. **Update plan checklists** — Mark C1/C2 rows `[x]` in `phase_e3_docs_plan.md`

## References

- `specs/ptychodus_api_spec.md:224-235` — New §4.8 Backend Selection & Dispatch
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_spec_patch.md` — Source draft
- `docs/findings.md:8` — POLICY-001 (PyTorch mandatory)
- `ptycho/workflows/backend_selector.py:121-165` — Implementation reference
- `tests/torch/test_backend_selection.py:59-170` — Test contract
