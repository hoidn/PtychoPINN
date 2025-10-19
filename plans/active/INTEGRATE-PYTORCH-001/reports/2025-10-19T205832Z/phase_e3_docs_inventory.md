# Phase E3 — Documentation Gap Inventory (INTEGRATE-PYTORCH-001)

**Date:** 2025-10-19
**Initiative:** INTEGRATE-PYTORCH-001 — PyTorch backend integration
**Phase:** E3 — Documentation & Handoff Planning
**Tasks:** A.A1 (Developer Docs), A.A2 (Spec & Findings), A.A3 (TEST-PYTORCH-001 Handoff)
**Mode:** Docs-only inventory loop (no file edits)

---

## Executive Summary

This inventory captures documentation gaps and spec amendments required to surface the now-operational PyTorch backend to downstream consumers (especially Ptychodus). The PyTorch backend achieved end-to-end parity in Phase D2 (Attempt #40), validated through integration testing in TEST-PYTORCH-001 (Attempt #6), yet developer-facing documentation still contains TensorFlow-only assumptions and lacks clear backend selection guidance.

**Key Findings:**
1. **Developer docs** (README.md, pytorch.md) accurately reflect PyTorch availability but lack backend selection API guidance
2. **Spec** (ptychodus_api_spec.md) documents PyTorch adapters but lacks explicit backend dispatch behavior
3. **Workflow docs** (pytorch.md) contain TensorFlow parity notes but no NotImplementedError warnings
4. **TEST-PYTORCH-001** needs CI guidance handoff from INTEGRATE-PYTORCH-001

---

## A.A1 — Developer Documentation Gaps

### 1. `docs/workflows/pytorch.md`

**Current Status:** Largely accurate as of Phase D2 updates (Attempt #13)
**Issues Found:**

#### ✅ NO ISSUES - Documentation Accurate
- Section 1 (Overview): Correctly describes parity with TensorFlow workflows
- Section 2 (Prerequisites): Accurately states PyTorch >=2.2 REQUIRED (POLICY-001 compliant)
- Section 5 (Complete Training Workflow): Correctly documents `run_cdi_example_torch` with `do_stitching=True` support
- Section 7 (Inference & Reconstruction): Accurately describes `_reassemble_cdi_image_torch` implementation (Phase D2.C complete)
- Section 9 (Differences from TensorFlow): Valid comparison table showing Lightning vs TensorFlow execution engines
- Section 11 (Regression Test & Runtime Expectations): Added by TEST-PYTORCH-001 Attempt #11, comprehensive
- Section 13 (Keeping Parity with TensorFlow): Good cross-reference guidance

**NotImplementedError Search:** ZERO stale warnings found (searched lines 1-343)

**Gap:** **No backend selection guidance.** The document describes "how to use PyTorch workflows" but does not explain:
- How users choose between TensorFlow and PyTorch backends in Ptychodus
- Whether there is a config flag/environment variable to control backend dispatch
- Default backend behavior if both are available

**Recommendation:** Add new Section 14 "Backend Selection in Ptychodus Integration" documenting:
```markdown
## 14. Backend Selection in Ptychodus Integration

When PtychoPINN is used via Ptychodus, backend selection is controlled through:
- [TBD: Config flag name, e.g., `ModelConfig.backend = 'pytorch' | 'tensorflow'`]
- [TBD: Default behavior (e.g., defaults to TensorFlow for backward compatibility)]
- [TBD: Fail-fast behavior if PyTorch not installed but requested]
```

**File:** `docs/workflows/pytorch.md`
**Line Anchors:** Insert after line 334 (end of Section 13)
**Severity:** Medium — Ptychodus integration needs this for Phase E completion

---

### 2. `README.md`

**Current Status:** Minimal PyTorch mention
**Issues Found:**

#### Line 30 — Accurate PyTorch Installation Note
```markdown
**Note:** This will automatically install PyTorch >= 2.2 as a required dependency. For GPU acceleration...
```
✅ This is correct per POLICY-001.

**Gap:** **No mention of dual-backend architecture.** Users installing PtychoPINN don't know:
- That both TensorFlow and PyTorch backends exist
- How to choose between them
- That PyTorch is now fully operational (as of Phase D2)

**Recommendation:** Add subsection under "## Features" (after line 16):
```markdown
### Dual-Backend Architecture (New as of 2025-10)
- **TensorFlow Backend:** Original implementation with full feature parity
- **PyTorch Backend:** Production-ready alternative using Lightning orchestration
  - See `docs/workflows/pytorch.md` for complete PyTorch workflow guide
  - Validated through comprehensive integration testing (TEST-PYTORCH-001)
```

**File:** `README.md`
**Line Anchors:** Insert around line 17 (after Scalability and Speed bullet)
**Severity:** Low — Nice-to-have for user awareness, not blocking Ptychodus integration

---

### 3. `docs/architecture.md`

**Current Status:** TensorFlow-centric component diagram
**Issues Found:**

#### Lines 13-42 — Component Diagram Lacks PyTorch Stack
The Mermaid diagram shows only TensorFlow modules (`params.py`, `model.py`, `tf_helper.py`). No mention of:
- `ptycho_torch/` parallel module structure
- `ptycho_torch.config_bridge` adapter layer
- PyTorch/TensorFlow backend switch logic

**Gap:** **Component diagram does not reflect dual-backend reality.**

**Recommendation:** Add note after diagram (insert at line 43):
```markdown
**Note:** This diagram shows the TensorFlow backend (`ptycho/`). The PyTorch backend (`ptycho_torch/`) mirrors this architecture with Lightning-based orchestration. See `docs/workflows/pytorch.md` for PyTorch component details and `specs/ptychodus_api_spec.md` §2.2 for configuration bridge adapters.
```

**File:** `docs/architecture.md`
**Line Anchors:** Insert after line 42 (after component diagram)
**Severity:** Low — Primarily affects new contributors understanding codebase layout

---

### 4. `CLAUDE.md` (Agent Guidance)

**Current Status:** PyTorch awareness documented
**Issues Found:**

#### Lines 53-59 — POLICY-001 Documented
✅ Directive level="critical" correctly states PyTorch requirement per POLICY-001.

**Gap:** **No CONFIG-001 reminder for PyTorch path.** The document warns about TensorFlow's `update_legacy_dict` requirement but doesn't mention that PyTorch workflows also need this bridge.

**Recommendation:** Add explicit note in Section 4.1 (after line 92):
```markdown
**PyTorch workflows:** The PyTorch backend (`ptycho_torch/workflows/components.py`) also requires `update_legacy_dict` before data loading, as it reuses the TensorFlow data pipeline (`ptycho.raw_data`, `ptycho.loader`). See <doc-ref type="workflow">docs/workflows/pytorch.md</doc-ref> §3 for PyTorch-specific CONFIG-001 compliance.
```

**File:** `CLAUDE.md`
**Line Anchors:** Insert after line 92 (after TensorFlow example)
**Severity:** Medium — Affects agent debugging discipline for PyTorch development

---

## A.A2 — Specification & Findings Coverage

### 1. `specs/ptychodus_api_spec.md`

**Current Status:** Comprehensive PyTorch adapter documentation
**Issues Found:**

#### Lines 75-81 — PyTorch Configuration Adapters Documented
✅ Section 2.2 describes `ptycho_torch.config_bridge` with complete function signatures and field mappings.

#### Line 14 — PyTorch Requirement Warning
✅ Section 1 includes POLICY-001 banner with actionable RuntimeError guidance.

#### Lines 140-223 — Reconstructor Contract (Section 4)
**Gap:** **No backend selection/dispatch behavior specified.** The spec documents:
- How `PtychoPINNTrainableReconstructor` interfaces with TensorFlow workflows (Section 4.1-4.6)
- PyTorch config adapters (Section 2.2)

But does NOT specify:
- Whether `PtychoPINNTrainableReconstructor` should support runtime backend switching
- How Ptychodus chooses which backend to invoke
- Expected fail-fast behavior if PyTorch backend requested but unavailable
- Persistence format compatibility (can PyTorch checkpoints be loaded by TensorFlow backend and vice versa?)

**Recommendation:** Add new subsection **§4.8 Backend Selection and Dispatch**:
```markdown
#### 4.8. Backend Selection and Dispatch

**Contract (TBD - Phase E3 design decision required):**

Option A (Static Backend at Instantiation):
- `PtychoPINNTrainableReconstructor` constructor accepts `backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'`
- Backend selection is immutable after instantiation
- Raises `RuntimeError` if PyTorch backend requested but `torch` not importable (POLICY-001)
- Checkpoints are backend-specific; cross-loading not supported in Phase E

Option B (Transparent Backend Abstraction):
- Reconstructor auto-detects available backends at import time
- Falls back to TensorFlow if PyTorch unavailable (backward compatible)
- Checkpoint format includes backend metadata; loader dispatches accordingly

**Recommendation for Phase E:** Prefer Option A for explicit control and clear error messages.
```

**File:** `specs/ptychodus_api_spec.md`
**Line Anchors:** Insert after Section 4.7 (line 222)
**Severity:** **HIGH — BLOCKING PHASE E COMPLETION** (requires design decision + implementation)

---

### 2. `docs/findings.md`

**Current Status:** POLICY-001 and FORMAT-001 documented
**Issues Found:**

#### Lines 8-9 — POLICY-001 (PyTorch Mandatory)
✅ Correctly documented with governance decision link.

#### Line 18 — FORMAT-001 (NPZ Auto-Transpose)
✅ Correctly documented with callchain summary link.

**Gap:** **No finding for backend selection behavior.** Once Phase E implements backend dispatch (per spec gap above), a new finding should document the chosen approach.

**Recommendation:** Prepare placeholder for **POLICY-002** (or similar ID):
```markdown
| POLICY-002 | 2025-10-XX | policy, backend-selection, dispatch, ptychodus | Backend selection in PtychoPINN is [static/dynamic] per `specs/ptychodus_api_spec.md` §4.8. Ptychodus controls backend via [config field/env var]. Fail-fast behavior enforced when PyTorch backend requested but unavailable. | [Link](specs/ptychodus_api_spec.md#48-backend-selection-and-dispatch) | Active |
```

**File:** `docs/findings.md`
**Line Anchors:** Append after line 18 (after FORMAT-001) once design finalized
**Severity:** Medium — Deferred until §4.8 implementation complete

---

## A.A3 — TEST-PYTORCH-001 Handoff Requirements

### Current Status of TEST-PYTORCH-001
Per `plans/active/TEST-PYTORCH-001/implementation.md`:
- **Phase A (Baseline):** ✅ Complete (Attempt #2) — Baseline passing in 35.92s mean runtime
- **Phase C (Modernization):** ✅ Complete (Attempt #6) — Pytest harness functional, GREEN
- **Phase D1 (Runtime Profile):** ✅ Complete (Attempt #10) — Guardrails defined (≤90s CI max)
- **Phase D2 (Documentation):** ✅ Complete (Attempt #11) — `docs/workflows/pytorch.md` §11 updated
- **Phase D3 (CI Integration):** ⏸️ **PENDING** — CI config review not yet executed

### Handoff Items Required from INTEGRATE-PYTORCH-001

#### 1. CI Execution Guidance
**Gap:** TEST-PYTORCH-001 Phase D3 requires CI workflow configuration decisions:
- Pytest markers for integration tests (`@pytest.mark.integration`, `@pytest.mark.slow`)
- Skip policy for PyTorch tests in TensorFlow-only CI runners
- Timeout settings (120s recommended per D1 runtime profile)
- Retry policy (1 retry on timeout per D1.C guidance)

**Handoff Action:** INTEGRATE-PYTORCH-001 should provide:
```markdown
### CI Policy for PyTorch Integration Tests

**Execution Environment:**
- CPU-only (`CUDA_VISIBLE_DEVICES=""` enforced via fixture)
- Python 3.11+, PyTorch 2.8.0+ (see `reports/2025-10-19T193425Z/phase_d_hardening/env_snapshot.txt`)
- Requires 128MB disk space for checkpoints + artifacts

**Pytest Configuration:**
- Selector: `pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`
- Markers: Add `@pytest.mark.integration` and `@pytest.mark.slow` to test function
- Timeout: 120s (conservative 3.3× baseline per runtime_profile.md)
- Retry: 1 retry on timeout/non-zero exit

**Skip Policy:**
- Auto-skip in TensorFlow-only environments via `tests/conftest.py` directory-based collection rules
- Local development expects PyTorch present (fail with actionable ImportError per POLICY-001)
```

**Owner:** INTEGRATE-PYTORCH-001 final documentation pass (this Phase E3)
**Destination:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_handoff.md`

---

#### 2. Parity Validation Selectors
**Gap:** TEST-PYTORCH-001 needs to know which additional selectors to monitor for ongoing parity.

**Handoff Action:** Provide list of critical parity tests:
```markdown
### Parity Validation Selectors for Ongoing Maintenance

**TensorFlow Baseline:**
- `pytest tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle -vv`
  (TensorFlow equivalent integration test for comparison)

**PyTorch Parity:**
- `pytest tests/torch/test_config_bridge.py -k parity -vv`
  (Config adapter parity per specs/ptychodus_api_spec.md §2.2)
- `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv`
  (Lightning orchestration contract per phase_d2_completion.md B1-B4)
- `pytest tests/torch/test_workflows_components.py -k ReassembleCdiImageTorch -vv`
  (Stitching path per phase_d2_completion.md C1-C4)
- `pytest tests/torch/test_lightning_checkpoint.py -vv`
  (Checkpoint serialization per phase_d2_completion.md D1c)
- `pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity -vv`
  (Decoder parity per phase_d2_completion.md D1e)

**Run Frequency:** After any changes to `ptycho_torch/model.py`, `ptycho_torch/workflows/components.py`, or `ptycho_torch/config_bridge.py`
```

**Owner:** INTEGRATE-PYTORCH-001 final documentation pass
**Destination:** Same `phase_e3_handoff.md` document

---

#### 3. Dataset & Fixture Coordination
**Gap:** TEST-PYTORCH-001 currently uses canonical dataset (`datasets/Run1084_recon3_postPC_shrunk_3.npz`, 35MB). INTEGRATE-PYTORCH-001 should confirm this is the approved fixture or provide alternative.

**Current Status:**
- Fixture location: Repository root `datasets/` directory
- Format: Canonical DATA-001 compliant NPZ (FORMAT-001 auto-transpose guard active)
- Size: 35MB (1087 diffraction patterns, 64x64)
- Runtime impact: Contributes ~30s of 35.92s total runtime (85% of execution)

**Handoff Action:** Confirm fixture strategy:
```markdown
### Fixture Strategy for PyTorch Integration Tests

**Decision:** Use existing canonical dataset (`datasets/Run1084_recon3_postPC_shrunk_3.npz`)
- **Rationale:** Real-world data exercises FORMAT-001 guard and validates production data path
- **Trade-off:** Longer runtime (35.92s) vs synthetic minimal fixture (<5s)
- **Governance:** Approved per TEST-PYTORCH-001 Attempt #2 baseline assessment

**Alternative (if runtime becomes issue):**
- Phase B fixture minimization (deferred) would create `tests/fixtures/pytorch_integration/minimal_train.npz`
- See `plans/active/TEST-PYTORCH-001/implementation.md` Phase B for design notes
```

**Owner:** Joint decision (INTEGRATE-PYTORCH-001 + TEST-PYTORCH-001)
**Destination:** `phase_e3_handoff.md`

---

#### 4. Ownership Matrix
**Gap:** Unclear who maintains PyTorch integration tests post-Phase E.

**Handoff Action:** Define responsibility boundaries:
```markdown
### Ownership Matrix — PyTorch Integration Testing

| Component | Owner | Scope |
|-----------|-------|-------|
| `ptycho_torch/` module code | INTEGRATE-PYTORCH-001 alumni | PyTorch backend implementation |
| `tests/torch/test_integration_workflow_torch.py` | TEST-PYTORCH-001 alumni | Integration regression maintenance |
| `tests/torch/test_config_bridge.py` | INTEGRATE-PYTORCH-001 alumni | Config adapter parity |
| `tests/torch/test_workflows_components.py` | INTEGRATE-PYTORCH-001 alumni | Lightning orchestration + stitching |
| CI workflow configuration (`.github/workflows/`) | DevOps / Project Lead | CI policy enforcement |
| `docs/workflows/pytorch.md` updates | INTEGRATE-PYTORCH-001 alumni | Workflow documentation |
| TEST-PYTORCH-001 plan updates | TEST-PYTORCH-001 owner | Test strategy evolution |

**Escalation Path:** New PyTorch backend issues → check `docs/fix_plan.md` [INTEGRATE-PYTORCH-001] history → file issue referencing Phase D2/E evidence
```

**Owner:** INTEGRATE-PYTORCH-001 final governance pass
**Destination:** `phase_e3_handoff.md`

---

## Summary of Recommendations

### Immediate Actions (Phase E3 Execution)

| ID | File | Action | Severity | Owner |
|----|------|--------|----------|-------|
| A1.1 | `docs/workflows/pytorch.md` | Add Section 14 "Backend Selection in Ptychodus Integration" | **HIGH** | INTEGRATE-PYTORCH-001 (this loop or next) |
| A2.1 | `specs/ptychodus_api_spec.md` | Add §4.8 "Backend Selection and Dispatch" | **HIGH BLOCKING** | INTEGRATE-PYTORCH-001 + design review |
| A3.1 | Create `phase_e3_handoff.md` | Consolidate CI, parity, fixture, ownership guidance | **HIGH** | INTEGRATE-PYTORCH-001 (Phase E3.D) |

### Optional Enhancements (Deferrable)

| ID | File | Action | Severity |
|----|------|--------|----------|
| A1.2 | `README.md` | Add "Dual-Backend Architecture" subsection | Low |
| A1.3 | `docs/architecture.md` | Add PyTorch backend note after component diagram | Low |
| A1.4 | `CLAUDE.md` | Add CONFIG-001 PyTorch reminder in Section 4.1 | Medium |
| A2.2 | `docs/findings.md` | Add POLICY-002 for backend selection (after §4.8 design) | Medium |

---

## Next Steps

**Phase E3.B (Documentation Updates):**
1. Implement A1.1 (pytorch.md Section 14 backend selection)
2. Draft A2.1 (spec §4.8) for design review
3. Optional: Implement A1.4 (CLAUDE.md CONFIG-001 reminder)

**Phase E3.D (Handoff Package):**
1. Create `phase_e3_handoff.md` consolidating A3.1-A3.4 guidance
2. Update `plans/active/TEST-PYTORCH-001/implementation.md` Phase D3 with handoff link
3. Append `docs/fix_plan.md` [TEST-PYTORCH-001] with Phase D3 completion artifact

**Blockers:**
- §4.8 design decision (static vs dynamic backend selection) — requires architecture review before implementation

---

## Artifact Metadata

- **Created:** 2025-10-19
- **Initiative:** INTEGRATE-PYTORCH-001
- **Phase:** E3.A (Gap Assessment)
- **Tasks:** A.A1 (Developer Docs), A.A2 (Spec & Findings), A.A3 (TEST-PYTORCH-001 Handoff)
- **Evidence:**
  - Phase D2 parity: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T201500Z/phase_d2_completion/parity_update.md`
  - TEST-PYTORCH-001 baseline: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/baseline/summary.md`
  - Runtime profile: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md`
- **Cross-References:**
  - `docs/findings.md#POLICY-001` (PyTorch mandatory)
  - `docs/findings.md#FORMAT-001` (NPZ auto-transpose guard)
  - `specs/ptychodus_api_spec.md` §2.2 (PyTorch config adapters)
  - `specs/ptychodus_api_spec.md` §4.1-4.7 (Reconstructor contract)
