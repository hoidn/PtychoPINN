# Phase F4.2 — Spec & Findings Synchronization

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** F4.2 (Spec & Findings)
**Timestamp:** 2025-10-17T205413Z
**Engineer:** ralph

## Summary

Successfully synchronized normative specifications and knowledge base with the torch-required baseline established in Phase F. Updated `specs/ptychodus_api_spec.md` to document PyTorch as mandatory dependency, added POLICY-001 finding to `docs/findings.md`, and updated cross-references in CLAUDE.md directive.

## Completed Tasks

### F4.2.A — Update Spec Prerequisites

**File:** `specs/ptychodus_api_spec.md`

**Location:** Section 1 (Overview), after line 12

**Change:** Added PyTorch requirement paragraph

**Anchor:** `specs/ptychodus_api_spec.md:14` (new paragraph starts)

**Content:**
```markdown
**⚠️ PyTorch Requirement:** As of Phase F (INTEGRATE-PYTORCH-001), PyTorch `>= 2.2` is a **mandatory runtime dependency** for the PyTorch backend (`ptycho_torch/`). The package specifies `torch>=2.2` in `setup.py` install_requires. The TensorFlow backend (`ptycho/`) continues to function independently, but callers integrating the PyTorch stack **must** ensure PyTorch is available; the system will raise an actionable `RuntimeError` if torch cannot be imported. This policy is documented in <doc-ref type="findings">docs/findings.md#policy-001</doc-ref> and reflects the governance decision archived at `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md`. For installation guidance, see the PyTorch workflow guide at <doc-ref type="workflow">docs/workflows/pytorch.md</doc-ref>.
```

**Rationale:** Explicitly documents the torch-required policy at the top of the spec so downstream implementers (e.g., Ptychodus integrators) understand the dependency requirement upfront.

---

**File:** `specs/ptychodus_api_spec.md`

**Location:** Section 2.3 (The Compatibility Bridge), after line 73

**Change:** Added PyTorch Configuration Adapters subsection

**Anchor:** `specs/ptychodus_api_spec.md:75-81` (new bullet block)

**Content:**
```markdown
- **PyTorch Configuration Adapters (`ptycho_torch.config_bridge`):**
  - **Purpose**: Translate PyTorch singleton configuration objects to TensorFlow dataclass instances, enabling PyTorch workflows to populate `params.cfg` via the standard `update_legacy_dict` function.
  - **Key Functions**:
    - `to_model_config(data: DataConfig, model: ModelConfig, overrides=None) -> TFModelConfig`: Converts PyTorch `DataConfig` and `ModelConfig` to TensorFlow `ModelConfig`, handling critical transformations such as `grid_size` tuple → `gridsize` int, `mode` enum → `model_type` enum, and activation name normalization.
    - `to_training_config(model: TFModelConfig, data: DataConfig, pt_model: ModelConfig, training: TrainingConfig, overrides=None) -> TFTrainingConfig`: Translates PyTorch training parameters to TensorFlow `TrainingConfig`, converting `epochs` → `nepochs`, `K` → `neighbor_count`, `nll` bool → `nll_weight` float, and requiring explicit `overrides` for fields missing in PyTorch configs (e.g., `train_data_file`, `n_groups`).
    - `to_inference_config(model: TFModelConfig, data: DataConfig, inference: InferenceConfig, overrides=None) -> TFInferenceConfig`: Converts PyTorch inference parameters to TensorFlow `InferenceConfig`, mapping `K` → `neighbor_count` and requiring `overrides` for `model_path` and `test_data_file`.
  - **Contract**: These adapters MUST produce dataclasses compatible with `update_legacy_dict` and maintain behavioral parity with direct TensorFlow dataclass instantiation. Consumers (e.g., `ptychodus` PyTorch integration) MUST call these adapters before invoking `update_legacy_dict` to ensure correct params.cfg population. Implementation details and field mappings are documented in `ptycho_torch/config_bridge.py:1-380` and tested via `tests/torch/test_config_bridge.py`.
```

**Rationale:** Documents the config bridge adapter contract (Phase B deliverable) in the normative spec, ensuring downstream integrators understand the required translation layer for PyTorch→TensorFlow config parity.

---

**File:** `specs/ptychodus_api_spec.md`

**Location:** Section 4.2 (Configuration Handshake), after line 161

**Change:** Added PyTorch Import Requirement bullet

**Anchor:** `specs/ptychodus_api_spec.md:162` (new bullet)

**Content:**
```markdown
- **PyTorch Import Requirement (Phase F)**: The PyTorch backend (`ptycho_torch/`) **must** raise an actionable `RuntimeError` with installation guidance if `torch` cannot be imported. Silent fallbacks or optional import guards are prohibited per <doc-ref type="findings">docs/findings.md#policy-001</doc-ref>. All modules in `ptycho_torch/` assume PyTorch availability and will fail fast with clear error messages directing users to install `torch>=2.2`. Test suites automatically skip `tests/torch/` in TensorFlow-only CI environments via directory-based pytest collection rules (`tests/conftest.py`), but local development expects PyTorch to be present.
```

**Rationale:** Codifies the fail-fast behavior and torch-required policy in the reconstructor contract section, explicitly prohibiting silent fallbacks and referencing the knowledge base entry.

---

### F4.2.B — Add Knowledge-Base Entry

**File:** `docs/findings.md`

**Location:** Line 8 (appended after CONVENTION-001, maintaining alphabetical ordering within table)

**Change:** Added POLICY-001 row

**Anchor:** `docs/findings.md:8`

**Content:**
```markdown
| POLICY-001 | 2025-10-17 | policy, PyTorch, dependencies, mandatory | PyTorch (torch>=2.2) is now a mandatory dependency for PtychoPINN as of Phase F (INTEGRATE-PYTORCH-001). All code in `ptycho_torch/` and `tests/torch/` assumes PyTorch is installed. Torch-optional execution paths were removed; modules raise actionable RuntimeError if torch import fails. Tests in `tests/torch/` are automatically skipped in TensorFlow-only CI environments via directory-based pytest collection rules in `tests/conftest.py`, but will fail with actionable ImportError messages if PyTorch is missing in local development. Migration rationale and implementation evidence documented in governance decision and Phase F implementation logs. | [Link](plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md) | Active |
```

**Rationale:** Creates authoritative policy reference (POLICY-001) capturing the torch-required transition, linking back to Phase F1 governance decision artifact, ensuring future developers understand the dependency baseline.

---

### F4.2.C — Verify Cross-References

**File:** `CLAUDE.md`

**Location:** Line 58 (directive block, after existing text)

**Change:** Appended doc-ref tag to directive

**Anchor:** `CLAUDE.md:58`

**Before:**
```markdown
Migration rationale and evidence documented in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md` and Phase F implementation logs (`docs/fix_plan.md` INTEGRATE-PYTORCH-001 history).
```

**After:**
```markdown
Migration rationale and evidence documented in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md` and Phase F implementation logs (`docs/fix_plan.md` INTEGRATE-PYTORCH-001 history). This policy is captured in <doc-ref type="findings">docs/findings.md#policy-001</doc-ref>.
```

**Rationale:** Completes the documentation loop by linking the critical CLAUDE directive back to the knowledge base entry, ensuring all policy references point to POLICY-001.

---

## Verification Checklist

- [x] Section 1 of spec updated with PyTorch requirement paragraph
- [x] Section 2.3 of spec extended with config bridge adapter documentation
- [x] Section 4.2 of spec augmented with fail-fast import behavior
- [x] POLICY-001 row added to docs/findings.md table (line 8)
- [x] CLAUDE.md directive (line 58) now references POLICY-001 via doc-ref tag
- [x] All doc-ref tags use correct type and anchor format
- [x] Spec paragraphs reference governance decision artifact path
- [x] Markdown table formatting preserved in findings.md (pipes aligned)
- [x] No numbering conflicts or ID reuse in findings table

## File Anchors Summary

| File | Section/Line | Purpose |
|------|--------------|---------|
| `specs/ptychodus_api_spec.md:14` | Section 1 (new paragraph) | PyTorch requirement statement |
| `specs/ptychodus_api_spec.md:75-81` | Section 2.3 (new bullet) | Config bridge adapter documentation |
| `specs/ptychodus_api_spec.md:162` | Section 4.2 (new bullet) | Fail-fast import requirement |
| `docs/findings.md:8` | Knowledge base table | POLICY-001 row |
| `CLAUDE.md:58` | Directive block | Cross-reference to POLICY-001 |

## Evidence Links

- Governance decision: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md`
- Phase F implementation logs: `docs/fix_plan.md` (INTEGRATE-PYTORCH-001 Attempts #64-#74)
- Config bridge implementation: `ptycho_torch/config_bridge.py:1-380`
- Config bridge tests: `tests/torch/test_config_bridge.py`
- Workflow guide: `docs/workflows/pytorch.md:17-22`

## Open Items

None — F4.2 complete per exit criteria.

## Next Actions

- **F4.3.A**: Map impacted initiatives & CI tasks (handoff notes)
- **F4.3.B**: Define verification cadence (pytest collection checks)
- **F4.3.C**: Update plan & ledger cross-references (phase_f_torch_mandatory.md, docs/fix_plan.md)

---

*Generated: 2025-10-17T205413Z*
