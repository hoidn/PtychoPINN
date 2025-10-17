# Phase B.B1 Redline Outline — INTEGRATE-PYTORCH-000

## Context
- Objective: Prepare canonical `plans/ptychodus_pytorch_integration_plan.md` for Phase B refactor so it reflects the rebased `ptycho_torch/` stack (commit bfc22e7).
- Inputs: `delta_log.md` (Critical Deltas 1-5), `module_inventory.md`, `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/parity_map.md`.
- Contract touchpoints: `specs/ptychodus_api_spec.md:11-274`, `specs/data_contracts.md:1-120`, `docs/workflows/pytorch.md`.

## Revision Outline (by Plan Phase)
| Plan Section | Required Updates | Rationale | Source Pointers |
| --- | --- | --- | --- |
| Phase 0 — Discovery & Design | Add decision gate for selecting PyTorch entry surface (`api/base_api.py` vs low-level modules); document new `api/` package structure and ownership of MLflow orchestration. | `delta_log.md` Δ2 shows high-level API addition absent from legacy plan. | `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md#🚨-delta-2-new-high-level-api-layer`; `ptycho_torch/api/base_api.py:30-410`. |
| Phase 1 — Config & Legacy Bridge | Replace singleton config assumptions with dataclass ingestion narrative; include mapping table for PyTorch config fields → spec-required fields (`gridsize`, `pad_object`, `gaussian_smoothing_sigma`, etc.). Call out need for `update_legacy_dict` and `KEY_MAPPINGS`. | Δ1 identifies schema mismatch blocking downstream modules. | `specs/ptychodus_api_spec.md:20-125`; `ptycho_torch/config_params.py:1-240`; `delta_log.md#🚨-delta-1-configuration-schema-mismatch`. |
| Phase 2 — Data Pipeline | Introduce `RawDataTorch` wrapper scope and memmap-to-cache bridging; list `datagen/` package for synthetic data parity and call out shared NPZ contract. | Δ3 + Δ5 show data ingestion differences (memory map, datagen). | `specs/data_contracts.md:1-120`; `ptycho_torch/dset_loader_pt_mmap.py:40-220`; `ptycho_torch/datagen/datagen.py:10-210`. |
| Phase 3 — Inference & Stitching | Document barycentric reassembly modules, multi-GPU path, and parity checks vs `ptycho/tf_helper.py` outputs. Add requirement for numeric comparators. | Δ4 identifies alternative stitching path requiring parity guardrails. | `ptycho_torch/reassembly.py:1-220`; `ptycho/tf_helper.py:20-260`; `delta_log.md#⚠️-delta-4-reassembly-module-suite`. |
| Phase 4 — Persistence & Orchestration | Describe Lightning + MLflow workflow, checkpoint formats, and proposal for archive shim to meet `MODEL_FILE_NAME = 'wts.h5.zip'`. Introduce mitigation for optional MLflow/Lightning dependencies. | Δ5 + parity summary highlight orchestration divergence. | `ptycho_torch/train.py:23-226`; `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/summary.md#2-critical-path-blockers`. |
| Phase 5 — Testing & Governance | Tie plan to TEST-PYTORCH-001 initiative; define backend-parametrized integration tests, fixture requirements, and spec updates queue. | Ensure governance sync with high-priority initiative. | `plans/pytorch_integration_test_plan.md`; `docs/TESTING_GUIDE.md`; `plans/active/INTEGRATE-PYTORCH-001/implementation.md#phase-e`. |

## Decision Inventory for B1 Handoff
| ID | Decision | Recommended Owner | Evidence |
| --- | --- | --- | --- |
| D1 | Integrate PyTorch via `api/` layer or direct module calls? | INTEGRATE-PYTORCH-001 Phase B lead | Evaluate maintenance cost vs spec alignment. Δ2 summary + parity map §"Workflow". |
| D2 | Harmonize config schemas vs translation shim? | INTEGRATE-PYTORCH-001 Phase B + spec maintainer | `config_params.py` fields vs `specs/ptychodus_api_spec.md` table; delta Δ1. |
| D3 | Persistence format strategy (unified archive vs dual filters). | INTEGRATE-PYTORCH-001 Phase D + Ptychodus maintainers | Parity summary §"Risk 3"; review `ptycho/model_manager.py:40-210`. |
| D4 | Lightning & MLflow dependency policy on CI. | Supervisors + TEST-PYTORCH-001 | Parity summary §"Risk 1"; `docs/workflows/pytorch.md`. |

## Canonical Plan Editing Order (for Phase B.B2)
1. Update Section 1 scope paragraph to mention dual backend surface and highlight config bridge as first milestone.
2. Overhaul Phase 0-4 subsections per table above; embed cross-links to delta artifacts for traceability.
3. Add new subsection "Spec & Ledger Sync" capturing requirement to update `specs/ptychodus_api_spec.md` once PyTorch semantics finalized.
4. Refresh deliverables list to include memory-mapped data shim + persistence adapter deliverables explicitly.
5. Append new risk entry for API layer drift vs low-level integration.

## Artifact Checklist
| File | Purpose | Status |
| --- | --- | --- |
| `plan_redline.md` (this file) | Draft outline for canonical plan edits | ✅ Created 2025-10-17T025633Z |
| `summary.md` | Capture key decisions + next steps for Phase B.B2/B3 | ✅ Created 2025-10-17T025633Z |

