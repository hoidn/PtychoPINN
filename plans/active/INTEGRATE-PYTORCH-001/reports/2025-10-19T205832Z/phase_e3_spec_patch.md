# Phase E3 — Backend Selection Spec Draft (INTEGRATE-PYTORCH-001)

## Context
- Initiative: INTEGRATE-PYTORCH-001 — Phase E3 (Documentation & Spec Handoff)
- Scope: Draft normative guidance for `specs/ptychodus_api_spec.md` §4.8 covering backend selection + fail-fast behaviour, and provide source notes for downstream documentation updates.
- Blocking issue: Inventory (`plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T205832Z/phase_e3_docs_inventory.md`) flagged absence of §4.8 as **HIGH BLOCKING** for Phase E exit.
- Dependencies: CONFIG-001 (params.cfg synchronization), POLICY-001 (PyTorch mandatory), FORMAT-001 (NPZ auto-transpose), backend selector implementation (`ptycho/workflows/backend_selector.py`), backend selection tests (`tests/torch/test_backend_selection.py`).

## Primary Sources Consulted
- `specs/ptychodus_api_spec.md` §4.1–§4.7 — existing reconstructor contract (no backend dispatch section).
- `ptycho/config/config.py:110`, `ptycho/config/config.py:142` — `backend: Literal['tensorflow', 'pytorch'] = 'tensorflow'` defaults.
- `ptycho/workflows/backend_selector.py:78-325` — canonical dispatcher behaviour (CONFIG-001 update, runtime routing, error handling, metadata injection).
- `tests/torch/test_backend_selection.py:1-228` — TDD contract outlining expected defaults, explicit selection, CONFIG-001 enforcement, torch unavailability messaging, inference parity.
- `docs/workflows/pytorch.md:246-336` — current documentation gap (no backend selection guidance).
- `docs/findings.md:8`, `docs/findings.md:18` — POLICY-001 (PyTorch mandatory), FORMAT-001 (legacy NPZ guard).

## Behaviour Snapshot (As Implemented Today)
1. **Configuration defaults** — `TrainingConfig.backend`/`InferenceConfig.backend` default to `'tensorflow'` (`ptycho/config/config.py:110`, `ptycho/config/config.py:142`) preserving backward compatibility.
2. **Dispatcher flow** — `run_cdi_example_with_backend()` updates `params.cfg` via `update_legacy_dict` before routing (`ptycho/workflows/backend_selector.py:121-141`), validates backend literal, delegates to TensorFlow or PyTorch workflows, and injects `results['backend']` for traceability (`ptycho/workflows/backend_selector.py:163-165`).
3. **Error handling** — Selecting `'pytorch'` without torch availability raises actionable `RuntimeError` recommending installation (`ptycho/workflows/backend_selector.py:142-156`) satisfying POLICY-001.
4. **Inference parity** — `load_inference_bundle_with_backend()` mirrors training dispatcher behaviour (`ptycho/workflows/backend_selector.py:241-323`) and relies on persistence shims (`ptycho_torch/workflows/components.py`, `tests/torch/test_model_manager.py:238-372`).
5. **TDD expectations** — Backend selection tests (currently marked as documentation/contract) expect identical API signatures, CONFIG-001 enforcement, and explicit backend metadata (`tests/torch/test_backend_selection.py:59-170`).

## Proposed Specification Additions — §4.8 Backend Selection & Dispatch

> ### 4.8. Backend Selection & Dispatch (NEW)
> - **Configuration Field**: `TrainingConfig.backend` and `InferenceConfig.backend` MUST accept the literals `'tensorflow'` or `'pytorch'` and SHALL default to `'tensorflow'` to maintain backward compatibility. Callers MAY override this field when invoking PtychoPINN through Ptychodus.
> - **CONFIG-001 Compliance**: Implementations MUST call `update_legacy_dict(ptycho.params.cfg, config)` before inspecting `config.backend` or importing backend-specific modules. This guarantees legacy subsystems observe synchronized parameters regardless of backend.
> - **Routing Guarantees**:
>   - When `config.backend == 'tensorflow'`, the dispatcher SHALL delegate to `ptycho.workflows.components` entry points without attempting PyTorch imports.
>   - When `config.backend == 'pytorch'`, the dispatcher SHALL delegate to `ptycho_torch.workflows.components` entry points and return the same `(amplitude, phase, results_dict)` structure expected by TensorFlow workflows.
> - **Torch Unavailability**: Selecting `'pytorch'` MUST raise an actionable `RuntimeError` if the PyTorch stack cannot be imported. The message SHALL include the phrases “PyTorch backend selected” and installation guidance (e.g., `pip install torch>=2.2`). Silent fallbacks to TensorFlow are prohibited (POLICY-001).
> - **Result Metadata**: Dispatchers MUST annotate the returned `results_dict` with `results['backend'] = config.backend` to aid downstream logging and regression harnesses.
> - **Persistence Parity**: Backends MUST persist archives in formats compatible with their load paths. Cross-backend artifact loading is OPTIONAL but, when unsupported, the dispatcher MUST raise a descriptive error (referenced in `tests/torch/test_model_manager.py:238-372`).
> - **Validation Errors**: Dispatcher MUST raise `ValueError` if `config.backend` is not one of the supported literals, guiding callers to correct usage.
> - **Inference Symmetry**: The same guarantees apply to `load_inference_bundle_with_backend()` to ensure train/save/load/infer workflows remain symmetric.

## Documentation Alignment Checklist (Phase E3.B Inputs)
- `docs/workflows/pytorch.md` — add Section “Backend Selection in Ptychodus Integration” summarizing §4.8 promises, referencing selector commands (`tests/torch/test_backend_selection.py`) and error handling.
- `docs/architecture.md` — annotate component diagram with backend selector node linking TensorFlow and PyTorch stacks.
- `CLAUDE.md` — inject CONFIG-001 reminder under §4.1 emphasising backend selector must hydrate `params.cfg` before PyTorch imports.
- `README.md` (optional) — short “Dual-backend architecture” blurb pointing to backend selector docs.

## Open Questions & Follow-ups
1. **Cross-backend archive loading** — Current implementation flags compatibility checks (`tests/torch/test_model_manager.py:238-372`). Need governance confirmation whether spec should require or forbid cross-backend resume; propose deferring as POLICY-002 note.
2. **Default backend overrides** — Inventory recommends exposing CLI/UI toggle; requires confirmation from Ptychodus maintainers before codifying in spec (Phase D handoff).
3. **CI guardrails** — TEST-PYTORCH-001 Phase D3 needs markers/timeout documentation; will be addressed in `phase_e3_handoff.md`.

## Plan Integration
- **Phase E3.C1**: This document satisfies the “Draft spec amendments” staging requirement — ready for review prior to editing `specs/ptychodus_api_spec.md`. Recommend marking C1 `[P]` until governance sign-off, then `[x]` once spec PR text finalised.
- **Phase E3.C2**: POLICY-002 placeholder to be added after spec text approved; include cross-reference to this draft.
- **Phase E3.C3**: Governance review should confirm the above decisions align with `phase_e_integration.md` and `phase_f_torch_mandatory.md`.

## Next Supervisor Actions
1. Secure feedback on §4.8 draft (architecture + governance stakeholders).
2. Once approved, delegate doc/spec edits:
   - Update `specs/ptychodus_api_spec.md` with §4.8 text.
   - Refresh docs per checklist above.
   - Log docs/fix_plan.md Attempt #14 referencing this draft and subsequent edits.
3. Prepare `phase_e3_handoff.md` capturing CI/test ownership for TEST-PYTORCH-001 Phase D3.

