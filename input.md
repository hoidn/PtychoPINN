Summary: Outline the Phase B config-bridge design so PyTorch can ingest Ptychodus dataclasses without regressing params.cfg parity.
Mode: Docs
Focus: INTEGRATE-PYTORCH-001 — Prepare for PyTorch Backend Integration with Ptychodus
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T021100Z/{config_bridge.md,notes.md}
Do Now: INTEGRATE-PYTORCH-001 — Phase B.B1 config bridge design; author `config_bridge.md` capturing the dataclass→singleton→params flow (no tests yet).
If Blocked: Capture raw field inventory with `python - <<'PY' ...` dumping `asdict(TrainingConfig(...))` into the same report folder, then stop and flag in docs/fix_plan.md Attempts History.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:3-152 — defines the mandatory dataclass→params handshake Ptychodus relies on.
- ptycho/config/config.py:221-279 — shows the exact KEY_MAPPINGS and `update_legacy_dict` contract the bridge must honor.
- ptycho_torch/config_params.py:1-110 — documents the singleton defaults that need mapping to modern configs.
- ptycho_torch/train.py:193-247 — illustrates current singleton seeding that will be superseded by the new adapter.
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:34-82 — Phase B tasks and exit criteria we are servicing this loop.
How-To Map:
- `mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T021100Z` before writing artifacts.
- Create `config_bridge.md` with sections: Context, Source Inputs (dataclasses), Target Outputs (`params.cfg` + PyTorch singletons), Field Mapping Table, Call Order, Open Questions.
- Populate the mapping table with columns: Dataclass Field, Legacy Key, PyTorch Singleton Key/Notes, Pending Work; source values from the spec + `ptycho/config/config.py` KEY_MAPPINGS + current singleton defaults.
- Document invocation order pseudocode showing when to call `update_legacy_dict` relative to PyTorch dataset/model construction; highlight required guardrails (params initialization before imports, per docs/debugging/QUICK_REFERENCE_PARAMS.md).
- Add `notes.md` summarizing unresolved items (e.g., fields missing in PyTorch today, questions for B2 test design) so we can queue them in the next loop.
Pitfalls To Avoid:
- Do not invent new legacy keys; stick to ones defined in `KEY_MAPPINGS` unless the spec calls for additions.
- Avoid editing production code or tests in this loop.
- Keep the artifact timestamped path exact; no writing outside `plans/active/INTEGRATE-PYTORCH-001`.
- Do not guess pytest selectors; this is a design-only pass.
- Note every open question with owner/action instead of leaving TODOs implicit.
- Respect the legacy params init order: design must call `update_legacy_dict` before touching raw_data/loader.
- Keep docs ASCII; avoid special formatting that complicates diff review.
- Cross-check probe/object flags (`object.big`, `probe.big`, etc.) so naming differences between dataclasses and singletons are explicit.
- Flag any singleton defaults that conflict with spec expectations (e.g., loss function naming).
- Capture references inline using repo-relative paths for traceability.
Pointers:
- specs/ptychodus_api_spec.md:144-151
- ptycho/config/config.py:231-269
- ptycho_torch/config_params.py:52-79
- docs/DEVELOPER_GUIDE.md:1-120
- docs/debugging/QUICK_REFERENCE_PARAMS.md:1-80
Next Up: 1) Draft failing parity test scaffold (Phase B.B2) once design approved; 2) Begin adapter implementation guided by the new document.
