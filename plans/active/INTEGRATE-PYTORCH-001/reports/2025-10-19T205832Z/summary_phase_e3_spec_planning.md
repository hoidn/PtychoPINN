# Phase E3 Spec Planning Summary — 2025-10-19T212700Z

**Focus:** INTEGRATE-PYTORCH-001-STUBS — Phase E3 backend selection documentation/spec sync  
**Action Type:** Planning (Docs mode)

## Loop Outcomes
- Drafted `phase_e3_spec_patch.md` outlining proposed §4.8 “Backend Selection & Dispatch” requirements for `specs/ptychodus_api_spec.md`.
- Captured current dispatcher behaviour with code anchors (`ptycho/workflows/backend_selector.py:121-165`) and linked TDD expectations (`tests/torch/test_backend_selection.py:59-170`).
- Enumerated documentation alignment tasks (pytorch workflow guide, architecture diagram, CLAUDE.md, README.md) to feed Phase E3.B updates.
- Recorded open governance questions (cross-backend archives, UI toggle policy) for follow-up before spec publication.

## Key References
- `specs/ptychodus_api_spec.md` (existing §4.1–§4.7 context)
- `ptycho/config/config.py:110`, `ptycho/config/config.py:142` (backend field defaults)
- `ptycho/workflows/backend_selector.py` (dispatcher implementation)
- `tests/torch/test_backend_selection.py` (expected behaviour contract)
- Findings: POLICY-001, FORMAT-001

## Next Steps
1. Socialize §4.8 draft with architecture/governance stakeholders for approval.
2. Upon approval, delegate spec + doc edits (Phase E3.B/C execution loop).
3. Author `phase_e3_handoff.md` capturing CI/runtime guidance for TEST-PYTORCH-001 Phase D3.

