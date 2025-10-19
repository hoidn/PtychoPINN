# Phase B Fixture Planning Summary — 2025-10-19T214052Z

## Objective
Document actionable guidance for TEST-PYTORCH-001 Phase B so engineers can construct a lightweight PyTorch integration fixture and deterministic configuration without sacrificing coverage.

## Key Decisions
- Target a <45s CPU runtime envelope once the fixture replaces `Run1084_recon3_postPC_shrunk_3.npz` in the regression.
- Require fixture datasets to adhere strictly to `specs/data_contracts.md` (diffraction float32, complex64 object/probe) and maintain at least two scan groups to exercise grouping code paths.
- Use a new generator script (`scripts/tools/make_pytorch_integration_fixture.py`) developed via TDD alongside pytest coverage to guarantee reproducibility.
- Treat documentation updates (implementation plan, workflow guide, runtime profile addendum) as part of Phase B completion to keep guardrails in sync.

## Next Actions (per plan.md)
1. Execute Phase B1 scope work: capture dataset stats, measure runtime sensitivity for 1 epoch/16 images, and lock acceptance criteria in `fixture_scope.md`.
2. Draft generator design + failing pytest in Phase B2 to drive fixture creation before touching production code.
3. After fixture is green, rewire integration test and refresh documentation (Phase B3).

## Artifacts
- `plan.md` — Detailed Phase B roadmap with checklist IDs B1.A–B3.C.
- (Upcoming) `fixture_scope.md`, `generator_design.md`, pytest logs, and fixture notes as engineers execute the plan.

## Dependencies Reviewed
- `docs/workflows/pytorch.md` §§4–8 — confirms CLI + CONFIG-001 requirements for backend parity.
- `specs/data_contracts.md` §1 — authoritative NPZ contract for new fixture.
- `docs/findings.md#POLICY-001` and `#FORMAT-001` — ensure fixture generation respects mandatory torch dependency and legacy transpose guardrails.

## Risks / Open Questions
- Need to verify that reducing `n_images` to ~16 still hits probe/object update paths; if not, adjust acceptance criteria.
- Dataset generator must preserve coordinate scaling; plan references downsampling script for guidance but implementation detail remains TBD.

## Supervisor Notes
Documented plan has been cross-referenced from `plans/active/TEST-PYTORCH-001/implementation.md` (to update) and will steer upcoming loops focused on Phase B.
