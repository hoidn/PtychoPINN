# Phase D4 Planning Summary — 2025-10-17T111014Z

## Question
How do we translate the completed persistence work (D3) into repeatable regression tests and prepare the PyTorch backend for TEST-PYTORCH-001 activation?

## Key Decisions
- Created dedicated plan `phase_d4_regression.md` with phased checklist (Alignment → Red tests → Green tests) and explicit artifact naming.
- Modeled tasks after canonical plan Phase 6 + configs from `plans/pytorch_integration_test_plan.md`; mapped CONFIG-001 finding as a standing guardrail.
- Established coordination gate: TEST-PYTORCH-001 plan gets activated once D4.B red evidence exists and torch-optional selectors are stable.
- Documented authoritative selectors: persistence (`tests/torch/test_model_manager.py`), orchestration (`tests/torch/test_workflows_components.py`), future integration test harness.

## Next Actions for Engineering
1. Execute D4.A1–A3 (alignment + selector map) before touching code.
2. Author failing regression tests (D4.B) capturing persistence + orchestration gaps.
3. Implement fixes under D4.C and prepare handoff package for TEST-PYTORCH-001.

## References
- specs/ptychodus_api_spec.md §4.5–4.6 (persistence contract)
- plans/ptychodus_pytorch_integration_plan.md Phase 6
- plans/pytorch_integration_test_plan.md (integration test roadmap)
- docs/TESTING_GUIDE.md (pytest selectors & policy)

Artifacts captured per Phase D4 plan; update docs/fix_plan.md Attempts History with this timestamp.
