Summary: Capture the full configuration parity test matrix (red phase) for the PyTorch config bridge.
Mode: TDD
Focus: INTEGRATE-PYTORCH-001 — Prepare for PyTorch Backend Integration with Ptychodus
Branch: feature/torchapi
Mapped tests: none — author TestConfigBridgeParity selectors per plan
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/{parity_test_plan.md,field_matrix.md,fixtures.py,testcase_design.md,baseline_params.json,pytest_red.log,summary.md}
Do Now: INTEGRATE-PYTORCH-001 Attempt #12 — execute Phase B.B4 plan by deriving the field matrix, adding the new parameterized parity tests, then run `pytest tests/torch/test_config_bridge.py -k parity -v 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/pytest_red.log` to capture the red state.
If Blocked: Record the blocker (stack trace, missing field mapping, torch import skip) in `summary.md`, keep the pytest log, and push the partially filled `field_matrix.md` so we can unblock during the next supervisor loop.
Priorities & Rationale:
- docs/fix_plan.md:63 — Attempt #12 mandates parity-test planning follow-up.
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:49 — Phase B.B4 instructions now point at the new parity test plan.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/parity_test_plan.md:1 — Defines the deliverables and checklists you must satisfy this loop.
- specs/ptychodus_api_spec.md:213 — Authoritative list of Model/Training/Inference fields to cover.
- tests/torch/test_config_bridge.py:1 — Extend the existing harness instead of inventing a new file so the suite stays discoverable.
- docs/TESTING_GUIDE.md:1 — Follow the documented TDD workflow (write failing tests first, keep scope tight).
How-To Map:
- Read `parity_test_plan.md` and create `field_matrix.md` plus `fixtures.py` in the same report directory capturing explicit values for every spec-required field (avoid defaults when possible).
- Extend `tests/torch/test_config_bridge.py` with a new `TestConfigBridgeParity` class that uses `pytest.mark.parametrize` to assert the adapter returns the expected TensorFlow dataclass values; use `pytest.param(..., marks=pytest.mark.xfail(...))` for gaps noted in the plan.
- Add helper utilities (if needed) inside the test file or under `plans/.../fixtures.py`; do not modify production modules yet.
- After authoring the tests, run `pytest tests/torch/test_config_bridge.py -k parity -v` once, teeing output to `pytest_red.log`; expect failures/xfails while PyTorch remains unavailable, and ensure the skip/xfail reasoning is explicit in assertions.
- Diff the adapter-driven `params.cfg` against a TensorFlow baseline using the helper functions described in the plan; store the baseline snapshot as `baseline_params.json` and reference it from the tests (e.g., by loading the JSON for comparisons).
Pitfalls To Avoid:
- Do not touch `ptycho_torch/config_bridge.py` in this loop; we are only expanding tests and fixtures.
- Keep PyTorch import skips localized (reuse existing `pytest.importorskip('torch')` in the test file) so CI without torch stays green.
- Avoid relying on default values in tests; explicit assignments prevent silent parity gaps.
- Do not over-constrain PyTorch-only fields — focus on spec-required ones and mark PyTorch-only parameters as expected skips for now.
- Ensure params.cfg is snapshotted/restored around each test (reuse existing setup/teardown patterns).
- Limit pytest execution to the single parity selector; no full-suite runs.
- Maintain artifact paths exactly as listed; add any new files inside the timestamped directory.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/parity_test_plan.md:1
- ptycho/config/config.py:70
- ptycho/config/config.py:231
- specs/ptychodus_api_spec.md:213
- tests/torch/test_config_bridge.py:1
- docs/debugging/QUICK_REFERENCE_PARAMS.md:1
Next Up: Once the red matrix is in place, we will iterate on the adapter to flip failing cases green and broaden coverage to PyTorch-only fields.
