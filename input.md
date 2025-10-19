Summary: Draft fixture generator spec and RED test for PyTorch regression fixture.
Mode: TDD
Focus: [TEST-PYTORCH-001] Author PyTorch integration workflow regression — Phase B2 fixture generator TDD
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_fixture_pytorch_integration.py::test_fixture_outputs_match_contract -vv
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T220500Z/phase_b_fixture/{generator_design.md,pytest_fixture_red.log}

Do Now:
1. TEST-PYTORCH-001 B2.A @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Author `generator_design.md` summarizing fixture creation steps + CLI contract, then scaffold `scripts/tools/make_pytorch_integration_fixture.py` with argparse stub (no functional logic yet); tests: none.
2. TEST-PYTORCH-001 B2.B @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Write failing pytest `tests/torch/test_fixture_pytorch_integration.py::test_fixture_outputs_match_contract` asserting Phase B1 criteria (shape/dtype/subset count/checksum stub) and confirm RED via `pytest tests/torch/test_fixture_pytorch_integration.py::test_fixture_outputs_match_contract -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T220500Z/phase_b_fixture/pytest_fixture_red.log`.
3. TEST-PYTORCH-001 B2.B @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Log Attempt #38 in docs/fix_plan.md summarizing RED run + new artifacts before ending loop; tests: none.

If Blocked: Capture partial design + test skeleton in generator_design.md with TODO markers, record failure reason + outstanding dependencies in docs/fix_plan.md Attempt, leave B2 rows `[P]`.

Priorities & Rationale:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Phase B2 entrance criteria require design doc + RED test before implementation.
- specs/data_contracts.md:12 — Fixture must emit canonical `(N,H,W)` float32 diffraction + complex64 probe/object per contract.
- docs/workflows/pytorch.md:210 — Reinforces CONFIG-001 bridging + CLI knobs referenced in generator + tests.
- docs/findings.md#FORMAT-001 — Legacy `(H,W,N)` datasets demand explicit transpose; encode expectation in generator design/test.
- docs/TESTING_GUIDE.md:150 — Mandates TDD flow (RED before GREEN) for new regression coverage.

How-To Map:
- mkdir -p plans/active/TEST-PYTORCH-001/reports/2025-10-19T220500Z/phase_b_fixture
- Write generator design covering: source dataset path (`datasets/Run1084_recon3_postPC_shrunk_3.npz`), deterministic subset (first 64 positions), dtype downcasts, axis reorder, checksum metadata, CLI flags. Quote acceptance criteria from `fixture_scope.md` §3.
- Stub script `scripts/tools/make_pytorch_integration_fixture.py` with argparse options (`--source`, `--output`, `--subset-size`, `--metadata-out`) and placeholder `main()` raising `NotImplementedError`.
- Create pytest module `tests/torch/test_fixture_pytorch_integration.py` importing `numpy` and referencing acceptance criteria constants (n_subset=64, dtype expectations). Structure test to call `np.load` on fixture path (expect to fail) or call stubbed helper to emphasize missing implementation.
- Run targeted RED command: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_fixture_pytorch_integration.py::test_fixture_outputs_match_contract -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T220500Z/phase_b_fixture/pytest_fixture_red.log`.
- Update `fixture_scope.md` appendix with pointer to new design doc if you add additional acceptance notes (optional, docs-only).
- After tests, record Attempt #38 (Phase B2 kickoff) in docs/fix_plan.md referencing artifact path + failing assertion text.

Pitfalls To Avoid:
- Do not implement fixture logic yet—leave `NotImplementedError` to keep test RED.
- Keep pytest module pure pytest style (no unittest.TestCase mixes).
- Store all artifacts under the 2025-10-19T220500Z hub; no logs at repo root.
- Preserve TODO markers for unimplemented functionality instead of partial code.
- Remember to call out CONFIG-001 bridge requirement in design doc; missing that will cause future regressions.
- Avoid guessing checksum in test—leave placeholder assertion (e.g., `pytest.skip` or `assert False, "Fixture not generated"`).
- Do not run full pytest suite; only the targeted selector for RED validation.
- Ensure stub script imports remain minimal (argparse, pathlib, typing) so unused torch imports don't break without implementation.
- Verify new files follow ASCII + shebang conventions and include module docstring referencing plan.
- Update docs/fix_plan immediately after work; no deferred ledger updates.

Pointers:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/fixture_scope.md#3
- specs/data_contracts.md:12
- docs/workflows/pytorch.md:200
- docs/findings.md:18

Next Up: 1. TEST-PYTORCH-001 B2.C — implement generator + GREEN run once RED artifacts captured.
