## Context
- Initiative: TEST-PYTORCH-001 — PyTorch integration regression
- Phase Goal: Deliver deterministic, lightweight fixtures + configuration overrides so the PyTorch train→infer regression remains <45s on CPU while exercising the full pipeline.
- Dependencies:
  - `plans/active/TEST-PYTORCH-001/implementation.md` Phase B table (fixture minimization & deterministic config).
  - `specs/data_contracts.md` §1 (canonical NPZ contract — amplitude `diffraction`, complex64 `Y`/`objectGuess`).
  - `docs/workflows/pytorch.md` §§4–8 (RawData expectations, CLI knobs, CONFIG-001 enforcement).
  - `docs/findings.md` entries POLICY-001 (torch required) & FORMAT-001 (legacy `(H,W,N)` auto-transpose heuristic).
  - `plans/pytorch_integration_test_plan.md` (original fixture intent; now needs modernization with deterministic dataset).

---

### Phase B1 — Fixture Requirements & Runtime Envelope
Goal: Quantify current dataset behaviour, define target size/runtime budget, and lock telemetry to guide fixture construction.
Prereqs: Phase A baseline artifacts (2025-10-19T115303Z) available; PyTorch regression currently green with canonical dataset.
Exit Criteria: Written analysis capturing dataset statistics, config knob sensitivities, and target fixture specs (size, dtype, coordinate coverage).

| ID | Task Description | State | How/Why & Guidance (including API / document / artifact / source file references) |
| --- | --- | --- | --- |
| B1.A | Capture canonical dataset profile | [ ] | Run `python plans/active/TEST-PYTORCH-001/scripts/profile_fixture_dataset.py` (**new script, see B2.A**) or ad-hoc notebook to log `diffraction` orientation, dtype, min/max, and counts (N,H,W). Store findings in `reports/<TS>/phase_b_fixture/fixture_scope.md`. Cross-check against data contract (`specs/data_contracts.md` §1) and FORMAT-001 auto-transpose notes. |
| B1.B | Measure runtime sensitivity to CLI knobs | [ ] | Execute two targeted dry runs of the PyTorch regression (training only) varying `--max_epochs` (1 vs 2) and `--n_images` (16 vs 64) with `CUDA_VISIBLE_DEVICES=""` to quantify impact on elapsed time. Record commands, timings, and resulting artifact sizes in `fixture_scope.md`. Use existing helper `_run_pytorch_workflow` in `tests/torch/test_integration_workflow_torch.py` as baseline and log outputs under `reports/<TS>/phase_b_fixture/logs/`. |
| B1.C | Define fixture acceptance criteria | [ ] | Based on B1.A/B1.B, draft bullet list in `fixture_scope.md` covering: desired `n_images`, target runtime (<45s CPU), required coordinate spread (ensure >1 scan group), dtype conversions (diffraction→float32, complex arrays→complex64), and compatibility with CONFIG-001 bridging. |

---

### Phase B2 — Fixture Construction & Validation (TDD)
Goal: Build a reproducible fixture generator that emits a canonical NPZ + probe/object pair satisfying the acceptance criteria.
Prereqs: Phase B1 scope documented; desired runtime + dataset shape agreed.
Exit Criteria: Fixture generator script committed, regression tests covering fixture shape/dtype created, and artifact stored under `tests/fixtures/pytorch_integration/`.

| ID | Task Description | State | How/Why & Guidance (including API / document / artifact / source file references) |
| --- | --- | --- | --- |
| B2.A | Author fixture generator spec + stub | [ ] | Create `plans/active/TEST-PYTORCH-001/reports/<TS>/phase_b_fixture/generator_design.md` outlining inputs (source dataset path), operations (axis reorder, dtype downcast, subsample strategy), and outputs. Include pseudocode referencing `numpy` ops and `specs/data_contracts.md` dtype requirements. Stub new script `scripts/tools/make_pytorch_integration_fixture.py` with CLI placeholders (no implementation yet) to enable TDD. |
| B2.B | TDD RED — fixture validation test | [ ] | Add pytest module `tests/torch/test_fixture_pytorch_integration.py` with initial failing test `test_fixture_outputs_match_contract` asserting: `diffraction.shape == (n_subset, H, W)`, dtype is `float32`, `Y/objectGuess/probeGuess` are `complex64`, and coordinates align with subset size. Use Phase B1 acceptance criteria for expected values. Record red log under `reports/<TS>/phase_b_fixture/pytest_fixture_red.log`. |
| B2.C | Implement generator + make test GREEN | [ ] | Implement `make_pytorch_integration_fixture.py` to read canonical dataset, apply deterministic slicing (e.g., first `n_subset` scan positions), convert dtypes, reorder axes to `(N,H,W)`, and write fixture to `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`. Update test to import fixture (without heavy I/O in tests). Capture green log `pytest_fixture_green.log`. Ensure artifact references metadata (source dataset, commit SHA) inside generator output via JSON sidecar if feasible. |
| B2.D | Document fixture metadata | [ ] | Write `fixture_notes.md` summarizing fixture dimensions, runtime savings, and commands to regenerate. Include hashes (`sha256sum`) and storage instructions (under `tests/fixtures/pytorch_integration/`). Reference POLICY-001 to reaffirm torch dependency. |

---

### Phase B3 — Regression Wiring & Determinism Guardrails
Goal: Integrate the new fixture into the PyTorch integration test, update configuration overrides, and document deterministic guardrails.
Prereqs: Fixture generated and validated (Phase B2 complete).
Exit Criteria: Integration test consumes fixture, runtime drops within target envelope, deterministic seeds documented, and plan/ledger updated.

| ID | Task Description | State | How/Why & Guidance (including API / document / artifact / source file references) |
| --- | --- | --- | --- |
| B3.A | Update integration test to use fixture | [ ] | Modify `tests/torch/test_integration_workflow_torch.py` `data_file` fixture to reference `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`. Adjust `_run_pytorch_workflow` CLI overrides (e.g., `--max_epochs=1`, `--n_images=<fixture_n>`). Ensure `update_legacy_dict` remains invoked via CLI by citing `docs/workflows/pytorch.md` §12. |
| B3.B | Validate runtime + determinism | [ ] | Run targeted pytest selector with new fixture, capture runtime log under `reports/<TS>/phase_b_fixture/pytest_fixture_integration.log`, and update statistics in `runtime_profile.md` (Phase D1) or append addendum noting new baseline. Confirm run time <45s CPU and variance ≤10%. |
| B3.C | Update documentation & ledger | [ ] | Refresh `plans/active/TEST-PYTORCH-001/implementation.md` B1–B3 rows with artifact links, append docs/fix_plan Attempt summarizing fixture integration, and update `docs/workflows/pytorch.md` §11 (Regression Test & Runtime Expectations) with new dataset/timeout info. |

---

## Deliverables & Artifact Discipline
- Use ISO timestamps per loop (e.g., `2025-10-19T214052Z`) under `plans/active/TEST-PYTORCH-001/reports/<TS>/phase_b_fixture/`.
- Required artifacts for each loop: `{plan_or_scope.md, summary.md, pytest_<tag>.log}` plus generator/test sources when implemented.
- Update `docs/fix_plan.md` Attempts history after every loop, citing checklist IDs (e.g., `B1.A`).

## Decision Rules
- If runtime with existing dataset already meets <45s but fixture work jeopardizes coverage (e.g., removing necessary scan diversity), escalate via new docs/fix_plan entry before dropping Phase B.
- Fixture must preserve at least two unique scan positions to exercise grouping logic; otherwise, adjust subset.
- Maintain dtype conversions exactly (`float32` diffraction, `complex64` complex arrays) to match DATA-001 contract. Deviations require spec update discussion.

## Next Supervisor Checkpoints
1. Verify B1 scope doc includes concrete numbers (N=1087 baseline, H=W=64, dtype float64 → target float32).
2. Ensure generator spec selects deterministic subset and documents seeding.
3. Confirm regression runtime log reflects new fixture improvements before closing Phase B.
