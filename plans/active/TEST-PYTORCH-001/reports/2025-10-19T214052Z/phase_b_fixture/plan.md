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
| B1.A | Capture canonical dataset profile | [x] | ✔ `reports/2025-10-19T215300Z/phase_b_fixture/dataset_probe.txt` + `fixture_scope.md` document `(H,W,N)=(64,64,1087)` orientation, dtype violations, and coordinate ranges (DATA-001 + FORMAT-001 references). |
| B1.B | Measure runtime sensitivity to CLI knobs | [x] | ✔ `reports/2025-10-19T215300Z/phase_b_fixture/logs/train_ep2_n64.log` + `train_ep1_n16.log` capture CPU-only dry runs (21.91s / 17.11s) with CLI overrides noted in `fixture_scope.md`. |
| B1.C | Define fixture acceptance criteria | [x] | ✔ `reports/2025-10-19T215300Z/phase_b_fixture/fixture_scope.md` §3 lists nine acceptance criteria (n_subset=64, dtype downcasts, runtime <45s, checksum metadata) derived from telemetry. |

---

### Phase B2 — Fixture Construction & Validation (TDD)
Goal: Build a reproducible fixture generator that emits a canonical NPZ + probe/object pair satisfying the acceptance criteria.
Prereqs: Phase B1 scope documented; desired runtime + dataset shape agreed.
Exit Criteria: Fixture generator script committed, regression tests covering fixture shape/dtype created, and artifact stored under `tests/fixtures/pytorch_integration/`.

| ID | Task Description | State | How/Why & Guidance (including API / document / artifact / source file references) |
| --- | --- | --- | --- |
| B2.A | Author fixture generator spec + stub | [x] | ✔ `reports/2025-10-19T220500Z/phase_b_fixture/generator_design.md` documents inputs/operations/outputs (axis reorder, dtype downcast, deterministic subset) and `scripts/tools/make_pytorch_integration_fixture.py` now ships argparse stub + `NotImplementedError` placeholder per TDD plan. |
| B2.B | TDD RED — fixture validation test | [x] | ✔ `tests/torch/test_fixture_pytorch_integration.py` authored with seven RED tests (shapes, dtypes, metadata, RawData/PyTorch smoke). RED log captured at `reports/2025-10-19T220500Z/phase_b_fixture/pytest_fixture_red.log` (all skipped pending fixture generation). |
| B2.C | Implement generator + make test GREEN | [x] | ✅ 2025-10-19 — Fixture generator implemented per design §4. Artifact bundle at `reports/2025-10-19T225900Z/phase_b_fixture/{fixture_generation.log,pytest_fixture_green.log}` shows `generate_fixture()` emitting dual-key NPZ + JSON metadata with stratified sampling (94.8% / 96.8% coverage). Core contract tests now GREEN (5/5) with remaining smoke failures deferred to B3. |
| B2.D | Document fixture metadata | [x] | ✅ 2025-10-19 — Documentation captured in `reports/2025-10-19T225900Z/phase_b_fixture/fixture_notes.md` and `summary.md`, including regeneration command, checksum `6c2fbea0dcadd950385a54383e6f5f731282156d19ca4634a5a19ba3d1a5899c`, storage guidance, and follow-up items for Phase B3. |

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
