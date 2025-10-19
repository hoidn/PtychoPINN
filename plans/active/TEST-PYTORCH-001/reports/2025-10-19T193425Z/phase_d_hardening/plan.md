# Phase D — Regression Hardening & Documentation (TEST-PYTORCH-001)

## Context
- **Initiative:** TEST-PYTORCH-001 — Author PyTorch integration workflow regression
- **Phase Goal:** Lock in the PyTorch regression test with durable documentation, runtime telemetry, and CI-ready guidance so the workflow remains green after Phase C modernization.
- **Dependencies:**  
  - `plans/active/TEST-PYTORCH-001/implementation.md` (Phase D checklist)  
  - Phase C3 summary & artifact audit (`plans/active/TEST-PYTORCH-001/reports/2025-10-19T130900Z/phase_c_modernization/`)  
  - PyTorch workflow guide (`docs/workflows/pytorch.md` §§5–8)  
  - POLICY-001 (PyTorch mandatory) & FORMAT-001 (NPZ transpose guard) in `docs/findings.md`
- **Artifact Hub:** `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/`
  - Store runtime reports, CI notes, updated summaries, and pytest logs for this phase here.

---

### Phase D1 — Runtime & Resource Profile
Goal: Capture authoritative runtime, hardware, and resource notes for the PyTorch integration pytest selector so CI maintainers know expected performance envelopes.

Prereqs:
- Phase C GREEN log available (`pytest_modernization_green.log`, rerun log from C3).
- Access to hardware specs for the host running the test (CPU model, RAM).

Exit Criteria:
- Runtime + environment profile documented under this phase directory.
- Any variability considerations (e.g., tmp_path cleanup, dataset size) recorded for future tuning.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1.A | Aggregate runtime evidence | [x] | Parse C2 GREEN (`2025-10-19T122449Z`) and C3 rerun logs to confirm typical duration (≈36s). Record stats in `runtime_profile.md` with citation snippets. **COMPLETE:** Aggregated C2 (35.86s), C3 (35.98s), D1 (35.92s) logs with 0.17% variance. Evidence in `runtime_profile.md` §1. |
| D1.B | Document environment & resource context | [x] | Capture `python -m ptycho_torch.env_report` (if available) or manual `python -V`, `pip show torch`, `lscpu` snippets. Note CPU cores, RAM, and whether GPU disabled. Store outputs in `runtime_profile.md` and `env_snapshot.txt`. **COMPLETE:** Captured Python 3.11.13, PyTorch 2.8.0+cu128, Lightning 2.5.5, Ryzen 9 5950X (32 CPUs), 128GB RAM via command sequence. Evidence in `env_snapshot.txt` (3.6 KB). |
| D1.C | Identify performance guardrails | [x] | Summarize acceptable runtime variance (e.g., ≤90s on CI CPU nodes). Reference `docs/workflows/pytorch.md` §§6–8 for device guidance. **COMPLETE:** Defined four guardrail thresholds (≤90s CI max, 60s warning, 36s±5s baseline, 20s minimum) with variance analysis (CPU freq, I/O, dataset). Evidence in `runtime_profile.md` §3. |

---

### Phase D2 — Documentation & Ledger Alignment
Goal: Ensure project documentation and fix ledger reflect the pytest modernization, artifact discipline, and runtime expectations.

Prereqs:
- D1 runtime profile drafted (so documentation can cite real numbers).
- Access to `docs/fix_plan.md`, `docs/workflows/pytorch.md`, and Phase C artifacts.

Exit Criteria:
- fix_plan attempts updated with Phase D evidence.
- Workflow docs include PyTorch regression selector + runtime guidance.
- Implementation plan Phase D table links to this phase’s artifacts.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D2.A | Update implementation plan | [ ] | Edit `plans/active/TEST-PYTORCH-001/implementation.md` Phase D rows to reference this plan + runtime profile, marking D1 tasks `[P]`/`[x]` as progress occurs. |
| D2.B | Append fix_plan attempt entry | [ ] | Record Phase D work in `docs/fix_plan.md` ([TEST-PYTORCH-001] section) with artifact paths (`runtime_profile.md`, `ci_notes.md`). |
| D2.C | Refresh workflow documentation | [ ] | Update `docs/workflows/pytorch.md` testing section (≈§§7–8) with the pytest selector, runtime budget, and artifact expectations. Note POLICY-001 & FORMAT-001 compliance. |

---

### Phase D3 — CI Integration & Follow-up Gates
Goal: Define how the new pytest selector integrates into CI (markers/skip rules) and capture any follow-up work needed for automation.

Prereqs:
- D2 documentation updated so CI changes reference authoritative guidance.
- Knowledge of existing CI configuration (see `docs/development/TEST_SUITE_INDEX.md` and any `.github/workflows/` configs if relevant).

Exit Criteria:
- CI integration strategy documented with clear next actions (if automation required).
- Any follow-up fix-plan items authored for work beyond this initiative’s scope.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D3.A | Assess existing CI runners | [ ] | Review `docs/development/TEST_SUITE_INDEX.md` and CI configs to determine where to hook the selector (likely an integration/torch job). Summarize findings in `ci_notes.md`. |
| D3.B | Define execution strategy | [ ] | Decide whether to tag the test with `@pytest.mark.integration` or similar. Document expected command (e.g., `pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`). Record skip conditions (requires torch, dataset). |
| D3.C | Capture follow-up actions | [ ] | If CI changes exceed current loop, add new fix plan entry or TODO in `ci_notes.md` with owners + rationale. Ensure docs/fix_plan.md references any new tickets. |

---

## Artifact Checklist
- `runtime_profile.md` — Runtime + environment notes (Phase D1)
- `env_snapshot.txt` — Raw command outputs (Phase D1)
- `ci_notes.md` — CI strategy, markers, follow-up (Phase D3)
- `summary.md` — Phase completion narrative per loop
- Pytest logs (`pytest_modernization_phase_d.log` or similar) if reruns executed during this phase

## References
- `plans/active/TEST-PYTORCH-001/implementation.md`
- `plans/active/TEST-PYTORCH-001/reports/2025-10-19T130900Z/phase_c_modernization/`
- `docs/workflows/pytorch.md`
- `docs/findings.md#POLICY-001`
- `docs/TESTING_GUIDE.md` (integration tier expectations)
