# Phase B.B5 — Config Bridge Parity Green Plan

## Context
- Initiative: INTEGRATE-PYTORCH-001 (Configuration & Legacy Bridge Alignment)
- Phase Goal: Flip the red-phase parity matrix green by implementing the remaining adapter logic and running the expanded tests end-to-end.
- Dependencies: `specs/ptychodus_api_spec.md §5.1-5.3` (field contract), `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/{field_matrix.md,summary.md}` (coverage + priority ranking), `ptycho_torch/config_bridge.py` (current adapter), `tests/torch/test_config_bridge.py` (red-phase parity suite), `ptycho/config/config.py` (KEY_MAPPINGS + defaults), and finding CONFIG-001 (update params.cfg first).
- Current State: Phase A is complete (Attempt #15) — parity selectors execute without torch; adapter P0 fixes (probe_mask translation, nphotons override enforcement, path normalization) landed in Attempt #17. Parity suite remains red because `TestConfigBridgeParity` still inherits `unittest.TestCase`, so parameterized cases raise `TypeError` until the harness is converted to pytest style (blocking B2/B4).
- Artifact storage: Capture design notes, diffs, and pytest logs under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/` (e.g., `implementation_notes.md`, `adapter_diff.md`, `pytest_green.log`). Reference each artifact from docs/fix_plan.md attempts.

---

### Phase A — Enable Test Harness Without Hard PyTorch Dependency
Goal: Allow the parity tests to execute (not skip) in environments lacking GPU-enabled torch by providing a safe import fallback.
Prereqs: Review `tests/conftest.py` skip logic and `ptycho_torch/config_params.py` import path.
Exit Criteria: Targeted selector `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -k N-direct -v` runs without SKIP when torch is missing, using documented fallback behaviour.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Audit skip mechanics | [x] | Completed 2025-10-17 — see `reports/2025-10-17T050930Z/implementation_notes.md` (Phase A.A1) for skip logic summary. |
| A2 | Introduce optional torch shim | [x] | Completed 2025-10-17 — guarded import + `TORCH_AVAILABLE` flag added to `ptycho_torch/config_params.py`; details in `implementation_notes.md` Phase A.A2. |
| A3 | Adjust pytest gating | [x] | Completed 2025-10-17 — `tests/conftest.py` now exempts config bridge tests when torch missing (Phase A.A3 notes). |
| A4 | Add fallback verification | [x] | Completed 2025-10-17 — pytest log `reports/2025-10-17T050930Z/pytest_phaseA.log` shows execution without skip; fallback verified. |

---

### Phase B — Resolve P0 Blockers (Probe Mask & nphotons)
Goal: Implement adapter logic so highest-risk parity assertions pass.
Prereqs: Phase A complete (tests runnable), review `summary.md` P0 section and spec defaults.
Exit Criteria: `pytest tests/torch/test_config_bridge.py -k "probe_mask or nphotons" -v` passes (no skips/xfails).

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B0 | Refactor parity harness to pytest style | [ ] | Convert `tests/torch/test_config_bridge.py::TestConfigBridgeParity` into pytest-style tests (drop `unittest.TestCase`, use fixtures) so parameterized cases run. Required before B2/B4; capture log under new reports directory. |
| B1 | Implement probe_mask conversion | [x] | Completed 2025-10-17 — adapter logic landed in `reports/2025-10-17T045936Z/adapter_diff.md`; defaults verified in `bridge_probe_mask_check.md`. |
| B2 | Extend tests for probe_mask | [ ] | After B0 lands, re-enable parameterized parity cases covering default (`None`→False) and explicit override paths per `field_matrix.md`. Use pytest-style assertions; capture results in new pytest log. |
| B3 | Enforce nphotons override | [x] | Completed 2025-10-17 — adapter now raises ValueError when overrides missing; see `adapter_diff.md` + Attempt #17 summary. |
| B4 | Tighten default divergence test | [ ] | Once pytest harness executes, assert ValueError message text for missing overrides and green-case success; store red vs green logs as planned. |

---

### Phase C — Address P1 High-Priority Cases (n_subsample & Error Handling)
Goal: Ensure semantic divergences and validation messaging meet spec expectations.
Prereqs: Phase B green.
Exit Criteria: `pytest tests/torch/test_config_bridge.py -k "n_subsample or error_handling" -v` passes and error messages match plan guidance.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Clarify n_subsample semantics | [ ] | Encode adapter behaviour so PyTorch `DataConfig.n_subsample` does **not** override TensorFlow training count; require explicit override. Add docstring comment referencing `field_matrix.md` row. |
| C2 | Update tests for n_subsample | [ ] | Add parameterized case verifying override vs. default interplay, marking as FAIL until behaviour implemented. |
| C3 | Review error messages | [ ] | Ensure ValueErrors raised by adapter include actionable guidance (e.g., "Provide train_data_file override via overrides['train_data_file']"). Update tests to assert message fragments recorded in summary.md P1 section. |

---

### Phase D — params.cfg Baseline Comparison & Override Matrix
Goal: Finish deferred Phase C tasks from the red plan and document override expectations.
Prereqs: Phases A-C complete.
Exit Criteria: New test `test_params_cfg_matches_baseline` passes; override matrix documented.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Implement baseline comparison test | [ ] | Add `test_params_cfg_matches_baseline` leveraging `baseline_params.json` to assert adapter + overrides reproduce canonical params.cfg. Place helper loader in test file. |
| D2 | Capture override matrix | [ ] | Create `override_matrix.md` summarizing required overrides, default behaviours, and failure modes. Link to docs/fix_plan.md attempt. |
| D3 | Validate override warnings | [ ] | Extend tests to assert missing overrides raise warnings/errors with guidance; reference override matrix. |

---

### Phase E — Final Verification & Reporting
Goal: Finalize artifacts, log outcomes, and update governance documents.
Prereqs: Phases A-D complete.
Exit Criteria: All parity tests green (or justified xfails), artifacts recorded, ledger updated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| E1 | Run full parity selector | [ ] | Execute `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/pytest_green.log`. Annotate pass/fail summary. |
| E2 | Update implementation notes | [ ] | Record decisions, follow-up risks, and outstanding questions in `implementation_notes.md`. Include environment details (torch present? fallback used?). |
| E3 | Refresh plan & ledger | [ ] | Mark B4 row complete in `plans/active/INTEGRATE-PYTORCH-001/implementation.md`, update B5 guidance, and log Attempt #14 in docs/fix_plan.md linking to this plan + artifacts. |

---

## Verification Checklist
- [x] Tests no longer skipped when torch absent; fallback strategy documented (see pytest_phaseA.log).
- [ ] P0 probe_mask/nphotons tests green with explicit overrides enforced.
- [ ] params.cfg baseline comparison test added and passing, with override matrix recorded.
- [ ] `pytest_green.log` stored under the new timestamped directory capturing the first green run.

