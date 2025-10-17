# Phase B.B5 — Config Bridge Parity Green Plan

## Context
- Initiative: INTEGRATE-PYTORCH-001 (Configuration & Legacy Bridge Alignment)
- Phase Goal: Flip the red-phase parity matrix green by implementing the remaining adapter logic and running the expanded tests end-to-end.
- Dependencies: `specs/ptychodus_api_spec.md §5.1-5.3` (field contract), `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/{field_matrix.md,summary.md}` (coverage + priority ranking), `ptycho_torch/config_bridge.py` (current adapter), `tests/torch/test_config_bridge.py` (red-phase parity suite), `ptycho/config/config.py` (KEY_MAPPINGS + defaults), and finding CONFIG-001 (update params.cfg first).
- Current State: Attempt #13 captured the complete red matrix (38 spec fields) with all tests SKIPPED when PyTorch is absent. Critical blockers: (1) `probe_mask` Tensor→bool handling, (2) enforcing explicit `nphotons` overrides, (3) targeted pytest skip due to hard torch dependency in config singletons.
- Artifact storage: Capture design notes, diffs, and pytest logs under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/` (e.g., `implementation_notes.md`, `adapter_diff.md`, `pytest_green.log`). Reference each artifact from docs/fix_plan.md attempts.

---

### Phase A — Enable Test Harness Without Hard PyTorch Dependency
Goal: Allow the parity tests to execute (not skip) in environments lacking GPU-enabled torch by providing a safe import fallback.
Prereqs: Review `tests/conftest.py` skip logic and `ptycho_torch/config_params.py` import path.
Exit Criteria: Targeted selector `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -k N-direct -v` runs without SKIP when torch is missing, using documented fallback behaviour.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Audit skip mechanics | [ ] | Document how `tests/conftest.py` applies skip markers (file path contains `torch`, explicit `@pytest.mark.torch`). Capture notes in `implementation_notes.md` summarizing required changes. |
| A2 | Introduce optional torch shim | [ ] | Implement guarded import layer (e.g., new module `ptycho_torch/_torch_optional.py` or lazy attribute) so `config_params.py` can be imported without torch. Provide minimal `TensorStub` for typing or gate out features requiring real tensors. Record design + risks in notes. |
| A3 | Adjust pytest gating | [ ] | Update `tests/conftest.py` (or parity tests) so config-bridge selectors opt out of the blanket skip while still respecting `@pytest.mark.torch` for true torch-dependent cases. Ensure guidance includes revert steps if torch becomes available. |
| A4 | Add fallback verification | [ ] | Extend tests to assert fallback path emits informative warning/log entry; store any helper fixture under `tests/torch/conftest_helpers.py` if needed. |

---

### Phase B — Resolve P0 Blockers (Probe Mask & nphotons)
Goal: Implement adapter logic so highest-risk parity assertions pass.
Prereqs: Phase A complete (tests runnable), review `summary.md` P0 section and spec defaults.
Exit Criteria: `pytest tests/torch/test_config_bridge.py -k "probe_mask or nphotons" -v` passes (no skips/xfails).

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Implement probe_mask conversion | [ ] | Translate `ModelConfig.probe_mask` (Optional[torch.Tensor]) into a boolean flag or explicit override consistent with spec §5.1:8. Options: detect `None`→False, tensor with any non-zero→True, allow override for explicit bool. Document conversion in `adapter_diff.md` with rationale. |
| B2 | Extend tests for probe_mask | [ ] | Add targeted parity test (probable new parameterization) ensuring both default (`None`→False) and manual override paths are covered. Mark with `@pytest.mark.parametrize` per `field_matrix` guidance. |
| B3 | Enforce nphotons override | [ ] | Update adapter to require explicit `nphotons` value in overrides when PyTorch default differs. Ensure failure message references spec requirement. Verify training config uses override instead of PyTorch default. |
| B4 | Tighten default divergence test | [ ] | Adjust `test_default_divergence_detection` expectations so absence of override fails with actionable message, and presence passes. Capture failing log before fix in `pytest_red.log` (Phase A) and passing log after fix in `pytest_green.log`. |

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
- [ ] Tests no longer skipped when torch absent; fallback strategy documented.
- [ ] P0 probe_mask/nphotons tests green with explicit overrides enforced.
- [ ] params.cfg baseline comparison test added and passing, with override matrix recorded.
- [ ] `pytest_green.log` stored under the new timestamped directory capturing the first green run.

