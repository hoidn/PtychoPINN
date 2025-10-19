# Phase D1e — Lightning Decoder Shape Mismatch Remediation

## Context
- Initiative: INTEGRATE-PYTORCH-001 — PyTorch backend integration (Phase D2 completion).
- Trigger: Post-dtype fix integration run (`pytest_integration_dtype_green.log`) now fails with `RuntimeError: The size of tensor a (572) must match the size of tensor b (1080)` at `ptycho_torch/model.py:366` (`outputs = x1 + x2` inside `Decoder_last`).
- Observed during: Attempt #37 follow-through (dtype enforcement). Failure surfaces immediately after Lightning inference begins feeding stitched predictions back into the decoder.
- Dependencies:
  - `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md` (Phase D checklist owner).
  - `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/pytest_integration_dtype_green.log` (latest failing log).
  - TensorFlow parity reference: `ptycho/workflows/components.py` decoder/stitching flow.
  - Torch implementation reference: `ptycho_torch/model.py` (`Decoder_last`, `Decoder_phase`, `Decoder_amp`).
- Findings ledger anchors: CONFIG-001 (params bridge), FORMAT-001 (auto-transpose). Shape-grid parity must not violate these policies.

## Objectives
1. Capture authoritative evidence of mismatched tensor shapes within the Lightning decoder / autoencoder path.
2. Restore spatial parity with TensorFlow baseline via TDD (unit tests exercising decoder combine path and integration test).
3. Unblock Phase D2 parity summary work (`phase_d2_completion.md` D2/D3) and TEST-PYTORCH-001 integration initiative.

## Phase A — Evidence & Shape Instrumentation
Goal: Localize the precise tensor dimensions responsible for the 572 vs 1080 mismatch and document a reproducible failing scenario for TDD.
Prereqs: params.cfg bridge (CONFIG-001), dtype fix (D1d) merged, artifact hub ready at `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/`.
Exit Criteria: Shape trace, hypothesis table, and failing selector log captured under this timestamp; plan checklist D1e.A tasks `[x]`.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1e.A1 | Capture fresh failing integration log with shape annotations | [ ] | Re-run `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv \| tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/pytest_integration_shape_red.log`. Confirm failure reproduces at `Decoder_last.forward`. |
| D1e.A2 | Instrument decoder for shape traces (temporary logging) | [ ] | Use targeted pytest (`pytest tests/torch/test_workflows_components.py::TestReassembleCdiImageTorchFloat32::test_batches_remain_float32 -vv`) with `TORCH_DECODER_TRACE=1` env var to enable temporary logging (add instrumentation via patch stack, commit excluded). Record tensor shapes of `x1`, `x2`, and `outputs` in `shape_trace.md`. Remove instrumentation after evidence captured. |
| D1e.A3 | Author triage memo summarising hypotheses | [ ] | Document hypotheses in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/shape_mismatch_triage.md` (e.g., Probe big branch using mismatched padding, incorrect upsample scale, missing crop/pad). List confirming/refuting evidence, reference TensorFlow baseline (`ptycho/model.py:360-410`). |

## Phase B — TDD Fix (Decoder Alignment)
Goal: Encode failure with targeted pytest coverage and implement decoder alignment to eliminate the mismatch.
Prereqs: Phase A artifacts complete; triage hypotheses narrowed to actionable fix.
Exit Criteria: New unit tests red→green, production fix merged, GREEN selector logs stored.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1e.B1 | Add failing decoder shape regression test | [ ] | Extend `tests/torch/test_workflows_components.py` with `TestDecoderLastShapeParity` (pytest style) that constructs representative tensors (with probe_big=True, gridsize=2) and asserts `Decoder_last.forward` returns channels matching input. Capture RED log `pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity::test_probe_big_shape_alignment -vv \| tee .../pytest_decoder_shape_red.log`. |
| D1e.B2 | Implement decoder alignment fix | [ ] | Update `ptycho_torch/model.py` to mirror TensorFlow padding/cropping logic: ensure `conv_up_block` upsample scale matches `conv1` output; apply center crop/pad so `x1`/`x2` spatial dims align (use TensorFlow helper parity). Cast/reshape only where necessary (avoid losing complex info). |
| D1e.B3 | Green the decoder regression test | [ ] | Run the same pytest selector, capturing GREEN log (`.../pytest_decoder_shape_green.log`). Ensure test asserts both spatial dims and channel counts (including `probe_big=False` guard). |

## Phase C — Integration Validation & Ledger Updates
Goal: Prove fix in end-to-end workflow, propagate documentation updates, unblock downstream phases.
Prereqs: Phase B GREEN tests.
Exit Criteria: Integration test passes past decoder stage (ideally green, or new blocker documented), checklist rows updated, fix plan attempt recorded.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1e.C1 | Re-run integration selector post-fix | [ ] | `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv \| tee .../pytest_integration_shape_green.log`. If still failing, capture new failure signature and loop back to Phase A with updated triage. |
| D1e.C2 | Refresh plans & summaries | [ ] | Update `phase_d2_completion.md` (add D1e row, mark state) and `summary.md` with new evidence. Record docs/fix_plan.md Attempt (include selector logs, artifacts). |
| D1e.C3 | Update findings/test references if semantics shift | [ ] | If decoder contract diverges from TensorFlow baseline, note in `specs/ptychodus_api_spec.md` (§4.6) and `docs/workflows/pytorch.md` parity section. Otherwise, confirm alignment and document parity proof. |

## Artifact Discipline
- Store all new artefacts under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/`.
- Required files (initial scaffolding):
  - `shape_mismatch_triage.md` — hypotheses & evidence
  - `shape_trace.md` — optional detailed tensor shape logs
  - `pytest_decoder_shape_red.log` / `pytest_decoder_shape_green.log`
  - `pytest_integration_shape_red.log` / `pytest_integration_shape_green.log`
- Reference these artefacts in docs/fix_plan.md Attempts and plan checklist updates.

## Decision Gates
- If Phase B tests remain red after implementation, re-open Phase A with additional instrumentation (e.g., inspect autoencoder intermediate layers, compare with TensorFlow via saved checkpoints).
- If integration test reveals downstream issues (e.g., stitching regressions) after decoder fix, fork a new Phase D2 TODO before marking D1e complete to maintain “one active blocker per loop”.

