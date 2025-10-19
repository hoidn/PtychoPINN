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
| D1e.A1 | Capture fresh failing integration log with shape annotations | [x] | **COMPLETE (Attempt #38):** Fresh integration failure log captured at `pytest_integration_shape_red.log` (16KB). Confirmed RuntimeError at `ptycho_torch/model.py:366` (`outputs = x1 + x2`): "The size of tensor a (572) must match the size of tensor b (1080) at non-singleton dimension 3". |
| D1e.A2 | Instrument decoder for shape traces (temporary logging) | [x] | **COMPLETE (Attempt #38):** Added temporary `TORCH_DECODER_TRACE` env-var controlled logging to `ptycho_torch/model.py:349-386` and `ptycho_torch/inference.py:382-385`. Captured shape trace at `shape_trace.md` (1.4KB) showing x1=(8,1,64,572), x2=(8,1,64,1080). Instrumentation removed after evidence gathering (not committed). |
| D1e.A3 | Author triage memo summarising hypotheses | [x] | **COMPLETE (Attempt #38):** Updated `shape_mismatch_triage.md` with "Evidence Collected" section documenting root cause (2× upsampling on x2 vs padding on x1), hypothesis validation (confirmed upsample scale factor mismatch), and recommended fix (Option A: center-crop x2 to match x1 spatial dims). |

## Phase B — TDD Fix (Decoder Alignment)
Goal: Encode failure with targeted pytest coverage and implement decoder alignment to eliminate the mismatch.
Prereqs: Phase A artifacts complete; triage hypotheses narrowed to actionable fix.
Exit Criteria: New unit tests red→green, production fix merged, GREEN selector logs stored.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1e.B1 | Add failing decoder shape regression test | [x] | **COMPLETE (Attempt #40):** Modernised `TestDecoderLastShapeParity` so `test_probe_big_shape_alignment` asserts successful forward pass with spatial parity. Captured RED evidence at `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/pytest_decoder_shape_red.log` showing pre-fix RuntimeError (572 vs 1080 mismatch). |
| D1e.B2 | Implement decoder alignment fix | [x] | **COMPLETE (Attempt #40):** Applied center-crop parity in `ptycho_torch/model.py:366-381`, trimming the x2 path to x1 spatial dims without device/dtype drift. Mirrors TensorFlow `trim_and_pad_output` behaviour. |
| D1e.B3 | Green the decoder regression test | [x] | **COMPLETE (Attempt #40):** Targeted selector `pytest tests/torch/test_workflows_components.py::TestDecoderLastShapeParity -vv` now passes 2/2; GREEN log stored at `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/pytest_decoder_shape_green.log`. |

## Phase C — Integration Validation & Ledger Updates
Goal: Prove fix in end-to-end workflow, propagate documentation updates, unblock downstream phases.
Prereqs: Phase B GREEN tests.
Exit Criteria: Integration test passes past decoder stage (ideally green, or new blocker documented), checklist rows updated, fix plan attempt recorded.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1e.C1 | Re-run integration selector post-fix | [x] | **COMPLETE (Attempt #40):** Integration selector green (`pytest_integration_shape_green.log`, 20.44s) stored under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/`. |
| D1e.C2 | Refresh plans & summaries | [x] | **COMPLETE (Attempt #40):** `phase_d2_completion.md` D1e row marked `[x]`; new summary + checklist updates published at `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/summary.md`; docs/fix_plan Attempt #40 logged with artifact references. |
| D1e.C3 | Update findings/test references if semantics shift | [x] | **COMPLETE (Attempt #40):** Decoder behaviour now matches TensorFlow baseline—no spec update required. Recorded parity confirmation in `shape_mismatch_triage.md` “GREEN Evidence” section. |

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
