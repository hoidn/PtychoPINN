# Phase D3 Evidence Prep — Persistence Callchain Brief (2025-10-17T101657Z)

## Context
- **Initiative:** INTEGRATE-PYTORCH-001
- **Focus:** Phase D3 — persistence bridge parity
- **Loop Role:** Supervisor (galph) — prepare evidence collection brief for ralph
- **Action Type:** Evidence collection (callchain tracing)
- **Mode:** Parity

## Analysis Question for Callchain Prompt
"How does the TensorFlow workflow produce and consume `wts.h5.zip` bundles (train → `ModelManager.save_multiple_models` → archive layout → `load_inference_bundle`), and which touchpoints must the PyTorch backend replicate to stay spec-compliant? Contrast with Lightning checkpoint + MLflow outputs in `ptycho_torch.train`."

## Recommended Prompt Parameters
- **initiative_id:** `INTEGRATE-PYTORCH-001`
- **scope_hints:** `["ModelManager", "persistence", "archives", "load_inference_bundle"]`
- **roi_hint:** "Single training run artefact (mock path acceptable); focus on control flow, not full training"
- **namespace_filter:** `ptycho`
- **time_budget_minutes:** `30`

## Key Documentation & Code Anchors
| Path | Rationale |
| --- | --- |
| `specs/ptychodus_api_spec.md` §4.5–4.6 | Defines reconstructor lifecycle + archive contract (`wts.h5.zip`). |
| `plans/ptychodus_pytorch_integration_plan.md` Phase 5 | Canonical persistence tasks + risks. |
| `ptycho/workflows/components.py:500-750` | TensorFlow orchestration calling ModelManager save/load helpers. |
| `ptycho/model_manager.py` | Authoritative save/load implementation for TensorFlow backend. |
| `ptycho_torch/train.py:100-220` | Current Lightning checkpoint + MLflow behaviour that lacks archive parity. |
| `plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md` (D3 rows) | Phase task definitions consuming this evidence. |
| `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T101500Z/phase_d2c_green.md` | Notes on pending `_reassemble_cdi_image_torch` stub + persistence hand-off expectations. |

## Deliverable Expectations for Ralph
1. Follow `prompts/callchain.md` using the analysis question above; capture
   - `callchain/static.md`
   - (optional) `callgraph/dynamic.txt`
   - `trace/tap_points.md`
   - `summary.md`
   - `env/trace_env.json`
   Store under `plans/active/INTEGRATE-PYTORCH-001/reports/<timestamp>/phase_d3_callchain/`.
2. Highlight archive composition (files within `wts.h5.zip`, params snapshots, custom objects) and list the side-effects (`params.cfg` updates) observed along the path.
3. Document PyTorch outputs to contrast (Lightning checkpoint path, MLflow artefacts) — include anchors to `ptycho_torch/train.py` and any supporting helpers.
4. Enumerate the minimum hooks the PyTorch persistence shim must expose (save hook, params snapshot, archive writer) and flag open questions for D3.B/D3.C.

## Findings Ledger Check
- `docs/findings.md` — no persistence-specific entries (CONFIG-001/ MIGRATION-001/ DATA-001 only). Treat persistence analysis as net-new; record any discoveries after callchain execution.

## Risks & Notes
- Ensure callchain stays within evidence-only bounds (no edits to `ptycho/model_manager.py`).
- If dynamic tracing is impractical, annotate why in `summary.md` and focus on static mapping.
- Capture assumptions about MLflow availability; this directly informs D3 warning strategy.

## Next-Step Gating
- Phase D3.B/C may not start until this callchain summary is filed and referenced from `docs/fix_plan.md` Attempt #48.
- Supervisor will review outputs next loop to decide whether to proceed with archive writer implementation or request additional evidence.
