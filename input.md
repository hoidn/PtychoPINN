Summary: Map TensorFlow model persistence so the PyTorch backend can mirror the wts.h5.zip contract before coding Phase D3.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 — Phase D3 persistence bridge
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T101657Z/phase_d3_callchain/{callchain/static.md,summary.md,trace/tap_points.md,env/trace_env.json}
Do Now:
1. INTEGRATE-PYTORCH-001 D3.A @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — run the callchain tracing workflow (analysis question in phase_d3_persistence_prep.md) via prompts/callchain.md; capture required deliverables under the artifact directory; tests: none.

If Blocked: If callchain execution stalls (missing tooling or unclear entrypoint), pivot to a static mapping: document entry→ModelManager→archive flow with path:line anchors in `phase_d3_callchain/static.md` and log open questions in `summary.md` so we can adjust scope next loop.

Priorities & Rationale:
- specs/ptychodus_api_spec.md:192-202 mandates archive parity; we need an exact flow map before designing adapters.
- ptycho/model_manager.py:346-420 shows how TensorFlow saves/loads multi-model archives; tracing it ensures we replicate manifests and params snapshots.
- ptycho/workflows/components.py:676-720 drives training/save orchestration; we must capture where ModelManager hooks in and how params.cfg is restored.
- ptycho_torch/train.py:100-220 highlights the current Lightning + MLflow persistence delta; documenting gaps will shape D3.B/D3.C scope.
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:44 gates implementation on this evidence, so the callchain is prerequisite for further work.

How-To Map:
- Read `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T101657Z/phase_d3_persistence_prep.md` for the analysis question and parameter suggestions.
- Follow prompts/callchain.md: build the candidate entrypoint table from docs, then trace from `run_cdi_example`/`ModelManager.save_multiple_models` through to `load_inference_bundle` with path:line anchors.
- If dynamic tracing is feasible, record a minimal run (e.g., scripted no-op harness) filtered to `ptycho.*`; otherwise note why it was skipped in `callchain/summary.md`.
- Contrast findings with PyTorch outputs: reference Lightning checkpoint creation, MLflow artifacts, and where those surfaces live; add anchors in the `summary.md` delta section.
- Drop all deliverables (static.md, tap_points.md, summary.md, env/trace_env.json, optional dynamic.txt) under `.../phase_d3_callchain/` and link any supporting scratch notes from there.

Pitfalls To Avoid:
- Do not modify production code or configs; this is evidence-only.
- Keep the callchain doc-first—cite specs/plan sections before code.
- Maintain torch-optional imports; no `import torch` in new analysis scripts.
- Record path:line anchors for every stage; avoid vague references.
- Don’t spend time on full training runs; focus on control flow and archive layout.
- Avoid assuming MLflow availability—note any dependency expectations explicitly.
- Ensure artifact filenames follow the template; no stray files at repo root.

Pointers:
- specs/ptychodus_api_spec.md:192-202 — Persistence contract requirements.
- ptycho/model_manager.py:346-420 — wts.h5.zip save/load implementation.
- ptycho/workflows/components.py:676-720 — Orchestration hook points for ModelManager.
- ptycho_torch/train.py:100-220 — Current Lightning checkpoint + MLflow outputs.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T101657Z/phase_d3_persistence_prep.md — Callchain brief and parameters.

Next Up: Execute D3.B (archive writer design) once the callchain evidence is reviewed and gaps are validated.
