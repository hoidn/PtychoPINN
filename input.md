Summary: Map TensorFlow workflow entry points and choose the PyTorch orchestration surface before coding.
Mode: Docs
Focus: INTEGRATE-PYTORCH-001 / Phase D1
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T085431Z/{phase_d_callchain.md,phase_d_asset_inventory.md,phase_d_decision.md}
Do Now:
- INTEGRATE-PYTORCH-001 D1.A @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — run callchain analysis for `ptycho.workflows.components.run_cdi_example`; tests: none
- INTEGRATE-PYTORCH-001 D1.B @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — author `phase_d_asset_inventory.md` summarizing reusable PyTorch modules vs gaps; tests: none
- INTEGRATE-PYTORCH-001 D1.C @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — draft `phase_d_decision.md` capturing API-vs-shim orchestration choice with pros/cons and MLflow policy; tests: none
- docs/fix_plan.md — log Attempt #40 with artifact links once docs are saved; tests: none
If Blocked: Capture the blocker in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T085431Z/blockers.md`, note mitigation ideas, and record the stall in docs/fix_plan.md before stopping.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:127 — defines reconstructor lifecycle that Phase D must satisfy.
- ptycho/workflows/components.py:676 — source of TensorFlow workflow we must mirror.
- ptycho/model_manager.py:1 — persistence contract informing D3 assumptions gathered during D1.
- ptycho_torch/train.py:1 — current PyTorch training surface to inventory.
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md — authoritative checklist for this loop.
How-To Map:
- For D1.A use `prompts/callchain.md` with: analysis_question="What entrypoints and side-effects does run_cdi_example trigger that PyTorch must replicate?", initiative_id="INTEGRATE-PYTORCH-001", scope_hints=["workflow orchestration","ModelManager","config bridge"], roi_hint="Minimal Fly dataset stub", namespace_filter="ptycho.workflows", time_budget_minutes=30. Save outputs to `phase_d_callchain.md` (static), optional dynamic trace, proposed tap points.
- For D1.B review `ptycho_torch/train.py`, `ptycho_torch/inference.py`, `ptycho_torch/api/`, and existing Lightning helpers. Document modules, readiness state, and gaps (e.g. missing `run_cdi_example` analog, MLflow coupling) plus cross-links back to integration plan Delta-2.
- For D1.C compare two orchestration options: (A) wrap `ptycho_torch/api` surface, (B) build thin shims around low-level modules. Include decision criteria (config bridge touchpoints, MLflow optionality, persistence strategy), note provisional owner, and record decision/next steps.
- After artifacts are written, update docs/fix_plan.md Attempts History referencing each new file and mark D1.A–D1.C progress.
Pitfalls To Avoid:
- Do not modify production code or tests this loop.
- Keep callchain ROI minimal; avoid loading full datasets or running training jobs.
- Honor torch-optional policy—no hard `import torch` in analysis notes.
- Follow prompts/callchain instructions exactly; capture required files under the artifact directory.
- Document any open questions in the decision log instead of deferring silently.
Pointers:
- specs/ptychodus_api_spec.md:127
- ptycho/workflows/components.py:676
- ptycho/model_manager.py:1
- ptycho_torch/train.py:1
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md
Next Up:
- D2.A–D2.C scaffolding @ plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md once D1 artifacts are approved.
