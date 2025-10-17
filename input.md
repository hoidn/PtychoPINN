Summary: Trace PyTorch dataloader neighbor indexing overflow to unblock integration parity
Mode: Parity
Focus: INTEGRATE-PYTORCH-001-DATALOADER-INDEXING — Fix PyTorch dataloader neighbor indexing overflow
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T230724Z/{pytest_integration_fail.log,callchain/static.md,callgraph/dynamic.txt,callchain/summary.md,trace/tap_points.md,env/trace_env.json}

Do Now:
1. INTEGRATE-PYTORCH-001-DATALOADER-INDEXING @ plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md:D2 — Reproduce the IndexError via `pytest tests/torch/test_integration_workflow_torch.py -vv` | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T230724Z/pytest_integration_fail.log (tests: targeted)
2. INTEGRATE-PYTORCH-001-DATALOADER-INDEXING @ docs/fix_plan.md:71 — Execute prompts/callchain.md with `analysis_question="Why does ptycho_torch.dataloader assign nn_indices beyond the diffraction stack during the integration workflow?"`, `initiative_id="INTEGRATE-PYTORCH-001-DATALOADER-INDEXING"`, `scope_hints=["neighbor indexing","group sampling","mmap"]`, `roi_hint="datasets/Run1084_recon3_postPC_shrunk_3.npz"`, `namespace_filter="ptycho_torch"`, `time_budget_minutes=30`; capture required artifacts under plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T230724Z/ (tests: none)
3. INTEGRATE-PYTORCH-001-DATALOADER-INDEXING @ docs/fix_plan.md:78 — Summarize findings in callchain/summary.md, list proposed tap points, and append Attempt #1 with artifact links to docs/fix_plan.md (tests: none)

If Blocked: Archive the failing pytest log under the timestamped reports directory, note the blocker and hypothesis in callchain/summary.md, and record the partial outcome in docs/fix_plan.md Attempts history.

Priorities & Rationale:
- docs/fix_plan.md:71 — New ledger item defines acceptance gates for resolving the IndexError.
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md:34 — D2 guidance now escalates the neighbor indexing blocker.
- specs/data_contracts.md:1 — Canonical dataset sizing rules govern valid index bounds.
- specs/ptychodus_api_spec.md:40 — Reconstructor contract requires consistent grouping geometry across backends.
- docs/workflows/pytorch.md:24 — PyTorch data pipeline expectations to honour when analysing callchain.

How-To Map:
- export TS=2025-10-17T230724Z; mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/$TS/{callchain,callgraph,trace,env}
- pytest tests/torch/test_integration_workflow_torch.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$TS/pytest_integration_fail.log
- Follow prompts/callchain.md using the parameters above; emit `callchain/static.md`, `callgraph/dynamic.txt` (optional), `trace/tap_points.md`, `callchain/summary.md`, and `env/trace_env.json`
- Keep ROI minimal (canonical dataset) and avoid modifying production code; stop tracing once the offending index calculations are understood
- Update docs/fix_plan.md Attempts history for INTEGRATE-PYTORCH-001-DATALOADER-INDEXING with artifact paths and key findings

Pitfalls To Avoid:
- Do not edit implementation code or tests during this evidence loop.
- Keep callchain instrumentation module/device neutral; avoid hard-coding CUDA assumptions.
- Ensure pytest run executes from repo root with editable install active.
- Respect artifact hygiene: no logs at repo root, only under the timestamped reports directory.
- Do not rely on legacy `diff3d` paths; focus on canonical `diffraction` loading flow.
- Avoid expanding ROI beyond the canonical dataset; large datasets make tracing intractable.
- Document any skipped callgraph outputs explicitly in summary.md if tooling limitations arise.
- Preserve CONFIG-001 ordering reminders when analysing params bridge interactions.
- Capture parameter values from code, not plan notes, when describing index bounds.
- Keep tap point proposals actionable (function + owner), per prompts/callchain.md.

Pointers:
- docs/fix_plan.md:71 — INTEGRATE-PYTORCH-001-DATALOADER-INDEXING scope + exit criteria.
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md:34 — D2 row referencing the IndexError and expected follow-up.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/parity_summary.md:56 — Prior parity run capturing the IndexError stack trace.
- specs/data_contracts.md:20 — Diffraction/object/probe dimensional requirements for canonical datasets.
- specs/ptychodus_api_spec.md:70 — Grouping parameters that must align across backends.

Next Up: Implement neighbor indexing fix (`INTEGRATE-PYTORCH-001-DATALOADER-INDEXING`) once callchain evidence isolates the faulty slice logic.
