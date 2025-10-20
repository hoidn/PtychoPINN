Summary: Ship Phase D.C C4 docs by aligning the workflow guide with the inference thin-wrapper behaviour and closing out plan/ledger bookkeeping.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase D.C (Inference CLI thin wrapper, C4)
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123820Z/phase_d_cli_wrappers_inference_docs/{docs_gap_analysis.md,docs_update_summary.md}

Do Now:
1. ADR-003-BACKEND-API C4 (workflow guide update) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — edit `docs/workflows/pytorch.md` §§12–13 to reflect the inference thin wrapper (flag defaults, helper delegation, artifact outputs, example command); tests: none.
2. ADR-003-BACKEND-API C4 (artifact + plan sync) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — document the doc edits in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123820Z/phase_d_cli_wrappers_inference_docs/docs_update_summary.md`, then mark plan row C4 `[x]` and refresh `summary.md` with completion notes; tests: none.
3. ADR-003-BACKEND-API C4 (ledger update) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — append docs/fix_plan.md Attempt #52 capturing the doc changes, artifact path, and confirmation that C4 exit criteria are satisfied; tests: none.

If Blocked: Capture the issue and current diff in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123820Z/phase_d_cli_wrappers_inference_docs/blocker.md`, leave plan row C4 at `[P]`, and log the blocker in docs/fix_plan.md before stopping.

Priorities & Rationale:
- docs/workflows/pytorch.md:360 — inference flag defaults must match the thin wrapper (`default='auto'`, `None` batch size) to prevent user misconfiguration.
- docs/workflows/pytorch.md:344 — add helper delegation narrative so training/inference guidance stays symmetrical per Phase D blueprint.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123820Z/phase_d_cli_wrappers_inference_docs/docs_gap_analysis.md — gap analysis lists required edits; follow it to avoid missing exit-criteria items.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48 — plan requires doc refresh + ledger sync before Phase D handoff.
- docs/fix_plan.md:159 — Attempt #51 recorded the outstanding doc work; the next entry must close C4 with artifact references.

How-To Map:
- Review `docs_gap_analysis.md` for the bullet list of required updates and keep it open while editing.
- Update `docs/workflows/pytorch.md` inference table defaults, add a "Helper Delegation" paragraph referencing `ptycho_torch/cli/shared.py` + `_run_inference_and_reconstruct`, and include an inference CLI example using the minimal dataset fixture with expected amplitude/phase outputs.
- Mention the `--device` deprecation timeline (Phase E removal) in the same way as the training section.
- Record the completed changes plus cross-links (spec section, tests touched) in `docs_update_summary.md` inside the artifact hub; note there that no tests were run.
- Set plan row C4 to `[x]`, add completion notes + artifact pointer in `plan.md` and `summary.md`, then append Attempt #52 in docs/fix_plan.md referencing the same artifact directory.
- Verify cleanliness with `git status` before committing.

Pitfalls To Avoid:
- Do not edit production Python modules—this loop is docs-only.
- Keep Markdown anchors (`<doc-ref>` tags) intact; add new ones only if you have authoritative targets.
- Ensure table formatting in `docs/workflows/pytorch.md` remains aligned (pipe-separated, consistent spacing).
- Don’t remove the training CLI guidance; mirror structure without duplicating content verbatim.
- Avoid mixing deprecated flag recommendations—make deprecation messaging consistent across training and inference.
- Store every new doc artifact under the timestamped reports directory; nothing belongs at repo root.
- Reference only evidence-backed defaults; do not invent new CLI behaviour.
- Update docs/fix_plan.md once, with a precise artifact link and Mode, to prevent ledger churn.
- Keep summary language concise; no placeholder text in plan or summary files.
- Run no tests and avoid invoking CLI commands—documentation only.

Pointers:
- docs/workflows/pytorch.md:344
- docs/workflows/pytorch.md:360
- ptycho_torch/inference.py:520
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123820Z/phase_d_cli_wrappers_inference_docs/docs_gap_analysis.md
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:48

Next Up: Phase D.D (smoke evidence + hygiene) once C4 documentation lands.
