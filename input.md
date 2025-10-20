Summary: Refresh training CLI docs/hygiene after thin-wrapper landing.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase D.B (Training CLI thin wrapper, B4 documentation + hygiene)
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T112811Z/phase_d_cli_wrappers_training_docs/{summary.md}

Do Now:
1. ADR-003-BACKEND-API B4 (docs refresh) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:37 — Update `docs/workflows/pytorch.md` CLI sections with the new `--quiet` behaviour, `--device` deprecation messaging, and helper-based flow; revise `tests/torch/test_cli_shared.py` module docstring/comments to reflect current GREEN status; tests: none
2. ADR-003-BACKEND-API B4 (hygiene + ledger) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:37 — Move `train_debug.log` into the new artifact hub, capture a brief summary.md for the doc/hygiene updates, mark implementation.md D1 row `[x]`, and append Attempt #44 in docs/fix_plan.md; tests: none

If Blocked: Document the blocker in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T112811Z/phase_d_cli_wrappers_training_docs/blocker.md`, keep B4 `[ ]`, and log the stall in docs/fix_plan.md before pausing.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:37 — B4 exit criteria require doc updates plus artifact hygiene before D1 can close.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/summary.md — Next steps section lists documentation and log relocation as immediate follow-ups.
- plans/active/ADR-003-BACKEND-API/implementation.md:71 — D1 row remains `[ ]` until docs + hygiene land; this loop should flip it.
- tests/torch/test_cli_shared.py:1 — Docstring still describes RED scaffolds; needs alignment with GREEN helper coverage.
- specs/ptychodus_api_spec.md:206 — CLI contract requires documenting accelerator/legacy flag behaviour consistently across guides.

How-To Map:
- `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T112811Z/phase_d_cli_wrappers_training_docs`
- Update `docs/workflows/pytorch.md` §§3–5 & CLI examples to mention `--quiet`, `--disable_mlflow` deprecation, and helper-based configuration flow; cite helper path `ptycho_torch/cli/shared.py`.
- In `tests/torch/test_cli_shared.py`, replace RED-phase language with notes that these were formerly RED and now guard helper behaviour; keep pytest style intact.
- Move `train_debug.log` → `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T112811Z/phase_d_cli_wrappers_training_docs/train_debug.log`.
- After edits, update `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T112811Z/phase_d_cli_wrappers_training_docs/summary.md` with doc changes + hygiene actions, set implementation.md D1 `[x]`, and record Attempt #44 in docs/fix_plan.md referencing the new artifact path.
- No pytest runs required this loop (docs/hygiene only).

Pitfalls To Avoid:
- Don’t leave `train_debug.log` at repo root after this loop.
- Keep doc changes aligned with CONFIG-001 ordering; don’t imply helpers bypass update_legacy_dict.
- Preserve pytest-native structure in tests; avoid reintroducing unittest mixins.
- Refrain from editing core model/workflow logic (only docs + comments this loop).
- Maintain artifact naming under the timestamped reports directory (no crosstalk with prior hubs).
- When moving logs, use `mv` instead of re-creating to keep timestamps intact.
- Keep doc text consistent with `specs/ptychodus_api_spec.md` terminology.
- Document any open questions in the new summary before closing the loop.
- Run `git status` before staging to ensure only doc files changed.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:37
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/summary.md
- plans/active/ADR-003-BACKEND-API/implementation.md:67
- docs/workflows/pytorch.md:1
- tests/torch/test_cli_shared.py:1

Next Up: Kick off Phase D.C blueprint (plan.md C1) once D1 closes.
