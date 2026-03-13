Read the injected study docs and session state first.

Your job in this step is to address one specific candidate crash for `lines_256`.

The candidate already exists, already has a crash record, and already has a crash log path. Your task is to inspect that crash, decide whether there is a clean targeted fix worth attempting, and if so prepare one replacement candidate attempt.

Your primary responsibility is to fix the concrete crash without drifting into a broad rewrite.
Bias toward the smallest change that makes the candidate viable again while preserving the candidate's underlying hypothesis when that still makes sense.

**Simplicity criterion**: all else being equal, simpler is better.
- Prefer a small targeted fix over a larger redesign.
- If the crash reveals the candidate idea is not worth pursuing cleanly, prefer `BLOCKED`.
- Do not pile speculative improvements on top of the crash fix.

Do not run the scored `20`-epoch experiment yourself. Do not append the TSV ledger. Do not perform post-evaluation git reset logic. Your responsibility ends after you have either produced a crash-fixed viable candidate package or a blocker report.

You must use the following injected files as authority:
- `docs/studies/lines_256_dataset.md`
- `docs/studies/lines_256_arch_improvement_loop.md`
- `state/lines_256_arch_improvement/protected_local_paths.json`
- `state/lines_256_arch_improvement/accepted_state.json`
- `state/lines_256_arch_improvement/candidate_context.json`
- `state/lines_256_arch_improvement/candidate_metadata.json`
- `state/lines_256_arch_improvement/candidate_assessment.json`

Task:
1. Read the injected files.
2. Read the crash log at the `log_path` named in `candidate_metadata.json`.
3. Identify the narrowest credible fix for that crash.
4. If the crash is not worth fixing cleanly, emit `BLOCKED`.
5. If it is worth fixing, make exactly one coherent bugfix/update addressing the crash.
6. Restrict edits to existing files only, and only when those files are not in the protected local-change set.
7. Run a cheap end-to-end smoke check that exercises both training and inference for the repaired candidate.
8. If the smoke check fails because of your fix, repair it and rerun the smoke check.
9. Only once the repaired candidate is smoke-green, stage only the intended candidate files.
10. Create exactly one replacement candidate commit.
11. Write the debug-candidate metadata JSON at the exact path named by `debug_candidate_metadata_path` in `candidate_context.json`.
12. If you created a replacement candidate commit, also write the staged candidate file list JSON at the exact path named by `debug_candidate_paths_file` in `candidate_context.json`.

If you cannot get the repaired candidate smoke-green after a small number of focused fixes, prefer `BLOCKED` over thrashing.

Allowed outcomes:

1. `READY`
Use this only when you repaired the crash, got the candidate through the smoke check, staged only candidate files, and created exactly one replacement commit.

2. `BLOCKED`
Use this when:
- the crash would require modifying a protected local-change file
- the cleanest fix lies outside the allowed editable surface
- the crash reveals missing or ambiguous runner/study behavior that you should not guess
- the candidate idea is no longer worth pursuing after understanding the crash
- you cannot get the repaired candidate smoke-green without thrashing or guessing

If you are blocked:
- make no candidate commit
- leave tracked source files unchanged
- write only the candidate metadata JSON with `status: "BLOCKED"` and a concise `blocker_reason`

Candidate metadata JSON schema:

For `READY`:

```json
{
  "status": "READY",
  "candidate_commit": "<full git sha>",
  "candidate_paths_file": "state/lines_256_arch_improvement/debug_candidate_paths.json",
  "smoke_command": "python scripts/studies/grid_lines_torch_runner.py ... --epochs 1 ...",
  "smoke_output_root": "outputs/lines_256_arch_improvement/<timestamp>__smoke",
  "smoke_log_path": "state/lines_256_arch_improvement/<timestamp>__smoke.log",
  "run_command": "python scripts/studies/run_lines_256_arch_experiment.py --output-dir outputs/lines_256_arch_improvement/<timestamp>_<short_commit> ...",
  "output_root": "outputs/lines_256_arch_improvement/<timestamp>_<short_commit>",
  "log_path": "state/lines_256_arch_improvement/<timestamp>_<short_commit>.log",
  "comparison_png_path": "outputs/lines_256_arch_improvement/comparison_pngs/<session_id>/<timestamp>_<short_commit>__compare_amp_phase.png",
  "note": "one short sentence explaining the crash fix",
  "hypothesis": "one short sentence explaining why this repaired candidate is still worth scoring"
}
```

For `BLOCKED`:

```json
{
  "status": "BLOCKED",
  "blocker_reason": "short explicit reason"
}
```

Rules:
- Use the `timestamp_utc`, `comparison_gallery_dir`, `output_root_base`, and `log_root` values from `candidate_context.json` when constructing output paths.
- The smoke check should use `scripts/studies/grid_lines_torch_runner.py` directly so you can run a cheap end-to-end sanity pass with `epochs=1` while preserving the same `lines_256` dataset, seed, `N=256`, `gridsize=1`, `hybrid_resnet` architecture family, and no-probe-mask contract.
- The smoke check must exercise both training and inference, not just CLI parsing or importability.
- The candidate run command must target the fixed `lines_256` wrapper and preserve the fixed dataset/epoch contract from the study docs.
- Do not stage or commit `state/` or `outputs/`.
- Keep the fix local to the crash you are addressing unless a nearby cleanup is necessary to make the fix coherent.

Before finishing:
- verify the smoke check succeeded
- verify the replacement commit exists if you returned `READY`
- verify `debug_candidate_metadata_path` exists and contains valid JSON
- if `READY`, verify `debug_candidate_paths_file` exists and contains a JSON list of relative file paths
