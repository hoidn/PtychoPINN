Read the injected study docs and session state first.

Your job in this step is only to prepare one candidate architecture-improvement attempt for `lines_256`.

Do not run the experiment yourself. Do not append the TSV ledger. Do not parse metrics. Do not perform keep/discard git reset logic. The workflow owns those deterministic actions.

You must use the following injected files as authority:
- `docs/studies/lines_256_dataset.md`
- `docs/studies/lines_256_arch_improvement_loop.md`
- `state/lines_256_arch_improvement/protected_local_paths.json`
- `state/lines_256_arch_improvement/accepted_state.json`
- `state/lines_256_arch_improvement/candidate_context.json`

Task:
1. Read the authoritative study docs and the session-state files.
2. Treat the accepted branch state from `accepted_state.json` as the current accepted reference.
3. Make one coherent architecture or training-configuration change within the allowed editable surface from the study prompt.
4. Restrict candidate edits to existing files only, and only when those files are not in the protected local-change set.
5. Stage only the intended candidate files.
6. Create exactly one candidate commit.
7. Write candidate metadata JSON to the exact path named by `candidate_metadata_path` in `candidate_context.json`.
8. If you created a candidate commit, also write the staged candidate file list JSON to the exact path named by `candidate_paths_file` in `candidate_context.json`.

Allowed outcomes:

1. `READY`
Use this when you successfully made one coherent change, staged only candidate files, and created exactly one commit.

2. `BLOCKED`
Use this when the best next change would:
- require modifying a protected local-change file
- require editing outside the allowed editable surface
- require guessing missing study/runner behavior
- or otherwise should stop the autonomous loop instead of guessing

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
  "candidate_paths_file": "state/lines_256_arch_improvement/candidate_paths.json",
  "run_command": "python scripts/studies/run_lines_256_arch_experiment.py --output-dir outputs/lines_256_arch_improvement/<timestamp>_<short_commit> ...",
  "output_root": "outputs/lines_256_arch_improvement/<timestamp>_<short_commit>",
  "log_path": "state/lines_256_arch_improvement/<timestamp>_<short_commit>.log",
  "comparison_png_path": "outputs/lines_256_arch_improvement/comparison_pngs/<session_id>/<timestamp>_<short_commit>__compare_amp_phase_probe.png",
  "note": "one short sentence explaining the change"
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
- The candidate run command must target the fixed `lines_256` wrapper and preserve the fixed dataset/epoch contract from the study docs.
- Keep `probe_mask` off unless the candidate is explicitly about probe masking.
- Do not fabricate evidence or claim improvement. This step only proposes and commits a candidate.
- Do not stage or commit `state/` or `outputs/`.

Before finishing:
- verify the commit exists
- verify `candidate_metadata_path` exists and contains valid JSON
- if `READY`, verify `candidate_paths_file` exists and contains a JSON list of relative file paths
