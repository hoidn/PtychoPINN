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
- the injected `protected_local_paths.json` for this task
- the injected `accepted_state.json` for this task
- the injected `candidate_context.json` for this task
- the injected `candidate_metadata.json` for this task
- the injected `candidate_assessment.json` for this task

Task:
1. Read the injected files.
2. Read the crash log at the `log_path` named in the injected candidate metadata file.
3. Identify the narrowest credible fix for that crash.
4. If the crash is not worth fixing cleanly, emit `BLOCKED`.
5. If it is worth fixing, prepare the repaired candidate in exactly one of these forms:
   - `source`: make one coherent bugfix/update addressing the crash
   - `run_config`: keep tracked source files at the accepted ref and repair the crash by changing only the smoke and scored run configuration
6. If you are preparing a `source` candidate, restrict edits to existing files only, and only when those files are not in the protected local-change set.
7. If you are preparing a `run_config` candidate, leave tracked source files unchanged.
8. Run a cheap end-to-end smoke check that exercises both training and inference for the repaired candidate.
9. If the smoke check fails because of your fix, repair it and rerun the smoke check.
10. Only once the repaired candidate is smoke-green:
   - for `source`, stage only the intended candidate files and create exactly one replacement candidate commit
   - for `run_config`, stage nothing and create no replacement candidate commit
11. Write the debug-candidate metadata JSON at the exact path named by `debug_candidate_metadata_path` in the injected candidate context file.
12. If and only if you created a replacement source candidate commit, also write the staged candidate file list JSON at the exact path named by `debug_candidate_paths_file` in the injected candidate context file.

If you cannot get the repaired candidate smoke-green after a small number of focused fixes, prefer `BLOCKED` over thrashing.

Allowed outcomes:

1. `READY`
Use this only when you repaired the crash, got the candidate through the smoke check, and wrote complete metadata for either a `source` or `run_config` replacement candidate.

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

Candidate metadata requirements:

For `READY` with `candidate_kind: "source"`, include:
- `status`
- `candidate_kind`
- `base_ref`
- `candidate_paths_file`
- `smoke_command`
- `smoke_output_root`
- `smoke_log_path`
- `run_command`
- `output_root`
- `log_path`
- `comparison_png_path`
- `note`
- `hypothesis`

For `READY` with `candidate_kind: "run_config"`, include:
- `status`
- `candidate_kind`
- `base_ref`
- `smoke_command`
- `smoke_output_root`
- `smoke_log_path`
- `run_command`
- `output_root`
- `log_path`
- `comparison_png_path`
- `note`
- `hypothesis`

For `BLOCKED`, include:
- `status`
- `blocker_reason`

For `source` replacement candidates, the controller resolves the authoritative candidate commit from the workspace `HEAD` after reading your metadata. Do not invent or guess a `candidate_commit` field. The stable provider-owned provenance surface is the staged `debug_candidate_paths_file`.

Rules:
- Use the `timestamp_utc`, `comparison_gallery_dir`, `output_root_base`, and `log_root` values from the injected candidate context file when constructing output paths.
- The smoke check should use `scripts/studies/grid_lines_torch_runner.py` directly so you can run a cheap end-to-end sanity pass with `epochs=1` while preserving the same `lines_256` dataset, seed, `N=256`, `gridsize=1`, `hybrid_resnet` architecture family, `--no-probe-mask`, and `--torch-mae-pred-l2-match-target`.
- The smoke check must exercise both training and inference, not just CLI parsing or importability.
- After the smoke check, inspect `<smoke_output_root>/runs/pinn_hybrid_resnet/randomness_contract.json` and compare it to `accepted_state.json["accepted_randomness_contract"]`.
- Only treat randomness as a blocker if the repaired smoke run cannot produce that contract or it does not match the accepted reference. Do not block merely because an internal effective seed value differs from the requested wrapper seed when it still matches the accepted session contract.
- The candidate run command must target the fixed `lines_256` wrapper and preserve the fixed dataset/epoch contract from the study docs.
- Set `comparison_png_path` to the session-gallery `compare_amp_phase_probe.png` artifact for this repaired candidate, not to a nested `visuals/` path.
- For `run_config`, prepare a parameter-only rerun of the accepted architecture. Do not edit tracked source files just to express the crash fix.
- For `run_config`, use unique output and log paths derived from `timestamp_utc` and `base_ref`, because there is no replacement candidate commit hash to name the run.
- Do not stage or commit `state/` or `outputs/`.
- Set `base_ref` to the accepted git sha from `accepted_state.json` that you started from before finalizing the replacement candidate package.
- Keep the fix local to the crash you are addressing unless a nearby cleanup is necessary to make the fix coherent.

Before finishing:
- verify the smoke check succeeded
- verify `debug_candidate_metadata_path` exists and contains valid JSON
- if `candidate_kind` is `source`, verify `HEAD` is on the intended replacement candidate commit
- if `candidate_kind` is `source`, verify `debug_candidate_paths_file` exists and contains a JSON list of relative file paths
- if `candidate_kind` is `run_config`, verify tracked source files are unchanged
