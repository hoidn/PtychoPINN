Read the injected study docs and session state first.

Your job in this step is only to prepare one candidate architecture-improvement attempt for `lines_256`.

Your primary responsibility is to propose and build a worthwhile improvement attempt, not just any change that satisfies the mechanical contract of this task.
Before returning `READY`, the candidate must also be smoke-green: it should survive a cheap end-to-end training+inference sanity run after your code changes.

Bias toward changes that have a plausible path to better `amp_ssim` under the fixed `lines_256` budget.
Prefer coherent architectural or training-configuration hypotheses over random parameter thrashing.

**Simplicity criterion**: all else being equal, simpler is better.
- A small improvement that adds ugly complexity is usually not worth it.
- Removing code or configuration and getting equal or better results is a strong outcome.
- If a change adds complexity, the expected improvement should be large enough to justify that cost.
- A tiny metric gain from hacky or brittle code is usually a bad trade.
- An approximately neutral result that materially simplifies the system is still a good candidate to test.

Prefer changes that are:
- simple to explain in one sentence
- localized to a coherent mechanism
- easy to verify from the resulting code and run command
- consistent with the study docs' intended search surface

Avoid changes that are:
- hacky, ad hoc, or difficult to justify mechanistically
- broad cleanup bundles instead of one hypothesis
- likely to confound the comparison by changing multiple ideas at once
- only weakly motivated knob twiddling with no clear reason to expect improvement

Do not run the scored `20`-epoch experiment yourself. Do not append the TSV ledger. Do not parse scored metrics to decide whether the candidate should ultimately be retained. Do not perform post-evaluation git reset logic. Your responsibility ends after you have either produced a viable candidate package or a blocker report.

You must use the following injected files as authority:
- `docs/studies/lines_256_dataset.md`
- `docs/studies/lines_256_arch_improvement_loop.md`
- `state/lines_256_arch_improvement/protected_local_paths.json`
- `state/lines_256_arch_improvement/accepted_state.json`
- `state/lines_256_arch_improvement/candidate_context.json`

Task:
1. Read the authoritative study docs and the session-state files.
2. Treat the accepted branch state from `accepted_state.json` as the current accepted reference.
3. Choose the next coherent architecture or training-configuration hypothesis that is worth testing against the accepted reference.
4. Make exactly one coherent change implementing that hypothesis within the allowed editable surface from the study prompt.
5. Restrict candidate edits to existing files only, and only when those files are not in the protected local-change set.
6. Run a cheap end-to-end smoke check that exercises both training and inference for the edited candidate.
7. If the smoke check fails because of your code changes, fix the problem and rerun the smoke check.
8. Only once the candidate is smoke-green, stage only the intended candidate files.
9. Create exactly one candidate commit.
10. Write candidate metadata JSON to the exact path named by `candidate_metadata_path` in `candidate_context.json`.
11. If you created a candidate commit, also write the staged candidate file list JSON to the exact path named by `candidate_paths_file` in `candidate_context.json`.

If you cannot identify a candidate that is both coherent and worth testing, prefer `BLOCKED` over inventing a low-quality experiment.
If you cannot get the candidate smoke-green after a small number of focused fixes, prefer `BLOCKED` over finalizing a broken candidate.

Allowed outcomes:

1. `READY`
Use this only when you successfully made one coherent change, got the candidate through the smoke check, staged only candidate files, and created exactly one commit.

2. `BLOCKED`
Use this when the best next change would:
- require modifying a protected local-change file
- require editing outside the allowed editable surface
- require guessing missing study/runner behavior
- have no credible next experiment that clears the simplicity and coherence bar
- fail the smoke check in a way you cannot repair cleanly without thrashing or guessing
- or otherwise should stop this task instead of guessing

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
  "smoke_command": "python scripts/studies/grid_lines_torch_runner.py ... --epochs 1 ...",
  "smoke_output_root": "outputs/lines_256_arch_improvement/<timestamp>__smoke",
  "smoke_log_path": "state/lines_256_arch_improvement/<timestamp>__smoke.log",
  "run_command": "python scripts/studies/run_lines_256_arch_experiment.py --output-dir outputs/lines_256_arch_improvement/<timestamp>_<short_commit> ...",
  "output_root": "outputs/lines_256_arch_improvement/<timestamp>_<short_commit>",
  "log_path": "state/lines_256_arch_improvement/<timestamp>_<short_commit>.log",
  "comparison_png_path": "outputs/lines_256_arch_improvement/comparison_pngs/<session_id>/<timestamp>_<short_commit>__compare_amp_phase_probe.png",
  "note": "one short sentence explaining the change",
  "hypothesis": "one short sentence explaining why this change could improve amp_ssim or simplify the system"
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
- Keep `probe_mask` off unless the candidate is explicitly about probe masking.
- Do not fabricate evidence or claim improvement. This task only prepares and validates a candidate package.
- Do not stage or commit `state/` or `outputs/`.
- The accepted reference is the comparison target. Optimize for a better next accepted state, not for novelty.
- Prefer deletions, simplifications, and cleaner mechanisms when they are plausible improvement paths.
- If a change meaningfully increases complexity, be able to justify that cost in the `hypothesis`.

Before finishing:
- verify the smoke check succeeded
- verify the commit exists
- verify `candidate_metadata_path` exists and contains valid JSON
- if `READY`, verify `candidate_paths_file` exists and contains a JSON list of relative file paths
