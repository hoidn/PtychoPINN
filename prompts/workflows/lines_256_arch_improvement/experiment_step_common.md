Read the injected study docs and session state first.

Your job in this step is only to prepare one candidate architecture-improvement attempt for `lines_256`.

Your primary responsibility is to propose and build a worthwhile improvement attempt, not just any change that satisfies the mechanical contract of this task.
Before returning `READY`, the candidate must also be smoke-green: it should survive a cheap end-to-end training+inference sanity run after your code changes.

Bias toward changes that have a plausible path to better `amp_ssim` under the fixed `lines_256` budget.
Prefer coherent architectural or training-configuration hypotheses over random parameter thrashing.
Use the injected full-session `search_summary`, not just the last few attempts, to understand which knob families are already saturated and which remain underexplored.
Use the injected `proposal_mode` and `proposal_mode_reason` as proposal-steering context only. They are meant to help you choose a better next hypothesis, not to replace your judgment.
If `proposal_context.json` includes an active `queued_workflow_idea`, treat that queued idea as the highest-priority direction for this proposal step. Interpret it faithfully, but still apply the same coherence, simplicity, smoke-check, and fixed-study-contract standards.
If a queued workflow idea is not viable to test cleanly, return `BLOCKED` or a clean failed attempt rather than silently substituting some unrelated free-form idea.

**Simplicity criterion**: all else being equal, simpler is better.
- A small improvement that adds ugly complexity is usually not worth it.
- Removing code or configuration and getting equal or better results is a strong outcome.
- If a change adds complexity, the expected improvement should be large enough to justify that cost.
- A small metric gain that materially increases training cost, memory use, or model size is usually a bad trade unless the improvement is large enough to justify that cost.
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
- poorly motivated knob sweeps with no clear reason to expect improvement

Do not run the scored `20`-epoch experiment yourself. Do not append the TSV ledger. Do not parse scored metrics to decide whether the candidate should ultimately be retained. Do not perform post-evaluation git reset logic. Your responsibility ends after you have either produced a viable candidate package or a blocker report.

You must use the following injected files as authority:
- `docs/studies/lines_256_dataset.md`
- `docs/studies/lines_256_arch_improvement_loop.md`
- the injected `protected_local_paths.json` for this task
- the injected `accepted_state.json` for this task
- the injected `candidate_context.json` for this task
- the injected `proposal_context.json` for this task

Task:
1. Read the authoritative study docs and the session-state files.
2. Treat the accepted branch state and accepted run configuration from `accepted_state.json` as the current accepted experiment state.
3. Choose the next coherent architecture or training-configuration hypothesis that is worth testing against the accepted experiment state.
4. Prepare that candidate in exactly one of these forms:
   - `source`: make one coherent source change within the allowed editable surface from the study docs
   - `run_config`: keep tracked source files at the accepted ref and prepare a parameter-only rerun of the accepted architecture
5. If you are preparing a `source` candidate, restrict edits to existing files only, and only when those files are not in the protected local-change set.
6. If you are preparing a `run_config` candidate, leave tracked source files unchanged and express the hypothesis only through the smoke and scored run commands.
7. Run a cheap end-to-end smoke check that exercises both training and inference for the candidate you prepared.
8. If the smoke check fails because of your code change or parameter change, fix the problem and rerun the smoke check.
9. Only once the candidate is smoke-green:
   - for `source`, stage only the intended candidate files and create exactly one candidate commit
   - for `run_config`, stage nothing and create no candidate commit
10. Write candidate metadata JSON to the exact path named by `candidate_metadata_path` in the injected candidate context file.
11. If and only if you created a source-changing candidate commit, also write the staged candidate file list JSON to the exact path named by `candidate_paths_file` in the injected candidate context file.

If you cannot identify a candidate that is both coherent and worth testing, prefer `BLOCKED` over inventing a low-quality experiment.
If you cannot get the candidate smoke-green after a small number of focused fixes, prefer `BLOCKED` over finalizing a broken candidate.

Allowed outcomes:

1. `READY`
Use this only when you prepared one coherent candidate package, got it through the smoke check, and wrote complete metadata for either a `source` or `run_config` candidate.

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

Candidate metadata requirements:

For `READY` with `candidate_kind: "source"`, include:
- `status`
- `candidate_kind`
- `base_ref`
- `candidate_commit`
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

Rules:
- Use the `timestamp_utc`, `comparison_gallery_dir`, `output_root_base`, and `log_root` values from the injected candidate context file when constructing output paths.
- The smoke check should use `scripts/studies/grid_lines_torch_runner.py` directly so you can run a cheap end-to-end sanity pass with `epochs=1` while preserving the same `lines_256` dataset, seed, `N=256`, `gridsize=1`, `hybrid_resnet` architecture family, `--no-probe-mask`, and `--torch-mae-pred-l2-match-target`.
- The smoke check must exercise both training and inference, not just CLI parsing or importability.
- After the smoke check, inspect `<smoke_output_root>/runs/pinn_hybrid_resnet/randomness_contract.json` and compare it to `accepted_state.json["accepted_randomness_contract"]`.
- Only treat randomness as a blocker if the smoke run cannot produce that contract or it does not match the accepted reference. Do not block merely because an internal effective seed value differs from the requested wrapper seed when it still matches the accepted session contract.
- The candidate run command must target the fixed `lines_256` wrapper and preserve the fixed dataset/epoch contract from the study docs.
- Set `comparison_png_path` to the session-gallery `compare_amp_phase_probe.png` artifact for this candidate, not to a nested `visuals/` path.
- For `run_config`, prepare a parameter-only rerun of the accepted architecture. Do not edit tracked source files just to encode a wrapper-level configuration experiment.
- For `run_config`, use unique output and log paths derived from `timestamp_utc` and `base_ref`, because there is no candidate commit hash to name the run.
- Keep `probe_mask` off unless the candidate is explicitly about probe masking.
- Do not fabricate evidence or claim improvement. This task only prepares and validates a candidate package.
- Do not stage or commit `state/` or `outputs/`.
- The accepted reference is the comparison target. Optimize for a better next accepted state, not for novelty.
- Set `base_ref` to the accepted git sha from `accepted_state.json` that you started from before finalizing the candidate package.
- Prefer deletions, simplifications, and cleaner mechanisms when they are plausible improvement paths.
- When two candidate directions look similarly promising, prefer the one with lower added complexity and lower incremental compute cost.
- If a change meaningfully increases complexity, be able to justify that cost in the `hypothesis`.
- If a candidate increases modes, width, depth, runtime, memory, or model size, justify why that extra cost is worth paying in the `hypothesis`.

Before finishing:
- verify the smoke check succeeded
- verify `candidate_metadata_path` exists and contains valid JSON
- if `candidate_kind` is `source`, verify the commit exists
- if `candidate_kind` is `source`, verify `candidate_paths_file` exists and contains a JSON list of relative file paths
- if `candidate_kind` is `run_config`, verify tracked source files are unchanged
