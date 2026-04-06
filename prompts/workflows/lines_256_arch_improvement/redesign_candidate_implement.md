Use the injected redesign brief, redesign design note, and redesign implementation plan as the authoritative inputs.

Task:
- Implement the redesign candidate in the current workspace.
- Produce a candidate package that the outer controller can later smoke/score.

Required outputs:
- Write the execution session log to the `execution_session_log_path` named in the redesign brief.
- Write `candidate_metadata.json` to the `candidate_metadata_output_path` named in the redesign brief.
- If the candidate is a `source` candidate, also write `candidate_paths.json` to the `candidate_paths_output_path` named in the redesign brief.

`candidate_metadata.json` must contain:
- `status`: `READY` or `BLOCKED`
- `candidate_kind`: `source` or `run_config`
- `base_ref`
- `note`
- `hypothesis`
- `smoke_command`
- `smoke_output_root`
- `smoke_log_path`
- `run_command`
- `output_root`
- `log_path`
- `comparison_png_path`
- plus `candidate_paths_file` when `candidate_kind` is `source`

For `source` candidates, the outer controller resolves the authoritative candidate commit from the workspace `HEAD` after reading your metadata. Do not invent or guess a `candidate_commit` field. The stable provider-owned provenance surface is `candidate_paths.json`.

Execution log requirements:
- files changed
- commands executed and outcomes
- targeted checks run
- any remaining risks

Constraints:
- Do not write `proposal_result.json`; the workflow finalizer owns that.
- Keep changes focused on this candidate.
- Do not move queue items or mutate session/accepted-state ledgers.
