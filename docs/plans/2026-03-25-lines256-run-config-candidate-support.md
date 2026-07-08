# Lines 256 Run-Config Candidate Support Plan

Goal: let the `lines_256` workflow test parameter-only reruns of the accepted architecture without forcing a fake source edit or candidate commit.

Scope:
- add a first-class `candidate_kind` contract with `source` and `run_config`
- make keep/discard handling no-op for `run_config` candidates
- keep source-edit behavior unchanged for `source` candidates
- update prompts/docs so a pure wrapper-level higher-`fno_modes` rerun is a valid next experiment

Implementation outline:
1. Add failing tests for helper behavior and workflow contract shape.
2. Update `lines_256_handle_candidate_outcome.py` to branch on `candidate_kind`.
3. Update both workflow YAMLs so experiment/debug metadata expose `candidate_kind` and `candidate_paths_file` is only required for `source`.
4. Update prompts/docs to allow parameter-only candidates and describe the two candidate kinds.
5. Re-run focused tests and both workflow dry-run validations.
