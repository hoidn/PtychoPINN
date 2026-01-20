## Summary
Close Phase D1 by proving the legacy `ptycho.params.cfg` defaults match the sim_lines loss weights so we can move on to normalization work.

## Focus
DEBUG-SIM-LINES-DOSE-001 — Phase D amplitude bias investigation

## Branch
paper

## Mapped tests
- pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v

## Artifacts
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/

## Do Now — DEBUG-SIM-LINES-DOSE-001.D1 (Loss-config parity)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py::main — extend the CLI with a helper that deep-copies `ptycho.params.cfg` so the Markdown/JSON diff includes the real framework defaults (mae_weight/nll_weight/realspace_*); add a `--output-legacy-defaults` flag that writes those values to the artifacts hub alongside the existing runtime loss snapshots.
- Verify: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/pytest_cli_smoke.log
- Archive: Re-run `compare_sim_lines_params.py` with the sim_lines snapshot + `dose_experiments_param_scan.md`, writing the refreshed `loss_config_diff.{md,json}`, `dose_loss_weights.json`, new `legacy_params_cfg_defaults.json`, and a short `summary.md` noting the matching defaults (cite ptycho/params.py:64) under plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/.

## How-To Map
1. Edit plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py: add `capture_legacy_params_defaults()` (imports `from ptycho import params as legacy_params`, returns deep copy of cfg keys), wire `--output-legacy-defaults`, and update the Markdown/JSON builders to append a "Legacy params.cfg defaults" table + JSON node when defaults are captured.
2. Re-run the diff:
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py \
     --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json \
     --dose-config plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_param_scan.md \
     --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/loss_config_diff.md \
     --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/loss_config_diff.json \
     --output-dose-loss-weights plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/dose_loss_weights.json \
     --output-legacy-defaults plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/legacy_params_cfg_defaults.json
   ```
3. Summarize findings in plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/summary.md (reference that `ptycho/params.py:64` sets mae_weight=0, nll_weight=1, matching TrainingConfig defaults) so we can mark D1 complete.
4. Run the pytest selector and stash the log as noted above.

## Pitfalls To Avoid
- Do NOT mutate `ptycho.params.cfg`; copy values then leave globals untouched.
- Keep the stubbed `execute_legacy_init_with_stubbed_cfg` path intact—new defaults capture must not break Phase D1a logic.
- Reference file paths via repo root; no absolute `/tmp` outputs except existing TrainingConfig placeholders.
- Maintain JSON determinism (sort keys) so diffs stay reviewable.
- Ensure new CLI args have defaults so existing automation (reports/2026-01-20T110227Z) can be reproduced later.
- Avoid introducing dependency on environment-specific interpreters; run via PATH `python` per PYTHON-ENV-001.
- Guard new markdown sections with clear headings; reviewers expect separate “runtime vs defaults” sections.
- Remember CONFIG-001: never call legacy modules with partially constructed configs (read-only copy is safe).
- Don’t skip the pytest guard even though changes touch only plan-local tooling.
- Keep artifacts small/plaintext; large raw dumps belong under .artifacts/ per hygiene rules.

## If Blocked
Capture the exact failure (traceback or CLI output) in plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T170000Z/blocker.log, note whether the import of `ptycho.params` or the CLI run failed, and ping me so we can decide whether to fall back to manual code citation.

## Findings Applied
- CONFIG-001 — treat legacy `params.cfg` as source of truth; copy values without mutating shared state before comparing against dataclass configs.
- NORMALIZATION-001 — reiterates that amplitude bias stems from normalization/loss pathways, hence we must prove loss weights truly match before moving to normalization probes.
- SIM-LINES-CONFIG-001 — all sim_lines runners/scripts must continue calling `update_legacy_dict` before legacy hand-offs; today’s tooling update should reinforce that assumption by showing defaults already align.

## Pointers
- specs/spec-ptycho-core.md:86 — Normative loss composition + normalization invariants.
- docs/DEVELOPER_GUIDE.md:157 — Three-tier normalization architecture (physics vs statistical vs display).
- docs/DATA_NORMALIZATION_GUIDE.md:1 — Detailed normalization responsibilities and pitfalls.
- docs/findings.md:CONFIG-001/SIM-LINES-CONFIG-001 entries — prior evidence that cfg sync fixes NaNs.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:320 — Phase D checklist defining D1 deliverables.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/summary.md — Existing runtime loss-mode evidence that needs the params.cfg corroboration.

## Next Up (optional)
- DEBUG-SIM-LINES-DOSE-001.D2 — instrument normalization stage parity once loss weights are definitively ruled out.
