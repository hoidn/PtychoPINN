# DEBUG-SIM-LINES-DOSE-001 — Phase D1 Loss Config Validation

## Summary
Capture the actual `dose_experiments` loss weights (default `loss_fn='nll'` plus the MAE override branch) and fix the comparison CLI so conditional assignments stop masquerading as defaults before we adjust sim_lines losses.

## Focus
DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase D amplitude bias investigation)

## Branch
paper

## Mapped Tests
`pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

## Artifacts
`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/`

## Do Now (Phase D1a–D1c)
- Implement: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py::main` — add helpers that execute the captured `dose_experiments_param_scan.md` module in-process with a stubbed `ptycho.params.cfg`, call `init()` twice (`loss_fn='nll'` and `'mae'`), and record the resulting cfg dictionaries without touching the production environment. Use the `loss_fn='nll'` snapshot as the canonical legacy baseline in the existing Markdown/JSON diff and add a new section emitting both loss modes so conditional assignments (MAE branch) are clearly labeled.
- Capture: Write the raw legacy snapshots to `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/dose_loss_weights.json` plus a short Markdown summary explaining which weights change between `nll` and `mae`. Rerun the CLI command: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py --snapshot plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json --dose-config plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_param_scan.md --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/loss_config_diff.md --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/loss_config_diff.json`.
- Validate: Re-run the CLI smoke guard via `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v 2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/pytest_cli_smoke.log`, then update `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/summary.md` with the corrected loss-weight interpretation and note whether H-LOSS-WEIGHT remains viable.

## How-To Map
1. **Stub the legacy cfg safely** — Import `sys`, `types`, and `importlib.util`. Before executing the captured script, save `sys.modules['ptycho.params']`, register a lightweight module (`types.ModuleType('ptycho.params')`) exposing a dict `cfg = {}` plus any attributes the script expects, and restore the original module in a `finally` block so the real `params.cfg` stays untouched.
2. **Load and call `init()` twice** — Use `runpy.run_path` or `exec(compile(...), module_dict)` to load `dose_experiments_param_scan.md`, retrieve its `init` function, and call it with `nphotons=1e9` (matches the snapshot) for both `loss_fn` options. Deep-copy the stub cfg after each call so you can emit JSON (keys sorted for repeatability) and Markdown tables describing the legacy defaults.
3. **Integrate with the diff** — Replace the previous `parse_dose_config()` output with the captured `loss_fn='nll'` snapshot, add a `dose_loss_modes` entry in the JSON payload (`{"nll": {...}, "mae": {...}}`), and append a Markdown section that tabulates `mae_weight/nll_weight/realspace_*` per loss_fn with clear labels such as “legacy override when loss_fn='mae'”. Keep the existing scenario table unchanged except for removing the misleading MAE delta.
4. **Archive evidence** — Save `dose_loss_weights.json` + `.md` beside the refreshed diff outputs, mention the corrected interpretation in `summary.md`, and ensure the pytest log lands under the same hub for traceability.

## Pitfalls To Avoid
- Do not leave the stubbed `ptycho.params` module in `sys.modules`; always restore it even if `init()` raises.
- Never mutate or serialize the real `params.cfg`; work with copies inside the stub so production code stays untouched.
- Avoid executing heavy functions from the legacy script (only call `init()`), otherwise you may import unavailable TensorFlow Addons stubs.
- Keep the CLI backward compatible: flags and JSON schema should extend rather than break earlier consumers.
- Clearly mark conditional values in Markdown; do not imply MAE branch weights apply to default runs.
- Remember to set `AUTHORITATIVE_CMDS_DOC` on every CLI/pytest command.
- Store every generated artifact (JSON/MD/pytest log) under the 2026-01-20T112029Z hub without overwriting prior evidence.
- Use deterministic ordering when dumping JSON so future diffs stay stable.
- Run the pytest guard after regenerating the diff to prove imports still work.

## If Blocked
Capture the failing command plus traceback under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/blocker.log`, note whether the blocker is due to module import or script execution, and update docs/fix_plan.md Attempts History + plan summary before pausing the focus.

## Findings Applied (Mandatory)
- CONFIG-001 — Stubbed execution must not desync the real `params.cfg`; restore the module immediately after snapshotting.
- NORMALIZATION-001 — We are validating loss weights before touching normalization, preserving the policy of changing one axis at a time.
- SIM-LINES-CONFIG-001 — Keep the sim_lines pipeline import guard running to ensure CONFIG-001 bridging stays intact while the CLI evolves.

## Pointers
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:320` — Phase D checklist with new D1a–D1c tasks.
- `docs/fix_plan.md:205` — Attempts history entries describing the D1 correction.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py:1` — CLI to update with legacy snapshot logic.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/dose_experiments_param_scan.md:1` — Captured legacy script whose `init()` we need to execute.

## Next Up (Optional)
1. Phase D2 — instrumentation for normalization parity once loss weights are confirmed.
2. Phase D3 — hyperparameter delta audit (epochs, batch size, scheduler) if H-LOSS-WEIGHT is cleared.

