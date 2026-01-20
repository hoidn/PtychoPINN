**Summary**: Enforce CONFIG-001 bridging for the sim_lines runner + pipeline, then rerun gs1_ideal/gs2_ideal (least_squares scaling) and refresh analyzer + CLI evidence under the reserved 2026-01-20T160000Z hub.
**Focus**: DEBUG-SIM-LINES-DOSE-001 — C4f CONFIG-001 sync (Phase C intensity audit)
**Branch**: paper
**Mapped tests**: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
**Artifacts**: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/

**Do Now**
- Implement: scripts/studies/sim_lines_4x/pipeline.py::{run_scenario, run_inference} — import `update_legacy_dict` + `params as legacy_params`, then call `update_legacy_dict(legacy_params.cfg, train_config)` before `run_training(...)` and `update_legacy_dict(legacy_params.cfg, infer_config)` inside `run_inference(...)` immediately before `load_inference_bundle_with_backend(...)` so loader/grouping see the active ModelConfig values (specs/spec-inference-pipeline.md §1.1).
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::{main, run_inference_and_reassemble} — add the same import, call the bridging helper right before invoking `run_training(...)`, and inside `run_inference_and_reassemble(...)` before the grouped-data call so `legacy_params.get_padded_size()` reflects the current scenario overrides.
- Run: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs1_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs1_ideal --prediction-scale-source least_squares --group-limit 64` (uses baked stable profile). Archive stdout/stderr as `gs1_ideal_runner.log` under the hub and ensure all generated JSON/PNG/NPY files live beneath the gs1_ideal directory.
- Run: same command for `--scenario gs2_ideal --output-dir .../gs2_ideal --prediction-scale-source least_squares --group-limit 64` so both scenarios share the new CONFIG-001 plumbing; capture `gs2_ideal_runner.log` alongside outputs.
- Run: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs1_ideal --scenario gs2_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/gs2_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/` to regenerate `bias_summary.{json,md}` with the fresh telemetry.
- Run: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` and save the log as `pytest_cli_smoke.log` in the same hub.
- Update `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md` + `docs/fix_plan.md` Attempts with the new bridging evidence, noting the artifact paths (`2026-01-20T160000Z/gs*_ideal`, analyzer outputs, pytest log); mention that CONFIG-001 is now enforced prior to every training/inference dispatch.

**How-To Map**
1. scripts/studies/sim_lines_4x/pipeline.py — add `from ptycho import params as legacy_params` and `from ptycho.config.config import update_legacy_dict` near the imports (or alongside the existing inline imports) and drop in the two bridging calls before `run_training` and before the grouped-data path inside `run_inference`.
2. plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py — extend the existing `from ptycho.config.config import InferenceConfig, ModelConfig` import to include `update_legacy_dict`; call it after the final `train_config` dataclass adjustments (before `run_training(...)`) and inside `run_inference_and_reassemble(...)` right after constructing `infer_config` but before fetching grouped data.
3. Create the hub directory `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/` if it does not already exist; ensure each runner invocation writes into child directories (`gs1_ideal`, `gs2_ideal`). Use `tee` or shell redirection to capture `gs*_ideal_runner.log` next to the scenario outputs.
4. Analyzer command (above) expects the scenario directories to contain `run_metadata.json`, `intensity_stats.json`, `train_outputs/history_summary.json`, etc. Verify these files exist before running.
5. After analyzer + pytest complete, prepend a new entry in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/summary.md` summarizing what changed (CONFIG-001 hook, rerun results, analyzer highlights) and mirror the same snippet into `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md`. Update `docs/fix_plan.md` Attempts History under DEBUG-SIM-LINES-DOSE-001 with the timestamp, focus (C4f), and evidence paths.

**Pitfalls To Avoid**
1. Don’t mutate the plan-local runner’s scenario metadata paths—only add CONFIG-001 bridging; leave amplitude-scaling and telemetry wiring intact.
2. Ensure `update_legacy_dict` targets `legacy_params.cfg`; do not create a new dict copy or the legacy modules will still read stale values.
3. Keep the scenario directories under the specified artifacts hub; do not spill outputs into repo root or tmp/ beyond plan-local expectations.
4. Make sure both runner commands use `--prediction-scale-source least_squares` so the analyzer sees consistent scaling metadata for this loop.
5. Preserve existing plan-local CLI flags (group_limit, stable profiles); avoid reverting to large workloads that could OOM.
6. Analyzer output must overwrite `bias_summary.{json,md}` in the 2026-01-20T160000Z hub; don’t leave stale data from 150500Z.
7. When updating docs, record only the new attempt details—do not rewrite historical entries or archive logs from prior loops.
8. Always invoke the pytest selector from repo root with `AUTHORITATIVE_CMDS_DOC` set; omitting the env var violates the testing guide contract.
9. Do not edit `docs/index.md` or other specs this loop; scope is limited to CONFIG-001 bridging + evidence refresh.
10. Keep GPU utilization low by limiting parallel runs; complete gs1 before starting gs2.

**If Blocked**
- Record the failing command + stack trace under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/blocked.log`, update `docs/fix_plan.md` Attempts with the blocker (e.g., unexpected sim failure or analyzer missing files), and ping the supervisor before retrying.

**Findings Applied (Mandatory)**
- CONFIG-001 — specs/spec-inference-pipeline.md §1.1 requires every legacy entrypoint to run `update_legacy_dict(params.cfg, config)` before loader/training/inference.
- NORMALIZATION-001 — keep the analyzer outputs intact so we can continue verifying amplitude/intensity symmetry after the reruns.
- POLICY-001 — no environment swaps; continue using the existing TensorFlow stack and document any GPU pressure in the hub logs.

**Pointers**
- scripts/studies/sim_lines_4x/pipeline.py:287-371 — builds train_config + inference config for the nongrid study runner.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py:940-1240 — orchestrates training/inference and writes stats for each scenario.
- specs/spec-inference-pipeline.md:1-80 — CONFIG-001 contract governing params.cfg synchronization.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py:1-200 — analyzer CLI invoked after reruns.
- docs/fix_plan.md: DEBUG-SIM-LINES-DOSE-001 section — record new Attempts entry referencing hub 2026-01-20T160000Z.

**Next Up (optional)**
1. If CONFIG-001 bridging still doesn’t close the amplitude gap, scope C4g around loader normalization corrections using the refreshed telemetry.

