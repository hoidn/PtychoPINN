Summary: Build the Phase B4 reassembly limits CLI so we can prove the gs2 offsets overflow the legacy padded-size math and record reassembly sum ratios for gs1 vs gs2.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
Branch: paper (sync with origin/paper)
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T050500Z/

Do Now (hard validity contract)
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py::main — add the B4 instrumentation script that rebuilds the SIM-LINES snapshot, logs padded-size math vs observed offsets, and runs the reassembly sum-preservation probe for `gs1_custom` and `gs2_custom`, saving JSON/Markdown plus the CLI log in the new hub.
- Pytest: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T050500Z/{reassembly_gs1_custom.json, reassembly_gs1_custom.md, reassembly_gs2_custom.json, reassembly_gs2_custom.md, reassembly_cli.log, pytest_sim_lines_pipeline_import.log}

How-To Map
1. export ARTIFACT_DIR=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T050500Z && mkdir -p "$ARTIFACT_DIR".
2. Implement `bin/reassembly_limits_report.py` (same snapshot-loading helpers as `grouping_summary.py`) so it:
   - builds RawData for the requested scenario, runs `update_legacy_dict(params.cfg, TrainingConfig(...))`, and captures `get_padded_size()`, `cfg['offset']`, and `cfg['max_position_jitter']`;
   - generates grouped data for train/test with consistent seeds, computes per-axis max offsets, and derives `required_canvas = math.ceil(N + 2 * max_offset)` alongside `fits_canvas = padded_size >= required_canvas` plus delta/ratio metrics;
   - slices the first `--group-limit` (default 64) samples, creates dummy complex patches (np.ones) shaped like grouped diffraction, and calls `tf_helper.reassemble_whole_object()` twice (`size=padded_size` vs `size=required_canvas`) to record total sums and percent loss when the canvas is undersized;
   - writes JSON + Markdown (metadata + per-subset stats + reassembly sum table) and prints a concise CLI stream (include both canvases’ sums) so we can tee it into `reassembly_cli.log`.
3. Run the CLI twice (identical seeds) with the new script:
   - python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py --scenario gs1_custom --group-limit 64 --output-json "$ARTIFACT_DIR/reassembly_gs1_custom.json" --output-markdown "$ARTIFACT_DIR/reassembly_gs1_custom.md" | tee "$ARTIFACT_DIR/reassembly_cli.log"
   - python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py --scenario gs2_custom --group-limit 64 --output-json "$ARTIFACT_DIR/reassembly_gs2_custom.json" --output-markdown "$ARTIFACT_DIR/reassembly_gs2_custom.md" | tee -a "$ARTIFACT_DIR/reassembly_cli.log"
4. pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$ARTIFACT_DIR/pytest_sim_lines_pipeline_import.log"

Pitfalls To Avoid
- No production edits: keep everything under plans/active/DEBUG-SIM-LINES-DOSE-001/bin/ and leave ptycho/ untouched.
- Always call `update_legacy_dict(params.cfg, config)` before touching legacy params; CONFIG-001 is mandatory for accurate `get_padded_size()` output.
- Limit the reassembly probe to ≤64 groups per subset so TensorFlow stays CPU-friendly; use float32 and avoid GPU-only assumptions.
- When computing offset maxima, use absolute values per axis before deriving the canvas requirement; don’t mix coords_offsets (global) with coords_relative (local) in the same metric.
- Ensure the CLI JSON schema extends existing patterns (metadata + per-subset dictionaries) so downstream analysis scripts can reuse them.
- Keep CLI stdout deterministic and tee-friendly (no progress bars); capture numerical outputs with reasonable precision.
- Do not delete previous artifact hubs or regenerate the Phase A snapshot; only write into 2026-01-16T050500Z/ for this loop.
- Respect PYTHON-ENV-001: invoke Python via `python`, not `python3.11` or virtualenv wrappers.
- Archive both CLI runs and the pytest log even if failures occur so the ledger has evidence.
- Avoid reusing legacy globals outside the CLI (e.g., clean up temporary params if the script stores state that could leak into other tests).

If Blocked
- If the CLI cannot import (missing module, TensorFlow error) or reassembly raises due to dtype/device constraints, capture the full traceback in $ARTIFACT_DIR/blocker.log, update docs/fix_plan.md Attempts with the command + error signature, set `<status>blocked</status>` here, and stop after saving the pytest output (even if pytest fails).

Findings Applied (Mandatory)
- CONFIG-001 — run `update_legacy_dict(params.cfg, TrainingConfig(...))` before RawData/grouping/reassembly so gridsize/neighbor_count/offsets reflect the scenario.
- MODULE-SINGLETON-001 — keep probe/raw-data creation scoped within the CLI; never rely on module-level singletons that persist between runs.
- NORMALIZATION-001 — do not change probe normalization; the CLI only reports geometry, so avoid rescaling diffraction beyond what synthetic_helpers already does.
- BUG-TF-REASSEMBLE-001 — this probe exercises `_reassemble_position_batched`; treat errors as telemetry only and do not patch tf_helper while we’re still proving the padded-size mismatch.

Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:106-136 — Phase B checklist defining the B4 reassembly limits deliverable and required outputs.
- docs/fix_plan.md:18-62 — Ledger status + latest Attempts History entry describing why B4 focuses on padded-size vs offsets.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/grouping_gs2_custom_default.md:13-36 — Evidence that gs2 offsets reach ≈381 px even though the legacy padded size is ~78 px.
- docs/specs/spec-ptycho-workflow.md:12-55 — Normative grouping + reassembly rules (`M ≥ N + 2·max(|dx|,|dy|)`), which the new CLI must cite rather than reinvent.
- plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md:1-8 — Current initiative summary describing the B4 scope and artifact hub reserved this loop.

Next Up (optional)
- If the reassembly limits evidence proves a padded-size deficit, prepare a C-phase Do Now to patch the legacy `get_padded_size()`/offset bridging logic.

Doc Sync Plan — not needed (no new pytest nodes added or renamed; guard already tracked).
Mapped Tests Guardrail: The CLI smoke selector collects (>0) and must stay green; keep it as the validation step for every Phase B instrumentation handoff.
Normative Math/Physics: Reference `docs/specs/spec-ptycho-workflow.md` §2–3 for grouping and §Reassembly Requirements for canvas math; don’t restate equations in ad-hoc prose—link to the spec in CLI output notes instead.
