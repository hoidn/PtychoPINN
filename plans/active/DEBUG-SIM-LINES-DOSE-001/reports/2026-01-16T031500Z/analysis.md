# Phase B1 Findings — Grid vs Nongrid Grouping

1. **SIM-LINES default scenario works exactly as expected.** Running `bin/grouping_summary.py` with the snapshot defaults (`gs1_custom`, gridsize=1, 2000 total images, 1000 requested groups) produced `actual_groups=1000` in both splits. The JSON report shows clean ranges for `coords_offsets` (≈10–382 px) and the CLI confirmed no warnings.
2. **Dose-experiments overrides fail immediately because the nongrid grouping pipeline cannot fabricate gridsize=2 groups from only two points per split.** When overriding the snapshot to `gridsize=2`, `group_count=2`, and `total_images=4` (matching the legacy config’s `nimgs_train=nimgs_test=2`), each subset errors with `Dataset has only 2 points but 4 coordinates per group requested.` This is consistent with KDTree grouping requiring at least `gridsize**2` samples per group, while the grid-based script pre-built four-patch groups in-place.
3. **Implication.** We can’t literally replay the legacy sampling regime inside the nongrid helper; to reach parity we either (a) provision at least `nsamples * gridsize**2` points or (b) add an explicit grid-mode branch. For Phase B we will stay in the evidence lane and keep the KDTree pipeline but record where the mismatch shows up.

# Phase B2 Scope — Probe Normalization A/B

Goal: quantify the difference between the legacy `set_default_probe()` normalization (disk probe, params-driven scaling) and the modern `make_probe`/`normalize_probe_guess` path used by sim_lines_4x. Holding the scenario constants (N, probe_scale, probe_big/mask), capture per-mode amplitude stats so we know whether the change in probe construction explains the reconstruction divergence.

Deliverables for B2:
- Plan-local CLI `bin/probe_normalization_report.py` that loads the snapshot scenarios, generates both probes (legacy vs modern), and writes JSON/Markdown summaries with amplitude min/max/mean, L2 norm, and relative ratios.
- Runs for `gs1_custom`, `gs1_ideal`, `gs2_custom`, and `gs2_ideal`, with artifacts saved under this hub.
- Guard test: rerun `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` after the CLI lands.
