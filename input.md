Summary: Enable amplitude supervision by setting realspace_weight > 0 in the sim_lines training config, then rerun gs2_ideal to verify amplitude improvement.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D6a realspace_weight fix
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T220000Z/

Do Now (DEBUG-SIM-LINES-DOSE-001 / D6a — see implementation.md §Phase D, D6 entry):

1. **Implement:** Update `scripts/studies/sim_lines_4x/pipeline.py::build_training_config()` to set `realspace_weight=0.1`:
   - At line ~196, after `TrainingConfig(`, add the loss weight parameter:
     ```python
     return TrainingConfig(
         model=model_config,
         n_groups=group_count,
         nphotons=params.nphotons,
         neighbor_count=params.neighbor_count,
         nepochs=nepochs,
         output_dir=output_dir,
         realspace_weight=0.1,  # Enable amplitude supervision (D6a fix)
     )
     ```
   - Reference: `ptycho/train_pinn.py:56` uses `realspace_weight=0.1` for PINN training.

2. **Implement:** Update `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::build_training_config()` (lines ~176-203) with the same fix so plan-local scenarios also use amplitude supervision.

3. **Validate:** Rerun gs2_ideal with the new loss weighting:
   ```bash
   python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py \
     --scenario gs2_ideal \
     --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T220000Z/gs2_ideal \
     --prediction-scale-source least_squares \
     2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T220000Z/logs/gs2_ideal_runner.log
   ```

4. **Compare metrics:** After the run completes, compare the new metrics against D5b baseline:
   - D5b baseline: `output_vs_truth_ratio=0.234` (predictions 23.4% of truth)
   - Expected improvement: ratio should increase significantly (ideally > 0.8)
   - Check `run_metadata.json::comparison::metrics::amplitude` for:
     - `mae` (should decrease)
     - `pearson_r` (should increase toward 1.0)
     - `pred_stats.mean` vs `truth_stats.mean` (should be closer)

5. **Test:** Run CLI smoke test:
   ```bash
   pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
     | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T220000Z/logs/pytest_cli_smoke.log
   ```

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
2. Edit `scripts/studies/sim_lines_4x/pipeline.py`:196 to add `realspace_weight=0.1`
3. Edit `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py` in `build_training_config()` to add the same
4. Create logs directory: `mkdir -p plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T220000Z/logs`
5. Run gs2_ideal scenario
6. Run pytest smoke test
7. Extract and document the comparison between D5b baseline and D6a results

Pitfalls To Avoid:
- The `TrainingConfig` dataclass must already have a `realspace_weight` field (check `ptycho/config/config.py:118`). If not, you need to add it.
- Don't modify core modules (`ptycho/model.py`, `ptycho/params.py`) — only update the pipeline configs.
- The training may take longer with realspace_loss enabled, but 5 epochs should complete in under 2 minutes on GPU.
- If training hits NaN loss, try reducing `realspace_weight` to 0.01 first — but NaN is unlikely since D5b verified the model trains successfully.
- After the run, inspect both the amplitude metrics AND the loss history to verify realspace_loss is being computed (should show non-zero values).

If Blocked: If `TrainingConfig` doesn't support `realspace_weight`, document the obstacle and propose adding the field to the dataclass.

Findings Applied (Mandatory):
- H-LOSS-WEIGHTS (D6 finding) — `realspace_weight=0.0` confirmed as root cause of amplitude gap
- PINN-CHUNKED-001 — training should use streaming mode for large datasets
- SIM-LINES-CONFIG-001 — all training configs flow through the same path

Pointers:
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T220000Z/summary.md (D6 root cause analysis)
- ptycho/config/config.py:118 (realspace_weight default)
- ptycho/train_pinn.py:56 (reference for realspace_weight=0.1)
- ptycho/model.py:597-601 (loss compilation with realspace_weight)
- docs/fix_plan.md:477 (D6 entry with root cause details)

Expected Outcome:
- With `realspace_weight=0.1`, the model will optimize `realspace_loss(trimmed_obj, Y_I_centered)` in addition to NLL loss.
- This should directly supervise the object amplitude, significantly reducing the ~4.3× amplitude gap observed in D5b.
- If the fix works, `output_vs_truth_ratio` should increase from 0.234 toward 1.0, and amplitude MAE should decrease.

Next Up (optional): If D6a succeeds, consider making `realspace_weight` a CLI argument for the runner script to enable experimentation with different values.
