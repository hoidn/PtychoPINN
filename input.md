# FNO-STABILITY-OVERHAUL-001 — Phase 4: Capped Stage B Deep Control

**Summary:** Run the Stage B deep-control arm with the new `max_hidden_channels` cap and capture metrics/logs so we can decide the next stability move.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 4 (Channel Cap rerun + Stage B logging)

**Branch:** fno2

**Mapped tests:**
- `pytest tests/torch/test_fno_generators.py::TestHybridUNOGenerator::test_max_hidden_channel_cap -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_capped_channels -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_max_hidden_channels -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T210000Z/`

**Next Up (optional):** If the capped Stage B run is stable, prep the LayerScale spike for `stable_hybrid` per docs/strategy/mainstrategy.md §LayerScale.

---

## Do Now — Stage B Deep Control with max_hidden_channels

1. **Sanity-check the channel-cap plumbing (tests + audit)**
   - Review `ptycho_torch/generators/fno.py`, `scripts/studies/grid_lines_torch_runner.py`, and `scripts/studies/grid_lines_compare_wrapper.py` to confirm `max_hidden_channels` is threaded end-to-end (ModelConfig → TorchRunnerConfig → CLI flag). This keeps legacy behavior when the flag is omitted.
   - Run the mapped selectors above; tee each `pytest` invocation to its own log (e.g., `pytest_hybrid_cap.log`) under `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T210000Z/` per `docs/TESTING_GUIDE.md`.

2. **Prepare the Stage B workspace (reuse Stage A dataset)**
   - Ensure `outputs/grid_lines_stage_b/deep_control` exists; wipe any stale runs: `rm -rf outputs/grid_lines_stage_b/deep_control/runs`.
   - Re-copy the Stage A control dataset so Stage B uses identical NPZs: `rsync -a --delete outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_b/deep_control/datasets/`.
   - Verify README in the 210000Z hub still lists the capped hyperparameters (N=64, gridsize=1, fno_blocks=8, max_hidden_channels=512).

3. **Run the capped Stage B deep-control arm** (see `docs/strategy/mainstrategy.md §Stage B` for the success criteria)
   - Execute the grid-lines compare wrapper with the new max-hidden flag and grad-norm logging:
     ```bash
     AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
     python scripts/studies/grid_lines_compare_wrapper.py \
       --N 64 --gridsize 1 \
       --output-dir outputs/grid_lines_stage_b/deep_control \
       --architectures hybrid \
       --seed 20260128 \
       --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
       --nepochs 50 --torch-epochs 50 \
       --fno-blocks 8 --torch-max-hidden-channels 512 \
       --torch-grad-clip 1.0 --torch-grad-clip-algorithm norm \
       --torch-loss-mode mae --torch-infer-batch-size 8 \
       --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
       2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T210000Z/stage_b_deep_control_max512.log
     ```
   - After the run, copy `runs/pinn_hybrid/{history.json,metrics.json,model.pt}` plus any auxiliary metrics (e.g., stitched outputs) into the 210000Z hub. Capture the top-level `outputs/grid_lines_stage_b/deep_control/metrics.json` as well.

4. **Capture stats + comparative summary**
   - Generate quick stats for the capped run:
     ```bash
     python scripts/internal/stage_a_dump_stats.py \
       --run-dir outputs/grid_lines_stage_b/deep_control/runs/pinn_hybrid \
       --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T210000Z/stage_b_deep_control_max512_stats.json
     ```
   - Extend the Stage A metrics helper (implementation plan §3.5) to emit `stage_b_metrics.json` under the same hub. Compare against the Stage A control numbers stored in `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_metrics.json` (val_loss, amp/phase SSIM, MAE, grad norms).
   - Update `stage_b_summary.md` in the 210000Z hub with a table showing Stage A control vs capped Stage B (val_loss, SSIMs, MAE, grad_norm max/median) and note whether depth-induced instability appears once memory is resolved. Call out any anomalies (NaNs, grad spikes).

5. **Docs + plan sync**
   - Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` Phase 4 to record the capped Stage B outcome (Task 4.5). If the cap solves the OOM, mark the task complete; if not, document the fallback (e.g., fno_blocks=6) and add next steps.
   - Refresh `docs/strategy/mainstrategy.md` Stage B section with the new metrics and decision tree (continue depth scaling vs try LayerScale/AGC). Reference the exact Stage B paragraph when editing so spec remains authoritative.
   - Append a new Attempts History bullet plus supervisor state update in `docs/fix_plan.md` describing the capped run and linking to the 210000Z artifacts.

6. **Knowledge base + hygiene**
   - If the capped depth reveals a durable lesson (e.g., "512-channel cap stabilizes fno_blocks=8" or "still OOM, need fno_blocks=6"), add a Finding (e.g., `FNO-DEPTH-002`) to `docs/findings.md` with log paths.
   - Keep the repo clean: do not commit datasets, but ensure the artifacts hub has README, log, stats JSON, model snapshot, `stage_b_metrics.json`, and pytest logs.
   - When everything passes, summarize successes/risks in `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T210000Z/summary.md` (append below today’s Turn Summary) so the supervisor can audit next loop.
