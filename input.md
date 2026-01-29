# FNO-STABILITY-OVERHAUL-001 — Phase 4: Stage B Deep Control Run

**Summary:** Run the Stage B deep-control experiment (hybrid, fno_blocks=8) with full instrumentation and publish the Stage B summary.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 3 (Stage B Stress Test)

**Branch:** fno2

**Mapped tests:**
- `pytest tests/torch/test_fno_generators.py -k stable -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_grad_clip_algorithm -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/`

**Next Up (optional):** LayerScale spike for stable_hybrid if Stage B confirms control stability.

---

## Do Now (Phase 4 Tasks 4.1–4.4)

1. **Prep Stage B workspace (Task 4.1)**
   - Ensure the artifacts hub exists: `mkdir -p plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z`.
   - Create Stage B output dir and reuse the Stage A dataset/probe:
     ```bash
     mkdir -p outputs/grid_lines_stage_b/deep_control
     rsync -a --delete outputs/grid_lines_stage_a/arm_control/datasets/ \
       outputs/grid_lines_stage_b/deep_control/datasets/
     rm -rf outputs/grid_lines_stage_b/deep_control/runs
     ```
   - Update the README in the artifacts hub if any hyperparameters change (currently: N=64, gridsize=1, nimgs_train/test=2, nphotons=1e9, seed=20260128, loss=MAE, clip=norm 1.0, fno_blocks=8).

2. **Run the deep control arm with grad-norm logging (Task 4.2)**
   - Command (mirrors Stage A control but `--fno-blocks 8` + grad norm instrumentation):
     ```bash
     AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
     python scripts/studies/grid_lines_compare_wrapper.py \
       --N 64 --gridsize 1 \
       --output-dir outputs/grid_lines_stage_b/deep_control \
       --architectures hybrid \
       --seed 20260128 \
       --nimgs-train 2 --nimgs-test 2 --nphotons 1e9 \
       --nepochs 50 --torch-epochs 50 \
       --fno-blocks 8 \
       --torch-grad-clip 1.0 --torch-grad-clip-algorithm norm \
       --torch-loss-mode mae --torch-infer-batch-size 8 \
       --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
       2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/stage_b_deep_control.log
     ```
   - Copy `history.json`, `metrics.json`, and `model.pt` from `outputs/grid_lines_stage_b/deep_control/runs/pinn_hybrid/` into the artifacts hub.

3. **Capture stats + Stage B summary (Task 4.3)**
   - Quick stats helper:
     ```bash
     python scripts/internal/stage_a_dump_stats.py \
       --run-dir outputs/grid_lines_stage_b/deep_control/runs/pinn_hybrid \
       --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/stage_b_deep_control_stats.json
     ```
   - Extend the Stage A metrics snippet to emit `stage_b_metrics.json` (single row with best val_loss, amp/phase SSIM, amp MAE, grad_norm extrema) under the same artifacts path.
   - Author `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/stage_b_summary.md` comparing Stage A control vs Stage B deep control (table + narrative). Cite `docs/strategy/mainstrategy.md §Stage B` for success criteria and conclude whether instability emerges at 8 blocks. If stable, recommend the next experiment (e.g., multi-seed sweep or higher depth); if unstable, propose mitigation (AGC rerun or LayerScale).
   - Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` (§Phase 4 status), `docs/strategy/mainstrategy.md` (Stage B outcome), and `docs/fix_plan.md` attempts history accordingly. Append any durable lessons (e.g., gradient drift behavior) to `docs/findings.md`.

4. **Regression guard + artifact drop (Task 4.4)**
   - Re-run the mapped selectors listed above (same commands) and archive their logs under the artifacts hub per `docs/TESTING_GUIDE.md`.
   - Ensure the Stage B CLI log, stats JSON, metrics JSON, copied run artifacts, and pytest logs are all present in `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/`.
