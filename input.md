# FNO-STABILITY-OVERHAUL-001 — Phase 3: Stage A Diagnostics + AGC Arm

**Summary:** Reproduce the stable_hybrid arm with instrumentation, run the AGC arm, and publish the Stage A metrics/summary to pick the Stage B candidate.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 3 (Stage A Validation)

**Branch:** fno2

**Mapped tests:**
- `pytest tests/torch/test_fno_generators.py -k stable -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_grad_clip_algorithm -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/`

**Next Up (optional):** Stage B 8-block rerun once Stage A winner is documented.

---

## Do Now (Stage A Tasks 3.3–3.5)

1. **Stable arm rerun + diagnostics**
   - Remove / archive `outputs/grid_lines_stage_a/arm_stable/runs/pinn_stable_hybrid/` so the rerun doesn’t mix artifacts.
   - Re-run Task 3.3 with grad-norm instrumentation and capture stdout/stderr:
     ```bash
     AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
     python scripts/studies/grid_lines_compare_wrapper.py \
       --N 64 --gridsize 1 \
       --output-dir outputs/grid_lines_stage_a/arm_stable \
       --architectures stable_hybrid \
       --seed 20260128 \
       --nimgs-train 2 --nimgs-test 2 --nphotons 1e9 \
       --nepochs 50 --torch-epochs 50 \
       --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
       --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
       --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
       2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_arm_stable.log
     ```
   - After the run, compute quick stats so we can reason about the collapse without loading notebooks:
     ```bash
     python scripts/internal/stage_a_dump_stats.py \
       --run-dir outputs/grid_lines_stage_a/arm_stable/runs/pinn_stable_hybrid \
       --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_arm_stable_stats.json
     ```
     (If `scripts/internal/stage_a_dump_stats.py` doesn’t exist yet, add a tiny helper that prints amp/phase mean/std and reports whether `StablePtychoBlock.norm.weight` stayed near zero.)

2. **Task 3.4 — AGC arm execution**
   - Refresh the shared dataset before running:
     ```bash
     rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_agc/datasets/
     rm -rf outputs/grid_lines_stage_a/arm_agc/runs
     ```
   - Run the AGC arm and capture its log:
     ```bash
     AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
     python scripts/studies/grid_lines_compare_wrapper.py \
       --N 64 --gridsize 1 \
       --output-dir outputs/grid_lines_stage_a/arm_agc \
       --architectures hybrid \
       --seed 20260128 \
       --nimgs-train 2 --nimgs-test 2 --nphotons 1e9 \
       --nepochs 50 --torch-epochs 50 \
       --torch-grad-clip 0.01 --torch-grad-clip-algorithm agc \
       --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
       2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_arm_agc.log
     ```
   - Archive `outputs/grid_lines_stage_a/arm_agc/runs/pinn_hybrid/{history.json,metrics.json,model.pt}` and `metrics.json` under the artifacts hub.

3. **Task 3.5 — Metrics aggregation + Stage A summary**
   - Run the updated snippet from `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` §3.5 (pulls best `val_loss` from each `runs/.../history.json`) to create `stage_a_metrics.json`.
   - Author `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_summary.md` with:
     - A table showing control vs stable vs AGC (best val_loss, phase SSIM, amp MAE, inference notes).
     - A short narrative (cite `docs/strategy/mainstrategy.md §2`) recommending which arm proceeds to Stage B and why stable_hybrid underperformed (include the amp/phase variance evidence).
   - Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` Phase 3 checklist and `docs/fix_plan.md` Attempts History with the Stage A outcome. If AGC wins, queue Stage B tasks in the plan.

4. **Regression guard**
   - Re-run the mapped selectors above to prove the `stable_hybrid` plumbing + grad-clip flag forwarding still pass.
   - Drop the pytest logs plus the three CLI logs/metrics artifacts into `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/` per `docs/TESTING_GUIDE.md`.
