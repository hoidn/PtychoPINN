# FNO-STABILITY-OVERHAUL-001 — Phase 7 LR/Clipping Stage A Sweep

**Summary:** Run the three Phase 7 Stage A experiments (low LR, WarmupCosine low LR, WarmupCosine + clip) for `stable_hybrid`, then publish metrics and doc updates per the new LR/gradient guard plan.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 7 (LR + gradient guard for STABLE-LS-001)

**Branch:** fno2-phase7-lr-sweep

**Mapped tests:**
- `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock::test_layerscale_grad_flow -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/`

**Next Up (optional):** If all three runs finish early, capture activation traces at the epoch-7 collapse using `scripts/debug_fno_activations.py` for whichever arm still fails.

---

## Do Now — Phase 7 plan_lr_sweep.md Tasks 7.1–7.5

1. **Re-read the plan + findings:** Study `plans/active/FNO-STABILITY-OVERHAUL-001/plan_lr_sweep.md` (mirrors `docs/plans/2026-01-29-stable-hybrid-lr-gradient-study.md`) and Finding STABLE-LS-001 in `docs/findings.md` so the acceptance criteria (amp_ssim ≥0.80 without post-epoch-7 collapse) are fresh.
2. **Workspace prep:** From the `fno2-phase7-lr-sweep` worktree, sync datasets per Task 7.1:
   ```bash
   rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_stable_lowlr/datasets/
   rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_stable_warmup_lowlr/datasets/
   rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_stable_warmup_clip/datasets/
   rm -rf outputs/grid_lines_stage_a/arm_stable_lowlr/runs outputs/grid_lines_stage_a/arm_stable_warmup_lowlr/runs outputs/grid_lines_stage_a/arm_stable_warmup_clip/runs
   ```
   Ensure `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/README.md` captures shared seeds/params (seed 20260128, N=64, gridsize=1, nimgs_train/test=1, nphotons=1e9, epochs=20).
3. **Arm 1 — Constant low LR (no scheduler):**
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
   python scripts/studies/grid_lines_compare_wrapper.py \
     --N 64 --gridsize 1 \
     --output-dir outputs/grid_lines_stage_a/arm_stable_lowlr \
     --architectures stable_hybrid \
     --seed 20260128 \
     --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
     --nepochs 20 --torch-epochs 20 \
     --torch-learning-rate 2.5e-4 \
     --torch-scheduler Default \
     --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
     --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
     --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
     2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_lowlr.log
   ```
4. **Arm 2 — WarmupCosine with low LR:** Repeat with the WarmupCosine knobs and the same LR cap:
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
   python scripts/studies/grid_lines_compare_wrapper.py \
     --N 64 --gridsize 1 \
     --output-dir outputs/grid_lines_stage_a/arm_stable_warmup_lowlr \
     --architectures stable_hybrid \
     --seed 20260128 \
     --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
     --nepochs 20 --torch-epochs 20 \
     --torch-learning-rate 2.5e-4 \
     --torch-scheduler WarmupCosine \
     --torch-lr-warmup-epochs 5 \
     --torch-lr-min-ratio 0.05 \
     --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
     --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
     --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
     2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_warmup_lowlr.log
   ```
5. **Arm 3 — WarmupCosine + norm clip:** Repeat with the original LR and clipping:
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
   python scripts/studies/grid_lines_compare_wrapper.py \
     --N 64 --gridsize 1 \
     --output-dir outputs/grid_lines_stage_a/arm_stable_warmup_clip \
     --architectures stable_hybrid \
     --seed 20260128 \
     --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
     --nepochs 20 --torch-epochs 20 \
     --torch-learning-rate 5e-4 \
     --torch-scheduler WarmupCosine \
     --torch-lr-warmup-epochs 5 \
     --torch-lr-min-ratio 0.05 \
     --torch-grad-clip 0.5 --torch-grad-clip-algorithm norm \
     --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
     --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
     2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_warmup_clip.log
   ```
6. **Archive + stats:** For each arm copy `history.json`, `metrics.json`, `model.pt`, and grad-norm traces into the artifacts hub with descriptive filenames, then run:
   ```bash
   python scripts/internal/stage_a_dump_stats.py --run-dir <arm_run_dir> --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_<arm>_stats.json
   ```
   Aggregate the rows into `stage_a_metrics_phase7.json` (or append to the existing metrics file) and update `stage_a_summary.md` with a new table highlighting which arms (if any) avoided collapse.
7. **Docs + findings sync:** Refresh `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` (Phase 7 status), `plans/active/FNO-STABILITY-OVERHAUL-001/summary.md`, `docs/strategy/mainstrategy.md`, `docs/fix_plan.md`, and `docs/findings.md` (STABLE-LS-001). Document whether LR/clipping resolved the failure mode; if not, record quantitative LR thresholds to justify next hypotheses.
8. **Testing + hygiene:** Run the mapped pytest selectors above (archive logs under the artifacts directory) and ensure git only stages intentional changes plus the new reports. Keep large artifacts out of git; everything for this loop stays under `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/`.
