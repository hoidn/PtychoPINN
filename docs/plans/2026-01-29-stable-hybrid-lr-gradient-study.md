# Stable Hybrid LR + Gradient Guard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether reducing the stable_hybrid learning rate or adding gradient clipping prevents the Stage A collapse documented in STABLE-LS-001.

**Architecture:** Reuse the existing Stage A grid_lines harness, cached datasets, and `stable_hybrid` architecture while varying only scheduler + clipping knobs so metrics remain comparable to prior arms. Capture stats with `stage_a_dump_stats.py` and archive under a new reports hub for reproducibility.

**Tech Stack:** Python 3.11, PyTorch Lightning runner (`scripts/studies/grid_lines_compare_wrapper.py`), rsync, pytest, jq/python for metrics aggregation.

---

### Task 1: Prep shared workspace + artifacts hub

**Files:**
- Modify: `outputs/grid_lines_stage_a/` (arm directories only)
- Create: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/README.md`
- Reference: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_metrics.json`

**Step 1:** Copy the canonical Stage A dataset from the control arm so each new sweep reuses identical NPZs.
```bash
rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ \
      outputs/grid_lines_stage_a/arm_stable_lowlr/datasets/
rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ \
      outputs/grid_lines_stage_a/arm_stable_warmup_lowlr/datasets/
rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ \
      outputs/grid_lines_stage_a/arm_stable_warmup_clip/datasets/
```

**Step 2:** Clear any stale runs so logs/model files never mix.
```bash
rm -rf outputs/grid_lines_stage_a/arm_stable_lowlr/runs
rm -rf outputs/grid_lines_stage_a/arm_stable_warmup_lowlr/runs
rm -rf outputs/grid_lines_stage_a/arm_stable_warmup_clip/runs
```

**Step 3:** Create the reports hub + README.
```bash
mkdir -p plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z
cat > plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/README.md <<'MARK'
Stage A LR + gradient guard sweep (stable_hybrid) — all runs reuse Stage A datasets and **must include `--set-phi`** so phase metrics stay meaningful.
- dataset: outputs/grid_lines_stage_a/arm_control/datasets (rsync copy per Task 1)
- seed: 20260128
- hyperparams vary per task (see individual logs)
- artifacts: stage_a_arm_stable_lowlr.log, stage_a_arm_stable_warmup_lowlr.log, stage_a_arm_stable_warmup_clip.log, *_stats.json, *_metrics.json
MARK
```

### Task 2: Constant low-LR baseline (no scheduler)

**Files:**
- Output: `outputs/grid_lines_stage_a/arm_stable_lowlr/`
- Artifacts: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_lowlr.*`

**Step 1:** Run the compare wrapper with LR halved and scheduler disabled.
```bash
AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --set-phi \
  --output-dir outputs/grid_lines_stage_a/arm_stable_lowlr \
  --architectures stable_hybrid \
  --seed 20260128 \
  --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
  --torch-epochs 20 --nepochs 20 \
  --torch-learning-rate 2.5e-4 \
  --torch-scheduler Default \
  --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
  --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
  --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
  2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_lowlr.log
```

**Step 2:** Archive run outputs.
```bash
cp outputs/grid_lines_stage_a/arm_stable_lowlr/runs/pinn_stable_hybrid/history.json \
   plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_lowlr_history.json
cp outputs/grid_lines_stage_a/arm_stable_lowlr/runs/pinn_stable_hybrid/metrics.json \
   plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_lowlr_metrics.json
cp outputs/grid_lines_stage_a/arm_stable_lowlr/runs/pinn_stable_hybrid/model.pt \
   plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_lowlr_model.pt
```

**Step 3:** Capture stats.
```bash
python scripts/internal/stage_a_dump_stats.py \
  --run-dir outputs/grid_lines_stage_a/arm_stable_lowlr/runs/pinn_stable_hybrid \
  --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_lowlr_stats.json
```

### Task 3: WarmupCosine with reduced peak LR

**Files:** same as Task 2 but under `arm_stable_warmup_lowlr`

**Step 1:** Execute the WarmupCosine run with the same LR cap and warmup parameters.
```bash
AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --set-phi \
  --output-dir outputs/grid_lines_stage_a/arm_stable_warmup_lowlr \
  --architectures stable_hybrid \
  --seed 20260128 \
  --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
  --torch-epochs 20 --nepochs 20 \
  --torch-learning-rate 2.5e-4 \
  --torch-scheduler WarmupCosine \
  --torch-lr-warmup-epochs 5 \
  --torch-lr-min-ratio 0.05 \
  --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
  --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
  --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
  2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_warmup_lowlr.log
```

**Step 2:** Copy `history.json`, `metrics.json`, and `model.pt` into the reports hub (same pattern as Task 2).

**Step 3:** Run the stats helper for this arm, outputting `stage_a_arm_stable_warmup_lowlr_stats.json`.

### Task 4: WarmupCosine + gradient clipping guard

**Files:** `outputs/grid_lines_stage_a/arm_stable_warmup_clip`

**Step 1:** Launch the WarmupCosine run with the original LR plus norm clipping to damp the spike.
```bash
AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --set-phi \
  --output-dir outputs/grid_lines_stage_a/arm_stable_warmup_clip \
  --architectures stable_hybrid \
  --seed 20260128 \
  --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
  --torch-epochs 20 --nepochs 20 \
  --torch-learning-rate 5e-4 \
  --torch-scheduler WarmupCosine \
  --torch-lr-warmup-epochs 5 \
  --torch-lr-min-ratio 0.05 \
  --torch-grad-clip 0.5 --torch-grad-clip-algorithm norm \
  --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
  --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
  2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_arm_stable_warmup_clip.log
```

**Step 2:** Archive `history.json`, `metrics.json`, `model.pt`, and run the stats helper to `stage_a_arm_stable_warmup_clip_stats.json`.

### Task 5: Aggregate metrics + doc sync

**Files:**
- Modify: `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`
- Modify: `docs/strategy/mainstrategy.md`
- Modify: `docs/fix_plan.md`
- Modify: `docs/findings.md` (if new evidence)
- Modify: `plans/active/FNO-STABILITY-OVERHAUL-001/summary.md`
- Modify: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/summary.md`
- Optional: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_metrics_phase7.json`

**Step 1:** Append the new arms to `stage_a_metrics.json` (or write a new `stage_a_metrics_phase7.json`). Example helper:
```bash
python - <<'PY'
import json, pathlib
arms = [
    ("stable_lowlr", "outputs/grid_lines_stage_a/arm_stable_lowlr/runs/pinn_stable_hybrid"),
    ("stable_warmup_lowlr", "outputs/grid_lines_stage_a/arm_stable_warmup_lowlr/runs/pinn_stable_hybrid"),
    ("stable_warmup_clip", "outputs/grid_lines_stage_a/arm_stable_warmup_clip/runs/pinn_stable_hybrid"),
]
rows = []
for name, run_dir in arms:
    run = pathlib.Path(run_dir)
    stats = json.loads((pathlib.Path("plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z") / f"stage_a_arm_{name}_stats.json").read_text())
    rows.append({"arm": name, **stats})
out = pathlib.Path("plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z/stage_a_metrics_phase7.json")
out.write_text(json.dumps(rows, indent=2))
PY
```

**Step 2:** Update documentation:
- Implementation plan: add Phase 7 status + references to this new plan file.
- Strategy doc: append a Stage A “Next Levers” subsection summarizing the three sweeps and criteria (e.g., success if amp_ssim > 0.8 and no post-epoch-7 collapse).
- Fix plan: log the attempt + set next action.
- Findings: close or update STABLE-LS-001 if collapse resolved; otherwise record observations (e.g., LR threshold) under the same ID.
- Summary files: prepend a new Turn Summary covering the sweep + pointing to artifacts.

**Step 3:** Regression selectors (unchanged code but required by policy):
```bash
pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock::test_layerscale_grad_flow -v
pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v
pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v
```
Archive the `pytest` logs alongside the run logs under the new reports hub.

**Step 4:** Commit docs + artifacts (if repo policy allows committing logs, otherwise leave logs under `.artifacts/`). Include selector outcomes in the commit message.
```bash
git add docs/strategy/mainstrategy.md docs/fix_plan.md docs/findings.md \
        plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md \
        plans/active/FNO-STABILITY-OVERHAUL-001/summary.md \
        plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T010000Z
git commit -m "FNO-STABILITY-OVERHAUL-001: Stage A LR sweep - tests: StableBlock/Runner/Wrapper"
```

**Step 5:** Update `input.md` for the next engineer iteration with the new focus + mapped tests (selectors above) referencing the new artifacts hub.

---

Plan complete and saved to `docs/plans/2026-01-29-stable-hybrid-lr-gradient-study.md`. Two execution options:

1. **Subagent-Driven (this session)** — dispatch tasks sequentially with reviews using superpowers:subagent-driven-development.
2. **Parallel Session** — open a fresh session/worktree, invoke superpowers:executing-plans, and work through the plan with checkpoints.

Which approach?
