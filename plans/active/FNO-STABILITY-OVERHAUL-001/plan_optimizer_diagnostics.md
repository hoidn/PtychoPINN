# Stable Hybrid Optimizer + Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Instrument `stable_hybrid` training to either find an optimizer configuration that prevents the amp-collapse or capture actionable activation evidence for the remaining hypotheses in STABLE-LS-001.

**Architecture:** Extend the existing PyTorch training configs/CLI so the lightning module can swap optimizers (Adam, AdamW, SGD) and log their parameters; enhance `scripts/debug_fno_activations.py` so it can load saved `stable_hybrid` checkpoints and dump per-layer stats; then rerun Stage A arms with SGD and AdamW and immediately run activation captures on their checkpoints.

**Tech Stack:** Python 3.11, PyTorch Lightning (torch>=2.2), pytest, existing grid_lines compare harness, `scripts/debug_fno_activations.py` instrumentation.

---

### Task 1: Optimizer selection plumbing

**Files:**
- Modify: `ptycho/config/config.py` (`TrainingConfig` literal list + default optimizer and params)
- Modify: `ptycho_torch/config_params.py` + `ptycho_torch/config_bridge.py` (mirror optimizer fields)
- Modify: `ptycho_torch/config_factory.py` / `scripts/internal/train_config_serializers.py` (pass overrides)
- Modify: `scripts/studies/grid_lines_torch_runner.py` + `scripts/studies/grid_lines_compare_wrapper.py` (CLI flags)
- Modify: `ptycho_torch/model.py` (`PtychoPINN_Lightning.configure_optimizers` + helper for optimizer-specific kwargs)
- Tests: `tests/torch/test_config_bridge.py`, `tests/torch/test_grid_lines_torch_runner.py`, `tests/test_grid_lines_compare_wrapper.py`, `tests/torch/test_model_training.py`

**Step 1: Write failing config tests**
- Extend `tests/torch/test_config_bridge.py::TestConfigBridgeParity` with `test_training_config_optimizer_roundtrip` that sets `optimizer='sgd'`, `momentum=0.9`, `weight_decay=1e-4` and expects `config_bridge.to_training_config()` to preserve the fields.
- Update `tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment` to assert `setup_torch_configs()` copies CLI `--torch-optimizer adamw` etc into the `TrainingConfig`.
- Update `tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid` (or new test) to assert parse_args accepts `--torch-optimizer` and forwards it to runner stub.
- Add a lightning smoke test to `tests/torch/test_model_training.py` verifying `PtychoPINN_Lightning.configure_optimizers()` returns `torch.optim.SGD` when `training_config.optimizer='sgd'` (and that momentum is respected).

**Step 2: Run targeted pytest selectors (expect FAIL)**
```bash
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_training_config_optimizer_roundtrip -vv
pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_optimizer -vv
pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_optimizer -vv
pytest tests/torch/test_model_training.py::TestOptimizerSelection::test_configures_sgd -vv
```

**Step 3: Add optimizer fields + CLI flags**
- Add `optimizer: Literal['adam', 'adamw', 'sgd'] = 'adam'` plus `momentum: float = 0.9`, `weight_decay: float = 0.0`, `adam_beta1/beta2` fields in `TrainingConfig` (TF + Torch) and bridge them.
- Extend CLI parsers: in `grid_lines_torch_runner.py`, add `--torch-optimizer`, `--torch-weight-decay`, `--torch-momentum`, `--torch-beta1`, `--torch-beta2`; propagate into `TorchRunnerConfig` → `setup_torch_configs()`.
- Mirror flag names in `grid_lines_compare_wrapper.py` (prefix `--torch-` for all) so Stage A harness can forward values.

**Step 4: Update Lightning optimizer factory**
- Refactor `PtychoPINN_Lightning.configure_optimizers()` to branch on `training_config.optimizer`:
    - `adam`: keep existing behavior but allow overriding betas + weight decay.
    - `adamw`: instantiate `torch.optim.AdamW` with betas + weight decay.
    - `sgd`: instantiate `torch.optim.SGD` with lr, momentum, nesterov flag (`momentum>0`), and weight decay.
- Ensure scheduler attaches to whichever optimizer is returned (existing WarmupCosine block should reuse `optimizer`).
- Guard against unsupported strings with `ValueError` logged.

**Step 5: Update docs + help text**
- Mention new flags in `docs/workflows/pytorch.md` (Torch runner CLI table) and `docs/strategy/mainstrategy.md` Phase 8 section when you sync results later.

**Step 6: Re-run pytest selectors (expect PASS)**
Same commands as Step 2; archive logs under the initiative report hub.

**Step 7: Commit checkpoint** (`feat: torch optimizer selection` once tests pass)

---

### Task 2: Activation debug script upgrades

**Files:**
- Modify: `scripts/debug_fno_activations.py`
- Modify: `tests/torch/test_debug_fno_activations.py`

**Step 1: Extend the test**
- Add a new test case using PyTest paramization to invoke the script with `--architecture stable_hybrid` and `--checkpoint <tmp/model.pt>` to ensure it loads a saved state dict.
- Generate a tiny synthetic checkpoint by instantiating `StableHybridUNOGenerator`, saving `state_dict()` to tmp/model.pt, and ensuring the script loads it.

**Step 2: Script enhancements**
- Support `--architecture stable_hybrid` by importing `StableHybridUNOGenerator` and updating `_build_model()` to instantiate it.
- Add optional `--checkpoint` flag: when provided, load `torch.load(checkpoint)` and strip the `model.` prefix entries so they map onto the generator modules before calling `load_state_dict`.
- Add `--layerscale-init` override so instrumentation can match training hyperparams if needed.
- Write helper `_load_state_dict(model, checkpoint_path)` with descriptive error messages; ensure missing keys raise.

**Step 3: CLI ergonomics**
- Add `--output-json-name` optional override (defaults to `activation_report.json`) so multiple captures (baseline vs collapse) can live in the same folder.
- Document new flags at the top of the script and in `docs/plans/2026-01-27-fno-activation-debug-plan.md` appendix.

**Step 4: Run `pytest tests/torch/test_debug_fno_activations.py -vv` and archive the log.**

**Step 5: Commit checkpoint** (`feat: activation script loads stable_hybrid checkpoints`).

---

### Task 3: Stage A optimizer sweep (SGD + AdamW)

**Files / Paths:**
- Worktree: `.worktrees/fno2-phase8-optimizers`
- Outputs: `outputs/grid_lines_stage_a/arm_stable_sgd`, `outputs/grid_lines_stage_a/arm_stable_adamw`
- Artifacts hub: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T050000Z/`

**Step 1: Prep workspace**
- `mkdir -p` the two arm directories and rsync Stage A control datasets into each.
- Create the new reports hub with README listing shared hyperparams (N=64, gridsize=1, nimgs=1, fno_blocks=4, seed=20260128, scheduler=WarmupCosine(5,0.05)).

**Step 2: Run SGD arm**
```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --output-dir outputs/grid_lines_stage_a/arm_stable_sgd \
  --architectures stable_hybrid \
  --seed 20260128 \
  --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
  --nepochs 20 --torch-epochs 20 \
  --torch-learning-rate 3e-4 \
  --torch-optimizer sgd --torch-momentum 0.9 --torch-weight-decay 0.0 \
  --torch-scheduler WarmupCosine --torch-lr-warmup-epochs 5 --torch-lr-min-ratio 0.05 \
  --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
  --torch-infer-batch-size 8 --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
  2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T050000Z/stage_a_arm_stable_sgd.log
```
- After completion, copy `history.json`, `metrics.json`, `model.pt`, and grad norm CSV to the hub; run `scripts/internal/stage_a_dump_stats.py` to produce `stage_a_arm_stable_sgd_stats.json`.

**Step 3: Capture activations for SGD checkpoint**
```bash
python scripts/debug_fno_activations.py \
  --input outputs/grid_lines_stage_a/arm_control/datasets/N64/gs1/train/train.npz \
  --output plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T050000Z \
  --architecture stable_hybrid \
  --checkpoint outputs/grid_lines_stage_a/arm_stable_sgd/runs/pinn_stable_hybrid/model.pt \
  --batch-size 1 --max-samples 1 --device cpu \
  --output-json-name activation_report_sgd.json
```

**Step 4: Run AdamW arm** (same command but `--output-dir .../arm_stable_adamw`, `--torch-optimizer adamw --torch-weight-decay 0.01 --torch-beta1 0.9 --torch-beta2 0.999`). Capture logs, stats, and activations (JSON named `activation_report_adamw.json`).

**Step 5: Regression selectors**
- `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock::test_layerscale_grad_flow`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid`

Archive all CLI logs + pytest outputs in the new hub.

**Step 6: Update summaries**
- Append SGD/AdamW entries to `stage_a_metrics.json` (new file `stage_a_metrics_phase8.json`).
- Write `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T050000Z/stage_a_optimizer_summary.md` comparing Adam vs SGD vs previous Phase 7 metrics, calling out whether either run avoided collapse (target: amp_ssim ≥ 0.80).

**Step 7: Commit checkpoint** (`exp: stable_hybrid optimizer sweep phase8`).

---

### Task 4: Doc + plan sync

**Files:**
- `docs/strategy/mainstrategy.md`
- `docs/findings.md` (update STABLE-LS-001 or add new finding if optimizer succeeds)
- `docs/fix_plan.md` (Attempts History + FSM entry)
- `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` (Phase 8 section)
- `plans/active/FNO-STABILITY-OVERHAUL-001/summary.md` (new turn summary)
- `input.md` (next iteration)

**Steps:**
1. Summarize optimizer experiment outcomes and activation insights in strategy doc Stage A section.
2. Update finding STABLE-LS-001 with optimizer verdict; add STABLE-OPT-001 if SGD/AdamW fixes collapse.
3. Append Phase 8 status + artifacts to implementation plan + summary.
4. Record fix_plan attempt with the new reports hub + mapped tests.
5. Refresh `input.md` for subsequent execution per supervisor template.
6. Commit documentation updates (`docs: phase8 optimizer report`).
