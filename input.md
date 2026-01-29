# FNO-STABILITY-OVERHAUL-001 — Phase 4: Task 4.5 Channel Cap + Deep Rerun

**Summary:** Implement the `max_hidden_channels` cap for Hybrid/StableHybrid so we can rerun Stage B at fno_blocks=8 without OOM.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 4 (Task 4.5 Channel Cap + Rerun)

**Branch:** fno2

**Mapped tests:**
- `pytest tests/torch/test_fno_generators.py::TestHybridUNOGenerator::test_max_hidden_channel_cap -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_capped_channels -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_max_hidden_channels -v`
- `pytest tests/torch/test_fno_generators.py -k stable -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T210000Z/`

**Next Up (optional):** If capped Stage B succeeds, kick off LayerScale spike for stable_hybrid.

---

## Do Now — Task 4.5 (Channel Cap + Deep Rerun)

1. **Plumb `max_hidden_channels` (config + CLI)**
   - Add `max_hidden_channels: Optional[int] = None` to canonical `ModelConfig` (`ptycho/config/config.py`) and to PyTorch singletons in `ptycho_torch/config_params.py`.
   - Update `ptycho_torch/config_bridge.py` + `config_factory.py` so the field round-trips between TF/Torch configs and shows up in the overrides audit trail.
   - Extend `grid_lines_compare_wrapper.py` with `--torch-max-hidden-channels` (default None) and thread it through `run_grid_lines_compare()`.
   - Add `max_hidden_channels` to `TorchRunnerConfig`, forward it inside `setup_torch_configs()`, and ensure `_train_with_lightning` receives it via the PyTorch model config.

2. **Hybrid encoder cap implementation**
   - Update `HybridUNOGenerator` (and `StableHybridUNOGenerator` via inheritance) to accept `max_hidden_channels`. Store a channel schedule so encoder/decoder agree on concat widths.
   - While building encoder/downsample blocks, compute `next_ch = ch * 2` and clamp via `max_hidden_channels` if provided.
   - Decoder/up-sample path must mirror the clamped schedule to keep skip concatenations aligned.
   - Add a test (`tests/torch/test_fno_generators.py`) that instantiates `HybridUNOGenerator(n_blocks=8, hidden_channels=32, max_hidden_channels=512)` and asserts all layers stay ≤512 channels while preserving output shape.

3. **Runner + CLI regression tests**
   - Extend `tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment` (or add a new test) verifying that `TorchRunnerConfig(max_hidden_channels=512)` leads to `training_config.model.max_hidden_channels == 512` when passed through `setup_torch_configs()`.
   - Add a CLI test (`tests/test_grid_lines_compare_wrapper.py`) confirming `--torch-max-hidden-channels 512` propagates to the mocked runner call.

4. **Stage B rerun (capped)**
   - Create artifacts hub `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T210000Z/` if missing and reuse Stage A dataset copy from Task 4.1.
   - Run the deep control arm with channel cap:
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
   - Copy `history.json`, `metrics.json`, `model.pt`, and new stats (`stage_b_deep_control_max512_stats.json`) into the artifacts hub. Regenerate `stage_b_metrics.json` (now includes capped data).
   - Update `stage_b_summary.md` comparing Stage A baseline vs Stage B capped run (val_loss, SSIMs, MAE, grad norms). Call out whether depth-induced instability appeared once memory was resolved.

5. **Regression guard + docs sync**
   - Re-run mapped selectors listed above, archive logs under the 210000Z hub per `docs/TESTING_GUIDE.md`.
   - Update `docs/strategy/mainstrategy.md` (Stage B remediation paragraph), `docs/fix_plan.md` attempts history / supervisor state, and append any durable lessons (e.g., `FNO-DEPTH-002` if new behavior emerges) to `docs/findings.md`.

6. **Artifacts hygiene**
   - Ensure the 210000Z hub contains: README (with capped hyperparams), stage_b CAP log, stats/metrics JSON, copied run artifacts, pytest logs, and the refreshed summary.md turn log.
