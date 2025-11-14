## Phase A Do Now — Patch-Stat Instrumentation + Short Torch Baseline

### Scope
- Files: `ptycho_torch/model.py`, `ptycho_torch/inference.py`, `ptycho_torch/train.py`, `ptycho_torch/inference.py`, `ptycho_torch/config_factory.py`, potentially `ptycho_torch/cli/shared.py` for config plumbing, and new pytest under `tests/torch/test_cli_train_torch.py`.
- Hub: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/`

### Instrumentation Requirements
1. Add two new CLI flags to both train and inference modules:
   - `--log-patch-stats` (bool) — enable per-patch metrics/dumps for the first N batches.
   - `--patch-stats-limit` (int, default=1) — number of batches to record.
2. Propagate the flags through `ptycho_torch.cli.shared.build_execution_config_from_args` and plumb to the Lightning module/config payload so they are available inside `PtychoPINN.forward` and inference reassembly helpers.
3. When enabled:
   - Capture per-patch mean/variance before and after normalization.
   - Record zero-mean variance (`var_zero_mean`) and running min/max for each patch.
   - Save JSON summary to `$HUB/analysis/torch_patch_stats.json`.
   - Save a normalized patch grid PNG to `$HUB/analysis/torch_patch_grid.png` (use torchvision make_grid or matplotlib).
   - Emit structured INFO logs tagged with `PATCH_STATS` so tests can assert on them easily.
4. Guard instrumentation via the new flags so normal runs are unaffected (no additional CUDA allocations when disabled).

### Evidence to Capture
1. **PyTest selector**  
   `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv | tee "$HUB"/green/pytest_patch_stats.log`
   - Test should invoke `cli_main` with the new flags (using tmp_path NPZ stubs) and assert JSON/PNG files are produced.
2. **Short Training Baseline (10 epochs, 256 groups)**  
   ```bash
   HUB=plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity
   OUT=outputs/torch_forward_parity_baseline
   python -m ptycho_torch.train \
     --train_data_file datasets/fly64_coord_variants/fly001_64_train_converted_identity.npz \
     --test_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
     --output_dir \"$OUT\" \
     --max_epochs 10 \
     --n_images 256 \
     --gridsize 2 \
     --batch_size 4 \
     --torch-loss-mode poisson \
     --accelerator gpu --deterministic \
     --log-patch-stats --patch-stats-limit 2 \
     --quiet \
     |& tee \"$HUB\"/cli/train_patch_stats.log
   ```
3. **Inference Debug Dump**  
   ```bash
   python -m ptycho_torch.inference \
     --model_path \"$OUT\" \
     --test_data datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
     --output_dir \"$OUT\"/inference \
     --n_images 128 \
     --accelerator gpu \
     --debug-dump \"$HUB\"/analysis/forward_parity_debug \
     --log-patch-stats --patch-stats-limit 2 \
     |& tee \"$HUB\"/cli/inference_patch_stats.log
   ```
   - Ensure the debug dump directory contains both the Lightning dump (existing) and the new JSON/PNG artifacts.

4. Update `$HUB/analysis/artifact_inventory.txt` summarizing the instrumentation outputs, pytest log, and CLI commands.

### Acceptance
- New CLI flags documented via `--help` and covered by pytest.
- `$HUB/analysis/torch_patch_stats.json` shows non-zero variance/mean data for at least the first 2 batches.
- `$HUB/analysis/torch_patch_grid.png` renders normalized patch images (no all-zero grid).
- Logs show instrumentation only runs for the first `--patch-stats-limit` batches.
- No regressions in existing PyTorch CLI tests.
