# C4.D3 Lightning Training Failure — Poisson Support Violation

## Reproduction
- Generated temporary fixture with `python scripts/tools/make_pytorch_integration_fixture.py --source datasets/Run1084_recon3_postPC_shrunk_3.npz --output tmp/minimal_dataset_v1.npz --subset-size 64`.
- Ran the ADR-003 CLI workflow with the minimal fixture:
  ```bash
  CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train \
    --train_data_file tmp/minimal_dataset_v1.npz \
    --test_data_file tmp/minimal_dataset_v1.npz \
    --output_dir plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/manual_cli_run \
    --max_epochs 2 --n_images 64 --gridsize 1 --batch_size 4 \
    --device cpu --disable_mlflow
  ```
- Captured stdout/stderr in `manual_train_cli.log` (same directory).

## Observed Failure
Lightning aborts during the first training epoch with:
```
ValueError: Expected value argument (Tensor of shape (4, 1, 64, 64)) to be within the support (IntegerGreaterThan(lower_bound=0)) of the distribution Poisson(rate: torch.Size([4, 1, 64, 64])), but found invalid values: tensor([...])
```
- Exception raised by `torch.distributions.Poisson.log_prob` inside `PoissonIntensityLayer` because `batch[0]['images']` contains normalized amplitude floats (0.0–0.07), not integer photon counts.
- `_train_with_lightning()` re-raises as `RuntimeError: Lightning training failed. See logs for details.`; integration test captures only this wrapper message.

## Immediate Implications
- Dataloader refactor fixed TensorDict shape, but batches still feed amplitude floats straight into Poisson loss.
- TensorFlow path squares amplitudes and applies `nphotons` scaling before Poisson log-likelihood; PyTorch equivalent never discretizes inputs, so Poisson support check fails.
- ADR-003 Phase C4.D3 cannot close until raw diffraction is converted to Poisson-compatible counts within the PyTorch workflow (likely inside dataloader or `compute_loss`).

## Next Questions
1. Where should amplitude→count conversion live? (`TensorDict` assembly vs. `compute_loss` parity with TensorFlow).
2. Should we apply `torch.floor`/`torch.round` after scaling by `nphotons` or reuse existing `physics_scaling_constant` pipeline?
3. Do existing TensorFlow tests document expected scaling factors (check `ptycho/model.py` Poisson branch + `tests/test_model_manager.py`)?

## Artifacts
- Command log: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/manual_train_cli.log`
- Temporary fixture: `tmp/minimal_dataset_v1.npz` (paired metadata at `tmp/minimal_dataset_v1.json`)
