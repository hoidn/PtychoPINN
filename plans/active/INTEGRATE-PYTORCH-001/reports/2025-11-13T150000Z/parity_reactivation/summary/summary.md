### Turn Summary
Implemented PyTorch inference execution branch in scripts/inference/inference.py so backend='pytorch' now calls ptycho_torch.inference._run_inference_and_reconstruct instead of TensorFlow's perform_inference().
Added test_pytorch_inference_execution_path to verify the new code path, reran all backend selector tests (3 PASSED), and executed full PyTorch CLI smoke (training + inference) with the minimal fixture.
Inference completed successfully and generated amplitude/phase PNGs (15K/19K) under inference_outputs/, confirming the PyTorch workflow is end-to-end functional from the canonical entry points.
Artifacts: green/pytest_backend_selector_cli.log, cli/pytorch_cli_smoke/{train.log,inference.log,inference_outputs/{reconstructed_amplitude.png,reconstructed_phase.png}}
