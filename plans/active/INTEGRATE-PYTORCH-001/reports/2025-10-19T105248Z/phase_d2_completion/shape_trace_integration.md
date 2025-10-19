============================= test session starts ==============================
platform linux -- Python 3.11.13, pytest-8.4.1, pluggy-1.6.0 -- /home/ollie/miniconda3/envs/ptycho311/bin/python3.11
cachedir: .pytest_cache
PyTorch: available
PyTorch version: 2.8.0+cu128
rootdir: /home/ollie/Documents/PtychoPINN2
configfile: pyproject.toml
plugins: anyio-4.9.0
collecting ... collected 1 item

tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle FAILED [100%]

=================================== FAILURES ===================================
___ TestPyTorchIntegrationWorkflow.test_pytorch_train_save_load_infer_cycle ____

self = <test_integration_workflow_torch.TestPyTorchIntegrationWorkflow testMethod=test_pytorch_train_save_load_infer_cycle>

    def test_pytorch_train_save_load_infer_cycle(self):
        """
        Tests the complete PyTorch train → save → load → infer workflow.
    
        This validates the PyTorch model persistence layer by simulating a real
        user workflow across separate processes, mirroring the TensorFlow integration test.
    
        Phase: E2.B1 (Red Test)
        Expected Behavior (Phase E2.C implementation target):
        1. Training subprocess creates Lightning checkpoint
        2. Checkpoint artifact saved to <output_dir>/checkpoints/ or <output_dir>/wts.pt
        3. Inference subprocess loads checkpoint and generates reconstructions
        4. Output images created in inference output directory
    
        Current Status: FAILING — ptycho_torch training/inference scripts not yet
        integrated with subprocess harness and backend dispatcher.
        """
        # --- 1. Define Paths ---
        data_file = project_root / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"
        training_output_dir = self.output_path / "training_outputs"
        inference_output_dir = self.output_path / "pytorch_output"
    
        # --- 2. Training Step (PyTorch) ---
        print("--- Running PyTorch Training Step (subprocess) ---")
    
        # NOTE: This command reflects the expected CLI interface after Phase E2.C implementation.
        # Currently, ptycho_torch/train.py does not support these exact flags.
        # The test documents the target API contract.
        train_command = [
            sys.executable, "-m", "ptycho_torch.train",
            "--train_data_file", str(data_file),
            "--test_data_file", str(data_file),
            "--output_dir", str(training_output_dir),
            "--max_epochs", "2",
            "--n_images", "64",
            "--gridsize", "1",
            "--batch_size", "4",
            "--device", "cpu",
            "--disable_mlflow",  # Suppress MLflow for CI (flag to be added per TEST-PYTORCH-001)
        ]
    
        # Expected to fail in Phase E2.B (red phase) because:
        # - ptycho_torch.train may not support these CLI flags yet
        # - Backend dispatcher not wired to route PyTorch workflows
        # - CONFIG-001 gate may not be enforced
        train_result = subprocess.run(train_command, capture_output=True, text=True)
    
        self.assertEqual(
            train_result.returncode, 0,
            f"PyTorch training script failed with stdout:\n{train_result.stdout}\nstderr:\n{train_result.stderr}"
        )
    
        # Check for PyTorch checkpoint artifact
        # Expected format: Lightning checkpoint or custom .pt bundle
        # Phase E2.C implementation should define exact artifact name
        checkpoint_candidates = [
            training_output_dir / "checkpoints" / "last.ckpt",  # Lightning default
            training_output_dir / "wts.pt",  # Custom bundle format
            training_output_dir / "model.pt",  # Alternative naming
        ]
    
        checkpoint_found = any(p.exists() for p in checkpoint_candidates)
        self.assertTrue(
            checkpoint_found,
            f"No PyTorch checkpoint found in {training_output_dir}. Searched: {[str(p) for p in checkpoint_candidates]}"
        )
    
        # --- 3. Inference Step (PyTorch) ---
        print("--- Running PyTorch Inference Step (subprocess) ---")
    
        # NOTE: ptycho_torch/inference.py does not exist yet (per TEST-PYTORCH-001 §Open Questions).
        # This test documents the expected CLI interface for Phase E2.C implementation.
        inference_command = [
            sys.executable, "-m", "ptycho_torch.inference",
            "--model_path", str(training_output_dir),
            "--test_data", str(data_file),
            "--output_dir", str(inference_output_dir),
            "--n_images", "32",
            "--device", "cpu",
        ]
    
        # Expected to fail in Phase E2.B because:
        # - ptycho_torch.inference module does not exist yet
        # - Inference helper needs to be authored (TEST-PYTORCH-001 §Next Steps #3)
        infer_result = subprocess.run(inference_command, capture_output=True, text=True)
    
>       self.assertEqual(
            infer_result.returncode, 0,
            f"PyTorch inference script failed with stdout:\n{infer_result.stdout}\nstderr:\n{infer_result.stderr}"
        )
E       AssertionError: 1 != 0 : PyTorch inference script failed with stdout:
E       Loading Lightning checkpoint from: /tmp/tmpja387_97/training_outputs/checkpoints/last.ckpt
E       Test data: /home/ollie/Documents/PtychoPINN2/datasets/Run1084_recon3_postPC_shrunk_3.npz
E       Output directory: /tmp/tmpja387_97/pytorch_output
E       Device: cpu
E       N images: 32
E       Decoder Block 1: No attention module added.
E       Decoder Block 2: No attention module added.
E       Decoder Block 1: No attention module added.
E       Decoder Block 2: No attention module added.
E       Successfully loaded model from checkpoint
E       Model type: PtychoPINN_Lightning
E       Loaded test data with keys: ['diffraction', 'probeGuess', 'objectGuess', 'xcoords', 'ycoords', 'xcoords_start', 'ycoords_start']
E       Running inference on 32 images...
E       
E       stderr:
E       2025-10-19 03:59:19.084121: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
E       WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E       E0000 00:00:1760871559.095127  832016 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E       E0000 00:00:1760871559.098779  832016 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
E       W0000 00:00:1760871559.109806  832016 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
E       W0000 00:00:1760871559.109817  832016 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
E       W0000 00:00:1760871559.109819  832016 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
E       W0000 00:00:1760871559.109820  832016 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
E       2025-10-19 03:59:19.112503: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
E       To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
E       ERROR: Inference failed with exception: The size of tensor a (572) must match the size of tensor b (1080) at non-singleton dimension 3
E       Traceback (most recent call last):
E         File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/inference.py", line 533, in cli_main
E           reconstruction = model.forward_predict(
E                            ^^^^^^^^^^^^^^^^^^^^^^
E         File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/model.py", line 1120, in forward_predict
E           x_combined = self.model.forward_predict(x, positions, probe, input_scale_factor)
E                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E         File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/model.py", line 882, in forward_predict
E           x_amp, x_phase = self.autoencoder(x)
E                            ^^^^^^^^^^^^^^^^^^^
E         File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
E           return self._call_impl(*args, **kwargs)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E         File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
E           return forward_call(*args, **kwargs)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E         File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/model.py", line 514, in forward
E           x_amp = self.decoder_amp(x)
E                   ^^^^^^^^^^^^^^^^^^^
E         File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
E           return self._call_impl(*args, **kwargs)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E         File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
E           return forward_call(*args, **kwargs)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E         File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/model.py", line 479, in forward
E           outputs = self.amp(x)
E                     ^^^^^^^^^^^
E         File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
E           return self._call_impl(*args, **kwargs)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E         File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
E           return forward_call(*args, **kwargs)
E                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E         File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/model.py", line 388, in forward
E           outputs = x1 + x2
E                     ~~~^~~~
E       RuntimeError: The size of tensor a (572) must match the size of tensor b (1080) at non-singleton dimension 3

tests/torch/test_integration_workflow_torch.py:144: AssertionError
----------------------------- Captured stdout call -----------------------------

Created temporary directory for PyTorch test run: /tmp/tmpja387_97
--- Running PyTorch Training Step (subprocess) ---
--- Running PyTorch Inference Step (subprocess) ---
Cleaned up temporary directory: /tmp/tmpja387_97
=========================== short test summary info ============================
FAILED tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle - AssertionError: 1 != 0 : PyTorch inference script failed with stdout:
Loading Lightning checkpoint from: /tmp/tmpja387_97/training_outputs/checkpoints/last.ckpt
Test data: /home/ollie/Documents/PtychoPINN2/datasets/Run1084_recon3_postPC_shrunk_3.npz
Output directory: /tmp/tmpja387_97/pytorch_output
Device: cpu
N images: 32
Decoder Block 1: No attention module added.
Decoder Block 2: No attention module added.
Decoder Block 1: No attention module added.
Decoder Block 2: No attention module added.
Successfully loaded model from checkpoint
Model type: PtychoPINN_Lightning
Loaded test data with keys: ['diffraction', 'probeGuess', 'objectGuess', 'xcoords', 'ycoords', 'xcoords_start', 'ycoords_start']
Running inference on 32 images...

stderr:
2025-10-19 03:59:19.084121: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1760871559.095127  832016 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1760871559.098779  832016 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1760871559.109806  832016 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1760871559.109817  832016 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1760871559.109819  832016 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1760871559.109820  832016 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-10-19 03:59:19.112503: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
ERROR: Inference failed with exception: The size of tensor a (572) must match the size of tensor b (1080) at non-singleton dimension 3
Traceback (most recent call last):
  File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/inference.py", line 533, in cli_main
    reconstruction = model.forward_predict(
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/model.py", line 1120, in forward_predict
    x_combined = self.model.forward_predict(x, positions, probe, input_scale_factor)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/model.py", line 882, in forward_predict
    x_amp, x_phase = self.autoencoder(x)
                     ^^^^^^^^^^^^^^^^^^^
  File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/model.py", line 514, in forward
    x_amp = self.decoder_amp(x)
            ^^^^^^^^^^^^^^^^^^^
  File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/model.py", line 479, in forward
    outputs = self.amp(x)
              ^^^^^^^^^^^
  File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/model.py", line 388, in forward
    outputs = x1 + x2
              ~~~^~~~
RuntimeError: The size of tensor a (572) must match the size of tensor b (1080) at non-singleton dimension 3
============================== 1 failed in 19.69s ==============================
