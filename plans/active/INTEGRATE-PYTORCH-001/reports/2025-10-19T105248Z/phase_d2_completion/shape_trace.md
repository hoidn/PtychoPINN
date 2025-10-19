2025-10-19 04:01:03.226668: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1760871663.237908  832985 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1760871663.241633  832985 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1760871663.252731  832985 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1760871663.252741  832985 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1760871663.252743  832985 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1760871663.252744  832985 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-10-19 04:01:03.255423: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-10-19 04:01:07,654 - ptycho_torch.model - [DECODER_TRACE] Input x shape: torch.Size([8, 64, 32, 540])
2025-10-19 04:01:07,669 - ptycho_torch.model - [DECODER_TRACE] After path1 (conv1 + padding): x1 shape: torch.Size([8, 1, 64, 572])
2025-10-19 04:01:07,669 - ptycho_torch.model - [DECODER_TRACE] Path2 input x2_in shape: torch.Size([8, 8, 32, 540])
2025-10-19 04:01:07,735 - ptycho_torch.model - [DECODER_TRACE] After conv_up_block: x2 shape: torch.Size([8, 64, 64, 1080])
2025-10-19 04:01:07,784 - ptycho_torch.model - [DECODER_TRACE] After conv2 + silu: x2 shape: torch.Size([8, 1, 64, 1080])
2025-10-19 04:01:07,784 - ptycho_torch.model - [DECODER_TRACE] About to add: x1.shape=torch.Size([8, 1, 64, 572]), x2.shape=torch.Size([8, 1, 64, 1080])
ERROR: Inference failed with exception: The size of tensor a (572) must match the size of tensor b (1080) at non-singleton dimension 3
Traceback (most recent call last):
  File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/inference.py", line 538, in cli_main
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
Loading Lightning checkpoint from: /tmp/shape_trace_model/checkpoints/last.ckpt
Test data: datasets/Run1084_recon3_postPC_shrunk_3.npz
Output directory: /tmp/shape_trace_infer
Device: cpu
N images: 8
Decoder Block 1: No attention module added.
Decoder Block 2: No attention module added.
Decoder Block 1: No attention module added.
Decoder Block 2: No attention module added.
Successfully loaded model from checkpoint
Model type: PtychoPINN_Lightning
Loaded test data with keys: ['diffraction', 'probeGuess', 'objectGuess', 'xcoords', 'ycoords', 'xcoords_start', 'ycoords_start']
Running inference on 8 images...
