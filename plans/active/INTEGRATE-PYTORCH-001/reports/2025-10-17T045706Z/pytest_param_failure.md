============================= test session starts ==============================
platform linux -- Python 3.11.13, pytest-8.4.1, pluggy-1.6.0 -- /home/ollie/miniconda3/envs/ptycho311/bin/python3.11
cachedir: .pytest_cache
PyTorch: not available (/home/ollie/miniconda3/envs/ptycho311/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommWindowRegister)

rootdir: /home/ollie/Documents/PtychoPINN
configfile: pyproject.toml
plugins: anyio-4.9.0
collecting ... collected 1 item

tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields FAILED [100%]

=================================== FAILURES ===================================
____________ TestConfigBridgeParity.test_model_config_direct_fields ____________

self = <unittest.case._Outcome object at 0x74ffd06badd0>
test_case = <test_config_bridge.TestConfigBridgeParity testMethod=test_model_config_direct_fields>
subTest = False

    @contextlib.contextmanager
    def testPartExecutor(self, test_case, subTest=False):
        old_success = self.success
        self.success = True
        try:
>           yield

../../miniconda3/envs/ptycho311/lib/python3.11/unittest/case.py:57: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../../miniconda3/envs/ptycho311/lib/python3.11/unittest/case.py:623: in run
    self._callTestMethod(testMethod)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <test_config_bridge.TestConfigBridgeParity testMethod=test_model_config_direct_fields>
method = <bound method TestConfigBridgeParity.test_model_config_direct_fields of <test_config_bridge.TestConfigBridgeParity testMethod=test_model_config_direct_fields>>

    def _callTestMethod(self, method):
>       if method() is not None:
           ^^^^^^^^
E       TypeError: TestConfigBridgeParity.test_model_config_direct_fields() missing 3 required positional arguments: 'field_name', 'pytorch_value', and 'expected_tf_value'

../../miniconda3/envs/ptycho311/lib/python3.11/unittest/case.py:579: TypeError
----------------------------- Captured stderr call -----------------------------
2025-10-16 21:57:29.408373: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1760677049.419178 3297699 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1760677049.422441 3297699 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1760677049.432024 3297699 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1760677049.432036 3297699 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1760677049.432039 3297699 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1760677049.432041 3297699 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-10-16 21:57:29.434754: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
=========================== short test summary info ============================
FAILED tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields - TypeError: TestConfigBridgeParity.test_model_config_direct_fields() missing 3 required positional arguments: 'field_name', 'pytorch_value', and 'expected_tf_value'
============================== 1 failed in 2.72s ===============================
