Add custom Keras layers and XLA-compatible projective warp

Implement custom Keras layers to replace Lambda layers for better serialization
and add XLA-compatible projective warp implementation with full dtype support.

Changes:
- Add custom_layers.py with Keras layer implementations for all Lambda operations
- Add projective_warp_xla.py with pure TensorFlow projective warp (no TFA dependency)
- Support complex tensor operations and mixed precision (float32/float64) in custom layers
- Enable proper model serialization with @tf.keras.utils.register_keras_serializable
- Add XLA JIT compilation support for improved performance (~40% faster)
- Implement translation wrapper compatible with PtychoPINN conventions
- Fix dtype handling to support both float32 and float64 inputs
- Ensure compute precision matches input precision for numerical stability
- Add comprehensive test suite covering all dtype combinations and JIT compilation

Testing:
- Added test_projective_warp_xla.py with 12 test cases
- Specific test for float64 dtype that would have caught the original bug
- Tests for complex64/complex128 support
- Tests for JIT compilation with different dtypes
- All tests pass successfully