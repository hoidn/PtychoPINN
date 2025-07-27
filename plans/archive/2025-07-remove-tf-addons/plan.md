# R&D Plan: Remove TensorFlow Addons Dependency

*Created: 2025-07-27*

## 🎯 **OBJECTIVE & HYPOTHESIS**

### **Objective**
Replace the usage of `tensorflow-addons` (specifically `tfa.image.translate`) in the PtychoPINN codebase with a custom implementation using core TensorFlow operations, then remove the TensorFlow Addons dependency from the project.

### **Hypothesis**
We can implement the image translation functionality using TensorFlow's native `tf.raw_ops.ImageProjectiveTransformV3` without loss of functionality or numerical accuracy, eliminating the maintenance-mode dependency on TensorFlow Addons.

### **Background & Motivation**
- TensorFlow Addons is in maintenance mode and will not receive updates for newer TensorFlow versions
- The project only uses one function from the entire library: `tfa.image.translate`
- Removing this dependency will improve long-term maintainability and compatibility

---

## 📝 **CURRENT STATE ANALYSIS**

### **Current Usage**
- **Single Import Location**: `ptycho/tf_helper.py` (line 515)
- **Function Used**: `tensorflow_addons.image.translate`
- **Where Used**: The `translate()` function and `Translation` class in tf_helper.py
- **Purpose**: Translating image patches with sub-pixel accuracy for ptychographic reconstruction

### **Technical Requirements**
- Must support batched translation of complex-valued tensors
- Must maintain sub-pixel translation accuracy
- Must support bilinear interpolation
- Must be numerically equivalent to the current implementation

---

## 🔬 **TECHNICAL APPROACH**

### **Core Strategy**
Replace `tfa.image.translate` with a custom implementation using `tf.raw_ops.ImageProjectiveTransformV3`:

```python
def translate_core(images, translations, interpolation='bilinear'):
    """Core translation using native TensorFlow ops."""
    # Implementation using tf.raw_ops.ImageProjectiveTransformV3
    # with proper transformation matrix construction
```

### **Key Technical Considerations**
1. **Coordinate System**: TensorFlow uses different coordinate conventions than TFA
2. **Batch Processing**: Must handle per-image translations in batched tensors
3. **Complex Number Support**: Leverage existing `@complexify_function` decorator
4. **Numerical Precision**: Ensure results match TFA within acceptable tolerance

---

## ⚠️ **RISKS & MITIGATION**

### **Risk 1: Numerical Differences**
- **Severity**: High
- **Mitigation**: Create comprehensive comparison tests before replacing
- **Validation**: Direct numerical comparison with tolerance of 1e-6

### **Risk 2: Performance Regression**
- **Severity**: Medium
- **Mitigation**: Benchmark both implementations
- **Validation**: Time comparison on typical workloads

### **Risk 3: Edge Case Handling**
- **Severity**: Medium
- **Mitigation**: Test with zero shifts, integer shifts, and sub-pixel shifts
- **Validation**: Comprehensive test suite covering all use cases

---

## ✅ **VALIDATION & VERIFICATION PLAN**

### **Phase 1: Implementation Validation**
1. **Direct Comparison Test**: Create test comparing old and new implementations
2. **Edge Case Testing**: Verify behavior with various translation values
3. **Complex Tensor Support**: Ensure complex number handling works correctly

### **Phase 2: Integration Testing**
1. **Unit Tests**: Run existing tf_helper tests
2. **Full Test Suite**: Execute complete project test suite
3. **End-to-End Verification**: Run training/inference workflow

### **Phase 3: Performance Validation**
1. **Benchmark Translation Speed**: Compare execution times
2. **Memory Usage**: Verify no memory regression
3. **GPU Compatibility**: Test on GPU if available

### **Success Criteria**
- ✅ New implementation produces numerically identical results (within 1e-6 tolerance)
- ✅ All existing tests pass without modification
- ✅ TensorFlow Addons removed from dependencies
- ✅ Full training/inference cycle completes successfully

---

## 📊 **IMPLEMENTATION PHASES**

### **Phase 1: Core Implementation** (1 day)
- Implement `translate_core()` function
- Update existing `translate()` to use new implementation
- Create comparison tests

### **Phase 2: Validation & Testing** (1 day)
- Run numerical comparison tests
- Execute full test suite
- Perform end-to-end validation

### **Phase 3: Cleanup & Documentation** (0.5 day)
- Remove tensorflow-addons from setup.py
- Update any relevant documentation
- Commit changes with descriptive message

---

## 📁 **File Organization**

**Initiative Path:** `plans/active/remove-tf-addons-dependency/`

**Next Step:** Run `/implementation` to generate the phased implementation plan.