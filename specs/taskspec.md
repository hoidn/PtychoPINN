# Specification Template
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- **Adapt the codebase to support per-sample probe tensors instead of a global probe variable.**

## Mid-Level Objective

- **Update Data Containers:**
  - Introduce a new data container class (e.g., `MultiPtychoDataContainer`) to manage multiple probe tensors.
  - Add a new attribute `probe_indices` to store the probe index for each sample.

- **Modify Data Loading and Preprocessing:**
  - Adapt data loading functions to handle multiple probes and assign appropriate probe indices.
  - Ensure that `probe_indices` are of dtype `int64` and align with the first dimension of data tensors like `X`, `Y_I`, etc.

- **Adjust Model Architecture:**
  - Update the model to accept per-sample probe tensors based on `probe_indices`.
  - Modify relevant layers to utilize the dynamic probe tensors instead of a single global probe.

- **Enhance Training Pipeline:**
  - Implement shuffling and interleaving of samples from multiple datasets during training.
  - Ensure that each training sample references the correct probe tensor via `probe_indices`.

- **Ensure Consistency in Testing:**
  - Maintain the integrated handling of test samples without shuffling.
  - Ensure that test samples correctly utilize their associated probe tensors without requiring dataset boundary reconstruction.

## Implementation Notes

- **Dependencies and Requirements:**
  - Ensure compatibility with existing modules: `./ptycho/raw_data.py`, `./ptycho/workflows/components.py`, `./ptycho/train_pinn.py`, `./ptycho/model.py`, `./ptycho/tf_helper.py`.
  - Utilize TensorFlow (`tf.Tensor`) for managing probe tensors.
  - Maintain consistency in probe tensor shapes and datatypes across the codebase.

- **Coding Standards to Follow:**
  - Adhere to existing coding conventions and documentation styles within the codebase.
  - Ensure thorough testing of data loading, preprocessing, and model training with the new multi-probe setup.

- **Other Technical Guidance:**
  - Validate that `probe_indices` correctly reference probes within the `probe_list`.
  - Implement error handling to manage potential mismatches between `probe_indices` and the `probe_list`.
  - Optimize data handling to efficiently manage multiple probes without significant performance degradation.

## Context

### Beginning Context

- **Existing Files:**
  - `./ptycho/loader.py`
  - `./ptycho/raw_data.py`
  - `./ptycho/workflows/components.py`
  - `./ptycho/train_pinn.py`
  - `./ptycho/model.py`
  - `./ptycho/tf_helper.py`

### Ending Context

- **Modified Files:**
  - `./ptycho/loader.py` (updated)
  - `./ptycho/raw_data.py` (updated)
  - `./ptycho/workflows/components.py` (updated)
  - `./ptycho/train_pinn.py` (updated)
  - `./ptycho/model.py` (updated)
  - `./ptycho/tf_helper.py` (updated)

## Low-Level Tasks
> Ordered from start to finish

1. **Update `loader.py` to Handle Multiple Probes**