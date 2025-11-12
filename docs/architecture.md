# PtychoPINN Core Architecture

This document provides a high-level overview of the `ptycho/` core library architecture, its main components, and how they interact. It is intended to be a "map" of the system.

For detailed development practices, anti-patterns, and the project's design philosophy, please see the **<doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref>**.

## 1. Component Diagram

This diagram shows the primary modules in the `ptycho/` library and their relationships. The flow generally moves from configuration and data loading on the left to model execution and evaluation on the right.

**Note:** The component highlighted in red (`params.py`) is a legacy part of the system that is being actively phased out.

**Backend Selection:** PtychoPINN supports dual backends via a `backend` configuration field (`'tensorflow'` or `'pytorch'`). The dispatcher routes to `ptycho.workflows.components` (TensorFlow) or, when available, to the PyTorch orchestration described in `docs/workflows/pytorch.md`. Both paths share the same data pipeline (`raw_data.py`, `loader.py`) and configuration system (`config/config.py` with a legacy `params.py` bridge). For API surface and contracts, see `docs/specs/spec-ptycho-interfaces.md`.

```mermaid
graph TD
    subgraph "Configuration"
        A[config/config.py] -- "Updates" --> B[params.py (Legacy)]
    end

    subgraph "Data Pipeline"
        C[NPZ Files] --> D[raw_data.py]
        D -- "RawData" --> E[loader.py]
        E -- "PtychoDataContainer" --> F[Model-Ready Data]
    end

    subgraph "Core Model & Physics"
        G[diffsim.py] -- "Physics Model" --> H[model.py]
        I[tf_helper.py] -- "TF Ops" --> H
        B -- "Global State (DEPRECATED)" --> H
    end
    
    subgraph "Workflows & Evaluation"
        J[workflows/components.py] -- "Orchestrates" --> H
        J -- "Uses" --> E
        K[evaluation.py]
    end

    F -- "Input" --> J
    H -- "Reconstruction" --> K

    style A fill:#cde4ff
    style B fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px
```

Note on coordinates: The data pipeline standardizes scan positions in channel format `(B, 1, 2, C)` with axis order `[x, y]`; channel index `c` maps to `(row, col)` via row‑major (`row=c//gridsize`, `col=c%gridsize`). See `docs/specs/spec-ptycho-interfaces.md` for the full contract.

## 2. Typical Workflow Sequence (Training Run)

This diagram illustrates the sequence of function calls and data object transformations during a standard training run initiated by a script like `scripts/training/train.py`.

```mermaid
sequenceDiagram
    participant Script as scripts/training/train.py
    participant W as workflows/components.py
    participant L as loader.py
    participant M as model.py
    participant E as evaluation.py

    Script->>W: run_cdi_example(train_data, test_data, config)
    W->>L: create_ptycho_data_container(train_data)
    L-->>W: returns train_container (PtychoDataContainer)
    
    W->>M: train(train_container)
    Note over M: Model training loop executes...
    M-->>W: returns training_history
    
    alt If test_data is provided
        W->>L: create_ptycho_data_container(test_data)
        L-->>W: returns test_container
        W->>M: model.predict(test_container)
        M-->>W: returns reconstructed_obj
        W->>E: eval_reconstruction(reconstructed_obj, ground_truth)
        E-->>W: returns metrics_dict
    end

    W-->>Script: returns results_dict
```

## 3. A Deep Dive: The Data Loading & Preprocessing Pipeline

The journey from a raw `.npz` file on disk to a model-ready batch of tensors is a multi-stage process designed for robustness, physical correctness, and performance. This pipeline is primarily handled by `ptycho/raw_data.py` and `ptycho/loader.py`.

### The Data Transformation Flow

```mermaid
graph TD
    A[1. NPZ File on Disk] --> B(2. Ingestion: RawData Object);
    B --> C{gridsize > 1?};
    C -- No --> D[3a. Legacy Sequential<br>Subsampling];
    C -- Yes --> E[3b. Group-Aware Subsampling<br>(Sample-then-Group)];
    D --> F[4. Grouped Data Dictionary<br>(NumPy Arrays)];
    E --> F;
    F --> G(5. Transformation: loader.py);
    G --> H[6. Model-Ready<br>PtychoDataContainer<br>(TensorFlow Tensors)];

    subgraph "ptycho/raw_data.py"
        B
        C
        D
        E
    end

    subgraph "ptycho/loader.py"
        G
        H
    end
```

### Stage 1: Ingestion (`raw_data.py`)
The process begins by creating a `RawData` object, which is a direct in-memory representation of the `.npz` file.

### Stage 2: Neighbor-Aware Grouping (`RawData.generate_grouped_data()`)
This is the most critical step, especially for **overlap-based training (`gridsize > 1`)**. To ensure training samples are both spatially representative and physically coherent, this function implements a performant **"sample‑then‑group"** strategy:
1.  **Sample Seed Points:** Select seed indices (random or sequential) from the dataset.
2.  **Find Neighbors for Seeds:** Build a KD‑tree once and query K nearest neighbors for each seed.
3.  **Form Groups:** For each seed, select `C=gridsize²` indices from its neighbor set to form a solution region.

This approach avoids generating or caching all possible groups and significantly reduces memory and compute overhead. For `gridsize = 1`, seeds are used directly without neighbor search.

### Stage 3: Transformation to Tensors (`loader.py`)
This final stage prepares the data for TensorFlow by converting the grouped NumPy arrays into a `PtychoDataContainer`, which holds the final, model-ready `tf.Tensor` objects (`X`, `Y`, `coords_nominal`, etc.) that are passed directly to the model.

## 4. Component Reference

-   **`config/config.py`**: The **modern, authoritative configuration system**.
    -   **Key Components:** `ModelConfig`, `TrainingConfig`, `InferenceConfig` (dataclasses).
    -   See the **<doc-ref type="guide">docs/CONFIGURATION.md</doc-ref>**.

-   **`params.py`**: The **legacy global state**. A global dictionary that is **DEPRECATED** but maintained for backward compatibility.

-   **`raw_data.py` & `loader.py`**: The **data ingestion and transformation layers**. They convert raw `.npz` files into model-ready `PtychoDataContainer` objects.
    -   **Key Functions:** `raw_data.generate_grouped_data()`, `loader.create_ptycho_data_container()`.
    -   For a visual breakdown of grouping and grids, see **<doc-ref type="technical">docs/GRIDSIZE_N_GROUPS_GUIDE.md</doc-ref>**.

-   **`diffsim.py`**: The **forward physics model**. Encapsulates the scientific domain knowledge of ptychography.
    -   **Key Functions:** `illuminate_and_diffract()`, `mk_simdata()`.

-   **`model.py`**: The **core deep learning model**. Defines the U-Net architecture and custom Keras layers that embed the physics constraints.
    -   **Key Functions:** `create_model_with_gridsize()`, `train()`.

-   **`tf_helper.py`**: A **low-level TensorFlow utility module**. Contains reusable tensor operations for patching, reassembly, and transformations.
    -   **Key Functions:** `reassemble_position()`, `extract_patches_position()`.

-   **`image/` (Package)**: The **modern, authoritative image processing toolkit**. Contains modules for:
    -   `registration.py`: Sub-pixel image alignment (`register_and_align`).
    -   `cropping.py`: Physically correct alignment for evaluation (`align_for_evaluation`).
    -   `stitching.py`: Legacy grid-based patch reassembly.

-   **`evaluation.py`**: The **metrics and quality control module**. Contains all logic for calculating performance metrics (PSNR, SSIM, FRC).
    -   **Key Function:** `eval_reconstruction()`.
    -   Usage examples are covered in **<doc-ref type="guide">docs/WORKFLOW_GUIDE.md</doc-ref>** and **<doc-ref type="workflow-guide">scripts/studies/README.md</doc-ref>**.

-   **`workflows/components.py`**: The **high-level orchestration layer**. Chains together calls to the core library modules to execute end-to-end tasks.
    -   **Key Functions:** `run_cdi_example()`, `setup_configuration()`.
    -   For usage examples, see **<doc-ref type="workflow-guide">scripts/training/README.md</doc-ref>**.

-   **`model_manager.py`**: Model bundle persistence and reload. Handles multi-model archives (`wts.h5.zip`) and restores `params.cfg` on load.

-   **`image/reassemble_patches`**: Utilities for assembling predicted patches into full-field reconstructions (used by workflows and inference scripts).

-   **`io/` (Package)**: Input/output helpers for consistent file handling across training, inference, and studies.

## 5. Key Design Principles

-   **Explicit over Implicit**: New code should favor passing configuration and data as explicit arguments rather than relying on global state. This principle is explained in detail in the **<doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref>**.

-   **Data Contracts**: All data exchange between components must adhere to the formats defined in **<doc-ref type="contract">docs/specs/spec-ptycho-interfaces.md</doc-ref>**.

-   **Separation of Concerns**: Physics simulation (`diffsim`), model architecture (`model`), and data handling (`loader`) are kept in separate, specialized modules.

## 6. Scripts Overview

This project includes a set of user-facing scripts that build on the workflow orchestration layer:

- `scripts/training/train.py`: CLI entry for end-to-end training. Uses `ptycho.workflows.components.run_cdi_example()`.
- `scripts/inference/inference.py`: Inference + optional stitching using a previously trained model. Uses `load_inference_bundle()` and reassembly helpers.
- `scripts/reconstruction/run_tike_reconstruction.py`: Integrates with Tike reconstructions for comparison and reassembly.
- `scripts/studies/README.md`: Study and comparison workflows driving multi-model evaluations and aggregations.

See also: **<doc-ref type="workflow-guide">docs/WORKFLOW_GUIDE.md</doc-ref>** and per-folder READMEs under `scripts/`.

## 7. Typical Workflow Sequence (Inference-Only)

The sequence below illustrates inference using a trained model bundle (no further training):

```mermaid
sequenceDiagram
    participant Script as scripts/inference/inference.py
    participant W as workflows/components.py
    participant MM as model_manager.py
    participant Img as image/reassemble_patches
    participant E as evaluation.py

    Script->>W: load_inference_bundle(model_dir)
    W->>MM: ModelManager.load_multiple_models(wts.h5.zip)
    MM-->>W: returns {'diffraction_to_obj': model}, config

    Script->>W: predict(test_container)
    W-->>Script: reconstructed_patches

    Script->>Img: reassemble_patches(reconstructed_patches, coords)
    Img-->>Script: amplitude, phase

    Script->>E: eval_reconstruction(amplitude, phase, ground_truth)
    E-->>Script: metrics_dict
```

## 8. Backend Architecture (PyTorch)

PtychoPINN provides a PyTorch stack with parity to the TensorFlow workflows:

- Orchestration: `ptycho_torch/workflows/components.py` mirrors the TensorFlow API.
- Guide: **<doc-ref type="guide">docs/workflows/pytorch.md</doc-ref>** covers configuration bridging, training via Lightning, checkpointing, and inference.
- Data contract and configuration: shared with TensorFlow (`config/config.py`, `RawData`, and grouping pipeline).

```mermaid
graph TD
    subgraph "Shared Config & Data"
        A[config/config.py] --> B[params.cfg (Legacy Bridge)]
        C[NPZ Files] --> D[ptycho.raw_data.RawData]
    end

    subgraph "PyTorch Orchestration"
        E[ptycho_torch/config_bridge.py]
        F[ptycho_torch/workflows/components.py]
        G[Lightning Trainer]
    end

    subgraph "PyTorch Core"
        H[ptycho_torch/dataloader.py]
        I[ptycho_torch/model.py]
        J[ptycho_torch/model_manager.py]
    end

    D --> H
    E --> F
    B --> F
    H --> G
    I --> G
    G --> J
```

## 9. Stable Modules and Config Lifecycle

- Stable/Do‑Not‑Modify (without an approved plan): `ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`.
- Mandatory config bridge before legacy modules (PyTorch policy also applies):

```python
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params

# ... create ModelConfig + TrainingConfig instances as needed ...
config = TrainingConfig(model=ModelConfig(...))

# Bridge to legacy dict (required before data loading / legacy usage)
update_legacy_dict(params.cfg, config)
```

See **<doc-ref type="debug">docs/debugging/QUICK_REFERENCE_PARAMS.md</doc-ref>** for critical notes on `params.cfg` initialization and shape-mismatch troubleshooting.
