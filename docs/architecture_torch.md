# PtychoPINN Architecture — PyTorch

This page documents the PyTorch implementation of PtychoPINN, focusing on modules under `ptycho_torch/` and their orchestration.

## 1. Component Diagram (PyTorch)

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

## 2. Training Workflow (PyTorch)

```mermaid
sequenceDiagram
    participant Script as scripts/training/train.py
    participant W as Torch workflows/components.py
    participant D as Torch dataloader/DataContainer
    participant L as Lightning Trainer
    participant M as Torch model.py

    Script->>W: run_cdi_example_torch(train_data, test_data, config)
    W->>D: create_container_from_raw(train_data)
    D-->>W: returns train_container (PtychoDataContainerTorch)

    W->>L: build Trainer(config) + fit(model)
    Note over L: Lightning training loop executes...
    L-->>W: returns training_history

    alt If test_data is provided
        W->>D: create_container_from_raw(test_data)
        D-->>W: returns test_container
        L->>M: predict(test_container)
        M-->>L: returns reconstructed_patches
    end

    W-->>Script: returns results_dict
```

## 3. Inference Workflow (PyTorch)

```mermaid
sequenceDiagram
    participant Script as scripts/inference/inference.py
    participant W as ptycho_torch/workflows/components.py
    participant MM as ptycho_torch/model_manager.py
    participant Img as reassembly (parity)
    participant E as evaluation.py

    Script->>W: load_inference_bundle_torch(model_dir)
    W->>MM: load_torch_bundle(wts.h5.zip)
    MM-->>W: returns models_dict, config

    Script->>W: predict(test_container)
    W-->>Script: reconstructed_patches

    Script->>Img: reassemble (TF helper used for MVP parity)
    Img-->>Script: amplitude, phase

    Script->>E: eval_reconstruction(amplitude, phase, ground_truth)
    E-->>Script: metrics_dict
```

See details and current status in **<doc-ref type="guide">docs/workflows/pytorch.md</doc-ref>**.

## 4. Component Reference (PyTorch)

- `ptycho_torch/config_bridge.py`: Translates TF dataclasses to Torch equivalents
- `ptycho_torch/data_container_bridge.py`: `PtychoDataContainerTorch` container factory
- `ptycho_torch/dataloader.py`: Datasets and DataLoaders compatible with Lightning
- `ptycho_torch/model.py`: U‑Net + physics-informed Torch model
- `ptycho_torch/model_manager.py`: Torch model bundle persistence and load
- `ptycho_torch/workflows/components.py`: Orchestration entry points (`run_cdi_example_torch`, etc.)
- Reassembly: Currently reuses TF helper for parity; native Torch reassembly planned
- Shared modules: `ptycho/raw_data.py`, `config/config.py`, `specs/spec-ptycho-interfaces.md`

Config Bridging:
- Normative config mapping and bridge flow: <doc-ref type="spec">specs/spec-ptycho-config-bridge.md</doc-ref>

## 5. Function & Container Mapping (PyTorch ↔ TF)

- Orchestration: `ptycho_torch.workflows.components.run_cdi_example_torch` ↔ `ptycho.workflows.components.run_cdi_example`
- Load model: `load_inference_bundle_torch` ↔ `load_inference_bundle`
- Container: `PtychoDataContainerTorch` ↔ `loader.PtychoDataContainer`
- Data loader: `ptycho_torch.dataloader.PtychoDataset` + Lightning DataLoader ↔ TF `loader.py` pipelines
- Model: `ptycho_torch/model.py` ↔ `ptycho/model.py`
- Reassembly: parity path uses TF helper; native Torch path planned
