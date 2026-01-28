# Strategy: FNO/Hybrid Stability & Optimization Overhaul

**Objective:** Stabilize deep FNO/Hybrid model training (currently prone to gradient explosion at >30 epochs) by eliminating the mathematical root cause of signal drift (GELU mean-shift), rather than suppressing it with aggressive global clipping.

**Core Hypothesis:** The instability is accumulative. The current `y = x + GELU(...)` topology adds a positive bias at every block. As depth (`fno_blocks`) increases, this drift accelerates, pushing latent statistics into saturation.

We will validate two solution classes: an **Architectural Fix** (preventing drift) and an **Optimization Fix** (managing drift), followed by a **Hyperparameter Stress Test** to ensure the solution holds for deeper networks.

---

## 1. Solution Specifications

### A. The Architectural Fix: `stable_hybrid`
A new generator variant designed to enforce zero-mean residual updates mathematically.

*   **Topology:** Implements a **"Norm-Last"** block structure:
    $$ y = x + \text{InstanceNorm}(\text{GELU}(\text{Branch}(x))) $$
*   **Initialization:** Enforces **Zero-Gamma** initialization on the final normalization layer (`weight=0, bias=0`).
*   **Behavioral Contract:**
    *   **Identity Init:** At step 0, the block acts as an Identity function, preserving gradient flow perfectly.
    *   **Zero-Mean Update:** The term added to the residual path always has $\mu=0$, preventing the "mean shift" explosion regardless of network depth.
*   **Implementation:** Registered as `architecture='stable_hybrid'`.

### B. The Optimization Fix: Adaptive Gradient Clipping (AGC)
An advanced clipping algorithm integrated into the training loop as a fallback stability mechanism.

*   **Algorithm:** Clips gradients layer-wise based on the unit-wise ratio of gradient norms to parameter norms ($\frac{\|G\|}{\|W\|}$), rather than a single global scalar.
*   **Benefit:** Allows the stable CNN decoder layers to learn even if the FNO encoder layers have high gradient variance, avoiding the "stagnation" seen with global clipping.
*   **Configuration:** New `TrainingConfig.gradient_clip_algorithm` field supporting `['norm', 'value', 'agc']`.

---

## 2. Experimental Validation

We will select the winner through a rigorous 2-stage process using the `grid_lines` workflow.

### Stage A: The "Shootout" (Standard Depth)
**Context:** `fno_blocks=4` (The known failure case). 50 Epochs.

| Arm | Architecture | Clipping | Hypothesis | Success Condition |
| :--- | :--- | :--- | :--- | :--- |
| **1. Control** | `hybrid` | Global Norm (1.0) | **Stagnation:** Training survives but loss plateaus early. | N/A (Baseline) |
| **2. Arch Fix** | `stable_hybrid` | **None (0.0)** | **Stability:** Converges smoothly with NO clipping. | Lowest final loss; No NaNs. |
| **3. Opt Fix** | `hybrid` | **AGC (0.01)** | **Survival:** Survives 50 epochs despite drift. | Lower loss than Control. |

### Stage B: The "Deep" Stress Test (Robustness)
**Context:** `fno_blocks=8` (2x Depth).
**Why:** Instability scales with depth. A solution is only valid if it allows us to scale up the model.

*   **Protocol:** Run the **Winner of Stage A** with `fno_blocks=8`.
*   **Success Condition:** The deep model trains for 50 epochs without NaNs and achieves lower loss than the shallow model (demonstrating effective use of increased capacity).

---

## 3. Implementation Plan

### Phase 1: Foundation (Config & Utilities)
Enable the necessary configuration switches and mathematical utilities.

1.  **Configuration Schema:** Update `TrainingConfig` (TF & Torch) and `config_bridge` to support `gradient_clip_algorithm`.
2.  **AGC Utility:** Implement `adaptive_gradient_clip_` in `ptycho_torch/train_utils.py`.
3.  **Training Loop:** Update `PtychoPINN_Lightning.training_step` to dispatch the requested clipping algorithm during manual optimization.
4.  **Runners:** Update `grid_lines_torch_runner` and `compare_wrapper` to expose these flags.

### Phase 2: Generator Engineering
Construct the stable architecture.

1.  **StablePtychoBlock:** Implement the Norm-Last/Zero-Gamma logic in `ptycho_torch/generators/fno.py`.
2.  **StableHybridGenerator:** Subclass `HybridUNOGenerator` to use the stable block.
3.  **Registry:** Expose as `stable_hybrid`.

---

## 4. Context & Tooling

This section maps the strategic goals to the specific components in the codebase required to execute them.

### **The "Grid Lines" Harness**
The project uses a canonical dataset/workflow for architectural comparisons called `grid_lines`.
*   **Orchestrator:** `scripts/studies/grid_lines_compare_wrapper.py`
    *   *Role:* The main CLI for A/B testing. It runs multiple architectures side-by-side, merges metrics, and generates comparison plots.
    *   *Usage:* `python scripts/studies/grid_lines_compare_wrapper.py --architectures hybrid,stable_hybrid ...`
*   **Torch Runner:** `scripts/studies/grid_lines_torch_runner.py`
    *   *Role:* Handles the actual training/inference loop for PyTorch models. It needs to be updated to accept the new `--gradient-clip-algorithm` flag.

### **Generator Architecture**
*   **Definition:** `ptycho_torch/generators/fno.py`
    *   *Current State:* Contains `PtychoBlock` (unstable) and `HybridUNOGenerator`.
    *   *Target State:* Will contain `StablePtychoBlock` and `StableHybridGenerator`.
*   **Registry:** `ptycho_torch/generators/registry.py`
    *   *Role:* Maps string names (e.g., `'stable_hybrid'`) to classes.

### **Configuration System**
The codebase uses a mirrored configuration system (TensorFlow $\leftrightarrow$ PyTorch) connected by a bridge.
*   **Canonical Config (TF):** `ptycho/config/config.py` (`TrainingConfig`).
*   **PyTorch Config:** `ptycho_torch/config_params.py` (`TrainingConfig`).
*   **The Bridge:** `ptycho_torch/config_bridge.py`
    *   *Requirement:* Any new parameter like `gradient_clip_algorithm` must be added to **all three** files to propagate correctly from the CLI to the model.

### **Model & Training**
*   **Lightning Module:** `ptycho_torch/model.py` (`PtychoPINN_Lightning`)
    *   *Role:* Defines the `training_step`. This is where the manual optimization loop resides and where the logic to switch between `clip_grad_norm_` and `adaptive_gradient_clip_` must be injected.
*   **Utilities:** `ptycho_torch/train_utils.py`
    *   *Role:* The home for shared math functions. The AGC logic (`unitwise_norm`) goes here.

### **Diagnostics**
*   **Activation Monitor:** `scripts/debug_fno_activations.py`
    *   *Role:* A pre-existing script that hooks into the model to record mean/std of activations. Use this *before* the full training run to verify that `StablePtychoBlock` actually maintains zero-mean outputs at initialization.

---

## 5. Decision Logic & Outcome

*   **Gold Standard:** If `stable_hybrid` passes Stage A (beats Control) AND Stage B (scales to depth), it becomes the **new default FNO architecture**. The legacy `PtychoBlock` will be deprecated.
*   **Silver Standard:** If `stable_hybrid` fails but `AGC` works, AGC becomes the **mandatory training policy** for FNO models.
*   **Pivot:** If both fail Stage B (Deep Test), the instability is not just mean-shift. We will pivot to investigating **LayerScale** (explicit depth damping).
