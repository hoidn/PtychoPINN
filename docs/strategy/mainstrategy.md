# Strategy: FNO/Hybrid Stability & Optimization Overhaul

**Objective:** Stabilize deep FNO/Hybrid model training (currently prone to gradient explosion at >30 epochs) by eliminating the mathematical root cause of signal drift, without relying on aggressive global gradient clipping.

**Core Hypothesis:** The instability is accumulative. The current `y = x + GELU(...)` topology adds a positive bias at every block. As depth (`fno_blocks`) increases, this drift accelerates, pushing latent statistics into saturation.
*   *Evidence (to link):* In run <RUN_ID>, `grad_norm_preclip` exceeded $10^5$ around epoch 30. (Add log path or remove this line if no citation.)
*   *Verification:* `scripts/debug_fno_activations.py` will be used to confirm activation mean growth vs. depth.

We will validate two solution classes: an **Architectural Fix** (preventing drift) and an **Optimization Fix** (managing drift), followed by a **Hyperparameter Stress Test** to ensure the solution holds for deeper networks.

---

## 1. Solution Specifications

### A. The Architectural Fix: `stable_hybrid`
A new generator variant designed to enforce normalized residual updates.

*   **Topology:** Implements a **"Norm-Last"** block structure:
    $$ y = x + \text{InstanceNorm}(\text{GELU}(\text{Branch}(x))) $$
*   **Initialization:** Enforces **Zero-Gamma** initialization on the final normalization layer (`weight=0, bias=0`).
*   **Behavioral Contract:**
    *   **Identity Init:** At step 0, the block acts as an Identity function ($y = x$), ensuring perfect gradient flow.
    *   **Centered Residuals (pre-affine):** The branch output is centered by InstanceNorm before residual addition, reducing GELU's positive-mean drift; affine parameters ($\beta$) may later reintroduce bias.
*   **Implementation:** Registered as `architecture='stable_hybrid'`.

### B. The Optimization Fix: Adaptive Gradient Clipping (AGC)
An advanced clipping algorithm integrated into the training loop as a fallback stability mechanism.

*   **Algorithm:** Clips gradients layer-wise based on the unit-wise ratio of gradient norms to parameter norms ($\frac{\|G\|}{\|W\|}$), rather than a single global scalar. This prevents the "all-or-nothing" stagnation of global clipping.
*   **Configuration:**
    *   **Algorithm:** `gradient_clip_algorithm` field in `PyTorchExecutionConfig` (execution-only). Options: `'norm'` (default), `'value'`, `'agc'`.
    *   **Threshold:** `gradient_clip_val` is re-purposed as the clipping factor $\lambda$ (typically $0.01-0.1$) when `agc` is selected.

---

## 2. Experimental Validation

We will select the winner through a rigorous 2-stage process using the `grid_lines` workflow.

### Stage A: The "Shootout" (Standard Depth)
**Context:** `fno_blocks=4` (The known failure case). 50 Epochs.

**Metrics:**
*   **Primary:** `model.val_loss_name` (e.g., `poisson_val_Amp_loss` or `mae_val_Phase_loss`, depending on mode).
*   **Secondary:** `ssim_phase` (Test set reconstruction quality).

| Arm | Architecture | Clipping Strategy | Hypothesis | Success Condition |
| :--- | :--- | :--- | :--- | :--- |
| **1. Control** | `hybrid` | Global Norm (`val=1.0`) | **Baseline Failure:** Stagnates or explodes. | N/A (Negative Control) |
| **2. Arch Fix** | `stable_hybrid` | **Disabled** (no clipping; `gradient_clip_val=None`) | **Stability:** Converges smoothly with NO clipping. | Lowest `model.val_loss_name`; No NaNs. |
| **3. Opt Fix** | `hybrid` | **AGC** (`val=0.01`) | **Survival:** Survives 50 epochs despite drift. | Lower `model.val_loss_name` than Control. |

### Stage B: The "Deep" Stress Test (Robustness)
**Context:** `fno_blocks=8` (2x Depth).
**Why:** Instability scales with depth. A solution is only valid if it allows scaling.

*   **Protocol:** Run the **Winner of Stage A** with `fno_blocks=8` and `gradient_clip_val=None` (if Arch Fix) or `val=0.01` (if Opt Fix).
*   **Success Condition:** The deep model trains for 50 epochs without NaNs and achieves lower `model.val_loss_name` than the shallow model.

---

## 3. Implementation Plan

### Phase 1: Foundation (Config & Utilities)
Enable the necessary configuration switches and mathematical utilities.

1.  **Execution Config:** Update `PyTorchExecutionConfig` in `ptycho/config/config.py` to add `gradient_clip_algorithm: Literal['norm', 'value', 'agc'] = 'norm'`.
    *   *Note:* Do **not** add this to `TrainingConfig` or `config_bridge`; clipping is a runtime execution concern.
2.  **AGC Utility:** Implement `adaptive_gradient_clip_(parameters, clip_factor, eps)` in `ptycho_torch/train_utils.py`.
3.  **Training Loop:** Update `PtychoPINN_Lightning.training_step` (`ptycho_torch/model.py`) to dispatch the requested clipping algorithm during manual optimization.
4.  **Factory:** Update `ptycho_torch/config_factory.py` to populate `execution_config.gradient_clip_algorithm` from overrides.
5.  **Runners:** Update `grid_lines_torch_runner` and `compare_wrapper` CLI to accept `--gradient-clip-algorithm`.

### Phase 2: Generator Engineering
Construct the stable architecture.

1.  **StablePtychoBlock:** Implement the Norm-Last/Zero-Gamma logic in `ptycho_torch/generators/fno.py`.
    *   `__init__`: `InstanceNorm2d(channels, affine=True)`, `weight.data.fill_(0)`, `bias.data.fill_(0)`.
    *   `forward`: `branch = self.norm(self.act(self.spectral(x) + self.local(x))); return x + branch`.
2.  **StableHybridGenerator:** Subclass `HybridUNOGenerator` to use the stable block.
3.  **Registry:** Register `stable_hybrid` in `ptycho_torch/generators/registry.py`.

---

## 4. Context & Tooling

This section maps the strategic goals to the specific components in the codebase.

### **The "Grid Lines" Harness**
*   **Orchestrator:** `scripts/studies/grid_lines_compare_wrapper.py`
    *   *Role:* The A/B test driver. Needs to support `--gradient-clip-algorithm` to pass to the runner.
*   **Torch Runner:** `scripts/studies/grid_lines_torch_runner.py`
    *   *Role:* The training executor. Parses `--gradient-clip-algorithm` into `PyTorchExecutionConfig`.

### **Configuration System**
*   **Canonical Config:** `ptycho/config/config.py` (`PyTorchExecutionConfig`).
*   **Bridge:** `ptycho_torch/config_bridge.py` is **NOT** used for execution knobs (CONFIG-002). Use `config_factory` overrides instead.

### **Model & Training**
*   **Lightning Module:** `ptycho_torch/model.py` (`PtychoPINN_Lightning`)
    *   *Role:* The `training_step` handles manual optimization. Currently calls `clip_grad_norm_`. Must be updated to switch on `self.training_config` (or rather, the execution config accessible to the trainer, though Lightning 2.0 style usually keeps this in the Trainer or Strategy. Since we do manual optimization, we must invoke the utility directly in `training_step` using config values stored in the module).

### **Diagnostics**
*   **Activation Monitor:** `scripts/debug_fno_activations.py`
    *   *Action:* Run this on `stable_hybrid` before the long training run to verify `mean` stays $\approx 0$ at initialization.

---

## 5. Decision Logic & Outcome

*   **Gold Standard:** If `stable_hybrid` passes Stage A (beats Control) AND Stage B (scales to depth), it becomes the **new default FNO architecture**. The legacy `PtychoBlock` will be deprecated.
*   **Silver Standard:** If `stable_hybrid` fails but `AGC` works, AGC becomes the **mandatory training policy** for FNO models.
*   **Pivot:** If both fail Stage B (Deep Test), pivot to investigating **LayerScale** (explicit depth damping).
