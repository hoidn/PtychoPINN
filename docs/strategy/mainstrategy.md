# Strategy: FNO/Hybrid Stability & Optimization Overhaul

**Objective:** Stabilize deep FNO/Hybrid model training (currently prone to gradient explosion at >30 epochs) by eliminating the mathematical root cause of signal drift, without relying on aggressive global gradient clipping.

**Core Hypothesis:** The instability is accumulative. The current `y = x + GELU(...)` topology adds a positive bias at every block. As depth (`fno_blocks`) increases, this drift accelerates, pushing latent statistics into saturation. Instability is treated as **probabilistic** (some seeds fail, others do not).
*   *Evidence (to link):* Report failure rate across multiple seeds (e.g., 3/10 runs explode) rather than a single run. (Add run IDs/log paths or remove this line if no citations.)
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
**Context:** `fno_blocks=4` (the known failure case). **Cap test/comparison runs at 20 epochs** (Stage A historical run used 50).

**Metrics:**
*   **Primary:** `model.val_loss_name` (e.g., `poisson_val_Amp_loss` or `mae_val_Phase_loss`, depending on mode).
*   **Secondary:** `ssim_phase` (Test set reconstruction quality).
*   **Stability:** Failure rate across seeds + time-to-failure distribution.
*   **Epoch cap:** Test/comparison runs must use **≤20 epochs** unless explicitly labeled as a stress test.

**Protocol:**
*   Dataset generation must use `--set-phi`; otherwise phase metrics are meaningless.
*   Run each arm across **N seeds** (suggested: 5–10). Report median + IQR for metrics **conditional on successful runs**.
*   **Run sizing:** use `nimgs_train=1` and `nimgs_test=1` for both test and full runs to cut runtime and GPU memory.

| Arm | Architecture | Clipping Strategy | Hypothesis | Success Condition |
| :--- | :--- | :--- | :--- | :--- |
| **1. Control** | `hybrid` | **Disabled** (no clipping; `gradient_clip_val=None`) | **Baseline Failure:** Stagnates or explodes. | N/A (Negative Control) |
| **2. Arch Fix** | `stable_hybrid` | **Disabled** (no clipping; `gradient_clip_val=None`) | **Stability:** Converges smoothly with NO clipping. | Lowest `model.val_loss_name`; No NaNs; lowest failure rate. |
| **3. Opt Fix** | `hybrid` | **AGC** (`val=0.01`) | **Survival:** Survives 20 epochs despite drift. | Lower failure rate than Control; `model.val_loss_name` not worse by >10% (median, successful runs only). |

**Stage A outcome (2026-01-29):** Single-seed execution (N=64, gridsize=1, `fno_blocks=4`, `nimgs_{train,test}=2`, `nphotons=1e9`, MAE loss, 50 epochs, seed=20260128) produced the following metrics. Going forward, full runs should use `nimgs_train=1` and `nimgs_test=1`, and **test/comparison runs must not exceed 20 epochs**.

| Arm | Architecture | Clipping | Best val_loss | Amp SSIM | Phase SSIM | Amp MAE | Notes |
|-----|--------------|----------|---------------|----------|------------|---------|-------|
| Control | hybrid | norm (1.0) | **0.0138** | **0.925** | **0.997** | **0.063** | Healthy variance, no failures |
| AGC | hybrid | agc (0.01) | 0.0243 | 0.811 | 0.989 | 0.102 | Stable but slower convergence |
| Stable | stable_hybrid | none (0.0) | 0.1783 | 0.277 | 1.000 | 0.513 | Collapsed amplitude; InstanceNorm weights stayed near zero |

Control unexpectedly dominated; the hypothesized drift did not appear at 4 blocks/50 epochs. `stable_hybrid` stagnated because zero-gamma InstanceNorm prevented residual learning, and AGC’s 0.01 threshold over-damped gradients. Stage B therefore promotes the control arm to the deep test while we revisit the architectural hypothesis.

**Clip sweep (control sensitivity):**
*   Run the control arm with global norm clip values across a wide range (e.g., `val` in {1, 5, 10, 20, 50, 100}) to confirm conclusions are not artifacts of aggressive clipping.

### Stage B: The "Deep" Stress Test (Robustness)
**Context:** `fno_blocks=8` (2x Depth).
**Why:** Instability scales with depth. A solution is only valid if it allows scaling.

*   **Protocol:** Run the **Winner of Stage A** (control arm: `hybrid`, norm clip 1.0) with `fno_blocks=8`, same dataset/probe, and log gradient norms every epoch (`--torch-log-grad-norm`).
*   **Success Condition:** The deep model avoids gradient explosion (no NaNs, grad_norm bounded) and matches or beats the Stage A val_loss/SSIM metrics.
*   **Run sizing:** use `nimgs_train=1` and `nimgs_test=1` for full runs; do not scale back to 2 unless explicitly required.
*   **Epoch cap:** Test/comparison runs must use **≤20 epochs** unless explicitly labeled as a stress test.

**Stage B plan (scheduled 2026-01-29T18:00Z):**
- Output dir: `outputs/grid_lines_stage_b/deep_control`
- Artifacts: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/`
- Command template mirrors Stage A control but with `--fno-blocks 8` and grad norm logging enabled.
- After the run, `scripts/internal/stage_a_dump_stats.py` captures stats, and `stage_b_summary.md` compares Stage A vs Stage B.

**Stage B outcome (2026-01-28):** BLOCKED — `fno_blocks=8` is infeasible on RTX 3090 (24 GB). The `HybridUNOGenerator` channel-doubling encoder produces 4.4B parameters (17.7 GB FP32) at 8 blocks vs 17M at 4 blocks. The bottleneck reaches 4096 channels, and spectral conv weights scale as `channels^2 * modes^2`. CUDA OOM occurs at model load before any training step. This is a **structural scalability issue**, not a gradient stability issue. See `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T180000Z/stage_b_summary.md` for full analysis.

**Remediation outcome (Task 4.5, 2026-01-29):** `max_hidden_channels=512` cap wired end-to-end (ModelConfig → TorchRunnerConfig → CLI). `fno_blocks=8` with cap still OOMs (18 GiB allocation at model-to-device; spectral conv activations at 512 channels dominate). **Fallback `fno_blocks=6`** succeeded: 276M params, 1.1 GB model, ~10.4 GB VRAM. 50 epochs completed with **no NaNs, no gradient explosions** (grad_norm median=8.56, p99=22.85, single max spike=39927 in 43K steps). Metrics: val_loss=0.024, amp_ssim=0.815, phase_ssim=0.987, amp_mae=0.105. Lower than Stage A (nimgs=2→1 dataset change, so not directly comparable). Stability stress test **partially passed**: cap mechanism works, 6-block depth feasible, 8-block requires gradient checkpointing or larger GPU. Artifacts: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T210000Z/`.

**Phase 5 outcome (LayerScale — 2026-01-29):** LayerScale (init=1e-3) replaced zero-gamma InstanceNorm in `StablePtychoBlock`. Norm weights now train to healthy values (mean~0.82-1.0 vs ~0.09 with zero-gamma), confirming `STABLE-GAMMA-001` resolved. However, final metrics match the zero-gamma run:

| Arm | Architecture | Clipping | Best val_loss | Final val_loss | Amp SSIM | Phase SSIM | Amp MAE | Notes |
|-----|--------------|----------|---------------|----------------|----------|------------|---------|-------|
| LayerScale | stable_hybrid | none (0.0) | **0.0237** | 0.1790 | 0.277 | 1.000 | 0.513 | Early convergence → late collapse; norm weights healthy |

The best val_loss (0.024) approaches control (0.014), but training collapses after epoch ~4, ending at 0.179 with constant amplitude output. Failure mode shifted from "can't learn" (architecture/STABLE-GAMMA-001) to "learns then collapses" (training dynamics/STABLE-LS-001). Next investigation: LR warmup/cosine schedule, reduced LR, or optimizer tuning for stable_hybrid.

**Phase 6 scope (2026-01-29):** Authored `docs/plans/2026-01-29-stable-hybrid-training-dynamics.md` to surface Warmup+Cosine scheduler knobs (config/CLI), implement a deterministic scheduler helper in `PtychoPINN_Lightning`, and rerun the Stage A stable arm with shared datasets. Success criteria: stable training past epoch 5 without amplitude collapse and improved val_loss vs. LayerScale-only runs.

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

*   **Gold Standard:** If `stable_hybrid` passes Stage A (beats Control on median loss **and** has lower failure rate) AND Stage B (scales to depth with lower failure rate), it becomes the **new default FNO architecture**. The legacy `PtychoBlock` will be deprecated.
*   **Silver Standard:** If `stable_hybrid` fails but `AGC` reduces failure rate without >10% median‑loss regression, AGC becomes the **mandatory training policy** for FNO models.
*   **Pivot:** If both fail Stage B (Deep Test), pivot to investigating **LayerScale** (explicit depth damping).
