# Strategy: Scalable, Robust Training for Deep FNOs

## Strategic Pivot

**Goal (updated):** Enable scalable, robust training for deep FNOs (depth 6+).

**Hypothesis (refined):** Instability is accumulative and stochastic. At shallow depths (4), some initializations survive. At greater depths, failure probability (P_crash) approaches 100%. We must validate solutions by reducing P_crash to 0% across multiple random seeds.

**Findings that drive the pivot:**
- **Zero-gamma dead branches:** Zero-gamma InstanceNorm suppresses the residual branch, causing collapsed reconstructions.
- **Stochastic instability:** Single runs are not reliable; stability must be measured by failure rate across seeds.
- **Depth blocked by memory:** Depth 8 models OOM on 24 GB GPUs without architectural caps.
- **Late-epoch explosions remain unexplained:** Damping at initialization (LayerScale) only helps if drift starts early. If explosions trigger late (e.g., LR schedule transitions, optimizer dynamics), fixes must target training dynamics directly.

---

## 1) Infrastructure Upgrades (Prerequisites)

Before stability testing, the architecture must be physically runnable at depth.

### A) Memory Cap
- **Parameter:** `max_channels` for `HybridUNOGenerator` (config name: `max_hidden_channels`).
- **Logic:** Channels double on downsample (32 -> 64 -> 128 -> ...) until they hit `max_channels` (default 512), then stay constant.
- **Rationale:** Prevents parameter explosion (>4B at depth 8) and enables deep runs on 24 GB GPUs.

### B) Data Validity (Phase Metrics)
- **Requirement:** All grid-lines dataset generation must use `--set-phi`.
- **Rationale:** Without `--set-phi`, phase metrics are not meaningful.

---

## 2) Redefined Solution Candidates

The prior fixes (Zero-Gamma stable_hybrid and aggressive AGC 0.01) failed in practice. Replace them with the following pivoted candidates.

### A) Architecture (Pivot): LayerScale
- **Replace:** StablePtychoBlock (Norm-Last + Zero-Gamma)
- **With:** LayerScalePtychoBlock
- **Mechanism:** y = x + diag(lambda) * Branch(x)
- **Initialization:** lambda initialized to a small non-zero value (e.g., 1e-2).
- **Rationale:** Damps residual updates for stability without dead-branch risk.
- **Scope note:** This is a stability prior, not a complete fix. If failures occur late, LayerScale alone is insufficient.

### B) Optimization (Relaxed Constraint): Soft AGC
- **Update:** AGC clip factor from 0.01 -> 0.1.
- **Rationale:** Avoids over-clipping while still limiting extreme outliers.

---

## 3) Experimental Protocol (New)

### Protocol A: The "Crash Hunt" (Calibration)
**Goal:** Find the depth where the control fails reliably.

- **Config:** Control = Hybrid, `max_channels=512`.
- **Sweep:** Depth in [4, 6, 8].
- **Stop Condition:** The shallowest depth where >= 1 out of 3 runs fails (NaN or loss spike).
- **Planning assumption:** Depth 6 is the likely crash depth.

### Protocol B: The Stochastic "Shootout"
**Goal:** Compare candidates at the crash depth using multiple seeds.

- **Runs:** 3 seeds per arm (minimum).
- **Success metric:** P_crash = 0% (3/3 runs succeed) and competitive loss.

| Arm | Configuration | Seeds | Success Metric |
| --- | --- | --- | --- |
| Control | Hybrid (Depth X), Clip 1.0 | 3 | Baseline failure: >=1 run fails or high variance |
| Arch Fix | LayerScale Hybrid (Depth X), No Clip | 3 | Robustness: 3/3 runs converge; lower variance than Control |
| Opt Fix | Hybrid (Depth X), AGC 0.1 | 3 | Survival: 3/3 runs survive; loss competitive |

---

## 4) Hypothesis Backlog (Supervisor-Owned)

The supervisor should prioritize and test the following hypotheses when late-epoch explosions persist:

1. **Training dynamics trigger:** LR warmup-to-cosine transitions or peak LR spikes cause the late collapse. (Test: lower peak LR, extend warmup, remove transition.)
2. **Optimizer-specific instability:** Adam/AdamW induces late drift; SGD+momentum may stabilize. (Test: optimizer sweep at fixed depth.)
3. **Targeted clipping efficacy:** Clip only at transition epochs, or use per-layer norm clipping. (Test: schedule-aware clipping.)
4. **Architecture normalization variant:** Pre-Norm/RMSNorm yields better late-phase stability than Norm-Last. (Test: swap norm placement.)
5. **Constant-width blocks:** Avoid channel doubling to reduce variance amplification. (Test: constant-width encoder/decoder.)
6. **Data/loss scaling:** Late explosions are driven by loss scaling or phase noise. (Test: loss re-weighting or normalization sweep.)

The supervisor should sequence these by expected impact and cost, and update the plan/Do Now each loop accordingly.

---

## 5) Evidence to Date (Historical)

### Stage A (single-seed) outcome, 2026-01-29
**Config:** N=64, gridsize=1, fno_blocks=4, nimgs_train=2, nimgs_test=2, nphotons=1e9, loss=MAE, seed=20260128, epochs=50.

| Arm | Architecture | Clipping | Best val_loss | Amp SSIM | Phase SSIM | Amp MAE | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Control | hybrid | norm (1.0) | 0.0138 | 0.925 | 0.997 | 0.063 | No failure |
| AGC | hybrid | agc (0.01) | 0.0243 | 0.811 | 0.989 | 0.102 | Stable but slower |
| Stable | stable_hybrid | none (0.0) | 0.1783 | 0.277 | 1.000 | 0.513 | Collapsed amplitude |

**Interpretation:** Control did not explode at depth 4. Stable_hybrid collapsed due to dead branches. AGC 0.01 over-damped learning.

### Stage B (depth 8) OOM blocker
- Depth 8 OOMs on RTX 3090 (24 GB) due to exponential channel growth.
- Even with `max_hidden_channels=512`, depth 8 still OOMs.
- Depth 6 with cap is feasible and should be treated as the current deep stress depth unless checkpointing or a larger GPU is available.

---

## 6) Retired Attempts (Do Not Reuse)

- **Zero-Gamma stable_hybrid:** Dead branch failure mode (collapsed amplitude output).
- **AGC 0.01:** Overly conservative; degraded performance without improving stability.

---

## 7) Implementation Plan (Updated)

1. Ensure `max_hidden_channels` is present in ModelConfig, PyTorch config, CLI, and generators.
2. Enforce `--set-phi` in dataset generation for any run reporting phase metrics.
3. Replace stable_hybrid block with LayerScalePtychoBlock (small non-zero lambda init).
4. Update AGC default threshold to 0.1 for experiments.
5. Run Protocol A (Crash Hunt) to identify crash depth.
6. Run Protocol B (Shootout) at the crash depth with 3 seeds per arm.
