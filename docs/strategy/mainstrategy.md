# Strategy: Scalable, Robust Training for Deep FNOs

## Strategic Pivot

**Goal (updated):** Enable scalable, robust training for deep FNOs (depth 6+).

**Hypothesis (refined):** Instability is accumulative and stochastic. At shallow depths (4), some initializations can survive. At greater depths, failure probability (P_crash) rises toward 100%. We must validate solutions by reducing P_crash to 0% across multiple random seeds.

**Key findings driving the pivot:**
- **Zero-Gamma dead branches:** Zero-gamma InstanceNorm in the stable block suppresses branch learning, producing near-identity behavior and collapsed reconstructions.
- **Stochastic instability:** Single-shot runs are not reliable. Stability must be measured as a failure rate across seeds.
- **Depth blocked by memory:** Deep FNOs (depth 8) are currently OOM on 24 GB GPUs without architectural caps.

---

## 1) Infrastructure Upgrades (Prerequisites)

Before stability testing, the architecture must be physically runnable at depth.

### A) Memory Cap
- **Parameter:** `max_channels` for `HybridUNOGenerator` (config name: `max_hidden_channels`).
- **Logic:** Channels double on downsample (32 -> 64 -> 128 -> ...) **until** they hit `max_channels` (default 512), then stay constant.
- **Rationale:** Prevents parameter explosion (>4B at depth 8) and enables deep runs on 24 GB GPUs.

### B) Data Validity (Phase Metrics)
- **Requirement:** All grid-lines dataset generation must use `--set-phi`.
- **Rationale:** Without `--set-phi`, phase metrics are not meaningful and should not be compared.

---

## 2) Redefined Solution Candidates

The prior fixes (Zero-Gamma stable_hybrid and aggressive AGC 0.01) failed in practice.
We replace them with the following pivoted candidates.

### A) Architecture (Pivot): LayerScale
- **Replace:** StablePtychoBlock (Norm-Last + Zero-Gamma)
- **With:** LayerScalePtychoBlock
- **Mechanism:**
  - y = x + diag(lambda) * Branch(x)
- **Initialization:** lambda initialized to a small non-zero value (e.g., 1e-2).
- **Rationale:** Damps residual updates for stability without dead-branch risk.

### B) Optimization (Relaxed Constraint): Soft AGC
- **Update:** AGC clip factor from 0.01 -> 0.1.
- **Rationale:** Avoids over-clipping and allows healthy learning dynamics while still limiting extreme outliers.

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

## 4) Decision Logic

- **Winner:** The candidate that drives P_crash to 0% at depth X **and** delivers comparable or better median loss.
- **If no winner:** Re-evaluate architecture (PreNorm or constant-width blocks) and/or pivot optimizer (AdamW/SGD).

