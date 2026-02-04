# Grid-Lines Torch CNN-PINN Parity Design (Two-Phase)

**Date:** 2026-02-01
**Owner:** Codex
**Scope:** Enable CNN-PINN to run via PyTorch on grid-lines workflows with parity to TensorFlow.

## Goals
- Phase 1: Run CNN-PINN in PyTorch **on the same TF-generated grid-lines NPZs**.
- Phase 2: Provide a **Torch-native grid-lines simulator** to remove TF dependency.
- Preserve existing data contract (`specs/data_contracts.md`) and metadata schema.

## Non-Goals
- Replacing TF training/inference paths.
- Changing the physics model, noise model, or grid-lines dataset semantics.

---

## Phase 1 — Reuse TF NPZ, Torch CNN-PINN

### Intent
Move training/inference to PyTorch while keeping the dataset identical to TF. This yields a clean A/B comparison on the same inputs and minimizes simulation risk.

### Approach
1. **Dataset Reuse**
   - Continue generating datasets via `ptycho/workflows/grid_lines_workflow.py` (TF legacy sim) with output under:
     `outputs/.../datasets/N{N}/gs{g}/{train,test}.npz`.

2. **Torch Runner Extension**
   - Extend `scripts/studies/grid_lines_torch_runner.py` to accept `architecture=cnn`.
   - Build the CNN generator from `ptycho_torch/generators/cnn.py` (already in registry).
   - Mirror TF training settings: `N`, `gridsize`, `nphotons`, `nll_weight`, `mae_weight`, `realspace_weight`.
   - Use `torch_loss_mode=poisson` when `model_type=pinn` for PINN-like loss parity.

3. **Artifacts and Metrics**
   - Save recon artifacts using `save_recon_artifact` and render visuals using `render_grid_lines_visuals`.
   - Compute metrics against `YY_ground_truth` in the NPZ to match TF.

### Success Criteria
- Torch CNN run completes on TF-generated NPZ without shape/scale errors.
- `recons/pinn_cnn/recon.npz`, `visuals/compare_amp_phase.png`, and `metrics.json` are produced.
- Comparable metrics between TF CNN and Torch CNN on the same dataset.

---

## Phase 2 — Torch-Native Simulation (End-to-End)

### Intent
Provide a Torch-native simulator that reproduces the TF grid-lines dataset semantics and outputs identical NPZ structure. This removes TF dependency for the grid-lines workflow.

### Approach
1. **Torch Simulator Module**
   - Create `ptycho_torch/sim/grid_lines_sim.py` that returns:
     `X_train`, `Y_I_train`, `Y_phi_train`, `X_test`, `Y_I_test`, `Y_phi_test`,
     `coords_nominal`, `coords_true`, `YY_full`, `YY_ground_truth`, `norm_Y_I`,
     plus `probeGuess` and metadata.

2. **Probe Handling Parity**
   - Factor probe scaling into a shared backend-agnostic helper (NumPy/Torch parity):
     - Scaling modes: `interpolate` and `pad_extrapolate`.
     - Smoothing sigma.
     - Optional centered disk mask.

3. **Noise + Scaling Parity**
   - Implement the same Poisson intensity scaling and photon count normalization used in TF.
   - Validate with a fixed seed vs TF outputs for small `N` (numerical tolerance).

4. **NPZ Writer + Metadata**
   - Persist with `MetadataManager.save_with_metadata` to keep downstream compatibility.

5. **Runner Wiring**
   - Add `--sim-backend tf|torch` to `grid_lines_torch_runner.py`.
   - Default to `tf` until parity tests pass.

### Success Criteria
- Torch sim produces NPZs with identical keys and shapes to TF sim.
- Torch sim matches TF sim within tolerance for fixed seeds.
- End-to-end Torch pipeline runs without TF dependencies.

---

## Risks / Open Questions
- Exact intensity scaling parity between TF PINN and Torch PINN.
- Grid-lines sim parity: coordinate generation and noise injection must match TF to avoid subtle metric drift.

## Testing Strategy
- Phase 1: add a Torch runner test asserting `architecture=cnn` path and artifact creation.
- Phase 2: add a parity test comparing TF vs Torch sim outputs on a fixed seed.

