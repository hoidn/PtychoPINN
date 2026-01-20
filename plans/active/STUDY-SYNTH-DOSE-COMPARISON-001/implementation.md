# Implementation Plan: Synthetic Dose Response & Loss Comparison Study

## Initiative
- **ID:** STUDY-SYNTH-DOSE-COMPARISON-001
- **Title:** Synthetic Dose Response & Loss Comparison Study
- **Owner:** Ralph
- **Spec Owner:** `specs/spec-ptycho-core.md` (Physics/Normalization)
- **Status:** pending
- **Unblocked By:** REFACTOR-MODEL-SINGLETON-001 complete (2026-01-07) — lazy loading fixes multi-N shape mismatch
- **Priority:** High (Scientific Validation)
- **Working Plan:** this file
- **Reports Hub:** `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/`

## Goals
1. Compare PtychoPINN reconstruction quality under High Dose (1e9 photons) vs. Low Dose (1e4 photons) conditions using identical scan trajectories.
2. Evaluate the robustness of Poisson NLL loss vs. standard MAE loss across these regimes.
3. Produce a publication-ready 6-panel figure visualizing the input diffraction and resulting reconstructions.
4. Demonstrate a "Pure Python" workflow that orchestrates the library directly without CLI subprocess overhead, adhering to the Modern Coordinate-Based System.

## Phases Overview
- **Phase A — Orchestration & Data Fabric:** Establish the Python script skeleton, generate consistent Ground Truth (Object/Probe), and simulate 4 distinct datasets (Train/Test x High/Low) using library imports and `TrainingConfig`.
- **Phase B — Training & Inference Loop:** Implement the training execution for the 4 experimental arms (High/NLL, High/MAE, Low/NLL, Low/MAE) and the inference logic using `ptycho.workflows.components`.
- **Phase C — Visualization & Delivery:** Implement the custom plotting logic to generate the specific 6-panel layout, archive the study artifacts, and update project tracking documentation.

## Exit Criteria
1. `scripts/studies/dose_response_study.py` exists and completes without exceptions.
2. **Training Evidence:** Four models trained with:
    - Each arm's output directory (`tmp/dose_study/<arm>/`) contains `wts.h5.zip`
    - `training_history.json` is non-empty with at least `train_loss` arrays for all 4 arms
    - Final training loss < initial loss (convergence evidence)
3. **Inference Evidence:** All 4 reconstructions successfully produced:
    - `run_inference()` returns non-None `reconstruction['amplitude']` for each arm
    - Reconstructed amplitude arrays have valid shape (not empty, not all zeros)
4. **Figure Verification:** `dose_comparison.png` shows:
    - All 6 panels contain image data (no "No Data" placeholders)
    - Diffraction panels show High/Low dose difference (visible noise difference)
    - Reconstruction panels show recognizable object structure
5. Test registry check: `pytest --collect-only -q tests/` shows no new failures. Archive to `reports/pytest_collect.log`.
6. Ledger entry in `docs/fix_plan.md` updated with results or blockers.
7. `PROJECT_STATUS.md` updated with initiative completion status.

## Compliance Matrix (Mandatory)
- [ ] **Spec Constraint:** `spec-ptycho-core.md` — Photon statistics and normalization (intensity_scale symmetry). Both X and Y_I divided by intensity_scale per line 141-142 of diffsim.py.
- [ ] **Spec Constraint:** `spec-ptycho-workflow.md` — Loss and Optimization: Positivity guard on Y_pred before log in Poisson NLL.
- [ ] **Finding/Policy ID:** `CONFIG-001` — Must call `update_legacy_dict(params.cfg, config)` before legacy module usage.
- [ ] **Finding/Policy ID:** `NORMALIZATION-001` — Respect distinction between physics scaling and statistical normalization.
- [ ] **Finding/Policy ID:** `PYTHON-ENV-001` — Use active interpreter if spawning subprocesses (goal is pure Python, but applicable if fallback needed).
- [ ] **Finding/Policy ID:** `DATA-001` — Any generated NPZ must follow `specs/data_contracts.md`.
- [ ] **Finding/Policy ID:** `BUG-TF-001` — `params.cfg['gridsize']` must be set before `generate_grouped_data`.
- [ ] **Finding/Policy ID:** `CONVENTION-001` — Declare operation within the Modern Coordinate-Based System.
- [ ] **Finding/Policy ID:** `PINN-CHUNKED-001` — Guard against OOM if test set size exceeds limits (mitigated: small test sets used).

## Spec Alignment
- **Normative Spec:** `specs/spec-ptycho-workflow.md`
- **Key Clauses:**
    - "Forward semantics: When gridsize > 1, forward-path reassembly semantics MUST match..."
    - "Loss and Optimization: ... Positivity guard: Y_pred (intensity) fed to log SHALL be strictly positive..."
    - `spec-ptycho-core.md` §Normalization Invariants: "symmetry SHALL hold: Training inputs X_scaled = s * X, Labels Y_amp_scaled = s * X"

## Architecture / Interfaces
- **Script Entry:** `scripts/studies/dose_response_study.py`
- **Key Data Types:**
    - `TrainingConfig`: Dataclass controlling all training parameters including `nphotons`, `nll_weight`, `mae_weight`
    - `RawData`: Container for simulated diffraction patterns and ground truth
    - `PtychoDataContainer`: GPU-ready tensor container for training
- **Boundary Definitions:**
    - `[Script] -> [TrainingConfig] -> [update_legacy_dict] -> [params.cfg]`
    - `[Ground Truth] -> [generate_simulated_data] -> [RawData] -> [train_cdi_model] -> [Model]`
- **Data Flow:**
    - `[In-Memory Ground Truth (complex64)]` -> `generate_simulated_data(config, obj, probe, buffer)` -> `RawData`
    - `RawData` -> `train_cdi_model(train_data, test_data, config)` -> `{model, history, containers}`
    - `Model` -> `load_inference_bundle(model_dir)` -> `diffraction_to_obj` -> `predict` -> `reassemble_position` -> `reconstruction`
- **Key Functions:**
    - `ptycho.nongrid_simulation.generate_simulated_data`: Conforming public API for simulation.
    - `ptycho.workflows.components.train_cdi_model`: Training orchestration.
    - `ptycho.workflows.components.load_inference_bundle`: Model loading for inference.
    - `ptycho.tf_helper.reassemble_position`: Patch stitching.

## Context Priming (read before edits)
- **Primary docs/specs:**
    - `docs/DEVELOPER_GUIDE.md` Section 3.5 (Normalization Architecture)
    - `docs/debugging/QUICK_REFERENCE_PARAMS.md` (params.cfg initialization)
    - `specs/spec-ptycho-core.md` (photon scaling, intensity_scale)
- **Required findings:**
    - `CONFIG-001`: update_legacy_dict before legacy import
    - `NORMALIZATION-001`: Three distinct normalization systems
- **Data dependencies:**
    - None external; ground truth generated in-memory from `ptycho.diffsim.mk_lines_img`

---

## Phase A — Orchestration & Data Fabric

### Checklist
- [ ] A0: **Nucleus / Test-first gate:** Create `scripts/studies/dose_response_study.py` with import structure. Verify `update_legacy_dict` populates params correctly by logging:
    - `params.cfg['N']` (expect: 64)
    - `params.cfg['gridsize']` (expect: 2)
    - `params.cfg['nphotons']` (expect: 1e9 or 1e4 depending on config)
    - **Artifact:** Console log showing params populated correctly.

- [ ] A1: **Ground Truth Gen:** Generate `objectGuess` and `probeGuess` in memory using conforming APIs.
    ```python
    # Object: Lines pattern (amplitude-only for controlled comparison)
    from ptycho.diffsim import mk_lines_img
    obj_full = mk_lines_img(N=128, nlines=400)  # Returns (N, N, 3) with Gaussian filtering
    # Crop to desired size and convert to complex (amplitude-only)
    obj_amp = obj_full[:, :, 0]  # Take first channel
    obj_amp = obj_amp / obj_amp.max()  # Normalize to [0, 1]
    obj_amp = 0.5 + obj_amp  # Shift to [0.5, 1.5] for reasonable amplitude range
    objectGuess = obj_amp.astype(np.complex64)

    # Probe: Default disk probe (requires params.cfg['default_probe_scale'])
    from ptycho import probe
    probeGuess = probe.get_default_probe(N=64, fmt='np').astype(np.complex64)
    ```
    - **Rationale for amplitude-only object:** Using real-valued (zero-phase) ground truth isolates the dose/loss comparison from phase-wrapping ambiguities, enabling cleaner interpretation of reconstruction quality differences.
    - **Constraint:** `objectGuess.imag == 0` (amplitude only).
    - **Prerequisite:** Call `update_legacy_dict` first to set `params.cfg['default_probe_scale']`.

- [ ] A2: **Simulation Loop:** Implement generation of 4 `RawData` objects using `ptycho.nongrid_simulation.generate_simulated_data(config, objectGuess, probeGuess, buffer)`.
    - **Config structure:** `TrainingConfig(model=ModelConfig(N=64, gridsize=2), n_groups=..., nphotons=...)`
    - **Arms:**
        | Arm | nphotons | n_groups | seed | Purpose |
        |-----|----------|----------|------|---------|
        | High Dose Train | 1e9 | 2000 | 1 | Training data |
        | High Dose Test | 1e9 | 128 | 2 | Evaluation data |
        | Low Dose Train | 1e4 | 2000 | 1 | Training data |
        | Low Dose Test | 1e4 | 128 | 2 | Evaluation data |
    - **Note:** `n_groups` in `TrainingConfig` controls raw scan positions generated. Grouping into solution regions happens downstream in `train_cdi_model` based on `gridsize` and `neighbor_count`. With `gridsize=2, K=7`, 2000 raw images yield ample groups for training.
    - **Note:** Same seed (1) for both train sets ensures identical scan trajectories, isolating photon statistics as the only variable between High/Low dose training data.
    - **⚠️ API Constraint:** `generate_simulated_data` internally hardcodes `random_seed=42` in `_generate_simulated_data_legacy_params`. External `np.random.seed()` calls are overridden. To achieve different seeds per arm, either:
        1. Modify the API to accept a seed parameter (preferred), or
        2. Generate all data with the same trajectory and accept this limitation for the study.

- [ ] A3: **Sanity Check:** Add a block to print shapes and mean intensities of the generated data to confirm dose scaling.
    - Expected: ~316× factor difference in mean amplitude between High and Low dose (since `diff3d` stores amplitude = sqrt(counts), the ratio is sqrt(1e9/1e4) = sqrt(1e5) ≈ 316).
    - **Artifact:** Console log with shape/intensity table.

### Dependency Analysis
- **Touched Modules:** None modified (new script only).
- **Circular Import Risks:** None; script imports from stable public APIs.
- **State Migration:** N/A.

### Notes & Risks
- **Risk:** Global state leakage between simulation calls if `params.cfg` isn't reset.
- **Mitigation:** `generate_simulated_data` handles legacy adapter logic internally via try/finally (see `nongrid_simulation.py:169-183`).
- **Risk:** `get_default_probe` fails if `params.cfg['default_probe_scale']` not set.
- **Mitigation:** Call `update_legacy_dict(params.cfg, config)` before probe generation.

### Rollback
- Revert: `git checkout HEAD -- scripts/studies/dose_response_study.py`
- Cleanup: `rm -rf tmp/dose_study/`

---

## Phase B — Training & Inference Loop

### Checklist
- [ ] B1: **Config Builder:** Implement helper function to construct `TrainingConfig` for the 4 experimental arms:
    ```python
    def make_config(nphotons: float, loss_type: str, output_subdir: str) -> TrainingConfig:
        nll_weight = 1.0 if loss_type == 'nll' else 0.0
        mae_weight = 1.0 if loss_type == 'mae' else 0.0
        return TrainingConfig(
            model=ModelConfig(N=64, gridsize=2),
            n_images=2000,
            nphotons=nphotons,
            nll_weight=nll_weight,
            mae_weight=mae_weight,
            nepochs=50,  # Adjust as needed
            output_dir=Path(f'tmp/dose_study/{output_subdir}'),
            # ... other defaults
        )
    ```
    - **Arms:**
        | Arm | nphotons | nll_weight | mae_weight |
        |-----|----------|------------|------------|
        | High/NLL | 1e9 | 1.0 | 0.0 |
        | High/MAE | 1e9 | 0.0 | 1.0 |
        | Low/NLL | 1e4 | 1.0 | 0.0 |
        | Low/MAE | 1e4 | 0.0 | 1.0 |

- [ ] B2: **Training Execution:** Implement the loop over the 4 arms.
    ```python
    results = {}
    for arm_name, (train_data, test_data, config) in arms.items():
        update_legacy_dict(params.cfg, config)  # Critical: before each training run
        result = train_cdi_model(train_data, test_data, config)
        results[arm_name] = result
    ```
    - Save models to `tmp/dose_study/<arm_name>/`.
    - Store training history for convergence analysis.
    - **Artifact:** 4 model directories with `wts.h5.zip`.

- [ ] B3: **Inference Execution:** Implement inference on the Test sets.
    - **⚠️ CRITICAL:** Use `load_inference_bundle(model_dir)` from `ptycho.workflows.components` to retrieve the `diffraction_to_obj` model from disk. Do NOT attempt to extract it from `train_cdi_model` results dict (it returns `model_instance` which is the autoencoder, not `diffraction_to_obj`).
    ```python
    from ptycho.workflows.components import load_inference_bundle, create_ptycho_data_container

    model, params_snapshot = load_inference_bundle(config.output_dir)
    test_container = create_ptycho_data_container(test_data, config)
    predictions = model.predict([test_container.X, test_container.coords_nominal])
    recon = reassemble_position(predictions, test_container.coords_nominal, M=20)
    ```
    - Store results in dictionary keyed by arm name.
    - **Verification:** `predictions` should have shape `(n_groups, N, N, 1)` with complex64 dtype.
    - **Artifact:** Reconstructed amplitude/phase arrays for each arm.

### Notes & Risks
- **Risk:** `intensity_scale` state pollution between arms.
- **Mitigation:** Explicit `update_legacy_dict(params.cfg, config)` call before every `train_cdi_model` invocation resets all relevant params.
- **Risk:** OOM during inference.
- **Mitigation:** Test set is small (128 images -> ~32 groups with gridsize=2), well under PINN-CHUNKED-001 threshold.

### Rollback
- Revert: `git checkout HEAD -- scripts/studies/dose_response_study.py`
- Cleanup: `rm -rf tmp/dose_study/`

---

## Phase C — Visualization & Delivery

### Checklist
- [ ] C1: **Plotting Logic:** Implement `generate_six_panel_figure` using `matplotlib.gridspec`.
    ```
    Layout:
    ┌─────────────────────────────────────────┐
    │  High Dose   │  High Dose   │ High Dose │
    │  Diffraction │  MAE Recon   │ NLL Recon │
    │  (log scale) │  (amplitude) │ (amplitude)│
    ├─────────────────────────────────────────┤
    │  Low Dose    │  Low Dose    │ Low Dose  │
    │  Diffraction │  MAE Recon   │ NLL Recon │
    │  (log scale) │  (amplitude) │ (amplitude)│
    └─────────────────────────────────────────┘
    ```
    - Crop: Central region (e.g., 128x128 pixels) to show detail.
    - Color scaling: Each reconstruction normalized to its own [min, max] for visibility.
    - Diffraction: Log scale with consistent vmin/vmax across doses to show noise difference.

- [ ] C2: **Execution & Archive:** Run the full study.
    - Save `dose_comparison.png` to `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/`.
    - Commit the script to repository.
    - **Artifact:** Final 6-panel figure.

- [ ] C3: **Cleanup:** Ensure `tmp/dose_study/` artifacts are cleaned up or gitignored.
    - Add `tmp/` to `.gitignore` if not already present.
    - Optionally archive model weights to `.artifacts/` with symlinks from reports hub.

- [ ] C4: **Doc Sync:** Update project tracking documentation.
    - `docs/fix_plan.md`: Add ledger entry or update existing with completion status.
    - `PROJECT_STATUS.md`: Mark initiative as complete or document findings.
    - **Artifact:** Updated ledger entries.

- [ ] C5: **Test Registry Check:** Run `pytest --collect-only -q tests/` and archive output.
    - Verify no test regressions from new script.
    - Save to `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/pytest_collect.log`.

### Notes & Risks
- **Risk:** Color scaling differences obscure true reconstruction quality comparison.
- **Mitigation:** Use consistent scaling within each row (High/Low) or add quantitative metrics (PSNR, SSIM) as annotations.
- **Risk:** Figure too busy for publication.
- **Mitigation:** Consider separate detailed figures per dose level if 6-panel is cluttered.

### Rollback
- Revert: `git checkout HEAD -- scripts/studies/dose_response_study.py`
- Cleanup: `rm -rf tmp/dose_study/ plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/*.png`

---

## Open Questions & Follow-ups
- [ ] **Quantitative metrics:** Should we add PSNR/SSIM/FRC metrics to the figure annotations or a separate table?
- [ ] **Phase reconstruction:** This study uses amplitude-only ground truth. Follow-up study with complex-valued objects to validate phase reconstruction under dose/loss variations?
- [ ] **Convergence analysis:** Should training curves be included in the deliverables to show loss convergence behavior?

## Artifacts Index
- Reports root: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/`
- Expected artifacts:
    - `dose_comparison.png` — Final 6-panel figure
    - `pytest_collect.log` — Test registry verification
    - `training_history.json` — Convergence data (optional)
