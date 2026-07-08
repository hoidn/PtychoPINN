# ADR: Non-ML Single-Shot CDI Benchmark

Status: drafted design for planning
Date: 2026-04-12
Study id: `non-ml-single-shot-cdi-benchmark`
Source seed: `/home/ollie/Documents/ptychopinnpaper2/revision_designs/non_ml_single_shot_cdi_benchmark.md`
Revision checklist: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`

This document rewrites the design-phase seed in place, as allowed by the adapter instructions. It is the design source of truth for the workflow after this pass; the paper-repo source seed remains provenance.

## Decision

Implement a bounded, reproducible non-ML single-shot CDI benchmark for the Table 2 synthetic line-pattern condition, using **PyNX.cdi** (`pynx.cdi`) as the primary classical baseline. PyNX is a maintained, pip/conda-installable library (CeCILL-B license, developed at ESRF) that ships production-quality HIO, ER, RAAR, and CF algorithms for 2D single-frame CDI with arbitrary support masks and random-phase restarts — exactly the reviewer-requested method class. It is the default precisely because writing HIO/ER from scratch creates avoidable maintenance burden and invites reviewer skepticism about implementation correctness for a *standard* comparator.

Primary implementation path (PyNX.cdi):

- Use the current PtychoPINN checkout, not a worktree.
- Add a narrow study script under `scripts/reconstruction/` or `scripts/studies/` that **wraps PyNX.cdi** as the solver and owns only the Table-2 I/O, probe-support construction, stitching, ambiguity policy, and manifest writing.
- Reuse the existing grid-lines simulation, probe-preparation, stitching, and evaluation conventions for Table 2.
- Reconstruct each `C_g=1` test frame by constructing a `pynx.cdi.CDI` object from the stored normalized Fourier magnitude `X` and a probe-amplitude-derived support mask, then running PyNX's HIO → ER operator chain with pre-registered iteration counts, beta, and restart seeds.
- Recover the object patch `O_norm = psi / P_safe` from the PyNX-returned exit wave, then stitch into the same test-split object used for Table 2 metrics.
- Evaluate with the existing `ptycho.evaluation.eval_reconstruction` metric convention where possible, after the ambiguity policy below is applied.
- Defer paper, changelog, and checklist edits until the implementation plan has reviewer-ready evidence or an explicit pivot outcome.

Secondary object-domain diagnostic/candidate path:

- Add an explicitly labeled repo-local solver mode, `known_probe_object_hio_er`, when the study needs to distinguish support-constrained exit-wave CDI from known-probe object-domain HIO/ER.
- This path treats the object patch `O` as the unknown and applies the known probe inside every Fourier projection: `O -> P * O -> FFT -> detector-amplitude projection -> least-squares projection back to O`.
- It may reuse the same support mask, restart seed policy, direct-stitch metric contract, and manifest machinery, but its metrics and row labels must include `known_probe_object_hio_er` so it cannot be confused with the PyNX exit-wave CDI row.
- This path is not evidence that PyNX was rejected and does not replace the external-standard PyNX row by default. Treat it as diagnostic or candidate reviewer-facing evidence until the outcome review accepts it under the same data, metric, ambiguity, and no-oracle-selection contracts.

Fallback path (only if PyNX is rejected by the solver-discovery gate below):

- In-repo support-constrained HIO/ER implementation with the same reviewer-facing contract. This is **not** the default because (a) reviewer 3 asked for a "standard" algorithm and a recognized library name is stronger evidence than a bespoke implementation, and (b) the bespoke path multiplies the surface area that has to be self-consistency-tested before any metric is trusted.
- The fallback may be adopted only after the solver-discovery manifest documents the specific reason PyNX cannot be used (install failure in `ptycho311`, API incompatibility with the Table-2 normalization convention, license concern, or licensing/provenance failure), and only for the duration of this study.

If neither the PyNX route nor the fallback HIO/ER route can produce a clean, comparable single-shot CDI baseline inside the bounded attempt, pivot to a scoped manuscript and reviewer-response revision. The pivot must explain the attempted solver paths, why they are not valid or reproducible comparators, and how the manuscript claims are narrowed.

## Problem and Scope

Reviewer 3 asks for a benchmark of overlap-free `C_g=1` PtychoPINN against a standard non-ML single-shot phase retrieval algorithm, naming ER/HIO with support constraints or ADMM. The reviewer points to Table 2, where the current overlap-free result is compared against overlapped PtychoPINN but not against a classical single-shot CDI method.

The relevant manuscript claim surface is narrow:

- Abstract: overlap-free single-shot reconstruction with experimental-probe amplitude SSIM `0.904`.
- Methods: "single-shot" means one diffraction measurement with structured probe and no lateral scanning.
- Results/Table 2: synthetic line-pattern overlap ablation, `C_g=1` versus `C_g=4`, idealized and experimental probes.
- Discussion: probe diversity and overlap are treated as partially substitutable constraints.

The study answers only this reviewer-facing question:

> Under the Table 2 synthetic line-pattern, known-probe, `C_g=1` setting, how does PtychoPINN compare to a support-constrained non-ML single-shot CDI reference, and what prior information does that reference use?

It does not attempt a broad survey of classical phase retrieval or a definitive proof that PtychoPINN dominates all non-ML CDI solvers.

## Core Contracts and Invariants

### Table 2 Comparability

The benchmark is only reviewer-facing if it matches the Table 2 evaluation contract or states deviations explicitly.

Required constants:

- `N=64`
- `gridsize=1`
- `data_source="lines"`
- `size=392`
- `offset=4`
- `outer_offset_train=8`
- `outer_offset_test=20`
- `nimgs_train=2`
- `nimgs_test=2`
- `nphotons=1e9`
- probe source: `datasets/Run1084_recon3_postPC_shrunk_3.npz`
- probe variants: idealized disk and experimental/custom probe, matching Table 2 labels
- custom probe transform: preserve the Table 2 path unless provenance proves a different setting
- test split: bottom-half spatial split used by the existing paper JSON
- metric source: `ptycho.evaluation.eval_reconstruction`, with amplitude SSIM and PSNR as primary paper metrics, after the metric subset and crop contract below is resolved

The first implementation task must reconcile the paper-side provenance mismatch:

- `/home/ollie/Documents/ptychopinnpaper2/data/README.md` names `.artifacts/sim_lines_4x_metrics_2026-01-27/gs2_ideal_nll/metrics.json`.
- The current repo artifact directory contains `.artifacts/sim_lines_4x_metrics_2026-01-27/gs2_ideal/metrics.json`, not `gs2_ideal_nll`.
- `/home/ollie/Documents/ptychopinnpaper2/data/sim_lines_4x_metrics.json` already reports the paper table values, including `gs1_custom` amplitude SSIM `0.9044216561120993` and PSNR `68.8864772792175`.

Do not regenerate or extend the paper table until this provenance mismatch is resolved or explicitly annotated.

### Table 2 Metric Subset and Crop Contract Gate

The Table 2 metric contract is not pinned by recipe constants alone. Before any HIO/ER metric or same-split PtychoPINN rerun metric is inspected, the implementation plan must decide whether the paper-side metric notes are authoritative or stale, and then write a metric-contract manifest.

Known paper-side notes to reconcile:

- `/home/ollie/Documents/ptychopinnpaper2/data/sim_lines_4x_metrics.json` says the table used `align_for_evaluation` with scan coordinates from `global_offsets` and `stitch_patch_size=20`.
- `/home/ollie/Documents/ptychopinnpaper2/tables/scripts/generate_sim_lines_4x_metrics.py` and `/home/ollie/Documents/ptychopinnpaper2/tables/sim_lines_4x_metrics.tex` say test evaluation used a random subsample with `nsamples=1000` and `seed=7`.
- The current `ptycho.workflows.grid_lines_workflow.run_tf_comparison_workflow` path stitches with `stitch_predictions(...)` and calls `eval_reconstruction(...)` directly, so the plan must verify whether the paper notes came from another table-production path, an older artifact, or stale caption text.

Required metric-contract manifest fields:

- Decision for each paper note: authoritative, stale, or unresolved, with evidence path and line reference.
- Exact reconstruction-to-ground-truth preparation path: direct `stitch_predictions(...)` evaluation versus scan-coordinate `align_for_evaluation(...)`, the crop/alignment function, the `global_offsets` source and shape/checksum when used, and `stitch_patch_size` when used.
- Exact metric subset policy: full test split versus subsample. If subsampled, record `nsamples`, seed, RNG implementation, sampling population, selected indices or a checksum of the selected index array, and ordering.
- Exact `eval_reconstruction(...)` arguments and metric conventions, including `phase_align_method`, `ms_ssim_sigma`, `frc_sigma`, amplitude mean scaling, and any debug/preprocessing flags. Do not use removed no-reference single-image FRC settings or `single_frc50` / `single_frc1over7` outputs.
- A deviation field explaining whether the HIO/ER result is a Table-2-compatible comparator, a fresh same-split rerun comparator, or only historical context.

If this contract cannot be resolved before metrics are inspected, HIO/ER may only be reported as a fresh non-Table-2-compatible exploratory row or the study must pivot. The design forbids using the same generated data with a different crop, subset, or alignment policy while calling the result Table-2-compatible.

### Table 2 Data Identity Gate

The paper Table 2 `gs1_custom` value must not be treated as the same-data comparator by recipe alone. Before any HIO/ER metric is inspected, the implementation plan must choose and record one of these branches for every probe variant reported:

1. Frozen-artifact branch.
   - Locate the exact Table 2 `C_g=1` data and reconstruction artifacts used for the paper row, including the test split, probe array after transform, `YY_ground_truth`, `norm_Y_I`, and the PtychoPINN `YY_pred` or metrics source for the row being compared.
   - Write a data-identity manifest with file paths, mtimes, sizes, command/provenance pointers, and key-level checksums for `X`, `Y_I`, `Y_phi`, `YY_full`, `YY_ground_truth`, `norm_Y_I`, `probeGuess`, `coords_nominal`, `coords_true`, `coords_offsets` when present, and `YY_pred` when a reconstruction artifact is used.
   - Only this branch may compare the HIO/ER row against the old paper Table 2 value as if it used the same data.

2. Same-split rerun branch.
   - If exact Table 2 inputs cannot be located or their checksums do not match the claimed row, generate a new deterministic Table 2-compatible `C_g=1` bundle and run both PtychoPINN and HIO/ER on that same generated test split.
   - Before data generation, choose one explicit data-generation control branch and record it in the data-identity manifest:
     - Loader-compatible branch (default): reuse `simulate_grid_data(...)`, which calls `data_preprocessing.generate_data()` and `load_simulated_data(...)`. In this branch, the authoritative object/noise seeds are the existing loader's hard-coded NumPy resets: train seed `1` immediately before the train `mk_simdata(...)` call and test seed `2` immediately before the test `mk_simdata(...)` call. Do not claim an external data-generation seed controls the split.
     - Study-local seeded branch: use this only if the implementation plan deliberately wants a new data seed such as `2026041207`. It must bypass or wrap `load_simulated_data(...)` so the declared seed actually controls object/noise generation for both train and test, and it must disable memoization with `PTYCHO_DISABLE_MEMOIZE=1` or include the data seed and branch id in the dataset cache key/provenance.
   - For either data-generation control branch, disable memoization with `PTYCHO_DISABLE_MEMOIZE=1` for freshly generated bundles or record the exact cache mode, cache key/hash, cache file path, cache file checksum, and branch label if a cached result is reused. A cache-reused bundle must not be described as freshly generated.
   - Persist the generated train/test NPZs, probe-transform manifest, invocation artifacts, data-generation branch manifest, and key-level checksums.
   - Follow the same-split PtychoPINN rerun randomness contract below before model construction, training, inference, or metric inspection.
   - Compare HIO/ER only against the new same-split PtychoPINN rerun. The old paper value, including `gs1_custom` amplitude SSIM `0.9044216561120993`, may be cited only as historical context and must not be presented as the same-data comparator.
   - Default reproducibility tolerance for calling the rerun Table-2-compatible is `abs(delta amplitude SSIM) <= 0.02` and `abs(delta amplitude PSNR) <= 2.0 dB` for `gs1_custom` relative to `/home/ollie/Documents/ptychopinnpaper2/data/sim_lines_4x_metrics.json`. If the rerun falls outside this tolerance, report it as a fresh rerun or pivot; do not silently merge it into the old Table 2 row.

### Same-Split PtychoPINN Rerun Randomness Contract

The frozen-artifact branch records existing artifacts and does not introduce new model-training randomness. The same-split rerun branch must pre-register a PtychoPINN comparator seed policy before any HIO/ER or rerun metric is inspected.

Default deterministic rerun mode:

- Use the data-generation control branch recorded by the Table 2 Data Identity Gate and the primary PtychoPINN training seed `2026041211` unless the implementation plan changes them before metrics exist. In the default loader-compatible branch, record train data seed `1`, test data seed `2`, and no externally controlling `data_generation_seed`. In the study-local seeded branch, default to `data_generation_seed=2026041207` and prove that the declared seed controls the generated object/noise arrays and cache identity.
- Launch the rerun in a fresh process with `PYTHONHASHSEED=2026041211`, `TF_DETERMINISTIC_OPS=1` when supported, and a manifest entry for CUDA/cuDNN, TensorFlow, NumPy, and Python versions.
- Before study-local data generation, model construction, and training, set Python `random`, NumPy, and TensorFlow seeds. Prefer `tf.keras.utils.set_random_seed(2026041211)` plus `tf.config.experimental.enable_op_determinism()` when available, and record any unavailable determinism API or failure. For the loader-compatible data branch, record that `load_simulated_data(...)` resets NumPy to train seed `1` and test seed `2` after any external seed setting; still set the training seeds before model construction and training.
- Preserve the Table 2 training recipe unless the metric-contract gate proves a change is required: `nepochs=60`, `batch_size=16`, the same loss weights and probe transform for the condition, and the existing Keras training-order behavior. If `model.fit(..., shuffle=True)` remains in use, the manifest must state how the shuffle RNG is controlled; if it cannot be controlled, deterministic rerun mode is invalid.
- Persist initial model weight checksums after construction, final model/checkpoint checksums, training history, invocation artifacts, environment variables, seed values, and the metric-contract manifest checksum.
- Optionally repeat the primary seed once as a determinism check. If repeated execution with the same seed is not bitwise stable or not metric-stable within a pre-registered tolerance, fall back to stochastic repeated-rerun mode or pivot.

Stochastic repeated-rerun mode:

- Use this mode when TensorFlow/Keras/GPU determinism cannot be verified or when training-order randomness cannot be controlled without changing the Table 2 recipe.
- Run at least three fixed training seeds `[2026041211, 2026041212, 2026041213]` on the same generated data bundle and metric contract.
- Report every seed separately and use a pre-registered aggregate, defaulting to median amplitude SSIM and median amplitude PSNR with the full min-to-max range. Do not compare HIO/ER only against the best PtychoPINN seed.
- If three reruns are not feasible, report the same-split PtychoPINN comparator as insufficiently controlled and pivot or label the result as exploratory.

### Single-Shot CDI Baseline Contract

The non-ML baseline must use one diffraction pattern per reconstruction problem. It must not use multi-position ptychographic overlap, ptychographic position updates, or a Tike/PtyChi reconstruction labeled as CDI.

Allowed baseline classes, in priority order:

1. **PyNX.cdi** (primary): HIO, ER, RAAR, or a standard HIO→ER operator chain applied to a single diffraction amplitude with a strict real-space support derived from the known probe. This is the default reviewer-facing solver.
2. **Repo-local known-probe object-domain HIO/ER** (`known_probe_object_hio_er`): allowed as a separately labeled diagnostic/candidate row when the scientific question is specifically whether fixed-probe object-domain HIO/ER behaves differently from exit-wave CDI followed by probe division.
3. In-repo support-constrained exit-wave HIO/ER on a single diffraction amplitude, **only** as the fallback when PyNX is rejected by the solver-discovery gate with a recorded reason.
4. Other external single-frame CDI phase-retrieval packages (e.g. PyPhaseRetrieve, PhaseRetrieval.jl via interop) only after provenance and license checks, and only if PyNX is unavailable.
5. ADMM phase retrieval only if a suitable single-frame implementation is discovered and does not expand the study.

Rejected as reviewer-facing unless explicitly framed as out of scope:

- Multi-frame ptychography solvers, including PtyChi/Tike RPIE/DM/LSQML/PIE.
- Supervised or learned baselines.
- Any baseline that consumes the full scan or overlap groups while being labeled `single-shot`.

### Physics and Normalization

The existing synthetic forward model is far-field diffraction of the probe-illuminated object patch:

```text
psi = O_norm * P
X_norm ~= sqrt(fftshift(|FFT(psi)|^2 / (N * N))) with Poisson noise scaled by intensity_scale
```

The HIO/ER implementation must operate in the same normalized units as the existing grid-lines workflow:

- Use stored normalized diffraction amplitude `X` as the Fourier magnitude constraint.
- Do not multiply `X` by `intensity_scale` for the reconstruction loop unless producing diagnostic photon-count outputs.
- Recover normalized object patches `O_norm` so that existing `stitch_predictions(..., norm_Y_I, part=...)` can restore the Table 2 object scale.
- Apply `norm_Y_I` only at the final stitching/evaluation boundary, mirroring the current workflow.

This prevents mixing the repository's separate physics normalization, statistical normalization, and display/evaluation scaling systems.

### CDI Ambiguity and Evaluation Policy

Single-frame Fourier-magnitude CDI has global phase, shift, and conjugate-inversion/twin-image ambiguities. The reviewer-facing baseline must pre-register how those ambiguities are handled before any metrics are inspected.

Main reported HIO/ER row:

- Anchor each patch only by the known probe support, the Table 2 FFT convention, and the nominal `C_g=1` patch position used by the grid-lines workflow.
- Do not run per-patch or per-object ground-truth alignment, translation search, conjugate-inversion selection, twin-image selection, phase-sign selection, or object-dependent recentering before stitching.
- Do not choose between symmetry-equivalent HIO/ER outputs using amplitude SSIM, PSNR, phase metrics, or visual agreement with the ground truth.
- Allow only deterministic coordinate-convention corrections that are fixed before the benchmark run and validated by a ground-truth-independent forward/amplitude self-consistency test, for example correcting an `fftshift` convention mismatch in the implementation.
- Use `eval_reconstruction` consistently with the Table 2 metric convention after direct stitching. Its existing phase-plane or mean phase preprocessing and amplitude mean scaling are metric conventions, not a license to add CDI-specific oracle alignment.
- Report ambiguity failures explicitly. If the baseline appears to recover a valid twin, shifted, or globally phase-rotated CDI solution that is penalized by the direct Table 2 metric path, label that as an ambiguity outcome instead of silently realigning it for the main row.

Oracle diagnostics are permitted only as separate, clearly labeled sensitivity checks. They may include ground-truth shift/twin/orientation alignment to understand whether a poor main-row score is an ambiguity artifact, but those diagnostics must not replace the main reviewer-facing row unless the manuscript response explicitly pivots to discussing why a directly comparable Table 2 row is not defensible.

### Support and Probe Priors

The baseline is a known-probe, support-constrained CDI reference. The support policy must be explicit because it materially affects scientific interpretation.

Default support policy:

- For the PyNX exit-wave path, reconstruct the exit wave `psi = O_norm * P`.
- For the known-probe object-domain path, reconstruct `O_norm` directly but still apply the known probe inside the forward projection `P * O_norm`.
- Define a strict support mask from the canonical Table 2 probe amplitude footprint after the same probe transform used for the condition.
- Use `support_threshold=0.05` as the primary pre-registered support: `abs(P) >= 0.05 * max(abs(P))`.
- Run the fixed sensitivity grid `support_threshold in [0.01, 0.05, 0.10]` for the first full attempt when runtime permits; report all completed threshold rows and keep `0.05` as the primary row regardless of ground-truth metrics.
- If runtime or numerical instability prevents the full sensitivity grid, still run the primary `0.05` row and document which sensitivity thresholds were skipped or failed.
- Do not select a threshold by amplitude SSIM, PSNR, phase metrics, or visual agreement with the ground truth. A ground-truth-free rejection is allowed only for invalid supports, such as empty masks, full-frame masks, NaNs, or manifest-documented numerical failure.
- Recover object patches only inside the support by `O_norm = psi / P_safe`.
- Set or mark outside-support object values by a documented policy; default to zero outside support for the baseline reconstruction artifact.
- Record `support_threshold`, support pixel count/fraction, `probe_division_epsilon`, threshold-grid status, rejected/failed threshold reasons, and whether the support is an oracle known-probe prior.

Do not use the constant zero phase of the synthetic line-pattern object as an undisclosed oracle prior. If an amplitude-only or zero-phase prior is tested, it must be labeled as an oracle diagnostic and kept out of the main classical-baseline row unless the reviewer response deliberately discusses that prior.

### Randomness and Restarts

HIO/ER output must not depend on unrecorded random state.

Required:

- Set deterministic restart seeds.
- Use at least three random-phase restarts per condition in the first full attempt.
- Use restart base seeds `[2026041201, 2026041202, 2026041203]` for the primary row and derive any per-patch RNG streams deterministically from condition id, patch index, and restart index.
- Select the reported restart by the lowest final Fourier-amplitude residual after ER cleanup, breaking ties by lower restart seed.
- Persist per-restart residual curves and metrics, not only the chosen-row metrics.

### HIO/ER Schedule and Residual Contract

The main reviewer-facing row must use this pre-registered primary schedule unless a later design/plan update changes it before any HIO/ER metric output is inspected:

- Initialize each restart from the stored normalized Fourier magnitude `X` and a random phase drawn from the deterministic restart stream. Do not initialize from ground-truth object phase or amplitude.
- Use Fourier projection with the same `fftshift(fft2(...)) / sqrt(N * N)` amplitude convention as the self-consistency check for the existing grid-lines forward model.
- Run `1000` HIO iterations with `beta=0.9`.
- Follow with `200` ER cleanup iterations.
- Use a fixed iteration budget for the primary row. Do not early-stop or extend based on amplitude SSIM, PSNR, phase metrics, images, or ground-truth agreement.
- Compute and record the Fourier-amplitude residual as `norm(abs(F(psi)) - X) / max(norm(X), 1e-12)` using the same normalized Fourier convention as the projection.
- Record residuals every `10` HIO iterations, every `10` ER cleanup iterations, and at the final state.
- Use `probe_division_epsilon = 1e-6 * max(abs(P))` when converting `psi` to `O_norm`; if `max(abs(P)) == 0`, treat the condition as an invalid support/probe failure.
- The only ground-truth-free reasons to reject the primary schedule are invalid support/probe masks, NaNs, non-finite residuals, or a manifest-documented implementation self-consistency failure. A different beta, iteration count, cleanup length, epsilon, or stopping rule becomes a separately labeled diagnostic unless the design/plan is updated before HIO/ER metrics are inspected.

### Output and Artifact Contract

Successful benchmark artifacts:

- Study script under `scripts/reconstruction/` or `scripts/studies/`.
- Unit tests for support mask construction, Fourier magnitude projection, HIO/ER update behavior, restart selection, and manifest writing.
- Solver/dependency manifest with package name, version, license, install command, import/API entry point, and acceptance/rejection reason.
- Table 2 data-identity manifest identifying either the frozen-artifact branch or the same-split rerun branch, with checksums and comparator policy.
- Table 2 metric-contract manifest resolving the crop/alignment path, metric subset or subsample indices, `eval_reconstruction(...)` arguments, and whether paper-side notes are authoritative or stale.
- Same-split PtychoPINN randomness manifest when the rerun branch is used, including seed policy, determinism settings, model/checkpoint checksums, training history, and stochastic-repeat aggregate policy when needed.
- Machine-readable run manifest under `.artifacts/revision_studies/non_ml_single_shot_cdi_benchmark/<run_id>/`.
- Per-condition metrics JSON containing all HIO/ER hyperparameters, ambiguity policy, support policy, support-threshold grid status, data-identity branch, metric-contract id, seeds, residuals, timing, and metrics.
- Reconstruction NPZs for the selected HIO/ER rows, with the same `YY_pred` style used by existing `save_recon_artifact`.
- Paper-side data JSON update only after the benchmark is accepted as comparable.
- Regenerated Table 2 or a supplementary result only after paper-side table-generation checks pass.

Abandoned/pivot artifacts:

- A short attempt note with commands, exact failure mode, and pivot reason.
- Rejected external-solver table, if solver discovery ran.
- Proposed manuscript/reviewer-response text that narrows the single-shot CDI comparison claim.
- Changelog and checklist updates only after the pivot text is approved for the paper revision.

## Implementation Shape

Prefer one narrow study entry point rather than broad workflow refactoring.

Proposed file:

- `scripts/reconstruction/hio_cdi_benchmark.py`

Acceptable alternative:

- `scripts/studies/hio_cdi_benchmark.py`, if the plan decides this belongs with revision-study scripts rather than general reconstruction scripts.

Study-local helpers should be kept in the same script or a small sibling module until the benchmark has proven value. Candidate local responsibilities:

- `discover_solvers(...)` — PyNX-first solver-discovery gate; records install/version/license/import manifest.
- `build_table2_condition(...)`
- `make_probe_support(...)`
- `forward_amplitude(...)` — used only for the ground-truth-independent self-consistency check against PyNX's internal convention.
- `run_pynx_cdi_restart(...)` — thin wrapper constructing `pynx.cdi.CDI` from `(X, support, initial_phase_seed)` and running the pre-registered HIO→ER operator chain. Owns the PyNX-specific argument mapping.
- `run_restarts(...)` — deterministic restart loop; delegates the per-restart solve to `run_pynx_cdi_restart`.
- `known_probe_forward_amplitude(...)`, `project_known_probe_fourier_magnitude(...)`, `known_probe_hio_update(...)`, `known_probe_er_cleanup(...)`, and `run_known_probe_restarts(...)` — repo-local fixed-probe object-domain diagnostic/candidate solver path. These helpers must be selected only by an explicit solver mode such as `--solver known_probe_object_hio_er` and must write distinct manifests/row labels.
- `project_fourier_magnitude(...)`, `hio_update(...)`, `er_cleanup(...)` — **fallback-only**. Present only when the solver-discovery gate rejects PyNX. The fallback path must additionally pass a PyNX-vs-in-repo cross-check on a small synthetic case (if PyNX is importable but rejected for non-numerical reasons) so the fallback is not silently numerically divergent from the standard reference.
- `stitch_hio_patches(...)`
- `write_benchmark_manifest(...)` — records which solver (`pynx.cdi` or `inrepo_hio`) was used, version, and the solver-discovery gate outcome.

Reuse existing code paths:

- `ptycho.workflows.grid_lines_workflow.GridLinesConfig`
- `load_probe_guess`
- `load_ideal_disk_probe`
- `normalize_probe_transform_pipeline`
- `apply_probe_transform_pipeline`
- `apply_probe_mask`
- `simulate_grid_data`
- `configure_legacy_params`
- `stitch_predictions`
- `save_recon_artifact`
- `ptycho.evaluation.eval_reconstruction`
- `scripts.studies.invocation_logging.write_invocation_artifacts`

Same-split data-generation caveat:

- `simulate_grid_data(...)` is compatible with the default loader-compatible branch because it preserves the existing Table 2-style `load_simulated_data(...)` path and its hard-coded train/test NumPy seeds `1` and `2`.
- Do not call `simulate_grid_data(...)` unmodified for the study-local seeded branch unless the implementation plan first changes or wraps the seed reset and memoization behavior so the declared data seed controls both generated arrays and cache/provenance identity.

Do not modify stable core modules during this study unless a later approved plan explicitly authorizes it:

- `ptycho/model.py`
- `ptycho/diffsim.py`
- `ptycho/tf_helper.py`

## Internal Refactoring and Debt Paydown

No broad internal refactoring is required before feature work.

The design intentionally treats this as a revision-study script that **wraps** PyNX.cdi rather than extending shared workflow code because:

- No existing in-repo single-frame CDI HIO/ER baseline was found, and rather than write one, the solver itself is delegated to PyNX.cdi.
- Existing in-repo classical baselines are ptychographic and multi-frame, so PyNX is the lowest-friction path to a reviewer-credible `single-shot` comparator.
- The I/O glue, support-policy, and restart-policy choices are study-specific scientific decisions, not yet reusable APIs.
- Adding shared workflow API surface — or re-implementing a standard algorithm the reviewer can already cite — before a successful benchmark would create avoidable debt and invite reviewer skepticism about the comparator's fidelity to the method they named.

Required pre-feature gates, not broad refactors:

- Verify `from ptycho.evaluation import eval_reconstruction` works in the target environment. If it fails because optional FRC code is unavailable, the implementation plan must either make a minimal import compatibility fix or block before benchmark implementation.
- Reconcile the paper-side Table 2 provenance mismatch before any paper table regeneration.
- Complete the Table 2 data-identity branch selection and write its manifest before any HIO/ER metric is inspected.
- Complete the Table 2 metric subset/crop contract decision and write its manifest before any HIO/ER or same-split PtychoPINN rerun metric is inspected.
- If using the same-split rerun branch, choose and manifest the data-generation control branch (default loader-compatible train seed `1`/test seed `2` versus study-local seeded branch such as `2026041207`) before generating the bundle.
- If using the same-split rerun branch, freeze the PtychoPINN seed/determinism policy and write the randomness manifest before model construction or training.
- Run the solver-discovery gate **primarily** to validate PyNX.cdi importability, version, license (CeCILL-B), and API compatibility with the Table-2 normalization convention. Record the concrete install/import commands actually used in the `ptycho311` environment. Only if PyNX cannot be imported or cannot be made numerically consistent with the Table-2 forward model may the fallback in-repo HIO/ER path be adopted, and only with the rejection reason recorded in the solver/dependency manifest.
- Confirm that `stitch_predictions(..., part="complex")` or an equivalent study-local stitch path produces the same scale and crop convention as the existing `amp`/`phase` stitching path for `C_g=1`.

Promotion rule:

- Promote HIO/ER helpers into shared modules only after the study produces useful reviewer-facing evidence and a follow-up design identifies at least one additional consumer.

## Non-Goals

- Do not build a general CDI package.
- Do not implement a full ADMM framework unless solver discovery finds a clean, bounded integration.
- Do not use ptychographic multi-position solvers as the requested single-shot CDI baseline.
- Do not change PtychoPINN training, architecture, probe estimation, or core physics.
- Do not add trainable-probe or joint probe/position-refinement variants.
- Do not rewrite `eval_reconstruction` or Table 2 generation unless a narrow compatibility fix is required and approved in the implementation plan.
- Do not claim the result proves superiority or inferiority against all classical CDI methods.
- Do not edit the manuscript, reviewer response, changelog, or checklist during design-only work.

## Sequencing Constraints

1. Preserve output contract and state.
   - Keep the design pointer at `state/revision-study-priority-stack/non-ml-single-shot-cdi-benchmark/non-ml-single-shot-cdi-benchmark/design-phase/design_path.txt`.
   - The pointer must contain a relpath under `docs/plans`.

2. Solver discovery (PyNX-first).
   - **Primary check**: attempt `import pynx.cdi` in the `ptycho311` environment (or a minimal bounded install on top of it: `pip install pynx` or the ESRF-provided conda/tarball path). Confirm version, license (CeCILL-B), and install command in the solver manifest.
   - Verify that `pynx.cdi.CDI` accepts a user-supplied normalized Fourier magnitude and a user-supplied support mask, and exposes HIO/ER operators with configurable `beta`, iteration count, and RNG-controlled initial phase. Verify the FFT normalization convention matches (or can be made to match) the Table-2 `fftshift(fft2(...)) / sqrt(N*N)` amplitude convention via a ground-truth-independent self-consistency test.
   - Only if the PyNX check fails, search repo and environment for other single-frame CDI phase-retrieval packages or scripts, record candidates/rejections with version/license/install provenance, and document the concrete PyNX rejection reason.
   - Reject multi-frame ptychography for the reviewer-facing row regardless of which solver is chosen.

3. Table 2 data reproduction preflight.
   - Choose the frozen-artifact branch or same-split rerun branch from the Table 2 Data Identity Gate.
   - If using the frozen-artifact branch, load the exact `C_g=1` idealized and experimental-probe test splits and reconstruction/metrics artifacts; write the key-level checksum manifest before HIO/ER runs.
   - Before either branch reports metrics, reconcile the Table 2 metric subset/crop contract, including whether `align_for_evaluation` with `global_offsets` and `stitch_patch_size=20` and the `nsamples=1000`, `seed=7` caption are authoritative or stale.
   - If using the same-split rerun branch, choose the data-generation control branch before generation: default loader-compatible `simulate_grid_data(...)` with train seed `1` and test seed `2`, or a study-local seeded path where the declared data seed, default `2026041207`, actually controls generation and cache/provenance identity. Generate a deterministic `C_g=1` data bundle, freeze and record the PtychoPINN seed/determinism policy, rerun the matching PtychoPINN baseline on that bundle, and compare HIO/ER only to that same-split baseline or its pre-registered stochastic aggregate.
   - Verify `X`, probe transform, `YY_ground_truth`, `norm_Y_I`, and test split match the chosen comparator contract or document any deviation.
   - Resolve the `gs2_ideal_nll` versus `gs2_ideal` provenance mismatch before touching paper outputs.
   - Freeze the metric subset/crop contract, ambiguity policy, `support_threshold=0.05` primary row, and HIO/ER schedule before inspecting HIO/ER metrics.

4. HIO/ER smoke.
   - Run a small subset, for example 8 to 16 test frames.
   - Confirm no NaNs, residuals are recorded, the `0.05` support is nonempty and not full-frame, and reconstructed patches can be stitched/evaluated without ground-truth shift/twin/orientation alignment.
   - Confirm a self-consistency check: projecting a known ground-truth exit wave through the HIO forward/amplitude code uses the same normalization convention as the simulated `X`.
   - Confirm any deterministic coordinate-convention correction is fixed by self-consistency checks, not by post-hoc metric improvement.

5. Full benchmark attempt.
   - Run `C_g=1` experimental/custom probe first because it is the reviewer-critical Table 2 row.
   - Run the fixed primary HIO/ER schedule with `support_threshold=0.05` first, then the fixed sensitivity rows `0.01` and `0.10` when runtime permits.
   - Run idealized probe only if the experimental row is stable or if a negative/failed result needs a controlled comparison.
   - Use exact launched PID tracking for long runs and do not duplicate a run writing to the same output root.

6. Reporting gate.
   - If comparable, update paper data JSON/table/text in a separate implementation phase.
   - If not comparable, write the pivot note and draft claim-narrowing text instead.

## Pivot Criteria

Pivot to a scoped textual response if any of these occur:

- Neither PyNX.cdi nor the in-repo HIO/ER fallback can be made numerically stable and self-consistent with the Table-2 forward/normalization convention after the bounded smoke/full attempt.
- The HIO/ER implementation cannot be cleanly mapped to the Table 2 stitched-test-split convention.
- The baseline requires fragile installation, unclear licensing, or non-reproducible build steps.
- The only available classical solvers are multi-frame ptychographic methods.
- Known-probe/support priors dominate the result so strongly that a paper-table row would mislead without extended discussion.
- The direct support-anchored ambiguity policy makes the HIO/ER row scientifically misleading, and only oracle shift/twin/alignment diagnostics produce a useful comparison.
- The HIO/ER baseline outperforms PtychoPINN under strong oracle support/probe priors and the manuscript claim must be softened rather than cherry-picked.
- The HIO/ER baseline is much weaker but only because of a disadvantaged support or initialization policy; in that case, report it as one reference baseline, not a definitive classical comparison.

## Verification

Pre-publication checks:

- `python` resolves to the intended environment; for long commands, use tmux with `ptycho311` activated or PATH configured to that environment.
- `pytest` passes for the new HIO/ER unit tests.
- A smoke benchmark writes a manifest, per-restart residuals, metrics JSON, and reconstruction NPZs.
- Required output artifacts exist and are freshly written after the tracked PID exits with code 0.
- A Table 2 data-identity manifest exists, identifies the frozen-artifact or same-split rerun branch, and contains checksums for the comparator data and reconstruction artifacts.
- A Table 2 metric-contract manifest exists and records the authoritative crop/alignment path, metric subset or subsample indices, `eval_reconstruction(...)` arguments, and any stale paper-side notes.
- If the same-split rerun branch is used, the data-identity manifest records the data-generation control branch, train/test data seeds or study-local data seed, memoization/cache policy, and generated-bundle checksums.
- If the same-split rerun branch is used, a PtychoPINN randomness manifest exists and records deterministic seed evidence or the required multi-seed stochastic aggregate.
- The new metrics use the same split, photon dose, probe variants, crop/subset policy, support disclosure, ambiguity policy, and metric convention as Table 2, or deviations are explicitly stated.
- The primary HIO/ER row uses the fixed `beta=0.9`, `1000` HIO iterations, `200` ER cleanup iterations, `probe_division_epsilon = 1e-6 * max(abs(P))`, and final-residual restart selection.
- The primary row uses `support_threshold=0.05` and direct support-anchored stitching/evaluation without ground-truth shift/twin/orientation alignment.
- Any `support_threshold in [0.01, 0.10]` sensitivity rows and any oracle alignment diagnostics are labeled separately and cannot replace the primary row by post-hoc metric selection.
- Any external dependency is importable from a clean command and records version/license/install provenance.
- No ptychographic multi-frame solver is labeled as single-shot CDI.
- Paper table generation succeeds if paper outputs are changed.
- The compiled paper is inspected for Table 2 formatting if the table or caption changes.
- The reviewer checklist and changelog are updated only when reviewer-facing paper changes are made.

# Technical notes

PyNX is the primary target; discovery should confirm it rather than search broadly. References:
- PyNX project page: https://ftp.esrf.fr/pub/scisoft/PyNX/doc/ (ESRF, CeCILL-B licensed).
- `pynx.cdi` module exposes `CDI` objects, support projection, HIO/ER/RAAR operators, and restart handling compatible with 2D arrays and user-supplied supports.
- Installation in `ptycho311`: prefer `pip install pynx` (pulls the pure-Python + OpenCL pieces); fall back to ESRF tarball only if the pip path fails. Record the exact working install command in the solver manifest.

Only if PyNX is unavailable should the discovery pass search GitHub/PyPI more broadly for HIO/ADMM implementations; in that case, record each candidate's name, URL, commit/version, license, install command, and reason for rejection or acceptance.

## Review Revision Notes

- `DESIGN-H1`: addressed by adding the CDI ambiguity policy, requiring direct support-anchored stitching/evaluation for the main row, and restricting ground-truth shift/twin/orientation alignment to labeled oracle diagnostics.
- `DESIGN-M1`: addressed by pre-registering `support_threshold=0.05` as the primary row, defining the fixed sensitivity grid `[0.01, 0.05, 0.10]`, and forbidding post-hoc threshold selection by ground-truth metrics.
- `DESIGN-H2`: addressed by adding the Table 2 Data Identity Gate. The design now requires either exact frozen Table 2 artifacts with key-level checksums, or a deterministic same-split rerun with explicit tolerance and a prohibition on comparing HIO/ER to the old `0.904` row as if it were the same data.
- `DESIGN-M2`: addressed by pre-registering the primary HIO/ER beta, iteration budget, ER cleanup length, residual definition, restart-selection rule, and `probe_division_epsilon`, with only ground-truth-free failure criteria allowed.
- `DESIGN-H3`: addressed by adding the Table 2 Metric Subset and Crop Contract Gate, which requires resolving the paper JSON `global_offsets`/`stitch_patch_size=20` note and the table-caption `nsamples=1000`, `seed=7` note before HIO/ER or same-split rerun metrics are inspected.
- `DESIGN-M3`: addressed by adding the Same-Split PtychoPINN Rerun Randomness Contract, including deterministic seed defaults, TensorFlow/Keras determinism requirements, manifests, and a three-seed stochastic fallback if determinism cannot be verified.
- `DESIGN-M4`: addressed by adding an explicit same-split data-generation control branch. The default branch now preserves and documents the reused loader's hard-coded train/test NumPy seeds `1` and `2`; the alternate study-local branch may use a declared seed such as `2026041207` only if it wraps or bypasses `load_simulated_data(...)` and records memoization/cache identity so the seed actually controls generation.
- 2026-04-13 known-probe update: added the explicit `known_probe_object_hio_er` secondary object-domain path to reflect the later implementation and diagnostic work. This path is distinct from the PyNX exit-wave CDI row and from the old fallback-only `study_local_hio_er` recurrence; it must remain separately labeled until outcome review accepts or rejects it as reviewer-facing evidence.

## Documents Read

For this review-revision pass:

- `docs/plans/revision-studies/non-ml-single-shot-cdi-benchmark-design-seed.md`
- `.artifacts/review/revision-studies/non-ml-single-shot-cdi-benchmark-design-review.json`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/DATA_GENERATION_GUIDE.md`
- `docs/DATA_NORMALIZATION_GUIDE.md`
- `specs/data_contracts.md`
- `ptycho/data_preprocessing.py`
- `ptycho/diffsim.py`
- `ptycho/workflows/grid_lines_workflow.py`
- `ptycho/misc.py`
- `state/revision-study-priority-stack/non-ml-single-shot-cdi-benchmark/current-item-inputs.json`
- `state/revision-study-priority-stack/non-ml-single-shot-cdi-benchmark/non-ml-single-shot-cdi-benchmark/design-phase/design_path.txt`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
