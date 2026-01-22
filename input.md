Summary: Capture photon_grid dose baseline facts and draft the maintainer-ready D0 parity response.
Focus: seed — Inbox monitoring and response (checklist S1–S2)
Branch: dose_experiments
Mapped tests: pytest tests/test_generate_data.py::test_placeholder -q
Artifacts: plans/active/seed/reports/2026-01-22T024002Z/

Do Now:
- Implement: plans/active/seed/bin/dose_baseline_snapshot.py::main (dataset/probe snapshot) and inbox/response_prepare_d0_response.md (maintainer reply). Bundled S1–S2 coverage.
- Test: pytest tests/test_generate_data.py::test_placeholder -q | tee $ARTIFACT_DIR/pytest_seed.log
- Artifacts: $ARTIFACT_DIR/{dose_baseline_summary.json,dose_baseline_summary.md,inbox_response.md,pytest_seed.log}

How-To Map:
1. export ARTIFACT_DIR=plans/active/seed/reports/2026-01-22T024002Z and mkdir -p "$ARTIFACT_DIR".
2. Create plans/active/seed/bin/dose_baseline_snapshot.py:
   • argparse inputs: --dataset-root (default photon_grid_study_20250826_152459), --baseline-params (default photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/params.dill), --scenario-id ("PGRID-20250826-P1E5-T1024"), --output "$ARTIFACT_DIR".
   • Walk dataset-root for data_p1e*.npz, load with numpy, capture count, per-array shape/dtype, and hashlib.sha256(filepath) (matching sums already spot-checked: data_p1e5 sha256 01007daf8afc67aad3ad037e93077ba8bfb28b58d07e17a1e539bd202ffa0d95, etc.).
   • Load baseline params via dill; extract keys we need (N, gridsize, nimgs_train/test, mae/ms_ssim/psnr/frc50 tuples, mae/nll weights, intensity_scale, label, output_prefix). Cast tensors to floats so JSON serializes.
   • Emit JSON (dose_baseline_summary.json) describing scenario metadata plus dataset table and metrics. Emit Markdown (dose_baseline_summary.md) summarizing bullet points plus table referencing JSON.
3. Run the script:
   python plans/active/seed/bin/dose_baseline_snapshot.py \
     --dataset-root photon_grid_study_20250826_152459 \
     --baseline-params photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/params.dill \
     --scenario-id PGRID-20250826-P1E5-T1024 \
     --output "$ARTIFACT_DIR"
4. Draft inbox/response_prepare_d0_response.md covering maintainer checklist (sections 1–7). Source facts from the Markdown/JSON snapshot plus repo files:
   • Baseline selection: name scenario PGRID-20250826-P1E5-T1024 referencing photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1 and explain why (complete artifacts, stable metrics: ms_ssim≈0.925/0.921, psnr≈71.3 dB amplitude).
   • Dataset parity: enumerate data_p1e{3,4,5,6,7,8,9}.npz paths (5000 diffraction patterns each, 64×64 diff3d, includes probeGuess/objectGuess). Include sha256 from script output.
   • Probe provenance: cite dataset-embedded probeGuess (64×64 complex) produced by lines-based simulator (see notebooks/dose.py lines 1–120), mention probe mask on, default_probe_scale=0.7, probe.trainable=False, intensity_scale.trainable=True.
   • Config snapshot: pull key params (N=64, gridsize=1 for stored run, nimgs_train=9, nimgs_test=3, loss weights nll-only). Note that dataset names encode actual photon dose (e.g., data_p1e5 ⇒ 1e5 photons) even though params.dill retains legacy nphotons=1e9.
   • Commands executed: describe how dose_dependence.ipynb issues `python - <<'PY' ... dose.run_experiment_with_photons([1e9,1e7,1e5,1e4], loss_fn='nll')` for simulation, followed by `python -m ptycho.train --train_data_file photon_grid_study_20250826_152459/data_p1e5.npz --output_dir photon_grid_study_20250826_152459/results_p1e5/train_1024/...` and `python -m ptycho.inference --model_path .../pinn_run/wts.h5.zip --test_data ...` in the dose_experiments branch root. Call out any overrides (batch_size=16, nepochs=50, intensity_scale.trainable=True, probe frozen).
   • Artifacts available: list dataset root, baseline_run checkpoint (baseline_model.h5, recon.dill) and pinn_run weights, plus pointer to summary JSON/MD under $ARTIFACT_DIR.
   • Preferred handoff: tell maintainer to drop additional bundles under plans/active/seed/reports/ or external storage, note expected size (~0.8 GB for all NPZs) so they can plan.
5. Run pytest tests/test_generate_data.py::test_placeholder -q | tee "$ARTIFACT_DIR"/pytest_seed.log.

Pitfalls To Avoid:
- Read the large NPZ files in streaming fashion; don’t modify or relocate photon_grid_study_* data.
- Loading params.dill pulls TensorFlow; ignore GPU warnings but fail fast on actual exceptions.
- Keep new scripts/docs under plans/active/seed or inbox; no changes in ptycho/* production modules.
- Reuse ASCII; avoid notebooks or Jupyter outputs.
- Do not delete the maintainer request file; add a new response next to it.
- No environment/package installs; if imports fail, capture the exact error and stop.
- Scripts should not write temp data outside $ARTIFACT_DIR.
- When quoting commands in inbox response, ensure paths are relative to repo root.
- Keep pytest scope limited to the single placeholder test to avoid GPU workloads.

If Blocked:
- If any photon_grid_study data/params files are missing, capture `ls -R photon_grid_study_20250826_152459 | head` output into $ARTIFACT_DIR/missing_data.log, note the missing path in fix_plan Attempts + galph_memory, and halt instead of guessing values.
- If dill load crashes, log the traceback to $ARTIFACT_DIR/dill_error.log and request maintainer guidance via inbox.

Findings Applied: No relevant findings in docs/findings.md (file absent in this branch).

Pointers:
- fix_plan.md:1 — status + current focus summary.
- plans/active/seed/implementation.md:1 — checklist S1–S2 requirements.
- inbox/README_prepare_d0_response.md:1 — maintainer request rubric.
- notebooks/dose.py:1 — canonical config changes for dose_experiments workflow.
- photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/params.dill — baseline metrics snapshot.

Next Up: If time remains, capture gs2 variant (photon_grid_study_20250826_152459/data_p1e9) for future parity logging once maintainer confirms scope.

Mapped Tests Guardrail: pytest tests/test_generate_data.py::test_placeholder -q already collects/imports quickly; no new selectors required. Hard Gate: selector unchanged—if it fails, stop and record the failure signature instead of downgrading tests.
