**Summary**: Finish the A1b manual override by stabilizing the legacy `dose_experiments` runner (compatibility shim) so the simulate→train→infer flow completes and produces archived artifacts.
**Focus**: DEBUG-SIM-LINES-DOSE-001 — A1b dose_experiments ground-truth run
**Branch**: paper
**Mapped tests**: none — evidence-only
**Artifacts**: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T092411Z/

**Do Now**
- Implement: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_dose_stage.py::main — extend the compatibility runner to clamp legacy params (neighbor_count/group_count) and cap `--nimages` to a GPU-safe value (≤512) before invoking the `/home/ollie/Documents/PtychoPINN` scripts; keep the non-production shim under `bin/tfa_stub/` untouched.
- Run: `PYTHONNOUSERSITE=1 DOSE_REAL_REPO=/home/ollie/Documents/PtychoPINN AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_dose_stage.py simulation <abs_input_npz> <.artifacts/.../simulation>` (use `/home/ollie/Documents/PtychoPINN/notebooks/{train,test}_data.npz`, persist logs in `reports/2026-01-20T092411Z/`), then invoke the same wrapper for `training` and `inference` so we have simulate→train→infer outputs plus archived CLI logs.
- Archive: copy the compatibility logs plus any runnable artifacts (`stats.json`, PNGs, model outputs) into `.artifacts/DEBUG-SIM-LINES-DOSE-001/2026-01-20T092411Z/` and summarize the outcome in `reports/2026-01-20T092411Z/summary.md`.

**How-To Map**
1. Edit `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_dose_stage.py` — before calling `runpy`, set `params.cfg['neighbor_count']=min(params.cfg.get('neighbor_count',5), nimages)` and clamp `nimages` to ≤512 to avoid GPU OOM; keep using the stub `tensorflow_addons` + components shim under `bin/tfa_stub/`.
2. Simulation: `PYTHONNOUSERSITE=1 DOSE_REAL_REPO=/home/ollie/Documents/PtychoPINN python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_dose_stage.py simulation /home/ollie/Documents/PtychoPINN/notebooks/train_data.npz .artifacts/DEBUG-SIM-LINES-DOSE-001/2026-01-20T092411Z/simulation --nimages 512 --seed 42 --nepochs 60 --output_prefix dose_experiments --data_source lines --gridsize 2 --intensity_scale_trainable --probe_scale 4 --train_data_file_path /home/ollie/Documents/PtychoPINN/notebooks/train_data.npz --test_data_file_path /home/ollie/Documents/PtychoPINN/notebooks/test_data.npz > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T092411Z/simulation.log 2>&1`.
3. Training: same wrapper with `training` stage and the desired output dir (`.artifacts/.../training`), capturing stdout/stderr in `reports/.../training.log`.
4. Inference: run `run_dose_stage.py inference --model_prefix <training output> --test_data /home/ollie/Documents/PtychoPINN/notebooks/test_data.npz --output_path .artifacts/.../inference` and store the log at `reports/.../inference.log`.
5. Update `reports/2026-01-20T092411Z/summary.md` with a short outcome plus links to the new evidence; keep the stub files under version control for traceability.

**Pitfalls To Avoid**
1. Always run wrapper commands with `PYTHONNOUSERSITE=1` and `DOSE_REAL_REPO=/home/ollie/Documents/PtychoPINN` so the legacy scripts import the correct checkout (the shim assumes those paths).
2. Do not edit production modules under `ptycho/`; keep all compatibility hacks confined to `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/` (non-production artifacts per plan).
3. Clamp GPU memory — cap `--nimages` ≤512 and avoid spawning multiple runs concurrently; the RTX 3090 OOMs when the raw generator allocates 20k+ patches.
4. Adjust `params.cfg['neighbor_count']`/`group_count` before simulation so the KD-tree logic never requests more neighbors than scan positions; otherwise `group_coords` will raise `IndexError: index … out of bounds`.
5. Keep log files under `reports/2026-01-20T092411Z/` (e.g., `simulation.log`, `training.log`, `inference.log`) and copy any large outputs under `.artifacts/.../` — do not litter the repo root with legacy artifacts.
6. Ensure wrapper reuses the stub `tensorflow_addons` package by leaving `bin/tfa_stub` on `sys.path`; deleting or renaming it will reintroduce the `keras.src` import crash.
7. Respect the PyTorch/TensorFlow policy: do not upgrade/downgrade packages or install Keras globally — the compatibility runner must work inside the existing env.
8. Keep the CLI arguments absolute; the legacy scripts resolve relative paths against `/home/ollie/Documents/PtychoPINN`, not the tmp repo.
9. If the legacy scripts emit NaNs or fail mid-run, capture the entire stderr tail in the corresponding log and exit rather than silently retrying — we need actionable evidence for the fix plan.
10. Avoid touching `.artifacts/` entries created by prior loops; append new runs under the same timestamp to keep provenance.

**If Blocked**
- Save the failing command + stack trace into `reports/2026-01-20T092411Z/<stage>_error.log`, mention whether the crash is due to OOM vs KD-tree shape mismatches, and update docs/fix_plan.md with the new failure so we can decide whether a deeper shim (or legacy env) is required.

**Findings Applied (Mandatory)**
- CONFIG-001 — ensure every legacy entrypoint runs `update_legacy_dict(params.cfg, config)` before training/inference so params.cfg matches the dataclass config.
- NORMALIZATION-001 — keep intensity-scaling evidence intact (store stats JSON/Markdown) so we can prove whether loader normalization causes the bias.
- POLICY-001 — continue running everything with TensorFlow (torch policy satisfied) and avoid swapping interpreters.

**Pointers**
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_dose_stage.py: compatibility runner to edit.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/tfa_stub/tensorflow_addons/image/__init__.py: shim that currently implements `translate`/`gaussian_filter2d`.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T092411Z/: contains import_path.log, simulation_attempt*.log, and should host the next run’s summary.
- docs/fix_plan.md: DEBUG-SIM-LINES-DOSE-001 Attempts entry for this loop.
- plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md: high-level log of the manual override progress and remaining blockers.

**Next Up (optional)**
1. Once simulation/training/inference complete, document the ground-truth artifacts (params snapshot, stats, model outputs) and update Phase A1b exit criteria.
2. If the runner still fails despite clamping, escalate to a dedicated environment initiative (legacy TF + tf-addons) before retrying A1b.
