Summary: Inspect Lightning checkpoint hyperparameters to unblock Phase D remediation
Mode: none
Focus: INTEGRATE-PYTORCH-001-STUBS — Phase D1b Lightning checkpoint inspection
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/{summary.md,checkpoint_dump.txt,checkpoint_inspection.md}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS D1b @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Run `python -m ptycho_torch.train --train_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz --test_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz --output_dir plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/checkpoint_run --max_epochs 2 --n_images 64 --gridsize 1 --batch_size 4 --device cpu --disable_mlflow` to reproduce `last.ckpt` under the new artifact directory (tests: none)
2. INTEGRATE-PYTORCH-001-STUBS D1b @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Load `checkpoint_run/checkpoints/last.ckpt` with torch and dump `hyper_parameters` key info to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/checkpoint_dump.txt` (tests: none)
3. INTEGRATE-PYTORCH-001-STUBS D1b @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Summarise findings in `checkpoint_inspection.md`, tick off `summary.md` + plan checklist D1b, and add docs/fix_plan.md Attempt #32 referencing the new evidence (tests: none)

If Blocked: Capture whatever stdout/stderr you have from the training command into `checkpoint_dump.txt`, note the blocker in `checkpoint_inspection.md`, leave D1b as `[P]`, and document the failure plus file paths in docs/fix_plan.md before pausing.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:66 — D1b requires checkpoint inspection evidence before we choose a remediation path.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/summary.md:24 — Outstanding checklist item tracks this inspection task.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/summary.md:1 — Reserved artifact hub for this loop; keep all outputs there.
- specs/ptychodus_api_spec.md:205 — Persistence contract demands Lightning checkpoints load without extra constructor args; inspection confirms which contract clause is violated.
- docs/workflows/pytorch.md:260 — Troubleshooting guidance highlights the current TypeError and informs expected checkpoint contents.

How-To Map:
- Ensure artifact directory exists (already provisioned). Training command (Step 1) writes into `checkpoint_run/`; allow it to finish (~2 min CPU).
- After training, inspect checkpoint with:
  `python - <<'PY' > plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/checkpoint_dump.txt`
  ```python
  from pathlib import Path
  import torch

  ckpt_path = Path('plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/checkpoint_run/checkpoints/last.ckpt')
  checkpoint = torch.load(ckpt_path, map_location='cpu')
  print('Checkpoint keys:', sorted(checkpoint.keys()))
  hyper = checkpoint.get('hyper_parameters')
  print('hyper_parameters type:', type(hyper))
  if isinstance(hyper, dict):
      for key, value in hyper.items():
          print(f"{key}: {type(value)} → {value!r}")
  else:
      print('hyper_parameters repr:', repr(hyper))
  ```
  `PY`
- If the checkpoint is large, keep only the dump + md note in git: delete `checkpoint_run/checkpoints/last.ckpt` once inspection is complete (`rm -rf` the `checkpoint_run` directory) to avoid committing big binaries.
- Author `checkpoint_inspection.md` in the same directory with summary bullets (presence/absence of config objects, serialization format, next hypotheses). Reference `checkpoint_dump.txt` and the training command used.
- Update `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/summary.md` pending tasks to `[x]`, mark `phase_d2_completion.md` D1b `[x]`, and record docs/fix_plan.md Attempt #32 with artifact links and findings.

Pitfalls To Avoid:
- Do not commit large `.ckpt` binaries; keep only textual summaries in git.
- Don’t run the full pytest integration suite again—this loop is evidence-only.
- Avoid editing production code or tests; stick to scripts and documentation artifacts.
- Keep CONFIG-001 compliance intact by running the CLI exactly as provided (no manual params.cfg hacks).
- Ensure `checkpoint_dump.txt` captures full key/type info; partial logs won’t unblock remediation.
- Maintain artifact hygiene—everything from this loop stays under the 2025-10-19T123000Z directory.
- Remember to increment docs/fix_plan.md Attempts and reference both dump + markdown outputs.
- If the training command fails, copy stderr into the artifact directory before cleaning up.
- Don’t leave temporary files outside the artifact directory; delete `checkpoint_run` after dumping info.
- Keep Mode `none` — no test runs expected this loop.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:66
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T095900Z/phase_d2_completion/summary.md:24
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T123000Z/phase_d2_completion/summary.md:1
- specs/ptychodus_api_spec.md:205
- docs/workflows/pytorch.md:260

Next Up:
- Phase D2 remediation: inspect checkpoint payload structure (e.g., evaluate `save_hyperparameters` usage) or draft failing test for Lightning load path once inspection confirms gaps.
