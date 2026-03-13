Use `lines_256` as the experiment dataset.
Resolve that dataset only through [docs/studies/lines_256_dataset.md](/home/ollie/Documents/tmp/PtychoPINN/docs/studies/lines_256_dataset.md). Do not substitute a different dataset or guess alternate paths.
Resolve the experiment loop, baseline setup, results ledger path, and keepability rule only through [docs/studies/lines_256_arch_improvement_loop.md](/home/ollie/Documents/tmp/PtychoPINN/docs/studies/lines_256_arch_improvement_loop.md).

Task:
Run a focused PyTorch architecture-improvement loop against `lines_256`. Your goal is to improve `amp_ssim` on this dataset by following the exact branch-advance loop in the loop document.

Metric:
- The metric is `amp_ssim`.
- `amp_ssim` is the only optimization and keep/discard metric.
- You may log supporting metrics, but they must not override `amp_ssim` when making decisions.

Scope:
- Stay on the PyTorch path.
- Use `scripts/studies/run_lines_256_arch_experiment.py` as the default experiment runner.
- Do not switch to sweep/runbook machinery unless a supervising human explicitly asks for it.
- Work on architecture and training configuration only unless a blocking bug forces a narrow fix.
- Do not change the dataset split.
- Do not invent new evaluation metrics.

Editable surface:
- You may change these implementation files by default:
  - `ptycho_torch/generators/hybrid_resnet.py`
  - `ptycho/config/config.py` only for `PyTorchExecutionConfig` fields
  - `ptycho_torch/config_params.py`
  - `ptycho_torch/workflows/components.py`
  - `scripts/studies/run_lines_256_arch_experiment.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py` only when the experiment needs runbook support
- You may change these verification/docs files when they directly support the experiment:
  - `tests/torch/test_fno_generators.py`
  - `tests/torch/test_grid_lines_torch_runner.py`
  - `tests/studies/test_run_lines_256_arch_experiment.py`
  - `tests/studies/test_hybrid_resnet_mode_skip_sweep.py`
  - `docs/CONFIGURATION.md`
  - `docs/workflows/pytorch.md`
  - `ptycho_torch/generators/README.md`
  - `docs/studies/index.md`
- Escalate before changing anything outside that surface.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not broaden the editable surface just because a change looks convenient.

Default starting point:
- Start from the current default hybrid-resnet control unless the supervising context gives a different anchor:
  - `fno_modes=12`
  - `fno_width=32`
  - `fno_blocks=4`
  - `hybrid_skip_connections=off`
  - `hybrid_downsample_steps=2`
  - `hybrid_downsample_op=stride_conv`
  - `hybrid_resnet_blocks=6`
  - `hybrid_skip_style=add`
  - `probe_mask=off`
  - `torch_mae_pred_l2_match_target=off`

The experiment loop:
LOOP FOREVER:
1. Read `docs/studies/lines_256_dataset.md` and `docs/studies/lines_256_arch_improvement_loop.md`.
2. Check tracked git state. If tracked files are dirty, stop and report a blocker.
3. Ensure the untracked TSV ledger exists at the exact path defined by the loop document.
4. Start a new session by regenerating a fresh baseline from the current `HEAD` with the default control and fixed budget from the loop document.
5. Publish the baseline comparison PNG to the session gallery dir defined by the loop document.
6. Record that baseline in the TSV ledger, including the gallery PNG path.
7. Use the latest `baseline` or `keep` row from the current session as the champion.
8. Make one coherent experiment change at a time unless multiple changes are tightly coupled.
9. Stage only the intended source changes. Never stage `state/` or `outputs/`.
10. Create exactly one candidate commit.
11. Run exactly one `lines_256` experiment through the thin wrapper with the same fixed dataset and epoch budget as the session baseline.
12. Publish the candidate comparison PNG to the session gallery dir.
13. Read candidate `amp_ssim`.
14. Append exactly one candidate row to the TSV ledger, including the gallery PNG path.
15. If candidate `amp_ssim` is strictly better than the champion, keep the commit. The branch has advanced.
16. If candidate `amp_ssim` is equal, worse, missing, or the run crashed, record that in the TSV and reset back exactly as defined by the loop document.

Working rules:
1. The TSV ledger is the authoritative experiment history. Keep it untracked so resets do not erase prior results.
2. Every new session starts by regenerating a fresh baseline from the current `HEAD`. Do not depend on an older study run.
3. Equal `amp_ssim` is not good enough. Only strict improvement advances the branch.
4. Prefer simple improvements over broader, riskier changes.
5. Keep `probe_mask` off unless the experiment is explicitly about probe masking.
6. Every run must publish one easy-to-find comparison PNG in the session gallery dir.
7. The comparison PNG must include the probe, not just the object views.
8. If the best next change requires edits outside the editable surface, stop and report that explicitly instead of guessing.

Reporting:
- After each experiment, print a short plain-text summary with:
  - candidate commit
  - champion reference it was compared against
  - candidate `amp_ssim`
  - champion `amp_ssim`
  - `delta_amp_ssim`
  - decision: `keep`, `discard`, or `crash`
  - TSV path
  - output root
  - comparison PNG path

Constraints:
- No unrelated edits.
- No fabricated evidence.
- No hidden dataset substitution.
- No “better” claim without an explicit `amp_ssim` comparison against the current champion from the TSV ledger.
