Use `lines_256` as the experiment dataset.
Resolve that dataset only through [docs/studies/lines_256_dataset.md](/home/ollie/Documents/tmp/PtychoPINN/docs/studies/lines_256_dataset.md). Do not substitute a different dataset or guess alternate paths.

Task:
Run a focused PyTorch architecture-improvement loop against `lines_256`. Your goal is to find changes that improve reconstruction quality on this dataset.

Metric:
- The metric is `amp_ssim`.
- `amp_ssim` is the only optimization and keep/discard metric.
- You may log supporting metrics, but they must not override `amp_ssim` when making decisions.

Scope:
- Stay on the PyTorch path.
- Prefer `scripts/studies/grid_lines_torch_runner.py` for single-run experiments.
- You may use `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py` only when you need sweep machinery and the invocation remains valid for a lines-only experiment.
- Work on architecture and training configuration only unless a blocking bug forces a narrow fix.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not change the dataset split.
- Do not invent new evaluation metrics.

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

Working rules:
1. Read the `lines_256` dataset note first and use it as the authoritative dataset contract.
2. Make one coherent experiment change at a time unless multiple changes are tightly coupled.
3. Use an apples-to-apples baseline:
   - same dataset
   - same dataset paths
   - same epoch budget
   - same seed policy
4. If an experiment lowers `amp_ssim`, discard it.
5. If an experiment crashes, summarize the failure and either fix the narrow blocker or abandon the idea.
6. Prefer simple improvements over broader, riskier changes.
7. Keep `probe_mask` off unless the experiment is explicitly about probe masking.

Communication contract:
After each experiment, report a compact result block in plain text:

BEGIN_EXPERIMENT_RESULT
experiment_id: <id>
status: <keep|discard|crash>
hypothesis: <one sentence>
dataset: lines_256
baseline_run: <run id or path>
candidate_run: <run id or path>
amp_ssim: <value or na>
baseline_amp_ssim: <value or na>
delta_amp_ssim: <value or na>
command: <exact command or invocation artifact path>
output_root: <path>
decision_reason: <short explanation tied to amp_ssim>
next_step: <one sentence>
END_EXPERIMENT_RESULT

Decision policy:
- `keep` only when the candidate produces a valid apples-to-apples `amp_ssim` improvement.
- `discard` when `amp_ssim` is worse, flat, confounded, or the change is not worth keeping.
- `crash` when the run fails and no narrow recovery is justified.

Constraints:
- No unrelated edits.
- No fabricated evidence.
- No hidden dataset substitution.
- No “better” claim without an explicit `amp_ssim` comparison against a valid baseline.
