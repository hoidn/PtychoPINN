# Implementation Plan — FIX-PYTORCH-FORWARD-PARITY-001

## Initiative
- ID: FIX-PYTORCH-FORWARD-PARITY-001
- Title: Stabilize Torch Forward Patch Parity
- Owner/Date: Ralph / 2025-11-13
- Status: in_progress
- Priority: High
- Working Plan: this file
- Reports Hub (primary): `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/`

## Context Priming (read before edits)
> Required reading before touching this initiative.
- [x] docs/index.md — Documentation map; use it to locate workflow/spec files referenced below.
- [x] docs/fix_plan.md — Master ledger; ensures this plan aligns with other active focuses and dwell rules.
- [x] docs/findings.md — POLICY-001, CONFIG-001, and known PyTorch/translation findings that constrain this work.
- [x] docs/workflows/pytorch.md — Canonical PyTorch training/inference workflow, CLI expectations, and reporting policy.
- [x] docs/DEVELOPER_GUIDE.md — Two-system architecture; highlights forbidden core files (`ptycho/model.py`, etc.).
- [x] docs/specs/spec-ptycho-workflow.md — Normative forward pipeline semantics (reassembly, scaling, offsets).
- [x] docs/specs/spec-ptycho-interfaces.md — Data contracts for grouped tensors, offsets, and stitching inputs.

## Problem Statement
PyTorch forward inference currently produces impulse-like patches with extremely low variance even after per-patch normalization, diverging from TensorFlow behavior before stitching occurs. Scaling telemetry shows training and inference operate with different normalization, and we lack a structured plan to instrument, align, and test the PyTorch forward path to achieve TF-level patch quality.

## Objectives
- Provide deterministic training/inference instrumentation that logs per-patch mean/variance and normalized patch grids under controlled baselines.
- Align PyTorch training/inference scaling (intensity_scale override, object_big, physics weighting) so Torch patches exhibit TF-like structure before stitching.
- Capture paired TF vs Torch debug dumps with reproducible commands and quantitative parity checks (variance ratios, MAE/SSIM of stitched canvases).
- Add regression tests/guards and documentation updates that fail fast if Torch patch variance collapses or scaling overrides drift.

## Deliverables
1. Updated PyTorch instrumentation and CLI plumbing (`ptycho_torch/model.py`, `ptycho_torch/inference.py`, training/inference scripts) that produce normalized patch grids and variance logs on demand.
2. Config/bridge fixes (object_big defaults, intensity_scale persistence/override) plus associated pytest coverage stored under the Reports Hub.
3. TF vs Torch comparison artifacts (scripts/notebooks, MAE/SSIM plots) archived under `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/...`.
4. Regression tests + doc/test-registry updates describing the instrumentation workflow and new selectors.

## Phases Overview
- Phase A — Diagnostics & Baseline Capture: implement instrumentation and archive the short Torch baseline.
- Phase B — Scaling & Config Alignment: fix scaling/object_big semantics, persist overrides, and validate via targeted tests.
- Phase C — TF vs Torch Parity Proof & Guards: capture TF baselines, compare metrics, and add regression guards/tests/docs.

## Exit Criteria
1. Torch training and inference logs show consistent `mean_input_scale`, `mean_physics_scale`, and per-patch `var_zero_mean` within ±10% of TF baselines for the reference dataset.
2. TF/Torch debug dumps (logs, `stats.json`, normalized PNGs, stitched canvases) are archived under the Reports Hub with commands recorded.
3. New/updated tests (e.g., `pytest tests/torch/test_inference_reassembly_parity.py`, any new scaling guards) pass locally with logs captured in the Reports Hub.
4. Documentation/test registry updates completed for any new selectors; `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` reflect changes.

## Phase A — Diagnostics & Baseline Capture
### Checklist
- [x] A0: Thread the CLI TrainingPayload (or an equivalent override hook) from `ptycho_torch/train.py::cli_main` into `run_cdi_example_torch → train_cdi_model_torch → _train_with_lightning` so the Lightning module actually receives `pt_inference_config.log_patch_stats` / `patch_stats_limit` before rerunning the baseline. (dc5415ba)
- [x] A1: Extend `ptycho_torch/model.py` and `ptycho_torch/inference.py` with one-time per-patch variance logging plus normalized patch grid dumps.
- [x] A2: Re-run the short Torch baseline (10 epochs, 256 samples) via `python -m ptycho_torch.train` and `python -m ptycho_torch.inference --debug-dump`, archiving logs/PNG/JSON artifacts under `.../reports/.../torch_baseline/`. (2025-11-14 rerun complete)
- [x] A3: Summarize baseline stats (mean, std, `var_zero_mean`) and normalized grid observations in a notebook/script stored under the Reports Hub. (Stats captured in torch_patch_stats.json and artifact_inventory.txt)

### Pending Tasks (Engineering)
- The CLI/workflow bridge now threads the TrainingPayload from commit `dc5415ba`, and instrumentation hooks already emit stats under the pytest selector.
- Evidence audit (2025-11-17): both `$HUB/cli/train_patch_stats_rerun.log` and `.../cli/inference_patch_stats_rerun.log` still begin with `2025-11-14`, and `analysis/artifact_inventory.txt` is stamped “Phase A … (2025-11-14)”, so we still need a post-fix rerun.
- Existing hub evidence (`analysis/artifact_inventory.txt`) still reflects the Nov 14 pre-fix rerun: `cli/train_patch_stats_rerun.log:1-4` and `cli/inference_patch_stats_rerun.log:1-4` both show 2025-11-14 timestamps, and `analysis/artifact_inventory.txt:3-35` still cites the same run. Repeat the short baseline + inference commands with `--log-patch-stats --patch-stats-limit 2`, capture the emitted JSON/PNG artifacts, and copy them into the Reports Hub alongside the CLI logs.
- Ralph’s latest evidence-only commit (`876eeb12`) refreshed `outputs/torch_forward_parity_baseline/analysis/torch_patch_stats*.json` plus `train_debug.log`, but it never touched `$HUB`; `analysis/artifact_inventory.txt` and the CLI logs are still Nov 14. After the rerun completes, copy the new stats/grid/debug artifacts into `$HUB/analysis/` and update `$HUB/summary.md` and the initiative summary before closing Phase A.
- Refresh `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and the initiative summary once artifacts land; log blockers in `$HUB/red/blocked_<timestamp>.md`.
- 2025-11-18 audit: the supposedly “_v2” hub files (`green/pytest_patch_stats_rerun_v2.log:1`, `cli/train_patch_stats_rerun_v2.log:1`, `cli/inference_patch_stats_rerun_v2.log:1`) still open with the original 2025-11-14 timestamps, so no post-dc5415ba rerun evidence exists inside `$HUB`. Overwrite them (or emit `_v3` suffixed copies) once the new selector/train/infer commands complete so reviewers can see a clearly post-2025-11-17 data set.
- `outputs/torch_forward_parity_baseline/analysis/` currently contains only the base `torch_patch_stats.json` + `torch_patch_grid.png` pair (no `_v2` artifacts to import), so the rerun must regenerate the stats/grids before copying them into `$HUB/analysis/forward_parity_debug_v2/` and updating the inventories.

### Actionable Do Now — Phase A Evidence Refresh
0. Capture context before clobbering: `head -n 5 "$HUB"/green/pytest_patch_stats_rerun_v2.log`, `head -n 5 "$HUB"/cli/train_patch_stats_rerun_v2.log`, and `head -n 5 "$HUB"/cli/inference_patch_stats_rerun_v2.log` all still print `2025-11-14 ...`, so archive those snippets in the Turn Summary when reporting the overwrite.
1. Guard env vars so selectors cite `docs/TESTING_GUIDE.md` (KB: POLICY-001 / CONFIG-001 / ANTIPATTERN-001):  
   `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`  
   `export HUB=$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity`  
   `export OUT=outputs/torch_forward_parity_baseline`
2. Re-run the instrumentation selector to prove the TrainingPayload-threaded flags still work:  
   `pytest tests/torch/test_cli_train_torch.py::TestPatchStatsCLI::test_patch_stats_dump -vv | tee \"$HUB\"/green/pytest_patch_stats_rerun.log`
3. Execute the short 10-epoch Torch baseline with instrumentation enabled (same dataset slices as Nov 14):  
   ```bash
   python -m ptycho_torch.train \
     --train_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz \
     --test_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
     --output_dir \"$OUT\" \
     --n_images 256 --gridsize 2 --batch_size 4 \
     --max_epochs 10 --neighbor_count 4 \
     --torch-loss-mode poisson --accelerator gpu --deterministic --quiet \
     --log-patch-stats --patch-stats-limit 2 \
     |& tee \"$HUB\"/cli/train_patch_stats_rerun.log
   ```
4. Immediately run inference with the debug dump + patch stats enabled:  
   ```bash
   python -m ptycho_torch.inference \
     --model_path \"$OUT\" \
     --test_data datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
     --output_dir \"$OUT\"/inference \
     --n_images 128 --accelerator gpu \
     --debug-dump \"$HUB\"/analysis/forward_parity_debug \
     --log-patch-stats --patch-stats-limit 2 \
     |& tee \"$HUB\"/cli/inference_patch_stats_rerun.log
   ```
5. Copy `torch_patch_stats*.json`, `torch_patch_grid*.png`, and the refreshed `forward_parity_debug/` bundle from `\"$OUT\"/analysis/` into `$HUB/analysis/`, then update `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md` so Phase A evidence clearly post-dates dc5415ba.
6. If CUDA/memory blocks any command, capture the minimal error signature and create `$HUB/red/blocked_<timestamp>.md` referencing POLICY-001 / CONFIG-001 before handing the focus back.
- 2025-11-17 audit: `$HUB/cli/train_patch_stats_rerun.log`, `$HUB/cli/inference_patch_stats_rerun.log`, and `analysis/artifact_inventory.txt` are still stamped 2025-11-14, so commit 876eeb12’s patch-stat outputs only exist under `outputs/torch_forward_parity_baseline/`; rerun steps 1–6 and overwrite the hub before starting Phase B (KB: POLICY-001 / CONFIG-001 / ANTIPATTERN-001).
- 2025-11-14T1323Z audit: Re-read `$HUB/analysis/artifact_inventory.txt` and the first lines of `$HUB/cli/train_patch_stats_rerun.log`/`.../inference_patch_stats_rerun.log` (timestamps still `2025-11-14 04:49:57` and `04:50:41`), confirming no new hub artifacts landed after the TrainingPayload follow-up commit `876eeb12`. Repeat steps 1–6 before touching Phase B, and label the refreshed logs `*_rerun_v2.log` plus `_v2` suffixes for the JSON/PNG artifacts so the summary can show the overwrite clearly. Update `$HUB/summary.md` with the new filenames and drop a red blocker if CUDA/memory interrupts the rerun.

### Notes & Risks
- Keep instrumentation gated (first batch or debug flag) to avoid log spam during full training.
- Do not modify TF core files (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`) without separate approval.

## Phase B — Scaling & Config Alignment
### Checklist
- [x] B1: Enforce `object_big=True` and physics weighting through `ptycho_torch/config_factory.py`, CLI wrappers, and inference payloads; document behavior in `docs/workflows/pytorch.md`. (Object defaults verified in `config_factory.py:205-234`.)
- [x] B2: Persist bundle `intensity_scale` during training export and reuse it in inference; add tests (e.g., `tests/torch/test_model_manager.py`) to ensure the scale is stored/restored. (Implemented 2025-11-14, commit 9a09ece2)
- [x] B3: Validate scaling by re-running the short baseline and `pytest tests/torch/test_inference_reassembly_parity.py -vv`, capturing logs under `.../reports/.../scaling_alignment/`. (Validated 2025-11-14T1431Z, commit 08cfe61b: intensity_scale=9.882118 loaded from bundle)

### Action Plan — B2 (Intensity Scale Persistence)
1. **Capture the real scale during training.**
   - Inspect `PtychoPINN_Lightning` and `PtychoPINN` (`ptycho_torch/model.py`) to read the learned value from `model.scaler.log_scale` after `trainer.fit`.
   - When `intensity_scale_trainable` is false or the parameter is missing, compute a fallback using the spec formula (`s ≈ sqrt(config.nphotons) / (config.model.N / 2)`, see `docs/specs/spec-ptycho-core.md:80-110`).
   - Expose the resolved scalar through `_train_with_lightning` (e.g., stash under `train_results['intensity_scale']`).
2. **Persist the value inside bundles.**
   - Thread the captured scalar into `save_torch_bundle(...)` via the existing `intensity_scale` argument so that `params_snapshot['intensity_scale']` holds the real value instead of `1.0`.
   - Update `load_inference_bundle_torch` / CLI inference wiring to prefer the stored value and log it (selector already prints `bundle_intensity_scale`, so the log should show the new number instead of `1.0`).
3. **Guard with tests.**
   - Extend `tests/torch/test_model_manager.py` (or add a new test in `tests/torch/test_workflows_components.py`) to assert that passing a non-default scale ends up in `diffraction_to_obj/params.dill` and that `load_inference_bundle_torch` returns it.
   - Cover the fallback path (no learned parameter) and the learned-parameter path by injecting a sentinel Lightning module with a mocked scaler.
4. **Docs + evidence.**
   - Document the persistence rule in `docs/workflows/pytorch.md` (short subsection under bundle persistence).
   - After code/test updates, rerun the short baseline to regenerate `cli/inference_patch_stats_rerun*.log` so the log line `Loaded intensity_scale …` reports the stored float; archive the new bundle params diff in the hub.

### Action Plan — B3 (Scaling Validation)
1. **Pre-flight + hub prep.**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, set `HUB="$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity"`, `OUT="$PWD/outputs/torch_forward_parity_baseline"`, and `SCALING="$HUB/scaling_alignment/phase_b3"`.
   - Create `"$SCALING/cli"`, `"$SCALING/analysis"`, and `"$SCALING/green"` so new evidence is isolated from the Phase A/B runs. Confirm `$OUT` contains the latest bundle from commit `9a09ece2`.
2. **Pytest guard (torch inference reassembly).**
   - Run `pytest tests/torch/test_inference_reassembly_parity.py -vv | tee "$SCALING/green/pytest_inference_reassembly.log"`.
   - Failure output (if any) should be copied to `$HUB/red/blocked_<timestamp>.md` referencing CONFIG-001 / POLICY-001 before exiting.
3. **Short baseline rerun with instrumentation.**
   - Reuse the canonical commands from `plan/plan.md` (10 epochs, 256 groups, `--log-patch-stats --patch-stats-limit 2`) but stream logs to the scaling folder:
     ```bash
     python -m ptycho_torch.train \
       --train_data_file datasets/fly64_coord_variants/fly001_64_train_converted_identity.npz \
       --test_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
       --output_dir "$OUT" \
       --max_epochs 10 \
       --n_images 256 \
       --gridsize 2 \
       --batch_size 4 \
       --torch-loss-mode poisson \
       --accelerator gpu --deterministic \
       --log-patch-stats --patch-stats-limit 2 \
       --quiet \
       |& tee "$SCALING/cli/train_patch_stats_scaling.log"

     python -m ptycho_torch.inference \
       --model_path "$OUT" \
       --test_data datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
       --output_dir "$OUT"/inference \
       --n_images 128 \
       --accelerator gpu \
       --debug-dump "$SCALING/analysis/forward_parity_debug_scaling" \
       --log-patch-stats --patch-stats-limit 2 \
       |& tee "$SCALING/cli/inference_patch_stats_scaling.log"
     ```
   - After inference, grep `Loaded intensity_scale` inside the log to prove it now reports the stored float instead of `1.000000` (see previous baseline at `cli/inference_patch_stats_rerun_v3.log:26`).
4. **Bundle + artifact capture.**
   - Copy the refreshed `outputs/torch_forward_parity_baseline/wts.h5.zip` and `diffraction_to_obj/params.dill` digest (e.g., `shasum`) into `"$SCALING/analysis"` so reviewers can verify the persisted scalar exists.
   - Update `$HUB/analysis/artifact_inventory.txt` with a “Phase B3 — scaling validation” section listing the pytest log, train/infer logs, and bundle digest paths. Summarize the observed intensity_scale value in `$HUB/summary.md`.

### Status
- ✅ Phase B3 completed 2025-11-14 (logs + stats under `scaling_alignment/phase_b3/` prove `intensity_scale=9.882118` is loaded at inference). Proceed to Phase C for cross-backend comparisons.

### Notes & Risks
- Changing defaults impacts existing CLI usage; maintain overrides and document changes clearly.
- Remain compliant with CONFIG-001: update `params.cfg` before invoking legacy modules.

## Phase C — TF vs Torch Parity Proof & Guards
### Action Plan — C1 (Matched TensorFlow baseline)
1. **Guardrails + directories.**
   - `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
   - `export TF_XLA_FLAGS="--tf_xla_auto_jit=0"` (disables the XLA pass that triggered the dynamic_padder RET_CHECK on 2025-11-14; note the value inside each CLI log header so we can prove the mitigation was applied).
   - `export USE_XLA_TRANSLATE=0` (forces `ptycho/tf_helper.should_use_xla()` to bypass `translate_xla()` per `ptycho/tf_helper.py:158-175`, which is the only knob that actually disables the `projective_warp_xla_jit` code path).
   - Before kicking off any CLI commands, emit a one-line capture of both env vars by running `printf 'TF_XLA_FLAGS=%s\nUSE_XLA_TRANSLATE=%s\n' "$TF_XLA_FLAGS" "$USE_XLA_TRANSLATE"` and teeing it into the matching CLI log so the mitigation is self-evident inside the artifact.
   - `HUB="$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity"`
   - `OUT_TORCH="$PWD/outputs/torch_forward_parity_baseline"`
   - `OUT_TF="$PWD/outputs/tf_forward_parity_baseline"`
   - `TF_BASE="$HUB/tf_baseline/phase_c1"`
   - `mkdir -p "$TF_BASE"/{cli,analysis,green}`
2. **Baseline health check.**
   - `pytest tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle -vv | tee "$TF_BASE/green/pytest_tf_integration.log"`
   - Include an `env | grep TF_XLA_FLAGS` snippet in the log (or append to the Turn Summary) to capture the disabled-XLA configuration. File blockers under `$HUB/red/blocked_<timestamp>.md` citing POLICY-001/CONFIG-001 if runtime/env gaps appear.
   - If the selector fails because TensorFlow still tries to JIT compile the identity dataset, capture the RET_CHECK signature immediately; do **not** proceed to the CLI commands.
3. **TensorFlow short training run (mirrors Torch Phase B3).**
   - **Dataset note:** The `fly001_64_train_converted_identity.npz` variant has now failed twice (see `tf_baseline/phase_c1/red/blocked_*tf_xla_*.md`) with the same XLA RET_CHECK even after exporting `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`. To keep Phase C moving, default to the non-identity dataset `datasets/fly64/fly001_64_train_converted.npz` (this file actually exists; the earlier `fly64_coord_variants` path was a typo). Record the dataset choice and whether a matching PyTorch rerun is required for Phase C2 parity inside `$HUB/summary.md`.
  ```bash
  python scripts/training/train.py \
    --backend tensorflow \
    --train_data_file datasets/fly64/fly001_64_train_converted.npz \
    --test_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
    --output_dir "$OUT_TF" \
    --n_groups 256 \
    --gridsize 2 \
    --neighbor_count 7 \
    --batch_size 4 \
    --nepochs 10 \
    --do_stitching \
    |& tee "$TF_BASE/cli/train_tf_phase_c1.log"
  ```
4. **TensorFlow inference with debug dump.**
   ```bash
   python scripts/inference/inference.py \
     --backend tensorflow \
     --model_path "$OUT_TF" \
     --test_data datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
     --output_dir "$OUT_TF/inference_phase_c1" \
     --n_images 128 \
     --debug_dump "$TF_BASE/analysis/forward_parity_debug_tf" \
     --comparison_plot \
     |& tee "$TF_BASE/cli/inference_tf_phase_c1.log"
   ```
   - Confirm `forward_parity_debug_tf` contains `stats.json`, `offsets.json`, and patch-grid PNGs (TF helper already emits these). Keep TF_XLA_FLAGS exported for the inference process as well so the mitigation remains active.
5. **Bundle + metrics bookkeeping.**
   - `shasum "$OUT_TF/wts.h5.zip" > "$TF_BASE/analysis/bundle_digest_tf_phase_c1.txt"`.
   - `python - <<'PY'` snippet to print the TF stats alongside the PyTorch Phase B3 stats (`scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json`) and jot the numbers into `$TF_BASE/analysis/phase_c1_stats.txt` for Phase C2 comparison.
   - If the TF baseline used a non-identity dataset, append a `Dataset note:` bullet to both `$HUB/analysis/artifact_inventory.txt` and `$HUB/summary.md` explicitly stating which dataset was used and whether a follow-up PyTorch rerun is required for apples-to-apples comparisons.
6. **Hub updates.**
   - Append a “Phase C1 — TF baseline” section to `$HUB/analysis/artifact_inventory.txt` (reference logs, stats, bundle digest).
   - Mention the TF stats + PyTorch reference deltas in `$HUB/summary.md`.

   - Include the TF_XLA_FLAGS line in both CLI logs’ headers so we can prove the mitigation remained active end-to-end.
   - Include the `USE_XLA_TRANSLATE` capture inside each CLI log as well so reviewers can confirm the mitigation was applied.
7. **Fallback path if XLA still blocks training.**
   - If TensorFlow still errors even on the non-identity dataset with `USE_XLA_TRANSLATE=0`, capture the error signature under `$TF_BASE/red/blocked_<timestamp>_tf_xla_disabled.md`, cite Finding XLA-DYN-DOT-001, and note whether the failure occurred before or after any data-dependent ops. The blocker must quote the env capture so we can prove both knobs were set.
   - Only fall back to additional dataset changes (or propose PyTorch-only Phase C evidence) after documenting the new blocker and updating `docs/fix_plan.md` + this plan with the proposed mitigation. Continue to record whether a corresponding PyTorch rerun is required before Phase C2 comparisons whenever the dataset diverges from the Phase B3 baseline.

### Action Plan — C1b (GS1 fallback to bypass translation)
1. **Environment + directories.**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`, `USE_XLA_TRANSLATE=0`.
   - Set `OUT_TORCH_GS1="$OUT_TORCH/gs1_phase_c1"`, `OUT_TF_GS1="$OUT_TF/gs1_phase_c1"`, and `TF_BASE_GS1="$HUB/tf_baseline/phase_c1_gs1"`.
   - Create `$HUB/scaling_alignment/phase_c1_gs1/{cli,analysis,green}` + `$TF_BASE_GS1/{cli,analysis,green,red}`; prepend the env capture to every CLI log (use `printf 'TF_XLA_FLAGS=%s\nUSE_XLA_TRANSLATE=%s\n' …`).
2. **PyTorch GS1 short baseline.**
   - Training:
     ```bash
     python -m ptycho_torch.train \
       --train_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz \
       --test_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
       --output_dir "$OUT_TORCH_GS1" \
       --n_images 256 --gridsize 1 --batch_size 4 \
       --max_epochs 10 --neighbor_count 1 \
       --torch-loss-mode poisson --accelerator gpu --deterministic --quiet \
       --log-patch-stats --patch-stats-limit 2 \
       |& tee "$HUB/scaling_alignment/phase_c1_gs1/cli/train_patch_stats_gs1.log"
     ```
   - Inference:
     ```bash
     python -m ptycho_torch.inference \
       --model_path "$OUT_TORCH_GS1" \
       --test_data datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
       --output_dir "$OUT_TORCH_GS1/inference" \
       --n_images 128 --accelerator gpu \
       --debug-dump "$HUB/scaling_alignment/phase_c1_gs1/analysis/forward_parity_debug_gs1" \
       --log-patch-stats --patch-stats-limit 2 \
       |& tee "$HUB/scaling_alignment/phase_c1_gs1/cli/inference_patch_stats_gs1.log"
     ```
   - Copy `torch_patch_stats*_gs1.{json,png}` + debug bundle into `$HUB/scaling_alignment/phase_c1_gs1/analysis/` and extend the hub inventory with a “PyTorch GS1 fallback” section.
3. **TensorFlow GS1 guard + CLI.**
   - Integration gate: `pytest tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle -vv | tee "$TF_BASE_GS1/green/pytest_tf_integration_gs1.log"`.
   - Training:
     ```bash
     python scripts/training/train.py \
       --backend tensorflow \
       --train_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_train.npz \
       --test_data_file datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
       --output_dir "$OUT_TF_GS1" \
       --n_groups 256 --gridsize 1 --neighbor_count 1 \
       --batch_size 4 --nepochs 10 --do_stitching \
       |& tee "$TF_BASE_GS1/cli/train_tf_phase_c1_gs1.log"
     ```
   - Inference:
     ```bash
     python scripts/inference/inference.py \
       --backend tensorflow \
       --model_path "$OUT_TF_GS1" \
       --test_data datasets/fly001_reconstructed_prepared/fly001_reconstructed_final_downsampled_data_test.npz \
       --output_dir "$OUT_TF_GS1/inference_phase_c1_gs1" \
       --n_images 128 \
       --debug_dump "$TF_BASE_GS1/analysis/forward_parity_debug_tf_gs1" \
       --comparison_plot \
       |& tee "$TF_BASE_GS1/cli/inference_tf_phase_c1_gs1.log"
     ```
4. **Bookkeeping.**
   - Bundle digests: `shasum "$OUT_TORCH_GS1/wts.h5.zip" > "$HUB/scaling_alignment/phase_c1_gs1/analysis/bundle_digest_torch_gs1.txt"` and `shasum "$OUT_TF_GS1/wts.h5.zip" > "$TF_BASE_GS1/analysis/bundle_digest_tf_phase_c1_gs1.txt"`.
   - Stats delta: `python - <<'PY'` to load `forward_parity_debug_gs1/stats.json` + `forward_parity_debug_tf_gs1/stats.json`, print mean/std/var_zero_mean deltas, append to `$TF_BASE_GS1/analysis/phase_c1_gs1_stats.txt`.
   - Hub updates: add a “Phase C1 — GS1 fallback” section to `$HUB/analysis/artifact_inventory.txt` + `$HUB/summary.md` (include dataset note + env capture).
5. **Blockers.**
   - PyTorch failures → `$HUB/scaling_alignment/phase_c1_gs1/red/blocked_<timestamp>_torch_gs1.md` (cite POLICY-001/CONFIG-001).
   - TensorFlow failures → `$TF_BASE_GS1/red/blocked_<timestamp>_tf_gs1.md` (cite XLA-DYN-DOT-001 or translate_core bug). Stop and reassess if GS1 still cannot complete.

### Action Plan — C1c (GS1 evidence consolidation & dataset note)
1. **Compute GS1 vs Phase B3 stats delta (analysis artifact).**
   - Goal: quantify how the GS1 fallback differs from the Phase B3 (gridsize 2) baseline so reviewers can see exactly what evidence exists even though TF remains blocked.
   - Command (runs entirely on existing artifacts):
     ```bash
     HUB="$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity"
     python - <<'PY'
     import json, os, pathlib
     hub = pathlib.Path(os.environ["HUB"])
     phase_b3 = hub / "scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json"
     gs1 = hub / "scaling_alignment/phase_c1_gs1/analysis/forward_parity_debug_gs1/stats.json"
     out = hub / "tf_baseline/phase_c1_gs1/analysis/phase_c1_gs1_stats.txt"
     data = {"phase_b3": json.loads(phase_b3.read_text()), "phase_c1_gs1": json.loads(gs1.read_text())}
     lines = [
         "# Phase C1 GS1 fallback vs Phase B3 scaling",
         f"phase_b3.patch.var_zero_mean={data['phase_b3']['patch_amplitude']['var_zero_mean']}",
         f"phase_c1_gs1.patch.var_zero_mean={data['phase_c1_gs1']['patch_amplitude']['var_zero_mean']}",
         f"ratio_patch_var_zero_mean={data['phase_c1_gs1']['patch_amplitude']['var_zero_mean'] / data['phase_b3']['patch_amplitude']['var_zero_mean'] if data['phase_b3']['patch_amplitude']['var_zero_mean'] else 'inf'}",
         f"phase_b3.canvas.var_zero_mean={data['phase_b3']['canvas_amplitude']['var_zero_mean'] if 'var_zero_mean' in data['phase_b3']['canvas_amplitude'] else 'n/a'}",
         f"phase_c1_gs1.canvas.var_zero_mean={data['phase_c1_gs1']['canvas_amplitude'].get('var_zero_mean', 0)}",
     ]
     out.write_text("\n".join(str(x) for x in lines))
     print(f"Wrote stats delta to {out}")
     PY
     ```
   - Copy (or symlink) the text file into `$HUB/scaling_alignment/phase_c1_gs1/analysis/` so both backend folders cite the same numbers, then sha1 the file for reviewers.
2. **Refresh the main artifact inventory.**
   - Append a “Phase C1 — GS1 fallback (PyTorch-only)” section to `$HUB/analysis/artifact_inventory.txt` that lists:
     - `scaling_alignment/phase_c1_gs1/cli/{train_patch_stats_gs1.log,inference_patch_stats_gs1.log}`
     - `scaling_alignment/phase_c1_gs1/analysis/{torch_patch_stats_gs1.json,forward_parity_debug_gs1/,bundle_digest_torch_gs1.txt,phase_c1_gs1_stats.txt}`
     - TF blocker files at `$HUB/tf_baseline/phase_c1_gs1/red/blocked_*`
   - Call out explicitly that TF integration + stitching still fail even at GS1, referencing the blocker filenames and error strings.
3. **Update the initiative summary + hub summary.**
   - Prepend the Turn Summary block in `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md` and `$HUB/summary.md` with:
     - Dataset note (GS1 fallback uses the same fly001 reconstructed dataset for both PyTorch commands).
     - Stats-delta path + sha1.
     - TF blocker references and next decision (PyTorch-only parity unless TF translation gets fixed).
4. **Blocker handling.**
   - If the stats JSON files are missing or malformed, create `$HUB/scaling_alignment/phase_c1_gs1/red/blocked_<timestamp>_missing_gs1_stats.md` describing the gap before touching inventories.
   - If TF blockers change (new env capture, different stack), add the file path to the inventory entry and summarize the delta in the Turn Summary.

### Action Plan — C2 (PyTorch-only comparison & guard rails)
1. **Promote reusable comparison tooling (scriptization Tier 2).**
   - Create `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/bin/phase_c2_compare_stats.py` with the header template from CLAUDE.md §scriptization.
   - Inputs:
     - `--baseline-stats` → default `$HUB/scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json`
     - `--candidate-stats` → default `$HUB/scaling_alignment/phase_c1_gs1/analysis/forward_parity_debug_gs1/stats.json`
     - Optional `--label-baseline/--label-candidate` for summary text overrides.
     - `--out` → metrics text/JSON path inside `$HUB/analysis/`.
   - Behavior:
     - Load both stats JSON files (schema documented in `docs/workflows/pytorch.md` and `specs/ptycho-workflow.md`).
     - Emit ratios/differences for `mean`, `std`, and `var_zero_mean` under both `patch_amplitude` and `canvas_amplitude`.
     - When the baseline var is zero, emit `ratio=nan` and set `status=blocked_baseline_zero` so reviewers understand the limitation.
     - Include summary sentences that cite POLICY-001 (PyTorch mandatory) and CONFIG-001 (shared config bridge) because we are operating PyTorch-only pending TF fixes.

2. **Execute the comparison script for the GS1 fallback.**
   ```bash
   export HUB="$PWD/plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity"
   python plans/active/FIX-PYTORCH-FORWARD-PARITY-001/bin/phase_c2_compare_stats.py \
     --baseline-stats "$HUB/scaling_alignment/phase_b3/analysis/forward_parity_debug_scaling/stats.json" \
     --candidate-stats "$HUB/scaling_alignment/phase_c1_gs1/analysis/forward_parity_debug_gs1/stats.json" \
     --label-baseline "Phase B3 (gridsize=2)" \
     --label-candidate "Phase C1 GS1 fallback (PyTorch-only)" \
     --out "$HUB/analysis/phase_c2_pytorch_only_metrics.txt"
   shasum "$HUB/analysis/phase_c2_pytorch_only_metrics.txt" > "$HUB/analysis/phase_c2_pytorch_only_metrics.sha1"
   ```
   - Tee stdout/stderr into `$HUB/cli/phase_c2_compare_stats.log` for reproducibility.
   - If either stats file is missing, immediately create `$HUB/scaling_alignment/phase_c1_gs1/red/blocked_<timestamp>_missing_stats_json.md` and stop (no partial edits).

3. **Summaries + inventory refresh.**
   - Append a “Phase C2 — PyTorch-only comparison” section to `$HUB/analysis/artifact_inventory.txt` referencing:
     - The new metrics + sha1 file(s).
     - Source stats JSON paths + their sha1s (already captured during Phase B3/C1).
     - The CLI log for the comparison script.
   - Prepend both `$HUB/summary.md` and `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md` with:
     - Headline sentence summarizing the metrics (e.g., “Patch variance collapsed from 8.97e9 (Phase B3) to 0.0 (GS1).”)
     - Reference to TF blocker files to remind reviewers why the comparison is PyTorch-only.
     - Statement that MAE/SSIM cannot be computed until TF baseline unblocks; note this as a dependency for C3 regression guards.

4. **Forward-looking guard rail.**
   - Capture the script usage (CLI example + expected outputs) inside this plan and in the Reports Hub README so future loops can reuse it when TF evidence appears.
   - When TF baseline artifacts land, re-run the script with `--candidate-stats` pointing at `$HUB/tf_baseline/.../stats.json` and update the metrics file rather than creating a new one to preserve diffability.

### Checklist
- [ ] C1: Run the matched TF baseline (`scripts/training/train.py --backend tensorflow` + `scripts/inference/inference.py --backend tensorflow --debug_dump`) and archive artifacts (`.../reports/.../tf_baseline/`).
- [ ] C2: Build a comparison script/notebook that ingests Torch/TF dumps, reports variance ratios and stitched MAE/SSIM, and summarize findings in `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md`.
- [ ] C3: Add regression tests/guards (e.g., small synthetic dataset test under `tests/torch/`) ensuring `var_zero_mean` stays above a threshold and the CLI applies scaling overrides. Update docs/test registries if selectors change.
- [x] C1c: Consolidate GS1 fallback evidence (stats delta artifact, inventory/summary update, TF blocker cross-links).

### Pending Tasks (Engineering)
- Implement comparison tooling, define acceptable thresholds, then codify guards/tests.
- Ensure TF/Torch runs share seeds/dataset slices; document commands in the Reports Hub.

### Notes & Risks
- TF baseline may require probe-mask tweaks; document differences clearly.
- Regression tests must stay within TESTING_GUIDE time budgets.

## Deprecation / Policy Banner
- Reminder: Do not edit `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` within this focus without explicit supervisor approval.

## Artifacts Index
- Reports root: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/`
- Torch baseline run: `2025-11-13T000000Z/forward_parity/torch_baseline/`
- TF baseline run: `2025-11-13T000000Z/forward_parity/tf_baseline/`

## Open Questions & Follow-ups
- Should we backfill existing PyTorch bundles with recorded `intensity_scale`, or is a forward-looking fix sufficient?
- Do we need a CLI flag to capture training-time per-patch stats for future initiatives?
