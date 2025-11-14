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
- [ ] B3: Validate scaling by re-running the short baseline and `pytest tests/torch/test_inference_reassembly_parity.py -vv`, capturing logs under `.../reports/.../scaling_alignment/`.

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

### Pending Tasks (Engineering)
- Execute the B3 validation plan (pytest selector + short baseline rerun + artifact capture) and update the hub inventory/summary once inference logs show the stored intensity scale.

### Notes & Risks
- Changing defaults impacts existing CLI usage; maintain overrides and document changes clearly.
- Remain compliant with CONFIG-001: update `params.cfg` before invoking legacy modules.

## Phase C — TF vs Torch Parity Proof & Guards
### Checklist
- [ ] C1: Run the matched TF baseline (`scripts/training/train.py --backend tensorflow` + `scripts/inference/inference.py --backend tensorflow --debug_dump`) and archive artifacts (`.../reports/.../tf_baseline/`).
- [ ] C2: Build a comparison script/notebook that ingests Torch/TF dumps, reports variance ratios and stitched MAE/SSIM, and summarize findings in `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md`.
- [ ] C3: Add regression tests/guards (e.g., small synthetic dataset test under `tests/torch/`) ensuring `var_zero_mean` stays above a threshold and the CLI applies scaling overrides. Update docs/test registries if selectors change.

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
