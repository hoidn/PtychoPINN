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
- Re-run the short baseline + inference commands with `--log-patch-stats --patch-stats-limit 2`, capture the emitted JSON/PNG artifacts, and copy them into the Reports Hub alongside the CLI logs.
- Refresh `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and the initiative summary once artifacts land; log blockers in `$HUB/red/blocked_<timestamp>.md`.

### Notes & Risks
- Keep instrumentation gated (first batch or debug flag) to avoid log spam during full training.
- Do not modify TF core files (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`) without separate approval.

## Phase B — Scaling & Config Alignment
### Checklist
- [ ] B1: Enforce `object_big=True` and physics weighting through `ptycho_torch/config_factory.py`, CLI wrappers, and inference payloads; document behavior in `docs/workflows/pytorch.md`.
- [ ] B2: Persist bundle `intensity_scale` during training export and reuse it in inference; add tests (e.g., `tests/torch/test_workflows_components.py`) to ensure the scale is stored/restored.
- [ ] B3: Validate scaling by re-running the short baseline and `pytest tests/torch/test_inference_reassembly_parity.py -vv`, capturing logs under `.../reports/.../scaling_alignment/`.

### Pending Tasks (Engineering)
- Update config bridge/factory code, rerun baselines, and ensure logs show consistent scaling.
- Record diffs/commands in the Reports Hub summary and `analysis/artifact_inventory.txt`.

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
