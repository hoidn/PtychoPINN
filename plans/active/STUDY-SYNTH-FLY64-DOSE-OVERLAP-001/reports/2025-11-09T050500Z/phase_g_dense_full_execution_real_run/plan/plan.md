# Phase G Dense Pipeline Execution (2025-11-09T050500Z)

## Objective
Run the dense (dose=1000) Phase C→G workflow end-to-end with `--clobber`, then
generate and archive the Phase G metrics digest so we can cite live MS-SSIM/MAE
deltas in the study summary.

## Scope
- Launch the dense train/test pipeline via `run_phase_g_dense.py` using the new
  hub path and `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
- Capture full CLI transcripts, stdout/stderr, and exit codes for each phase in
  `reports/.../cli/`.
- Once `metrics_summary.json` and `aggregate_highlights.txt` land, execute
  `analyze_dense_metrics.py` to emit `metrics_digest.md` plus log output in the
  `analysis/` folder.
- Update the summary with measured MS-SSIM/MAE deltas and reference all new
  artifacts.
- Refresh `docs/fix_plan.md` Attempts History with this execution evidence and
  record findings adherence.

## Deliverables
1. Fresh Phase C→G CLI transcripts (`phase_c_generation.log`, `phase_d_training.log`,
   `phase_e_evaluation.log`, `phase_f_summary.log`, `phase_g_dense.log`) under
   `cli/`, each including command header + exit code.
2. Analysis artifacts in `analysis/`: `metrics_summary.json`,
   `aggregate_highlights.txt`, `aggregate_report.md`, `metrics_digest.md`,
   `metrics_digest.log`.
3. Updated `summary/summary.md` prepending this loop’s Turn Summary with measured
   deltas and links to new artifacts.
4. `docs/fix_plan.md` Attempts History entry noting the pipeline run, digest
   generation, and findings observed (POLICY-001, CONFIG-001, DATA-001,
   TYPE-PATH-001, OVERSAMPLING-001, STUDY-001).

## Acceptance Criteria
- `run_phase_g_dense.py --clobber` completes without error and regenerates the
  dense hub outputs (train/test) beneath this timestamp.
- `analyze_dense_metrics.py` exits 0, writes `metrics_digest.md`, prints the
  success banner, and summarizes MS-SSIM/MAE deltas (PtychoPINN vs Baseline,
  PtychoPINN vs PtyChi).
- Summary and fix plan reference the new digest/log paths and reported metrics.
- All artifacts remain under
  `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T050500Z/`.

## Findings to Observe
- POLICY-001 — PyTorch dependency required for comparison helpers.
- CONFIG-001 — Export `AUTHORITATIVE_CMDS_DOC` before running pytest or CLI.
- DATA-001 — Generated NPZ outputs must conform to the contract.
- TYPE-PATH-001 — Prefer `Path` objects and deterministic hub paths.
- OVERSAMPLING-001 — Verify overlap statistics in digest align with design.
- STUDY-001 — Track phase performance deltas across fly64 study conditions.

## Risks / Mitigations
- **Long runtime (hours):** Run in tmux/screen or persistent shell; log progress
  timestamps into CLI transcripts.
- **Partial artifact generation:** If a phase fails, capture stdout/stderr,
  exit code, and immediate tree listing under `cli/` before troubleshooting.
- **Digest mismatch vs summary:** Cross-check digest tables against
  `metrics_summary.json` before updating docs.
