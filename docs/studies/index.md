# Studies Index

## PDE / Forward-Modeling Studies

### `born-rytov-dt-candidate-preflight` (candidate; active backlog)

- Purpose: Evaluate whether 2D Born/Rytov diffraction tomography is practical
  as an additional inverse-scattering evidence lane for the SRU-Net manuscript.
- Design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`
- Backlog item: `docs/backlog/active/2026-04-29-brdt-candidate-preflight.md`
- Operator validation authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
  with machine-readable contract at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`.
  Downstream BRDT items must consume the operator contract from this report
  rather than re-reading the candidate-lane design.
- Dataset preflight authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
  with smoke dataset and machine-readable manifest under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/`
  (`dataset_manifest.json`, `dry_run_manifest.json`,
  `dry_run_summary.json`, and the
  `dataset/brdt128_sparse_fullview_preflight_{train,val,test}.h5`
  files). The dry-run manifest skeleton mirrors the live schema while
  leaving normalization/SNR unset. Smoke dataset only; not manuscript
  evidence.
- Task-adapter authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md` with
  loader/Born-init/adapter/loss/train/eval surfaces under
  `scripts/studies/born_rytov_dt/` and adapter-readiness sanity
  artifacts at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-task-adapters/fast_dev_run/`
  (`adapter_contract.json`, schema `brdt_adapter_contract_v1`). Adapter
  readiness only; not manuscript evidence.
- Roadmap phase: `candidate-brdt-preflight`
- Scope: physical-target and normalization lock, differentiable Born operator
  validation, synthetic dataset feasibility, task-specific adapters, and a
  four-row decision-support preflight.
- Boundary: additive candidate work only. It does not replace CDI `lines128`,
  does not replace PDEBench CNS, and cannot support manuscript result claims
  without a later roadmap/evidence-package amendment.

### `wavebench-inverse-source-candidate-preflight` (candidate; active backlog)

- Purpose: Evaluate whether WaveBench inverse source reconstruction is
  practical as an additional 2D known-forward-model inverse-wave lane for the
  SRU-Net manuscript.
- Design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_benchmark_design.md`
- Backlog item: `docs/backlog/active/2026-04-29-wavebench-inverse-source-preflight.md`
- Roadmap phase: `candidate-wavebench-inverse-source-preflight`
- Scope: dataset/checkpoint feasibility, native FNO/U-Net baseline inspection,
  exact tensor contract, local adapter feasibility, and forward-model
  reproduction checks for physics-informed variants.
- Boundary: additive candidate work only. It is on equal footing with BRDT, does
  not replace CDI `lines128` or PDEBench CNS, and cannot support manuscript
  result claims without a later roadmap/evidence-package amendment.

### `pdebench-128x128-image-suite` (planned)

- Purpose: Run the amended Roadmap Phase 2 native `128x128` PDEBench image suite covering SWE, Darcy Flow, and 2D Compressible Navier-Stokes.
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- Paper evidence index: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`; this is the cross-study outcome map for completed NeurIPS backlog items, summary authorities, artifact roots, evidence tiers, protocol/cap labels, and claim boundaries.
- Evidence matrix: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`; this is the human-facing master matrix for completed NeurIPS/SRU-Net CDI and CNS model rows, ablation families, generated outputs, summary authorities, and artifact roots.
- Model variant index: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`; this is the machine-readable index keyed by dataset contract, row id, architecture id, training mode, metrics, source backlog item, summary authority, and artifact paths.
- Ablation index: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`; this is the machine-readable index grouping completed backlog outputs by changed factor, fixed contract, row set, artifact root, and current interpretation.
- Preflight summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_preflight.md`; this is the discoverable source for staged-file status, raw HDF5 shapes, axis orders, and available supervision-unit counts.
- Darcy execution plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`; this is the source for the beta `1.0` static-operator contract, strong U-Net/FNO baseline gates, and literature calibration targets.
- 2D CNS design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`; this is the source for official `128x128` 2D_CFD file selection, storage gates, four-field stacking, `history_len=2` primary input, `history_len=1` ablation, trajectory/sample-level split rules, and required `fRMSE_high` shock-capture reporting.
- 2D CNS implementation summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`; this is the discoverable source for the active canonical CNS Hybrid profile `hybrid_resnet_cns` and the skip-add promotion evidence.
- CNS paper-contract decision: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`; this is the discoverable source for the selected bounded capped `history_len=2` headline lane, the locked normalization and training-recipe contract, the locked row roster, the authored-FFNO cutoff/status, and the explicit decision to keep `history_len=3` as adjacent capped context instead of mixing it into the headline table.
- CNS paper row-lock summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`; this is the discoverable source for the accepted bounded capped run roots, the machine-readable locked-row manifest, the audit of excluded adjacent context, and the explicit note that the reused roots remain `capped_decision_support` rather than paper-grade provenance-complete rows.
- CNS paper table/figure bundle summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_table_figure_bundle_summary.md`; this is the discoverable source for the bounded CNS paper bundle outcome, the `1024`-audit fallback versus `512` lock decision, the fixed sample/scaling policy, and the emitted table/figure artifact paths.
- 2D CNS upsampler compare summary: `docs/plans/2026-04-21-hybrid-upsampler-artifact-study-results.md`; this is the discoverable source for the post-skip-add upsampler rerun, including the result that `pixelshuffle` beat the transpose and bilinear CNS-shell decoders on aggregate capped error and was promoted into the canonical CNS Hybrid row.
- Spectral ResNet bottleneck design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`; this is the source for the separate `spectral_resnet_bottleneck_net` model family that keeps the current PDE adapter shell but replaces only the constant-resolution bottleneck with a ResNet-local stack plus a shared factorized spectral residual branch.
- FFNO-close bottleneck compare summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`; this is the discoverable source for the first same-shell three-row CNS bottleneck compare, including the result that `spectral_resnet_bottleneck_base` beat both `hybrid_resnet_cns` and `ffno_bottleneck_base` on the capped 10-epoch slice.
- The same FFNO/bottleneck summary also records the later capped 40-epoch spectral follow-up where `spectral_resnet_bottleneck_base` beat the earlier 40-epoch `hybrid_resnet_base`, `fno_base`, and `unet_strong` rows and links the cross-run comparison PNGs.
- Author FFNO equal-footing summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`; this is the discoverable source for the actual authored `fourierflow` FFNO baseline on the local CNS contract, including source provenance, fresh author-only run roots, and merged cross-run compare artifacts against the fixed local spectral/FNO/U-Net rows.
- Spectral modes-32 compare summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_summary.md`; this is the discoverable source for the manual higher-mode spectral CNS follow-up, including the reused `10`-epoch row, the authoritative fresh `40`-epoch row, the anchored `10`/`40`-epoch sidecars, and the result that modes-32 only improved `fRMSE_mid/high` at `40` epochs while losing aggregate error to the shared `12/12` spectral row.
- Spectral modes-24 convergence summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md`; this is the discoverable source for the fresh paired `80`-epoch shared spectral `12/12` versus `24/24` convergence follow-up on the larger `1024 / 128 / 128` capped CNS slice, including the resolved batch-size record, the emitted convergence audit, and the inconclusive result that both rows were still materially improving at stop time while the shared base row held the better final eval metrics.
- Hybrid-spectral architecture ablation summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`; this is the discoverable source for the fresh same-shell sharing and shared-depth pilots plus the larger-cap finalist confirmation, including the result that the `10`-block shared bottleneck won the `512 / 64 / 64` depth tranche but the shared base row recovered the aggregate lead on the `1024 / 128 / 128` confirmation slice.
- Markov history-1 compare summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`; this is the discoverable source for the controlled `history_len=1` versus frozen `history_len=2` CNS compare, including the audited reference manifest, the missing `40`-epoch hybrid-anchor backfill, the fresh `10`/`40`-epoch pilot runs, and the cross-history compare sidecars.
- History length 3+ compare summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`; this is the discoverable source for the controlled `history_len=3` versus frozen `history_len=2` CNS compare, including the raw eligible-window shrinkage, the fresh `10`/`40`-epoch pilot runs, the anchored cross-history compare sidecars, and the closed `history_len=4` gate after mixed spectral signals.
- Spectral history 4+ compare summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_spectral_history_len4plus_compare_summary.md`; this is the discoverable source for the later spectral-only `history_len=4` and `history_len=5` follow-up, including the inspect proofs, fresh `10`/`40`-epoch pilots, anchored multi-reference compare sidecars, and the conclusion that the spectral row kept improving at `40` epochs while remaining adjacent capped context only.
- Authored-FFNO history-length compare summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md`; this is the discoverable source for the within-model authored-FFNO temporal-context follow-up, including the frozen `history_len=2` reference manifest, the `history_len=3/4/5` inspect proofs, the fresh `40`-epoch authored-FFNO pilots, the anchored multi-reference compare sidecars, the explicit `history_len=4` and `history_len=5` gate decisions, and the result that authored FFNO improved cleanly through `history_len=4` and traded a small aggregate-error regression for further high-band gains at `history_len=5` while remaining adjacent capped context only.
- GNOT CNS compare summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`; this is the discoverable source for the official GNOT host environment, the first fairness-probe row, the fresh paper-default smoke gate, the same-contract paper-default `40`-epoch follow-up, and the anchored compare sidecar against the pinned spectral `40`-epoch row.
- FFNO local-convolution compare summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`; this is the discoverable source for the bounded FFNO-family local-branch follow-up, including the Task 1 inspection audit, the required `40`-epoch FFNO-close backfill, the authoritative `10`/`40`-epoch local-conv rows, the anchored compare sidecars, and the result that local convolution materially improved the repo-local FFNO proxy and beat the capped shared-spectral local row while still trailing the official authored FFNO `40`-epoch row.
- Hybrid-spectral to FFNO parameter-space summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_ffno_parameter_space_summary.md`; this is the discoverable source for the bounded shell-bridge follow-up, including the frozen study matrix, the fresh `down1` and transpose decoder probes, the anchored `10`-epoch sidecar, the closed `40`-epoch promotion gate, and the carry-forward result that the spectral anchor remains the aggregate local shell reference while local-conv stays the stronger repo-local FFNO-family alternative.
- Script: `scripts/studies/run_pdebench_image128_suite.py`
- Bundle builder: `scripts/studies/pdebench_image128/cns_paper_bundle.py`
- Reusable bundle surface: `cns-paper-table-figure-bundle`
- Expected official files: `2D_rdb_NA_NA.h5` (`swe`), `2D_DarcyFlow_beta1.0_Train.hdf5` (`darcy`), and first CNS target `2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5` (`2d_cfd_cns`). The CNS file is staging locally under `/home/ollie/Documents/pdebench-data/2d_cfd_cns/` and is not ready until byte-count, MD5, and schema checks pass.
- Scope: shared data/schema preflight, task-specific split and metric contracts, capped pilot/triage runs where needed, full available training-split Hybrid ResNet/FNO/U-Net benchmark runs, and focused spectral/local ablations where budget permits.
- Boundary: smoke and capped pilot outputs are readiness/triage artifacts only and must not rank models, trigger performance pivots, or satisfy benchmark-performance gates.
- Training recipe guardrail: PDE benchmark-performance rows use the inherited Hybrid ResNet recipe with `ReduceLROnPlateau` scheduler floor no higher than `1e-5` unless a later plan records a justified pre-run override.

### `pdebench-darcy-static-operator-benchmark` (implemented; full benchmark pending)

- Purpose: Implement and run the Darcy Flow beta `1.0` static operator-map member of the PDEBench `128x128` image suite.
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`
- Summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`
- Data file: `/home/ollie/Documents/pdebench-data/darcy/2D_DarcyFlow_beta1.0_Train.hdf5`
- Data contract: `nu` `(10000,128,128)` as one input channel to `tensor` `(10000,1,128,128)` as one target channel; deterministic sample-level split, no time axis, no one-step expansion.
- Strong-baseline rule: local performance interpretation requires Hybrid ResNet, FNO, and `unet_strong`; the tiny smoke U-Net is readiness-only and cannot satisfy the strong-baseline gate.
- Published-context values: PDEBench beta `1.0` U-Net RMSE/nRMSE about `6.4e-3`/`3.3e-2`, FNO about `1.2e-2`/`6.4e-2`; HAMLET/OFormer nRMSE context about `1.40e-2`/`2.05e-2`, with protocol caveats.
- Readiness artifact: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/readiness-cap-20260420T222155Z`
- Full-run budget: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json`
- Boundary: no `/home/ollie/Documents/neurips/` artifacts and no benchmark-performance claim until full available training-split runs complete for `hybrid_resnet_cns`, `fno_base`, and `unet_strong`.

### `pdebench-swe-primary-smoke-gate`

- Purpose: Run the Roadmap Phase 2 smoke/data-contract gate for the PDEBench 2D Shallow Water Equations task.
- Script: `scripts/studies/run_pdebench_swe_smoke.py`
- Official file: `2D_rdb_NA_NA.h5` from the PDEBench `swe` download path or DaRUS datafile `133021`.
- Scope: one-step next-state smoke only, with deterministic trajectory splits, local `err_nRMSE`/`err_RMSE`, and tiny Hybrid ResNet-compatible, FNO, and U-Net runs or explicit blockers. Smoke metrics are readiness/sanity artifacts, not benchmark-performance evidence.
- Output artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/`.
- Boundary: this is a Phase 2 prerequisite smoke gate for the SWE task and now feeds the broader PDEBench `128x128` image-suite plan; it is not full PDE training, rollout evaluation, ablations, CDI regeneration, or paper-facing artifact assembly.

### `pdebench-swe-longer-execution`

- Purpose: Run the Roadmap Phase 2 longer one-step execution gate for PDEBench 2D Shallow Water Equations after the smoke gate authorizes the primary path.
- Script: `scripts/studies/run_pdebench_swe_longer.py`
- Official file: `2D_rdb_NA_NA.h5` from the PDEBench `swe` download path or DaRUS datafile `133021`, pinned locally by SHA256 `28f0c33723d70eebb420fc170e94b675c18e032fb697dcef080e114ca9645e3a`.
- Horizon: one-step next-state prediction only.
- Output artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-longer-execution/`.
- Local model profiles: `hybrid_resnet_base`, `fno_base`, `unet_base`, with `hybrid_resnet_spectral_reduced` and `hybrid_resnet_local_reduced` ablations gated behind primary viability.
- Reproducibility guard: new longer runs must provide `training_seed=20260420` through the run budget or `--training-seed`; any live `runs/<run_id>/logs/longer.pid` marker is rejected even if stale `longer.exit_code` evidence exists, any PID marker without exit-code evidence is treated as incomplete-run evidence, fresh starts remove stale per-run completion markers, and freshness validation rejects invocation metadata whose `output_root` does not resolve to the selected run root.
- Current selected-run caveat: run `20260420T115509.961336393Z` predates training-seed provenance and is documented only as unseeded observed SWE pivot evidence.
- Boundary: this is longer Phase 2 execution only, not CDI, OpenFWI fallback execution, rollout evaluation, `256x256` scaling, or paper-facing artifact assembly.

### `openfwi-flatvel-a-fallback-smoke-gate`

- Purpose: Run the Roadmap Phase 2 fallback smoke/data-access gate for OpenFWI FlatVel-A. This study is currently deferred as an optional fallback or adjacent inverse-wave extension while the PDEBench `128x128` image-suite plan is viable.
- Script: `scripts/studies/run_openfwi_flatvel_a_smoke.py`
- Required shards: `data1.npy`, `model1.npy`, `data49.npy`, and `model49.npy` under an external or ignored FlatVel-A shard root supplied with `--data-root`.
- Output artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/`.
- Decision boundary: produce exactly one smoke-gate readiness decision: proceed to longer OpenFWI execution, block for storage/data/human decision, or reject fallback as operationally nonviable. Smoke metrics are sanity/provenance artifacts only and must not rank models or decide benchmark performance.
- Local profiles: `hybrid_resnet_smoke` and `unet_smoke`, with optional `fno_smoke` and optional official InversionNet probe through `--official-openfwi-repo`.
- Boundary: this is a fallback smoke gate only, not full 43 GB OpenFWI training, CDI regeneration, `256x256` scaling, PDEBench SWE rescue, or paper-facing artifact assembly.

## Grid-Lines Studies

### `lines_256` dataset note

- Purpose: Document the repo-local `N=256` lines dataset alias used for single-dataset architecture experiments.
- Document: `docs/studies/lines_256_dataset.md`
- Runbook profile name: `custom_npz_pair_n256`
- Storage note: the current loop still reads the pair from `outputs/...`, but persistent datasets should not live under cleanup-prone `outputs/` long term.
- Probe scaling contract: `pad_preserve` for the working `lines_256` pair; `pad_extrapolate` remains available as an explicit alternative mode.
- Preferred use: `scripts/studies/run_lines_256_arch_experiment.py` for fixed-budget experiments or explicit diagnostic-mode runbook invocations when you want lines-only `N=256` runs.

### `lines_256` architecture-improvement loop

- Purpose: Fix the exact autonomous loop for `lines_256` architecture experiments, including fresh baseline generation at session start, the untracked TSV ledger path, the session-local champion rule, and the keep/discard reset behavior.
- Document: `docs/studies/lines_256_arch_improvement_loop.md`
- Baseline inheritance: inherits the project Hybrid ResNet baseline from `docs/model_baselines.md` and overrides only study-specific items such as the `epochs=20` budget unless the loop doc says otherwise.
- Rollout status: legacy path retained while the v2 controller path is validated in parallel.
- Run-checkout note: this workflow uses DSL-level git rollback/checkpoint behavior; the study doc requires a dedicated run checkout and explains how that relates to normal branch work.
- Workflow: `workflows/agent_orchestration/lines_256_arch_improvement_session_loop.yaml`
- Encapsulated variant: `workflows/agent_orchestration/lines_256_arch_improvement_session_loop_v2_call.yaml`
- Iteration subworkflow: `workflows/library/lines_256_arch_improvement_iteration.yaml`
- Provider prompts:
  - `prompts/workflows/lines_256_arch_improvement/experiment_step.md`
  - `prompts/workflows/lines_256_arch_improvement/debug_crash.md`
- Ledger path: `state/lines_256_arch_improvement/results.tsv`
- Baseline rule: regenerate from the current `HEAD` at the beginning of each session using the default control and fixed budget in the loop document
- Thin wrapper: `scripts/studies/run_lines_256_arch_experiment.py`
- Comparison gallery: `outputs/lines_256_arch_improvement/comparison_pngs/<session_id>/`

### `lines_256` controller loop (v2)

- Purpose: Run the parallel controller-based `lines_256` path that keeps session mechanics in Python while the legacy YAML loop remains available.
- Document: `docs/studies/lines_256_controller_loop.md`
- Thin wrapper workflow: `workflows/agent_orchestration/lines_256_session_controller.yaml`
- Controller script: `scripts/studies/lines_256_session_controller.py`
- Session state root: `state/lines_256_arch_improvement_v2/sessions/<session_id>/`
- Session outputs root: `outputs/lines_256_arch_improvement_v2/sessions/<session_id>/`

### `hybrid-resnet-mode-skip-sweep`

- Purpose: Run staged `hybrid_resnet` search loops over `mode x skip x width` (Stage A) and later structural axes (Stages B-E) with strict stage/substage guardrails, promotion-source validation, seed-rerank aggregation, and retention-tier cleanup.
- Script: `scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py`
- Stage IDs: `A|B|C|D|E` with `substage_id=none` for `A/B/E`, `C1|C2` for stage `C`, and `D1|D2|D3|D4` for stage `D`.
- Stage-D branch-capacity axes:
  - canonical scale knobs: `--encoder-conv-hidden-scale-values` and `--encoder-spectral-hidden-scale-values`
  - legacy aliases: `--encoder-conv-hidden-values` / `--encoder-spectral-hidden-values` (diagnostic compatibility only)
  - summary/manifest provenance includes configured scales plus deterministic resolved-width metadata (`encoder_*_resolved_*` fields).
- Seed-rerank aggregation mode:
  - inputs: `--aggregate-seed-rerank-root` + `--source-summary`
  - outputs: `--emit-robust-promotion-summary` and `--emit-stage-anchor-summary`
  - coverage gate: boundary candidates (`top-K + next 2`) must include seeds `{3,11,17}`.
- Output/artifact contract:
  - run root: `invocation.json`, `invocation.sh`, `sweep_manifest.json`, `summary.csv`, `summary.md`
  - per-run: `runs/<run_id>/metrics.json` and `runs/<run_id>/cleanup_report.json`
  - summary rows persist confounder controls (`probe_mask_enabled`, `torch_mae_pred_l2_match_target`) and stage identity (`stage_id`, `substage_id`).
- Promotion governance gates:
  - rank feasible candidates with amplitude SSIM as the primary objective and `train_wall_time_sec` as the efficiency objective.
  - enforce feasibility before promotion: `phase_ssim_drop_vs_baseline <= max_phase_ssim_drop` (default `0.03`), train-time and model-parameter limits, and inference SLA (`<=60s` at `N=128`, `<=240s` at `N=256`).
  - require robustness validation before every promotion event: boundary candidate set (`top-K + next 2`) reranked across seeds `{3,11,17}` and promoted by median Pareto rank.
  - Stage A is not complete until this `hybrid-resnet-mode-skip-sweep` index entry is present and verified.
- Stop/go diagnostics:
  - pause-and-diagnose when two consecutive stages deliver `<1%` median relative gain and the rerank confidence interval overlaps zero.
  - pause-and-diagnose when all new-stage candidates regress on amplitude SSIM at `N=256` and the same direction appears in `N=128` robustness summaries.
  - before halting an axis, run one bounded rescue mini-sweep; if still failing, pause expansion on that axis and carry at least one hedge candidate forward under low budget.

Runbook CLI (Stage A full N=128 example):

```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --modes 12,16,24,32,48 \
  --skip-values off,on \
  --widths 32,48,64 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1,fly001_external_n128_top_bottom_v1 \
  --fly001-external-train-npz <path/to/fly001_n128_train_top_half.npz> \
  --fly001-external-test-npz <path/to/fly001_n128_test_bottom_half.npz> \
  --epochs-n128 20 \
  --top-k-n256 6 \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n128 2700 \
  --max-inference-seconds-n128 60 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
  --output-root outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221 \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

Runbook CLI (seed-rerank aggregation example):

```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id A \
  --aggregate-seed-rerank-root outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/seed_rerank \
  --source-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/summary.csv \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --top-k-n256 6 \
  --emit-stage-anchor-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/stage_anchor_summary.csv \
  --emit-robust-promotion-summary outputs/hybrid_resnet_mode_skip_sweep_full_n128_20260221/promotion/summary_seed_robust.csv
```

### `nersc-scan807-cameraman-ptychovit-hybrid-orchestration`

- Purpose: Run checkpoint-restored `pinn_ptychovit` inference on `scan807` and `cameraman256`, train `pinn_hybrid_resnet` on cameraman top/bottom-half (`N=128`, 40 epochs), run checkpoint-reuse hybrid inference across both full datasets, and aggregate per-dataset metrics/visuals.
- Script: `scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py`
- Canonical PtychoViT checkpoint for this study: `datasets/run145/best_model.pth` (required; do not substitute ad-hoc `tmp/ptychovit_initial_*` checkpoints).
- Position reassembly backend for external hybrid inference is pinned to `shift_sum` (`--position-reassembly-backend shift_sum` only).
- N=128 prep semantics are configurable via `--downsample-policy`:
  - `bin-crop` (default): diffraction is block-binned; `objectGuess`/`probeGuess` are center-cropped; coordinates remain in the same pixel frame.
  - `crop-bin`: diffraction is center-cropped; `objectGuess`/`probeGuess` are block-binned; coordinates are scaled by `1/factor`.
- Multimode probe collapse policy is configurable via `--probe-mode-policy`:
  - `incoherent_aggregate` (default): single-probe collapse with incoherent amplitude aggregation.
  - `first_mode`: compatibility fallback that uses only mode 0.
- Core helpers:
  - `scripts/studies/nersc_pair_adapter.py`
  - `scripts/studies/prepare_nersc_hybrid_dataset.py`
  - `scripts/studies/nersc_orchestration.py`
  - `scripts/studies/hybrid_checkpoint_inference.py`

CLI entry point (full command):

```bash
python scripts/studies/runbooks/run_nersc_scan807_cameraman_study.py \
  --scan807-dp /home/ollie/Downloads/nersc/testdata/scan807_dp.hdf5 \
  --scan807-para /home/ollie/Downloads/nersc/testdata/scan807_para.hdf5 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --ptychovit-checkpoint datasets/run145/best_model.pth \
  --half top \
  --downsample-policy bin-crop \
  --probe-mode-policy incoherent_aggregate \
  --position-reassembly-backend shift_sum \
  --output-dir outputs/nersc_scan807_cameraman_study \
  --seed 3
```

### `nersc-scan807-cameraman-ptychovit-hybrid-orchestration-n256-no-downsample`

- Purpose: Companion to the `N=128` orchestration that keeps both model arms on `N=256` end-to-end (no 256->128 conversion), while preserving the same staged workflow and strict `shift_sum` reassembly policy.
- Script: `scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n256.py`
- Smoke/full rule: use the same command, changing only `--epochs`:
  - smoke: `--epochs 5`
  - full: `--epochs 40`
- Core helpers:
  - `scripts/studies/prepare_nersc_hybrid_dataset.py` (explicit no-downsample path when `target_n == source_n`)
  - `scripts/studies/nersc_orchestration.py` (`target_n` + `epochs` threaded through training/inference)

CLI entry point (smoke example):

```bash
python scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n256.py \
  --scan807-dp /home/ollie/Downloads/nersc/testdata/scan807_dp.hdf5 \
  --scan807-para /home/ollie/Downloads/nersc/testdata/scan807_para.hdf5 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --ptychovit-checkpoint datasets/run145/best_model.pth \
  --half top \
  --probe-mode-policy incoherent_aggregate \
  --epochs 5 \
  --output-dir outputs/nersc_scan807_cameraman_study_n256_no_downsample_smoke \
  --seed 3
```

### `nersc-scan807-cameraman-n128-factorial-probe-mask-mae-downsample`

- Purpose: Run a sequential `N=128` factorial sweep over:
  - probe mask mode: `off`, `on_soft`, `on_hard`
  - Torch MAE prediction-L2 matching: `off`, `on`
  - downsample policy: `bin-crop`, `crop-bin`
- Script: `scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n128_factorial.py`
- Matrix size: `3 * 2 * 2 = 12` runs per sweep.
- Epoch policy: fixed for the whole sweep (`--epochs 20` or `--epochs 40`), not a matrix axis.
- Output contract:
  - `factorial_manifest.json` at study root
  - per-run outputs under `runs/<run_id>/`

Runbook CLI (full example):

```bash
python scripts/studies/runbooks/run_nersc_scan807_cameraman_study_n128_factorial.py \
  --scan807-dp /home/ollie/Downloads/nersc/testdata/scan807_dp.hdf5 \
  --scan807-para /home/ollie/Downloads/nersc/testdata/scan807_para.hdf5 \
  --cameraman-dp /home/ollie/Downloads/nersc/data/cameraman256_dp.hdf5 \
  --cameraman-para /home/ollie/Downloads/nersc/data/cameraman256_para.hdf5 \
  --ptychovit-checkpoint datasets/run145/best_model.pth \
  --half top \
  --epochs 40 \
  --soft-mask-sigma 1.0 \
  --output-root outputs/nersc_scan807_cameraman_study_n128_factorial_$(date +%Y%m%d_%H%M%S) \
  --seed 3
```

Collation script:

```bash
python scripts/studies/collate_nersc_n128_factorial_results.py \
  --factorial-root outputs/<factorial_run_dir> \
  --shared-dir outputs/<factorial_run_dir>/comparison_bundle
```

Collation outputs:
- `comparison_bundle/metrics_summary.csv`
- `comparison_bundle/metrics_summary.md`
- `comparison_bundle/shared_pngs/{run_id}__dataset-{dataset}__compare_amp_phase.png`

### `grid-lines-external-fly001-n128-top-train-full-test-e40`

- Purpose: Run external-raw `fly001` study at `N=128` with top-half train and full-object test (no additional subsampling), comparing Torch `cnn` and `hybrid_resnet`.
- Script: `scripts/studies/runbooks/grid_lines_external_fly001_n128_top_train_full_test_e40.sh`
- Output directory: `outputs/grid_lines_external_fly001_n128_top_train_full_test_e40_seed3_cnn_hybrid_resnet`
  - Rerun output: `outputs/grid_lines_external_fly001_n128_top_train_full_test_e40_seed3_cnn_hybrid_resnet_rerun_20260216_213242_pty`
- Dataset inputs:
  - `datasets/fly001_128/fly001_128_top_half_converted.npz`
  - `datasets/fly001_128/fly001_128_full_test_converted.npz`
  - `datasets/fly001_128/manifest.json`
- Position reassembly strategy (external mode):
  - Default is `auto` (`--torch-position-reassembly-backend auto`)
  - `auto` prefers `shift_sum` and falls back to `batched` on TF OOM.
  - Use explicit batched mode only as an opt-in override:
    - `--torch-position-reassembly-backend batched`
    - `--torch-position-reassembly-batch-size 32`

CLI entry point (full command):

```bash
bash scripts/studies/runbooks/grid_lines_external_fly001_n128_top_train_full_test_e40.sh
```

### `grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3`

- Purpose: Preserve the strongest currently known historical `N=128` grid-lines `pinn_hybrid_resnet` result as a durable decision-support reference, even though it does not satisfy the paper-grade anchor provenance bar.
- Status: decision-support only; do not treat as the NeurIPS paper anchor.
- Historical artifact root (legacy backup checkout): `/home/ollie/trash/tmp.bak/PtychoPINN/outputs/grid_lines_n128_compare_padex_lr2e4_plateau_e40_seed3`
- Key recovered artifacts:
  - `/home/ollie/trash/tmp.bak/PtychoPINN/outputs/grid_lines_n128_compare_padex_lr2e4_plateau_e40_seed3/runs/pinn_hybrid_resnet/metrics.json`
  - `/home/ollie/trash/tmp.bak/PtychoPINN/outputs/grid_lines_n128_compare_padex_lr2e4_plateau_e40_seed3/runs/pinn_hybrid_resnet/history.json`
  - `/home/ollie/trash/tmp.bak/PtychoPINN/outputs/grid_lines_n128_compare_padex_lr2e4_plateau_e40_seed3/visuals/amp_phase_pinn_hybrid_resnet.png`
- Recovered contract:
  - `N=128`, `gridsize=1`
  - synthetic lines, `set_phi=True`
  - custom Run1084 probe
  - `probe_scale_mode=pad_extrapolate`
  - `probe_smoothing_sigma=0.5`
  - `nimgs_train=2`, `nimgs_test=2`
  - `nphotons=1e9`
  - `seed=3`
  - `torch_epochs=40`
  - `torch_learning_rate=2e-4`
  - `torch_scheduler=ReduceLROnPlateau`
  - `torch_plateau_factor=0.5`
  - `torch_plateau_patience=2`
  - `torch_plateau_min_lr=1e-4`
  - `torch_plateau_threshold=0.0`
  - `torch_loss_mode=mae`
  - `torch_mae_pred_l2_match_target=off`
  - `torch_output_mode=real_imag`
  - `probe_mask=off`
  - `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`
- Best recovered metrics:
  - amp/phase MAE: `0.039216023 / 0.06845270933163407`
  - amp/phase SSIM: `0.9779267790199384 / 0.9903324391137435`
  - final train/val epoch loss: `0.03572520241141319 / 0.043088942766189575`
- Provenance caveat:
  - this root lives outside the active repo in a backup checkout;
  - no durable child `runs/pinn_hybrid_resnet/invocation.json` was recovered in the historical root;
  - the contract above is reconstructed from the historical metrics/history artifacts plus the contemporaneous wrapper command documented in the legacy workflow docs.

### `grid-lines-n128-ffno-vs-hybrid-resnet-best-contract`

- Purpose: Run and preserve one repaired stable `ffno` versus
  `hybrid_resnet` pair on the study-indexed
  `grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3` contract without
  changing the CDI dataset, probe, scheduler, or output contract.
- Status: completed on `2026-04-29`; the stable root is ready as prerequisite
  CDI evidence for later `lines128` paper packaging.
- Preflight:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
- Summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
- Output directory:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
- Artifact contract notes:
  - wrapper root includes `metrics.json`, `metrics_table.csv`,
    `metrics_table.tex`, `metrics_table_best.tex`, and the compare visuals
  - both `runs/pinn_hybrid_resnet/` and `runs/pinn_ffno/` include
    `invocation.json`, `invocation.sh`, metrics/history, model, and randomness
    provenance
  - the row-level `invocation.*` files were reconstructed during the repair
    pass from the fixed wrapper contract after the original compare finished,
    so they should be read as backfilled provenance rather than original
    runner-emitted start-time records
- Models:
  - `pinn_hybrid_resnet`
  - `pinn_ffno`
- Final metrics:
  - `pinn_hybrid_resnet` amp/phase MAE `0.026939474 / 0.072063477`, amp/phase
    SSIM `0.988114297 / 0.994739987`
  - `pinn_ffno` amp/phase MAE `0.062772475 / 0.082838669`, amp/phase SSIM
    `0.934830340 / 0.981591519`
- Boundary: this is the prerequisite FFNO-versus-Hybrid CDI row pair for later
  `lines128` paper packaging, not the final four-row paper benchmark.

### `grid-lines-n128-paper-benchmark-harness-readiness`

- Purpose: freeze the later `lines128` paper benchmark harness contract,
  capture the selected FNO comparator and row roster, and emit readiness-only
  schema/collation artifacts without launching the full multi-row benchmark.
- Status: harness-ready on `2026-04-29`; full benchmark still unlaunched.
- Harness preflight:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_preflight.md`
- Harness summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_harness_summary.md`
- Decision manifest:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/preflight/benchmark_decisions.json`
- Readiness bundle:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-harness/validation/readiness_only_preflight`
- Minimum supported subset:
  - `pinn_hybrid_resnet`
  - `pinn`
  - `pinn_fno_vanilla`
- Additional supported rows:
  - `pinn_spectral_resnet_bottleneck_net`
  - `pinn_ffno`
- Boundary:
  - the readiness bundle is intentionally `benchmark_incomplete`
  - the later execution item still owns fresh row runs, paper-grade visuals,
    and final table completeness

### `grid-lines-n128-complete-paper-benchmark`

- Purpose: publish the authoritative six-row `N=128` CDI paper bundle under
  the frozen `fno_vanilla` comparator, fixed `seed=3`, fixed samples `{0,1}`,
  and shared visual-scale contract.
- Status: completed on `2026-04-30`; the tmux-backed repaired root is the
  authoritative `paper_complete` Lines128 CDI benchmark bundle.
- Design:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
- Summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
- Authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Repair note:
  the earlier roots
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T134500Z`
  and
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T141325Z_repair`
  are superseded because promoted-row launcher-completion recovery and the
  wrapper tmux invocation contract were repaired afterward.
- Accepted rows:
  - `baseline`
  - `pinn`
  - `pinn_hybrid_resnet`
  - `pinn_fno_vanilla`
  - `pinn_spectral_resnet_bottleneck_net`
  - `pinn_ffno`
- Row provenance:
  - minimum-subset promoted rows:
    `baseline`, `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla`
  - fresh spectral row:
    `pinn_spectral_resnet_bottleneck_net` reran in the superseded complete
    root and was promoted into the repaired authoritative root
  - prerequisite FFNO row:
    `pinn_ffno` promoted from the fixed-contract FFNO-vs-Hybrid prerequisite
    root with repaired row-local completion proof rebuilt from durable
    promoted-source wrapper logs
- Bundle artifacts:
  - `metrics.json`
  - `metric_schema.json`
  - `model_manifest.json`
  - `metrics_table.csv`
  - `metrics_table.tex`
  - `metrics_table_best.tex`
  - `visuals/compare_amp_phase.png`
  - `visuals/frc_curves.png`
- Boundary:
  - this is the authoritative complete CDI paper bundle for Lines128
  - prerequisite pair evidence and the four-row minimum subset remain distinct,
    preserved evidence roots and should not be mistaken for the final six-row
    package

### `grid-lines-n128-supervised-ffno-extension`

- Purpose: publish the adjacent same-contract `FFNO + PINN` versus
  `FFNO + supervised` comparison under the frozen `lines128` `N=128`,
  `seed=3`, fixed-sample, shared-visual-scale contract without rewriting the
  primary six-row CDI benchmark claim.
- Status: completed on `2026-04-30`; the extension root is
  `paper_complete`.
- Summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_summary.md`
- Execution authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_execution_authority.md`
- Authoritative extension root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z`
- Included bundle rows:
  - `pinn_ffno`
  - `supervised_ffno`
- Reference-only same-contract supervised CNN row:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-minimum-paper-table/runs/minimum_subset_20260430T084339Z`
- Comparison audit:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/execution/supervised_ffno_parity_audit.json`
- Main result:
  the corrected supervised FFNO rerun executed successfully under the locked
  contract and produced a truthful `paper_complete` adjacent extension root,
  but not exact parity with the preserved `pinn_ffno` comparator. The rebuilt
  comparison audit records
  `comparison_outcome: non_identical_same_contract_comparison`.
- Boundary:
  - this is adjacent evidence for the Lines128 CDI lane
  - it does not replace the preserved six-row primary CDI benchmark root
- the supervised CNN minimum-subset evidence remains a referenced sibling,
  not a rerun or silently merged extension row

### `grid-lines-n128-hybrid-spectral-ffno-parameter-space`

- Purpose: run a bounded same-contract CDI bridge study between the current
  Hybrid and spectral shell anchors and FFNO-adjacent architectural changes.
- Status: completed on `2026-04-30` with closeout verification refreshed on
  `2026-05-01`; the output remains decision-support only with no paper
  promotion.
- Preflight:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_hybrid_spectral_ffno_parameter_space_preflight.md`
- Summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_hybrid_spectral_ffno_parameter_space_summary.md`
- Artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-hybrid-spectral-ffno-parameter-space-cdi`
- Reused anchors:
  - `pinn_hybrid_resnet`
  - `pinn_spectral_resnet_bottleneck_net`
  - `pinn_ffno`
- Fresh bridge rows:
  - `pinn_spectral_resnet_bottleneck_ds1`
  - `pinn_spectral_resnet_bottleneck_linear_decoder`
  - `pinn_hybrid_resnet_ffno_bottleneck`
- Main result:
  no fresh bridge row displaced the existing CDI anchors. The DS1 shell created
  a phase-leaning trade with weaker amplitude fidelity, the linear-decoder
  bridge regressed badly, and the FFNO bottleneck did not improve the Hybrid
  anchor.
- Boundary:
  - CDI-only decision-support evidence
  - does not change the current paper-grade `lines128` complete-table
    authority
  - does not promote any fresh row into manuscript claim territory

### `grid-lines-n128-hybrid-resnet-encoder-fusion-variants`

- Purpose: run a bounded same-contract CDI ablation over Hybrid ResNet
  encoder-fusion controls (per-block LayerScale and per-block branch gates)
  without reopening the six-row paper bundle.
- Status: completed on `2026-05-02`; the output remains decision-support only
  with no paper-grade promotion. The authoritative fresh rerun is
  `runs/encoder_fusion_rerun_20260502T121829Z`, where each mandatory fresh row
  now has true row-local launch evidence, checkpoints, and Lightning logs.
- Plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/execution_plan.md`
- Summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_encoder_fusion_variants_summary.md`
- Artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-hybrid-resnet-encoder-fusion-variants/`
- Reused anchor:
  - `pinn_hybrid_resnet`
- Fresh rows (per-block scalars only):
  - `pinn_hybrid_resnet_encoder_layerscale`
  - `pinn_hybrid_resnet_encoder_branch_gated`
  - `pinn_hybrid_resnet_encoder_branch_gated_layerscale`
- Optional deferred row:
  - `pinn_hybrid_resnet_encoder_fusion_norm`
- Boundary:
  - CDI-only decision-support evidence
  - does not replace the paper-grade `pinn_hybrid_resnet` anchor
  - does not replace the six-row `lines128` CDI headline bundle
  - shared-across-encoder-block scalar placement is a distinct future architecture axis

### `grid-lines-n128-hybrid-resnet-skip-residual-ablation`

- Purpose: run a bounded same-contract CDI ablation over Hybrid decoder
  skip-fusion mode and bottleneck residual-scale mode without reopening the
  six-row paper bundle.
- Status: completed on `2026-05-01`; the output remains decision-support only
  with no paper-grade promotion.
- Plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation/execution_plan.md`
- Summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md`
- Artifact root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation`
- Reused anchor:
  - `pinn_hybrid_resnet`
- Fresh rows:
  - `pinn_hybrid_resnet_skip_add`
  - `pinn_hybrid_resnet_residual_fixed`
  - `pinn_hybrid_resnet_skip_add_residual_fixed`
- Optional deferred row:
  - `pinn_hybrid_resnet_skip_gated_add`
- Main result:
  skip-add is the clearest phase-oriented trade, fixed residual scale is the
  clearest amplitude-oriented trade, and the combined row does not beat the
  simpler single-factor variants.
- Boundary:
  - CDI-only decision-support evidence
  - does not replace the paper-grade `pinn_hybrid_resnet` anchor
  - does not replace the six-row `lines128` CDI headline bundle
  - leaves the optional gated-add follow-up deferred for later bounded work

### `grid-lines-n64-pinn-hybrid-resnet-e20`

- Purpose: Run `N=64` grid-lines with `pinn` (TF) and `pinn_hybrid_resnet` (Torch) at `20` epochs, then render combined visuals.
- Script: `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20/run_study.sh`
- Output directory: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet`
- Invocation artifacts:
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/invocation.json`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/invocation.sh`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/runs/pinn_hybrid_resnet/invocation.json`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/runs/pinn_hybrid_resnet/invocation.sh`

CLI entry points (full commands):

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 \
  --gridsize 1 \
  --output-dir outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet \
  --models pinn \
  --nimgs-train 2 \
  --nimgs-test 1 \
  --nphotons 1e9 \
  --nepochs 20 \
  --batch-size 16 \
  --seed 3 \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --set-phi
```

```bash
python scripts/studies/grid_lines_torch_runner.py \
  --output-dir outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet \
  --architecture hybrid_resnet \
  --train-npz outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/datasets/N64/gs1/train.npz \
  --test-npz outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet/datasets/N64/gs1/test.npz \
  --N 64 \
  --gridsize 1 \
  --epochs 20 \
  --batch-size 16 \
  --infer-batch-size 16 \
  --learning-rate 2e-4 \
  --scheduler ReduceLROnPlateau \
  --plateau-factor 0.5 \
  --plateau-patience 2 \
  --plateau-min-lr 1e-4 \
  --plateau-threshold 0.0 \
  --seed 3 \
  --optimizer adam \
  --weight-decay 0.0 \
  --beta1 0.9 \
  --beta2 0.999 \
  --torch-loss-mode mae \
  --output-mode real_imag \
  --probe-source custom \
  --fno-modes 12 \
  --fno-width 32 \
  --fno-blocks 4 \
  --fno-cnn-blocks 2 \
  --torch-logger mlflow
```

```bash
python - <<'PY'
from pathlib import Path
from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals

out = Path("outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed3_pinn_hybrid_resnet")
render_grid_lines_visuals(out, order=("gt", "pinn", "pinn_hybrid_resnet"))
print("Rendered visuals under", out / "visuals")
PY
```

### `grid-lines-n64-pinn-only-retry1-e20-seed13`

- Purpose: Rebuild `pinn` recon for `N=64` (`seed=13`, `20` epochs), reuse an existing `pinn_hybrid_resnet` recon, then regenerate merged comparison metrics/visuals in one output directory.
- Script: `.artifacts/studies/grid_lines_n64_pinn_hybrid_resnet_e20_retry1/finish_from_completed_pinn.sh`
- Output directory: `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed13_pinn_only_retry1`
- Invocation artifacts:
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed13_pinn_only_retry1/invocation.json`
  - `outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed13_pinn_only_retry1/invocation.sh`

CLI entry points (full commands):

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 \
  --gridsize 1 \
  --output-dir /home/ollie/Documents/tmp/PtychoPINN/outputs/grid_lines_n64_compare_padex_lr2e4_plateau_e20_seed13_pinn_only_retry1 \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --models pinn,pinn_hybrid_resnet \
  --reuse-existing-recons \
  --seed 13 \
  --nimgs-train 2 \
  --nimgs-test 1 \
  --nphotons 1e9 \
  --nepochs 20 \
  --batch-size 16 \
  --nll-weight 0 \
  --mae-weight 1 \
  --realspace-weight 0 \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --set-phi \
  --torch-epochs 20 \
  --torch-batch-size 16 \
  --torch-learning-rate 2e-4 \
  --torch-scheduler ReduceLROnPlateau \
  --torch-plateau-patience 2 \
  --torch-plateau-factor 0.5 \
  --torch-plateau-min-lr 1e-4 \
  --torch-plateau-threshold 0 \
  --torch-output-mode amp_phase \
  --torch-loss-mode poisson \
  --torch-grad-clip 0 \
  --torch-grad-clip-algorithm norm \
  --torch-resnet-width 64
```
