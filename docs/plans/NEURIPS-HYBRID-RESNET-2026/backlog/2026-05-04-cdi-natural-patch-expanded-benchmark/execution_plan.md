# CDI Natural-Patch Expanded Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the first standalone expanded-object CDI benchmark on the locked `natural_patches128_fixedprobe_v1` dataset, publish a single-seed natural-patch bundle plus summary, and keep every existing `lines128` authority unchanged.

**Architecture:** Start from the already checked-in natural-patch harness and only patch the exact surfaces needed to preserve the frozen three-way split, row roster, provenance, and bundle contract. Treat this as Phase `3.3i` execution and publication work, not as a fresh framework build and not as a reopening of the `lines128` benchmark authority.

**Tech Stack:** PATH `python`; `ptycho311` for long-running launches; existing CDI/Torch/TF study surfaces under `scripts/studies/`; pytest; `compileall`; tmux; git-ignored `.artifacts/data/` and `.artifacts/work/`.

---

## Planning Inputs Read

- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/workflows/pytorch.md`
- `docs/model_baselines.md`
- `docs/steering.md`
- `docs/backlog/index.md`
- `docs/studies/index.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/execution_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-05-04-cdi-natural-patch-expanded-benchmark/selected-item-context.md`
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark-plan-review.json`
- `scripts/studies/cdi_natural_patch_benchmark.py`
- `scripts/studies/run_cdi_natural_patch_benchmark.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/grid_lines_compare_wrapper.py`
- `scripts/studies/metrics_tables.py`
- `scripts/studies/paper_provenance.py`
- `tests/studies/test_cdi_natural_patch_benchmark.py`
- `tests/studies/test_cdi_natural_patch_dataset.py`

## Selected Backlog Objective

- Execute backlog item `2026-05-04-cdi-natural-patch-expanded-benchmark`.
- Consume the completed dataset authority `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md`.
- Run the approved natural-patch CDI row family on the frozen dataset `natural_patches128_fixedprobe_v1`.
- Publish the benchmark root under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/`.
- Write the durable summary `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_expanded_benchmark_summary.md`.

## Scope And Explicit Non-Goals

### In Scope

- Consume the locked dataset root `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/` exactly as written.
- Run exactly this row roster where protocol-compatible:
  - `baseline`
  - `pinn`
  - `pinn_hybrid_resnet`
  - `pinn_fno_vanilla`
  - `pinn_ffno`
  - `pinn_neuralop_uno`
- Preserve the frozen `8000 / 1000 / 1000` train/validation/test split and use the full `8000` training objects after holdout.
- Preserve the fixed probe lineage, single-seed contract, metric schema, and fixed-sample visual policy.
- Emit standalone JSON, CSV, TeX, manifest, and visual artifacts for the natural-patch bundle.
- Update discoverability surfaces so this benchmark is visible as an expanded-object CDI lane that remains separate from `lines128`.

### Explicit Non-Goals

- Do not regenerate, overwrite, or mutate `natural_patches128_fixedprobe_v1`.
- Do not alter split membership, source-image partitioning, probe preprocessing, object-count cap, or diffraction contract.
- Do not rerun or rewrite the completed `lines128` six-row table or the append-only U-NO extension.
- Do not add `pinn_spectral_resnet_bottleneck_net`, `supervised_ffno`, or `supervised_neuralop_uno` to this natural-patch run.
- Do not expand this item into multi-seed aggregation, new source corpora, hyperparameter sweeps, or roadmap Phase 4/5 work.
- Do not create manuscript-facing outputs under `/home/ollie/Documents/neurips/`.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not force this work through `grid_lines_compare_wrapper.py` unless a specific shared helper is missing and the narrower harness path cannot satisfy the contract.

## Binding Constraints, Claim Boundaries, And Prerequisite Status

- This is Roadmap Phase `3.3i` only: standalone expanded-object CDI evidence after the frozen natural-patch dataset exists.
- The result widens CDI object-distribution evidence beyond `lines128`; it does not replace the `lines128` claim authority.
- Steering is binding:
  - preserve apples-to-apples comparisons;
  - do not silently relax fairness or split rules;
  - if a row cannot satisfy the protocol, record `blocked` or `not_protocol_compatible` explicitly instead of drifting the contract.
- The result must state the scikit-image-derived source boundary and the `<= 10_000` object cap, and must not claim broader natural-image generalization than that locked source corpus supports.
- `progress_ledger.json` shows early tranches complete and no global blocked tranche. Phase 2 PDE work remains open, but that is not a blocker for this Phase 3 item. This benchmark is allowed parallel CDI work and must not be presented as satisfying Phase 2 PDE evidence.
- Prerequisite authority already exists and must be treated as immutable input:
  - `cdi_natural_patch_fixedprobe_dataset_summary.md`
  - `lines128_paper_benchmark_summary.md`
  - `lines128_supervised_equivalent_rows_summary.md`
  - `lines128_uno_table_extension_summary.md`
  - `docs/backlog/index.md`
  - `evidence_matrix.md`
- PyTorch policy remains binding: use PATH `python`, and call `update_legacy_dict(params.cfg, config)` before data loading or legacy-module access.
- Long-running commands remain under implementation ownership until terminal success or recoverable failure handling is complete. Use tmux, activate `ptycho311`, track the launched PID exactly, do not reuse a live output root, and require both `exit_code == 0` and fresh required artifacts before treating a run as complete.
- Normal import, path, environment, schema, or test-harness failures are not sufficient to mark the item `BLOCKED`. Diagnose, patch narrowly, and rerun first.
- Reserve `BLOCKED` for missing or corrupted locked dataset artifacts, unavailable hardware/storage, a roadmap or user-authority conflict outside this item, or a row-level incompatibility that remains unrecoverable after a documented narrow fix attempt and prevents the intended minimum bundle.

## Locked Benchmark Contract

### Immutable Dataset Authority

- Dataset summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md`
- Dataset root:
  `.artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1/`
- Required frozen artifacts:
  `dataset_manifest.json`, `source_manifest.json`, `split_manifest.json`, `probe_manifest.json`, `simulation_manifest.json`, `adapter_contract.json`, `train.npz`, `val.npz`, `test.npz`, `probe.npz`, `verification/post_audit.json`

### Frozen Data And Probe Invariants

- `N=128`
- total object cap `<= 10_000`
- split counts `8000 / 1000 / 1000`
- no source overlap across train/validation/test
- fixed probe from `probe.npz["probeGuess"]`
- probe pipeline string `pad_extrapolate:128|smooth:0.5`
- reuse the frozen stored diffraction; do not resimulate with a different forward operator
- consume the adapter contract as written:
  one scan group per object patch, one zero coordinate per sample, derive `Y_I` / `Y_phi` from the stored complex object, and preserve the three-way split

### Frozen Row And Training Contract

- row roster:
  `baseline`, `pinn`, `pinn_hybrid_resnet`, `pinn_fno_vanilla`, `pinn_ffno`, `pinn_neuralop_uno`
- training recipe:
  `seed=3`, `epochs=40`, `learning_rate=2e-4`, `scheduler=ReduceLROnPlateau`, `plateau_factor=0.5`, `plateau_patience=2`, `plateau_min_lr=1e-4`, `plateau_threshold=0.0`, `torch_loss_mode=mae`, `torch_output_mode=real_imag`, `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`
- split usage:
  - `train` for fitting
  - `val` for convergence and model-selection behavior where supported
  - `test` for final reported metrics and fixed-sample visuals
- preserve single-seed execution only
- preserve the existing `neuralop_uno` determinism carve-out only for that architecture
- final bundle must say explicitly:
  - this is standalone natural-patch expanded-object CDI evidence
  - it is single-seed evidence on `natural_patches128_fixedprobe_v1`
  - it expands object-distribution evidence beyond `lines128` without replacing it
  - unsupported rows are surfaced as explicit statuses

## Implementation Architecture

- **Unit 1: Contract intake and prepared-input reuse**
  Validate the frozen dataset, reuse or regenerate item-local grouped `train` / `val` / `test` inputs under the item root, and prove those inputs came from the locked dataset without mutating it.
- **Unit 2: Row execution and bundle collation**
  Use the existing natural-patch harness as the default path for TF and Torch rows, preserving the three-way split and row-status accounting; patch only the exact provenance, evaluation, or routing gaps exposed by dry-run or live execution.
- **Unit 3: Durable publication and discoverability**
  Write the natural-patch summary and update the evidence index surfaces so the benchmark is discoverable as a completed expanded-object CDI outcome with the correct claim boundary.

## Concrete File And Artifact Targets

### Primary Code Surfaces

- Modify `scripts/studies/cdi_natural_patch_benchmark.py`
- Modify `scripts/studies/run_cdi_natural_patch_benchmark.py`
- Modify `tests/studies/test_cdi_natural_patch_benchmark.py`

### Conditional Code Surfaces

- Modify `scripts/studies/grid_lines_torch_runner.py` only if a narrow natural-patch provenance, validation, or final-test-evaluation gap is exposed
- Modify `scripts/studies/metrics_tables.py` only if the current bundle writer cannot represent the required natural-patch row statuses or metadata
- Modify `scripts/studies/paper_provenance.py` only if the current provenance schema cannot carry the required dataset/split boundary
- Modify `scripts/studies/grid_lines_compare_wrapper.py` only if a helper must be shared; do not expand scope by moving natural-patch execution behind the compare wrapper
- Modify `tests/torch/test_grid_lines_torch_runner.py` and `tests/studies/test_metrics_tables.py` only when their owning production surfaces change

### Mandatory Contract Outputs

- Item root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/`
- Prepared-input root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/prepared_inputs/`
- Required prepared-input artifacts:
  `train_grouped.npz`, `val_grouped.npz`, `test_grouped.npz`, `prepared_input_manifest.json`, `grouped_input_identity_audit.json`
- Contract root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/contract/`
- Required contract artifacts:
  `natural_patch_benchmark_contract.json`, `fixed_sample_manifest.json`, `shared_visual_scales.json`
- Authoritative run root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/runs/<run_id>/`
- Required run artifacts:
  `metrics.json`, `metric_schema.json`, `model_manifest.json`, `paper_benchmark_manifest.json`, `metrics_table.csv`, `metrics_table.tex`, per-row provenance/config/history/metrics/randomness artifacts, per-row recon outputs, and fixed-sample visuals
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_expanded_benchmark_summary.md`

### Preferred Packaging

- `metrics_table_best.tex`
- item-local `verification/` logs for dry-run, launcher PID / exit-code proof, and any row-recovery notes
- additional audit JSONs that explain row-level `blocked` or `not_protocol_compatible` outcomes without changing the mandatory bundle contract

### Required Discoverability Updates

- Modify `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify `docs/studies/index.md`
- Modify `docs/index.md`
- Update `docs/findings.md` only if implementation discovers a durable repo-wide lesson broader than this item

## Execution Checklist

### Task 1: Preflight The Existing Harness And Lock The Execution Contract

**Files:**

- Reuse `scripts/studies/cdi_natural_patch_benchmark.py`
- Reuse `scripts/studies/run_cdi_natural_patch_benchmark.py`
- Reuse `tests/studies/test_cdi_natural_patch_benchmark.py`

- [ ] Confirm the selected backlog item’s prerequisite evidence surfaces are present before touching code or launching training.
- [ ] Run the backlog-item required check commands unchanged as the baseline deterministic gate.
- [ ] Run one stronger dataset-contract preflight that validates dataset id, split counts, probe pipeline, and post-audit status before any expensive launch.
- [ ] Execute the canonical dry-run command against the current checked-in harness and inspect the emitted prepared-input and contract manifests.
- [ ] If the current dry-run already preserves the locked contract, do not refactor for its own sake. Move to Task 3 unless dry-run inspection exposes a real contract or provenance gap.

**Verification:**

- [ ] **Blocking:** selected-item required input presence gate:
  ```bash
  python - <<'PY'
  from pathlib import Path
  required = [
      Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md"),
      Path("scripts/studies/grid_lines_torch_runner.py"),
      Path("scripts/studies/grid_lines_compare_wrapper.py"),
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
      raise SystemExit(f"missing natural-patch expanded benchmark inputs: {missing}")
  print("natural-patch expanded benchmark inputs present")
  PY
  ```
- [ ] **Blocking:** selected-item required test gate:
  `pytest -q tests/studies/test_cdi_natural_patch_dataset.py tests/studies/test_cdi_natural_patch_benchmark.py`
- [ ] **Blocking:** selected-item required compile gate:
  `python -m compileall -q scripts/studies ptycho_torch`
- [ ] **Blocking:** stronger dataset-contract preflight before any live benchmark:
  ```bash
  python - <<'PY'
  import json
  from pathlib import Path

  dataset_root = Path(".artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1")
  required = [
      dataset_root / "dataset_manifest.json",
      dataset_root / "source_manifest.json",
      dataset_root / "split_manifest.json",
      dataset_root / "probe_manifest.json",
      dataset_root / "simulation_manifest.json",
      dataset_root / "adapter_contract.json",
      dataset_root / "train.npz",
      dataset_root / "val.npz",
      dataset_root / "test.npz",
      dataset_root / "probe.npz",
      dataset_root / "verification" / "post_audit.json",
  ]
  missing = [str(path) for path in required if not path.exists()]
  if missing:
      raise SystemExit(f"missing locked natural-patch dataset artifacts: {missing}")

  dataset_manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
  split_manifest = json.loads((dataset_root / "split_manifest.json").read_text())
  probe_manifest = json.loads((dataset_root / "probe_manifest.json").read_text())
  post_audit = json.loads((dataset_root / "verification" / "post_audit.json").read_text())

  if dataset_manifest.get("dataset_id") != "natural_patches128_fixedprobe_v1":
      raise SystemExit(f"unexpected dataset id: {dataset_manifest.get('dataset_id')}")
  if split_manifest.get("split_counts") != {"train": 8000, "val": 1000, "test": 1000}:
      raise SystemExit(f"unexpected split counts: {split_manifest.get('split_counts')}")
  if probe_manifest.get("canonical_pipeline") != "pad_extrapolate:128|smooth:0.5":
      raise SystemExit(f"unexpected probe pipeline: {probe_manifest.get('canonical_pipeline')}")
  if not post_audit.get("manifests_present") or not post_audit.get("no_source_overlap"):
      raise SystemExit(f"dataset post-audit failed: {post_audit}")
  print("locked natural-patch dataset artifacts present and match the frozen contract")
  PY
  ```
- [ ] **Blocking:** canonical dry-run prelaunch check:
  ```bash
  python scripts/studies/run_cdi_natural_patch_benchmark.py \
    --dataset-root .artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1 \
    --item-root .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark \
    --mode dry-run \
    --rows baseline,pinn,pinn_hybrid_resnet,pinn_fno_vanilla,pinn_ffno,pinn_neuralop_uno \
    --seed 3
  ```

### Task 2: Patch Only The Gaps Exposed By Preflight

**Files:**

- Modify `scripts/studies/cdi_natural_patch_benchmark.py`
- Modify `tests/studies/test_cdi_natural_patch_benchmark.py`
- Modify `scripts/studies/run_cdi_natural_patch_benchmark.py` only if the CLI contract itself is incomplete
- Modify conditional code surfaces only when a failing preflight or row execution proves they are the narrowest fix location

- [ ] Preserve or tighten the current prepared-input contract: item-local grouped `train`, `val`, and `test` inputs; zero-coordinate grouping; explicit dataset-identity audit; no mutation of the dataset root.
- [ ] Preserve or tighten the three-way split contract in row execution: `val` remains the convergence split and `test` remains the final reporting split for both TF and Torch rows.
- [ ] Preserve the exact row roster and explicit row-status accounting.
- [ ] Tighten provenance only where needed so the final bundle records dataset root, validation/test paths, fixed-sample policy, runtime/determinism caveats, and the natural-patch claim boundary.
- [ ] If the current harness already satisfies a surface, leave it alone. Do not migrate logic into the compare wrapper or invent a broader abstraction layer during this item.

**Verification:**

- [ ] **Blocking:** rerun the selected-item required test gate after each meaningful code patch:
  `pytest -q tests/studies/test_cdi_natural_patch_dataset.py tests/studies/test_cdi_natural_patch_benchmark.py`
- [ ] **Blocking:** rerun the selected-item required compile gate after code patches:
  `python -m compileall -q scripts/studies ptycho_torch`
- [ ] **Supporting:** if `grid_lines_torch_runner.py` changed:
  `pytest -q tests/torch/test_grid_lines_torch_runner.py`
- [ ] **Supporting:** if `metrics_tables.py` changed:
  `pytest -q tests/studies/test_metrics_tables.py`
- [ ] **Supporting:** if compare-wrapper helpers changed:
  `pytest -q tests/test_grid_lines_compare_wrapper.py`

### Task 3: Launch The Authoritative Natural-Patch Benchmark

**Files:**

- Reuse the implementation from Tasks 1-2
- Emit results under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/runs/<run_id>/`

- [ ] Choose the authoritative `run_id` and output root before launch, and confirm no live process is already writing there.
- [ ] Launch the full row roster in tmux under `ptycho311`, track the exact PID, and wait on that PID.
- [ ] If a row hits a recoverable harness or environment bug, patch narrowly and rerun the affected row or relaunch the benchmark cleanly.
- [ ] Only leave a row in `blocked` or `not_protocol_compatible` state after a documented narrow fix attempt still fails to preserve the locked contract.
- [ ] Collate the final bundle into the mandatory contract outputs and preserve row-level explicit statuses.

**Verification:**

- [ ] **Blocking:** rerun every cheap deterministic gate from Tasks 1-2 immediately before the live launch.
- [ ] **Blocking:** canonical tmux launch shape:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate ptycho311
  ITEM_ROOT=.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark
  RUN_ID=natural-patch-benchmark-$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "$ITEM_ROOT/verification/$RUN_ID"
  python scripts/studies/run_cdi_natural_patch_benchmark.py \
    --dataset-root .artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1 \
    --item-root "$ITEM_ROOT" \
    --mode benchmark \
    --run-id "$RUN_ID" \
    --rows baseline,pinn,pinn_hybrid_resnet,pinn_fno_vanilla,pinn_ffno,pinn_neuralop_uno \
    --seed 3 \
    > "$ITEM_ROOT/verification/$RUN_ID/launch.log" 2>&1 & pid=$!; printf '%s\n' "$pid" > "$ITEM_ROOT/verification/$RUN_ID/pid.txt"; wait "$pid"; rc=$?; printf '%s\n' "$rc" > "$ITEM_ROOT/verification/$RUN_ID/exit_code.txt"; exit "$rc"
  ```
- [ ] **Blocking:** treat the live run as complete only when:
  - the tracked PID exits `0`
  - the authoritative run root contains fresh `metrics.json`, `metric_schema.json`, `model_manifest.json`, `paper_benchmark_manifest.json`, `metrics_table.csv`, and `metrics_table.tex`
  - the run root records explicit row statuses for every requested row
- [ ] **Supporting:** archive tmux transcript and any row-recovery notes under the item-local `verification/` directory

### Task 4: Publish The Durable Summary And Synchronize Discoverability

**Files:**

- Add `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_expanded_benchmark_summary.md`
- Modify `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify `docs/studies/index.md`
- Modify `docs/index.md`
- Modify `docs/findings.md` only if a repo-wide durable lesson emerged

- [ ] Write the summary with the authoritative run root, row roster, row-status outcomes, dataset/source boundary, object cap, fixed-sample policy, single-seed caveat, and the explicit statement that this benchmark does not replace `lines128`.
- [ ] Update `evidence_matrix.md` with the natural-patch benchmark as a separate expanded-object CDI authority.
- [ ] Update `model_variant_index.json` for each completed natural-patch row that has real metrics and artifact roots. Do not fabricate variant entries for rows that remain blocked.
- [ ] Update `paper_evidence_index.md` with the new natural-patch outcome, tier, claim boundary, and artifact root.
- [ ] Update `docs/studies/index.md` and `docs/index.md` so the summary, entrypoint, and artifact root are discoverable.
- [ ] If the benchmark cannot be completed because of a legitimate blocker, still write a durable summary that records the blocker, the attempted narrow fix, and why the discoverability surfaces were not promoted beyond the blocked state.

**Verification:**

- [ ] **Blocking:** rerun the selected-item required input presence gate after final edits
- [ ] **Blocking:** rerun the selected-item required test gate after final edits:
  `pytest -q tests/studies/test_cdi_natural_patch_dataset.py tests/studies/test_cdi_natural_patch_benchmark.py`
- [ ] **Blocking:** rerun the selected-item required compile gate after final edits:
  `python -m compileall -q scripts/studies ptycho_torch`
- [ ] **Blocking if production workflow code changed in this item:** rerun the repo integration marker before claiming completion:
  `pytest -q -m integration`
- [ ] **Supporting:** if only documentation and artifact-index surfaces changed after the last green code verification, a focused JSON/doc validation pass is sufficient in addition to the blocking checks above

## Completion Criteria

- The frozen dataset contract is consumed, not regenerated.
- The benchmark has one authoritative natural-patch run root with the mandatory bundle artifacts, or a durable blocked summary that explains why an authoritative run could not be produced after narrow recovery attempts.
- The result is clearly labeled as single-seed expanded-object CDI evidence on `natural_patches128_fixedprobe_v1`.
- The summary and discoverability surfaces make the benchmark easy to locate without confusing it with `lines128`.
- Unsupported or failed rows are explicit statuses with rationale; no row disappears silently.
