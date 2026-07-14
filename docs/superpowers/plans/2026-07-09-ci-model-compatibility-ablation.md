# CI Model Compatibility Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. The authoritative task-status ledger below governs current state; body checkboxes in Tasks 1--13 are retained historical procedure, not open-work signals.

**Goal:** Build a reusable, schema-validated Torch ablation driver and use it to determine whether Hybrid ResNet is compatible with the manuscript CI contract under recognizable-quality, absolute-scale, and physical-consistency gates.

> **Historical preliminary revision 2026-07-10 (superseded):** The dose sweep was removed and the preliminary Dead Leaves execution used `ci_3p5m` + `legacy_amp`. This two-descriptor statement is retained only to explain Tasks 10--13 and is superseded for all future execution by the four-descriptor corrective contract below.

> **Corrective revision 2026-07-10:** The completed 12-run execution covered
> Dead Leaves only and is preliminary. Tasks 14--18 supersede the claim-grade
> portions of Tasks 8, 10, 11, 12, and 13: fix report publication and empty-plot
> handling; materialize deterministic Dead Leaves and canonical lines CI/legacy
> twins; add CNN legacy NLL; expand to five arms per family and 30 runs; then
> rerun and adjudicate both families. The full repository suite remains deferred
> under the explicit session constraint; focused suites are required.

> **Quality corrective revision 2026-07-11:** User review rejected the visual
> quality of both architecture families relative to the known-good Hybrid
> ResNet integration reference. The 30-run PASS is withdrawn and retained only
> as diagnostic evidence. Tasks 19--24 and 30 supersede its claim: reproduce known-good
> Hybrid ResNet and CNN grid-lines SSIM through the generic driver, isolate configuration drift
> one variable at a time, diagnose Poisson/rectangular convergence and CNN
> saturation, repair the generic `C>1` component and physical-lines contracts,
> lock meaningful quality/physics gates, rerun, and republish.

**Architecture:** Extend the production Torch runtime only where the accepted design proves missing capability: effective execution controls and structured VarPro/count diagnostics. Build the study layer as focused modules that resolve versioned TOML into authoritative Torch configs, select immutable synthetic or experimental dataset descriptors, execute the canonical training/checkpoint/mmap/reassembly path, and emit typed metrics, verdicts, and visual reports. The historical matrix materialized photon-floor-compliant CI/legacy twins and ran 30 diagnostic runs. The corrective path now requires performance-reference qualification and a one-variable ladder before locking a replacement matrix. Required report figures consume typed metric records and cannot publish empty eligible panels.

**Tech Stack:** Python 3.11, PyTorch, Lightning, NumPy, SciPy, scikit-image, Matplotlib, TOML via `tomllib`, pytest, SHA-256/JSON provenance.

**Design authority:** `docs/superpowers/specs/2026-07-09-ci-model-compatibility-ablation-design.md`

**Repository constraint:** Preserve unrelated dirty-worktree files. Each implementation task owns only its listed files, follows TDD, receives spec-compliance and quality review, and lands as a separate commit.

## Authoritative Task Status Ledger

This ledger is the current status authority for Tasks 1--30. Status values are
closed: `complete_final` means the task's result remains part of the final
contract; `complete_preliminary_superseded` means its historical work completed
but a named later task replaced it for final-claim purposes;
`partial_preliminary_superseded` means focused preliminary work completed, a
required portion was explicitly deferred, and a named later task replaced the
entire preliminary contract. `in_progress` means an atomic implementation
milestone has landed but a named final verification remains open. `pending`
identifies open corrective work. The
three completed statuses are terminal. The unchecked body
checkboxes retained in Tasks 1--13 describe historical execution procedure and
are not open-work indicators. Automation and readers must use this ledger, not
those historical checkboxes, to determine task state.

| Task | Status | Final-status authority |
|---:|---|---|
| 1 | `complete_final` | Effective Torch execution contract completed. |
| 2 | `complete_final` | Structured reassembly and VarPro diagnostics completed. |
| 3 | `complete_final` | Versioned manifest and matrix expansion framework completed. |
| 4 | `complete_final` | Synthetic and experimental dataset descriptor framework completed. |
| 5 | `complete_final` | Authoritative config resolution completed. |
| 6 | `complete_final` | Scientific fingerprints and atomic artifacts completed. |
| 7 | `complete_final` | Anchor-aware closed metric registry completed. |
| 8 | `complete_final` | Typed verdict and report framework completed; Task 14 subsequently hardened publication semantics. |
| 9 | `complete_final` | Canonical runtime adapter and CLI completed. |
| 10 | `complete_preliminary_superseded` | Preliminary Dead Leaves materialization completed; Task 15 replaced it with four claim-grade twins. |
| 11 | `complete_preliminary_superseded` | Preliminary four-arm manifest completed; Task 16 replaced it with the two-family, ten-arm matrix. |
| 12 | `partial_preliminary_superseded` | Focused preliminary verification completed, but required `ci/run_ci_tests.sh` was explicitly deferred and never run; Task 17 then superseded the entire task with the revised focused-verification contract. |
| 13 | `complete_preliminary_superseded` | Preliminary Dead Leaves execution completed and is archived; Task 17 replaced it as final evidence. |
| 14 | `complete_final` | Report evidence and publication semantics completed. |
| 15 | `complete_final` | Four Dead Leaves/canonical-lines claim-grade twins completed. |
| 16 | `complete_final` | Five arms across two object families completed. |
| 17 | `complete_preliminary_superseded` | Thirty-run execution completed, but Task 19 withdraws its PASS because the quality contract admitted degraded and collapsed reconstructions. |
| 18 | `complete_preliminary_superseded` | Historical finding/index publication completed; Task 24 must replace it after corrected evidence exists. |
| 19 | `complete_final` | Gate contract corrected and revised to performance-reference semantics (88edbcfda + 6811ba251); both reviews approved 2026-07-11. |
| 20 | `complete_final` | Both references qualified through the study driver 2026-07-11: Hybrid PASS 0.8984/0.9624; CNN PASS 0.8541/0.9079 (bit-exact cross-path reproduction; floors frozen from recreated 20-epoch reference, commit 46e2c0f14). |
| 21 | `complete_final` | Ladder + diagnostics identified and one-variable-confirmed the root cause of the CI-vs-reference gap: probe batch-tensor rank (flat `(B,H,W)` dictionary emission broadcast into P=B pseudo-modes; accidental ×batch-size physics gain). Confirmation rung1f: amp 0.4856→0.8959. Evidence: `.superpowers/sdd/task-21a-report.md`, commits `e6b97547d..2a9ee2ad9`, sealed rungs under `.artifacts/bridge_ladder/seed3_split/`. |
| 22 | `complete_final` | Completed 2026-07-13. Exact milestone capture, complete checked provenance declarations, and nested milestone reload passed focused review. The tracked six-arm seed-3 run completed 6/6 trajectories at epochs 5/20/40/80 under `.artifacts/ci_compatibility/task22/baseline_seed3/`. Manual grid review found all Hybrid ResNet profiles recognizable and all CNN profiles collapsed/saturated; the fixed common rule selects epoch 80 for every arm. Summary: `task22_summary.md` under the baseline root. |
| 23 | `complete_final` | The coherent 36-arm matrix completed 36/36 at source commit `5fcfd1e80`; its numeric report is sealed as `FAIL`, while manual visual status remains explicitly pending. |
| 24 | `pending` | Not eligible: Task 23's numeric verdict is `FAIL` and manual visual status is pending, so the all-gates-pass publication condition is unsatisfied. |
| 25 | `complete_final` | Probe-rank physics contract enforced 2026-07-12: fail-fast `ProbeLayoutError`, dictionary-emission migration, explicit `amplitude_physics_gain` (commits `8e9c16a79..5a28d1d8a`; spec + quality reviews APPROVED; PROBE-RANK-001 in docs/findings.md; contract shard docs/specs/spec-ptycho-torch-probe-layout.md). Design authority: `docs/superpowers/specs/2026-07-12-probe-rank-physics-contract-fix-design.md`. |
| 26 | `complete_final` | Gain calibration completed at execution commit `72816630e`: Rule A selected fixed gain 16 for the locked legacy-amplitude reference regime (amp/phase SSIM `0.8858652644`/`0.9618665959`); Rule B `4.2441268779` was outside plateau `[16,16]`. Evidence: `.artifacts/gain_calibration_v2_commit-72816630/sweep_summary.json`; report: `.superpowers/sdd/task-26-report.md`. |
| 27 | `complete_final` | Corrected-physics gain-16 reference requalification and atomic floor re-pinning completed. Inclusive commits: `46cc4d5bf` through `7c221d7e1` (harness/preparation `46cc4d5bf`, `c5dced526`, `a7e0dfa8b`, `8aa239881`, `7f0e3dde5`; pin promotion/hardening `532761c59`, `a2bc9634d`, `7c221d7e1`). Both final qualification verdicts and visual reviews PASS. Report: `.superpowers/sdd/task-27-report.md`. |
| 28 | `complete_final` | Canonical rung1a PASS under unit `dictionary_parity`: amp/phase SSIM `0.8913340876617375`/`0.9632217816205675`, absolute deltas from rung0 `0.0054688232603687`/`0.0013551856818027` within locked `0.02`/`0.01` gates. Root `.artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun`; evidence/report SHA-256 `a8741f93511cc90eb26777b1ff2cab0235ce812cff3661bad168632fce9ce711`/`2f95c0e4651a2a48449797f4a34aed557ce70962c8e82dced0d40b4aa95a0cf6`. Implementation commits `65c802e17`, `9c6b08ca2`, `292b18be8`; spec and code-quality reviews APPROVED. Historical fail/diagnostic evidence remains immutable. Report: `.superpowers/sdd/task-28-report.md`. |
| 29 | `pending` | Retire the components.py inline dataset producer in favor of the mmap `PtychoDataset` path (fallback: shared emission core). Depends on Task 28 PASS; executes after Task 24; coordinated with the refactoring roadmap. Added 2026-07-12 per user direction. |
| 30 | `complete_final` | CNN contract recovery passed: equal semantic component channels, bounded physical lines, active-support diagnostics, derived legacy normalization, and full-support seed-3 qualification. Fresh corrected Hybrid completions and prior support-on CNN metrics remain identity-distinct and are not a fabricated aggregate. |

**Revised execution order (2026-07-13):** 21 → 25 → 26 → 27 → 28 → 22 → 30 → 23 → 24 → 29.
Tasks 25–28 implement the physics-contract fix selected by the user after the
Task 21 root cause was confirmed. Tasks 22, 23, and 30 are `complete_final`.
Task 23's coherent matrix produced a sealed numeric `FAIL` with manual visual
status pending, so Task 24 is not eligible and no later task is currently
ready. A CI result under the old flat-probe physics,
raw lines magnitude, or asymmetric `C>1` component broadcast cannot be
published.

---

## File Map

### Shared Runtime

- Modify `ptycho/config/config.py`: add validated execution-owned `devices` and `precision` fields.
- Modify `ptycho_torch/train_lightning_only.py`: consume effective execution fields and explicit seed; persist effective runtime.
- Create `ptycho_torch/reassembly_diagnostics.py`: structured VarPro/reassembly diagnostics and objective math.
- Modify `ptycho_torch/reassembly.py`: populate structured diagnostics and expose a physical count-metric second pass without breaking legacy return shapes.

### Reusable Study Framework

- Create `scripts/studies/ablation/__init__.py`: public study-framework exports.
- Create `scripts/studies/ablation/manifest.py`: TOML schema v1, matrix expansion, selectors, gates/comparisons.
- Create `scripts/studies/ablation/datasets.py`: immutable synthetic/experimental descriptor validation and capabilities.
- Create `scripts/studies/ablation/configuration.py`: authoritative config registry, allowlists, coercion, aliases, CI validation.
- Create `scripts/studies/ablation/artifacts.py`: canonical JSON, fingerprints, completion records, resume/rerun.
- Create `scripts/studies/ablation/metrics.py`: anchor-aware truth/reference alignment and closed metric registry.
- Create `scripts/studies/ablation/verdicts.py`: typed gate evaluation and manual-review schema.
- Create `scripts/studies/ablation/visual_review.py`: manual-review schema/parse (Task 8 split, re-exported via `verdicts.py`).
- Create `scripts/studies/ablation/reporting.py`: CSV/JSON/report/visual aggregation.
- Create `scripts/studies/ablation/reporting_figures.py`: figure renderers (Task 8 split, consumed by `reporting.py`).
- Create `scripts/studies/ablation/runtime.py`: architecture-neutral canonical train/reload/mmap/reconstruct execution.
- Create `scripts/studies/torch_ablation_driver.py`: thin CLI.

### Study Definition And Data

- Create `scripts/studies/materialize_ci_compatibility_datasets.py`: deterministic CI-count and legacy-amplitude twin materializer.
- Create `scripts/studies/specs/hybrid_resnet_ci_compatibility.toml`: checked corrected six-arm, two-family study matrix and conditional gates (historical Task 16 first landed five arms).

### Tests

- Create `tests/torch/test_train_lightning_execution_contract.py`.
- Create `tests/torch/test_structured_reassembly_diagnostics.py`.
- Create `tests/studies/test_torch_ablation_manifest.py`.
- Create `tests/studies/test_torch_ablation_datasets.py`.
- Create `tests/studies/test_torch_ablation_configuration.py`.
- Create `tests/studies/test_torch_ablation_artifacts.py`.
- Create `tests/studies/test_torch_ablation_metrics.py`.
- Create `tests/studies/test_torch_ablation_verdicts.py`.
- Create `tests/studies/test_torch_ablation_reporting.py`.
- Create `tests/studies/test_torch_ablation_runtime.py`.
- Create `tests/studies/test_ci_compatibility_materializer.py`.
- Create `tests/studies/test_torch_ablation_driver_integration.py`.

---

### Task 1: Effective Torch Execution Contract

**Files:**
- Modify: `ptycho/config/config.py:213-430`
- Modify: `ptycho_torch/train_lightning_only.py:55-95,130-430`
- Create: `tests/torch/test_train_lightning_execution_contract.py`
- Test: `tests/torch/test_config_factory.py`
- Test: `tests/torch/test_workflows_components.py`

- [ ] **Step 1: Write failing execution-config tests**

Add tests proving:

```python
cfg = PyTorchExecutionConfig(devices=1, precision="32-true")
assert cfg.devices == 1
assert cfg.precision == "32-true"

with pytest.raises(ValueError, match="devices"):
    PyTorchExecutionConfig(devices=0)
with pytest.raises(ValueError, match="precision"):
    PyTorchExecutionConfig(precision="fp12")
```

Cover `devices="auto"`, positive integers, and the three accepted precision values.

- [ ] **Step 2: Write failing trainer-wiring tests**

Monkeypatch `lightning.Trainer`, seed setup, and the data module. Call `train_lightning_only.main(..., seed=11, execution_config=cfg)` and assert effective kwargs include `devices`, mapped accelerator, strategy, deterministic, enable-progress, enable-checkpointing, and precision. Assert `TrainingConfig.n_devices/strategy/device` are derived from execution and that no environment-only seed path is used.

Request `return_training_result=True` and assert a typed result carries the
run directory, in-memory trained model, finalized configs, and effective
runtime. The default call must remain backward compatible and return only the
run directory.

- [ ] **Step 3: Run RED tests**

Run:

```bash
pytest -q tests/torch/test_train_lightning_execution_contract.py
```

Expected: failures for missing fields/signature and hardcoded Trainer values.

- [ ] **Step 4: Implement minimal execution ownership**

Add to `PyTorchExecutionConfig`:

```python
devices: Union[int, Literal["auto"]] = 1
precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
```

Validate both in `__post_init__`. Extend `train_lightning_only.main` with keyword-only `seed: Optional[int] = None`; use `_resolve_seed()` only when absent. Derive compatibility training aliases, construct callbacks conditionally, and pass all effective fields into `L.Trainer`.

Add a small `TrainingRunResult` dataclass and opt-in
`return_training_result: bool = False`. The opt-in result is the generic
handoff for pre/post checkpoint reload verification and must not retain an open
Trainer or worker process.

- [ ] **Step 5: Persist and test effective runtime**

Write `effective_runtime.json` under the run directory with resolved seed, Trainer kwargs, dataloader worker/pinning settings, and precision. Tests must compare persisted values to the actual fake Trainer kwargs.

- [ ] **Step 6: Run GREEN and regression tests**

```bash
pytest -q \
  tests/torch/test_train_lightning_execution_contract.py \
  tests/torch/test_config_factory.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_absolute_scaling_entrypoints.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add ptycho/config/config.py ptycho_torch/train_lightning_only.py \
  tests/torch/test_train_lightning_execution_contract.py
git commit -m "feat(torch): honor effective execution config"
```

---

### Task 2: Structured Reassembly And VarPro Diagnostics

**Files:**
- Create: `ptycho_torch/reassembly_diagnostics.py`
- Modify: `ptycho_torch/reassembly.py:316-710,848-930,1011-1110,1153-1524`
- Create: `tests/torch/test_structured_reassembly_diagnostics.py`
- Test: `tests/torch/test_absolute_scaling_varpro.py`
- Test: `tests/torch/test_varpro_probe_weighted_reassembly.py`
- Test: `tests/torch/test_varpro_solve_units.py`

- [ ] **Step 1: Write failing objective and serialization tests**

Define the expected interface:

```python
stats = VarProSufficientStatistics(ata=ata, atb=atb, sum_i2=sum_i2, n_pixels=n)
diag = ReassemblyDiagnostics.from_statistics(
    stats, s1=2.0, s2=0.5, profile="ci_intensity_v2", ...
)
assert diag.fitted_objective <= diag.unit_objective + 1e-12 + 1e-10 * abs(diag.unit_objective)
assert diag.to_jsonable()["schema_version"] == 1
```

Pin `z=[s1**2,s2**2,s1*s2]`, objective normalization by pixel count, condition number, mask digest, canvas anchor/weights metadata, and accepted/total patch counts.

- [ ] **Step 2: Write failing count-pass tests**

Use a two-batch synthetic model/dataset with a known physical probe and scales. Assert:

```python
metrics.relative_l2_intensity_error == pytest.approx(0.0, abs=1e-6)
metrics.n_samples == expected_samples
metrics.n_pixels == expected_pixels
```

Pin `mean(I_pred - I_meas*log(clamp(I_pred,1e-8)))`, multimode mode-sum, effective mask digest, and no storage of full detector stacks.

- [ ] **Step 3: Run RED tests**

```bash
pytest -q tests/torch/test_structured_reassembly_diagnostics.py
```

Expected: missing module/API failures.

- [ ] **Step 4: Implement diagnostic value objects**

Keep objective math and JSON conversion in `reassembly_diagnostics.py`. Add `VarProScaler.sufficient_statistics()` without exposing mutable tensors. Instrument `VectorizedWeightedAccumulator` with accepted/total counters.

- [ ] **Step 5: Add opt-in structured return without breaking callers**

Extend `reconstruct_image_barycentric` with:

```python
structured_diagnostics: bool = False
```

Preserve all current tuple shapes when false. When true, return the reconstructed canvas, dataset subset, `ReassemblyDiagnostics`, and prescale canvas. The structured object must include aggregate statistics, never final-batch `Psi_a/Psi_b`.

- [ ] **Step 6: Implement and test effective inference precision**

Add `precision: Optional[Literal["32-true", "16-mixed", "bf16-mixed"]] = None`
and change the legacy alias to `use_mixed_precision: Optional[bool] = None`.
Resolve: neither -> `32-true`; legacy true/false -> `16-mixed`/`32-true`;
precision-only -> precision; both -> require equivalent values or raise
`ValueError`. Wrap model forward in `torch.autocast` for mixed modes using
FP16/BF16 respectively; keep complex physics accumulation in
complex64/float32 or float64 sufficient statistics. Tests cover omitted,
legacy-only, precision-only, equivalent-both, conflicting-both, selected
autocast dtype, and `effective_runtime.json` agreement.

- [ ] **Step 7: Implement the deterministic physical count pass**

Add `evaluate_fitted_count_metrics(...)` in `reassembly.py`. Reuse CI named fields, physical masked probe, `compute_varpro_basis`, and solved scales. Reject legacy profiles with a typed not-applicable result.

- [ ] **Step 8: Run GREEN and regression tests**

```bash
pytest -q \
  tests/torch/test_structured_reassembly_diagnostics.py \
  tests/torch/test_absolute_scaling_varpro.py \
  tests/torch/test_varpro_probe_weighted_reassembly.py \
  tests/torch/test_varpro_solve_units.py \
  tests/torch/test_multimode_probe_and_from_np.py
```

- [ ] **Step 9: Commit**

```bash
git add ptycho_torch/reassembly.py ptycho_torch/reassembly_diagnostics.py \
  tests/torch/test_structured_reassembly_diagnostics.py
git commit -m "feat(torch): expose structured VarPro diagnostics"
```

---

### Task 3: Versioned TOML Manifest And Matrix Expansion

**Files:**
- Create: `scripts/studies/ablation/__init__.py`
- Create: `scripts/studies/ablation/manifest.py`
- Create: `tests/studies/test_torch_ablation_manifest.py`

- [ ] **Step 1: Write failing schema and expansion tests**

Cover schema version rejection, declaration-order Cartesian expansion, complete include assignments, matching excludes, duplicate ids, dimension override collisions, base specialization, CLI precedence, and exact run ids including dataset id.

- [ ] **Step 2: Write failing selector and gate-schema tests**

Pin exact-id selection and conjunction grammar:

```python
selected = select_runs(runs, "architecture=hybrid_resnet,physics_profile=ci_nll")
assert {r.dimensions["architecture"] for r in selected} == {"hybrid_resnet"}
```

Validate dataset-relative gate/comparison targets expressed as dimension
selectors, resolved only after `--dataset` selection. Pin zero-match and
multi-match failures, optional dataset narrowing, the closed operators,
aggregations, metric paths, condition keys, unique gate ids, and required
fields for `status_count_ge` and `paired_ratio_ge`.

Pin `on_missing_capability = "error" | "not_applicable"` with default
`error`; reject unknown values and prove reference-only gates become typed
not-applicable on experimental no-reference data when explicitly configured.

- [ ] **Step 3: Run RED tests**

```bash
pytest -q tests/studies/test_torch_ablation_manifest.py
```

- [ ] **Step 4: Implement immutable manifest value objects and parser**

Use `tomllib`; do not execute Python configuration. Canonicalize with sorted-key JSON and preserve declaration order separately for expansion and ids.

- [ ] **Step 5: Implement deterministic expansion and CLI overrides**

Apply the six-step precedence contract exactly. `--dataset` filters a dataset
dimension or replaces base selection. Resolve gate/comparison selectors against
the resulting logical arms, not pre-replacement ids. Conflicting non-base
dimension assignments fail.

- [ ] **Step 6: Run GREEN tests**

```bash
pytest -q tests/studies/test_torch_ablation_manifest.py
```

- [ ] **Step 7: Commit**

```bash
git add scripts/studies/ablation/__init__.py \
  scripts/studies/ablation/manifest.py \
  tests/studies/test_torch_ablation_manifest.py
git commit -m "feat(studies): add versioned ablation manifests"
```

---

### Task 4: Synthetic And Experimental Dataset Descriptors

**Files:**
- Create: `scripts/studies/ablation/datasets.py`
- Create: `tests/studies/test_torch_ablation_datasets.py`

- [ ] **Step 1: Write failing closed-schema tests**

Build minimal synthetic, experimental-reference, and experimental-no-reference descriptors. Assert valid descriptors load and unknown fields, bad enums, malformed hashes, forbidden dose tables, or missing role-dependent fields fail.

- [ ] **Step 2: Write failing content-validation tests**

Create tiny NPZ fixtures and validate file hashes, required keys, detector/probe shape, coordinate lengths, probe modes, nonnegative count dtype, truth role, canonical probe-array hash, dose statistics, and provenance agreement.

- [ ] **Step 3: Write failing capability and compatibility tests**

Assert derived capabilities and reject CI without physical count gauge, unsupported `C`, count datasets selected by legacy normalized-amplitude arms, and truth gates against reference-only data.

- [ ] **Step 4: Run RED tests**

```bash
pytest -q tests/studies/test_torch_ablation_datasets.py
```

- [ ] **Step 5: Implement descriptor parser and capability derivation**

Accept repository-relative checked descriptors and absolute standalone descriptors. Never infer the measurement profile from numeric ranges. Return frozen descriptor/capability objects.

- [ ] **Step 6: Run GREEN tests**

```bash
pytest -q tests/studies/test_torch_ablation_datasets.py
```

- [ ] **Step 7: Commit**

```bash
git add scripts/studies/ablation/datasets.py \
  tests/studies/test_torch_ablation_datasets.py
git commit -m "feat(studies): validate immutable dataset descriptors"
```

---

### Task 5: Authoritative Config Resolution

**Files:**
- Create: `scripts/studies/ablation/configuration.py`
- Create: `tests/studies/test_torch_ablation_configuration.py`
- Test: `tests/torch/test_absolute_scaling_contract.py`

- [ ] **Step 1: Write failing ownership/allowlist tests**

Prove each dotted path maps to exactly one owner. Accept representative effective fields from data/model/training/inference/execution. Reject TF fields, execution duplicates, unimplemented fields, `intensity_scale_trainable`, unknown fields, and typos with suggestions.

- [ ] **Step 2: Write failing coercion and invariant tests**

Cover literals, optionals, tuples/lists, booleans, numeric ranges, `C=grid_x*grid_y`, `C_model=C_forward=C`, execution-to-training compatibility aliases, fixed `DatagenConfig`, and full explicit resolved output.

- [ ] **Step 3: Write failing CI-profile tests**

Assert the four activation predicates, CI+MAE/supervised rejection, amplitude-mode non-activation under default CI metadata, and explicit legacy pair behavior.

- [ ] **Step 4: Run RED tests**

```bash
pytest -q tests/studies/test_torch_ablation_configuration.py
```

- [ ] **Step 5: Implement registry, allowlists, coercion, and validation**

Construct `DataConfig`, `ModelConfig`, `TrainingConfig`, `InferenceConfig`, fixed `DatagenConfig`, and `PyTorchExecutionConfig` directly. Do not call the flat override factory.

- [ ] **Step 6: Run GREEN and contract regressions**

```bash
pytest -q \
  tests/studies/test_torch_ablation_configuration.py \
  tests/torch/test_absolute_scaling_contract.py \
  tests/torch/test_loss_modes.py
```

- [ ] **Step 7: Commit**

```bash
git add scripts/studies/ablation/configuration.py \
  tests/studies/test_torch_ablation_configuration.py
git commit -m "feat(studies): resolve authoritative Torch configs"
```

---

### Task 6: Scientific Fingerprints And Atomic Artifacts

**Files:**
- Create: `scripts/studies/ablation/artifacts.py`
- Create: `tests/studies/test_torch_ablation_artifacts.py`

- [ ] **Step 1: Write failing canonicalization/fingerprint tests**

Pin canonical JSON bytes and SHA-256 sensitivity to schema, config, seed, Git identity, environment digest, dataset/probe hashes, and checkpoint hash.

- [ ] **Step 2: Write failing completion/resume/rerun tests**

Cover atomic temp-to-complete transition, required artifact hash validation, clean versus dirty claim grade, incomplete attempt restart, corrupt completion refusal, and archival `--rerun` behavior.

- [ ] **Step 3: Run RED tests**

```bash
pytest -q tests/studies/test_torch_ablation_artifacts.py
```

- [ ] **Step 4: Implement artifact store**

Keep run state in `attempt-N/`; write data then hashes then atomically replace `completion.json.tmp` with `completion.json`. Never overwrite a completed attempt.

- [ ] **Step 5: Run GREEN tests and commit**

```bash
pytest -q tests/studies/test_torch_ablation_artifacts.py
git add scripts/studies/ablation/artifacts.py \
  tests/studies/test_torch_ablation_artifacts.py
git commit -m "feat(studies): add reproducible ablation artifacts"
```

---

### Task 7: Anchor-Aware Closed Metric Registry

**Files:**
- Create: `scripts/studies/ablation/metrics.py`
- Create: `tests/studies/test_torch_ablation_metrics.py`
- Test: `tests/test_evaluation_single_image_frc.py`

- [ ] **Step 1: Write failing alignment/mask tests**

Use nonzero and half-pixel scan COM fixtures. Pin canvas-to-truth coordinates, real/imag bilinear sampling, validity mask, deterministic maximal rectangle with full tie ordering, and centered square FRC footprint. Prove object-center cropping gives a different result.

- [ ] **Step 2: Write failing absolute and recognizability tests**

Pin no-amplitude-gauge absolute MAE/NRMSE, unit-magnitude global phase alignment, raw amplitude Pearson, mean-normalized SSIM/MS-SSIM data ranges, wrapped phase residuals, square FRC calls, and near-zero failure behavior.

- [ ] **Step 3: Write failing namespace tests**

The same formula must emit `truth_quality.*` for object truth and `reference_agreement.*` for a conventional reference. Count/VarPro/dose values must emit only `measurement_consistency.*`. Unknown paths fail.

- [ ] **Step 4: Run RED tests**

```bash
pytest -q tests/studies/test_torch_ablation_metrics.py
```

- [ ] **Step 5: Implement explicit metric adapter**

Reuse low-level FRC/SSIM helpers where their preprocessing is bypassed or explicit. Do not call `eval_reconstruction`, resize images, fit translations, or apply hidden global trim.

- [ ] **Step 6: Run GREEN and FRC regressions**

```bash
pytest -q \
  tests/studies/test_torch_ablation_metrics.py \
  tests/test_evaluation_single_image_frc.py
```

- [ ] **Step 7: Commit**

```bash
git add scripts/studies/ablation/metrics.py \
  tests/studies/test_torch_ablation_metrics.py
git commit -m "feat(studies): add anchor-aware ablation metrics"
```

---

### Task 8: Typed Verdicts And Interpretable Reports

**Files:**
- Create: `scripts/studies/ablation/verdicts.py`
- Create: `scripts/studies/ablation/reporting.py`
- Create: `tests/studies/test_torch_ablation_verdicts.py`
- Create: `tests/studies/test_torch_ablation_reporting.py`

- [ ] **Step 1: Write failing status/gate tests**

Cover `PASS|FAIL|INCONCLUSIVE`, terminal success denominator, missing attempts,
completed failed attempts, successful-only medians, paired-seed ratios, dose
CV, conditional capabilities, both `on_missing_capability` modes,
not-applicable gates, and mandatory missing operands.

- [ ] **Step 2: Write failing manual-review tests**

Require schema version, reviewer, UTC timestamp, figure SHA, approve/reject, recognizable, flat, checkerboard, mirrored, saturation, collapse, and notes. Approval must agree with all component fields.

- [ ] **Step 3: Write failing report/visual tests**

Given fixture rows, assert generation of `report.md`, tidy CSV/JSON, verdict tables, reconstruction/truth/error grids with shared row limits, curves, seed plots, scale plots, and stability dashboard. Assert all figure rows map to run ids and labels distinguish absolute/normalized/reference metrics.

Generic reusable metric support may retain typed dose records for other studies;
the compatibility report has no dose sweep, no dose figure, and no mandatory
dose-CV gate.

- [ ] **Step 4: Run RED tests**

```bash
pytest -q \
  tests/studies/test_torch_ablation_verdicts.py \
  tests/studies/test_torch_ablation_reporting.py
```

- [ ] **Step 5: Implement verdict and reporting modules**

Keep plotting free of training/runtime imports. Render failed/missing arms explicitly. Write a pending `visual_review.json` template without self-approval.

- [ ] **Step 6: Run GREEN tests and commit**

```bash
pytest -q tests/studies/test_torch_ablation_verdicts.py \
  tests/studies/test_torch_ablation_reporting.py
git add scripts/studies/ablation/verdicts.py scripts/studies/ablation/reporting.py \
  tests/studies/test_torch_ablation_verdicts.py \
  tests/studies/test_torch_ablation_reporting.py
git commit -m "feat(studies): add typed ablation reports"
```

---

### Task 9: Canonical Runtime Adapter And CLI

**Files:**
- Create: `scripts/studies/ablation/runtime.py`
- Create: `scripts/studies/torch_ablation_driver.py`
- Create: `tests/studies/test_torch_ablation_runtime.py`
- Create: `tests/studies/test_torch_ablation_driver_integration.py`

- [ ] **Step 1: Write failing canonical-call tests**

Monkeypatch exact entry points and assert this order:

```text
train_lightning_only.main(return_training_result=True)
find_best_checkpoint
torch.load(best_checkpoint)["state_dict"] -> in_memory_model.load_state_dict(strict=True)
PtychoDataset
reconstruct_image_barycentric(best_state_reference_model, structured_diagnostics=True)
load_checkpoint_with_configs
reconstruct_image_barycentric(reloaded_model, structured_diagnostics=True)
evaluate_fitted_count_metrics  # CI only
```

Assert staged train-only NPZ isolation, full config round-trip including
inference/fixed datagen, physical count pass only for CI, typed legacy
not-applicable records, and effective runtime/precision matching. Persist the
best-checkpoint SHA, fixed-batch input identity, and reference texture/full
stitched canvas produced after loading that exact best state into the existing
model without using the production checkpoint loader. Then invoke the
production loader, enforce `rtol=1e-5, atol=1e-6`, and publish max error under
the stability namespace. Never compare final-epoch in-memory weights against an
earlier validation-best checkpoint.

- [ ] **Step 2: Write failing CLI tests**

Cover `--dry-run`, selectors, dataset-relative gate targets, seed/epoch/output
overrides, `--dataset`, `--dataset-spec`, `--resume`, `--rerun`, `--fail-fast`,
and `--visual-review`. Dry-run must not load NPZ data or allocate CUDA.

- [ ] **Step 3: Write failing tiny integration test**

Run a two-arm, one-seed, one-epoch CPU fixture through training stub,
pre-reload capture, reload parity, reconstruction stub, metrics, report,
completion, and resume. Add dataset-selection cases for synthetic truth,
experimental reference, and experimental no-reference descriptors through the
same CLI; assert correct gate targeting, namespaces, deterministic run ids, and
artifact links.

- [ ] **Step 4: Run RED tests**

```bash
pytest -q \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_driver_integration.py
```

- [ ] **Step 5: Implement architecture-neutral runtime and thin CLI**

The runtime sees only resolved configs and dataset descriptors. Architecture selection remains `model.architecture`; no architecture `if` statements are allowed in driver/runtime modules.

- [ ] **Step 6: Run GREEN tests and commit**

```bash
pytest -q tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_driver_integration.py
git add scripts/studies/ablation/runtime.py scripts/studies/torch_ablation_driver.py \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_driver_integration.py
git commit -m "feat(studies): add reusable Torch ablation driver"
```

---

### Task 10: Preliminary Dead-Leaves Dataset Materializer (Superseded By Task 15)

This task records the completed Dead Leaves-only implementation. Do not use its
two-dataset output as final claim-grade scope; Task 15 extends it to both object
families.

**Files:**
- Create: `scripts/studies/materialize_ci_compatibility_datasets.py`
- Create: `tests/studies/test_ci_compatibility_materializer.py`
- Read/reuse: `scripts/studies/make_synthetic_truth_datasets.py`
- Read/reuse: `scripts/studies/make_dose_ladder_datasets.py`

- [ ] **Step 1: Write failing deterministic materialization tests**

On N=16 fixtures, assert stable checksums, same latent object/coordinates across CI and legacy bundles, calibrated probe/count co-scaling for the mean-photons-per-image target, profile metadata, photon-floor and saturation statistics, and refusal to overwrite mismatched output. (Revision 2026-07-10: two-dataset family `ci_3p5m` + `legacy_amp`; dose-only twins removed.)

- [ ] **Step 2: Run RED tests**

```bash
pytest -q tests/studies/test_ci_compatibility_materializer.py
```

- [ ] **Step 3: Implement the materializer using shared builders**

Do not duplicate diffraction physics. Materialize CI train/test calibrated to a 3.5M mean-photons-per-image target and normalized-amplitude legacy train/test twins from identical latent truth/coordinates. Emit the closed descriptor fields and provenance hashes.

- [ ] **Step 4: Run GREEN tests**

```bash
pytest -q tests/studies/test_ci_compatibility_materializer.py \
  tests/torch/test_absolute_scaling_dataset_generation.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/studies/materialize_ci_compatibility_datasets.py \
  tests/studies/test_ci_compatibility_materializer.py
git commit -m "feat(studies): materialize CI compatibility datasets"
```

---

### Task 11: Preliminary Four-Arm Manifest (Superseded By Task 16)

This task records the completed 12-run manifest. Task 16 replaces its
claim-grade matrix and hash contract.

**Files:**
- Create: `scripts/studies/specs/hybrid_resnet_ci_compatibility.toml`
- Create: `scripts/studies/specs/examples/experimental_npz_dataset.toml`
- Modify: `docs/index.md`
- Test: `tests/studies/test_torch_ablation_manifest.py`
- Test: `tests/studies/test_torch_ablation_configuration.py`
- Test: `tests/studies/test_torch_ablation_datasets.py`

- [ ] **Step 1: Materialize the N=64 study bundle**

```bash
python scripts/studies/materialize_ci_compatibility_datasets.py \
  --output-root .artifacts/ci_compatibility/datasets \
  --detector-size 64 --train-positions 5000 --test-positions 1250 \
  --probe-src /home/ollie/Documents/PtychoPINN/datasets/fly/fly001.npz
```

Expected: descriptor/provenance report all CI images above one million photons (target mean 3.5M photons/image), no saturation, and stable hashes.

- [ ] **Step 2: Write the checked TOML with explicit values**

Define the synthetic descriptors, all allowlisted base values, architecture and
physics-profile dimensions, two CNN legacy excludes, three seeds, CI/legacy
dataset pairing, dataset-relative selector targets, conditional truth and
experimental gates, status gate, paired ratio, and manual review.
Reference-agreement gates use `on_missing_capability="not_applicable"` so an
experimental no-reference descriptor remains a valid shared-manifest input;
CI physical-consistency requirements retain `error`.

- [ ] **Step 3: Add manifest contract tests**

Assert dry expansion is exactly four logical arms and twelve runs; CI arms use
CI/count/rectangular/Poisson/VarPro; legacy arms use legacy/amplitude/amplitude-
path/VarPro-off; C=4/object_big/probe training weighting are effective.
Replace the dataset with experimental-reference and
experimental-no-reference descriptors and assert conditional gates resolve to
the replacement arms without embedding stale dataset ids.

- [ ] **Step 4: Run tests and dry-run**

Before running, create the checked example experimental descriptor with the
complete closed schema, explicit placeholder NPZ paths/hashes, and a comment
that dry-run validates schema/capability routing only. Content validation of
real files remains mandatory outside dry-run.

```bash
pytest -q tests/studies/test_torch_ablation_manifest.py \
  tests/studies/test_torch_ablation_configuration.py \
  tests/studies/test_torch_ablation_datasets.py
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --dry-run
```

Expected: 12 valid runs, no data loading/GPU allocation, explicit resolved deltas.

Also dry-run a temporary standalone experimental descriptor:

```bash
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --dataset-spec scripts/studies/specs/examples/experimental_npz_dataset.toml \
  --dataset experimental_fixture \
  --only physics_profile=ci_nll --dry-run
```

Expected: only compatible CI arms, experimental measurement/reference gates,
and dataset-qualified run ids.

- [ ] **Step 5: Index the accepted design, plan, driver, and examples**

Add a concise `docs/index.md` entry linking the design, implementation plan,
checked study TOML, CLI, and experimental descriptor example.

- [ ] **Step 6: Commit**

```bash
git add scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  scripts/studies/specs/examples/experimental_npz_dataset.toml \
  tests/studies/test_torch_ablation_manifest.py docs/index.md
git commit -m "study: define Hybrid ResNet CI compatibility matrix"
```

---

### Task 12: Preliminary Verification Before Dead-Leaves GPU Execution (Superseded By Task 17)

This historical task includes the preliminary full-suite and 12-run projection
instructions. Do not execute them for the corrective study; Task 17 is the only
current verification and execution authority. Terminal status:
`partial_preliminary_superseded`. Focused preliminary verification completed,
but Step 3's required `ci/run_ci_tests.sh` gate was explicitly deferred and
never run; Task 17 subsequently superseded this entire preliminary task with
its revised focused-only verification contract.

**Files:**
- Modify only if a root-cause regression is found in owned files.

- [ ] **Step 1: Run focused study/runtime suite**

```bash
pytest -q \
  tests/torch/test_train_lightning_execution_contract.py \
  tests/torch/test_structured_reassembly_diagnostics.py \
  tests/studies/test_torch_ablation_manifest.py \
  tests/studies/test_torch_ablation_datasets.py \
  tests/studies/test_torch_ablation_configuration.py \
  tests/studies/test_torch_ablation_artifacts.py \
  tests/studies/test_torch_ablation_metrics.py \
  tests/studies/test_torch_ablation_verdicts.py \
  tests/studies/test_torch_ablation_reporting.py \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_ci_compatibility_materializer.py \
  tests/studies/test_torch_ablation_driver_integration.py
```

- [ ] **Step 2: Run scaling/loader/reassembly regressions**

```bash
pytest -q \
  tests/torch/test_absolute_scaling_contract.py \
  tests/torch/test_absolute_scaling_math.py \
  tests/torch/test_absolute_scaling_loss.py \
  tests/torch/test_absolute_scaling_mmap.py \
  tests/torch/test_absolute_scaling_dict.py \
  tests/torch/test_absolute_scaling_varpro.py \
  tests/torch/test_multimode_probe_and_from_np.py \
  tests/torch/test_varpro_probe_weighted_reassembly.py \
  tests/torch/test_reassembly_sign_parity.py
```

- [ ] **Step 3: Run CI-equivalent gate**

```bash
bash ci/run_ci_tests.sh
```

Expected: all pass. Archive logs under `.artifacts/ci_compatibility/verification/`.

- [ ] **Step 4: Run one-arm one-epoch GPU smoke**

```bash
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --only architecture=hybrid_resnet,physics_profile=ci_nll \
  --seeds 3 --epochs 1 \
  --output-root .artifacts/ci_compatibility/smoke
```

Expected: complete training/reload/reassembly/count metrics/report with pending visual review.

- [ ] **Step 5: Record runtime projection**

Record measured one-epoch training, inference, and assembly times plus peak
memory. Project the 12-run ten-epoch wall time with a stated serial/parallel
assumption and verify it remains practical before the full launch.

- [ ] **Step 6: Commit any verified root-cause fixes separately**

Do not bundle test-only workarounds. Re-run the failing selector and the focused suite before each fix commit.

---

### Task 13: Preliminary Dead-Leaves Execution (Superseded By Task 17)

The completed artifact is development evidence only. It lacks canonical lines,
CNN legacy NLL, and the corrected report-publication contract. Task 17 owns the
fresh claim-grade execution.

**Files/Artifacts:**
- Create: `.artifacts/ci_compatibility/full/`
- Create: `.artifacts/ci_compatibility/full/visual_review.json`
- Create: `.artifacts/ci_compatibility/full/report.md`

- [ ] **Step 1: Launch claim-grade clean-checkout execution**

Create a disposable clean clone at the verified commit, stage the ignored
immutable datasets at the manifest's expected relative path via a read-only
symlink, and verify all declared hashes before launch. Use one absolute output
root for every subsequent command:

```bash
SOURCE=/home/ollie/Documents/PtychoPINN/.worktrees/ci-compatibility-ablation
SHA=$(git -C "$SOURCE" rev-parse HEAD)
CLEAN=/tmp/ptychopinn-ci-compat-$SHA
DATA_ROOT=$SOURCE/.artifacts/ci_compatibility/datasets
RUN_ROOT=$SOURCE/.artifacts/ci_compatibility/full
git clone --shared "$SOURCE" "$CLEAN"
git -C "$CLEAN" checkout --detach "$SHA"
mkdir -p "$CLEAN/.artifacts/ci_compatibility"
ln -s "$DATA_ROOT" "$CLEAN/.artifacts/ci_compatibility/datasets"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$SHA" > "$RUN_ROOT/claim_commit.txt"
cd "$CLEAN"
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --output-root "$RUN_ROOT"
```

Expected: preflight verifies staged hashes, then four arms x three seeds x ten
epochs.

- [ ] **Step 2: Monitor to terminal state**

Do not leave training sessions running at turn completion. Resume only through the fingerprinted `--resume` path. Record any terminal run failures without deleting them.

- [ ] **Step 3: Inspect interpretable visuals**

Review the shared-limit reconstruction/error grid, training curves, seed plots, fitted scales, and stability dashboard. Record all required booleans and notes in `visual_review.json`, including its reviewed-figure hash.

- [ ] **Step 4: Import review and render final verdict**

```bash
SOURCE=/home/ollie/Documents/PtychoPINN/.worktrees/ci-compatibility-ablation
RUN_ROOT=$SOURCE/.artifacts/ci_compatibility/full
SHA=$(cat "$RUN_ROOT/claim_commit.txt")
CLEAN=/tmp/ptychopinn-ci-compat-$SHA
test "$(git -C "$CLEAN" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$CLEAN" status --porcelain --untracked-files=no)"
cd "$CLEAN"
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --output-root "$RUN_ROOT" \
  --visual-review "$RUN_ROOT/visual_review.json" \
  --resume
```

- [ ] **Step 5: Verify final evidence**

Assert the report distinguishes numeric and visual gates, synthetic truth from measurement consistency, all requested seeds from successful seeds, and bounded compatibility from model superiority.

- [ ] **Step 6: Record study outcome**

Add a concise durable finding to `docs/findings.md` only after the final typed verdict exists. Cite the manifest, verified commit, artifact root, seed table, physics gates, quality gates, and visual adjudication.

- [ ] **Step 7: Commit outcome documentation**

```bash
git add docs/findings.md
git commit -m "docs: record Hybrid ResNet CI compatibility evidence"
```

- [ ] **Step 8: Clean up the disposable clone**

After the final report and outcome commit are verified, remove only the exact
recorded `$CLEAN` path. Do not remove the shared dataset or artifact roots.

---

### Task 14: Correct Report Evidence And Publication Semantics

**Status:** Complete. Typed rendering, claim-grade eligibility, atomic publication,
and completed-report verification landed and were exercised by the final sealed
report.

**Files:**
- Modify: `scripts/studies/ablation/reporting.py`
- Modify: `scripts/studies/ablation/reporting_figures.py`
- Modify: `scripts/studies/ablation/runtime_study.py`
- Modify: `scripts/studies/ablation/runtime_attempts.py`
- Modify: `scripts/studies/ablation/manifest.py`
- Test: `tests/studies/test_torch_ablation_reporting.py`
- Test: `tests/studies/test_torch_ablation_runtime.py`
- Test: `tests/studies/test_torch_ablation_artifacts.py`

- [x] **Step 1: Write failing typed-plot tests**

Build fixture rows containing `measurement_consistency.varpro.s1/s2`,
`truth_quality.amp_mean_ratio`, `truth_quality.absolute_amp_nrmse`, VarPro
unit/fitted objectives, and zero reload error. Assert VarPro `s1/s2`, derived
`c_A/c_phi`, absolute-scale, objective-ratio, and annotated reload-error marks
exist and map to eligible run ids. Assert `training.log_grad_norm=false` cannot
empty either required figure.

- [x] **Step 2: Write failing completion/publication tests**

Assert eligible metrics with zero plotted marks fail publication; legitimate
inapplicability renders a visible reason. Assert `--rerun` invalidates old
visual review, figures, and sidecars. Inject failures before every root-level
replacement and prove rollback restores the prior bundle. Require frozen source
manifest/config, `invocation.json`, and `expansion.json` in the transaction and
completion hash list. Require every successful attempt to contain one-row CSV
and source manifest/config copies.

Assert the canonical seeds/epochs/full-family/full-arm invocation is
claim-grade eligible while `--epochs`, `--seeds`, `--only`, dataset replacement,
dirty checkout, or partial selection is sealed only as `NON_CLAIM_GRADE` and
cannot publish a bounded compatibility `PASS`.
Cover all closed reasons, including `external_dataset_spec` and
`manifest_budget_mismatch`; assert canonical reason ordering and
de-duplication. Assert `--output-root`, fingerprint-identical `--resume`, and a
matching visual-review import remain eligible. A terminal run failure must
change the verdict without changing protocol eligibility.

- [x] **Step 3: Run RED tests**

```bash
pytest -q tests/studies/test_torch_ablation_reporting.py \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_artifacts.py
```

- [x] **Step 4: Implement typed rendering and atomic publication**

Read scale values directly from metric records; do not use the unpopulated
`ReportRow.varpro_scales` side channel. Remove `dose_response.png` from required
artifacts. Stage the complete root bundle, validate figure eligibility, replace
the completion marker last, and restore it last on rollback. Distinguish
fingerprint-identical `--resume` from evidence-invalidating `--rerun`.
Expose `verify_completed_report(root)` and one public required-artifact set;
the verifier must reject missing, extra, or hash-mismatched sealed artifacts.

- [x] **Step 5: Run GREEN tests and commit**

```bash
pytest -q tests/studies/test_torch_ablation_reporting.py \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_artifacts.py
git add scripts/studies/ablation/reporting.py \
  scripts/studies/ablation/reporting_figures.py \
  scripts/studies/ablation/runtime_study.py \
  scripts/studies/ablation/runtime_attempts.py \
  scripts/studies/ablation/manifest.py \
  tests/studies/test_torch_ablation_reporting.py \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_artifacts.py
git commit -m "fix(studies): seal complete nonempty report evidence"
```

---

### Task 15: Materialize Dead Leaves And Canonical Lines Twins

**Status:** Complete. The four claim-grade twins are materialized at
`.artifacts/ci_compatibility/datasets_v2/` with closed provenance-v2 validation.

**Files:**
- Modify: `scripts/studies/materialize_ci_compatibility_datasets.py`
- Modify: `scripts/studies/ablation/datasets.py`
- Modify: `scripts/studies/ablation/dataset_provenance.py`
- Modify: `ptycho/diffsim.py`: add the explicit local-RNG path while preserving
  the existing omitted-RNG behavior
- Test: `tests/studies/test_ci_compatibility_materializer.py`
- Test: `tests/studies/test_torch_ablation_dataset_provenance.py`
- Test: `tests/torch/test_absolute_scaling_dataset_generation.py`
- Test: `tests/test_grid_lines_workflow.py`

- [x] **Step 1: Write failing two-family materialization tests**

Assert exactly four descriptors: `deadleaves_ci_3p5m`,
`deadleaves_legacy_amp`, `lines_ci_3p5m`, and `lines_legacy_amp`. Pin canonical
line generation to `ptycho.diffsim.mk_lines_img` plus the grid-lines
`set_phi=true` mapping (`dummy_phi` supplies phase). Assert deterministic local RNG, no ambient NumPy RNG
change, distinct object-family digests, shared cross-family coordinates/raw
probe, and within-family CI/legacy truth/coordinate identity.

Add a seeded equivalence fixture against the grid-lines workflow's lower-level
object helpers. Do not invoke `grid_lines_torch_runner.py`: it is a training
consumer with a different cached-NPZ schema, normalization, geometry, and
reassembly path. The dedicated materializer must change only morphology while
keeping compatibility-study physics and execution identical between families.
Use the design's exact N=640/nlines=400/crop-160/`dummy_phi` complex64 oracle,
`rtol=1e-6`, and `atol=1e-7`.

Define and validate closed `ci_compatibility_provenance_v2`. Claim-grade
preflight must recompute truth, coordinate, and probe array hashes from all four
staged NPZ pairs and enforce both within-family twin parity and mandatory
cross-family coordinate/raw-probe identity before expansion.
Implement every nested field/type/enum and unknown-field rejection exactly as
specified in the design; do not treat the provenance payload as an open dict.
Use bundle-relative split paths so the complete immutable directory can be
staged under a clean checkout while in-bundle relocation remains detectable.
Mark exact N=64/320, 5000/1250 output as `claim_grade`; all smaller test
materializations are `fixture` and cannot satisfy claim-grade eligibility;
validated bundle profile must propagate to the study invocation and emit the
closed `fixture_dataset` disqualification reason.

- [x] **Step 2: Run RED tests**

```bash
pytest -q tests/studies/test_ci_compatibility_materializer.py \
  tests/studies/test_torch_ablation_dataset_provenance.py \
  tests/test_grid_lines_workflow.py
```

- [x] **Step 3: Implement object-family selection and provenance**

Keep diffraction physics shared. Generate 5000/1250 positions for each family,
calibrate each CI twin independently to approximately 3.5M mean photons/image,
enforce the 1M weakest-frame floor and no saturation, and emit family-specific
source-object and output hashes. Never silently alias Dead Leaves as lines.
Stream one family/split at a time into private staging, batch diffraction, and
release arrays before the next split; do not retain all NPZ payloads or serialized
bytes in memory. Reject symlinked output roots. Publish with a recoverable
directory transaction that restores an interrupted backup before new work.

- [x] **Step 4: Run GREEN tests and materialize immutable N=64 data**

```bash
pytest -q tests/studies/test_ci_compatibility_materializer.py \
  tests/studies/test_torch_ablation_dataset_provenance.py \
  tests/torch/test_absolute_scaling_dataset_generation.py \
  tests/test_grid_lines_workflow.py
python scripts/studies/materialize_ci_compatibility_datasets.py \
  --output-root .artifacts/ci_compatibility/datasets_v2 \
  --detector-size 64 --train-positions 5000 --test-positions 1250 \
  --probe-src /home/ollie/Documents/PtychoPINN/datasets/fly/fly001.npz
```

- [x] **Step 5: Verify all hashes/statistics and commit**

```bash
git add scripts/studies/materialize_ci_compatibility_datasets.py \
  scripts/studies/ablation/datasets.py \
  scripts/studies/ablation/dataset_provenance.py ptycho/diffsim.py \
  tests/studies/test_ci_compatibility_materializer.py \
  tests/studies/test_torch_ablation_dataset_provenance.py \
  tests/torch/test_absolute_scaling_dataset_generation.py \
  tests/test_grid_lines_workflow.py
git commit -m "feat(studies): add canonical lines compatibility twins"
```

---

### Task 16: Expand To Five Arms Across Two Object Families

**Status:** Complete. The final manifest expands to 10 family-qualified logical
arms and 30 runs, with family-aware mandatory gates and non-gating diagnostics.

**Files:**
- Modify: `scripts/studies/specs/hybrid_resnet_ci_compatibility.toml`
- Modify: `scripts/studies/ablation/visual_review.py`
- Modify: `scripts/studies/ablation/verdicts.py`
- Modify: `scripts/studies/ablation/manifest.py`
- Modify: `scripts/studies/ablation/runtime_records.py`
- Modify: `scripts/studies/ablation/runtime_attempts.py`
- Modify: `scripts/studies/ablation/runtime_execution.py`
- Test: `tests/studies/test_torch_ablation_manifest.py`
- Test: `tests/studies/test_torch_ablation_verdicts.py`
- Test: `tests/studies/test_torch_ablation_reporting.py`
- Test: `tests/studies/test_torch_ablation_runtime.py`
- Test: `tests/studies/test_torch_ablation_runtime_records.py`

- [x] **Step 1: Write failing 30-run expansion tests**

Assert two object families, five valid arms per family, and three seeds produce
exactly 30 unique runs. Add CNN legacy NLL and keep CNN legacy MAE excluded.
Assert every profile resolves to the matching family dataset and that CI/VarPro
remain inactive for both legacy-NLL architectures.

- [x] **Step 2: Write failing family-gate and visual-schema tests**

Require separate Dead Leaves and lines Hybrid-CI seed, Pearson, SSIM, absolute,
physical-count, VarPro, reload, and manual-review gates. Overall compatibility
passes only when both family verdicts pass. Require the per-family median
matched-pair Hybrid CI/Hybrid legacy MAE Pearson ratio to be at least `0.70`
with at least two pairs; this is a mandatory bounded compatibility floor, not
an MAE-superiority claim. Also report diagnostic, non-superiority comparisons
for Hybrid CI/legacy NLL, CNN CI/legacy NLL, Hybrid CI/legacy MAE, and
Hybrid/CNN CI within each family. Require one visual-review record per family.

Finite physics and reload gates use `all_successful`: every successful
Hybrid-CI seed must supply a finite applicable operand. Add typed
`stability.reload_allclose`, computed over both fixed-batch texture and stitched
canvas with `rtol=1e-5, atol=1e-6`, and require it to equal one for every
successful seed. Missing/incomplete requested seeds remain `INCONCLUSIVE`;
missing or invalid operands on a successful seed are `FAIL`.

- [x] **Step 3: Run RED tests**

```bash
pytest -q tests/studies/test_torch_ablation_manifest.py \
  tests/studies/test_torch_ablation_verdicts.py \
  tests/studies/test_torch_ablation_reporting.py \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_runtime_records.py
```

- [x] **Step 4: Implement the manifest and family-aware verdict/report labels**

Use explicit valid-arm includes/excludes rather than generating invalid CI/MAE
or CNN/legacy-MAE combinations. Keep selectors parameterized by architecture,
physics profile, and dataset/object family; do not branch in the driver.

- [x] **Step 5: Run GREEN tests and dry-run**

```bash
pytest -q tests/studies/test_torch_ablation_manifest.py \
  tests/studies/test_torch_ablation_verdicts.py \
  tests/studies/test_torch_ablation_reporting.py \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_runtime_records.py
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml --dry-run
```

Expected: exactly 10 logical arms and 30 runs with family-qualified ids.

- [x] **Step 6: Commit**

```bash
git add scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  scripts/studies/ablation/visual_review.py \
  scripts/studies/ablation/verdicts.py \
  scripts/studies/ablation/manifest.py \
  scripts/studies/ablation/runtime_records.py \
  scripts/studies/ablation/runtime_attempts.py \
  scripts/studies/ablation/runtime_execution.py \
  tests/studies/test_torch_ablation_manifest.py \
  tests/studies/test_torch_ablation_verdicts.py \
  tests/studies/test_torch_ablation_reporting.py \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_runtime_records.py
git commit -m "study: cover Dead Leaves and lines compatibility arms"
```

---

### Task 17: Verify And Execute The 30-Run Claim-Grade Study

**Status:** Complete. Final claim commit
`6ace974f9ad9dbce210be7a03d1e13cc5075ef02`; final root
`.artifacts/ci_compatibility/full_v2/`; 30/30 runs succeeded on attempt 1 at 20
epochs in 43:10 (`2026-07-11T16:22:04Z`--`17:05:14Z`). The aggregate verdict is
**PASS** after both family reviews; `verify_completed_report` passes with
claim-grade eligibility, no disqualifiers, and a sealed completed review. The
reviewed grid SHA-256 is
`61f81c2bef9e654418596a90ec5e217f14aeafa989aba867105c980ab5c8f900`.

**Execution deviations:** The final evidence imported the completed review from
`.artifacts/ci_compatibility/reviews/visual_review-6ace974f9ad9dbce210be7a03d1e13cc5075ef02.json`
because the documented same-path review import still had a pre-fix recovery bug.
The same-path flow was fixed afterward by `b78020d3b` and `85bc8767c`, and
external-review symlink handling by `35fdae929`; these reviewed workflow fixes did
not modify claim artifacts. The prior successful but illegible-report execution
is archived at
`.artifacts/ci_compatibility/full_v2-superseded-legibility-220dd8dc285a65e1103f9648a48b588efed6e459/`
and is not final evidence.

**Files/Artifacts:**
- Preserve: `.artifacts/ci_compatibility/full/` as preliminary evidence under a
  claim-SHA-qualified `deadleaves_only_preliminary` archive
- Create: `.artifacts/ci_compatibility/full_v2/`
- Copy final visual outputs to: `tmp/ci_compatibility_visual_outputs_v2/`

- [x] **Step 1: Run focused verification**

```bash
pytest -q tests/studies/test_torch_ablation_manifest.py \
  tests/studies/test_torch_ablation_datasets.py \
  tests/studies/test_torch_ablation_configuration.py \
  tests/studies/test_torch_ablation_artifacts.py \
  tests/studies/test_torch_ablation_dataset_provenance.py \
  tests/studies/test_torch_ablation_metrics.py \
  tests/studies/test_torch_ablation_verdicts.py \
  tests/studies/test_torch_ablation_reporting.py \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_runtime_records.py \
  tests/studies/test_torch_ablation_driver_integration.py \
  tests/studies/test_ci_compatibility_materializer.py \
  tests/torch/test_absolute_scaling_dataset_generation.py \
  tests/torch/test_train_lightning_execution_contract.py \
  tests/torch/test_structured_reassembly_diagnostics.py \
  tests/test_grid_lines_workflow.py
```

Do not run repository-wide pytest or `ci/run_ci_tests.sh` under the current
session constraint. Record this limitation in the final report.

- [x] **Step 2: Run one-seed lines smoke and inspect all required figures**

```bash
SOURCE=/home/ollie/Documents/PtychoPINN/.worktrees/ci-compatibility-ablation
SPEC=scripts/studies/specs/hybrid_resnet_ci_compatibility.toml
cd "$SOURCE"
python scripts/studies/torch_ablation_driver.py --spec "$SPEC" \
  --only object_family=lines,architecture=hybrid_resnet,physics_profile=ci_nll \
  --seeds 3 --epochs 1 \
  --output-root .artifacts/ci_compatibility/smoke_lines_v2/hybrid_ci
python scripts/studies/torch_ablation_driver.py --spec "$SPEC" \
  --only object_family=lines,architecture=cnn,physics_profile=legacy_nll \
  --seeds 3 --epochs 1 \
  --output-root .artifacts/ci_compatibility/smoke_lines_v2/cnn_legacy_nll
python - <<'PY'
from pathlib import Path
from scripts.studies.ablation.reporting import verify_completed_report
for name in ("hybrid_ci", "cnn_legacy_nll"):
    verify_completed_report(Path(".artifacts/ci_compatibility/smoke_lines_v2") / name)
PY
```

Expected: both reports are visibly `NON_CLAIM_GRADE`; Hybrid CI VarPro and
dashboard panels contain typed marks; the legacy VarPro panel is visibly
not-applicable; semantic sidecars contain the selected run id.

- [x] **Step 3: Archive the preliminary root and create the clean checkout**

```bash
SOURCE=/home/ollie/Documents/PtychoPINN/.worktrees/ci-compatibility-ablation
PRELIM=$SOURCE/.artifacts/ci_compatibility/full
PRELIM_SHA=$(cat "$PRELIM/claim_commit.txt")
PRELIM_ARCHIVE=$SOURCE/.artifacts/ci_compatibility/deadleaves_only_preliminary-$PRELIM_SHA
test "$PRELIM_SHA" = 53623725bf294e26026c083940dbca5375677cca
test ! -e "$PRELIM_ARCHIVE"
mv "$PRELIM" "$PRELIM_ARCHIVE"

SHA=$(git -C "$SOURCE" rev-parse HEAD)
CLEAN=/tmp/ptychopinn-ci-compat-$SHA
DATA_ROOT=$SOURCE/.artifacts/ci_compatibility/datasets_v2
RUN_ROOT=$SOURCE/.artifacts/ci_compatibility/full_v2
test ! -e "$CLEAN"
test -f "$DATA_ROOT/ci_compatibility_descriptors.json"
git clone --shared "$SOURCE" "$CLEAN"
git -C "$CLEAN" checkout --detach "$SHA"
test -z "$(git -C "$CLEAN" status --porcelain --untracked-files=no)"
mkdir -p "$CLEAN/.artifacts/ci_compatibility" "$RUN_ROOT"
ln -s "$DATA_ROOT" "$CLEAN/.artifacts/ci_compatibility/datasets_v2"
printf '%s\n' "$SHA" > "$RUN_ROOT/claim_commit.txt"
```

- [x] **Step 4: Verify expansion and launch all 30 runs**

```bash
cd "$CLEAN"
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --output-root "$RUN_ROOT" --dry-run \
  > "$RUN_ROOT/dry_run.txt"
test "$(rg -c '^run ' "$RUN_ROOT/dry_run.txt")" -eq 30
rg -q '^runs 30$' "$RUN_ROOT/dry_run.txt"
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --output-root "$RUN_ROOT"
```

Expected: provenance-v2 preflight verifies all four staged datasets before
training; 10 logical arms x 3 seeds x 20 epochs reach terminal state.

- [x] **Step 5: Monitor every run to terminal state**

Use only fingerprinted `--resume`. A true `--rerun` must invalidate manual
review. Do not leave required training sessions running at turn completion.

```bash
cd "$CLEAN"
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --output-root "$RUN_ROOT" --resume
```

- [x] **Step 6: Perform family-scoped visual adjudication**

Review Dead Leaves and lines independently in the shared-limit grid. Also
inspect typed VarPro scales, derived contrasts, absolute-scale metrics,
objective ratios, reload errors, seed distributions, and training curves.
Complete both closed family records in `$RUN_ROOT/visual_review.json`; record
the reviewed reconstruction-grid SHA-256 and family-specific notes.

The documented in-place edit is imported transactionally by the next command.
The existing completion marker must remain present: the driver permits only the
validated `visual_review.json` bytes to differ from that seal, verifies every
other sealed artifact unchanged, then publishes a new fully valid seal.
This exception requires the literal in-root path to be a regular non-symlink
file. External review paths, including symlinks, remain ordinary unprivileged
imports and receive no exception from strict existing-report verification.

- [x] **Step 7: Import review, verify the seal, and export visuals**

```bash
cd "$CLEAN"
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/hybrid_resnet_ci_compatibility.toml \
  --output-root "$RUN_ROOT" \
  --visual-review "$RUN_ROOT/visual_review.json" --resume
python - <<PY
import json
from pathlib import Path
from scripts.studies.ablation.reporting import verify_completed_report
root = Path("$RUN_ROOT")
verify_completed_report(root)
expansion = json.loads((root / "expansion.json").read_text())
invocation = json.loads((root / "invocation.json").read_text())
assert len(expansion["selected_runs"]) == 30
assert invocation["claim_grade_eligible"] is True
assert invocation["claim_grade_disqualifying_reasons"] == []
PY

DEST=$SOURCE/tmp/ci_compatibility_visual_outputs_v2
mkdir -p "$DEST"
cp "$RUN_ROOT"/*.png "$DEST"/
cp "$RUN_ROOT/plot_metadata.json" "$RUN_ROOT/figure_row_mapping.json" \
  "$RUN_ROOT/report.md" "$RUN_ROOT/verdicts.json" "$DEST"/
test "$(git -C "$CLEAN" rev-parse HEAD)" = "$(cat "$RUN_ROOT/claim_commit.txt")"
rm -rf -- "$CLEAN"
test ! -e "$CLEAN"
```

The typed overall verdict may be `PASS`, `FAIL`, or `INCONCLUSIVE`, but `PASS`
is valid only when both family manual reviews and both Hybrid-CI numeric gate
sets pass. `verify_completed_report` checks the exact public required-artifact
set, every hash, invocation/expansion inclusion, and nonempty figure semantics.

Focused verification at reviewed implementation commit
`35fdae9295475c148eaaf8283fbafe13b5b6f2ef` reached `1191 passed, 27 warnings`.
The tracked [verification receipt](../../evidence/2026-07-11-ci-compatibility-focused-verification.json)
records the exact command, UTC interval, duration, exit code, summary, and SHA-256
of the complete workspace-local log at
`.artifacts/ci_compatibility/verification/final-focused-35fdae92.log`.
Repository-wide `pytest` and `ci/run_ci_tests.sh` were explicitly deferred under
the session constraint and remain a final limitation.

---

### Task 18: Replace Preliminary Finding With Two-Family Evidence

**Status:** Complete. The finding and index now route to the sealed two-family
evidence while preserving the Dead Leaves-only and illegible-report runs as
explicitly superseded history.

**Files:**
- Modify: `docs/findings.md`
- Modify: `docs/index.md`
- Modify: `docs/superpowers/plans/2026-07-09-ci-model-compatibility-ablation.md`

- [x] **Step 1: Correct preliminary evidence wording now**

Mark `CI-HYBRID-RESNET-COMPAT-001` as Dead Leaves-only preliminary evidence,
correct the figure path to the artifact-root top level, and state that it does
not satisfy the revised two-family contract.

- [x] **Step 2: Record final evidence only after Task 17 completes**

Replace the preliminary conclusion with the final two-family verdict, exact
claim commit, dataset and report roots, 30-run seed table, family metrics,
physics gates, and reviewed figure hashes. Preserve limitations: twenty epochs,
no superiority claim, no experimental-reference claim, and deferred full suite.

- [x] **Step 3: Verify docs consistency and commit**

```bash
! rg -n "Approved [f]our-arm|[e]xactly four logical arms|Expected: [1]2 valid runs|[f]our arms x three" \
  docs/superpowers/specs/2026-07-09-ci-model-compatibility-ablation-design.md \
  docs/index.md docs/findings.md
! sed -n '/^### Task 14:/,$p' \
  docs/superpowers/plans/2026-07-09-ci-model-compatibility-ablation.md | \
  rg -n "[f]our-arm|[e]xactly four logical arms|[1]2 valid runs|[f]our arms x three"
git diff --check
git add docs/findings.md docs/index.md \
  docs/superpowers/plans/2026-07-09-ci-model-compatibility-ablation.md
git commit -m "docs: record two-family CI compatibility evidence"
```

---

### Task 19: Withdraw Weak PASS And Correct The Gate Contract

**Status:** Complete 2026-07-11. Gate-contract implementation (88edbcfda,
quality re-review approved after the unavailable-center amend) plus the
performance-reference contract revision (6811ba251, review approved: SSIM
floors primary, hash equality diagnostic with mandatory classification,
CNN floors lock from a frozen artifact, bridge schema v3/evidence v2).

**Files:**
- Modify: `docs/superpowers/specs/2026-07-09-ci-model-compatibility-ablation-design.md`
- Modify: `scripts/studies/ablation/metrics.py`
- Modify: `scripts/studies/ablation/verdicts.py`
- Modify: `scripts/studies/specs/hybrid_resnet_ci_compatibility.toml`
- Modify: `tests/studies/test_torch_ablation_metrics.py`
- Modify: `tests/studies/test_torch_ablation_verdicts.py`

- [x] Add amplitude and phase quality gates, valid-mask variance/dynamic-range
  collapse gates, decoder-head saturation metrics, convergence evidence, bounded
  absolute-scale metrics, and a truth-forward Poisson-noise oracle.
- [x] Report quality both before and after VarPro. VarPro may improve the count
  objective without silently destroying recognizable texture; the historical
  post-VarPro lines Hybrid SSIM fell from about `0.765` to `0.693`.
- [x] Render separate structural/gauge-normalized and absolute-scale panels.
  Historical shared absolute limits correctly exposed legacy gauge mismatch but
  made structurally valid Hybrid controls appear featureless.
- [x] Add scan-utilization and canvas-coverage gates. The historical C=4 held-out
  grouping used only 756 of 1250 unique scans and covered about 71% of the
  canvas; successful completion may not hide dropped measurements.
- [x] Replace finite-only count/scale gates with predeclared numeric bounds.
- [x] Add CNN legacy MAE controls for both families. A ratio against a degraded
  control may not establish compatibility.
- [x] Require Hybrid ResNet and CNN reference-performance qualification before
  claim-grade matrix execution. SSIM is primary; internal hash equality is
  diagnostic rather than mandatory. (6811ba251: bridge schema v3, SSIM floors
  primary + MAE guard, three-class difference adjudication failing closed,
  LockedSsimFloors frozen-artifact mechanism; review approved 2026-07-11.)
- [x] Mark `.artifacts/ci_compatibility/full_v2/` and its visual review as
  historical diagnostic evidence; never mutate or silently reuse its seal.

Task 19 evidence:

- The checked matrix contains six arms per family (36 requested runs), with CI
  confined to rectangular Poisson NLL and CNN legacy-MAE controls present.
- Reassembly emits unique used/expected scan IDs and raw bounded-CNN decoder
  saturation fractions. Participating center/neighbor IDs are validated against
  all source IDs, while used centers are tracked independently against the
  bounds-eligible center set. Source utilization is unique participants / all
  1250 source scans; filtered utilization is used centers / eligible centers.
  Legacy grouped records without trustworthy centers carry `-1` plus an
  explicit unavailable flag; they retain source utilization but suppress
  center-derived metrics rather than reporting fabricated values.
  CNN real
  rails (`-0.8`, `+1.2`) and imaginary rails (`-1.2`, `+1.2`) are measured
  separately and each is capped at `0.05`. Synthetic truth-bearing CI attempts
  recompute the truth-forward Poisson floor and model/oracle error ratio from
  source arrays.
- Count metrics persist the exact ordered source-scan stream, including grouping
  multiplicity; the truth-forward oracle consumes that same stream and caches
  by dataset plus sample-policy digest. Structural grids normalize and crop on
  the common-valid mask. Saturation counters remain on device through the timed
  assembly loop and transfer once afterward.
- The current code records exact Run1084/probe/mask/boundary and dual-evaluator
  hashes. Under the revised contract these remain sealed provenance and
  debugging evidence. They do not gate reference qualification unless a
  difference invalidates the SSIM comparison, introduces hidden resize or
  undeclared gauge fitting, or changes the declared experimental condition.
- Lines gates require amplitude Pearson `>=0.90` and SSIM `>=0.75`. Every
  architecture/family CI arm must retain at least `0.85` of its matched legacy
  MAE control's amplitude SSIM and also pass an absolute SSIM floor of `0.50`.
  CNN legacy MAE arms are present for both families and CI remains inactive for
  MAE.
- Convergence evidence uses validation-loss slope, learning-rate trajectory,
  selected-checkpoint identity/epoch, and enabled gradient logging. The
  claim-grade budget and phase thresholds remain unlocked pending Tasks 20--22,
  so the manifest is explicitly claim-ineligible despite its diagnostic
  20-epoch setting.
- `.artifacts/ci_compatibility/full_v2/` was not modified and remains
  superseded diagnostic history.
- Verification after code-quality corrections: all ablation, structured
  reassembly, and loader-guard tests (`1281 passed`, 7 warnings), adjacent
  loader/probe/multimode/VarPro/reassembly regressions (`100 passed`, 1
  warning), scoped Ruff, byte compilation, and `git diff --check`.
- [ ] Re-approval after corrected focused/regression verification and review.

### Task 20: Hybrid ResNet And CNN Reference-Performance Qualification

**Status:** Complete 2026-07-11. Plumbing 038cd290c (review approved), CNN
floor freeze 46e2c0f14 (review approved), both GPU qualifications PASS
(evidence: .artifacts/reference_qualification/run1/, run2/).

**Files:**
- Modify: `scripts/studies/ablation/runtime.py`
- Modify: `scripts/studies/ablation/datasets.py`
- Create: `scripts/studies/specs/grid_lines_reference_performance.toml`
- Create: `tests/studies/test_grid_lines_reference_performance.py`

- [x] Run the `tests/torch/test_grid_lines_hybrid_resnet_integration.py`
  reference condition: N=128, gridsize/C=1, Run1084 probe, smoothing `0.5`,
  `pad_extrapolate`, no mask, Hybrid convolutional hidden scale `2.0`, central
  mask, legacy amplitude physics, MAE, seed 3, and five epochs.
  (2026-07-11 run1: PASS — amp SSIM 0.8984 / phase SSIM 0.9624 vs floors
  0.8408511/0.9404217; MAE 0.0741/0.1265 within guards.)
- [x] Run the sibling known-good CNN N=128, gridsize/C=1 grid-lines MAE
  condition through the study driver. Freeze the authoritative historical
  artifact and metric method before locking its floors; the existing result is
  approximately amplitude SSIM `0.886` and phase SSIM `0.928`.
  (2026-07-11: no on-disk historical artifact carried those values; per user
  decision the reference was recreated via the historical runner — the
  known-good recipe is the 20-epoch run — measuring amp SSIM 0.8541 / phase
  SSIM 0.9079 on the current NPZ materialization (drift vs 0.8864/0.9276
  documented). Frozen artifact sha 04c25f9d…91e0; atomic pin commit 46e2c0f14
  (review approved). run2 driver qualification: PASS with BIT-EXACT
  reproduction of the recreation metrics — strongest cross-path fidelity
  evidence. Interpretation caveat for Task 24: the resulting CNN amp floor
  0.8191 is more permissive than a faithful historical reproduction would
  imply (0.8514) and sits below the Hybrid floor.)
- [x] Use SSIM as the primary pass criterion. Hybrid floors are amplitude
  `0.8409` and phase `0.9404` from the integration fixture tolerances. Lock CNN
  amplitude/phase floors from its frozen reference artifact before execution.
  Retain MAE as a supporting guard, not the primary compatibility decision.
- [x] Record checkpoint, probe/mask/boundary configuration, patch/canvas/mask
  hashes, crop, gauge handling, and sample identities for diagnosis. Exact
  historical/generic equality is not required when the study-driver result
  passes the performance floors under a fair metric contract.
  (Sealed evidence: .artifacts/reference_qualification/run1/ and run2/.)
- [x] Prohibit hidden resizing, truth-dependent alignment, and undeclared gauge
  fitting. Any such operation invalidates the SSIM comparison.
  (Enforced fail-closed in plumbing + harness; verified in Task 20a review.)

### Task 21: One-Variable Configuration Bridge Ladder

**Status:** Complete (2026-07-12). Root cause identified and one-variable-confirmed:
probe batch-tensor rank (see ledger row 21 and the Task 25 design authority).
The checkboxes below are the historical execution contract; the rung 2-8 sweep
was superseded by the diagnostic sub-rung split (1a-1f) once rung 1 localized
the regression — the remaining rungs would have carried a bundled physics
defect and are replaced by Task 28's convergence validation under corrected
physics.

- [ ] Starting from the passing bridge, change only one group per rung:
  generic loader/schema; generic evaluator/reassembly; probe source/border
  transform; N=128 to N=64; C=1 to C=4 and training weighting; measurement
  domain/loss; rectangular scaling; inference VarPro.
- [ ] Keep the previous rung as a control and run seed 3 first. Record amplitude
  and phase metrics, patch quality, stitched quality, coverage, probe hashes,
  scaling quantities, and count error.
- [ ] Gate each rung on retained amplitude/phase SSIM relative to the immediately
  preceding passing rung. Internal tensor/hash differences explain a gap but do
  not independently fail a rung.
- [ ] Identify the first material degradation. Split a rung further if more
  than one effective value changed.
- [ ] In the C1-to-C4 rung, account for every source scan, duplicate use, group
  count, accepted patch, and reconstructed pixel. Correct grouping before
  evaluating model quality if scans are silently omitted.
- [ ] Prove inference reuses training normalization statistics. The historical
  runtime recomputed held-out statistics; preserve that run only as a measured
  diagnostic deviation.
- [ ] Promote only settings that retain the locked quality floor. Do not assume
  the current FLY/N64/C4 endpoint is valid because it completed.

### Task 22: Poisson/Rectangular Convergence And CNN Saturation

**Status:** Complete final. Slices A and B passed specification and
code-quality review; Slice C completed the smoke, all six 80-epoch trajectories,
manual compact review, and fixed common epoch-80 rule. The result exposed the
CNN contract failure now owned by Task 30; Task 23 remains blocked. Task 24
remains dependency-pending on Task 23. The
2026-07-13 proportionality review replaced the prior overbuilt milestone
subsystem plan with the three completed slices below.

**Proportional scope:** Task 22 answers one question: which single common epoch
budget is defensible for the corrected six-arm lines matrix, and do CNN arms
remain collapsed or saturated under corrected physics? It does not build a
general publication or checkpoint-evidence framework.

| Retained requirement | Problem solved | Required files/artifacts |
|---|---|---|
| Six-arm seed-3 lines manifest, gain 1, 80 epochs, milestones 5/20/40/80 | Prevents arm, data, gain, normalization, and budget drift | `scripts/studies/specs/grid_lines_ci_convergence.toml` using current pinned lines datasets |
| Optional exact-epoch capture from one trajectory | Shows convergence shape without four independent trainings | Four milestone checkpoints in each Task 22 run directory; ordinary best-checkpoint behavior remains unchanged |
| Canonical evaluator reuse | Prevents Task 22-only reconstruction semantics | Existing `runtime_execution.execute_canonical_run -> runtime_records` evaluator path |
| Compact trajectory table | Exposes the minimum convergence, quality, collapse, saturation, and CI-physics signals | One JSON and one CSV per arm |
| Compact reconstruction grid and review | Catches visible collapse/saturation hidden by scalar trends | One four-column grid and concise review per arm |
| Common-budget summary | Prevents per-arm or best-within-budget cherry-picking | One Task 22 summary linking the six arm outputs |

The checked manifest contains exactly Hybrid/CNN crossed with
`legacy_mae`, `legacy_nll`, and `ci_nll`, seed `3`, 80 epochs, and
post-epoch milestones `[5, 20, 40, 80]`. Every arm explicitly uses
`model.amplitude_physics_gain=1.0`; legacy remains explicit
`data.normalize="Batch"`; CI remains count/rectangular/Poisson NLL; CI+MAE is
invalid. Reuse the current pinned `lines_ci_3p5m` and `lines_legacy_amp`
bytes. Task 27 gain 16 and Task 28 `dictionary_parity` do not transfer.

Raw-patch metrics, Pearson/FRC, gradient norms, detailed residual plots, typed
N/A objects, per-milestone reload parity, generic milestone CLI overrides,
nested sidecar schemas/seals, per-visual hashes, gain sweeps, and architecture
changes are optional and nonblocking. They are not Task 22 completion criteria
and do not invalidate Task 22's diagnostic completion; Task 30 separately
blocks Task 23. Do not modify `ptycho_torch/reassembly.py` unless a
failing focused test or smoke proves existing CNN rail diagnostics insufficient.

#### Slice A: Milestone Capture And Evaluator Reuse

**Files:**
- Modify: `ptycho_torch/train_lightning_only.py` only for an optional additive
  exact-epoch callback hook.
- Modify: `scripts/studies/ablation/runtime_execution.py`.
- Modify: `scripts/studies/ablation/runtime_records.py`.
- Modify: `scripts/studies/ablation/runtime.py` only if the public facade must
  expose the optional milestone request/result.
- Test: `tests/torch/test_train_lightning_execution_contract.py`.
- Test: `tests/studies/test_torch_ablation_runtime.py`.
- Test: `tests/studies/test_torch_ablation_runtime_records.py`.

- [x] Write failing tests proving one fit captures post-epoch 5/20/40/80
  checkpoints and that the ordinary validation-best callback, selected
  checkpoint, main metrics, and verdict inputs are unchanged when capture is
  absent or enabled.
- [x] Write a failing call-order test proving every milestone reuses the existing
  strict checkpoint load and canonical held-out evaluator rather than duplicating
  inference.
- [x] Run RED:

```bash
pytest -q tests/torch/test_train_lightning_execution_contract.py \
  tests/studies/test_torch_ablation_runtime.py \
  tests/studies/test_torch_ablation_runtime_records.py \
  -k "milestone or best_checkpoint or checkpoint_evaluation"
```

- [x] Implement the smallest optional callback and evaluation loop that passes
  these tests. Keep the helper local unless extraction materially improves the
  existing file.
- [x] Run the same selector GREEN and commit Slice A.

Slice A completion evidence: implementation/no-drift commits `8728f448c` and
`cf46831e3`; CUDA peak isolation and prompt tensor-release fixes `e81afb1fa`
and `cbb6dc70b`. The final focused selector passed `9` tests with `163`
deselected; all three owned modules passed `172` tests. The required
integration marker passed `6` tests with `10` skips after provisioning the
worktree's pinned legacy fixture bytes. Specification and code-quality reviews
both APPROVED with no open findings.

#### Slice B: Six-Arm Manifest And Compact Outputs

**Files:**
- Create: `scripts/studies/specs/grid_lines_ci_convergence.toml`.
- Modify: `scripts/studies/ablation/manifest.py` only for the checked
  `diagnostics.milestones` field needed by this study.
- Modify: `scripts/studies/ablation/runtime_study.py`.
- Modify: `scripts/studies/ablation/reporting.py`.
- Modify: `scripts/studies/ablation/reporting_figures.py`.
- Test: `tests/studies/test_torch_ablation_manifest.py`.
- Test: `tests/studies/test_torch_ablation_reporting.py`.
- Test: `tests/studies/test_torch_ablation_driver_integration.py`.
- Modify: `scripts/studies/ablation/configuration.py` and its focused tests
  only to resolve the already-authorized explicit
  `model.amplitude_physics_gain` manifest field through the canonical
  configuration path.

- [x] Write failing manifest tests for exactly six seed-3 runs, 80 epochs,
  milestones 5/20/40/80, explicit gain 1, legacy `Batch`, CI count/NLL, and
  the current pinned lines dataset paths and hashes.
- [x] Write failing collation tests requiring one JSON/CSV trajectory pair per
  arm with exactly: epoch, validation loss, learning rate, amplitude SSIM, phase
  SSIM, stitched amplitude standard deviation, centered phase variance, CNN rail
  occupancy when available, and CI Poisson NLL, relative count error, and fitted
  scales when applicable. Blank cells are sufficient where a column does not
  apply; typed N/A objects are optional.
- [x] Write a failing rendering test requiring one compact four-column
  reconstruction grid per arm and a concise
  `recognizable/collapsed/saturated` review record.
- [x] Run RED:

```bash
pytest -q tests/studies/test_torch_ablation_manifest.py \
  tests/studies/test_torch_ablation_reporting.py \
  tests/studies/test_torch_ablation_driver_integration.py \
  -k "grid_lines_ci_convergence or trajectory or milestone_grid"
```

- [x] Implement only the checked manifest, compact tables/grids, and concise
  review. Preserve the canonical stitched path with no truth-dependent
  alignment, hidden resize, crop, gauge, placement, or weighting change.
- [x] Run the same selector GREEN and commit Slice B.

Slice B completion evidence: implementation commit `a7016ed6a`; focused
selector `13 passed`; the canonical gain-resolution/configuration module
`317 passed`; generic dry-run expanded exactly six seed-3 arms at gain 1 with
milestones 5/20/40/80; integration marker `6 passed, 10 skipped`.
Specification and code-quality reviews both APPROVED with no open findings.

Task 22 verification is scoped to the checked
`grid_lines_ci_convergence.toml` contract, the focused selector above, the
generic six-arm dry-run, Slice C smoke, and the resulting trajectories. A
failure that binds the explicitly superseded
`hybrid_resnet_ci_compatibility.toml` / `full_v2` protocol pin remains visible
historical lineage but is nonblocking here. Do not repin or suppress that
historical check during Task 22; Task 24 owns historical re-adjudication.


#### Slice C: Smoke And Execution

**Artifacts:**
- Create: `.artifacts/ci_compatibility/task22/milestone_smoke/`.
- Create: `.artifacts/ci_compatibility/task22/baseline_seed3/`.
- Create: one `task22_summary.md` under the baseline root.

- [x] Dry-run the checked manifest. Expected: six runs, seed 3, gain 1, 80
  epochs, milestones 5/20/40/80, current datasets, legacy `Batch`, and CI
  count/NLL.

```bash
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/grid_lines_ci_convergence.toml --dry-run
```

- [x] Run one selected arm for one epoch through the ordinary best-checkpoint
  path as the cheap training/evaluator smoke. This smoke does not need milestone
  output; focused Slice A tests prove exact-epoch capture.

```bash
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/grid_lines_ci_convergence.toml \
  --only architecture=hybrid_resnet,physics_profile=legacy_mae \
  --epochs 1 \
  --output-root .artifacts/ci_compatibility/task22/milestone_smoke
```

- [x] Run the checked manifest once from a clean verified checkout. Expected:
  six deterministic 80-epoch trajectories and four evaluated milestone
  checkpoints per arm.

```bash
python scripts/studies/torch_ablation_driver.py \
  --spec scripts/studies/specs/grid_lines_ci_convergence.toml \
  --output-root .artifacts/ci_compatibility/task22/baseline_seed3
```

- [x] Review the six compact grids and record only recognizable, collapsed, and
  saturated for each arm.
- [x] Write `task22_summary.md` linking the six trajectory tables and grids,
  recording failures honestly, and selecting one common budget/rule from
  5/20/40/80. Do not select per-arm budgets or best-within-budget checkpoints.
- [x] Task 22 completes only when the six runs, compact review, summary, and
  common budget/rule exist. The later CNN contract audit supersedes its original
  routing effect: Task 30, not Task 22 alone, now gates Task 23.

Task 22 completion evidence (2026-07-13): complete-family and milestone-budget
fixes landed in `aed71a5e1` and `e79c6084d`; nested milestone checkpoint reload
support landed in `ec76fbdcb` and `320adc2c0`. Focused re-verification passed
the two checkpoint-layout tests and three milestone-planning/execution tests;
all scoped reviews APPROVED with no open findings. The ordinary one-epoch smoke
completed under `.artifacts/ci_compatibility/task22/milestone_smoke/`. The
tracked baseline driver PID `357576` exited `0`; all six attempts contain
`completion.json`, no `failure.json`, milestone trajectory JSON/CSV at
5/20/40/80, and a milestone reconstruction grid. Manual review and the fixed
epoch-80 rule are recorded in
`.artifacts/ci_compatibility/task22/baseline_seed3/task22_summary.md`.

### Task 30: CNN Component And Physical-Lines Contract Recovery

**Status:** Complete final (`complete_final`) 2026-07-13. Depends on Task 22.

**Rationale:** Task 22 proved that additional epochs do not recover the CNN:
all three CNN profiles were saturated and collapsed by epoch 5 and remained so
through epoch 80. The failed protocol combined two invalid conditions. Its
lines object used raw `mk_lines_img` magnitude as physical transmission
(mean held-out patch amplitude `3.532`; `99.97%` outside the CNN box at unit
component scale), and its `C=4` instance of the generic `C>1` CNN path emitted
one real channel against four imaginary channels before implicit broadcast.
Git history provides no shared-amplitude scientific contract; the TensorFlow
object-big path requires one channel per patch for both components. Task 30
repairs both contracts before any multi-seed claim run.

Before any Task 30 run, set the checked claim lock to false; restore it only
after the corrected seed-3 run passes. Task 30 has three implementation steps:

1. **Fix the generic CNN output contract.** In `ptycho_torch/model.py` and its
   config resolution, both component heads emit `C_model` channels when
   `object_big=true` and one channel otherwise. Reject unequal component shapes
   before complex combination. Cover `C=1`, `C=2`, and `C=4` in focused model
   and adapter tests. Historical asymmetric checkpoints may use an explicit
   diagnostic-only override, but normal study runs may not broadcast.
2. **Fix the lines object and legacy normalization.** In the compatibility
   materializer, normalize `mk_lines_img` to `t in [0,1]` and construct
   `A=0.3+0.7*t`, `phi=0.5*(2*t-1)`, `O=A*exp(i*phi)`. CI and legacy twins keep
   identical truth, positions, probe geometry, and splits. Preserve old data as
   historical evidence and write the corrected data to a new root. Derive one
   physical legacy normalization from the exact sealed training input consumed
   by the qualification run and the exact forward:

   ```text
   P_eff = normalize_probe_like_tf(P_stored, probe_scale) / probe_scale
   A0 = fftshift(abs(sum_p FFT(O * P_eff[p]))) / N
   r = sqrt(N^2 / mean_samples(sum_hw(Y^2)))
   G_phys = r * sqrt(sum(Y^2) / sum(A0^2))
   ```

   The loss-side physics scale and scalar loss divisor cancel. Record the
   sealed input identities and the resolved probe, FFT, RMS, and loss factors;
   compute the expression once, not once per architecture or loss. For the
   sealed Task 30 v3 lines training archive (SHA-256
   `97e3933abf1ff27e443d1d0541e776ebb5e52c0d6edb2f2e3f2e3a744bdbf38f`)
   the result is `G_phys=12.452229360013307`. Verify it narrowly against the
   same sealed truth forward: amplitude least squares `12.450350059079451` and
   exact MAE weighted-median `12.450986331825641` must remain within relative
   `2e-4` of the Poisson expression. Freeze the derived value for both legacy MAE and
   legacy NLL across CNN and Hybrid ResNet. CI remains at gain 1; held-out test
   data, initialization RMS matching, reconstruction-quality selection, gain
   sweeps, boundary expansion, and architecture/loss-specific gain choices are
   inadmissible.

   The terminated `{1,4,16,64}` run under
   `.artifacts/ci_compatibility/task30/legacy_gain_calibration_seed3` exited
   `143` after its exact tracked PID was stopped. Preserve it as invalid
   diagnostic history; it cannot select or verify the physical normalization.
3. **Qualify the existing six roles.** Keep the Task 22 seed, probe, optimizer,
   epoch-80 budget, stitching, and metrics. Correct the active-support rail
   calculation so a fully railed 32x32 output inside a 64x64 frame reports near
   1.0, and retain the existing variance and gradient diagnostics. The six
   roles pass only if every CNN and Hybrid result is recognizable,
   non-collapsed, and non-saturated, and each CI result also passes count,
   VarPro, and reload checks. Pre-qualification evidence may remain split
   between fresh corrected completions and earlier already-correct metric
   records; never import or reseal those identities into one report.

**Primary files:** `ptycho_torch/config_params.py`, `ptycho_torch/model.py`,
`scripts/studies/materialize_ci_compatibility_datasets.py`, the ablation
configuration/metrics modules touched by those contracts, and their focused
tests. Do not add a new study driver or a second reporting schema.

**Task 30 completion gate:** focused tests pass, the sealed-input derivation and
narrow numerical verification above pass, and all corrected six-arm seed-3
roles pass the existing quality and physics gates. Then update the two-family
manifest hashes, the single derived legacy normalization, and protocol hash and
restore the claim lock in one change. If a CNN arm still collapses, diagnose
that arm directly before Task 23; do not expand the roadmap pre-emptively.

Task 30 completion evidence: the fresh full-support Hybrid arms under
`.artifacts/ci_compatibility/task30/corrected_lines_seed3_probe_big_true/`
have completion markers and amp/phase SSIM of `0.9420/0.99435` (legacy MAE),
`0.9411/0.99433` (legacy NLL), and pre/post-VarPro
`0.9423/0.9351` amplitude plus `0.99437/0.99431` phase (CI); CI physical-count
error is `0.08498` and reload error is zero. The already support-on CNN metric
records under `.artifacts/ci_compatibility/task30/corrected_lines_seed3/` are
retained as prior compatibility evidence, not merged into that identity:
legacy MAE `0.8335/0.98636`, legacy NLL `0.8231/0.98481`, and CI pre/post
amplitude `0.8309/0.7864`, phase `0.98758/0.98662`, count error `0.12746`,
zero head-saturation fractions, and exact reload. The source-manifest guard
prevented a common report identity; no completion marker was fabricated and no
scientifically redundant CNN rerun is required for this pre-qualification.

### Task 23: Corrected Multi-Seed Execution

**Status:** Complete final (`complete_final`) 2026-07-14 with a typed numeric
`FAIL`. Depends on Tasks 19--22, 27, and 30.

- [x] Verify Task 30's corrected manifest, protocol hash, datasets, probe
  transforms, semantic component contract, derived legacy normalization,
  convergence budget, and health/quality thresholds before launch. Pass the
  sealed Task 27 Hybrid reference evidence through the claim preflight; its
  exact SHA-256 is
  `2d297b391101909ebe9757359e28506a8130471a9ceade82a7a511a8e3527866`.
- [x] Launch all three seeds for both families under one coherent locked
  manifest. A failed mandatory arm
  yields FAIL or INCONCLUSIVE; it cannot be hidden as a diagnostic comparison.
- [x] Inspect training curves and all reconstruction/truth/error figures before
  importing manual review.

The coherent production execution at `.artifacts/ci_compatibility/full_v3/`
used source commit `5fcfd1e80967bb5458b1ee61a4c35fa77bcff753` and protocol
SHA-256 `dfd8720a8fcf2e432cff5fd917521ba90edd5092d80953c9c8343bb7ae95506b`.
The exact driver exited zero, all 36 requested arms completed, no arm failed,
and the report is claim-grade eligible and sealed. The locked report verdict is
nevertheless `FAIL`: DeadLeaves Hybrid CI missed the amplitude, physical-count,
and Poisson-oracle floors; its seed-29 post-VarPro phase SSIM was
`0.8995186708` against `0.90`; and DeadLeaves CNN CI/legacy-MAE amplitude SSIM
ratio was `0.8204951703` against `0.85`. Lines CI quality, all CNN saturation
gates, finiteness, and checkpoint reload passed. The figures contain no blank
or collapsed outputs, while the DeadLeaves amplitude reconstructions visibly
lose fine structure. Manual review remains explicitly pending and cannot
override the numeric `FAIL`.

### Task 24: Re-Adjudicate And Publish Corrected Evidence

**Status:** Pending and not eligible. Task 23 completed with numeric `FAIL`, so
the existing all-gates-pass publication condition is unsatisfied.

- [ ] Publish to a new immutable artifact root and preserve `full_v2` as
  superseded diagnostic history.
- [ ] Replace the finding and index language only after numeric, visual,
  bridge, and physics gates pass and the report seal verifies.
- [ ] State the exact relationship to the integration reference, the first
  degrading ladder rung, convergence evidence, remaining quality gap, and
  whether CNN compatibility was established or rejected.
- [ ] Run the focused suites plus the previously deferred repository CI command,
  unless the user explicitly re-establishes a narrower verification boundary.

### Task 25: Probe-Rank Physics Contract Enforcement

**Status:** Complete final (`complete_final`) 2026-07-12. Implemented in
commits `8e9c16a79..5a28d1d8a`; design, root-cause, spec-compliance, and quality
reviews approved. Evidence: `.superpowers/sdd/task-25-report.md`; design
authority: `docs/superpowers/specs/2026-07-12-probe-rank-physics-contract-fix-design.md`
§3, §8; reviewed root-cause commit: `2a9ee2ad9`.
This task explicitly authorizes edits to `ptycho_torch/model.py`
(`ProbeIllumination`) and `ptycho_torch/workflows/components.py`; TF-side
core files remain off-limits.

- [x] Enforce the documented probe layout in `ProbeIllumination.forward`:
  typed `ProbeLayoutError` on any sub-rank-5 probe (kills the silent
  batch-into-modes broadcast in every physics mode).
- [x] Migrate the inline-dataset amplitude branch to the documented
  per-sample layout (remove the `8b3d7a011` pre-82da7796 exception); extend
  the `82da77960` regression test to pin amplitude-mode layout.
- [x] Plumb `amplitude_physics_gain` (default 1.0) through config_params /
  config_factory / scaling contract / Lightning hparams; applied once in the
  amplitude forward; scaling contract rejects non-1.0 for
  `rectangular_scaled`/CI modes fail-closed.
- [x] Quantitative tie-back test: gain 16 at fixed weights reproduces the
  step-parity loss band (0.9361 -> ~0.0755).
- [x] RED-phase repo inventory of any other flat-probe producers bound for
  the model.
- [x] Add PROBE-RANK-001 to `docs/findings.md` (mechanism, `82da77960` /
  `8b3d7a011` history, cross-refs POISSON-NORM-001,
  TORCH-REASSEMBLY-SIGN-001); add the §3 contracts to the docs/specs shard
  and cross-reference from docstrings.

### Task 26: Amplitude Physics Gain Calibration

**Status:** Complete final (`complete_final`). Depends on Task 25. Design §4.
Execution commit `72816630ea76318c65641e1214953a965a3c0404`; evidence
`.artifacts/gain_calibration_v2_commit-72816630/sweep_summary.json`; report
`.superpowers/sdd/task-26-report.md`. Rule A fixed gain 16 was selected for the
locked legacy-amplitude reference regime: amp/phase SSIM
`0.8858652644`/`0.9618665959`. Rule B `4.2441268779` fell outside the quality
plateau `[16,16]`; the halt criterion passed and no confirmation run was
required. Tasks 22, 23, 27, 28, and 30 are `complete_final`; Task 24 is not
eligible under its all-gates-pass publication condition.

- [x] Predeclared sweep `amplitude_physics_gain in {1, 4, 16, 64}` on the
  exact reference recipe (dictionary flow, corrected emission, seed 3,
  5 epochs); record val-loss trajectory, amp/phase SSIM, decoder output
  statistics.
- [x] Evaluate init-time self-calibration (TF `intensity_scale` convention)
  against the sweep plateau; select per the design's decision criterion.
- [x] Halt criterion: chosen rule must reach amp SSIM >= 0.85 on the
  reference recipe or the fix phase returns to design.

### Task 27: Reference Re-Qualification And Floor Re-Pinning

**Status:** Complete final (`complete_final`) 2026-07-12. Depends on Task 26.
Design §5. Inclusive commits: `46cc4d5bf` through `7c221d7e1`.
Harness/preparation commits: `46cc4d5bf`, `c5dced526`, `a7e0dfa8b`,
`8aa239881`, and `7f0e3dde5`. Pin promotion/hardening commits: `532761c59`,
`a2bc9634d`, and `7c221d7e1`. Both reviews were APPROVED after fixes. Report:
`.superpowers/sdd/task-27-report.md`.

- [x] Re-qualify Hybrid (5-epoch) and CNN (20-epoch recreated recipe)
  references under corrected physics + calibrated gain via the Task 20
  harness; mandatory visual review passed for both fresh run2 outputs before
  any pin moved. Hybrid evidence:
  `.artifacts/reference_qualification/task27_gain16_hybrid_prequalification_run2/grid_lines_hybrid_resnet_reference/reference_evidence.json`
  (SHA-256 `17a323102950d64ad4a3712486f155205a7806b15c6c1e782c69ce84b3b21962`;
  measured amp/phase MAE `0.08168590068817139`/`0.12818376669684495`,
  SSIM `0.8858652644013688`/`0.9618665959387648`). CNN evidence:
  `.artifacts/reference_qualification/task27_gain16_cnn_prequalification_run2/grid_lines_cnn_reference/reference_evidence.json`
  (SHA-256 `bb069ce3ba1288e2ce34e2dc1ec8c9abccc16ae7c7723faca37c596eb189b251`;
  measured amp/phase MAE `0.08112537115812302`/`0.18809200960630929`,
  SSIM `0.8846891595066123`/`0.9150671199457723`).
- [x] Promote the sealed CNN candidate without clobber to
  `.artifacts/reference_qualification/task27_gain16_cnn_prequalification_run2/grid_lines_cnn_reference_floors.json`;
  exact SHA-256 `7b4b1d5b319031094979faa4aeb0c8ef884b23e8b2d21302235795b25caa7346`.
- [x] One atomic commit re-pinning all floors (both spec TOMLs; reference-
  performance, ablation-verdict, and ladder-control test pins), old->new
  values enumerated in the commit message and this ledger:
  Hybrid amp MAE max `0.0996316` -> `0.09668590068817139`; phase MAE max
  `0.1583743` -> `0.15318376669684494`; amp SSIM min `0.8408511` ->
  `0.8508652644013688`; phase SSIM min `0.9404217` ->
  `0.9468665959387648`. CNN amp MAE max `0.10603325754404068` ->
  `0.09612537115812302`; phase MAE max `0.21913804466287312` ->
  `0.21309200960630928`; amp SSIM min `0.8191430035603822` ->
  `0.8496891595066123`; phase SSIM min `0.8928981347342532` ->
  `0.9000671199457723`. CNN artifact path moved from
  `.artifacts/reference_qualification/grid_lines_cnn_reference_floors.json`
  to the Task 27 run2 path above; SHA moved from
  `04c25f9d4766d7758184c8b94a926b627a37d40fdb606f2b8d7e494fbf8391e0`
  to `7b4b1d5b319031094979faa4aeb0c8ef884b23e8b2d21302235795b25caa7346`.
  The strict Task 27 xfail was removed and the integration command now passes
  `--amplitude-physics-gain 16`.
- [x] Preserve superseded qualification runs and the old canonical CNN floor
  artifact read-only as diagnostic history. Task 20's recorded numbers above
  remain superseded historical authority and are not rewritten.
- [x] Final fresh post-pin controller requalification passed for both arms.
  Hybrid root:
  `.artifacts/reference_qualification/task27_gain16_hybrid_final_commit-7c221d7`
  (amp/phase SSIM `0.8858652644013688`/`0.9618665959387648`, MAE
  `0.08168590068817139`/`0.12818376669684495`; evidence SHA-256
  `2d297b391101909ebe9757359e28506a8130471a9ceade82a7a511a8e3527866`;
  visual SHA-256
  `11280df5ee0e8f90d9abe4e6e4b4b2afd4bae9d658d8ba131fb5e178ffc785d5`).
  CNN root:
  `.artifacts/reference_qualification/task27_gain16_cnn_final_commit-7c221d7`
  (amp/phase SSIM `0.8846891595066123`/`0.9150671199457723`, MAE
  `0.08112537115812302`/`0.18809200960630929`; evidence SHA-256
  `80035e854686343ff5ec5ed9b0d712efc4ac23b0012a33c012c5bb295779ce4a`;
  visual SHA-256
  `e5fcce8be6f83742079cac3fe5db7918abdda72f857ec4e2499f1c16468bfb70`).
  Manual review passed for both, including recognizable line morphology and
  no blank or malformed panels. Final visual and canvas hashes exactly match
  the approved pre-pin runs.

### Task 28: Bridge Convergence Validation And Ladder Trim

**Status:** Complete final (`complete_final`) 2026-07-12. Canonical
`rung1a_mmap_full_scanset` passed `absolute_ssim_delta_v1` under unit
`dictionary_parity` at
`.artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun`. Amp/phase
MAE is `0.07664361596107483`/`0.11914721730481034`; amp/phase SSIM is
`0.8913340876617375`/`0.9632217816205675`. Absolute deltas from rung0 are
`0.0054688232603687`/`0.0013551856818027`, within the locked `0.02`/`0.01`
gates. Verdict is `pass`; reason and protocol-failure reason are null. The
sealed rung evidence is
`.artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun/rung1a_mmap_full_scanset/rung_evidence.json`
(SHA-256 `a8741f93511cc90eb26777b1ff2cab0235ce812cff3661bad168632fce9ce711`),
and the report is
`.artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun/ladder_report.json`
(SHA-256 `2f95c0e4651a2a48449797f4a34aed557ce70962c8e82dced0d40b4aa95a0cf6`).
Inference reuses training normalization; training/inference normalization
SHA-256 values are
`757559a3e42ecab03b244ffc8202092948cc4b65547799e491c00b68d10f99a1` and
`279dbef649a7ddb2ca329c3977b652a541b8d0cd5cf6248212db7310953cfc36`.
The exact generic twin train/test SHA-256 values remain
`628cac77ef85c3927e3d5407f509556f054267e71e567aed67500b8de5f6ae4e` and
`17b2aea9a9deeb3ead2ab78771f19b33a2612b2666196e20dd45fa1a51f2275b`.
Implementation commits are `65c802e17`, `9c6b08ca2`, and `292b18be8`; spec
and code-quality reviews are APPROVED. Closeout report:
`.superpowers/sdd/task-28-report.md`.
Task 27 is `complete_final`.

The `absolute_ssim_delta_v1` gate uses locked 0.02 amplitude and 0.01 phase
thresholds. Fresh rung0 dictionary evidence is the checked current baseline
(PASS at seed 3, 5 epochs; amp/phase SSIM
`0.8858652644013688`/`0.9618665959387648`, MAE
`0.08168590068817139`/`0.12818376669684495`; evidence SHA-256
`155ee5961e31f9cf82c012d6bb61591bd776551f728d66bb19e0f3abee6ad298`;
approved visual SHA-256
`11280df5ee0e8f90d9abe4e6e4b4b2afd4bae9d658d8ba131fb5e178ffc785d5`).
The first, now historical, rung1a mmap run completed with amp/phase SSIM
`0.856505683935826`/`0.9498293416806348`. Its absolute deltas from rung0 are
`0.0293595804655428`/`0.01203725425813`, exceeding the locked `0.02`/`0.01`
gate. The verdict is `fail`, reason
`ladder_absolute_amp_ssim_delta_exceeded`, with no protocol failure. The CLI
returned `1` for this quality FAIL. Sealed evidence is
`.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1a_mmap_full_scanset/rung_evidence.json`
(SHA-256 `f7c3eba38f0a93529cc37315d1d3e42f43f1dd5b1e5b239300d68cdd71f9c132`);
the derived report is
`.artifacts/bridge_ladder/task28_gain16_seed3/convergence/ladder_report.json`
(SHA-256 `130096cf45fb9e193308f272c84b0179f1948d2c0abacf700634dfec762303c7`)
and names `rung1a_mmap_full_scanset` as the first material degradation.
The derived report uses repo-relative/logical spec and evidence paths so its
bytes do not depend on the checkout or output-root location.
Design §6.

The rung1a execution used datasets root
`.artifacts/bridge_ladder/task28_gain16_seed3/datasets`; dataset
`n128_run1084_generic` contains train `n=8978`, SHA-256
`628cac77ef85c3927e3d5407f509556f054267e71e567aed67500b8de5f6ae4e`, and
test `n=729`, SHA-256
`17b2aea9a9deeb3ead2ab78771f19b33a2612b2666196e20dd45fa1a51f2275b`.
They were generated from the dictionary source train/test with SHA-256
`c7615e11b2a500c891ed13be0747adba467b451a12e5a31dd18b7f338e89c916` and
`01dd9ff64d84e56b5950865640e79895d82813c0caa451f9552338c07a700699`.
The source paths are
`.artifacts/integration/grid_lines_hybrid_resnet/datasets/N128/gs1/{train,test}.npz`;
the twin outputs are
`.artifacts/bridge_ladder/task28_gain16_seed3/datasets/n128_run1084_generic/{train,test}.npz`.
Generation provenance is tracked at
`.artifacts/bridge_ladder/task28_gain16_seed3/datasets/n128_run1084_generic/generic_twin_provenance.json`
(SHA-256 `3f97e27de19a28eca85528893741e3558f035e338b8fda7c8a5f8636b8cbf569`);
the source probe archive SHA-256 is
`9f82cb9eb2c5a853764b98c1657b778600c0e90425296a7d1fdc6e8fdb53c906`.

Diagnostic conclusion: the dictionary and generic twin measurement arrays are
byte-identical. Dictionary batches use `rms_scaling_constant=1` and
`physics_scaling_constant=1`; old rung1a used
`DataConfig.normalize="Batch"` and produced RMS approximately `1.33047` and
physics scaling approximately `1.9797e-4`. Rung1c selected the existing
`normalize="None"` path, restored unit constants, and passed at amp/phase SSIM
`0.8913340876617375`/`0.9632217816205675`. Rung1d and rung1e sampler controls
also passed, exonerating sampler policy. The grid-lines simulator already owns
amplitude conditioning in `ptycho.diffsim.illuminate_and_diffract`, so the
canonical generic bridge must use `mmap_scale_convention="dictionary_parity"`.
This is dataset ownership, not a global-default change: `DataConfig` remains
default `Batch`, explicit loader/Batch remains valid for studies that request
it, and CI/count behavior is unchanged.

Immutable historical evidence paths and seals:

- old failing rung1a:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1a_mmap_full_scanset/rung_evidence.json`
  (SHA-256 `f7c3eba38f0a93529cc37315d1d3e42f43f1dd5b1e5b239300d68cdd71f9c132`);
- rung1c normalization:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1c_normalization_regime/rung_evidence.json`
  (SHA-256 `b9886b498880c35d4ef5e1a7c18b8c229e41704fd407879d431e2226e65940da`);
- rung1d sampler:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1d_sampler_shuffle/rung_evidence.json`
  (SHA-256 `6df72e84ece6203f8c76326b635dd4835abb59cb9d00757cd3de6d75cd47fcad`);
- rung1e sampler plus unit normalization:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1e_sampler_plus_unit_norm/rung_evidence.json`
  (SHA-256 `92f61a63870ea59993938e206aec52053f901907be7b23d6f9de4b76018cb897`).

Historical rung1f evidence remains proof of the old flat-rank mechanism only
and must not be rerun; its status is `historical_only`/non-runnable. It remains at
`.artifacts/bridge_ladder/seed3_split/rung1f_probe_layout/rung_evidence.json`
(SHA-256 `230b35b9511483e6e409ab5a3e611e925e7ba09fd22bf10c8b3efbbdb2aae324`).
Its `mmap_probe_batch_shape="dictionary_flat"` lever recreates the prohibited
rank and now raises `ProbeLayoutError`; explicit gain 16 plus accidental rank
gain is invalid.

- [x] Run rung0' (dictionary, corrected) and rung1a' (mmap full scan-set,
  corrected) at seed 3; gate |d amp SSIM| <= 0.02, |d phase SSIM| <= 0.01,
  and record the quality FAIL above.
- [x] On FAIL: reopen isolation with the ladder machinery; Tasks 22-24 stay
  blocked.
- [x] Diagnose normalization ownership from rung1c and exonerate sampler policy
  with rung1d/rung1e; preserve all old evidence paths and seals above.
- [x] Make the executable baseline convention `dictionary_parity`, archive the
  now-no-op diagnostic rung definitions from the current TOML, and explicitly
  restore loader/`Batch` at the CI/count rung. Do not tombstone historical
  parser/runtime support before convergence passes.
- [x] Run canonical `rung1a_mmap_full_scanset` under the new output root
  `.artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun`; preserve
  the historical root. The unchanged locked 0.02 amplitude and 0.01 phase
  absolute-delta gates passed.
- [x] Record the conservative retention decision after canonical PASS: rungs
  1c-1f are already archived from the current TOML. Retain the remaining
  rung1b-through-rung8 scaffold and historical injection/parser support until
  Task 29 producer retirement; do not tombstone them in Task 28. Task 28's
  regression adjudication is only rung0 versus canonical rung1a. This is a
  conservative retention decision, not remaining Task 28 work.

Tasks 22, 23, and 30 are `complete_final`. Task 23 ran under the corrected CNN
component, physical-lines, support, saturation, and physical-normalization
contracts and returned a sealed numeric `FAIL` with manual visual status
pending. Task 24 is not eligible under its existing publication condition.

### Task 29: Retire The Inline Dataset Producer

**Status:** Pending. Depends on Task 28 PASS. Executes after Task 24 (no
producer change while the corrected multi-seed matrix runs or before its
evidence is sealed). Coordinated with the refactoring roadmap (P3 moves) —
structural change to a tree shared by concurrent initiatives.

**Rationale:** the components.py inline Dataset `__getitem__` is a
hand-synced mimic of `PtychoDataset.__getitem__` (components.py:790); mimic
drift produced PROBE-RANK-001. A single emission producer eliminates the
drift class and obsoletes Task 28's standing two-run convergence gate.
Task 28's PASS evidence (|d amp SSIM| <= 0.02 dictionary vs mmap under
corrected physics) is the behavior-preservation bound for this refactor.

- [ ] Assess mmap materialization cost for in-memory/test flows; decide full
  retirement vs the fallback (extract one shared emission core called by
  both Dataset implementations — drift class still eliminated, but the
  convergence gate then stays).
- [ ] If full retirement: add an in-memory/NPZ ingestion API to
  `PtychoDataset`; make the sampler semantics decision explicit (inline
  RandomSampler vs mmap SequentialSampler — exonerated for quality by
  rungs 1d/1e, but batch composition changes are a deliberate choice, not
  plumbing); migrate callers (`base_api.py`, `grid_lines_torch_runner.py`,
  tests); delete the inline Dataset path from components.py.
- [ ] Re-run the re-pinned reference floors (Task 27 gates) as the
  behavior-preservation check; retire the Task 28 two-run gate on success
  (single producer, nothing left to converge).

## Final Verification Checklist

- [x] Generic driver contains no architecture-specific branch.
- [x] Synthetic lines and external experimental NPZ descriptors both dry-run through the same CLI.
- [x] CI activates only for unsupervised rectangular Poisson.
- [x] Legacy arms are explicit and their CI-only metrics are `not_applicable`.
- [x] The C=4 probe-weighted training reassembly and generic `C>1` component symmetry execute under the corrected full-support policy.
- [x] Canonical inference uses physical-probe VarPro and probe-weighted barycentric stitching.
- [x] Anchor-aware metrics do not call `eval_reconstruction` or resize.
- [x] Absolute, reference-agreement, and measurement-consistency namespaces cannot mix.
- [x] Structured diagnostics contain aggregate sufficient statistics, not final-batch bases.
- [x] Historical study artifacts pass fingerprints and required-artifact hashes.
- [x] Historical Dead Leaves and canonical lines matrices each have five arms and three requested seeds.
- [x] Historical CNN legacy NLL controls and the corrected CNN legacy MAE control are present.
- [x] Required VarPro/dashboard figures contain eligible marks or explicit not-applicable reasons.
- [x] `--rerun` cannot reuse prior figures or visual approval.
- [x] Invocation/expansion provenance is sealed before the completion marker.
- [x] Historical verdict is typed and sealed; its quality conclusion is now superseded.
- [x] Hybrid ResNet and CNN study-driver references pass their locked amplitude/phase SSIM floors.
- [x] One-variable ladder identifies and resolves the first material quality regression.
- [x] Every `C>1` CNN component representation emits equal per-patch branch shapes without implicit broadcast.
- [x] Canonical lines use the declared physical amplitude/phase mapping and CI/legacy twins share latent truth and geometry.
- [x] The legacy physical normalization is derived once per exact sealed training input/data-grid normalization contract from the exact forward, verified narrowly, and shared across loss profiles and architectures; CI gain remains 1.
- [x] Active-support rail occupancy is measured correctly and rejects saturated CNN output.
- [x] CNN lines outputs pass non-collapse and saturation gates in the accepted seed-3 qualification evidence.
- [x] Corrected seed-3 amplitude, phase, absolute-scale, and count-error gates pass; Task 23 owns the coherent multi-seed claim.
- [ ] New final verdict is `PASS`, `FAIL`, or `INCONCLUSIVE` with separate visual status.
