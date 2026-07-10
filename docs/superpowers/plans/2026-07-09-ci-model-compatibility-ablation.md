# CI Model Compatibility Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, schema-validated Torch ablation driver and use it to determine whether Hybrid ResNet is compatible with the manuscript CI contract under recognizable-quality, absolute-scale, and physical-consistency gates.

**Architecture:** Extend the production Torch runtime only where the accepted design proves missing capability: effective execution controls and structured VarPro/count diagnostics. Build the study layer as focused modules that resolve versioned TOML into authoritative Torch configs, select immutable synthetic or experimental dataset descriptors, execute the canonical training/checkpoint/mmap/reassembly path, and emit typed metrics, verdicts, and visual reports. Materialize a dose-compliant synthetic CI/legacy twin bundle separately, then run the four-arm, three-seed study and calibrated dose sweep.

**Tech Stack:** Python 3.11, PyTorch, Lightning, NumPy, SciPy, scikit-image, Matplotlib, TOML via `tomllib`, pytest, SHA-256/JSON provenance.

**Design authority:** `docs/superpowers/specs/2026-07-09-ci-model-compatibility-ablation-design.md`

**Repository constraint:** Preserve unrelated dirty-worktree files. Each implementation task owns only its listed files, follows TDD, receives spec-compliance and quality review, and lands as a separate commit.

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
- Create `scripts/studies/ablation/reporting.py`: CSV/JSON/report/visual aggregation.
- Create `scripts/studies/ablation/runtime.py`: architecture-neutral canonical train/reload/mmap/reconstruct execution.
- Create `scripts/studies/torch_ablation_driver.py`: thin CLI.

### Study Definition And Data

- Create `scripts/studies/materialize_ci_compatibility_datasets.py`: deterministic CI count-dose and legacy-amplitude twin materializer.
- Create `scripts/studies/specs/hybrid_resnet_ci_compatibility.toml`: checked four-arm study matrix and conditional gates.

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

Given fixture rows, assert generation of `report.md`, tidy CSV/JSON, verdict tables, reconstruction/truth/error grids with shared row limits, curves, seed plots, dose plots, scale plots, and stability dashboard. Assert all figure rows map to run ids and labels distinguish absolute/normalized/reference metrics.

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

### Task 10: Deterministic Compatibility Dataset Materializer

**Files:**
- Create: `scripts/studies/materialize_ci_compatibility_datasets.py`
- Create: `tests/studies/test_ci_compatibility_materializer.py`
- Read/reuse: `scripts/studies/make_synthetic_truth_datasets.py`
- Read/reuse: `scripts/studies/make_dose_ladder_datasets.py`

- [ ] **Step 1: Write failing deterministic materialization tests**

On N=16 fixtures, assert stable checksums, same latent object/coordinates across CI and legacy bundles, calibrated probe/count co-scaling for target means 432/864/1728, profile metadata, photon-floor and saturation statistics, and refusal to overwrite mismatched output.

- [ ] **Step 2: Run RED tests**

```bash
pytest -q tests/studies/test_ci_compatibility_materializer.py
```

- [ ] **Step 3: Implement the materializer using shared builders**

Do not duplicate diffraction physics. Materialize base CI train/test at 432, dose-only CI test twins at 864/1728, and normalized-amplitude legacy train/test twins from identical latent truth/coordinates. Emit the closed descriptor fields and provenance hashes.

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

### Task 11: Checked Hybrid ResNet CI Study Manifest

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
  --N 64 --train-count 512 --test-count 128 \
  --target-means 432,864,1728
```

Expected: descriptor/provenance report all CI images above one million photons, no saturation, and stable hashes.

- [ ] **Step 2: Write the checked TOML with explicit values**

Define the synthetic descriptors, all allowlisted base values, architecture and
physics-profile dimensions, two CNN legacy excludes, three seeds, CI/legacy
dataset pairing, dataset-relative selector targets, conditional truth and
experimental gates, status gate, paired ratio, dose CV, and manual review.
Reference-agreement gates use `on_missing_capability="not_applicable"` so an
experimental no-reference descriptor remains a valid shared-manifest input;
CI physical-consistency requirements retain `error`.

- [ ] **Step 3: Add manifest contract tests**

Assert dry expansion is exactly four logical arms and twelve runs; CI arms use
CI/count/rectangular/Poisson/VarPro; legacy arms use legacy/amplitude/amplitude-
path/VarPro-off; C=4/object_big/probe training weighting are effective; only CI
arms receive dose datasets. Replace the dataset with experimental-reference and
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

### Task 12: Full Verification Before GPU Execution

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

### Task 13: Execute And Adjudicate The Full Study

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
SOURCE=/home/ollie/Documents/PtychoPINN
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
epochs followed by CI-only dose inference.

- [ ] **Step 2: Monitor to terminal state**

Do not leave training sessions running at turn completion. Resume only through the fingerprinted `--resume` path. Record any terminal run failures without deleting them.

- [ ] **Step 3: Inspect interpretable visuals**

Review the shared-limit reconstruction/error grid, training curves, seed plots, dose response, fitted scales, and stability dashboard. Record all required booleans and notes in `visual_review.json`, including its reviewed-figure hash.

- [ ] **Step 4: Import review and render final verdict**

```bash
SOURCE=/home/ollie/Documents/PtychoPINN
RUN_ROOT=/home/ollie/Documents/PtychoPINN/.artifacts/ci_compatibility/full
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

## Final Verification Checklist

- [ ] Generic driver contains no architecture-specific branch.
- [ ] Synthetic lines and external experimental NPZ descriptors both dry-run through the same CLI.
- [ ] CI activates only for unsupervised rectangular Poisson.
- [ ] Legacy arms are explicit and their CI-only metrics are `not_applicable`.
- [ ] C=4 probe-weighted training reassembly is proven effective.
- [ ] Canonical inference uses physical-probe VarPro and probe-weighted barycentric stitching.
- [ ] Anchor-aware metrics do not call `eval_reconstruction` or resize.
- [ ] Absolute, reference-agreement, and measurement-consistency namespaces cannot mix.
- [ ] Structured diagnostics contain aggregate sufficient statistics, not final-batch bases.
- [ ] Full study artifacts pass fingerprints and required-artifact hashes.
- [ ] Final verdict is `PASS`, `FAIL`, or `INCONCLUSIVE` with separate visual status.
