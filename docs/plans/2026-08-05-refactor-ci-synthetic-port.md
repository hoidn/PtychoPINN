# Port the CI Synthetic-Generation Surface to `refactor`

**Status:** IMPLEMENTATION COMPLETE — revised 2026-08-05 after the user
selected the public-CNN option. The port-specific CPU, integration, CUDA, and
documentation gates pass. The branch-wide suite was also run and its unrelated
pre-existing/pruned-surface failures are classified below rather than hidden.
The public `refactor` branch keeps its existing generator boundary: this port
does not add `hybrid_resnet` or describe a CNN run using hybrid quality
evidence.

**Goal:** Make the public synthetic runner generate, train on, reconstruct, and
evaluate the count-intensity contract through a new `cnn-lines-ci` profile,
while keeping the existing `synthetic-lines-v1` workflow byte-identical.

**Architecture:** Add the count-intensity acquisition contract and a second
synthetic profile around the already registered `cnn` generator. Reuse the
existing Torch CI validation and dose-closure implementation, preserve raw
grouped detector counts at the training boundary, and make reconstruction
evaluation validate diagnostics for the resolved measurement domain.

**Source material:** Curate the applicable contract implementation from the
local `fno-stable` branch with `git show`; do not cherry-pick because the public
resolver and workflow structure differs. Relevant references are:

- `ptycho/simulation/flat_acquisition.py` for count emission and physical-probe
  manifest identity;
- `ptycho/workflows/synthetic_config.py` for coherent CI field locking;
- `scripts/simulation/synthetic_pipeline.py` for explicit CLI overrides;
- `ptycho/workflows/training.py` and
  `ptycho_torch/workflows/components.py` for the raw-count container bridge;
- `ptycho_torch/reconstruction_evaluation.py` for measurement-domain-specific
  diagnostic validation;
- `fno-stable:docs/plans/2026-08-04-ci-gauge-invariant-scaling.md` for the
  acquisition-gauge and dose-closure semantics.

The applicable public documentation routes from `README.md`,
`scripts/simulation/README.md`, `docs/CONFIGURATION.md`, and
`docs/workflows/pytorch.md`. This branch intentionally has no `docs/index.md`,
so this port must not recreate the pruned documentation hub.

---

## 1. Corrected gap inventory

| Missing piece | `refactor` state | Required result |
|---|---|---|
| Count-intensity flat acquisitions | Amplitude only | Poisson-realized counts, one training-derived count scale, CI-scaled physical `probeGuess`, and strict manifest fields |
| Public CI synthetic profile | Only `synthetic-lines` | Add `cnn-lines-ci`; keep `synthetic-lines-v1` bytes and digest unchanged |
| Contract CLI overrides | Six fields absent | Expose `--scale-contract-version`, `--measurement-domain`, `--physics-forward-mode`, `--cnn-output-mode`, `--torch-loss-mode`, and `--rect-s1s2-init` |
| Raw detector-count training bridge | `generate_grouped_data()` returns raw `grouped["diffraction"]`, but `_materialize_backend_container()` discards it when constructing the backend container | Attach exact grouped values as `raw_grouped_diffraction` and use them for CI Poisson training while preserving the legacy path |
| Count-domain evaluation | Assumes amplitude-domain marker | Require fitted, finite count diagnostics in count mode and retain the explicit legacy marker in amplitude mode |
| Pipeline manifest verification | Does not know the CI probe/count fields | Verify the physical-probe digest and count scale for CI; reject stray CI fields from amplitude artifacts |

The synthetic workflow already forwards `DataConfig`, `ModelConfig`, and
training fields through `_synthetic_factory_overrides()` into
`resolve_training_payload()`. It must not pass a redundant `profile="ci"`:
the resolved synthetic snapshot is already the authority, and the shared Torch
factory validates its coherence.

## 2. Contract design

### 2.1 Flat acquisition

For `measurement_domain="count_intensity"`:

1. derive one scale from the normalized-amplitude training split,
   `S = sqrt(nphotons / mean(sum(amplitude**2)))`;
2. transform every split to detector counts, `(amplitude * S) ** 2`;
3. store `probeGuess = probe_unscaled * S` as the CI-scaled physical forward
   probe, not the normalized model-input probe;
4. persist `count_amplitude_scale` and a digest of that stored physical probe;
5. apply the transformation before writing source/train/test artifacts so all
   persisted surfaces agree.

This fixes a deterministic acquisition gauge; it does not claim identifiable
physical calibration. The existing dose-closure startup record remains the
runtime diagnostic for a probe/object decomposition mismatch.

The legacy `legacy_v1` / `normalized_amplitude` path must remain byte-identical,
including its sealed array hashes and workflow identity.

### 2.2 Profile and CLI

Replace the single-profile hard rejection with a registry containing:

- `synthetic-lines`: the unchanged `synthetic-lines-v1` amplitude recipe;
- `cnn-lines-ci`: the exact new recipe version `cnn-lines-ci-v1`, using
  `architecture="cnn"` and the coherent
  contract set:

  ```text
  simulation.scale_contract_version = ci_intensity_v2
  simulation.measurement_domain      = count_intensity
  model.architecture                 = cnn
  model.physics_forward_mode         = rectangular_scaled
  model.cnn_output_mode              = real_imag
  model.loss_function                = Poisson
  model.rect_s1s2_init               = dose_closure
  training.torch_loss_mode           = poisson
  training.nll                       = true
  training.gradient_clip_val         = 1.0
  training.gradient_clip_algorithm   = norm
  ```

For `cnn-lines-ci`, `architecture`, `scale_contract_version`,
`measurement_domain`, `physics_forward_mode`, `cnn_output_mode`,
`loss_function`, `torch_loss_mode`, and `nll` are profile locks: explicit
matching values are accepted and contradictions fail closed. This prevents a
complete amplitude override from retaining a misleading CI profile identity.
`rect_s1s2_init` remains an overridable profile default so `ones` is available
as a control. The six CLI flags apply as explicit overrides and use the same
resolver validation as structured YAML/TOML/JSON input.

The new profile receives its own recipe and workflow digest. No new field may
appear in the serialized `synthetic-lines-v1` snapshot.

### 2.3 Training and evaluation

`RawData.generate_grouped_data()` already returns the unnormalized grouped
detector values as `grouped["diffraction"]`. The loss occurs in
`ptycho.workflows.training._materialize_backend_container()`, which must attach
an exact copy as `container.raw_grouped_diffraction` alongside normalized
`container.X`. `ptycho_torch.workflows.components._build_lightning_dataloaders()`
selects those raw values only when the resolved data contract is
`ci_intensity_v2` / `count_intensity`, and uses the training-derived
normalization statistics when adapting validation data. All legacy consumers
continue receiving their existing normalized representation.

Evaluation takes the resolved `measurement_domain` explicitly:

- count mode requires fitted, finite `relative_l2_intensity_error` and
  `mean_raw_poisson_nll` diagnostics with positive sample/pixel counts;
- amplitude mode retains its explicit `not_applicable` legacy diagnostic
  marker;
- unknown domains and legacy/deferred markers in count mode fail closed.

## 3. TDD and implementation sequence

- [x] Seal current `synthetic-lines-v1` 50-epoch and five-epoch payload bytes
  and digests before broadening accepted types.
- [x] Complete count-scale, count-emission, stored-probe, manifest, and
  amplitude-regression tests in `tests/test_flat_acquisition.py`; implement in
  `ptycho/simulation/flat_acquisition.py` and verify through
  `ptycho/workflows/synthetic_pipeline.py`.
- [x] Add resolver tests for both profiles, fail-closed mixed contracts,
  overrides, and distinct CI identities in
  `tests/test_synthetic_workflow_config.py`; implement the registry and widened
  literals in `ptycho/workflows/synthetic_config.py`.
- [x] Add CLI help/plumbing/rejection tests in
  `tests/scripts/test_synthetic_pipeline_cli.py`; implement the six flags in
  `scripts/simulation/synthetic_pipeline.py`.
- [x] Add raw-count boundary tests in `tests/torch/test_ci_container_bridge.py`
  and a workflow-level preservation test in
  `tests/test_training_workflow_initialization_summary.py`; implement the
  attachment in `ptycho.workflows.training._materialize_backend_container()`
  and contract-selective selection in
  `ptycho_torch.workflows.components._build_lightning_dataloaders()`.
- [x] Add count-domain evaluator tests in
  `tests/torch/test_reconstruction_evaluation.py`; implement the keyword and
  pipeline forwarding in `ptycho_torch/reconstruction_evaluation.py` and
  `ptycho/workflows/synthetic_pipeline.py`.
- [x] Run the focused CPU batteries after each TDD slice, then execute and
  classify the complete branch suite once the tree is settled.
- [x] Run a fresh five-epoch CUDA `cnn-lines-ci` workflow with
  `rect_s1s2_init=dose_closure`, then a matching `ones` control.
- [x] Update public runner/configuration/workflow documentation only after the
  executable contract is verified.

## 4. Acceptance evidence

### CPU contract gate

The following focused battery must pass freshly:

```bash
python -m pytest \
  tests/test_flat_acquisition.py \
  tests/test_synthetic_workflow_config.py \
  tests/scripts/test_synthetic_pipeline_cli.py \
  tests/torch/test_ci_container_bridge.py \
  tests/torch/test_reconstruction_evaluation.py -q
```

Then run the branch's complete supported CPU suite according to
`docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` when those
files are present. Collection failures must be classified against this
branch's supported surface rather than silently deleted or bypassed.

### GPU functional gate

CNN CI end-to-end completion is an open feasibility prerequisite because this
branch has no validated short-run CNN CI baseline. First run this reduced fresh
CUDA smoke test:

```bash
python -m scripts.simulation.synthetic_pipeline \
  --profile cnn-lines-ci \
  --output-root .artifacts/integration/cnn-lines-ci-smoke \
  --gridsize 1 --epochs 1 --batch-size 16 --seed 3 \
  --probe-source custom \
  --probe-path datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-transform 'pad_extrapolate:128|smooth:0.5' \
  --train-patterns 256 --test-patterns 64 \
  --train-raw-selection 256 --training-groups 256 --validation-groups 64 \
  --neighbor-count 1 --neighbor-pool-size 1 --groups-per-center 1 \
  --photons-per-pattern 1e9 \
  --rect-s1s2-init dose_closure \
  --gradient-clip-val 1.0 --gradient-clip-algorithm norm \
  --plateau-factor 0.5 --plateau-patience 2 --plateau-threshold 0.0 \
  --accelerator cuda --devices 1 --precision 32-true --workers 0 \
  --logger csv --deterministic
```

If this fails because the stock CNN cannot produce an evaluable
reconstruction, record that as the missing CNN capability and stop; do not
weaken the reconstruction/evaluation contract or import hybrid evidence. If it
passes, run the exact five-epoch functional recipe:

```bash
python -m scripts.simulation.synthetic_pipeline \
  --profile cnn-lines-ci \
  --output-root .artifacts/integration/cnn-lines-ci-5ep-dose-closure \
  --gridsize 1 --epochs 5 --batch-size 16 --seed 3 \
  --probe-source custom \
  --probe-path datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --probe-transform 'pad_extrapolate:128|smooth:0.5' \
  --train-patterns 4489 --test-patterns 729 \
  --train-raw-selection 4489 --training-groups 4489 --validation-groups 729 \
  --neighbor-count 1 --neighbor-pool-size 1 --groups-per-center 1 \
  --photons-per-pattern 1e9 \
  --rect-s1s2-init dose_closure \
  --gradient-clip-val 1.0 --gradient-clip-algorithm norm \
  --plateau-factor 0.5 --plateau-patience 2 --plateau-threshold 0.0 \
  --accelerator cuda --devices 1 --precision 32-true --workers 0 \
  --logger csv --deterministic
```

Acceptance requires:

> **Historical acceptance record:** This section records the prefix-era run
> executed on 2026-08-05. Its v1 requirement was correct for that run but is not
> the current producer contract. Fresh runs now emit strict
> `rect-s1s2-initialization-v2`; valid prefix-era v1 records remain readable
> without rewrite.

- exit code zero and all four stage-manifest-v2 stages freshly complete;
- resolved `cnn` architecture and coherent CI contract persisted;
- count-domain dataset and evaluation diagnostics finite and mode-matched;
- a strict `rect-s1s2-initialization-v1` record with mode `dose_closure`, a
  positive finite solved gauge, and 256 sampled patterns;
- reload/reconstruction completes from the saved bundle without contract drift.

Run the same five-epoch command into
`.artifacts/integration/cnn-lines-ci-5ep-ones`, changing only
`--rect-s1s2-init ones`. Acceptance requires an explicit `ones`
initialization record (`solved_gauge=1`, zero sampled patterns) and successful
completion.

There is deliberately **no SSIM floor or expected solved-gauge value** for
these CNN runs. The public guide already records that short N=128 count-Poisson
CNN training is collapse-prone without the complete TF-parity preset, and this
port does not expose that separate runtime preset. The hybrid-resnet
0.78/0.93 floors and approximately 0.815/0.939 results are not evidence for
the CNN profile. Establishing and sealing a CNN quality baseline is separate,
user-gated work.

### Recorded execution evidence (2026-08-05)

The final focused contract battery passed with `202 passed` in 88.17 seconds. The
supported integration selection passed with `3 passed, 10 skipped, 3136
deselected` in 180.20 seconds after two independently required cleanup fixes:
the unsupported external-orchestrator runtime test was retired, and the
grid-lines runner was updated to construct the nested sampling, optimizer, and
scheduler records expected by the current public configuration model.

The fresh CUDA runs completed all four `synthetic-stage-manifest-v2` stages:

| Run | Initialization record | Amplitude SSIM | Phase SSIM | Relative L2 intensity error |
|---|---:|---:|---:|---:|
| one-epoch feasibility smoke | `dose_closure`, gauge 3.036541, 256 samples | 0.215032 | 0.839536 | 0.389526 |
| five-epoch functional run | `dose_closure`, gauge 3.115810, 256 samples | 0.650057 | 0.896875 | 0.070796 |
| five-epoch control | `ones`, gauge 1.0, 0 samples | 0.236867 | 0.846586 | 0.416824 |

The two five-epoch runs used byte-identical train and test NPZs, so their only
intended contract difference was initialization. These measurements are
functional evidence, not a quality baseline.

After correcting the evaluation serializer to retain fitted count evidence
instead of emitting the legacy not-applicable marker, a fresh repeat of the
one-epoch smoke completed all four stages at
`.artifacts/integration/cnn-lines-ci-smoke-metric-validity`. Its persisted
`metric_validity.count_diagnostics` has `status=complete` and exactly matches
the finite error, NLL, sample-count, and pixel-count values validated from
`reassembly.count_metrics`.

The unfiltered branch suite completed with `2956 passed, 154 failed, 27
skipped, 11 xfailed, 1 xpassed` in 679.99 seconds. None of the failures exercises
the new CI acquisition/profile/raw-count/evaluation contract:

- 132 failures target pruned orchestration, paper-authority, workflow, study,
  or absent-documentation surfaces;
- 20 reproduce in older callers and tests that still use flat
  `TrainingConfig` fields after the earlier nested-config refactor; the relevant
  schema and caller files were untouched by this port;
- one pre-existing TensorFlow generator test expects its input configuration
  not to receive resolved model defaults and fails by itself;
- one Matplotlib figure-isolation assertion failed only in the monolithic run
  and passed in a fresh process.

The full log is retained at
`.artifacts/test-runs/refactor-ci-full-20260805-2.log`. The non-port failures are
not promoted into this plan's scope or represented as a green branch-wide
suite.

## 5. Documentation gate

After executable verification:

- document `cnn-lines-ci`, its six explicit overrides, the count-domain NPZ
  meaning, and the two initialization modes in `scripts/simulation/README.md`;
- update the synthetic examples and routing in `README.md`;
- replace the cross-branch CI recipe pointer in `docs/CONFIGURATION.md` with
  the native CNN functional example and its no-quality-baseline caveat;
- reconcile `docs/workflows/pytorch.md` so the data-contract section admits
  both legacy amplitudes and count-intensity NPZs and does not imply the CNN
  functional run inherits hybrid quality;
- perform a focused text sweep for stale current claims involving
  `hybrid-resnet-lines-ci`, the public generator emitting amplitudes only, or
  the old cross-branch recipe.

## 6. Non-goals and guardrails

- No `hybrid_resnet` or other excluded architecture port.
- No raster/`--scan-position-layout` work.
- No changes to protected physics modules.
- No change to the sealed `synthetic-lines-v1` identity or legacy arrays.
- No CNN quality threshold or quality-gate pin.
- No push beyond the local repository.
- No recreation of documentation files intentionally absent from public
  `refactor`.

## 7. Risks

- **Legacy identity drift:** widening live literals or adding registry data can
  accidentally alter serialization. The committed byte/digest tests are the
  first gate.
- **Raw/count ambiguity:** using normalized `container.X` for CI makes the
  Poisson objective numerically wrong even when config resolution looks
  coherent. The bridge tests must inspect the actual attached values.
- **Probe convention ambiguity:** `probeGuess` is the scaled physical
  acquisition probe for these flat CI artifacts, not the normalized model
  input. Tests and user docs must say so explicitly.
- **CNN convergence:** a functional five-epoch run may be low quality. That is
  expected under the current public runtime surface and cannot be promoted to
  a quality claim without a separately validated preset and predeclared gate.
