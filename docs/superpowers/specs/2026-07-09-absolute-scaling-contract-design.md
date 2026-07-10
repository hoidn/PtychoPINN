# Absolute Scaling Contract Design

**Status:** Approved for implementation by the user on 2026-07-09.

## Purpose

Establish a manuscript-conformant absolute-scaling contract for the PyTorch
contrast-invariant (`rectangular_scaled`) path while keeping legacy amplitude
behavior available only through an explicit compatibility selection.

The contract distinguishes detector measurement consistency from the
object/probe gauge. A reconstruction is in absolute object units only when the
supplied physical probe is calibrated in the same gauge as the measured
detector intensity.

## Contract Profiles

These profiles govern the `rectangular_scaled` contrast-invariant path. The
amplitude forward remains the existing legacy amplitude pipeline and does not
activate CI scaling merely because default configuration objects contain the CI
field defaults. A rectangular workflow resolves and validates the profile
before data loading.

### Profile resolution

The two configuration fields are resolved independently, with missing values
defaulted before validation:

| `scale_contract_version` | `measurement_domain` | Result |
|---|---|---|
| absent | absent | `ci_intensity_v2` + `count_intensity` |
| `ci_intensity_v2` | absent | `ci_intensity_v2` + `count_intensity` |
| absent | `count_intensity` | `ci_intensity_v2` + `count_intensity` |
| `ci_intensity_v2` | `count_intensity` | valid CI |
| `legacy_v1` | `normalized_amplitude` | valid legacy |
| any other partial, contradictory, or unknown combination | any | fail |

Explicit legacy selection therefore requires both fields. The resolved fields
are persisted in `DataConfig` within Lightning checkpoint hyperparameters, run
configuration JSON, and generated dataset metadata. A checkpoint or dataset
without the fields defaults to CI. Loading a metadata-free legacy checkpoint
requires an operator override setting both legacy values; it is never inferred.

### `ci_intensity_v2` (default)

Each absent field first receives its CI default according to the profile table;
the resulting pair is then validated. New code MUST NOT infer legacy behavior
from missing metadata.

- `measurement_domain = "count_intensity"`.
- `measured_intensity` contains detector intensity/counts and is the only target
  used by rectangular Poisson loss and VarPro.
- `network_input = measured_intensity * rms_input_scale`.
- `rms_input_scale` targets the manuscript energy `(N / 2)^2`:
  `sqrt((N / 2)^2 / mean_{samples,channels}(sum_HW(measured_intensity^2)))`.
  It is one scalar per experiment, computed from that experiment's training
  split and reused for validation and inference.
- `mean_measured_intensity` is
  `mean_{samples,channels,H,W}(measured_intensity)`, one positive scalar per
  experiment. It is persisted with the profile and attached per sample as a
  broadcastable tensor.
- `probe_physical` is the unnormalized probe in the detector measurement gauge.
- `probe_training = probe_normalization * probe_physical` is used during
  training only.
- The training forward explicitly compensates probe normalization so its
  physical field is unchanged.
- VarPro uses `probe_physical`, the same effective probe mask as training, and
  the full detector frame. It MUST NOT fold the training output scale into its
  basis.
- Rectangular Poisson compares predicted intensity directly to
  `measured_intensity`.
- The Poisson rate is `clamp(predicted_intensity, min=1e-8)`. Nonnegative,
  finite real-valued observations are accepted; negative or non-finite values
  fail. The per-sample NLL sums `(C,H,W)`, divides by that experiment's detached
  `mean_measured_intensity`, then averages over the batch. The matching raw
  count NLL (sum `(C,H,W)`, mean batch, before the denominator) is logged
  separately.
- The profile is active only with Poisson NLL. `ci_intensity_v2` plus MAE is an
  invalid configuration and fails before data loading or model construction.

`TrainingConfig.torch_loss_mode` is the authoritative Lightning primary-loss
selector. Its value must be `"poisson"` for CI; Poisson is the code name for the
manuscript NLL. `ModelConfig.loss_function` is legacy metadata and cannot
override it. Supervised mode and every other primary loss are invalid for CI.
Auxiliary object regularizers remain allowed, but physical-intensity
normalization applies only to the Poisson data term.

### `legacy_v1` (explicit compatibility only)

- `measurement_domain = "normalized_amplitude"`.
- Existing amplitude-domain scale routing, RMS behavior, normalized-probe
  VarPro compatibility, and frozen parity behavior remain available.
- MAE training is supported only through this explicit legacy profile; the
  historical rectangular double-square behavior remains compatibility-only.
- Selecting `legacy_v1` must be explicit in configuration or an operator
  compatibility override.
- Outputs from this profile MUST NOT be labeled manuscript-CI absolute scale.

## Legacy Amplitude Adaptation Into CI

An adapter may convert a normalized-amplitude dataset into `ci_intensity_v2`
when a count-amplitude scale `S` is available:

```text
measured_intensity = (S * normalized_amplitude)^2
probe_physical = S * probe_in_normalized_amplitude_gauge
```

The conversion is performed once at the data boundary. The resulting batch
must not retain `S` in an overloaded generic physics-scale field.

## Batch Contract

The CI batch exposes unambiguous named fields:

- `images`: detector intensity used as the network source before input RMS
  scaling.
- `measured_intensity`: detector intensity used by loss and VarPro.
- `rms_input_scale`: input-only RMS factor.
- `probe_training`: `(B,C,P,H,W)` normalized training probe.
- `probe_physical`: `(B,C,P,H,W)` raw/physical inference probe.
- `probe_normalization`: broadcastable multiplier relating the probes.
- `experiment_id`: dataset identity for dataset-level scalars.
- `mean_measured_intensity`: the persisted per-experiment loss normalizer.

`physics_scaling_constant` is not consumed by `ci_intensity_v2`. Existing
writers may retain it only for `legacy_v1` compatibility until removed.

Shared and multi-mode probes always use the explicit `(B,C,P,H,W)` shape.
Incoherent modes are combined by summing per-mode intensities, never fields.

For each experiment, define the effective mask `M` and scalar `q` from the
pre-channel-expansion physical probe `(P,H,W)` using the existing joint
multi-mode normalization statistic:

```text
probe_norm = probe_scale * mean_{P,H,W}(|M * probe_physical|)
q = 1 / probe_norm
probe_training = q * probe_physical
```

`q` is stored once per experiment and its named CI field collates as
`(B,1,1,1,1)`, broadcastable over `(B,C,P,H,W)`. The deprecated tuple
`batch[2]` alias remains `(B,1,1,1)` and must be unsqueezed once before probe
use. Training forms `M * probe_training / q`, exactly equal to
`M * probe_physical`. VarPro forms `M * probe_physical` directly. The same mask
resolver is used in both locations.

## Existing-Field Migration

| Existing surface | `ci_intensity_v2` meaning | `legacy_v1` meaning |
|---|---|---|
| NPZ `diff3d` / `diffraction` | count intensity | normalized amplitude |
| `images` | count-intensity network source | legacy model source |
| `observed_images` | alias of `measured_intensity` | legacy amplitude target |
| `rms_scaling_constant` | read only as a deprecated alias when explicitly migrated; new writers emit `rms_input_scale` | unchanged |
| `probe_scaling` / tuple `batch[2]` | `probe_normalization = q`; retained as a tuple compatibility alias | unchanged |
| tuple `batch[1]` | `probe_training` with `(B,C,P,H,W)` | unchanged compatibility shape |
| `physics_scaling_constant` | ignored and rejected as a CI scale source | unchanged |

New CI TensorDict writers emit all named CI fields. Existing metadata-free
amplitude NPZs are interpreted as CI count intensity by default and therefore
must be loaded with both explicit legacy fields or through the explicit
amplitude-to-CI adapter. No value-range heuristic is permitted.

## Gauge And Dataset Requirements

Scaling a detector intensity while leaving the probe unchanged changes the
object/probe gauge. Synthetic builders that change dose must either:

1. scale the stored physical probe by the matching amplitude factor, or
2. mark the probe gauge as arbitrary and make no absolute object-amplitude
   claim.

Synthetic count generation starts from a noiseless expected intensity, applies
the requested dose, then draws Poisson counts. It must not rescale an already
Poisson-sampled amplitude and describe the result as an independently sampled
count measurement.

## Error Handling

- Missing profile fields receive their independent CI defaults and the resolved
  pair is validated exactly as specified in the profile-resolution table.
- `ci_intensity_v2` rejects `normalized_amplitude` without an explicit adapter
  scale.
- `ci_intensity_v2` rejects every non-NLL loss mode, including MAE.
- `legacy_v1` rejects rectangular absolute-scale claims at reporting surfaces.
- Loading a checkpoint known by provenance to use the legacy contract requires
  an explicit `legacy_v1` compatibility override when its profile metadata is
  absent. Without that override, the missing fields default to CI exactly like
  every other runtime configuration.
- Non-positive or non-finite measurement/probe normalization statistics fail
  before training.

## Inference Routing

The canonical absolute-scale inference path is
`reconstruct_image_barycentric` with VarPro enabled under `ci_intensity_v2`.
Simplified checkpoint inference must route through that path or identify its
stitched output as unscaled texture. It must not compute and discard an output
scale.

## Acceptance Evidence

The implementation is accepted only with independent, non-circular tests:

1. Known object plus calibrated raw probe produces a known detector intensity.
2. Loader and training forward reproduce that intensity after training-probe
   normalization.
3. Raw-probe VarPro recovers known `s1/s2` without the training output scale.
4. Recovered object scale is invariant across dose when the probe is calibrated
   with dose.
5. An unchanged-probe dose sweep is retained as a negative control and must
   produce `s1/s2` proportional to square-root dose.
6. Poisson optimization gradients are invariant to global photon scale after
   physical-intensity normalization.
7. Probe behavior is invariant to batch size and correct for multiple
   incoherent modes.
8. Training and inference use identical effective masking.
9. mmap, in-memory, and dict adapters produce equivalent CI batches for the
   same physical inputs.
10. Explicit `legacy_v1` reproduces frozen legacy fixtures.

## Documentation Ownership

- This design owns the implementation architecture and migration decision.
- `docs/specs/spec-ptycho-core.md` owns the normative measurement-domain and
  scaling formulas after implementation.
- `docs/specs/spec-ptycho-interfaces.md` owns the batch fields and shapes.
- `docs/DATA_NORMALIZATION_GUIDE.md` explains operator-facing profile usage.
- `docs/findings.md` records superseded historical interpretations and measured
  evidence without becoming the contract authority.

The implementation change is incomplete unless the core spec, interface spec,
normalization guide, and findings supersession notes are updated atomically in
the same implementation series.
