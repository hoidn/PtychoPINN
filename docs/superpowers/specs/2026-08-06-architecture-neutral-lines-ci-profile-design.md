# Architecture-Neutral `lines-ci` Synthetic Profile Design

## Purpose

Replace the architecture-named synthetic count-intensity profiles with one
active profile:

```text
profile        lines-ci
recipe_version lines-ci-v1
```

The profile owns the lines simulation recipe and coherent CI data/physics/loss
contract. Architecture remains part of the resolved model identity, defaults
to `cnn`, and is explicitly overrideable.

Apply the contract to `refactor`, `fno-stable`, and `refactor-internal` without
reintroducing branch-excluded architecture families or documentation surfaces.

## Authority and conflict

This design supersedes only the active-profile clauses that name or lock
`cnn-lines-ci`, `cnn-lines-ci-v1`, `hybrid-resnet-lines-ci`, or
`hybrid-resnet-lines-ci-v1`. Historical plans, commands, artifact paths, and
recorded results retain their original names as provenance.

The current inconsistency is a `semantic_conflict`: the count-intensity scale
contract and rectangular physics are architecture-independent, while active
synthetic profile names and, on `refactor`, profile locks make a generator
family part of CI identity. Architecture-specific grid support is a separate
capability question and must not be inferred from profile selection.

The applicable authority surfaces are:

- `ptycho/workflows/synthetic_config.py` for profile resolution and persisted
  workflow identity;
- `ptycho/workflows/synthetic_pipeline.py` and
  `ptycho/simulation/flat_acquisition.py` for exact stage and dataset identity;
- `ptycho_torch/scaling_contract.py` and the synthetic resolver's coherence
  checks for the CI contract;
- branch-native architecture registries and grouped-patch allowlists for
  architecture capability;
- `docs/specs/spec-ptycho-core.md` on branches where that normative spec is
  present.

`refactor` intentionally has no `docs/index.md`; this change must not create
one there. `fno-stable` and `refactor-internal` retain their full documentation
routing surfaces.

## Current branch state

| Branch | Active synthetic CI profile | Architecture behavior |
|---|---|---|
| `refactor` | `cnn-lines-ci` | defaults to and locks `cnn` |
| `fno-stable` | `hybrid-resnet-lines-ci` | inherits Hybrid ResNet, but permits architecture overrides |
| `refactor-internal` | none | must receive the final neutral CI profile directly |

All three branches have architecture-independent count-intensity and
rectangular-forward machinery. Their validated `gridsize > 1` surfaces differ:
`refactor` currently exposes CNN, while `fno-stable` and
`refactor-internal` expose CNN and Hybrid ResNet. Registry membership alone is
not evidence of grouped-workflow support.

## Public contract

### One active CI profile

There is exactly one active synthetic CI profile entry:

```python
_CI_PROFILE_NAME = "lines-ci"
_CI_RECIPE_VERSION = "lines-ci-v1"
```

The retired names are not registry entries, aliases, or accepted authoring
inputs. A request using a retired name fails with a targeted message:

```text
synthetic profile 'cnn-lines-ci' is retired; use 'lines-ci' and, when needed,
set model.architecture='cnn' explicitly
```

The equivalent message for `hybrid-resnet-lines-ci` names
`model.architecture='hybrid_resnet'`.

This preserves one active identity rather than maintaining parallel profile
names that resolve to the same contract.

### Defaults and locks

`lines-ci` uses the same default architecture on every branch:

```text
model.architecture = cnn
```

That value is a default, not a lock. File and explicit CLI values may replace
it with any architecture supported by the branch and compatible with the
requested grid geometry.

The profile locks only the semantic CI identity:

```text
simulation.scale_contract_version = ci_intensity_v2
simulation.measurement_domain      = count_intensity
model.physics_forward_mode         = rectangular_scaled
model.loss_function                = Poisson
training.torch_loss_mode           = poisson
training.nll                       = true
```

An authored contradiction fails before model construction. The following
remain overrideable defaults:

```text
model.architecture                 = cnn
model.rect_s1s2_init               = dose_closure
model.rect_s1s2_trainable          = true
training.gradient_clip_val         = 1.0
training.gradient_clip_algorithm   = norm
```

An explicit `rect_s1s2_init="ones"` remains the supported no-solve control.

### Effective object representation

The rectangular forward requires a real/imaginary object representation, not
a particular generator family. Synthetic validation resolves the effective
representation by architecture:

```python
if model.architecture == "cnn":
    effective_output = model.cnn_output_mode
else:
    effective_output = model.generator_output_mode
```

`lines-ci` supplies `real_imag` as the default for both knobs, but neither
irrelevant architecture-specific knob is a CI identity lock. Validation fails
when the selected architecture's effective output is not `real_imag` and names
the field that controls that architecture.

The remaining CI coherence requirements stay unchanged: unsupervised mode,
count-intensity data, rectangular physics, Poisson training, unit amplitude
physics gain, VarPro-enabled inference, and probe-weighted patch handling.

### Architecture and grid capability

Profile selection never grants architecture capability. One branch-owned,
named capability set is the source of truth for complete grouped-workflow
support and is consumed by both the synthetic resolver and the grid-lines
runner.

Initial capability sets remain evidence-conservative:

| Branch | Validated `gridsize > 1` architectures |
|---|---|
| `refactor` | `cnn` |
| `fno-stable` | `cnn`, `hybrid_resnet` |
| `refactor-internal` | `cnn`, `hybrid_resnet` |

At `gridsize=1`, `lines-ci` accepts every registered architecture whose
effective representation and architecture-specific constraints satisfy the
CI contract. `neuralop_uno` retains its existing `N=128`, `gridsize=1`, and
`real_imag` restrictions.

Adding FFNO, FNO, FNO Vanilla, or another family to a grouped capability set
requires branch-native evidence covering the actual registered generator,
grouped data, training, checkpoint reload, and position-aware reconstruction.
The profile rename does not claim that evidence.

## Configuration and CLI flow

JSON, YAML, TOML, and CLI inputs continue through the same resolver:

```text
config root profile / --profile
                 |
                 v
       select active profile
                 |
                 v
   apply file values, then CLI values
                 |
                 v
   enforce CI locks and effective output
                 |
                 v
  validate architecture/grid capability
                 |
                 v
 persist complete resolved workflow identity
```

Examples:

```bash
ptycho_synthetic --profile lines-ci
ptycho_synthetic --profile lines-ci --architecture ffno --gridsize 1
```

On branches where Hybrid ResNet is retained and validated for grouped data:

```bash
ptycho_synthetic \
  --profile lines-ci \
  --architecture hybrid_resnet \
  --gridsize 2
```

The profile name is an authoring convenience. Inference continues to consume
the model, data, training, and artifact identity persisted in the bundle; it
does not reselect `lines-ci`.

## Identity and historical artifacts

Fresh resolution persists and hashes exactly:

```text
profile        lines-ci
recipe_version lines-ci-v1
```

The full resolved workflow, including `model.architecture`, remains in the
digest. Two architectures selected under `lines-ci` therefore have different
workflow identities even though they share a profile.

No synthetic workflow schema bump is required because the record shape is
unchanged. The new profile and recipe strings deliberately create a new
semantic identity and digest.

Historical mappings retain their literal hashes. Existing
`resolved_workflow.json`, dataset manifests, stage manifests, commands, and
artifact directories are not rewritten or canonicalized. The old names remain
valid provenance but are not accepted for new authoring.

Because stage reuse compares profile and recipe identity exactly, a
`lines-ci` request does not resume an output root created as
`cnn-lines-ci-v1` or `hybrid-resnet-lines-ci-v1`. The resolver requires a new
output root and reports the identity mismatch. Users who must resume an old
pipeline use the historical code revision. Already completed model bundles
remain loadable for inference through their persisted bundle identity.

There is no retired-name alias and no legacy identity normalization in reuse
checks. This keeps the migration small and prevents an old architecture-named
recipe from being silently relabeled as the broader neutral recipe.

## Branch application

### `refactor`

- Replace `cnn-lines-ci` with `lines-ci`.
- Keep CNN as the unlocked default.
- Remove architecture and universal `cnn_output_mode` assumptions from CI
  locks; validate the effective representation.
- Centralize the existing CNN-only grouped-workflow capability without
  expanding it.
- Update public configuration, workflow, simulation, normalization, command,
  and top-level usage documentation. Do not create `docs/index.md`.

### `fno-stable`

- Replace `hybrid-resnet-lines-ci` with `lines-ci`.
- Override the inherited Hybrid ResNet base default with the common unlocked
  CNN default in the CI profile patch.
- Converge CI locks and effective-output validation with the reference design.
- Retain the validated grouped capability set `{cnn, hybrid_resnet}`.
- Keep historical Hybrid ResNet plans and evidence under their original name;
  update normative and current user-facing documentation.

### `refactor-internal`

- Add the final `lines-ci` profile directly; do not add an intermediate
  architecture-named CI profile.
- Port the settled count-intensity profile fields and coherence validation
  required by the branch if they are not present at its tip.
- Use the common unlocked CNN default while retaining explicit Hybrid ResNet
  selection and the validated grouped capability set.
- Update the complete routed documentation set, including `docs/index.md` and
  normative specs. Documentation exclusion rules do not apply to this branch.

## Feasibility boundary

The architecture-neutral profile at `gridsize=1` is already supported by the
underlying contracts:

- the synthetic resolver can express a coherent FFNO count-intensity,
  rectangular, Poisson workflow when profile locking is bypassed;
- the registered FFNO/FNO/FNO-Vanilla modules accept channel-count `C` and
  emit the shared real/imaginary adapter shape;
- the shared rectangular forward is architecture-independent;
- existing FFNO CI integration covers the count-intensity path at
  `gridsize=1`.

The same evidence is insufficient to claim complete non-CNN
`gridsize > 1` support. That remains behind the capability set until a separate
end-to-end gate passes.

## Validation and acceptance

The change is accepted only when all applicable assertions hold on each
branch:

1. `_PROFILES` contains one active CI name, `lines-ci`, and no retired CI
   names or aliases.
2. Fresh resolution persists `lines-ci` / `lines-ci-v1`.
3. Omitted architecture resolves to CNN on all three branches.
4. A registered non-CNN architecture resolves under `lines-ci` at
   `gridsize=1` with its effective `real_imag` knob.
5. Architecture contradictions are no longer rejected as profile-lock
   contradictions; unsupported architectures, output modes, and grid
   combinations still fail through their owning validators.
6. The branch-native grouped capability set is consumed consistently by the
   synthetic resolver and grid-lines runner.
7. `synthetic-lines-v1` or the branch's existing amplitude-profile identity
   remains unchanged.
8. Retired authoring names fail with migration guidance; historical files are
   not rewritten.
9. An old CI output root is not silently reused under `lines-ci`.
10. A completed historical model bundle still loads through the ordinary
    inference bundle path.
11. CLI, JSON, YAML, and TOML resolution produce the same canonical workflow
    for equivalent `lines-ci` inputs.
12. Current user-facing and normative documentation contains no active command
    using a retired name. Historical plans and evidence remain labeled as
    historical.
13. Integration tests run before comprehensive branch suites, and any
    integration failure is investigated before proceeding.

## Non-goals

- Maintaining selectable aliases for retired CI profile names.
- Rewriting historical artifacts or changing their literal digests.
- Resuming an old architecture-named output root under the new profile.
- Changing the count-emission, dose-closure, representative-sampling,
  `N/2` input-conditioning, Poisson normalization, or rectangular-forward
  equations.
- Enabling every registered architecture at `gridsize > 1`.
- Reintroducing architecture families excluded from `refactor`.
- Changing the default amplitude profile or its sealed identity.
