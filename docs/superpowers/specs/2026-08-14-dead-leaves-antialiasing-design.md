# Dead Leaves Anti-Aliasing Design

## Scope

Add opt-in four-sample anti-aliasing to the local Dead Leaves generator, expose
it through executable configuration and command-line paths, and make the public
Dead Leaves producer available on both `refactor` and `fno-stable`.

The implementation must preserve existing generated arrays when the option is
omitted or false. The `refactor` port includes only the deterministic Lines and
Dead Leaves producer behavior needed by its current single-object synthetic
workflow. Frozen object banks, multiple-object split machinery, and other
unrelated `fno-stable` features are out of scope.

## Public Contract

`SyntheticObjectConfig` gains one family-specific Boolean field, appended after
all existing fields:

```python
dead_leaves_anti_aliasing: bool = False
```

Structured JSON, TOML, and YAML configurations use:

```yaml
simulation:
  object:
    kind: dead_leaves
    dead_leaves_anti_aliasing: true
```

The public synthetic CLI exposes the same setting as
`--dead-leaves-anti-aliasing` and `--no-dead-leaves-anti-aliasing` through
`argparse.BooleanOptionalAction`. Because the parser suppresses omitted
defaults, an omitted flag does not override a config-file value.

Enabling the setting is valid only for the generated
`dead_leaves` / `dead-leaves-object-v2` pair. The synthetic workflow resolver
rejects it for Lines, the historical v1 compatibility recipe, and source-backed
`frozen-object-bank-v1` inputs, where rasterization has already happened and the
setting would otherwise be ignored. Explicit false is semantically the same as
omission. The canonical simulation serializer therefore omits the field when
false, preserving historical default configuration dictionaries and SHA-256
identities; it includes the field when true, which changes dataset identity.

The low-level generator reads the same choice from its existing option mapping:

```python
obj_arg={"anti_aliasing": True}
```

The direct mapping value must be a real Boolean. Its default is false. The
specialized `make_synthetic_truth_datasets.py` producer exposes the same Boolean
CLI switch, passes it through this mapping, and records the resolved mapping in
its existing provenance output.

## Rasterization

The false path retains the current OpenCV integer-mask drawing and Boolean map
updates without changing random draws or arithmetic.

The true path uses the fixed four subpixel positions used by Ptychodus:

```text
(1/3, 1/3), (1/3, 2/3), (2/3, 1/3), (2/3, 2/3)
```

Each leaf's geometry is selected once. The hard and sampled paths share the
same integer center and radius for circles and the same polygon vertices after
the existing NumPy-to-`int32` truncation. For output pixel `(y, x)`, a sample
offset `(u_y, u_x)` represents the point `(y + u_y, x + u_x)`. The equivalent
OpenCV draw translates the geometry by `(-u_y, -u_x)` before evaluating the
integer output lattice.

The translation uses one byte-defined fixed-point contract on both branches:

- `shift=8` and fixed-point scale `Q=256`;
- each translated coordinate is encoded with
  `int(numpy.rint((coordinate - offset) * Q))`;
- a circle radius is encoded as `int(radius * Q)` because the existing radius
  is integral;
- filled shapes use `LINE_8`, not `LINE_AA`; and
- the four sample masks are accumulated as integer counts before division.

Thus one third is represented as `85/256` and two thirds as `171/256`, avoiding
branch-specific floating-point-to-integer choices. The implementation divides
the accumulated count by four to obtain coverage in
`{0, 0.25, 0.5, 0.75, 1}`.

The sampling contract is anchored to
[Ptychodus `generate_dead_leaves_object` at commit `085dcd3`](https://github.com/AdvancedPhotonSource/ptychodus/blob/085dcd3c56b5f0c70aacc5e715967dafbf9e2e1a/src/ptychodus/api/object_gen.py#L368-L424).
The local implementation extends those four offsets to polygons while retaining
the local generator's existing quantized geometry.

For coverage `c`, the new material replaces the previous topmost material by
linear interpolation:

```text
beta  <- (1 - c) * beta  + c * leaf_beta
delta <- (1 - c) * delta + c * leaf_delta
```

This matches Ptychodus's four-sample sequential coverage semantics while also
covering the local generator's circles, oriented squares, rectangles,
triangles, and quadrilaterals. Anti-aliasing consumes no random values, so it
does not perturb leaf family, geometry, or material sampling. Its expected cost
is roughly four raster operations per leaf; the false path pays no such cost.

## Deterministic Producer Port

On `refactor`, add the narrow registered producer boundary already used on
`fno-stable` for:

- `lines` with `lines-object-v1`; and
- `dead_leaves` with `dead-leaves-object-v2`.

The Dead Leaves v2 producer retains independent named geometry and material RNG
streams derived from the existing object seed, the fixed Dead Leaves phase law,
and the existing fixed radius/iteration arguments. It additionally passes the
resolved `dead_leaves_anti_aliasing` value into `create_dead_leaves`.

The synthetic resolver derives the default recipe from `object.kind`, validates
the kind/recipe pair, and rejects unsupported pairs before simulation. Its
recipe-aware validation also rejects enabled anti-aliasing for any pair except
`dead_leaves` / `dead-leaves-object-v2`. The flat acquisition path builds its
truth object through that registry instead of its inline Lines-only constructor.
Object provenance records the recipe, producer symbols, source commit, realized
array hash, RNG identity, and phase identity. The simulation configuration hash
separately records an enabled anti-aliasing choice.

On `fno-stable`, retain its existing registry and route the new field through
the existing Dead Leaves builder. Both branches must produce the same v2 Dead
Leaves object for the same canonical settings and seed.

## Compatibility And Failure Behavior

- Omitted and explicit-false anti-aliasing preserve historical low-level Dead
  Leaves output exactly.
- Existing Lines behavior, default serialized configuration, and default
  simulation digest remain unchanged.
- True anti-aliasing is never silently ignored for another object family,
  historical compatibility recipe, or source-backed object bank.
- Invalid direct `obj_arg` values fail before generating leaves.
- No new dependency is introduced; NumPy and the existing OpenCV dependency are
  sufficient.
- No generic sampling-count abstraction or new recipe version is added. A
  configurable sample pattern should be introduced only if another supported
  mode is actually needed.

## Verification

Focused tests must establish:

1. omitted and explicit-false low-level calls are array-identical under the same
   random streams;
2. true anti-aliasing is deterministic, changes the hard-edged result, and
   produces quarter-step fractional coverage for both circle and polygon paths;
3. anti-aliasing consumes no additional RNG draws;
4. config-file and CLI values resolve to the same canonical field, explicit
   true changes the simulation digest, and the default dictionary/digest remain
   unchanged;
5. true with a non-Dead-Leaves kind, the v1 compatibility recipe, or a frozen
   object-bank recipe is rejected;
6. `refactor` accepts and executes the registered Dead Leaves producer instead
   of failing the current Lines-only preflight;
7. manifests bind the selected recipe, RNG and phase identities, enabled
   anti-aliasing setting, and realized object hash; and
8. the same seeded v2 producer settings yield equal object arrays on
   `refactor` and `fno-stable`; both branches carry the same fixed expected
   array hash for one small seeded anti-aliased fixture.

Run the focused generator, simulation-config, CLI, synthetic-workflow, and flat
acquisition tests on each branch before broader regression tests.
