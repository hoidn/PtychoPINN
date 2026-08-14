# Dead Leaves Anti-Aliasing and Producer Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in Ptychodus-style four-sample anti-aliasing to Dead Leaves generation and expose the same deterministic Dead Leaves v2 producer on `refactor` and `fno-stable`.

**Architecture:** Keep the existing integer hard-mask path byte-for-byte when the option is omitted or false. Add one fixed four-sample coverage branch inside the shared low-level generator, carry a strict Boolean through canonical simulation config and CLI resolution, and route public generation through the registered kind/recipe producer. Port only Lines v1 and Dead Leaves v2 producer behavior to `refactor`; adapt the existing richer registry on `fno-stable` without removing its v1 or frozen-bank compatibility paths.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, Pydantic dataclass adapters, `argparse.BooleanOptionalAction`, pytest, Git.

---

## File map

Shared behavior on both branches:

- Modify `ptycho_torch/datagen/objects.py`: strict low-level option validation, split geometry/material RNG support, and four-sample rasterization.
- Create `tests/torch/test_dead_leaves_antialiasing.py`: branch-identical low-level behavior, RNG, coverage, and golden-hash tests.
- Modify `ptycho/config/config.py`: append the canonical Boolean field and conditionally serialize only `true`.
- Modify `ptycho/workflows/synthetic_config.py`: resolve recipes from object kind and reject invalid anti-aliasing/recipe pairs.
- Modify `scripts/simulation/synthetic_pipeline.py`: public Boolean CLI and nested config mapping.
- Modify `scripts/studies/make_synthetic_truth_datasets.py`: specialized CLI forwarding and provenance.
- Modify `tests/test_simulation_config.py`, `tests/test_synthetic_workflow_config.py`, `tests/scripts/test_synthetic_pipeline_cli.py`, and `tests/torch/test_absolute_scaling_dataset_generation.py`: public contract coverage.
- Modify `docs/CONFIGURATION.md` and `scripts/simulation/README.md`: user-facing config and producer contract.

`refactor` producer port:

- Create `ptycho/simulation/object_producers.py`: narrow Lines v1 and Dead Leaves v2 registry copied from the corresponding `fno-stable` behavior.
- Modify `ptycho/simulation/flat_acquisition.py`: replace the inline Lines-only constructor with the registry and record RNG/phase identity.
- Modify `ptycho/workflows/synthetic_pipeline.py`: validate dynamic producer symbols, RNG identity, and phase identity.
- Modify `tests/test_flat_acquisition.py` and `tests/test_synthetic_pipeline.py`: producer execution and manifest verification.

`fno-stable` adaptation:

- Modify its existing `ptycho/simulation/object_producers.py`: thread the Boolean through generated v2 builds while retaining Dead Leaves v1 and frozen object banks.
- Modify its existing `ptycho/simulation/flat_acquisition.py`: bind the field into shared-split validation and seeded builds.
- Extend the same test files in place; do not copy the narrower `refactor` registry over this branch.

## Branch discipline

- Work on `refactor` first and commit every task before switching branches.
- Preserve the unrelated existing workspace entries under `notebooks/archive/ePIE_recon_simulation`, `.claude/`, and `scripts/orchestration/`.
- `fno-stable` is already checked out at `.worktrees/fno-principled-quality`. Do not create another worktree or switch the root checkout to that branch. Run Tasks 6--8 with that existing path as the command working directory.
- The existing fno worktree has unrelated unstaged documentation and `pyproject.toml` edits. Before editing, confirm none of the feature paths below overlap; leave those edits unstaged and commit only exact task paths.
- Do not merge `refactor` wholesale into `fno-stable`. Reapply the small shared generator/config changes and manually adapt registry-specific code.

### Task 1: Port independent Dead Leaves RNG streams to `refactor`

**Files:**

- Modify: `ptycho_torch/datagen/objects.py:1005-1198`
- Create: `tests/torch/test_dead_leaves_antialiasing.py`

- [ ] **Step 1: Write the failing split-stream test**

Add a `RecordingGenerator(np.random.Generator)` test double and one test that calls `dead_leaves_ptycho(..., rng=numeric, shape_rng=geometry)` for one leaf. Assert that `choice`, radius, center, angle, and vertex draws are made only by `geometry`, while beta and delta draws are made only by `numeric`.

Use a small fixed call helper so later anti-aliasing tests reuse the exact parameters:

```python
DEAD_LEAVES_KWARGS = {
    "res": 32,
    "r_sigma_param": 3,
    "max_iters": 1,
    "r_min_frac": 0.40,
    "r_max_frac": 0.48,
    "beta_pareto_alpha": 1.5,
    "beta_scale": 0.001,
    "delta_beta_mean": 100,
    "delta_beta_std": 10,
    "thickness": 3.0,
    "min_phase": -np.pi,
    "max_phase": np.pi,
    "min_amp": 0.6,
    "max_amp": 1.1,
}
```

- [ ] **Step 2: Run the focused test and verify the missing API failure**

Run:

```bash
python -m pytest -q tests/torch/test_dead_leaves_antialiasing.py -k split_stream
```

Expected: FAIL because `dead_leaves_ptycho` and `create_dead_leaves` do not yet accept `shape_rng` on `refactor`.

- [ ] **Step 3: Port the minimal split-stream behavior**

In `create_dead_leaves`, add the keyword-only `shape_rng=None` argument and forward it. In `dead_leaves_ptycho`, validate both generators and use:

```python
numeric_source = np.random if rng is None else rng
geometry_source = numeric_source if shape_rng is None else shape_rng
```

Route only shape family, radius, center, orientation, and polygon vertices through `geometry_source`; keep beta/delta draws on `numeric_source`. Preserve the historical `random.choice` and `np.random.randint` fallbacks when both explicit generators are absent. Mirror the already-tested `fno-stable` implementation rather than inventing another RNG contract.

- [ ] **Step 4: Run the split-stream and existing reproducibility tests**

Run:

```bash
python -m pytest -q tests/torch/test_dead_leaves_antialiasing.py -k split_stream tests/torch/test_absolute_scaling_dataset_generation.py::test_full_dead_leaves_dataset_generation_is_seed_reproducible
```

Expected: PASS.

- [ ] **Step 5: Commit the prerequisite port**

```bash
git add ptycho_torch/datagen/objects.py tests/torch/test_dead_leaves_antialiasing.py
git commit -m "feat(synthetic): split dead leaves random streams"
```

### Task 2: Implement opt-in four-sample rasterization on `refactor`

**Files:**

- Modify: `ptycho_torch/datagen/objects.py:1005-1198`
- Modify: `tests/torch/test_dead_leaves_antialiasing.py`

- [ ] **Step 1: Add failing default-compatibility and strict-type tests**

Add tests that:

1. call `dead_leaves_ptycho` with the same numeric and geometry seeds once with the option omitted and once with `anti_aliasing=False`, then use `np.testing.assert_array_equal` on all four returned arrays;
2. call `create_dead_leaves` with `obj_arg={"anti_aliasing": value}` for `1`, `"true"`, and `None`, and expect `TypeError("anti_aliasing must be a bool")`; and
3. snapshot both bit-generator states before the invalid call and assert validation consumed no random values.

- [ ] **Step 2: Run the tests and verify they fail before generation**

Run:

```bash
python -m pytest -q tests/torch/test_dead_leaves_antialiasing.py -k "default or rejects"
```

Expected: FAIL because the option is not accepted or validated.

- [ ] **Step 3: Add the strict option without changing the hard path**

At the start of `create_dead_leaves`, before any draw, use:

```python
anti_aliasing = obj_arg.get("anti_aliasing", False)
if type(anti_aliasing) is not bool:
    raise TypeError("anti_aliasing must be a bool")
```

Add `anti_aliasing=False` to `dead_leaves_ptycho` and forward the resolved value. Do not touch the existing false-branch mask drawing or assignment statements.

- [ ] **Step 4: Add failing circle, polygon, determinism, and RNG tests**

Parameterize `shape_seed` as `11` (circle) and `0` (quadrilateral). For each seed, generate one hard and two anti-aliased leaves. Normalize nonzero `beta_map` values by their maximum and assert:

```python
np.testing.assert_allclose(4.0 * coverage, np.rint(4.0 * coverage), atol=2e-6)
assert np.any((coverage > 0.0) & (coverage < 1.0))
assert not np.array_equal(hard_beta, first_aa_beta)
np.testing.assert_array_equal(first_aa_beta, second_aa_beta)
```

Also compare the final `bit_generator.state` for hard and anti-aliased numeric streams and for hard and anti-aliased geometry streams.

- [ ] **Step 5: Run the new raster tests and verify hard output still appears**

Run:

```bash
python -m pytest -q tests/torch/test_dead_leaves_antialiasing.py -k "coverage or rng"
```

Expected: FAIL because `anti_aliasing=True` still uses the hard mask.

- [ ] **Step 6: Implement the exact fixed-point sample branch**

Keep the hard branch unchanged under `if not anti_aliasing`. Under the true branch:

```python
sample_offsets = (
    (1.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 2.0 / 3.0),
    (2.0 / 3.0, 1.0 / 3.0),
    (2.0 / 3.0, 2.0 / 3.0),
)
shift = 8
fixed_scale = 1 << shift
leaf_counts = np.zeros((res, res), dtype=np.uint8)
sample_mask = np.zeros((res, res), dtype=np.uint8)
```

For every `(offset_y, offset_x)`, clear `sample_mask`, encode translated circle centers with `int(np.rint((coordinate - offset) * fixed_scale))`, use `int(radius_pixels * fixed_scale)`, and call `cv2.circle(..., lineType=cv2.LINE_8, shift=shift)`. For polygons, translate the already-truncated `corners_abs` in `(x, y)` order by `[offset_x, offset_y]`, apply the same `np.rint(... * fixed_scale)` conversion to `np.int32`, and call `cv2.fillPoly(..., lineType=cv2.LINE_8, shift=shift)`.

Accumulate each byte mask into `leaf_counts`, compute `coverage = leaf_counts.astype(np.float32) / 4.0`, and apply the sequential top-leaf interpolation:

```python
beta_map = (1.0 - coverage) * beta_map + coverage * current_beta
delta_map = (1.0 - coverage) * delta_map + coverage * current_delta
```

Do not use `LINE_AA`, supersampling, a new dependency, or any additional random draw.

- [ ] **Step 7: Run all low-level tests**

Run:

```bash
python -m pytest -q tests/torch/test_dead_leaves_antialiasing.py tests/torch/test_absolute_scaling_dataset_generation.py::test_full_dead_leaves_dataset_generation_is_seed_reproducible
```

Expected: PASS.

- [ ] **Step 8: Commit the low-level feature**

```bash
git add ptycho_torch/datagen/objects.py tests/torch/test_dead_leaves_antialiasing.py
git commit -m "feat(synthetic): add dead leaves anti-aliasing"
```

### Task 3: Add canonical config and public CLI ownership on `refactor`

**Files:**

- Modify: `ptycho/config/config.py:216-349`
- Modify: `scripts/simulation/synthetic_pipeline.py:62-110,237-270`
- Modify: `tests/test_simulation_config.py:155-240,591-767`
- Modify: `tests/scripts/test_simulation_config_cli.py:26-212`
- Modify: `tests/scripts/test_synthetic_pipeline_cli.py:75-140`

- [ ] **Step 1: Write failing config tests**

Update the exact dataclass field/asdict assertions to append `dead_leaves_anti_aliasing`, with default `False`. Add a test that constructs a Dead Leaves config with `true`, asserts round-trip inclusion at `object.dead_leaves_anti_aliasing`, and asserts its SHA-256 differs from the same config with false. Parameterize non-Booleans `0`, `1`, `"true"`, and `None` and require the error path `simulation.object.dead_leaves_anti_aliasing`.

Keep `test_default_simulation_canonical_dictionary_and_digest_are_exact` unchanged: false must not appear and the digest must remain `f149d2d29e2e105643f9ee44087e3e0a562b9621be24f210301194302348772d`.

- [ ] **Step 2: Run config tests and verify the field is unknown**

Run:

```bash
python -m pytest -q tests/test_simulation_config.py tests/scripts/test_simulation_config_cli.py
```

Expected: FAIL on the new field/round-trip tests while existing default fixtures continue to describe the historical payload.

- [ ] **Step 3: Append and conditionally serialize the field**

Append this field last in `SyntheticObjectConfig`:

```python
dead_leaves_anti_aliasing: _StrictBool = False
```

Build `object_config` in `simulation_config_to_dict`, then conditionally add:

```python
if config.object.dead_leaves_anti_aliasing:
    object_config["dead_leaves_anti_aliasing"] = True
```

Return that mapping. Do not change legacy `DatagenConfig` or add a schema version.

- [ ] **Step 4: Add failing CLI precedence tests**

Test `_cli_values` and request resolution for:

- `--dead-leaves-anti-aliasing` mapping to `simulation.object.dead_leaves_anti_aliasing=True`;
- `--no-dead-leaves-anti-aliasing` mapping to false; and
- an omitted flag leaving a config-file `true` value present rather than overwriting it.

- [ ] **Step 5: Add the public BooleanOptionalAction mapping**

In the simulation argument group add:

```python
simulation.add_argument(
    "--dead-leaves-anti-aliasing",
    action=argparse.BooleanOptionalAction,
)
```

Add:

```python
"dead_leaves_anti_aliasing": (
    "simulation", "object", "dead_leaves_anti_aliasing"
),
```

to `_ARG_PATHS`. Retain `argument_default=argparse.SUPPRESS`.

- [ ] **Step 6: Run config and parser tests**

Run:

```bash
python -m pytest -q tests/test_simulation_config.py tests/scripts/test_simulation_config_cli.py tests/scripts/test_synthetic_pipeline_cli.py -k "anti_aliasing or simulation_config"
```

Expected: PASS for canonical config and parser mapping. Recipe-dependent workflow tests are added in Task 4.

- [ ] **Step 7: Commit config ownership**

```bash
git add ptycho/config/config.py scripts/simulation/synthetic_pipeline.py tests/test_simulation_config.py tests/scripts/test_simulation_config_cli.py tests/scripts/test_synthetic_pipeline_cli.py
git commit -m "feat(config): expose dead leaves anti-aliasing"
```

### Task 4: Port the deterministic producer registry and manifest route to `refactor`

**Files:**

- Create: `ptycho/simulation/object_producers.py`
- Modify: `ptycho/workflows/synthetic_config.py:124-132,459-495,826-916,1268-1338,1394-1425`
- Modify: `ptycho/simulation/flat_acquisition.py:35-170,529-680,704-729`
- Modify: `ptycho/workflows/synthetic_pipeline.py:841-898,1104-1135`
- Modify: `tests/test_synthetic_workflow_config.py`
- Modify: `tests/scripts/test_synthetic_pipeline_cli.py:540-590`
- Modify: `tests/test_flat_acquisition.py:168-208,340-430,560-614`
- Modify: `tests/test_synthetic_pipeline.py:1162-1305`

- [ ] **Step 1: Write failing registry and recipe-resolution tests**

Port the focused `fno-stable` tests for:

- `object_recipe_for_kind("lines") == "lines-object-v1"`;
- `object_recipe_for_kind("dead_leaves") == "dead-leaves-object-v2"`;
- a seeded v2 object using separate `PCG64` shape and numeric streams with spawn keys `[0]` and `[1]`;
- the fixed Dead Leaves phase identity;
- default false passing the original four-key `DEAD_LEAVES_OBJECT_ARGUMENTS` unchanged;
- true passing a copied mapping with only `"anti_aliasing": True` added; and
- true rejection for Lines, `dead-leaves-object-v1`, and `frozen-object-bank-v1`.

Add workflow tests proving an unpinned `object.kind=dead_leaves` derives v2, an explicitly mismatched recipe fails, and `dataclasses.replace` cannot forge an invalid enabled pair past sealed revalidation.

- [ ] **Step 2: Run the registry/resolver tests and verify the module is absent**

Run:

```bash
python -m pytest -q tests/test_synthetic_workflow_config.py tests/test_flat_acquisition.py -k "dead_leaves or object_recipe"
```

Expected: FAIL because `ptycho.simulation.object_producers` is absent and the resolver still retains the Lines recipe.

- [ ] **Step 3: Create the narrow registry**

Port only the following v2 contracts from `fno-stable:ptycho/simulation/object_producers.py`:

- Lines and Dead Leaves recipe/symbol constants;
- `LinesObject`, `DeadLeavesObject`, `ObjectRandomStreams`, and `ObjectProducer` records;
- array validation and `fixed_dead_leaves_phase`;
- Lines and Dead Leaves v2 builders;
- `OBJECT_PRODUCERS`, `DEFAULT_OBJECT_RECIPES`, `registered_object_kinds`, `object_recipe_for_kind`, and `validate_object_recipe`;
- `rng_identity_for_seed`, `phase_identity_for_recipe`, `build_object`, `build_object_from_seed`, and the two convenience wrappers.

Exclude Dead Leaves v1 construction, frozen object banks, multiple-object banks, and source-backed machinery. Add one shared validator:

```python
def validate_dead_leaves_anti_aliasing(kind, recipe, enabled):
    if type(enabled) is not bool:
        raise TypeError("dead_leaves_anti_aliasing must be a bool")
    if enabled and (kind, recipe) != (
        "dead_leaves", DEAD_LEAVES_OBJECT_RECIPE
    ):
        raise ValueError(
            "dead_leaves_anti_aliasing=True requires "
            "dead_leaves/dead-leaves-object-v2"
        )
```

Call it at the builder boundary. Preserve the false build path exactly; for true, copy the fixed argument mapping and add the low-level option without mutating the module constant.

- [ ] **Step 4: Derive and revalidate recipes in the workflow resolver**

Add `dead_leaves_anti_aliasing: False` to the profile object mapping. Detect whether `simulation.object_recipe` was explicitly supplied before merging patches, pass that fact into `_resolve_simulation`, and otherwise derive the recipe with `object_recipe_for_kind(train.object.kind)`. Validate both recipe and anti-aliasing after resolution and again in `_validate_resolved_workflow`.

- [ ] **Step 5: Run registry and workflow tests**

Run:

```bash
python -m pytest -q tests/test_synthetic_workflow_config.py tests/scripts/test_synthetic_pipeline_cli.py -k "dead_leaves or object_recipe or anti_aliasing"
```

Expected: PASS, including fake-executor dispatch for the Dead Leaves v2 CLI route.

- [ ] **Step 6: Write failing flat-acquisition and manifest tests**

Add tests that resolve a small Dead Leaves workflow, stub only probe/diffraction execution, and assert:

- generation calls `build_object_from_seed` with kind, v2 recipe, the object seed, and the resolved Boolean;
- Lines output remains unchanged for the same seed;
- the manifest object record contains recipe, producer symbols, array hash, seed, RNG identity, and phase identity;
- each split recipe identity contains that complete object identity; and
- changing or deleting producer symbols, RNG identity, phase identity, or the enabled simulation field is rejected by `_load_matching_dataset_manifest` / split verification.

- [ ] **Step 7: Replace the Lines-only flat path with the registry**

In `flat_acquisition.py`:

1. import `ptycho.simulation.object_producers` and re-export its existing Lines wrapper for compatibility;
2. add `object.dead_leaves_anti_aliasing` to `_SHARED_SPLIT_RECIPE_FIELDS`;
3. replace the hardcoded recipe/kind preflight with registry and anti-aliasing validation;
4. build the shared object through `build_object_from_seed(..., dead_leaves_anti_aliasing=train_simulation.object.dead_leaves_anti_aliasing)`; and
5. generalize `lines_object` locals to `synthetic_object` and record `rng_identity` plus `phase_identity` in `object_identity`.

In `ptycho/workflows/synthetic_pipeline.py`, derive the expected producer from `validate_object_recipe`, compare its symbols dynamically, compare `rng_identity_for_seed` and `phase_identity_for_recipe`, and include both fields when reconstructing split recipe identity.

- [ ] **Step 8: Run generation and manifest tests**

Run:

```bash
python -m pytest -q tests/test_flat_acquisition.py tests/test_synthetic_pipeline.py -k "object or manifest or dead_leaves"
```

Expected: PASS.

- [ ] **Step 9: Commit the producer port**

```bash
git add ptycho/simulation/object_producers.py ptycho/workflows/synthetic_config.py ptycho/simulation/flat_acquisition.py ptycho/workflows/synthetic_pipeline.py tests/test_synthetic_workflow_config.py tests/scripts/test_synthetic_pipeline_cli.py tests/test_flat_acquisition.py tests/test_synthetic_pipeline.py
git commit -m "feat(synthetic): port dead leaves producer to refactor"
```

### Task 5: Expose the specialized study option, document it, and pin `refactor`

**Files:**

- Modify: `scripts/studies/make_synthetic_truth_datasets.py:26-49,90-97,362-380,431-463`
- Modify: `tests/torch/test_absolute_scaling_dataset_generation.py:161-200`
- Modify: `tests/torch/test_dead_leaves_antialiasing.py`
- Modify: `docs/CONFIGURATION.md:700-767`
- Modify: `scripts/simulation/README.md:203-217,440-459`

- [ ] **Step 1: Write the failing specialized CLI/provenance test**

Parameterize `[]`, `["--dead-leaves-anti-aliasing"]`, and `["--no-dead-leaves-anti-aliasing"]`. Monkeypatch the study's `SPECS`, probe loader, object builder, and per-split builder so `main(argv)` remains small. Assert the mapping passed into `frozen_raw_object` and written to `provenance_deadleaves.json` is the historical four-key mapping for false and that same copied mapping plus `"anti_aliasing": True` for true. Assert `DEAD_LEAVES_ARG` itself is unchanged.

- [ ] **Step 2: Run the test and verify `main` rejects argv**

Run:

```bash
python -m pytest -q tests/torch/test_absolute_scaling_dataset_generation.py -k anti_aliasing
```

Expected: FAIL because the script has no parser and `main()` accepts no arguments.

- [ ] **Step 3: Add the specialized parser and copied mapping**

Add `argparse`, `build_parser()`, the same `BooleanOptionalAction`, and `main(argv=None)`. Resolve with:

```python
dead_leaves_arg = dict(DEAD_LEAVES_ARG)
if args.dead_leaves_anti_aliasing:
    dead_leaves_arg["anti_aliasing"] = True
```

Pass that mapping into `frozen_raw_object` and record it as provenance. Do not mutate the global mapping and do not add false to historical provenance.

- [ ] **Step 4: Pin one anti-aliased v2 object hash**

After the semantic raster tests pass, compute the full registered fixture once:

```bash
python - <<'PY'
from ptycho.simulation.identity import array_sha256
from ptycho.simulation.object_producers import build_object_from_seed

obj = build_object_from_seed(
    "dead_leaves",
    "dead-leaves-object-v2",
    123,
    dead_leaves_anti_aliasing=True,
)
print(array_sha256(obj.array))
PY
```

Add the printed 64-character digest as a literal constant in `tests/torch/test_dead_leaves_antialiasing.py` and assert the same build equals it. This golden is protected by the independent circle/polygon coverage tests, so it is a cross-branch drift detector rather than the only correctness oracle.

- [ ] **Step 5: Document the public setting**

In `docs/CONFIGURATION.md`, add the nested TOML field, default false/default-elision rule, v2-only validity, and dataset-identity effect. In `scripts/simulation/README.md`, replace the Lines-only producer statement with Lines v1 / Dead Leaves v2 selection and show `--dead-leaves-anti-aliasing` plus its negative form. Keep `simulate_and_save.py` documented as consuming an already-built object.

- [ ] **Step 6: Run the complete focused suite on `refactor`**

Run:

```bash
python -m pytest -q \
  tests/torch/test_dead_leaves_antialiasing.py \
  tests/test_simulation_config.py \
  tests/scripts/test_simulation_config_cli.py \
  tests/test_synthetic_workflow_config.py \
  tests/scripts/test_synthetic_pipeline_cli.py \
  tests/test_flat_acquisition.py \
  tests/test_synthetic_pipeline.py \
  tests/torch/test_absolute_scaling_dataset_generation.py
python -m compileall -q \
  ptycho/config/config.py \
  ptycho/simulation/object_producers.py \
  ptycho/simulation/flat_acquisition.py \
  ptycho/workflows/synthetic_config.py \
  ptycho/workflows/synthetic_pipeline.py \
  ptycho_torch/datagen/objects.py \
  scripts/simulation/synthetic_pipeline.py \
  scripts/studies/make_synthetic_truth_datasets.py
```

Expected: all tests PASS; `compileall` exits 0.

- [ ] **Step 7: Commit study/docs/golden coverage**

```bash
git add scripts/studies/make_synthetic_truth_datasets.py tests/torch/test_absolute_scaling_dataset_generation.py tests/torch/test_dead_leaves_antialiasing.py docs/CONFIGURATION.md scripts/simulation/README.md
git commit -m "docs(synthetic): document dead leaves anti-aliasing"
```

### Task 6: Apply the shared surface in the existing `fno-stable` worktree

**Files:**

- Modify: `ptycho_torch/datagen/objects.py:1005-1198`
- Create: `tests/torch/test_dead_leaves_antialiasing.py`
- Modify: `ptycho/config/config.py:211-382`
- Modify: `scripts/simulation/synthetic_pipeline.py:64-90,303-340`
- Modify: `tests/test_simulation_config.py:593-779`
- Modify: `tests/scripts/test_simulation_config_cli.py`
- Modify: `tests/scripts/test_synthetic_pipeline_cli.py`

- [ ] **Step 1: Confirm `refactor` is committed and audit the existing fno worktree**

Run:

```bash
git status --short
git -C .worktrees/fno-principled-quality status --short --branch
git -C .worktrees/fno-principled-quality status --short -- \
  ptycho_torch/datagen/objects.py \
  ptycho/config/config.py \
  ptycho/workflows/synthetic_config.py \
  ptycho/simulation/object_producers.py \
  ptycho/simulation/flat_acquisition.py \
  scripts/simulation/synthetic_pipeline.py \
  scripts/studies/make_synthetic_truth_datasets.py \
  tests/torch/test_dead_leaves_antialiasing.py \
  tests/test_simulation_config.py \
  tests/test_synthetic_workflow_config.py \
  tests/scripts/test_simulation_config_cli.py \
  tests/scripts/test_synthetic_pipeline_cli.py \
  tests/test_flat_acquisition.py \
  tests/torch/test_absolute_scaling_dataset_generation.py \
  docs/CONFIGURATION.md \
  scripts/simulation/README.md
```

Expected: `refactor` has no task-related tracked changes, the existing worktree reports branch `fno-stable`, and the path-limited fno diff is empty. Its pre-existing unstaged changes outside this list remain untouched. Run every remaining Task 6 and Task 7 command with `/home/ollie/Documents/PtychoPINN/.worktrees/fno-principled-quality` as the working directory.

- [ ] **Step 2: Copy the branch-identical tests and run them red**

Bring `tests/torch/test_dead_leaves_antialiasing.py` from `refactor` onto `fno-stable`, then run:

```bash
python -m pytest -q tests/torch/test_dead_leaves_antialiasing.py \
  -k "not seeded_v2_antialiased_object_hash"
```

Expected: FAIL because the fno generator lacks the option, while its existing split-RNG behavior remains available.

- [ ] **Step 3: Apply only the anti-aliasing delta to the fno generator**

Add the same strict option and raster branch from Task 2, preserving fno's existing `shape_rng` implementation. Do not replace the entire file with the `refactor` version. Rerun the Step 2 command and require PASS for every low-level test; the producer-hash node remains intentionally deselected until Task 7 threads the option through fno's registry.

- [ ] **Step 4: Add failing fno config/CLI tests and implement the field**

Append `dead_leaves_anti_aliasing` after fno's existing `source_path` field. Conditionally add only true to `object_config` in `simulation_config_to_dict`; `simulation_config_digest_input` then needs no special case. Add the same public CLI action/mapping and update fno's exact field/asdict tests while preserving its current default dictionary and historical digest.

Run:

```bash
python -m pytest -q tests/test_simulation_config.py tests/scripts/test_simulation_config_cli.py tests/scripts/test_synthetic_pipeline_cli.py -k "anti_aliasing or simulation_config"
```

Expected: PASS.

- [ ] **Step 5: Commit the fno shared surface**

```bash
git add ptycho_torch/datagen/objects.py tests/torch/test_dead_leaves_antialiasing.py ptycho/config/config.py scripts/simulation/synthetic_pipeline.py tests/test_simulation_config.py tests/scripts/test_simulation_config_cli.py tests/scripts/test_synthetic_pipeline_cli.py
git commit -m "feat(synthetic): add dead leaves anti-aliasing option"
```

### Task 7: Route anti-aliasing through the existing `fno-stable` producer

**Files:**

- Modify: `ptycho/simulation/object_producers.py:154-208,287-312,491-537`
- Modify: `ptycho/workflows/synthetic_config.py:491-519,845-990,1561-1590,1696-1731,1852-1927`
- Modify: `ptycho/simulation/flat_acquisition.py:71-86,890-970,1605-1665`
- Modify: `scripts/studies/make_synthetic_truth_datasets.py:26-49,90-97,362-380,431-463`
- Modify: `tests/test_synthetic_workflow_config.py`
- Modify: `tests/test_flat_acquisition.py:562-778,920-983`
- Modify: `tests/torch/test_absolute_scaling_dataset_generation.py`
- Modify: `docs/CONFIGURATION.md`
- Modify: `scripts/simulation/README.md`

- [ ] **Step 1: Write failing producer and invalid-pair tests**

Extend fno's existing v2 builder tests to assert false retains the original four-key mapping and true adds only `anti_aliasing=True`. Add workflow/resolved-record tests rejecting true with Lines, Dead Leaves v1, and `frozen-object-bank-v1`. Keep all existing v1 golden and frozen-bank tests unchanged.

- [ ] **Step 2: Run the focused fno tests red**

Run:

```bash
python -m pytest -q tests/test_synthetic_workflow_config.py tests/test_flat_acquisition.py -k "anti_aliasing or dead_leaves"
```

Expected: FAIL because the field is not routed into the existing builder.

- [ ] **Step 3: Adapt the existing registry without narrowing it**

Add the same shared `validate_dead_leaves_anti_aliasing` guard. Thread a keyword-only Boolean through `build_object`, `build_object_from_seed`, and `build_dead_leaves_object`. For false, preserve the current producer callable path and exact four-key argument mapping. For true, allow only v2 and pass a copied argument mapping with the low-level key. Do not modify v1 RNG/phase behavior, registry membership, or frozen-bank loaders.

- [ ] **Step 4: Validate at fno workflow boundaries**

Add false to the profile object mapping. After recipe selection in `_resolve_simulation` and after recipe validation in `_validate_resolved_workflow`, call the anti-aliasing pair validator. Because `SimulationConfig` serialization already omits false, workflow digest normalization needs no new special case; true remains identity-bearing.

- [ ] **Step 5: Bind the field into fno flat acquisition**

Add `object.dead_leaves_anti_aliasing` to `_SHARED_SPLIT_RECIPE_FIELDS`. Pass the resolved Boolean to every generated-object `build_object_from_seed` call for shared and split object banks. Leave source-backed frozen-bank calls unchanged; preflight rejects true before generation.

- [ ] **Step 6: Apply the specialized study CLI and docs**

Make the same `main(argv=None)`/copied-mapping/provenance changes from Task 5. Update the branch's configuration and simulation guides without overwriting fno-specific object-bank documentation.

- [ ] **Step 7: Run fno producer, workflow, study, and golden tests**

Run:

```bash
python -m pytest -q \
  tests/torch/test_dead_leaves_antialiasing.py \
  tests/test_synthetic_workflow_config.py \
  tests/scripts/test_synthetic_pipeline_cli.py \
  tests/test_flat_acquisition.py \
  tests/torch/test_absolute_scaling_dataset_generation.py
```

Expected: PASS, including the literal anti-aliased v2 hash copied from `refactor` and all pre-existing v1/frozen tests.

- [ ] **Step 8: Commit the fno producer adaptation**

```bash
git add ptycho/simulation/object_producers.py ptycho/workflows/synthetic_config.py ptycho/simulation/flat_acquisition.py scripts/studies/make_synthetic_truth_datasets.py tests/test_synthetic_workflow_config.py tests/test_flat_acquisition.py tests/torch/test_absolute_scaling_dataset_generation.py docs/CONFIGURATION.md scripts/simulation/README.md
git commit -m "feat(synthetic): route anti-aliasing through object producers"
```

### Task 8: Verify both branch heads and cross-branch identity

**Files:**

- Verify only; no planned production changes.

- [ ] **Step 1: Run the complete focused suite on `fno-stable`**

Run the same eight-file pytest and `compileall` commands from Task 5. Expected: PASS / exit 0.

- [ ] **Step 2: Record fno identity outputs**

Run:

```bash
python - <<'PY'
from ptycho.config import simulation_config_from_mapping, simulation_config_sha256
from ptycho.simulation.identity import array_sha256
from ptycho.simulation.object_producers import build_object_from_seed

cfg = simulation_config_from_mapping({
    "object": {"kind": "dead_leaves", "dead_leaves_anti_aliasing": True}
})
obj = build_object_from_seed(
    "dead_leaves", "dead-leaves-object-v2", 123,
    dead_leaves_anti_aliasing=True,
)
print(simulation_config_sha256(cfg))
print(array_sha256(obj.array))
PY
```

Save the two printed values in the execution notes.

- [ ] **Step 3: Rerun the identity command in the root `refactor` checkout and compare**

Run:

```bash
git -C /home/ollie/Documents/PtychoPINN status --short --branch
python -m pytest -q \
  /home/ollie/Documents/PtychoPINN/tests/torch/test_dead_leaves_antialiasing.py::test_seeded_v2_antialiased_object_hash
```

Run this step with `/home/ollie/Documents/PtychoPINN` as the working directory, then rerun the Task 8 Step 2 Python command there. Expected: both printed hashes exactly match `fno-stable`.

- [ ] **Step 4: Run the complete focused suite on `refactor` once more**

Run the same eight-file pytest and `compileall` commands from Task 5. Expected: PASS / exit 0.

- [ ] **Step 5: Inspect final branch state**

Run:

```bash
git -C /home/ollie/Documents/PtychoPINN status --short --branch
git -C /home/ollie/Documents/PtychoPINN log --oneline -8
git -C /home/ollie/Documents/PtychoPINN/.worktrees/fno-principled-quality status --short --branch
git -C /home/ollie/Documents/PtychoPINN/.worktrees/fno-principled-quality log --oneline -8
```

Expected: both branch heads contain their feature commits, `refactor` has no task-related uncommitted tracked changes, and every unrelated pre-existing change in both worktrees remains untouched and unstaged.
