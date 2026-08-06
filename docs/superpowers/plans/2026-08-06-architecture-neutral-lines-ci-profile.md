# Architecture-Neutral `lines-ci` Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace architecture-named synthetic CI profiles with one active, architecture-neutral `lines-ci` / `lines-ci-v1` profile on `refactor`, `fno-stable`, and `refactor-internal`.

**Architecture:** The synthetic profile resolver owns the count-intensity, rectangular-forward, Poisson contract but treats architecture as an unlocked model choice with a common CNN default. A small Torch capability module owns the branch-specific set of architectures validated for complete grouped (`gridsize > 1`) workflows, and both the synthetic resolver and grid-lines runner consume it. Effective output representation is selected by architecture (`cnn_output_mode` for CNN, `generator_output_mode` otherwise). Retired profile names remain historical provenance only: they are rejected for new authoring, old roots are not relabeled, and bundle inference remains independent of profile selection.

**Tech Stack:** Python 3.11, stdlib dataclasses, Pydantic `TypeAdapter`, PyTorch, Lightning, NumPy/NPZ, JSON/YAML/TOML, pytest, Git. Work serially in the existing checkout; do not create worktrees.

---

**Design authority:** [`docs/superpowers/specs/2026-08-06-architecture-neutral-lines-ci-profile-design.md`](../specs/2026-08-06-architecture-neutral-lines-ci-profile-design.md)

## Fixed decisions

| Concern | Required outcome |
|---|---|
| Active CI profile | `lines-ci` only |
| Recipe identity | `lines-ci-v1` |
| Retired names | Rejected with targeted migration text; no aliases |
| Default architecture | `cnn` on all three branches |
| Architecture override | Allowed when registered and compatible with the requested grid |
| CI locks | `ci_intensity_v2`, `count_intensity`, `rectangular_scaled`, `Poisson`/`poisson`, `nll=true` |
| Overrideable defaults | Architecture, `rect_s1s2_init`, `rect_s1s2_trainable`, gradient clipping |
| Effective representation | CNN reads `cnn_output_mode`; all other architectures read `generator_output_mode`; selected value must be `real_imag` |
| `gridsize > 1` on `refactor` | `cnn` |
| `gridsize > 1` on `fno-stable` / `refactor-internal` | `cnn`, `hybrid_resnet` |
| Historical workflow roots | Never rewritten or resumed as `lines-ci` |
| Historical bundles | Continue through the ordinary persisted-bundle inference path |
| Physics/sampling math | Unchanged |

The amplitude profile and its sealed identity are outside this migration. On
`fno-stable` and `refactor-internal`, the fixture-backed
`tests/torch/test_synthetic_hybrid_resnet_gs2_integration.py` command uses the
amplitude profile `hybrid-resnet-lines`, not the retired CI profile. Do not
rename that command or repin
`tests/fixtures/generic_runner_hybrid_resnet_gs2_5ep_metrics.json` as part of
this work. Those Hybrid/ResNet files are intentionally absent from `refactor`.

## Execution rules and branch baseline

The design was drafted from these local tips:

```text
refactor           bcc39a39c03fc44e742b3b3bba7271c90b01bad9
fno-stable         e3e94a7a4502b01f0701ed5eb9bd7cb9806360ca
refactor-internal  b6a3a494eac8437ca4dcb2cfb55a9c1454c8ed0d
```

The design commit itself advances `refactor`; record fresh tips before
implementation. At execution time:

- [ ] Run `git fetch --all --prune`, inspect `git status --short`, and compare
      local/remote ancestry before editing any branch. Fast-forward from the
      branch's established remote when possible; do not manufacture a merge
      merely to hide divergence.
- [ ] Preserve `.claude/`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/`,
      `scripts/orchestration/`, and the dirty notebook submodule. Stage only
      named files.
- [ ] Implement and settle `refactor` first. Port the settled behavior—not raw
      whole-file replacements—to `fno-stable`, then `refactor-internal`.
- [ ] Keep the standard branch exclusions: do not restore Hybrid/ResNet code or
      `docs/index.md` on `refactor`; documentation exclusions do not apply to
      `refactor-internal`.
- [ ] Invoke Python through PATH as `python`. Run long integration and full
      suites in tmux after activating `ptycho311`; track and wait for the exact
      launched PID. Use a distinct output root for every run.
- [ ] On every branch, run integration selectors before the comprehensive
      suite. If integration fails, diagnose and repair it before launching the
      comprehensive suite.
- [ ] Do not push any branch unless separately requested.

### Task 1: Establish the neutral profile contract with failing tests on `refactor`

**Files:**

- Modify: `tests/test_synthetic_workflow_config.py`
- Modify: `tests/scripts/test_synthetic_pipeline_cli.py`
- Modify: `tests/test_synthetic_pipeline.py`
- Modify: `tests/test_flat_acquisition.py`
- Create: `tests/torch/test_architecture_capabilities.py`
- Modify: `tests/torch/test_artifact_schema.py`
- Modify: `tests/torch/test_fno_lightning_integration.py`
- Modify: `tests/torch/test_grid_lines_torch_runner_ci_inference.py`

- [ ] **Step 1: Pin the one-profile identity and migration errors**

Replace architecture-named assertions with exact expectations for
`lines-ci` / `lines-ci-v1`. Add a test that filters `_PROFILES` for CI recipes
and finds exactly one entry. Parameterize both retired names:

```python
@pytest.mark.parametrize(
    ("retired", "architecture"),
    [
        ("cnn-lines-ci", "cnn"),
        ("hybrid-resnet-lines-ci", "hybrid_resnet"),
    ],
)
def test_retired_ci_profile_names_are_not_authoring_aliases(
    retired,
    architecture,
):
    with pytest.raises(
        ValueError,
        match=rf"{retired}.*retired.*lines-ci.*{architecture}",
    ):
        resolve_synthetic_workflow(profile=retired)
```

Require the omitted architecture to resolve to `cnn`, while explicit matching
semantic locks are accepted.

- [ ] **Step 2: Pin architecture-aware representation behavior**

Add table-driven cases proving:

- CNN + `cnn_output_mode="real_imag"` succeeds even if the irrelevant
  `generator_output_mode` is explicitly `amp_phase`;
- FFNO + `generator_output_mode="real_imag"` succeeds at `gridsize=1` even if
  the irrelevant `cnn_output_mode` is `amp_phase`;
- CNN + `cnn_output_mode="amp_phase"` fails and names
  `model.cnn_output_mode`;
- FFNO + `generator_output_mode="amp_phase"` fails and names
  `model.generator_output_mode`;
- architecture overrides are not reported as profile-lock contradictions;
- `neuralop_uno` retains its existing `N=128`, `gridsize=1`, `real_imag`
  restrictions.

Use `file_values` and `cli_values` variants so both precedence paths are
covered.

- [ ] **Step 3: Pin branch-owned grid capability**

Create tests expecting a named immutable capability set and one validator:

```python
assert GROUPED_WORKFLOW_ARCHITECTURES == frozenset({"cnn"})
validate_grouped_workflow_architecture(
    architecture="ffno",
    gridsize=1,
    field="model.architecture",
)
with pytest.raises(ValueError, match="model.architecture.*ffno.*gridsize=2"):
    validate_grouped_workflow_architecture(
        architecture="ffno",
        gridsize=2,
        field="model.architecture",
    )
```

Exercise both public consumers: the synthetic resolver and
`setup_torch_configs()` must accept the same `gridsize=1` case and reject the
same unsupported grouped case. Do not assert that registry membership implies
grouped support.

- [ ] **Step 4: Pin file-format and CLI convergence**

Build equivalent JSON, YAML, TOML, and explicit CLI inputs selecting
`lines-ci` with FFNO at `gridsize=1`. Assert equal canonical resolved mappings
and equal workflow digests. Keep the config-root `profile` as the selector;
do not add a duplicate nested profile field.

- [ ] **Step 5: Pin identity boundaries**

Add or update tests that require:

- fresh records persist `profile="lines-ci"` and
  `recipe_version="lines-ci-v1"`;
- CNN and FFNO resolutions under `lines-ci` have different workflow digests;
- the sealed `synthetic-lines-v1` payload bytes and digests remain unchanged;
- a stage root whose persisted workflow says `cnn-lines-ci-v1` or
  `hybrid-resnet-lines-ci-v1` fails on profile/recipe identity mismatch before
  any stage is reused; current documentation directs the user to a new output
  root;
- flat count-acquisition manifests carry the new literal profile/recipe and
  retain all count/probe/determinism assertions.

Do not normalize historical names before hashing or reuse comparison.

- [ ] **Step 6: Characterize strict historical CI bundle loading**

Add
`tests/torch/test_artifact_schema.py::test_retired_synthetic_profile_record_does_not_gate_strict_ci_bundle_load`.
Build and save the real transitional CI bundle used by the neighboring strict
load test, write a parent `resolved_workflow.json` carrying each retired
profile/recipe pair, and call `load_inference_bundle_torch(bundle_dir)` without
calling the synthetic resolver. Assert state-dict equality and the decoded CI
scale/domain identity. This characterization must pass before and after the
profile migration and proves that a completed bundle is not reselected by its
synthetic authoring profile.

- [ ] **Step 7: Pin an actual registered non-CNN model path**

Extend `test_fno_lightning_integration.py` with a small CPU forward/backward
test that starts from a resolved `lines-ci` FFNO configuration, constructs the
registered generator through the production factory, consumes `C=1`
count-intensity-shaped input, produces the expected two-channel complex
adapter output, and has finite gradients. This is a mechanics gate, not a
quality threshold.

- [ ] **Step 8: Run the focused tests red**

```bash
python -m pytest \
  tests/test_synthetic_workflow_config.py \
  tests/scripts/test_synthetic_pipeline_cli.py \
  tests/test_synthetic_pipeline.py \
  tests/test_flat_acquisition.py \
  tests/torch/test_architecture_capabilities.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_fno_lightning_integration.py \
  tests/torch/test_grid_lines_torch_runner_ci_inference.py -q
```

Expected failures must be limited to the missing neutral profile/capability
behavior; the historical-bundle characterization from Step 6 must already be
green. Investigate unrelated failures before implementation.

### Task 2: Implement `lines-ci` and shared grouped capability on `refactor`

**Files:**

- Modify: `ptycho/workflows/synthetic_config.py`
- Create: `ptycho_torch/architecture_capabilities.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: files listed in Task 1 only as needed to finish the tests

- [ ] **Step 1: Add the branch-owned capability module**

Define one immutable set and one pure validator in
`ptycho_torch/architecture_capabilities.py`:

```python
GROUPED_WORKFLOW_ARCHITECTURES = frozenset({"cnn"})


def validate_grouped_workflow_architecture(
    *,
    architecture: str,
    gridsize: int,
    field: str = "architecture",
) -> None:
    ...
```

The validator is a no-op for `gridsize == 1`; for larger grids it checks the
set and reports the supplied field, architecture, grid size, and supported
values. It does not duplicate the model registry or claim every architecture
in that registry is grouped-capable.

- [ ] **Step 2: Replace the profile identity and patch**

In `synthetic_config.py`:

```python
_CI_PROFILE_NAME = "lines-ci"
_CI_RECIPE_VERSION = "lines-ci-v1"
```

Keep `model.architecture="cnn"` as a profile patch default. Also default both
representation knobs to `real_imag`, so the default remains coherent when the
caller changes architecture. Keep `rect_s1s2_init="dose_closure"` and gradient
clipping in the patch as overrideable defaults.

Limit `_PROFILE_LOCKS` to the six semantic fields in the design. Remove
architecture and representation knobs from the lock map.

- [ ] **Step 3: Add targeted retirement handling without aliases**

Keep a private message map for `cnn-lines-ci` and
`hybrid-resnet-lines-ci`, checked only on the unknown-profile error path. Do
not insert either string into `_PROFILES`, do not resolve it, and do not alter
serialized historical records.

- [ ] **Step 4: Resolve effective output by selected architecture**

Add a small private helper returning `(field_name, value)`:

```python
if model.architecture == "cnn":
    return "model.cnn_output_mode", model.cnn_output_mode
return "model.generator_output_mode", model.generator_output_mode
```

Use it in rectangular/CI coherence validation. Require `real_imag` and name
the selected field on failure. Do not require or lock the irrelevant knob.

- [ ] **Step 5: Route both grouped consumers through one validator**

Replace the inline CNN-only checks in `_resolve_model()` and
`setup_torch_configs()` with
`validate_grouped_workflow_architecture()`. Preserve existing
`neuralop_uno` constraints and branch-specific architecture typing.

- [ ] **Step 6: Run focused tests green and commit the code boundary**

```bash
python -m pytest \
  tests/test_synthetic_workflow_config.py \
  tests/scripts/test_synthetic_pipeline_cli.py \
  tests/test_synthetic_pipeline.py \
  tests/test_flat_acquisition.py \
  tests/torch/test_architecture_capabilities.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_fno_lightning_integration.py \
  tests/torch/test_grid_lines_torch_runner_ci_inference.py -q
git diff --check
```

Stage only Task 1–2 code/tests and commit:

```bash
git commit -m "refactor(ci): make synthetic lines profile architecture-neutral"
```

### Task 3: Update current `refactor` documentation without rewriting history

**Files:**

- Modify: `README.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/DATA_NORMALIZATION_GUIDE.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `scripts/simulation/README.md`
- Modify: `docs/plans/2026-08-05-refactor-ci-synthetic-port.md` (status note only)

- [ ] **Step 1: Replace active commands and tables**

Document `lines-ci` / `lines-ci-v1`, the common CNN default, explicit
architecture override, architecture-aware representation knob, semantic
locks, overrideable `ones` and gradient settings, and the branch's CNN-only
grouped capability. Show at least:

```bash
ptycho_synthetic --profile lines-ci
ptycho_synthetic --profile lines-ci --architecture ffno --gridsize 1
```

Keep the distinction between training-only `profile="ci"` and synthetic
`profile="lines-ci"` explicit.

- [ ] **Step 2: Preserve historical provenance**

Do not rewrite old commands, paths, measured results, or SHAs in the completed
2026-08-05 port plan. Add only a short leading note that its
`cnn-lines-ci-v1` name is historical and points to the new design for current
authoring.

- [ ] **Step 3: Enforce the refactor exclusions and documentation gate**

```bash
test ! -e docs/index.md
git grep -n -E "cnn-lines-ci|hybrid-resnet-lines-ci" -- \
  README.md docs scripts/simulation/README.md
```

Only the design, the migration note, and genuinely historical plan/evidence
contexts may match. Confirm no Hybrid/ResNet implementation or guide was
restored.

- [ ] **Step 4: Commit the documentation boundary**

```bash
git diff --check
git commit -m "docs(ci): document the architecture-neutral lines profile"
```

### Task 4: Verify the settled `refactor` implementation

- [ ] **Step 1: Run the focused contract surface**

```bash
python -m pytest \
  tests/test_synthetic_workflow_config.py \
  tests/scripts/test_synthetic_pipeline_cli.py \
  tests/test_synthetic_pipeline.py \
  tests/test_flat_acquisition.py \
  tests/torch/test_architecture_capabilities.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_fno_lightning_integration.py \
  tests/torch/test_grid_lines_torch_runner_ci_inference.py \
  tests/torch/test_absolute_scaling_contract.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_public_barycentric_workflow.py \
  tests/torch/test_workflows_components.py -q
```

- [ ] **Step 2: Run one fresh public non-CNN smoke**

Run this reduced `lines-ci --architecture ffno --gridsize 1` public workflow.
The 256 train groups at `C=1` satisfy the fixed 256-slot dose-closure sample;
the model uses one small FFNO block and all four production stages:

```bash
mkdir -p .artifacts/integration
lines_ci_run_root=$(mktemp -d \
  "$PWD/.artifacts/integration/lines-ci-ffno-gs1-refactor-XXXXXX")
python -m scripts.simulation.synthetic_pipeline \
  --profile lines-ci \
  --architecture ffno \
  --N 64 \
  --gridsize 1 \
  --train-patterns 256 \
  --test-patterns 64 \
  --train-raw-selection 256 \
  --training-groups 256 \
  --validation-groups 64 \
  --neighbor-count 1 \
  --neighbor-pool-size 1 \
  --groups-per-center 1 \
  --epochs 1 \
  --batch-size 16 \
  --inference-batch-size 16 \
  --fno-modes 4 \
  --fno-width 8 \
  --fno-blocks 1 \
  --fno-cnn-blocks 1 \
  --accelerator cpu \
  --devices 1 \
  --precision 32-true \
  --workers 0 \
  --logger csv \
  --deterministic \
  --no-progress-bar \
  --stages simulate,train,reconstruct,evaluate \
  --output-root "$lines_ci_run_root"
test -s "$lines_ci_run_root/resolved_workflow.json"
test -s "$lines_ci_run_root/stage_manifest.json"
```

Require exit code zero, fresh dataset/stage manifests, strict bundle reload,
mmap reconstruction, finite raw metrics, and persisted FFNO identity. This is
a structural integration smoke; do not introduce quality thresholds or a
tracked fixture. Record the concrete `lines_ci_run_root` alongside the result.

- [ ] **Step 3: Run integration tests before the full suite**

In tmux with `ptycho311` active, launch `python -m pytest -m integration -q`,
capture its exact PID and exit status, and wait for completion. Investigate any
failure before continuing. Do not launch another run against the same output
root.

- [ ] **Step 4: Run the comprehensive suite**

Only after Step 3 passes, run `python -m pytest -q` in tmux with the same exact
PID discipline. Record the branch SHA, commands, exit codes, skip counts, and
log locations. Classify any supplemental failure against this plan's contract
before changing code.

### Task 5: Port the settled contract to `fno-stable`

**Files:**

- Modify: `ptycho/workflows/synthetic_config.py`
- Create or modify: `ptycho_torch/architecture_capabilities.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/test_synthetic_workflow_config.py`
- Modify: `tests/scripts/test_synthetic_pipeline_cli.py`
- Modify: `tests/test_synthetic_pipeline.py`
- Modify: `tests/test_flat_acquisition.py`
- Create or modify: `tests/torch/test_architecture_capabilities.py`
- Modify: `tests/torch/test_artifact_schema.py`
- Modify as needed: `tests/torch/test_fno_lightning_integration.py`
- Modify as needed: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `README.md`
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/DATA_NORMALIZATION_GUIDE.md`
- Modify: `docs/findings.md`
- Modify: `docs/specs/spec-ptycho-core.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/TESTING_GUIDE.md`
- Modify: `docs/development/TEST_SUITE_INDEX.md`
- Modify: `docs/index.md`
- Modify: `scripts/simulation/README.md`
- Add: the approved design and this plan if absent on the branch

- [ ] **Step 1: Write/adapt red tests before porting production code**

Port the Task 1 assertions, changing only the grouped capability expectation:

```python
assert GROUPED_WORKFLOW_ARCHITECTURES == frozenset(
    {"cnn", "hybrid_resnet"}
)
```

Add explicit cases for default CNN and
`--architecture hybrid_resnet --gridsize 2`. Require FFNO to remain accepted
at `gridsize=1` but rejected at `gridsize=2`. Require both retired CI names to
fail. Run the focused selector set and observe only the expected red failures.

- [ ] **Step 2: Adapt the settled implementation**

Replace `hybrid-resnet-lines-ci` with the final neutral identity. Set CNN in
the CI patch even though the branch's amplitude base profile defaults to
Hybrid ResNet. Keep Hybrid ResNet registered and include it in the grouped
capability set. Port semantic locks, effective-output validation, and retired
name diagnostics without overwriting branch-only architecture fields.

Do not copy `refactor`'s architecture registry or strip-family patches over
this branch. Do not change `hybrid-resnet-lines` or its sealed identity.

- [ ] **Step 3: Update active and normative documentation**

Replace active `hybrid-resnet-lines-ci` usage with `lines-ci`; show explicit
`--architecture hybrid_resnet --gridsize 2` where a Hybrid example is intended.
Update the core spec to separate CI semantics from architecture capability,
and route the new design/plan from `docs/index.md`. Keep historical plans,
fixture records, and findings about completed runs literal; amend current
interpretive prose only.

- [ ] **Step 4: Preserve the sealed quality gate**

Run its exact contract-only selector and assert the tracked fixture bytes are
unchanged:

```bash
python -m pytest \
  tests/torch/test_synthetic_hybrid_resnet_gs2_integration.py \
  -m "not integration" -q
git diff --exit-code -- \
  tests/fixtures/generic_runner_hybrid_resnet_gs2_5ep_metrics.json
```

The quality command remains on `hybrid-resnet-lines`; do not make it a
`lines-ci` migration test.

- [ ] **Step 5: Run focused, integration, then comprehensive tests**

Run the Task 4 focused selectors adapted to existing branch files. Then, in
tmux with exact PID tracking:

```bash
python -m pytest -m integration -q
python -m pytest -q
```

The second command starts only after the first exits zero. Existing Hybrid
ResNet grouped integration and FFNO/CNN CI tests must retain their governing
expectations. If an integration fails, investigate before advancing.

- [ ] **Step 6: Commit a bounded branch port**

After green evidence, stage only the named implementation, tests, governing
design/plan, and current docs. Use one coherent branch-port commit unless a
separate documentation commit materially improves reviewability; squash
temporary TDD checkpoints before handoff.

```bash
git commit -m "refactor(ci): adopt architecture-neutral lines profile"
```

### Task 6: Port the final contract directly to `refactor-internal`

**Prerequisite production files:**

- Modify: `ptycho/workflows/synthetic_config.py`
- Modify: `ptycho/simulation/flat_acquisition.py`
- Modify: `ptycho/workflows/synthetic_pipeline.py`
- Modify: `scripts/simulation/synthetic_pipeline.py`
- Modify: `ptycho_torch/scaling_contract.py`
- Modify: `ptycho_torch/rect_s1s2_initialization.py`
- Modify: `ptycho_torch/rect_s1s2_sampling.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `ptycho_torch/reconstruction_evaluation.py`

**Final profile/capability production files:**

- Modify: `ptycho/workflows/synthetic_config.py`
- Create or modify: `ptycho_torch/architecture_capabilities.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`

**Tests:**

- Modify: `tests/test_synthetic_workflow_config.py`
- Modify: `tests/scripts/test_synthetic_pipeline_cli.py`
- Modify: `tests/test_flat_acquisition.py`
- Modify: `tests/test_synthetic_pipeline.py`
- Modify: `tests/torch/test_absolute_scaling_contract.py`
- Modify: `tests/torch/test_artifact_schema.py`
- Modify: `tests/torch/test_reconstruction_evaluation.py`
- Modify: `tests/torch/test_rect_s1s2_initialization.py`
- Modify: `tests/torch/test_rect_s1s2_sampling.py`
- Modify: `tests/torch/test_workflows_components.py`
- Create or modify: `tests/torch/test_architecture_capabilities.py`
- Modify as needed: `tests/torch/test_fno_lightning_integration.py`
- Modify as needed: `tests/torch/test_grid_lines_torch_runner.py`

**Documentation:** The full Task 5 documentation surface, including
`docs/index.md`, normative specs, the design, and this plan. Do not apply
documentation exclusions on this branch.

- [ ] **Step 1: Re-audit branch prerequisites at its actual tip**

Read only: confirm whether the branch contains the settled count-intensity
fields, representative dose-closure runtime, strict initialization record,
CLI flags, manifest validation, and physical-count evaluation. The audited
local tip had legacy-only synthetic scale/domain/forward literals and no
synthetic CI registry or locks, so expect the prerequisite path to be needed.

Use these settled `fno-stable` commits as provenance, not as blind whole-commit
cherry-picks: `b21b47101` (resolver), `91a0eaafe` (CLI), `8c71ff5cd`
(count acquisition), `199f4043d`/`0763b22ed`/`b9d466829` (Torch count path),
`afc2f6674` (evaluation), and
`b54b8ec14`/`9c2d1df2e`/`5c5be68b9`/`f69351fcf`/`f33fa2d2e`
(dose-closure convergence and representative sampling). Record which
branch-local behaviors are already present before preparing tests.

- [ ] **Step 2: Write prerequisite and final-state tests before production ports**

First port/adapt tests for the missing CI literals, count acquisition, strict
manifest identity, physical-count training/evaluation, representative
dose-closure record, and public CLI flags into the exact test files named
above. In the same test boundary, port the neutral profile, migration,
format-parity, identity, historical strict-bundle characterization, effective
output, and capability cases from Tasks 1 and 5. Expect:

```python
GROUPED_WORKFLOW_ARCHITECTURES == frozenset({"cnn", "hybrid_resnet"})
```

The first CI profile ever added to this branch must be `lines-ci`; no test or
implementation checkpoint should expose an architecture-named profile.

- [ ] **Step 3: Run the complete internal focused surface red**

```bash
python -m pytest \
  tests/test_synthetic_workflow_config.py \
  tests/scripts/test_synthetic_pipeline_cli.py \
  tests/test_flat_acquisition.py \
  tests/test_synthetic_pipeline.py \
  tests/torch/test_absolute_scaling_contract.py \
  tests/torch/test_artifact_schema.py \
  tests/torch/test_reconstruction_evaluation.py \
  tests/torch/test_rect_s1s2_initialization.py \
  tests/torch/test_rect_s1s2_sampling.py \
  tests/torch/test_workflows_components.py \
  tests/torch/test_architecture_capabilities.py \
  tests/torch/test_fno_lightning_integration.py \
  tests/torch/test_grid_lines_torch_runner.py -q
```

Require the existing branch contracts to stay green and the new prerequisite
and final-state cases to fail for their named missing behavior. Do not port
production code until this red boundary is recorded.

- [ ] **Step 4: Port the named prerequisite implementation**

Adapt only the missing behavior from the provenance commits into the nine
prerequisite production files listed above. Preserve internal-only
architectures and configuration fields. Land the settled count-intensity and
representative dose-closure behavior directly; do not create an intermediate
architecture-named synthetic profile.

- [ ] **Step 5: Port the settled final profile implementation**

Apply the common CNN default, semantic lock set, architecture-aware
representation check, branch-owned grouped capability, and targeted retired
name errors. Preserve every internal-only architecture field and registered
family. Explicit Hybrid selection must continue to resolve at `gridsize=2`.

- [ ] **Step 6: Update the complete documentation graph**

Add the design and plan, update active commands/guides, amend the core spec,
and add routing entries to `docs/index.md`, `docs/TESTING_GUIDE.md`, and
`docs/development/TEST_SUITE_INDEX.md`. Do not copy `refactor`'s missing-doc
exclusions onto this branch.

- [ ] **Step 7: Run focused, integration, then comprehensive tests**

Use the same order and exact-PID tmux discipline as Task 5. The branch is not
complete if a prerequisite port makes the focused tests pass but an existing
Hybrid ResNet integration regresses.

- [ ] **Step 8: Commit the direct final-state port**

Stage only the intended branch files. Use one coherent port commit unless the
prerequisite convergence is independently reviewable and needs its own commit;
squash temporary checkpoints before handoff.

```bash
git commit -m "refactor(ci): add architecture-neutral lines profile"
```

### Task 7: Perform the three-tip consistency and evidence audit

- [ ] **Step 1: Verify one active profile per tip**

For each of `refactor`, `fno-stable`, and `refactor-internal`, inspect the
committed tree and assert:

```text
active name       lines-ci
active recipe     lines-ci-v1
default arch      cnn
retired aliases   none
```

Confirm the full workflow digest still changes with architecture.

- [ ] **Step 2: Verify branch capability sets**

Read the committed capability module and exercise both consumers on each
branch. Require `{cnn}` on `refactor` and `{cnn, hybrid_resnet}` on the other
two. Confirm FFNO/FNO families were not added to grouped support merely because
their C=4 forward/backward mechanics work.

- [ ] **Step 3: Audit current documentation and historical exceptions**

Search code, tests, README, current guides, normative specs, and CLI examples
for retired names. Every remaining match must be one of:

- the targeted retirement diagnostic/tests;
- the migration design/plan;
- an explicitly labeled historical plan, command, artifact path, or evidence
  record.

There must be no active command or profile table selecting a retired name.

- [ ] **Step 4: Verify artifacts were not rewritten**

Confirm no historical `resolved_workflow.json`, manifest, tracked fixture, or
sealed amplitude-profile digest was changed solely to adopt the new name.
Confirm the inference loader remains profile-selection-independent and its
direct strict characterization passes:

```bash
python -m pytest \
  tests/torch/test_artifact_schema.py::test_retired_synthetic_profile_record_does_not_gate_strict_ci_bundle_load \
  -q
```

- [ ] **Step 5: Record final evidence**

Report, for each branch:

- final local SHA and compared remote SHA;
- focused test command/result;
- public FFNO smoke result where applicable;
- integration command/result and skip count;
- comprehensive command/result and skip count;
- allowed retired-name documentation matches;
- whether anything remains unpushed.

Do not claim a branch passed from a test run performed at another checked-out
SHA. Do not push as part of this audit unless separately instructed.

## Completion criteria

This plan is complete when all thirteen acceptance clauses in the design hold
at the recorded tips of all three branches, integration ran before the full
suite on each branch, no historical artifact was silently re-identified, and
the final report distinguishes architecture-neutral `gridsize=1` authoring
from the smaller evidence-backed grouped capability sets.
