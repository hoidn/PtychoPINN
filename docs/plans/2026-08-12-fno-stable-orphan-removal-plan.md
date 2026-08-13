# `fno-stable` Package Boundary and Orphan Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Make `pyproject.toml` the sole packaging authority, constrain the
wheel to supported runtime packages and commands, and delete the approximately
5.0 kLOC of freshly verified zero-caller code without changing runtime or
scientific behavior.

**Architecture:** This is Phase 0 of the
[subtractive refactoring roadmap](2026-08-12-subtractive-refactoring-roadmap.md).
It changes the distribution boundary and removes orphans only. It does not
consolidate live loaders, trainers, or reconstruction paths.

**Tech stack:** setuptools/PEP 517, Python standard-library wheel inspection,
pytest, existing TensorFlow and PyTorch packages.

---

## Plan summary

- **Status:** Complete on `fno-stable`, `refactor-internal`, and `refactor`.
- **Initial branch:** `fno-stable`.
- **Audit base:** `acd6c2aa6129ca984c5213238b509db0bcabcc4e`.
- **Execution base:** `9b9a8e448a5cd30cb73c64508458cd806b44f124`.
- **Authority:** `specs/data_contracts.md`, `specs/ptychodus_api_spec.md`,
  `docs/specs/spec-ptycho-interfaces.md`, the approved roadmap above, and
  repository policy in `AGENTS.md`.
- **Scope:** `pyproject.toml`, `setup.py`, the exact orphan paths in Task 2,
  their wrapper-only test, and live documentation made stale by their removal.
- **Out of scope:** Loader/grouping consolidation; `MemmapDatasetBridge`;
  `PtychoDataset.from_np`; `InMemoryPtychoDataModule`; `RawDataTorch`;
  canonical `ptycho_torch/reassembly.py`; `ptycho_torch.api`; native Torch CLI
  retirement; any numerical, configuration, artifact-schema, or scientific
  behavior change.
- **Invariants:** The four installed commands retain the same entry points;
  `ptycho` and `ptycho_torch` remain importable; the tracked Run1084 NPZ remains
  unchanged at repo-relative `datasets/Run1084_recon3_postPC_shrunk_3.npz` and
  is not installed as code; supported training, inference, synthetic, and study
  behavior is unchanged; installed metadata continues to require `torch>=2.2`.
- **Workspace rule:** Execute in the current checkout. Do not create a worktree
  and do not touch unrelated dirty or untracked files. Capture
  `git status --short` before Task 1 and preserve every pre-existing unrelated
  entry unchanged.
- **Preserved pre-existing workspace entries:** modified submodules
  `notebooks/archive/ePIE_recon_simulation` and `scripts/orchestration`, plus
  untracked `.orchestrate/` and
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_efficiency_table_summary.md.refactor-untracked-backup`.

No scientific metric run is required: every production deletion is zero-caller
and this tranche changes neither a live numerical path nor a dataset.

## Task 1: Bound the installed distribution

**Files**

- Modify: `pyproject.toml`
- Delete: `setup.py`
- Modify: `specs/ptychodus_api_spec.md`

**Contract and constraints**

- Keep namespace discovery because the installed commands target
  `scripts.training`, `scripts.inference`, and `scripts.simulation`, which do
  not currently use `__init__.py`.
- Restrict discovery to `ptycho`, `ptycho_torch`, the existing top-level `frc`
  compatibility namespace, and those three exact script namespaces. Do not
  package study trees, tests, docs, outputs, notebooks, archives, top-level
  `loaders`, or top-level `torch`.
- Exclude `ptycho.trash`, `ptycho_torch.configs`,
  `ptycho_torch.notebooks`, and the retired `ptycho_torch.beta_modules` path.
  Each exclusion must also include its `.*` descendant pattern.
- Disable implicit VCS-derived package data so excluded tracked directories and
  repository datasets are not copied back into the wheel. The tracked Run1084
  fixture remains a repo-relative scientific input, not installed package code.
- Preserve POLICY-001 by changing the unbounded `torch` dependency in
  `pyproject.toml` to `torch>=2.2`. This is the only dependency spelling carried
  from `setup.py`; its other stale pins and extras are not migrated.
- Preserve these `[project.scripts]` mappings unchanged:
  `ptycho_train`, `ptycho_inference`, `ptycho_synthetic`, and `ptycho_study`.
- Delete `setup.py`; do not merge its stale dependency pins or metadata into
  `pyproject.toml`.
- Update the existing PyTorch requirement sentence in
  `specs/ptychodus_api_spec.md` to point to `pyproject.toml` rather than the
  deleted `setup.py`; the normative minimum does not change.

**Steps**

1. In `[tool.setuptools.packages.find]`, add exact include patterns for
   `ptycho`, `ptycho.*`, `ptycho_torch`, `ptycho_torch.*`, `frc`,
   `scripts.training`, `scripts.inference`, and `scripts.simulation`; add only
   the exclusions named above.
2. Set `include-package-data = false` under `[tool.setuptools]`; do not declare
   the repository's NPZ inputs as package data.
3. Set the existing Torch dependency minimum and update the spec's packaging
   source pointer.
4. Delete `setup.py`.
5. Build an isolated wheel and inspect its contents. Do not add package marker
   files or a packaging helper script.

**Task check**

Run from the repository root:

```bash
set -e
for generated in build ptychopinn.egg-info; do
  if test -e "$generated"; then
    git check-ignore -q "$generated"
    find "$generated" -depth -delete
  fi
done
package_audit_dir=$(mktemp -d)
trap 'rm -rf "$package_audit_dir"' EXIT
python -m build --wheel --outdir "$package_audit_dir/wheel"
wheel_path=$(find "$package_audit_dir/wheel" -maxdepth 1 -name '*.whl' -print -quit)
python - "$wheel_path" <<'PY'
import sys
import zipfile

wheel = sys.argv[1]
allowed = (
    "frc/",
    "ptycho/",
    "ptycho_torch/",
    "scripts/training/",
    "scripts/inference/",
    "scripts/simulation/",
)
forbidden = (
    "ptycho/trash/",
    "ptycho_torch/beta_modules/",
    "ptycho_torch/configs/",
    "ptycho_torch/notebooks/",
)
required = {
    "frc/__init__.py",
    "ptycho/workflows/study_runner.py",
    "ptycho_torch/model.py",
    "scripts/training/train.py",
    "scripts/inference/inference.py",
    "scripts/simulation/synthetic_pipeline.py",
}
retired = {
    "ptycho_torch/reassembly_alpha.py",
    "ptycho_torch/reassembly_beta.py",
    "ptycho_torch/datagen.py",
    "ptycho_torch/model_finetuner_modified.py",
    "scripts/simulation/run_with_synthetic_lines.py",
}
with zipfile.ZipFile(wheel) as archive:
    names = {name for name in archive.namelist() if not name.endswith("/")}
bad = sorted(
    name
    for name in names
    if not name.startswith(allowed)
    and not (name.startswith("ptychopinn-") and ".dist-info/" in name)
)
assert not bad, bad[:20]
assert not any(name.startswith(forbidden) for name in names)
assert names.isdisjoint(retired), sorted(names & retired)
assert not any(name.endswith(".npz") for name in names)
assert required <= names, sorted(required - names)
PY
python -m pip install --no-deps --target "$package_audit_dir/site" "$wheel_path"
(
  cd "$package_audit_dir"
  PYTHONPATH="$package_audit_dir/site" python - <<'PY'
from importlib.metadata import distribution

import frc
from ptycho.workflows.study_runner import main as study_main
from scripts.inference.inference import main as inference_main
from scripts.simulation.synthetic_pipeline import main as synthetic_main
from scripts.training.train import main as training_main

installed = distribution("ptychopinn")
commands = {
    entry.name: entry.value
    for entry in installed.entry_points
    if entry.group == "console_scripts"
}
assert commands == {
    "ptycho_train": "scripts.training.train:main",
    "ptycho_inference": "scripts.inference.inference:main",
    "ptycho_synthetic": "scripts.simulation.synthetic_pipeline:main",
    "ptycho_study": "ptycho.workflows.study_runner:main",
}
requirements = {requirement.replace(" ", "") for requirement in installed.requires or ()}
assert any(requirement.startswith("torch>=2.2") for requirement in requirements)
assert frc.__all__ == []
assert all(callable(fn) for fn in (training_main, inference_main, synthetic_main, study_main))
PY
)
```

Expected: the wheel builds, contains only the allowed runtime roots and
distribution metadata, contains no repository NPZ inputs, contains the required
modules, installs into the temporary target, and exposes four callable entry
points.

Stop if isolated build dependencies cannot be obtained. Do not revive
`setup.py` or weaken the wheel assertion as a fallback.

## Task 2: Delete only confirmed orphans

**Files**

- Delete: `ptycho_torch/reassembly_alpha.py`
- Delete: `ptycho_torch/reassembly_beta.py`
- Delete: `ptycho_torch/beta_modules/__init__.py`
- Delete: `ptycho_torch/beta_modules/model.py`
- Delete: `ptycho_torch/beta_modules/model_test.py`
- Delete: `ptycho_torch/beta_modules/model_unet.py`
- Delete: `ptycho_torch/beta_modules/reassembly.py`
- Delete: `ptycho_torch/datagen.py`
- Delete: `ptycho_torch/model_finetuner_modified.py`
- Delete: `loaders/__init__.py`
- Delete: `loaders/als.py`
- Delete: `loaders/xpp.py`
- Delete: `torch/tf_helper.py`
- Delete: `torch/tests/tf_helper.py`
- Delete: `tests/torch/test_tf_helper.py`
- Delete: `scripts/simulation/run_with_synthetic_lines.py`
- Delete: `tests/scripts/test_synthetic_lines_wrapper.py`
- Modify: `ptycho/workflows/components.py`
- Modify: `ptycho_torch/model.py`
- Modify: `tests/torch/test_training_forward_probe_weighted_reassembly.py`
- Modify: `README.md`
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify: `docs/DEVELOPER_GUIDE.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `scripts/simulation/README.md`
- Modify: `docs/findings.md`
- Modify:
  `docs/superpowers/specs/2026-08-03-generic-runner-hybrid-resnet-gs2-quality-design.md`
- Regenerate: `docs/development/TEST_SUITE_INDEX.md`

**Contract and constraints**

- Delete, do not deprecate or replace, the zero-caller code listed above.
- The `ptycho_torch/datagen/` package is canonical; only its shadowed sibling
  file `ptycho_torch/datagen.py` is deleted.
- Top-level `torch/` is not the third-party Torch package and is not the
  canonical `ptycho_torch` package; its two tracked `tf_helper.py` copies are
  identical zero-caller remnants and are both deleted.
- `tests/torch/test_tf_helper.py` imports a nonexistent relative
  `tests.torch.tf_helper`, skips all three tests when that import fails, and
  provides no coverage for the canonical `ptycho_torch.helper`; delete it with
  the dead helper copies rather than preserving a false-green test.
- Remove `load_and_prepare_data` and its module-docstring entry from
  `ptycho/workflows/components.py`; keep the live `load_data` implementation.
- Remove historical `beta_modules` call-site wording from the surviving model
  and test docstrings without changing their executable assertions.
- Replace live wrapper instructions with `ptycho_synthetic`; historical plans,
  archives, change logs, and old DDP records remain untouched.
- Update finding `TORCH-REASSEMBLY-COM-STALE-001` to record that the two orphan
  modules named by its follow-up were deleted. Do not rewrite the finding's
  historical diagnosis.
- Regenerate the test index after deleting the wrapper-only test; do not hand
  edit unrelated rows.

**Steps**

1. Re-run the zero-live-caller search below before deletion. Stop if a new
   executable caller exists outside the paths being deleted.
2. Delete the exact files above and remove the empty tracked directories.
3. Remove `load_and_prepare_data` and stale live prose.
4. Regenerate the test-suite index:

   ```bash
   python scripts/tools/generate_test_index.py > docs/development/TEST_SUITE_INDEX.md
   ```

**Pre-deletion caller check**

```bash
rg -n \
  'reassembly_alpha|reassembly_beta|beta_modules|model_finetuner_modified|from loaders|import loaders|run_with_synthetic_lines|torch[./]tf_helper|load_and_prepare_data' \
  ptycho ptycho_torch scripts tests
```

Expected before deletion: only documentation/comments in surviving files, the
deprecated wrapper and its wrapper-only test, the deprecated function itself,
and definitions inside the paths being deleted. Any other executable caller is
a stop condition.

**Post-deletion task check**

```bash
test ! -e ptycho_torch/reassembly_alpha.py
test ! -e ptycho_torch/reassembly_beta.py
test ! -e ptycho_torch/beta_modules/__init__.py
test ! -e ptycho_torch/beta_modules/model.py
test ! -e ptycho_torch/beta_modules/model_test.py
test ! -e ptycho_torch/beta_modules/model_unet.py
test ! -e ptycho_torch/beta_modules/reassembly.py
test ! -e ptycho_torch/datagen.py
test ! -e ptycho_torch/model_finetuner_modified.py
test ! -e loaders/__init__.py
test ! -e loaders/als.py
test ! -e loaders/xpp.py
test ! -e torch/tf_helper.py
test ! -e torch/tests/tf_helper.py
test ! -e tests/torch/test_tf_helper.py
test ! -e scripts/simulation/run_with_synthetic_lines.py
test ! -e tests/scripts/test_synthetic_lines_wrapper.py
if rg -n \
  'reassembly_alpha|reassembly_beta|beta_modules|model_finetuner_modified|from loaders|import loaders|run_with_synthetic_lines|torch[./]tf_helper|load_and_prepare_data' \
  ptycho ptycho_torch scripts tests README.md \
  docs/COMMANDS_REFERENCE.md docs/DEVELOPER_GUIDE.md docs/workflows/pytorch.md; then
  exit 1
fi
python - <<'PY'
import ptycho_torch.datagen
from ptycho.workflows.components import load_data

assert ptycho_torch.datagen.__file__.endswith("ptycho_torch/datagen/__init__.py")
assert callable(load_data)
PY
```

Expected: every retired path is absent, no live runtime/test/current-guide
reference remains, and the canonical datagen package plus workflow loader still
import.

## Task 3: Verify the surviving paths and close the plan

**Files**

- Modify after successful checks:
  `docs/plans/2026-08-12-fno-stable-orphan-removal-plan.md`
- Modify after successful checks:
  `docs/plans/2026-08-12-subtractive-refactoring-roadmap.md`
- Modify routing/status documents:
  `docs/index.md`, `plans/README.md`,
  `docs/plans/2026-07-06-pipeline-consolidation.md`,
  `docs/plans/2026-07-06-pipeline-consolidation-tiers-0-2.md`,
  `docs/plans/2026-07-07-refactoring-roadmap.md`, and its four phase plans.

**Steps**

1. Re-run the Task 1 wheel/install check on the final deletion tree.
2. Run the focused selector below. Use tmux with the `ptycho311` environment if
   the command is long-running; inside the pane, launch one process and wait on
   that exact PID.
3. Inspect the scoped diff and check Markdown links.
4. Record exact pass counts and commands in this plan and mark `fno-stable`
   complete. Mark Phase 0 complete overall only after branch-native propagation
   and verification finish on both destination branches.

**Focused behavior check**

```bash
python -m pytest -q \
  tests/scripts/test_synthetic_pipeline_cli.py \
  tests/torch/test_inference_reassembly_parity.py \
  tests/torch/test_public_barycentric_workflow.py \
  tests/torch/test_varpro_probe_weighted_reassembly.py \
  tests/torch/test_training_forward_probe_weighted_reassembly.py \
  tests/torch/test_construction_consolidation.py \
  tests/torch/test_generator_registry.py \
  tests/torch/test_lightning_checkpoint.py
```

Expected: all collected tests pass; no unexpected skip, collection error, or
import error. A failure in a surviving path is investigated before completion;
do not restore an orphan merely to satisfy a stale import.

**Review checks**

```bash
git diff --check -- \
  pyproject.toml setup.py specs/ptychodus_api_spec.md \
  ptycho/workflows/components.py \
  ptycho_torch/reassembly_alpha.py ptycho_torch/reassembly_beta.py \
  ptycho_torch/beta_modules ptycho_torch/datagen.py \
  ptycho_torch/model_finetuner_modified.py ptycho_torch/model.py \
  loaders torch/tf_helper.py torch/tests/tf_helper.py \
  scripts/simulation/run_with_synthetic_lines.py \
  tests/scripts/test_synthetic_lines_wrapper.py tests/torch/test_tf_helper.py \
  tests/torch/test_training_forward_probe_weighted_reassembly.py \
  README.md docs/COMMANDS_REFERENCE.md docs/DEVELOPER_GUIDE.md \
  docs/workflows/pytorch.md scripts/simulation/README.md docs/findings.md \
  docs/development/TEST_SUITE_INDEX.md docs/index.md plans/README.md \
  docs/superpowers/specs/2026-08-03-generic-runner-hybrid-resnet-gs2-quality-design.md \
  docs/plans/2026-07-06-pipeline-consolidation.md \
  docs/plans/2026-07-06-pipeline-consolidation-tiers-0-2.md \
  docs/plans/2026-07-07-refactoring-roadmap.md \
  docs/plans/2026-07-07-refactor-phase-0-cleanup.md \
  docs/plans/2026-07-07-refactor-phase-1-safety-net.md \
  docs/plans/2026-07-07-refactor-phase-2-consolidate.md \
  docs/plans/2026-07-07-refactor-phase-3-core-extraction.md \
  docs/plans/2026-08-12-subtractive-refactoring-roadmap.md \
  docs/plans/2026-08-12-fno-stable-orphan-removal-plan.md
git diff --stat -- \
  pyproject.toml setup.py specs/ptychodus_api_spec.md \
  ptycho/workflows/components.py \
  ptycho_torch/reassembly_alpha.py ptycho_torch/reassembly_beta.py \
  ptycho_torch/beta_modules ptycho_torch/datagen.py \
  ptycho_torch/model_finetuner_modified.py ptycho_torch/model.py \
  loaders torch/tf_helper.py torch/tests/tf_helper.py \
  scripts/simulation/run_with_synthetic_lines.py \
  tests/scripts/test_synthetic_lines_wrapper.py tests/torch/test_tf_helper.py \
  tests/torch/test_training_forward_probe_weighted_reassembly.py \
  README.md docs/COMMANDS_REFERENCE.md docs/DEVELOPER_GUIDE.md \
  docs/workflows/pytorch.md scripts/simulation/README.md docs/findings.md \
  docs/development/TEST_SUITE_INDEX.md docs/index.md plans/README.md \
  docs/superpowers/specs/2026-08-03-generic-runner-hybrid-resnet-gs2-quality-design.md \
  docs/plans/2026-07-06-pipeline-consolidation.md \
  docs/plans/2026-07-06-pipeline-consolidation-tiers-0-2.md \
  docs/plans/2026-07-07-refactoring-roadmap.md \
  docs/plans/2026-07-07-refactor-phase-0-cleanup.md \
  docs/plans/2026-07-07-refactor-phase-1-safety-net.md \
  docs/plans/2026-07-07-refactor-phase-2-consolidate.md \
  docs/plans/2026-07-07-refactor-phase-3-core-extraction.md \
  docs/plans/2026-08-12-subtractive-refactoring-roadmap.md \
  docs/plans/2026-08-12-fno-stable-orphan-removal-plan.md
git status --short
python - <<'PY'
from pathlib import Path
import re

for source in (
    Path("docs/index.md"),
    Path("plans/README.md"),
    Path("docs/plans/2026-07-06-pipeline-consolidation.md"),
    Path("docs/plans/2026-07-06-pipeline-consolidation-tiers-0-2.md"),
    Path("docs/plans/2026-07-07-refactoring-roadmap.md"),
    Path("docs/plans/2026-07-07-refactor-phase-0-cleanup.md"),
    Path("docs/plans/2026-07-07-refactor-phase-1-safety-net.md"),
    Path("docs/plans/2026-07-07-refactor-phase-2-consolidate.md"),
    Path("docs/plans/2026-07-07-refactor-phase-3-core-extraction.md"),
    Path("docs/superpowers/specs/2026-08-03-generic-runner-hybrid-resnet-gs2-quality-design.md"),
    Path("docs/plans/2026-08-12-subtractive-refactoring-roadmap.md"),
    Path("docs/plans/2026-08-12-fno-stable-orphan-removal-plan.md"),
):
    text = source.read_text()
    for target in re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", text):
        if "://" not in target:
            assert (source.parent / target).resolve().exists(), (source, target)
PY
```

Expected: no whitespace errors in task-scoped paths, no unrelated change was
introduced relative to the recorded pre-task status, the test index drops the
two deleted-test rows and refreshes the surviving reassembly-test docstring
row, and all changed routing links resolve.

No repository-wide or scientific-quality suite is required for this
zero-caller, packaging-only tranche. Do not broaden the plan in response to an
unrelated supplemental failure.

## `fno-stable` execution evidence

Executed 2026-08-13 from
`9b9a8e448a5cd30cb73c64508458cd806b44f124`:

- The pre-deletion caller audit found no executable caller outside the planned
  deletion set. The post-deletion absence/import check passed and confirmed
  that `ptycho_torch.datagen` resolves to the retained package.
- A clean isolated wheel contained 173 files, no NPZ input, no retired member,
  and only the approved roots. Target installation exposed the four exact
  commands and metadata retained `torch>=2.2`.
- The focused selector ran in `ptycho311` under tracked PID `2862551`, exited
  `0`, and reported `144 passed, 19 warnings in 19.14s`; the warnings only note
  that an available GPU was not selected by CPU Lightning tests.
- `git diff --check`, the scoped caller/reference audit, and routing-link checks
  passed. Independent code and documentation reviews passed after their
  findings were resolved.

Two plan corrections were required during execution. The local
`ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz` was an ignored artifact
different from the tracked top-level fixture, so repository NPZ inputs are
explicitly excluded from the wheel. A prior build also proved that setuptools
can reuse stale `build/` contents, so the final gate now clears validated
generated build state and rejects each retired in-package member explicitly.
The test-index generator exposed unrelated pre-existing index drift; its
output was compared in full, then only the two removed rows and the required
surviving-docstring refresh were applied.

## `refactor-internal` propagation evidence

Propagated semantically on 2026-08-13 from
`56db36ef5ea4e355f64e8b3a39a1e47e1609cf53`. The branch intentionally has no
`ptycho_study` command or study runner, so the package gate retained its three
installed commands and did not reintroduce either surface. Its clean isolated
wheel contained 169 files, including the initialized `ptycho/FRC` submodule,
no NPZ input, and no retired member. Target installation exposed the three
exact commands and retained `torch>=2.2`.

The branch-local focused selector ran in `ptycho311` under tracked PID
`2884999`, exited `0`, and reported `129 passed, 19 warnings in 14.96s`.
Post-deletion caller/import checks and `git diff --check` passed. The generated
test index again contained unrelated pre-existing drift, so only the two
deleted rows and required surviving-docstring refresh were applied.

## `refactor` propagation evidence

Before propagation, `origin/refactor` was fetched and a fast-forward-only pull
confirmed the local tip
`f2523bfb23a0f8a0c79c12c14a987dd59777cfdf` was current and seven commits
ahead. The three top-level `loaders` files and the internal documentation
surfaces were already absent, so they were not recreated. The public branch
retained its four commands, study/Hydra surface, `pydantic-settings`
dependency, and existing pytest markers.

With the required `ptycho/FRC` submodule initialized, the clean isolated wheel
contained 163 files, no NPZ input, and no retired member. Target installation
exposed the four exact commands, retained `torch>=2.2`, and preserved the
branch's Hydra and Pydantic Settings requirements. The focused selector ran in
`ptycho311` under tracked PID `2894039`, exited `0`, and reported
`107 passed, 10 warnings in 12.67s`. Post-deletion caller/import checks and
`git diff --check` passed.

## Completion criteria

- [x] `pyproject.toml` is the sole packaging authority.
- [x] A clean wheel contains only the approved runtime roots, no repository NPZ
  inputs, and four installed command mappings.
- [x] Every Task 2 orphan and the wrapper-only test are absent.
- [x] Surviving canonical synthetic, model-construction, checkpoint, and
  reassembly selectors pass freshly.
- [x] Current guides and generated test routing contain no stale instruction to
  use a removed path.
- [x] The tranche introduced no unrelated workspace change and preserved the
  recorded pre-existing dirty baseline.
- [x] The plan records fresh evidence and the roadmap accurately marks Phase 0
  complete on all three branches.

## Stop conditions

Stop instead of expanding the tranche if:

- a supposedly orphaned path has a new executable caller;
- removing a path requires a compatibility shim or changes a public signature;
- the wheel cannot retain all installed commands within the explicit package
  boundary;
- a focused failure is attributable to executable behavior in a surviving
  path; or
- completion would require changing a normative scientific or external
  contract.

Route any such result into a small amendment or the relevant later roadmap
phase. Do not turn Phase 0 into loader, trainer, or API consolidation.
