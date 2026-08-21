# Propagating the Architecture-Consolidation Initiative onto `refactor`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the *semantics* of the 2026-08-18 architecture-consolidation initiative (Phases 0–4) and the 2026-08-19/20 fno-stable quality closeout on branch `refactor`, as branch-native changes that respect `refactor`'s older runner surfaces, leaner docs, and its own published artifact-era identifiers.

**Architecture:** Six sequential phases, each a self-contained landing on `refactor` with its own test gate and commit. Phases 0–2 are structural (dead-route deletion, backend boundary, door/facade split) and carry no persistence risk. Phases 3–5 touch the archive contract and therefore each pair a code change with a same-change amendment to `specs/ptychodus_api_spec.md`. The artifact era advances along `refactor`'s **own** `torch-artifact-portable-*` lineage — `portable-v3` for honest fields, `portable-v4` for the group/neighbor renames — never by importing `fno-stable`'s `torch-artifact-v*` identifiers.

**Tech Stack:** Python 3.11 (`ptycho311` conda env), PyTorch ≥ 2.2 + Lightning, TensorFlow (TF-side only), pytest.

**Spec / source of truth:**
- `docs/plans/2026-08-12-subtractive-refactoring-roadmap.md` §5 — the branch-propagation rules this plan obeys
- `docs/plans/2026-08-18-architecture-consolidation-roadmap.md` §3 — Decisions 1–8
- Per-phase source plans on `fno-stable` (read for the code, not for the diff):
  `docs/plans/2026-08-18-consolidation-phase-{0,1,2,3,4}-*.md`,
  `docs/plans/2026-08-19-fno-stable-refactoring-plan.md`,
  `docs/plans/2026-08-20-quality-closeout-hours-days.md`
- `specs/ptychodus_api_spec.md` §4.6 — the archive contract, amended by Tasks 4–6

---

## Global Constraints

Copied verbatim from the governing documents. Every task's requirements implicitly include this section.

**Branch-propagation rules** (`2026-08-12-subtractive-refactoring-roadmap.md` §5):

> `refactor` | Reconcile its older `scripts/training/train.py`, smaller grid-runner caller set, and leaner docs rather than importing `fno-stable` study history. Pull its upstream before integration.

> 1. re-audit live callers and governing contracts at the destination tip;
> 2. reproduce the phase's outcome with the smallest branch-native diff;
> 3. omit source-only features that do not exist on the destination rather than adding them as migration scaffolding;
> 4. run the destination's affected selectors; and
> 5. update this roadmap's status only after all intended branches have either landed the outcome or recorded a concrete branch-specific non-applicability.

**Decisions** (`2026-08-18-architecture-consolidation-roadmap.md` §3):

> 1. **Specs outrank tidiness.** Spec-pinned facades (twin `components` module paths, `load_inference_bundle*` names, CONFIG-001 load side effects, the archive contract) change only with a same-change amendment to the owning spec.
> 3. **Delete before abstracting.** A zero-caller path is removed, not wrapped.
> 4. **Loud before silent.** Where a consolidation is not yet landed, a config request the path cannot honor raises; it is never ignored.
> 5. **One door per operation.**
> 6. **Artifacts are versioned data, never code.** A format change is a new schema version with a declared upgrade path; existing schema identifiers are never reinterpreted.
> 7. **Monotone phases.** Every commit reduces path/representation/check count or moves code to its owning side.
> 8. **Propagate semantics, not commits.**

**Repository directives** (`CLAUDE.md`):
- Do **not** modify `ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py` — this plan authorizes none of them.
- Do **not** create worktrees. Work in the existing `refactor` checkout at `/home/ollie/Documents/PtychoPINN`.
- Invoke Python via PATH `python` (PYTHON-ENV-001).
- Fresh passing pytest evidence is mandatory for every task.

**Exclusions — never propagate these to `refactor`:**

| Surface | On `fno-stable` | Rule |
| --- | --- | --- |
| `docs/plans/NEURIPS-HYBRID-RESNET-2026/` | 269 files | study history — excluded |
| `docs/backlog/` | 108 files | study history — excluded |
| `docs/studies/` | 3 files | study history — excluded |
| `scripts/studies/` beyond `refactor`'s 47 files | 167 files | source-only callers — rule 3 |
| `studies/` beyond `refactor`'s 5 files | 10 files | study history — excluded |
| `.superpowers/sdd/` ledgers, `docs/plans/2026-08-2*-quality-closeout-*.md` | source-only process artifacts | not a destination deliverable |
| `torch-artifact-v1..v4` identifiers | source era lineage | Decision 6 — `refactor` keeps `portable-*` |

**Do NOT copy this source-branch defect.** `fno-stable`'s `specs/ptychodus_api_spec.md` §4.6 still declares `artifact_schema_version='torch-artifact-v2'` while `ptycho_torch/artifact_schema.py:40` stamps `torch-artifact-v4`. The external-contract spec was never amended for Phase 4 or the closeout rename — a live violation of Decision 1 and CLAUDE.md Fundamental Directive 2. On `refactor`, every era bump amends `specs/ptychodus_api_spec.md` **in the same commit**. See Task 7 for the recommended back-fix on `fno-stable`.

---

## File Structure

**Created on `refactor`:**

| File | Responsibility | Task |
| --- | --- | --- |
| `ptycho/workflows/bundle_loading.py` | TF bundle load, split out of `components.py` | 3 |
| `ptycho/workflows/config_cli.py` | `load_data`, `setup_configuration` | 3 |
| `ptycho/workflows/workflow_orchestration.py` | `run_cdi_example`, `train_cdi_model`, `save_outputs` | 3 |
| `ptycho_torch/workflows/{bundle_io,containers,dataloaders,legacy,lightning_service,orchestration,rect_s1s2}.py` | the seven owned slabs behind the torch facade | 3 |
| `ptycho_torch/checkpoint_decode.py` | one checkpoint-decode boundary for agreement checks | 3 |
| `scripts/migrate_legacy_bundle.py` | offline dill-era → JSON-manifest migration | 4 |
| `ptycho_torch/migrate_bundle.py` | `python -m ptycho_torch.migrate_bundle` era-detecting CLI | 6 |
| `docs/specs/spec-ptycho-config-bridge.md` | internal config-bridge spec shard (absent on `refactor`) | 5 |
| `tests/torch/era_fixtures.py` | per-era bundle fixtures | 4 |
| `tests/torch/test_bundle_era_matrix.py` | era × load-path matrix | 4 |
| `tests/torch/test_boundary_ratchets.py` | TF-import + serialization ratchets | 2 |
| `tests/torch/test_no_impl_to_facade_imports.py` | AST import-direction lint | 3 |
| `tests/torch/test_rename_elimination.py` | one-name-per-quantity pins | 6 |

**Modified (major):** `ptycho/workflows/components.py` (744 → ~70-line shim), `ptycho_torch/workflows/components.py` (3379 → ~110-line shim), `ptycho_torch/artifact_schema.py` (476 → ~700), `ptycho_torch/config_params.py`, `ptycho/config/config.py`, `ptycho_torch/inference.py`, `specs/ptychodus_api_spec.md`.

**Not created (rule 3 — source-only, no destination caller):** `tests/torch/test_module_size_gates.py`. See Task 7 for the rationale.

---

## Task 0: Entry gate — upstream, baseline, and caller re-audit

**Files:**
- Create: `docs/plans/2026-08-20-refactor-propagation-evidence.md`

**Interfaces:**
- Produces: a recorded green-or-known-red baseline every later task diffs against; the live-caller inventory Tasks 1–3 delete against.

- [ ] **Step 1: Pull `refactor`'s upstream** (§5 rule: "Pull its upstream before integration")

```bash
cd /home/ollie/Documents/PtychoPINN
git checkout refactor
git pull --ff-only origin refactor
git rev-parse HEAD
```

Expected: fast-forward or already-up-to-date. If it is not a fast-forward, STOP and report — this plan assumes a linear `refactor` tip.

- [ ] **Step 2: Capture the baseline suite**

```bash
bash ci/run_ci_tests.sh 2>&1 | tail -20
```

Record the exact `N passed, M failed, K skipped` line. Every later task compares against this number. A pre-existing failure is a baseline fact, not a regression — name it in the evidence file rather than fixing it inside a propagation task.

**Measured at `2948e52e1` on 2026-08-20 (4m19s):**

```
1 failed, 1482 passed, 4 skipped, 13 deselected, 11 xfailed, 1 xpassed
FAILED tests/torch/test_workflows_components.py::TestLightningCheckpointCallbacks::test_model_checkpoint_callback_configured
```

That one failure is the **known-red baseline**. Do not fix it inside a propagation task. Task 3 rewrites `test_workflows_components.py`'s import surface when the facade splits, so re-check it there: if the split resolves it, say so in the evidence file; if it still fails, it stays a named baseline fact through Task 7.

- [ ] **Step 3: Re-audit live callers at the destination tip** (§5 rule 1)

```bash
git grep -n "from ptycho.workflows.components import\|from ptycho.workflows import components" -- '*.py' | tee /tmp/refactor-tf-facade-callers.txt | wc -l
git grep -n "from ptycho_torch.workflows.components import\|from ptycho_torch.workflows import components" -- '*.py' | tee /tmp/refactor-torch-facade-callers.txt | wc -l
git grep -n "manifest.dill\|params.dill\|weights_only" -- '*.py' | tee /tmp/refactor-dill-sites.txt | wc -l
git grep -n "\bn_groups\b\|\bn_subsample\b\|\.K\b\|C_model\|C_forward\|grid_size" -- 'ptycho_torch/*.py' 'ptycho/config/*.py' | wc -l
```

- [ ] **Step 4: Write the evidence file**

Create `docs/plans/2026-08-20-refactor-propagation-evidence.md` with: the `refactor` tip SHA from Step 1, the baseline line from Step 2, and the four counts from Step 3 under a `## Entry state` heading. Add an empty `## Phase closeouts` section — each later task appends one row.

- [ ] **Step 5: Commit**

```bash
git add docs/plans/2026-08-20-refactor-consolidation-propagation.md docs/plans/2026-08-20-refactor-propagation-evidence.md
git commit -m "docs: refactor-branch consolidation propagation plan + entry evidence"
```

---

## Task 1: Phase 0 — delete verified-dead routes, make silent drift loud

**Source plan:** `fno-stable:docs/plans/2026-08-18-consolidation-phase-0-dead-code-and-guards.md`

**Files:**
- Delete: `ptycho_torch/api/example_predict_lightning.py`, `ptycho_torch/api/trainer_api.py`
- Modify: `ptycho_torch/inference.py` (unreachable barycentric branch), `ptycho/raw_data.py` (dead `diffsim` import), `ptycho/grouping.py` (oversampling guard)
- Test: `tests/torch/test_boundary_ratchets.py` (create)

**Interfaces:**
- Consumes: Task 0's caller inventory.
- Produces: `ptycho_torch/api/` reduced to `__init__.py`, `api_helper.py`, `base_api.py`, `mlflow_utils.py` — the set Task 3 splits against.

- [ ] **Step 1: Prove the two API modules are dead on `refactor`**

```bash
git grep -n "example_predict_lightning\|trainer_api" -- '*.py' '*.toml' '*.md' | grep -v "^ptycho_torch/api/"
```

Expected: no hits outside the files themselves. If anything on `refactor` imports them (`refactor` has an older caller set than `fno-stable`), do **not** delete — record the caller in the evidence file and skip to Step 4.

- [ ] **Step 2: Write the failing ratchet test**

Create `tests/torch/test_boundary_ratchets.py`:

```python
"""Self-retiring boundary ratchets: no dead API modules, no TF import in torch."""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RETIRED_API_MODULES = ("example_predict_lightning.py", "trainer_api.py")


def test_retired_api_modules_are_absent():
    present = [
        name
        for name in RETIRED_API_MODULES
        if (REPO_ROOT / "ptycho_torch" / "api" / name).exists()
    ]
    assert not present, f"retired API modules still present: {present}"


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_torch_tree_has_no_module_level_tensorflow_import():
    offenders = []
    for path in sorted((REPO_ROOT / "ptycho_torch").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if any(name.split(".")[0] == "tensorflow" for name in _module_imports(path)):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, f"module-level tensorflow import in torch tree: {offenders}"
```

- [ ] **Step 3: Run it to verify it fails**

```bash
python -m pytest tests/torch/test_boundary_ratchets.py -v
```

Expected: `test_retired_api_modules_are_absent` FAILS naming both files. `test_torch_tree_has_no_module_level_tensorflow_import` may already pass — that is fine, it is a guard against regression.

- [ ] **Step 4: Delete the dead routes**

```bash
git rm ptycho_torch/api/example_predict_lightning.py ptycho_torch/api/trainer_api.py
```

Then port the two `fno-stable` deletions that have live equivalents on `refactor`, reading the source plan for the exact hunks:
- the unreachable second barycentric branch in `ptycho_torch/inference.py` (locate with `git grep -n "_run_barycentric_inference_and_reconstruct" -- ptycho_torch/inference.py`; confirm unreachability by finding the earlier `return` on the same route before deleting);
- the dead `from ptycho import diffsim as datasets` import in `ptycho/raw_data.py` (confirm with `git grep -n "datasets\." -- ptycho/raw_data.py` returning nothing).

- [ ] **Step 5: Make the oversampling request loud** (Decision 4)

In `ptycho/grouping.py`, the grouping entry point must raise rather than silently ignore an oversampling request it cannot honor. Locate the call site with `git grep -n "enable_oversampling" -- ptycho/`. Add, at the top of the grouping function that receives it:

```python
    if enable_oversampling and (gridsize <= 1 or neighbor_pool_size < gridsize ** 2):
        raise ValueError(
            "enable_oversampling requires gridsize>1 and "
            f"neighbor_pool_size>=gridsize**2; got gridsize={gridsize!r}, "
            f"neighbor_pool_size={neighbor_pool_size!r}"
        )
```

- [ ] **Step 6: Run the ratchets and the destination selector**

```bash
python -m pytest tests/torch/test_boundary_ratchets.py -v
bash ci/run_ci_tests.sh 2>&1 | tail -5
```

Expected: ratchets PASS; suite total ≥ Task 0 baseline passed-count minus any tests that covered the deleted routes (name those in the evidence file), zero new failures.

- [ ] **Step 7: Commit**

```bash
git add -A ptycho_torch/api ptycho_torch/inference.py ptycho/raw_data.py ptycho/grouping.py tests/torch/test_boundary_ratchets.py
git commit -m "refactor: delete verified-dead torch API routes; loud oversampling guard (Phase 0)"
```

---

## Task 2: Phase 1 — backend boundary

**Source plan:** `fno-stable:docs/plans/2026-08-18-consolidation-phase-1-backend-boundary.md`

**Files:**
- Modify: `ptycho/raw_data.py`, `ptycho/workflows/training.py`, `ptycho/grouping.py`
- Test: `tests/torch/test_boundary_ratchets.py` (extend)

**Interfaces:**
- Consumes: Task 1's ratchet module.
- Produces: `ptycho_torch` importable in a TF-free process; a single `group_from_config` call site both backends use.

- [ ] **Step 1: Write the failing cold-import test**

Append to `tests/torch/test_boundary_ratchets.py`:

```python
import subprocess
import sys


def test_torch_workflow_import_does_not_load_tensorflow():
    code = (
        "import sys; import ptycho_torch.workflows.components as m; "
        "assert m is not None; "
        "sys.exit(1 if 'tensorflow' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "importing the torch workflow facade pulled in tensorflow\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/torch/test_boundary_ratchets.py::test_torch_workflow_import_does_not_load_tensorflow -v
```

Expected: FAIL (exit code 1) — `refactor`'s `ptycho_torch/workflows/components.py` imports `ptycho.raw_data`, which imports TensorFlow at module level.

- [ ] **Step 3: De-taint `ptycho/raw_data.py`**

Move the two TF-touching functions' imports inside their bodies. Only `get_image_patches` and `normalize_data` touch TF; confirm with `git grep -n "tf\.\|tf_helper" -- ptycho/raw_data.py`. Replace the module-level `import tensorflow as tf` / `from ptycho import tf_helper` with a local import inside each of those two functions, and replace any `tf.reduce_sum`-style call that has a NumPy equivalent with `np.sum`.

- [ ] **Step 4: Relocate torch orchestration out of the TF-side module**

`ptycho/workflows/training.py` on `refactor` carries torch branches and lazy `ptycho_torch` adapters. Move each torch body into `ptycho_torch/`, leaving `ptycho/workflows/backend_selector.py` as the thin, spec-pinned dispatch boundary (`specs/ptychodus_api_spec.md` mandates `run_cdi_example_with_backend` and its six `torch_*` kwargs — the dispatcher signature does not change). Find the branches with:

```bash
git grep -n "ptycho_torch" -- ptycho/workflows/training.py
```

- [ ] **Step 5: Collapse to one grouping call site**

The TF mirror passes `enable_oversampling`/`neighbor_pool_size` and no `seed`; the torch mirror passes `seed` and neither oversampling kwarg. Route both through `ptycho.grouping.group_from_config`, which takes all three. A torch caller enabling oversampling must now reach the Task 1 guard instead of being dropped.

- [ ] **Step 6: Run the ratchets and the destination selector**

```bash
python -m pytest tests/torch/test_boundary_ratchets.py -v
bash ci/run_ci_tests.sh 2>&1 | tail -5
```

Expected: all ratchets PASS; suite zero new failures. **Record the behavior change** in the evidence file: TF training now honors `subsample_seed` (it did not before the shared call site).

- [ ] **Step 7: Commit**

```bash
git add ptycho/raw_data.py ptycho/workflows/training.py ptycho/grouping.py ptycho_torch/ tests/torch/test_boundary_ratchets.py
git commit -m "refactor: de-taint raw_data; relocate torch orchestration; one grouping call site (Phase 1)"
```

---

## Task 3: Phase 2 — doors, facade split, checkpoint-decode boundary

**Source plan:** `fno-stable:docs/plans/2026-08-18-consolidation-phase-2-doors.md`; import-direction contract from `fno-stable:tests/torch/test_no_impl_to_facade_imports.py`

This is the largest task: `ptycho_torch/workflows/components.py` goes 3379 → ~110 lines and `ptycho/workflows/components.py` goes 744 → ~70. Both remain importable at their spec-pinned paths.

**Files:**
- Create: `ptycho_torch/workflows/{bundle_io,containers,dataloaders,legacy,lightning_service,orchestration,rect_s1s2}.py`, `ptycho_torch/checkpoint_decode.py`, `ptycho/workflows/{bundle_loading,config_cli,workflow_orchestration}.py`
- Modify: `ptycho_torch/workflows/components.py`, `ptycho/workflows/components.py` → pure re-export shims
- Test: `tests/torch/test_no_impl_to_facade_imports.py` (create)

**Interfaces:**
- Consumes: Task 2's TF-free torch tree.
- Produces: the module names Tasks 4–6 import directly (`bundle_io.load_inference_bundle_torch`, `lightning_service._train_with_lightning`, `checkpoint_decode`).

- [ ] **Step 1: Write the failing import-direction test**

Copy `fno-stable:tests/torch/test_no_impl_to_facade_imports.py` and trim `IMPL_MODULES` to the modules that will exist on `refactor` after this task:

```bash
git show fno-stable:tests/torch/test_no_impl_to_facade_imports.py > tests/torch/test_no_impl_to_facade_imports.py
```

Then edit `IMPL_MODULES` to remove entries `refactor` will not have. Verify each remaining entry against the file list in the **File Structure** section above; delete any that name a module this plan does not create.

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/torch/test_no_impl_to_facade_imports.py -v
```

Expected: FAIL — the impl modules do not exist yet, and both facades contain executable code rather than re-exports.

- [ ] **Step 3: Split the torch facade by responsibility**

Move each responsibility slab out of `ptycho_torch/workflows/components.py` into its own module, then replace the facade body with re-export blocks plus `__all__`. The facade must end as: a docstring, relative `from .<module> import (...)` blocks, and one `__all__` assignment — no functions, classes, `try`/`except`, calls, or absolute imports. Slab boundaries (mirror `fno-stable`'s, they are responsibility-shaped, not size-shaped):

| Module | Owns |
| --- | --- |
| `bundle_io.py` | bundle decode/encode, `load_inference_bundle_torch`, scaling metadata |
| `containers.py` | container adaptation, CI probe canonicalization, finalized CI statistics |
| `dataloaders.py` | `_build_lightning_dataloaders`, seed resolution |
| `rect_s1s2.py` | rect_s1s2 initialization, training-summary publication, dataloader settings |
| `lightning_service.py` | serving-checkpoint state machine, callbacks, `_train_with_lightning` |
| `orchestration.py` | workflow orchestration entry points |
| `legacy.py` | `run_cdi_example_torch`, `train_cdi_model_torch` |

- [ ] **Step 4: Split the TF facade the same way**

`ptycho/workflows/components.py` → `bundle_loading.py` (`load_inference_bundle`), `config_cli.py` (`load_data`, `setup_configuration`), `workflow_orchestration.py` (`run_cdi_example`, `train_cdi_model`, `save_outputs`); facade becomes a re-export shim.

- [ ] **Step 5: Point live callers at owned modules, not the facade**

```bash
git grep -ln "from ptycho.workflows import components\|from ptycho_torch.workflows import components" -- '*.py'
```

For each hit **outside** the facades themselves, rewrite the import to name the owning module. `ptycho/workflows/backend_selector.py` is the highest-value case — it must import `run_cdi_example`, `train_cdi_model`, `load_inference_bundle` (TF) and `run_cdi_example_torch`, `train_cdi_model_torch`, `load_inference_bundle_torch` (torch) directly. Update the adjacent `logger.info` strings that name `ptycho.workflows.components` so the log text matches the real route — `fno-stable` left these stale; do not copy that.

- [ ] **Step 6: Add the checkpoint-decode boundary**

Create `ptycho_torch/checkpoint_decode.py` owning the bundle-vs-runtime agreement checks that currently live inline in `ptycho_torch/inference.py` and the model constructor. Every Lightning-native loader calls it; `PtychoPINN_Lightning.__init__` keeps only construction, Lightning hparams capture, and dict→spec coercion.

- [ ] **Step 7: Run the import lint and the destination selector**

```bash
python -m pytest tests/torch/test_no_impl_to_facade_imports.py -v
bash ci/run_ci_tests.sh 2>&1 | tail -5
```

Expected: import lint PASSES; suite zero new failures. Spec-pinned names still resolve — verify explicitly:

```bash
python -c "from ptycho.workflows.components import load_inference_bundle, run_cdi_example; from ptycho_torch.workflows.components import load_inference_bundle_torch; print('facade names intact')"
```

- [ ] **Step 8: Commit**

```bash
git add ptycho/workflows/ ptycho_torch/workflows/ ptycho_torch/checkpoint_decode.py tests/torch/test_no_impl_to_facade_imports.py
git commit -m "refactor: split both workflow facades into owned modules behind pinned shims (Phase 2)"
```

---

## Task 4: Phase 3 — JSON manifest era + one sealed restore path

**Source plan:** `fno-stable:docs/plans/2026-08-18-consolidation-phase-3-bundle.md`

**Era decision for `refactor`:** the schema identifier stays `torch-artifact-portable-v2`. What changes is the *manifest container* for PyTorch archives (`manifest.dill` → `manifest.json`, `params.dill` → `params.json`) and the removal of `weights_only=False` loads. Decision 6 is satisfied because no existing identifier is reinterpreted — the payload fields are unchanged.

**Files:**
- Modify: `ptycho_torch/model_manager.py`, `ptycho_torch/workflows/bundle_io.py`, `ptycho_torch/artifact_schema.py`, `specs/ptychodus_api_spec.md`
- Create: `scripts/migrate_legacy_bundle.py`, `tests/torch/era_fixtures.py`, `tests/torch/test_bundle_era_matrix.py`

**Interfaces:**
- Consumes: Task 3's `bundle_io` module.
- Produces: `TORCH_MANIFEST_MEMBER = "manifest.json"`, `TORCH_MANIFEST_JSON_VERSION = "torch-manifest-v1"`, and `scripts/migrate_legacy_bundle.py` — the migration door Tasks 5–6 extend.

- [ ] **Step 1: Write the failing era-matrix test**

Create `tests/torch/era_fixtures.py` with a builder per era (`dill_era`, `portable_v1`, `portable_v2_json`) that writes a real `wts.h5.zip` into `tmp_path`, then `tests/torch/test_bundle_era_matrix.py`:

```python
"""Era x load-path matrix: every supported era loads, unsupported eras fail loudly."""
from __future__ import annotations

import pytest

from tests.torch.era_fixtures import build_bundle

SUPPORTED_ERAS = ["portable_v1", "portable_v2_json"]


@pytest.mark.parametrize("era", SUPPORTED_ERAS)
def test_supported_era_loads_through_strict_path(tmp_path, era):
    from ptycho_torch.workflows.bundle_io import load_inference_bundle_torch

    bundle = build_bundle(tmp_path, era)
    models, params = load_inference_bundle_torch(bundle)
    assert set(models) >= {"autoencoder", "diffraction_to_obj"}
    assert params["artifact_schema_version"].startswith("torch-artifact-portable-")


def test_dill_era_fails_loudly_naming_the_migration_script(tmp_path):
    from ptycho_torch.workflows.bundle_io import load_inference_bundle_torch

    bundle = build_bundle(tmp_path, "dill_era")
    with pytest.raises(Exception, match="migrate_legacy_bundle"):
        load_inference_bundle_torch(bundle)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/torch/test_bundle_era_matrix.py -v
```

Expected: FAIL — `portable_v2_json` does not exist yet and the dill era loads silently rather than raising.

- [ ] **Step 3: Write the JSON manifest**

`refactor`'s `ptycho_torch/artifact_schema.py:32-35` currently declares only:

```python
TORCH_ARTIFACT_BACKEND = "pytorch"
ARTIFACT_SCHEMA_V1_VERSION = "torch-artifact-portable-v1"
CURRENT_ARTIFACT_SCHEMA_VERSION = "torch-artifact-portable-v2"
TORCH_BUNDLE_VERSION = "2.0-pytorch"
```

Note the v2 identifier is a bare literal with no named constant, and there is no supported-versions tuple. Give v2 a name and add the manifest members (`TORCH_ARTIFACT_BACKEND` already exists — do not re-add it):

```python
ARTIFACT_SCHEMA_V2_VERSION = "torch-artifact-portable-v2"
CURRENT_ARTIFACT_SCHEMA_VERSION = ARTIFACT_SCHEMA_V2_VERSION
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = (
    ARTIFACT_SCHEMA_V1_VERSION,
    ARTIFACT_SCHEMA_V2_VERSION,
)
TORCH_MANIFEST_JSON_VERSION = "torch-manifest-v1"
TORCH_MANIFEST_MEMBER = "manifest.json"
TORCH_PARAMS_MEMBER = "params.json"
```

Every later era in Tasks 5–6 extends `SUPPORTED_ARTIFACT_SCHEMA_VERSIONS` rather than replacing the check.

Change `ptycho_torch/model_manager.py` to write `manifest.json` (carrying `manifest_version`, `backend: 'pytorch'`, `models`, `version`) and per-model `params.json`. TensorFlow archives keep `manifest.dill` — this change is PyTorch-scoped.

- [ ] **Step 4: Eliminate `weights_only=False` and `import dill` from `ptycho_torch/`**

The sealed-metadata payload is str/primitive/tuple dicts, so flipping the load flag is sufficient — do not re-encode. Verify with the ratchet added in the same commit; append to `tests/torch/test_boundary_ratchets.py`:

```python
def test_torch_tree_has_no_dill_and_no_unsafe_torch_load():
    offenders = []
    for path in sorted((REPO_ROOT / "ptycho_torch").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        if "import dill" in text or "weights_only=False" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, f"dill or unsafe torch.load in torch tree: {offenders}"
```

- [ ] **Step 5: Add the migration script and delete the in-process legacy restore**

Create `scripts/migrate_legacy_bundle.py` covering the dill era → `portable_v2_json`. Then delete the metadata-free / dill-era restore path from `model_manager.py`; unmigrated bundles must fail loudly naming the script (that is what the Step 1 test asserts).

- [ ] **Step 6: Amend the spec in this same commit** (Decision 1)

Edit `specs/ptychodus_api_spec.md` §4.6. Replace the single-manifest sentence with the backend-split form, keeping `refactor`'s identifiers:

```markdown
- Manifest: TensorFlow archives SHALL include a `manifest.dill` at the root with, at minimum, `{'models': [...], 'version': 'X.Y'}`.
  PyTorch archives SHALL instead include a `manifest.json` at the root with, at minimum, `{'models': [...], 'version': 'X.Y'}`, an
  explicit `manifest_version` marker (currently `'torch-manifest-v1'`), and `backend: 'pytorch'`; TensorFlow MAY omit `backend`
  and defaults to `'tensorflow'`. Per-model config projections are stored as `params.json` (PyTorch) rather than `params.dill`.
  Pre-JSON PyTorch archives (`manifest.dill` + per-model `params.dill`) are supported exclusively via
  `python scripts/migrate_legacy_bundle.py`, which migrates the manifest, params, and sealed identity together.
```

Leave the `torch-artifact-portable-v2` identity paragraph unchanged — this task does not bump the era.

- [ ] **Step 7: Run the matrix and the destination selector**

```bash
python -m pytest tests/torch/test_bundle_era_matrix.py tests/torch/test_boundary_ratchets.py -v
bash ci/run_ci_tests.sh 2>&1 | tail -5
```

Expected: all PASS; suite zero new failures.

- [ ] **Step 8: Commit**

```bash
git add ptycho_torch/ scripts/migrate_legacy_bundle.py specs/ptychodus_api_spec.md tests/torch/
git commit -m "feat(torch): JSON manifest era for PyTorch archives; sealed restore; spec §4.6 amended (Phase 3)"
```

---

## Task 5: Phase 4 — honest fields under `torch-artifact-portable-v3`

**Source plan:** `fno-stable:docs/plans/2026-08-18-consolidation-phase-4-facts.md`

This is where the C-family stops being stated four ways and `n_subsample` stops meaning two things.

**Files:**
- Modify: `ptycho_torch/config_params.py`, `ptycho_torch/artifact_schema.py`, `ptycho_torch/model_spec.py`, `ptycho/config/config.py`, `ptycho_torch/inference.py`, `specs/ptychodus_api_spec.md`
- Create: `docs/specs/spec-ptycho-config-bridge.md`
- Test: `tests/torch/test_bundle_era_matrix.py` (extend), `tests/torch/test_model_spec_v3.py` (create)

**Interfaces:**
- Consumes: Task 4's `portable_v2_json` era and migration door.
- Produces: `ARTIFACT_SCHEMA_V3_VERSION` (value `"torch-artifact-portable-v3"`), frozen `ARTIFACT_V2_*_FIELDS` tuples, and `gridsize: int` as the single channel-identity statement.

- [ ] **Step 1: Freeze the v2 era tuples before changing anything** (Decision 6)

In `ptycho_torch/artifact_schema.py`, capture the current v2 field sets as immutable literal tuples — `ARTIFACT_V2_DATA_FIELDS`, `ARTIFACT_V2_TRAINING_FIELDS`, `ARTIFACT_V2_INFERENCE_FIELDS`. They must be literals, never derived from the live dataclasses, or a later dataclass edit silently rewrites history.

- [ ] **Step 2: Write the failing honest-field test**

Create `tests/torch/test_model_spec_v3.py`:

```python
"""portable-v3 states channel identity once, as gridsize."""
from __future__ import annotations

from dataclasses import fields

import pytest

from ptycho_torch.config_params import DataConfig, ModelConfig


def test_data_config_states_channel_identity_once():
    names = {f.name for f in fields(DataConfig)}
    assert "gridsize" in names
    for retired in ("C", "grid_size"):
        assert retired not in names, f"{retired} is a duplicated channel statement"


def test_model_config_has_no_channel_twins():
    names = {f.name for f in fields(ModelConfig)}
    for retired in ("C_model", "C_forward"):
        assert retired not in names


def test_raw_selection_and_groups_per_center_are_separate():
    names = {f.name for f in fields(DataConfig)}
    assert "n_subsample" not in names, (
        "n_subsample meant raw-frame selection AND groups-per-center"
    )
```

- [ ] **Step 3: Run it to verify it fails**

```bash
python -m pytest tests/torch/test_model_spec_v3.py -v
```

Expected: all three FAIL — `refactor`'s `DataConfig` carries `C`, `K`, `n_subsample`, `grid_size`.

- [ ] **Step 4: Introduce the v3 era with a declared upgrade path**

```python
ARTIFACT_SCHEMA_V3_VERSION = "torch-artifact-portable-v3"
CURRENT_ARTIFACT_SCHEMA_VERSION = ARTIFACT_SCHEMA_V3_VERSION
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = (
    ARTIFACT_SCHEMA_V1_VERSION,
    ARTIFACT_SCHEMA_V2_VERSION,
    ARTIFACT_SCHEMA_V3_VERSION,
)
```

Add `ARTIFACT_V3_{DATA,TRAINING,INFERENCE}_FIELDS` and per-section `v2 → v3` upgrade functions, keyed by era in a `_DATA_FIELDS_BY_ERA` / `_TRAINING_FIELDS_BY_ERA` / `_INFERENCE_FIELDS_BY_ERA` lookup. The v1/v2 upgrade must **reject** an internally inconsistent payload (stored `C` disagreeing with `grid_size**2`) rather than silently preferring one — copy the `validate_legacy_channel_faithfulness` semantics from `fno-stable:ptycho_torch/artifact_schema.py`, renaming the era strings.

- [ ] **Step 5: Make the dataclasses honest**

`DataConfig`: `C`/`grid_size` → `gridsize: int`; `n_subsample` splits into the raw-frame selection field, with groups-per-center becoming a runtime argument, never persisted. `ModelConfig`: delete `C_model`/`C_forward`, derive from `gridsize`. Mirror the `n_subsample` split on `ptycho/config/config.py`'s `TrainingConfig`/`InferenceConfig`.

- [ ] **Step 6: Delete only the genuinely duplicated joins**

In `ptycho_torch/inference.py`, remove the C-join and channels fallback from `_validate_loaded_reconstruction_identity`. **Keep** the workflow-conformance comparisons — they compare the bundle against the caller-resolved workflow, which a second source decode can never see. Account for the LOC honestly in the evidence file: this is roughly a 5% deletion, not the whole ~176-line function.

- [ ] **Step 7: Create the internal config-bridge spec shard**

`refactor` has no `docs/specs/spec-ptycho-config-bridge.md`. Port `fno-stable`'s, rewriting every `torch-artifact-v3` reference to `torch-artifact-portable-v3` and dropping sections describing modules `refactor` does not have (rule 3).

- [ ] **Step 8: Amend `specs/ptychodus_api_spec.md` in this same commit**

Update the PyTorch object-policy identity paragraph to name `torch-artifact-portable-v3` / `torch-model-spec-portable-v3` as what newly written archives use, and extend the compatibility-decoding paragraph to list `portable-v1` and `portable-v2` as immutable historical schemas with a deterministic upgrade to the v3 in-memory identity.

- [ ] **Step 9: Run the tests and the destination selector**

```bash
python -m pytest tests/torch/test_model_spec_v3.py tests/torch/test_bundle_era_matrix.py -v
bash ci/run_ci_tests.sh 2>&1 | tail -20
```

Expected: new tests PASS. **This is the step where `refactor`'s existing suite will go red in bulk** — `fno-stable` took 314 failures here from tests still constructing `DataConfig(C=…)`, `ModelConfig(C_model=…)`, `DataConfig(grid_size=…)`. Migrate every such call site in `tests/` **in this same commit**; do not defer it. Verify none remain:

```bash
git grep -n "DataConfig(.*\bC=\|ModelConfig(.*C_model=\|ModelConfig(.*C_forward=\|DataConfig(.*grid_size=" -- tests/ | wc -l
```

Expected: `0`. Then re-run the suite and require zero new failures against the Task 0 baseline before committing.

- [ ] **Step 10: Commit**

```bash
git add ptycho_torch/ ptycho/config/config.py docs/specs/spec-ptycho-config-bridge.md specs/ptychodus_api_spec.md tests/
git commit -m "feat(torch): honest fields under torch-artifact-portable-v3; spec §4.6 amended (Phase 4)"
```

---

## Task 6: Closeout wave — one name per quantity under `torch-artifact-portable-v4`

**Source plans:** `fno-stable:docs/plans/2026-08-19-fno-stable-refactoring-plan.md`, `fno-stable:docs/plans/2026-08-20-quality-closeout-hours-days.md`

**Files:**
- Modify: `ptycho/config/config.py`, `ptycho/config/resolution.py`, `ptycho_torch/config_params.py`, `ptycho_torch/config_resolution.py`, `ptycho_torch/artifact_schema.py`, `ptycho_torch/inference.py`, `ptycho/workflows/training.py`, `scripts/inference/inference.py`, `specs/ptychodus_api_spec.md`
- Create: `ptycho_torch/migrate_bundle.py`, `tests/torch/test_rename_elimination.py`

**Interfaces:**
- Consumes: Task 5's `portable-v3` era.
- Produces: `training_groups` / `inference_groups` / `train_raw_selection` / `inference_raw_selection` / `neighbor_count` as the only spellings, and `python -m ptycho_torch.migrate_bundle`.

- [ ] **Step 1: Write the failing rename-elimination test**

```bash
git show fno-stable:tests/torch/test_rename_elimination.py > tests/torch/test_rename_elimination.py
```

Then edit the module docstring's era references from `torch-artifact-v4` to `torch-artifact-portable-v4`.

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/torch/test_rename_elimination.py -v
```

Expected: FAIL — `refactor`'s `TrainingConfig` still has `n_groups`, `DataConfig` still has `K`.

- [ ] **Step 3: Rename, one name per quantity**

| Old | New | Owner |
| --- | --- | --- |
| `TrainingConfig.n_groups` | `training_groups` | `ptycho/config/config.py` |
| `InferenceConfig.n_groups` | `inference_groups` | `ptycho/config/config.py` |
| `TrainingConfig.n_subsample` | `train_raw_selection` | `ptycho/config/config.py` |
| `InferenceConfig.n_subsample` | `inference_raw_selection` | `ptycho/config/config.py` |
| `DataConfig.K` | `neighbor_count` | `ptycho_torch/config_params.py` |
| `DataConfig.groups_per_center` | *(deleted; runtime constructor argument)* | `ptycho_torch/config_params.py` |

`_resolve_group_alias` gains a `canonical_name` parameter so the training and inference resolvers each name their own field in the conflict error. The `n_images` deprecation alias stays.

- [ ] **Step 4: Fence the external-contract alias, do not add a fallback**

`specs/ptychodus_api_spec.md` historically names `training_groups` for the **inference** patch. Keep accepting it, permanently and deliberately:

```python
_INFERENCE_ALIASES = {
    # Documented external-contract fence (not a fallback): the legacy
    # inference-group-count spelling "training_groups" is permanently
    # accepted; specs/ptychodus_api_spec.md and config_factory docstrings
    # historically named this key for the inference patch. Normalization
    # maps it to the canonical "inference_groups".
    "training_groups": "inference_groups",
}
```

- [ ] **Step 5: Add the v4 era and the offline migration CLI**

```python
ARTIFACT_SCHEMA_V4_VERSION = "torch-artifact-portable-v4"
CURRENT_ARTIFACT_SCHEMA_VERSION = ARTIFACT_SCHEMA_V4_VERSION
```

Add `ARTIFACT_V4_{DATA,TRAINING,INFERENCE}_FIELDS` and the v3 → v4 upgrade. Create `ptycho_torch/migrate_bundle.py` as an era-detecting offline CLI (`python -m ptycho_torch.migrate_bundle`) covering every era from the dill era through `portable-v3`.

- [ ] **Step 6: Port the two behavioral fixes, skip the ratchets**

Port from the closeout: the `inference_groups` end-to-end fix (H2 — the inference path must not require the *training* field in its phase patch) and the typed lightning seams (D1 — `_assemble_trainer` returns a typed record, not a 14-key dict).

Do **not** port these, and record why in the evidence file:
- `_build_callbacks`' 10-element positional tuple return (D2) — reproduce D1's typed record and keep `_assemble_trainer` whole; the tuple seam is a known defect on the source branch.
- `tests/torch/test_module_size_gates.py` — 31 hand-maintained line-count pins encoding `fno-stable`'s file sizes, which do not describe `refactor`. Rule 3: source-only, omit rather than adapt.
- `test_coordinator_size_gate` in `test_lightning_service_seams.py` — asserts `len(inspect.getsource(...).splitlines()) <= 100`, a shape assertion. If porting the seam tests, port only the behavioral ones (datamodule-not-rebuilt, seeding-before-construction).

- [ ] **Step 7: Amend the spec in this same commit**

Update `specs/ptychodus_api_spec.md` §4.6 to name `torch-artifact-portable-v4` / `torch-model-spec-portable-v4` as current, list `portable-v1..v3` as historical, and name `python -m ptycho_torch.migrate_bundle` as the recovery door.

- [ ] **Step 8: Verify no stale spelling survives**

```bash
git grep -n "\bn_groups\b" -- 'ptycho/config/*.py' 'ptycho_torch/*.py' | wc -l   # expect 0
git grep -n "\bn_subsample\b" -- 'ptycho/config/*.py' 'ptycho_torch/*.py' | wc -l # expect 0
python -m pytest tests/torch/test_rename_elimination.py -v
bash ci/run_ci_tests.sh 2>&1 | tail -5
```

Expected: both counts `0`; rename tests PASS; suite zero new failures. Local variables named `n_groups` inside array-shape docstrings are not config fields and may stay.

- [ ] **Step 9: Commit**

```bash
git add ptycho/ ptycho_torch/ scripts/ specs/ptychodus_api_spec.md tests/
git commit -m "feat(torch): one name per config quantity under torch-artifact-portable-v4; spec §4.6 amended"
```

---

## Task 7: Roadmap status reconciliation and source-branch back-fix

**Files:**
- Modify: `docs/plans/2026-08-12-subtractive-refactoring-roadmap.md`, `docs/plans/2026-08-18-architecture-consolidation-roadmap.md`, `docs/plans/2026-08-20-refactor-propagation-evidence.md`, `docs/index.md`

- [ ] **Step 1: Record the closeout evidence**

Append one row per phase to the `## Phase closeouts` section of the evidence file: phase, commit SHA on `refactor`, the `N passed / M failed` line, and any recorded behavior change (Task 2's `subsample_seed`) or non-applicability (Task 6's omitted ratchets).

- [ ] **Step 2: Update both roadmaps** (§5 rule 5)

In `2026-08-18-architecture-consolidation-roadmap.md`, change each Phase 0–4 Status cell from `**Complete**` to `**Complete on fno-stable and refactor**`, or record a concrete branch-specific non-applicability where one applies. Add `refactor` to the §2 audited-commit table with its post-propagation tip.

- [ ] **Step 3: Route the new docs**

Add `docs/specs/spec-ptycho-config-bridge.md` and this plan to `docs/index.md` so the documentation hub resolves them.

- [ ] **Step 4: File the source-branch defect**

`fno-stable`'s `specs/ptychodus_api_spec.md` §4.6 declares `torch-artifact-v2` while `ptycho_torch/artifact_schema.py:40` stamps `torch-artifact-v4` — Phase 4 and the closeout each changed the archive contract without the same-change spec amendment Decision 1 requires. Record this in `docs/findings.md` on `refactor` as a cross-branch finding, and open the back-fix on `fno-stable` separately. It is out of scope for this plan, which fixes the pattern on `refactor` rather than inheriting it.

- [ ] **Step 5: Commit and push**

```bash
git add docs/
git commit -m "docs: propagation closeout — roadmaps reconciled, spec-drift finding filed"
git push origin refactor
```

---

## Notes on sequencing

Tasks 1–3 carry no persistence risk and can land on consecutive days. **Task 5 is the sharp edge**: on `fno-stable` the equivalent change broke 314 tests, and that branch shipped the breakage as its Phase 4 "complete" commit. Task 5 Step 9 exists specifically so that does not repeat here — the test migration is part of the commit, not a follow-up.

Stopping after any completed task leaves `refactor` in a coherent state. Stopping *inside* Task 5 or Task 6 does not, because the era bump and the dataclass rename must land together with their spec amendment.
