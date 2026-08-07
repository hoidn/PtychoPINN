# Compact Object-Family Reproduction YAML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact `reproduce.yaml` with one relative `output/` root while preserving the exact 12-arm scientific recipe and automatically recording runtime evidence.

**Architecture:** Keep the completed `study.yaml` and result bundle immutable. Adapt the artifact-local runner and collator to accept direct dataset paths, derive phase outputs and figure names, and observe source/input identity at runtime. Preserve compatibility with the original YAML only where it makes focused unit testing and artifact inspection useful; the new user-facing contract is `reproduce.yaml`.

**Tech Stack:** Python 3.11, PyYAML, pathlib, Git subprocesses, hashlib, pytest, existing `grid_lines_torch_runner.py`.

**Governing design:** `docs/superpowers/specs/2026-08-07-object-family-reproduction-yaml-design.md`

**Workspace constraint:** Work in the existing checkout; do not create a worktree. The implementation files live under ignored `.artifacts/`; do not force-add generated run outputs or overwrite the completed result bundle.

---

## File Map

- Create: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/reproduce.yaml` — compact authored study recipe.
- Modify: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/run_study.py` — compact-schema loading, output resolution, runtime source/input observation, and phase execution.
- Modify: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/collate_results.py` — derived figure names, provenance-free summaries, and removal of duplicate `tmp/` copies.
- Modify: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/test_run_study.py` — compact-schema, path, command-equivalence, and mutation tests.
- Preserve: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/study.yaml` and the existing `preflight/` and `full/` directories.

### Task 0: Capture The Immutable Evidence Baseline

**Files:**
- Verify only: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/study.yaml`
- Verify only: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/preflight/`
- Verify only: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/full/`

- [ ] **Step 1: Record recursive content manifests before any harness edit**

For each target, sort all regular files by relative path, compute each file's
SHA-256, then hash lines of the form `<file-sha256>  <relative-path>\n`.
The captured baseline is:

```text
study.yaml  files=1     bytes=5020        manifest_sha256=93ccfb4ec91ccd542282ec211f7db4a662139f04e4e2dd5c23ca6d0c527ce89f
preflight   files=575   bytes=562022019   manifest_sha256=8f7b9f5e5d919fd8354d4e5b35218fb2005e65e61cc29a184ce87e9e247431da
full        files=3385  bytes=3374308976  manifest_sha256=22f54f9c47c181ab42198f1c7496136254db19a8880501bb18b17217ebb75762
```

Do not add a manifest file inside either immutable tree.

### Task 1: Specify The Compact Schema In Failing Tests

**Files:**
- Modify: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/test_run_study.py`
- Test: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/test_run_study.py`

- [ ] **Step 1: Add a failing compact-config fixture test**

Add a test that writes a minimal config containing direct dataset strings and
`output: output/`, then asserts:

```python
config_path = tmp_path / "reproduce.yaml"
config_path.write_text(COMPACT_CONFIG)
config = RUN_STUDY.load_config(config_path)

assert RUN_STUDY.phase_output_root(config_path, config, "full") == (
    tmp_path / "output" / "full"
)
assert RUN_STUDY.phase_output_root(config_path, config, "preflight") == (
    tmp_path / "output" / "preflight"
)
```

- [ ] **Step 2: Add a failing direct-dataset-path test**

Create small train/test files under `tmp_path`, pass their paths as strings, and
assert `validate_datasets()` records their resolved paths, SHA-256 values, and
sizes without requiring hashes in the authored config.

- [ ] **Step 3: Add a failing command-equivalence test**

Load immutable `study.yaml` and future `reproduce.yaml`, expand all 12 rows, and
compare each command after normalizing only `--output-dir`. Assert every other
token, including options equal to runner defaults, is identical.

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
python -m pytest \
  .artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/test_run_study.py \
  -q
```

Expected: the new tests fail because `phase_output_root`, direct split strings,
and `reproduce.yaml` do not yet exist. Existing tests remain green.

### Task 2: Implement Compact Loading, Output Resolution, And Runtime Observation

**Files:**
- Modify: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/run_study.py`
- Test: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/test_run_study.py`

- [ ] **Step 1: Make config loading structural rather than provenance-versioned**

Keep the mapping check, but remove the mandatory `schema_version` comparison.
Add:

```python
def phase_output_root(config_path: Path, config: dict[str, Any], phase: str) -> Path:
    base = Path(config["output"])
    if not base.is_absolute():
        base = config_path.parent / base
    return (base / phase).resolve()
```

For legacy inspection compatibility only, fall back to
`config["outputs"][phase]` resolved against the repository root when `output`
is absent.

- [ ] **Step 2: Accept direct dataset split paths**

Add a helper that returns `record` when it is a string and `record["path"]`
for the immutable legacy YAML. Always compute the actual hash; compare against
an authored hash only for the legacy mapping form.

- [ ] **Step 3: Replace authored source pins with an observed source contract**

Define required artifact-local source files:

```python
REQUIRED_SOURCE_FILES = (
    "ptycho/FRC/fourier_ring_corr.py",
    "ptycho/FRC/spin_average.py",
)
```

Implement `observe_source()` to:

1. resolve the worktree and runner;
2. reject missing required files;
3. reject non-empty `git status --porcelain=v1 --untracked-files=no`;
4. record `git rev-parse HEAD`;
5. record the raw `git submodule status --recursive` output, allowing `-` for
   unrelated uninitialized submodules;
6. resolve the required `ptycho/FRC` checkout directly with
   `git -C <path> rev-parse HEAD` and require that revision to match the parent
   `HEAD` gitlink even when the parent command prints `-`; and
7. hash the runner and required FRC files.

Return a JSON-serializable observation. Add
`assert_source_unchanged(expected, actual)` with a clear mismatch error.

- [ ] **Step 4: Observe source around every arm**

Record the launch observation in `source_provenance.json`. Include SHA-256
records for `reproduce.yaml`, `run_study.py`, and `collate_results.py` in the
launch observation. Re-observe and compare the source plus all three harness
files immediately before and after each arm, and once after collation. Keep
per-arm invocation validation against the observed launch commit. Re-hash all
datasets at completion and compare them to launch-time records.

- [ ] **Step 5: Add fixture tests for source and dataset mutation**

Create a temporary Git repository with a committed stub runner and required FRC
files. Assert the launch observation succeeds, then modify a tracked source
file and assert observation fails because the tree is dirty. Separately record
a temporary dataset, mutate its bytes, and assert the completion comparison
rejects the changed hash.

- [ ] **Step 6: Add a no-GPU orchestration fixture test**

Invoke `main()` against a compact temporary config while monkeypatching
`run_logged()` and `validate_arm()` so no training occurs. The fake arm call
must create the expected exit marker; the fake collator call must create the
three required report artifacts. Assert:

- `source_provenance.json` contains observed source, dataset, and all three
  harness hashes;
- source observation happens before and after the arm;
- completion revalidation is written; and
- changing the runner during the fake arm causes `main()` to fail before
  completion.

- [ ] **Step 7: Run the focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: compact path/dataset tests pass; the
command-equivalence test may remain RED until `reproduce.yaml` is created.

### Task 3: Simplify Collation And Derived Outputs Test-First

**Files:**
- Modify: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/test_run_study.py`
- Modify: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/collate_results.py`

- [ ] **Step 1: Add failing tests for derived figure names**

Import the collator as a module and assert:

```python
assert COLLATOR.comparison_filename("preflight") == "preflight_comparison_table.png"
assert COLLATOR.comparison_filename("full") == "full_comparison_table.png"
```

Also assert the compact config contains no `tmp` destination or explicit
figure filename.

Build a one-row fixture containing small complex reconstruction NPZs, metrics,
resolved config, and source provenance. Invoke the real collator against it and
assert it writes the derived table, `summary.json`, and
`report_completion.json` only beneath the phase input root, with no `tmp_copy`
field and no repository-level copied output. Set `CUDA_VISIBLE_DEVICES=""`
before importing the collator's evaluation dependency.

- [ ] **Step 2: Run the focused tests and verify RED**

Expected: failure because `comparison_filename()` does not exist.

- [ ] **Step 3: Remove provenance-only collation inputs**

Delete historical-summary loading and delta generation. Read the actual source
commit from `<input-root>/source_provenance.json`. Do not require `study.id`.
Derive the filename with `comparison_filename(phase)`.

- [ ] **Step 4: Remove duplicate copied outputs**

Remove `shutil` and the `tmp_*` copy block. Keep the table, `summary.json`, and
`report_completion.json` only in the selected phase directory. Remove
`tmp_copy` from the completion payload.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all current tests pass except any test
waiting for the compact YAML file.

### Task 4: Author `reproduce.yaml` And Prove Exact Command Expansion

**Files:**
- Create: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/reproduce.yaml`
- Test: `.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/test_run_study.py`

- [ ] **Step 1: Create the compact YAML**

Write only these top-level keys:

```yaml
source:
  worktree: .worktrees/fno-dose-closure
  runner: scripts/studies/grid_lines_torch_runner.py
  cuda_visible_devices: "0"

output: output/

datasets: {}
matrix: {}
preflight: {}
common: {}
profiles: {}
collation:
  crop_border: 2
```

Populate datasets, matrix, preflight rows, `common`, and `profiles` from the
immutable YAML without dropping or changing any explicit experiment option.

- [ ] **Step 2: Run tests and verify exact equivalence**

Run the Task 1 pytest command. Expected: all tests pass, including 12/12 command
equivalence after output-path normalization.

- [ ] **Step 3: Run both dry-run phases**

```bash
python .artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/run_study.py \
  --config .artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/reproduce.yaml \
  --phase preflight --dry-run

python .artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/run_study.py \
  --config .artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/reproduce.yaml \
  --phase full --dry-run
```

Expected: two and twelve rows respectively; output roots end in
`output/preflight` and `output/full`; both commands exit `0` without creating
phase directories.

- [ ] **Step 4: Run the no-GPU orchestration and collation fixture tests**

Run their exact pytest selectors with `-q`. Expected: provenance recording,
source/dataset mutation rejection, compact collation, and absence of duplicate
copies all pass without invoking CUDA or Lightning.

### Task 5: Final Integrity And Scope Verification

**Files:**
- Verify only; do not modify completed artifacts.

- [ ] **Step 1: Run syntax and focused tests**

```bash
python -m py_compile \
  .artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/run_study.py \
  .artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/collate_results.py

python -m pytest \
  .artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/test_run_study.py \
  -q
```

Expected: compilation succeeds and all tests pass.

- [ ] **Step 2: Prove the complete immutable evidence trees are unchanged**

Recompute the Task 0 recursive manifests using the identical algorithm and
compare file count, byte count, and manifest SHA-256 for root `study.yaml`, the
entire `preflight/` tree, and the entire `full/` tree. Expected: exact matches
to all three recorded baselines.

- [ ] **Step 3: Inspect scope**

Run `git status --short` and verify no user-owned dirty path was altered. Inspect
the artifact-local files directly and verify `output/` was not created by dry
runs. Do not launch a GPU run: schema/command equivalence plus fixture-level
orchestration/collation is the acceptance claim.
