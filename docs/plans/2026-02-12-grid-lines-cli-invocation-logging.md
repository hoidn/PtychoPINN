# Grid-Lines CLI Invocation Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the grid-lines study runner and comparison scripts persist the exact command lines they were invoked with, alongside parsed args metadata, in deterministic output locations.

**Architecture:** Add a small shared invocation-logging helper used by three CLI entrypoints: `scripts/studies/grid_lines_workflow.py`, `scripts/studies/grid_lines_torch_runner.py`, and `scripts/studies/grid_lines_compare_wrapper.py`. Each CLI will write a machine-readable `invocation.json` plus a copy-paste `invocation.sh` near existing artifacts (`output_dir` root or `runs/pinn_<arch>`). Keep runtime behavior unchanged except for emitting invocation artifacts.

**Tech Stack:** Python 3, `argparse`, `sys.argv`, `shlex.join`, `pathlib`, `json`, `pytest`.

**Execution discipline:**
- Use `@superpowers:test-driven-development` for each behavior change.
- Use `@superpowers:verification-before-completion` before completion claims.

---

### Task 1: Add RED tests for shared invocation artifact writer

**Files:**
- Create: `tests/test_grid_lines_invocation_logging.py`
- Create: `scripts/studies/invocation_logging.py`

**Step 1: Write failing tests for helper contract**

Add tests validating:
- writes `invocation.json` and `invocation.sh`
- `invocation.json` includes `command`, `argv`, `script`, `cwd`
- `invocation.sh` contains one exact command line
- output dir is created if missing

```python
def test_write_invocation_artifacts_writes_json_and_shell(tmp_path):
    from scripts.studies.invocation_logging import write_invocation_artifacts
    out = tmp_path / "runs" / "pinn_hybrid_resnet"
    path_json, path_sh = write_invocation_artifacts(
        output_dir=out,
        script_path="scripts/studies/grid_lines_torch_runner.py",
        argv=["--output-dir", "outputs/x", "--architecture", "hybrid_resnet"],
        parsed_args={"output_dir": "outputs/x", "architecture": "hybrid_resnet"},
    )
    assert path_json.exists()
    assert path_sh.exists()
```

**Step 2: Run tests to verify RED**

Run: `pytest tests/test_grid_lines_invocation_logging.py -v`
Expected: FAIL (module/function missing).

**Step 3: Implement minimal helper**

In `scripts/studies/invocation_logging.py`, implement:
- `build_command_line(script_path: str, argv: list[str]) -> str`
- `write_invocation_artifacts(output_dir: Path, script_path: str, argv: list[str], parsed_args: dict, extra: dict | None = None) -> tuple[Path, Path]`

Artifacts:
- `<output_dir>/invocation.json`
- `<output_dir>/invocation.sh`

**Step 4: Run tests to verify GREEN**

Run: `pytest tests/test_grid_lines_invocation_logging.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/invocation_logging.py tests/test_grid_lines_invocation_logging.py
git commit -m "feat(studies): add shared CLI invocation artifact writer"
```

### Task 2: Log invocation for Torch runner CLI

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write failing test for Torch CLI invocation artifacts**

Add test that:
- monkeypatches `run_grid_lines_torch` to avoid heavy training
- calls `main([...])`
- asserts `output_dir/runs/pinn_<arch>/invocation.json` and `invocation.sh` exist
- asserts JSON `command` contains `grid_lines_torch_runner.py` and key flags (`--architecture`, `--output-dir`)

```python
def test_main_writes_cli_invocation_artifacts(tmp_path, monkeypatch):
    ...
```

**Step 2: Run selector to verify RED**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k invocation_artifacts -v`
Expected: FAIL.

**Step 3: Implement minimal wiring**

In `scripts/studies/grid_lines_torch_runner.py`:
- change `main()` signature to `main(argv=None)`
- parse with `parser.parse_args(argv)`
- compute run artifact dir deterministically:
  - `run_dir = args.output_dir / "runs" / f"pinn_{args.architecture}"`
- call shared `write_invocation_artifacts(...)` before training call
- include invocation paths in result logging (optional)

**Step 4: Re-run test selector to verify GREEN**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k invocation_artifacts -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "feat(grid-lines-torch): persist CLI invocation artifacts per run"
```

### Task 3: Log invocation for compare wrapper CLI

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write failing compare-wrapper test**

Add test that:
- monkeypatches `run_grid_lines_compare` (no heavy work)
- calls `main([...])`
- asserts `<output_dir>/invocation.json` and `<output_dir>/invocation.sh` exist
- validates JSON `command` includes `grid_lines_compare_wrapper.py` and `--models` or `--architectures`

**Step 2: Run selector to verify RED**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k invocation_artifacts -v`
Expected: FAIL.

**Step 3: Implement minimal wiring**

In `scripts/studies/grid_lines_compare_wrapper.py`:
- in `main(argv=None)`, after `args = parse_args(argv)`, call shared helper with:
  - `output_dir=args.output_dir`
  - script path + argv + normalized parsed args
- then call `run_grid_lines_compare(...)` unchanged

**Step 4: Re-run selector to verify GREEN**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k invocation_artifacts -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(grid-lines-compare): log wrapper CLI invocation to output dir"
```

### Task 4: Log invocation for grid-lines TF workflow CLI

**Files:**
- Modify: `scripts/studies/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`

**Step 1: Write failing CLI test**

Add test in `TestGridLinesCLI` that:
- monkeypatches `run_grid_lines_workflow` to no-op
- runs `main([...])` (or parse/build path if needed)
- asserts `<output_dir>/invocation.json` + `invocation.sh`
- checks command contains `grid_lines_workflow.py`

**Step 2: Run selector to verify RED**

Run: `pytest tests/test_grid_lines_workflow.py -k invocation_artifacts -v`
Expected: FAIL.

**Step 3: Implement minimal wiring**

In `scripts/studies/grid_lines_workflow.py`:
- change `main()` to `main(argv=None)`
- parse via `parse_args(argv)`
- call shared `write_invocation_artifacts(args.output_dir, ...)`
- continue existing workflow execution

**Step 4: Re-run selector to verify GREEN**

Run: `pytest tests/test_grid_lines_workflow.py -k invocation_artifacts -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat(grid-lines-workflow): persist CLI invocation artifacts"
```

### Task 5: Documentation updates

**Files:**
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify: `docs/studies/index.md` (if present; otherwise create in this task)
- Modify: `docs/index.md` (ensure studies index remains discoverable)

**Step 1: Document invocation artifacts**

Add short “Invocation Artifacts” note under grid-lines command sections:
- Torch runner: `runs/pinn_<arch>/invocation.json`, `runs/pinn_<arch>/invocation.sh`
- Compare wrapper: `invocation.json`, `invocation.sh` at run root
- Grid workflow CLI: `invocation.json`, `invocation.sh` at run root

**Step 2: Update studies index entry**

Ensure the study row lists full command lines and references invocation artifact paths.

**Step 3: Run docs verification**

Run:

```bash
rg -n "invocation.json|invocation.sh|grid_lines_torch_runner|grid_lines_compare_wrapper|grid_lines_workflow" docs/COMMANDS_REFERENCE.md docs/studies/index.md docs/index.md
```

Expected: all references present.

**Step 4: Commit**

```bash
git add docs/COMMANDS_REFERENCE.md docs/studies/index.md docs/index.md
git commit -m "docs(grid-lines): document invocation artifact logging"
```

### Task 6: Add Repo-Wide Developer Guide for Invocation Logging

**Files:**
- Create: `docs/development/INVOCATION_LOGGING_GUIDE.md`
- Modify: `docs/index.md`
- Modify: `docs/DEVELOPER_GUIDE.md`

**Step 1: Write the new developer-facing guide**

Create `docs/development/INVOCATION_LOGGING_GUIDE.md` describing a reusable pattern
for **any** script/orchestration entrypoint in this repository. The guide must cover:
- why invocation logging exists (reproducibility, auditability, debugging)
- canonical artifact names (`invocation.json`, `invocation.sh`)
- recommended output locations for:
  - single-model script runs
  - multi-model wrappers/orchestrators
  - nested orchestration chains
- exact implementation pattern:
  - capture `argv` from CLI entrypoint
  - write parsed args snapshot
  - include working directory and timestamp
  - include optional parent run metadata for nested calls
- testing checklist template for new scripts
- minimal “drop-in” code example using the shared helper
- anti-patterns (logging only partial flags, non-deterministic paths, lossy formatting)

**Step 2: Link guide from docs hub**

Add an entry in `docs/index.md` under Architecture/Development or Workflows:
- title: `Invocation Logging Guide`
- link: `development/INVOCATION_LOGGING_GUIDE.md`
- short description: repo-wide pattern for CLI/orchestration invocation artifacts

**Step 3: Link guide from developer guide**

In `docs/DEVELOPER_GUIDE.md`, add a short section or pointer under development
practices so engineers can find the invocation logging standard during routine work.

**Step 4: Verify discoverability**

Run:

```bash
rg -n "INVOCATION_LOGGING_GUIDE|Invocation Logging Guide|invocation.json|invocation.sh" docs/index.md docs/DEVELOPER_GUIDE.md docs/development/INVOCATION_LOGGING_GUIDE.md
```

Expected: all references present and discoverable from `docs/index.md`.

**Step 5: Commit**

```bash
git add docs/development/INVOCATION_LOGGING_GUIDE.md docs/index.md docs/DEVELOPER_GUIDE.md
git commit -m "docs(dev): add repo-wide invocation logging guide for scripts and orchestrators"
```

### Task 7: Final verification sweep

**Step 1: Run focused test suite**

Run:

```bash
pytest tests/test_grid_lines_invocation_logging.py -v
pytest tests/torch/test_grid_lines_torch_runner.py -k invocation_artifacts -v
pytest tests/test_grid_lines_compare_wrapper.py -k invocation_artifacts -v
pytest tests/test_grid_lines_workflow.py -k invocation_artifacts -v
```

Expected: all PASS.

**Step 2: Run broader regression targets**

Run:

```bash
pytest tests/torch/test_grid_lines_torch_runner.py -v
pytest tests/test_grid_lines_compare_wrapper.py -v
```

Expected: PASS (or only known pre-existing skips/failures).

**Step 3: Archive logs**

Store command logs under:
- `.artifacts/studies/invocation_logging/`

**Step 4: Final commit (if prior commits squashed policy is not required)**

```bash
git status
```
