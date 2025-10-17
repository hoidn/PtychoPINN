# CLAUDE.md

This file provides the core instructions for the Claude AI agent working on the PtychoPINN repository. It defines the rules, workflows, and authoritative sources of truth for all development tasks.

---

## 1. ⚙️ Core Agentic Workflow

**Your primary directive is to operate within the Supervisor/Engineer agentic loop.**

1.  **Supervisor (`supervisor.sh`)**: This agent plans and reviews. It reads the `docs/fix_plan.md` ledger and generates a single, focused task in `input.md`.
2.  **Engineer (`loop.sh`)**: This agent (you) executes the task defined in `input.md`. Your work must be guided by the prompts in the `prompts/` directory.
3.  **The Ledger (`docs/fix_plan.md`)**: This is the master task list. You MUST update this file at the end of every loop to document your attempt, linking to generated artifacts, even if the attempt failed. Follow the template in `prompts/update_fix_plan.md`.
4.  **"One Thing Per Loop"**: You will attempt to complete exactly one high-priority item from the `fix_plan.md` in each loop.

---

## 2. ⚠️ Core Project Directives

<directive level="critical" purpose="Consult the knowledge base first">
  Before starting any analysis or debugging, you **MUST** first search the consolidated findings ledger:
  <doc-ref type="findings">docs/findings.md</doc-ref>.
  If the issue is documented there, follow the known patterns instead of re-investigating from scratch.
</directive>

<directive level="critical" purpose="Follow formal specifications">
  The project's normative requirements are defined in the `specs/` directory. These are the **single source of truth** for all behavior.
  - For data formats: <doc-ref type="contract">specs/data_contracts.md</doc-ref>
  - For API contracts: <doc-ref type="spec">specs/ptychodus_api_spec.md</doc-ref>
</directive>

<directive level="critical" purpose="Follow the debugging methodology">
  For all new defects, you **MUST** execute the standard procedure documented in
  <doc-ref type="debugging">docs/debugging/debugging.md</doc-ref>. Record each step in the active task notes within <doc-ref type="plan">docs/fix_plan.md</doc-ref>.
</directive>

<directive level="guidance" purpose="Reuse context efficiently">
  At the start of a loop, review <doc-ref type="plan">docs/fix_plan.md</doc-ref>, the most recent plan artifacts, and the focused task in `input.md`. If those sources have not changed since your last loop, you may rely on your cached understanding instead of re-reading the full document stack. Record that decision in your attempt notes.
</directive>

<directive level="critical" purpose="Enforce Test-Driven Development">
  For any task involving new feature implementation or bug fixing, you **MUST** follow a Test-Driven Development (TDD) methodology as defined in the <doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref>.
</directive>

<directive level="important" purpose="Avoid modifying stable core logic">
  The core ptychography physics simulation and the TensorFlow model architecture are considered stable. **Do not modify the core logic in `<code-ref type="module">ptycho/model.py</code-ref>`, `<code-ref type="module">ptycho/diffsim.py</code-ref>`, or `<code-ref type="module">ptycho/tf_helper.py</code-ref>` unless explicitly directed by a plan.**
</directive>

<directive level="guidance" purpose="Store generated artifacts correctly">
  All generated reports, logs, and artifacts from a development loop **MUST** be saved to a timestamped subdirectory within `plans/active/<initiative-name>/reports/`. Do **not** leave artifacts (e.g., `*.log`, JSON dumps) at the repository root. Rule of thumb: if a file helps explain work from a loop, put it under that loop’s `plans/active/<initiative>/reports/<timestamp>/`; if it’s temporary scratch data, keep it under `tmp/` and delete before committing. The artifact path **MUST** be recorded in <doc-ref type="plan">docs/fix_plan.md</doc-ref> for traceability.
</directive>

<directive level="critical" purpose="Acknowledge the PyTorch Backend Initiative">
  A primary goal of this project is the development of a PyTorch backend. Tasks related to `ptycho_torch/` and the integration plan at `plans/ptychodus_pytorch_integration_plan.md` are of high priority.
</directive>

<directive level="critical" purpose="Enforce PyTorch Requirement">
  PyTorch is a required dependency for this project as of Phase F (INTEGRATE-PYTORCH-001). All code in `ptycho_torch/` and `tests/torch/` assumes PyTorch is installed. The package specifies `torch>=2.2` in `setup.py` install_requires. Tests in `tests/torch/` are automatically skipped in TensorFlow-only CI environments via directory-based pytest collection rules in `tests/conftest.py`, but will fail with actionable ImportError messages if PyTorch is missing in local development. Migration rationale and evidence documented in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md` and Phase F implementation logs (`docs/fix_plan.md` INTEGRATE-PYTORCH-001 history).
</directive>

---

## 3. 📚 Authoritative Document Pointers

Instead of browsing, use these as your primary entry points into the project's knowledge base.

- **Master Index:** <doc-ref type="index">docs/index.md</doc-ref> (for a full map of all documents)
- **Architectural Bible:** <doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref> (for the "two-system" architecture and core principles)
- **Test Suite Map:** <doc-ref type="test-index">docs/development/TEST_SUITE_INDEX.md</doc-ref> (to find existing tests and run commands)
- **Agentic Workflow:** <doc-ref type="workflow-guide">docs/INITIATIVE_WORKFLOW_GUIDE.md</doc-ref> (for the human-AI collaboration process)

---

## 4. 🛑 Critical Gotchas & Anti-Patterns

### 4.1. Parameter Initialization (The #1 Bug Source)

**Before calling ANY data loading or model construction functions**, you **MUST** initialize the legacy `params.cfg` dictionary. Failure to do so will cause silent shape mismatch errors.

```python
from ptycho.config.config import update_legacy_dict
from ptycho import params as p

# 1. Create modern dataclass config
config = setup_configuration(args, yaml_path)

# 2. Bridge to legacy system (MANDATORY)
update_legacy_dict(p.cfg, config)

# 3. NOW it is safe to import modules that depend on global state
from ptycho import loader, model
```
**Solution:** See <doc-ref type="troubleshooting">docs/debugging/TROUBLESHOOTING.md#shape-mismatch-errors</doc-ref>.

### 4.2. Data Format Requirements

All `.npz` datasets **MUST** conform to the specifications in `<doc-ref type="contract">specs/data_contracts.md</doc-ref>`.
-   **`diffraction`**: MUST be **amplitude** (sqrt of intensity), not intensity.
-   **`objectGuess`**: MUST be significantly larger than `probeGuess`.
-   **`Y` patches**: MUST be `complex64`. A silent `float64` conversion was the source of a major historical bug.

### 4.3. Test Harness Compatibility

Do **not** mix `unittest.TestCase` classes with pytest parametrization. Write new tests using native pytest style (plain functions or `pytest`-managed classes with fixtures) so parametrized parity tests and skip logic behave correctly. When modernizing existing tests, move entirely to pytest in a single loop.

---

## 5. Key Commands

### Environment Verification
```bash
# Install in editable mode
pip install -e .

# Run a quick, small training job to verify setup
ptycho_train --train_data_file datasets/fly/fly001_transposed.npz --n_groups 512 --output_dir verification_run
```

### Running Tests
```bash
# Run all tests (preferred)
pytest tests/

# Run a specific test or selector
pytest tests/torch/test_config_bridge.py -k parity -vv
```
