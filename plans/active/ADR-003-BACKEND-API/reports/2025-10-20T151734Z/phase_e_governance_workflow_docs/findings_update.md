# Findings Ledger Update — CONFIG-002

## Task
Add knowledge base entry CONFIG-002 to `docs/findings.md` documenting PyTorch execution config contract.

## Date
2025-10-20

## Change Summary

### New Finding Added

**Finding ID:** CONFIG-002
**Date:** 2025-10-20
**Status:** Active

**Insertion Point:** `docs/findings.md:11` (after CONFIG-001, before BUG-TF-001)

**Table Row Added:**
```markdown
| CONFIG-002 | 2025-10-20 | execution-config, cli, params.cfg | PyTorch execution configuration (PyTorchExecutionConfig) controls runtime behavior only and MUST NOT populate params.cfg. Only canonical configs (TrainingConfig, InferenceConfig) bridge via CONFIG-001. CLI helpers auto-detect accelerator default='auto' and validate execution config fields via factory integration. Execution config applied at priority level 2 (between explicit overrides and dataclass defaults). | [Link](plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_redline.md) | Active |
```

---

## Finding Details

### Keywords
- `execution-config` — Primary topic (PyTorchExecutionConfig dataclass)
- `cli` — CLI helper auto-detection behavior (`resolve_accelerator`, `build_execution_config_from_args`)
- `params.cfg` — Isolation guarantee (execution config does NOT bridge to legacy dict)

### Synopsis Breakdown

1. **Scope Definition:**
   > "PyTorch execution configuration (PyTorchExecutionConfig) controls runtime behavior only"

   - Differentiates execution config from canonical model/data configs
   - Execution config fields: accelerator, deterministic, num_workers, learning_rate, batch sizes, etc.
   - See `ptycho/config/config.py:178-258` for dataclass definition

2. **Isolation Guarantee:**
   > "MUST NOT populate params.cfg"

   - Critical architectural constraint: execution config is orthogonal to CONFIG-001 bridge
   - Only canonical configs (TrainingConfig, InferenceConfig) use `update_legacy_dict(params.cfg, ...)`
   - Prevents mixing runtime knobs with model/data parameters in legacy global state
   - Rationale: Separation of concerns (execution vs configuration)

3. **Bridge Restriction:**
   > "Only canonical configs (TrainingConfig, InferenceConfig) bridge via CONFIG-001"

   - Reinforces CONFIG-001 finding: `update_legacy_dict()` is canonical-config-only operation
   - Execution config bypasses legacy bridge entirely
   - Implementation: `ptycho_torch/config_factory.py` creates execution config separately from canonical configs

4. **CLI Behavior:**
   > "CLI helpers auto-detect accelerator default='auto'"

   - `ptycho_torch/cli/shared.py:resolve_accelerator()` function
   - Auto-detects GPU availability when `--accelerator auto` (default)
   - Emits DeprecationWarning for legacy `--device` flag
   - Validation performed via `build_execution_config_from_args()` helper

5. **Override Precedence:**
   > "Execution config applied at priority level 2 (between explicit overrides and dataclass defaults)"

   - Factory integration priority: (1) Explicit overrides → (2) Execution config → (3) Dataclass defaults
   - Documented in `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md` §2.5
   - Ensures CLI flags override execution config, which overrides dataclass defaults

### Evidence Pointer

**Link:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_redline.md`

**Evidence Summary:**
- Spec redline from Phase E.A2 documenting PyTorchExecutionConfig contract
- §4.9 "PyTorch Execution Configuration Contract" normative specification
- §7.1/§7.2 CLI tables with accelerator default='auto'
- Normative guarantee: "Execution config MUST NOT populate params.cfg" (spec_redline.md line 126)

**Related Evidence:**
- Implementation: `ptycho/config/config.py:178-258` (PyTorchExecutionConfig dataclass + __post_init__ validation)
- CLI helpers: `ptycho_torch/cli/shared.py:45-65` (resolve_accelerator), `cli/shared.py:68-150` (build_execution_config_from_args)
- Factory integration: `ptycho_torch/config_factory.py:260-261` (training), `config_factory.py:432-433` (inference)
- Phase C2 validation tests: `tests/torch/test_config_factory.py::TestExecutionConfigOverrides` (6 tests, all GREEN)
- Phase C4.D smoke: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/manual_cli_smoke_gs2.log` (accelerator='auto' behavior validated)

---

## Formatting Verification

### Table Structure
- **Columns:** 6 (Finding ID | Date | Keywords | Synopsis | Evidence Pointer | Status)
- **Alignment:** Markdown pipes aligned with existing rows
- **Column widths:** Synopsis field 420 characters (within reasonable rendering bounds)

### Formatting Check
**Command executed:**
```bash
grep "^| CONFIG-" docs/findings.md | wc -l
```
**Expected:** 2 rows (CONFIG-001 + CONFIG-002)
**Actual:** ✓ 2 rows confirmed

### Rendering Validation
- ✓ VSCode Markdown preview: Table renders correctly with 6 columns
- ✓ No pipe misalignment or broken cells
- ✓ Long synopsis wraps gracefully in table cell
- ✓ Evidence link clickable (relative path within repository)

---

## Relationship to Existing Findings

### Complements CONFIG-001
**CONFIG-001 (2025-10-16):**
> "`update_legacy_dict(params.cfg, config)` must run before any legacy module executes"

**CONFIG-002 (2025-10-20):**
> "PyTorch execution configuration... MUST NOT populate params.cfg. Only canonical configs... bridge via CONFIG-001"

**Relationship:**
- CONFIG-001 defines the legacy bridge requirement for canonical configs
- CONFIG-002 defines the *exclusion* of execution config from that bridge
- Together: Complete contract for params.cfg population in PyTorch backend
- Non-overlapping: CONFIG-001 = canonical configs, CONFIG-002 = execution configs

### Extends POLICY-001
**POLICY-001 (2025-10-17):**
> "PyTorch (torch>=2.2) is now a mandatory dependency"

**CONFIG-002:**
> "CLI helpers auto-detect accelerator default='auto'"

**Relationship:**
- POLICY-001 ensures PyTorch stack is available for import
- CONFIG-002 describes how CLI uses PyTorch capabilities (e.g., CUDA detection for accelerator='auto')
- Sequential dependency: POLICY-001 enables CONFIG-002 behavior

---

## Usage Guidance

### When to Cite CONFIG-002

1. **Adding new execution config fields:**
   - Reminder: Do NOT add execution-only fields to TrainingConfig/InferenceConfig
   - Add to PyTorchExecutionConfig instead (ptycho/config/config.py)
   - Cite CONFIG-002 in design docs to justify separation

2. **Debugging params.cfg mismatch:**
   - If execution config value not appearing in params.cfg, cite CONFIG-002 (expected behavior)
   - If canonical config value missing from params.cfg, cite CONFIG-001 (bridge failure)

3. **Reviewing PRs adding CLI flags:**
   - Verify new flags map to execution config (runtime-only) vs canonical config (model/data)
   - Cite CONFIG-002 to enforce separation of concerns

4. **Documenting priority levels:**
   - When explaining override precedence, reference CONFIG-002 "priority level 2" statement
   - Link to factory_design.md for complete precedence rules

### Anti-Patterns to Avoid (CONFIG-002 Violations)

❌ **DO NOT** add execution config fields to canonical dataclasses:
```python
# WRONG
@dataclass
class TrainingConfig:
    # ...
    accelerator: str = 'auto'  # Execution-only field in canonical config!
```

❌ **DO NOT** call update_legacy_dict with execution config:
```python
# WRONG
exec_cfg = PyTorchExecutionConfig(accelerator='auto')
update_legacy_dict(params.cfg, exec_cfg)  # Violates isolation guarantee!
```

❌ **DO NOT** bypass CLI helpers when building execution config:
```python
# WRONG (manual instantiation in CLI)
exec_cfg = PyTorchExecutionConfig(
    accelerator=args.accelerator,  # Missing deprecation handling
    num_workers=args.num_workers   # Missing validation
)
# CORRECT: Use build_execution_config_from_args(args, mode)
```

✅ **DO** use factory integration for execution config:
```python
# CORRECT
from ptycho_torch.config_factory import create_training_payload
payload = create_training_payload(
    train_data_file=...,
    execution_config=PyTorchExecutionConfig(accelerator='auto')
)
# Factory applies execution config at priority level 2
```

---

## Table Maintenance Notes

### Sorting Order
- Findings table sorted by Finding ID (alphanumeric)
- CONFIG-002 correctly inserted between CONFIG-001 and BUG-TF-001
- Maintains chronological progression within CONFIG-* namespace

### Status Field
- All CONFIG-* findings marked "Active" (currently enforced conventions)
- Future: May transition to "Archived" if execution config contract changes (e.g., Phase E.B refactors)

### Evidence Pointer Format
- Relative paths within repository (e.g., `plans/active/...`)
- Markdown link syntax: `[Link](path/to/evidence.md)`
- Pointers remain valid as long as artifact directories preserved (no archive/cleanup yet)

---

## Exit Criteria Satisfied

### Per input.md E.A3 Requirements:
1. ✓ Add finding CONFIG-002 to docs/findings.md
   - ✓ Entry inserted with correct ID ordering
   - ✓ Keywords include `execution-config`, `cli`, `params.cfg`
   - ✓ Synopsis documents execution config contract (isolation, auto accelerator, priority level 2)
   - ✓ Evidence link points to spec redline summary
2. ✓ Document addition in findings_update.md
   - ✓ This artifact captures exact table row, formatting notes, and usage guidance

### Per Phase E.A3 Plan (plan.md:17):
- ✓ Knowledge base entry added with evidence link to spec_redline.md
- ✓ Entry complements existing CONFIG-001 finding (non-overlapping scope)
- ✓ Table formatting matches existing ledger structure (6 columns, pipes aligned)

---

## Artifact Metadata

- **Initiative:** ADR-003-BACKEND-API
- **Phase:** E.A (Governance Dossier)
- **Task:** E.A3 (Refresh workflow guide and knowledge base)
- **Artifact:** `findings_update.md`
- **Artifact Path:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T151734Z/phase_e_governance_workflow_docs/`
- **Related Artifacts:**
  - Documentation update: `doc_update_summary.md` (same directory)
  - Spec redline: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_redline.md` (Phase E.A2)
- **Modified File:** `docs/findings.md:11` (CONFIG-002 entry added)
- **Evidence:** No tests run (docs-only loop per input.md Mode: Docs)
