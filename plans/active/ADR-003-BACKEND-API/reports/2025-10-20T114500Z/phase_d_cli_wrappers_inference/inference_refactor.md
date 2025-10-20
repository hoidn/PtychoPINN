# Phase D.C Inference CLI Thin Wrapper Blueprint

**Initiative:** ADR-003-BACKEND-API — Standardize PyTorch backend API
**Phase:** D.C — Inference CLI Thin Wrapper Refactoring
**Date:** 2025-10-20
**Author:** Ralph (Engineer Loop)

---

## Executive Summary

This document specifies the architectural blueprint for refactoring `ptycho_torch/inference.py` from a 671-line module containing heavy orchestration logic into a lightweight argument-parsing shim that delegates to shared factories and workflow components. The refactor mirrors the training CLI thin wrapper (Phase D.B) and prepares for Phase E deprecation of legacy interfaces.

**Key Design Principles:**
1. **Single Responsibility:** CLI parses arguments and delegates; workflow components handle orchestration
2. **Shared Helpers:** Reuse `ptycho_torch/cli/shared.py` for device mapping, validation, execution config construction
3. **CONFIG-001 Compliance:** Factory layer remains responsible for `update_legacy_dict()` call
4. **Test-Driven:** RED tests capture expected behavior before implementation (Phase C2)
5. **Backward Compatible:** Legacy MLflow interface preserved unchanged

---

## Current State Analysis

### Problems Identified (from baseline.md + training blueprint)

1. **Monolithic `cli_main()`:** 348-line function handles argparse, execution config, factory invocation, bundle loading, AND inference orchestration
2. **Duplicate Device Mapping:** Identical `--device` → `--accelerator` resolution logic in inference.py:418-429 (already refactored in training CLI)
3. **Duplicate Validation:** `num_workers >= 0`, `inference_batch_size > 0` checks in CLI (should be in dataclass `__post_init__`)
4. **RawData Loading in CLI:** Lines 550-561 load data before workflow call (duplication + CONFIG-001 risk)
5. **Legacy Interface Burden:** MLflow-based inference path requires if-branch logic
6. **Manual Checkpoint Discovery:** Lines 506-546 manually search for bundle instead of using validated factory path

### Current Call Graph (Phase C4 State)

```
cli_main() [inference.py:293]
  ↓
  argparse.ArgumentParser() [319-410]
    - New interface: --model_path, --test_data, --output_dir, --n_images
    - Execution config (Phase C4): --accelerator, --num-workers, --inference-batch-size
    - Legacy: --quiet
    - Deprecated: --device
  ↓
  Execution Config Construction [414-442] ← EXTRACT to shared.py
  ↓
  create_inference_payload() [463-504] ← KEEP (factory delegation)
  ↓
  load_inference_bundle_torch() [506-546] ← KEEP (factory-validated bundle loader)
  ↓
  RawData Loading [548-561] ← REMOVE (delegate to workflow OR keep for Phase D, decide below)
  ↓
  Simplified Inference [563-641] ← REPLACE with workflow helper delegation
  ↓
  save_individual_reconstructions() [629] ← KEEP (CLI output responsibility)
```

---

## Target Architecture

### New Module Structure (Reuses Training Helpers)

```
ptycho_torch/
├─ cli/
│  ├─ __init__.py (empty)
│  ├─ shared.py (EXISTS from Phase D.B)
│  │  ├─ resolve_accelerator(args) -> str
│  │  ├─ build_execution_config_from_args(args, mode='inference') -> PyTorchExecutionConfig
│  │  └─ validate_paths(train_file, test_file, output_dir) -> None
└─ inference.py (REFACTORED)
   ├─ cli_main() (SIMPLIFIED: argparse → helpers → delegation)
   ├─ load_and_predict() (UNCHANGED: legacy MLflow path preserved)
   ├─ save_individual_reconstructions() (KEPT: CLI output artifact generation)
   └─ plot_amp_and_phase() (KEPT: legacy plotting utility)
```

### Proposed Thin Wrapper Flow

```
cli_main() [inference.py, ~100 lines after refactor]
  ↓
  argparse.ArgumentParser() [inline, existing structure]
  ↓
  Interface Detection (legacy MLflow vs new Lightning CLI)
    ├─ [LEGACY] → load_and_predict(...) [UNCHANGED]
    └─ [NEW] ↓
  ↓
  validate_paths(None, test_data_file, output_dir) [cli/shared.py]
    - Checks test file existence (train_file=None for inference mode)
    - Creates output_dir
    - Raises FileNotFoundError with clear message
  ↓
  build_execution_config_from_args(args, mode='inference') [cli/shared.py]
    - Calls resolve_accelerator(args.accelerator, args.device)
    - Constructs PyTorchExecutionConfig with validated fields
    - Maps quiet to enable_progress_bar
  ↓
  create_inference_payload(..., execution_config=...) [config_factory.py]
    - Factory handles CONFIG-001 compliance (update_legacy_dict)
    - Validates model_path contains wts.h5.zip
    - Returns InferencePayload with tf_inference_config, execution_config, etc.
  ↓
  load_inference_bundle_torch(model_path) [workflows/components.py]
    - Loads wts.h5.zip bundle (spec-compliant format per §4.6)
    - Restores params.cfg from archive (CONFIG-001 ordering)
    - Returns (models_dict, params_dict)
  ↓
  RawData.from_file(test_data_path) [raw_data.py]
    - DECISION POINT: Keep in CLI or delegate to workflow? (see Option Analysis below)
  ↓
  run_inference_and_reconstruct(model, test_data, config, output_dir) [NEW helper OR inline]
    - Wraps existing simplified inference logic (lines 563-641)
    - Calls save_individual_reconstructions() for PNG output
    - Returns (amplitude, phase, metadata_dict)
  ↓
  Success Logging
    - Prints output_dir, reconstruction artifact paths
    - Exit code 0
```

---

## Detailed Component Specifications

### Component 1: `ptycho_torch/cli/shared.py` (REUSE EXISTING)

**Status:** Implemented in Phase D.B (training refactor).

**Functions Used by Inference CLI:**

#### `resolve_accelerator(accelerator: str, device: Optional[str]) -> str`

**Usage:** Handle `--device` → `--accelerator` deprecation for inference CLI.

**Inference-Specific Notes:**
- Same logic as training CLI
- No deterministic+num_workers warning needed (inference mode)
- Emits DeprecationWarning for `--device` usage

---

#### `build_execution_config_from_args(args: argparse.Namespace, mode: str = 'inference') -> PyTorchExecutionConfig`

**Usage:** Construct `PyTorchExecutionConfig` from parsed CLI arguments with mode='inference'.

**Inference-Specific Fields:**
- `accelerator`: From `resolve_accelerator()`
- `num_workers`: DataLoader worker count (default: 0)
- `inference_batch_size`: Optional batch size override (default: None → use training batch_size)
- `enable_progress_bar`: From `not args.quiet`

**Validation:**
- `num_workers >= 0` (checked in dataclass `__post_init__`)
- `inference_batch_size` is None OR > 0 (checked in dataclass)
- `accelerator` in whitelist (checked in dataclass)

**Example:**
```python
from ptycho_torch.cli.shared import build_execution_config_from_args

args = argparse.Namespace(
    accelerator='cpu',
    device=None,
    num_workers=0,
    inference_batch_size=64,
    quiet=False,
)

execution_config = build_execution_config_from_args(args, mode='inference')
# Returns PyTorchExecutionConfig(accelerator='cpu', num_workers=0, inference_batch_size=64, enable_progress_bar=True)
```

---

#### `validate_paths(train_file: Optional[Path], test_file: Optional[Path], output_dir: Path) -> None`

**Usage:** Validate test data file exists and create output directory.

**Inference-Specific Call Pattern:**
```python
from ptycho_torch.cli.shared import validate_paths

try:
    validate_paths(
        train_file=None,  # Inference mode: no training file
        test_file=Path(args.test_data),
        output_dir=Path(args.output_dir),
    )
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)
```

**Notes:**
- Accepts `train_file=None` for inference mode
- Raises `FileNotFoundError` if test_file missing (CLI catches + formats)
- Creates `output_dir` with `mkdir -p` behavior

---

### Component 2: `ptycho_torch/inference.py` Refactored CLI

**Target State:** Thin wrapper (~100 lines for `cli_main()`, legacy path unchanged).

**Key Changes:**
1. **Remove duplicate device mapping** (lines 418-429) → call `resolve_accelerator()`
2. **Remove duplicate validation** (lines 432-435) → rely on dataclass `__post_init__()`
3. **Extract execution config construction** (lines 437-442) → call `build_execution_config_from_args()`
4. **Remove manual bundle discovery logic** → trust factory-validated path
5. **Decide RawData loading ownership** (see Option Analysis below)
6. **Extract/simplify inference orchestration** → potential helper function

**Proposed `cli_main()` Structure (Pseudocode):**

```python
def cli_main():
    """
    Inference CLI entry point: argparse → factory → workflow delegation.

    Supports two interfaces:
    1. Legacy: --run_id <mlflow_run> --infer_dir <dir> [--file_index N] (MLflow-based)
    2. New: --model_path <dir> --test_data <npz> --output_dir <dir> --n_images <int> ...

    Exit Codes:
        0: Success
        1: Argument parsing error or validation failure
        2: Inference workflow error
    """
    # --- Argument Parsing (existing structure preserved) ---
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning checkpoint inference for ptychography reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on trained model (new CLI)
  python -m ptycho_torch.inference \\
      --model_path training_outputs \\
      --test_data datasets/test.npz \\
      --output_dir inference_outputs \\
      --n_images 32 \\
      --accelerator cpu

  # Legacy MLflow-based inference
  python -m ptycho_torch.inference \\
      --run_id abc123 \\
      --infer_dir datasets/ \\
      --file_index 0
        """
    )

    # New interface flags
    parser.add_argument('--model_path', type=Path, help='Path to training output directory containing wts.h5.zip')
    parser.add_argument('--test_data', type=Path, help='Path to test NPZ file (DATA-001 compliant)')
    parser.add_argument('--output_dir', type=Path, help='Directory to save reconstruction PNGs')
    parser.add_argument('--n_images', type=int, default=32, help='Number of grouped samples')

    # Execution config flags (Phase C4)
    parser.add_argument('--accelerator', type=str, default='auto', choices=['auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'])
    parser.add_argument('--num-workers', type=int, default=0, dest='num_workers')
    parser.add_argument('--inference-batch-size', type=int, default=None, dest='inference_batch_size')

    # Shared flags
    parser.add_argument('--quiet', action='store_true', help='Suppress progress bars and verbose output')

    # Deprecated flag
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='[DEPRECATED] Use --accelerator')

    # Legacy MLflow flags (mutually exclusive with new interface)
    parser.add_argument('--run_id', type=str, help='[LEGACY] MLflow run ID')
    parser.add_argument('--infer_dir', type=str, help='[LEGACY] Inference directory')
    parser.add_argument('--file_index', type=int, default=0, help='[LEGACY] File index for multi-file inference')

    args = parser.parse_args()

    # --- Interface Detection ---
    using_legacy = args.run_id or args.infer_dir
    using_new = args.model_path or args.test_data

    if using_legacy and using_new:
        print("ERROR: Cannot mix legacy (--run_id/--infer_dir) and new (--model_path/--test_data) interfaces.")
        sys.exit(1)

    # --- Legacy Path (UNCHANGED) ---
    if using_legacy:
        # Legacy MLflow-based inference: Preserved for backward compatibility.
        # Scheduled for removal in Phase E (ADR-003).
        # If you are using this interface, please migrate to the new CLI flags.
        try:
            load_and_predict(
                run_id=args.run_id,
                ptycho_files_dir=args.infer_dir,
                file_index=args.file_index,
                config_override_path=getattr(args, 'config', None),
            )
            return 0
        except Exception as e:
            print(f"ERROR: Legacy inference failed: {e}")
            sys.exit(2)

    # --- New Path (REFACTORED) ---

    # 1. Validate paths (creates output_dir, checks test file existence)
    from ptycho_torch.cli.shared import validate_paths
    try:
        validate_paths(
            train_file=None,  # Inference mode: no training file
            test_file=args.test_data,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # 2. Build execution config (handles device deprecation, validation, warnings)
    from ptycho_torch.cli.shared import build_execution_config_from_args
    try:
        execution_config = build_execution_config_from_args(args, mode='inference')
    except ValueError as e:
        print(f"ERROR: Invalid execution config: {e}")
        sys.exit(1)

    # 3. Invoke factory (CONFIG-001 compliance handled internally)
    from ptycho_torch.config_factory import create_inference_payload

    overrides = {
        'n_groups': args.n_images,  # Map CLI arg to config field
    }

    try:
        payload = create_inference_payload(
            model_path=args.model_path,
            test_data_file=args.test_data,
            output_dir=args.output_dir,
            overrides=overrides,
            execution_config=execution_config,
        )

        # Extract configs from payload (factory already populated params.cfg)
        pt_data_config = payload.pt_data_config
        tf_inference_config = payload.tf_inference_config
        execution_config = payload.execution_config

        if not args.quiet:
            print(f"Loaded configuration from model checkpoint")
            print(f"Test data: {args.test_data}")
            print(f"Output directory: {args.output_dir}")
            print(f"N groups: {tf_inference_config.n_groups}")
            print(f"Execution config: accelerator={execution_config.accelerator}, "
                  f"num_workers={execution_config.num_workers}")

    except Exception as e:
        print(f"ERROR: Failed to create inference payload: {e}")
        sys.exit(1)

    # 4. Load bundle via spec-compliant loader (Phase C4.C6/C4.C7)
    from ptycho_torch.workflows.components import load_inference_bundle_torch
    import torch

    try:
        models_dict, params_dict = load_inference_bundle_torch(
            bundle_dir=args.model_path,
            model_name='diffraction_to_obj'
        )

        model = models_dict['diffraction_to_obj']
        model.eval()

        # Resolve device from execution config
        device_map = {
            'cpu': 'cpu',
            'gpu': 'cuda',
            'cuda': 'cuda',
            'mps': 'mps',
            'auto': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        device = device_map.get(execution_config.accelerator, 'cpu')
        model.to(device)

        if not args.quiet:
            print(f"Loaded model bundle from: {args.model_path / 'wts.h5.zip'}")
            print(f"Model device: {device}")
            print(f"Restored params.cfg from bundle (N={params_dict.get('N', 'N/A')}, "
                  f"gridsize={params_dict.get('gridsize', 'N/A')})")

    except Exception as e:
        print(f"ERROR: Failed to load inference bundle: {e}")
        sys.exit(1)

    # 5. Load test data (DECISION: Keep in CLI per Option A, or delegate to workflow per Option B)
    # Option A (Phase D): Keep RawData loading in CLI
    from ptycho.raw_data import RawData

    try:
        raw_data = RawData.from_file(str(args.test_data))

        if not args.quiet:
            print(f"Loaded test data: {raw_data.diff3d.shape[0]} scan positions")

    except Exception as e:
        print(f"ERROR: Failed to load test data: {e}")
        sys.exit(1)

    # 6. Run inference and save outputs
    # OPTION 1: Keep simplified inference logic inline (current state)
    # OPTION 2: Extract to helper function run_inference_and_reconstruct()
    # OPTION 3: Delegate to workflow component _reassemble_cdi_image_torch()
    #
    # Recommendation: Option 2 (extract to helper) for Phase D.C C3
    # Rationale:
    # - Inline logic (563-641) is too long for thin wrapper
    # - Helper function maintains testability without full workflow integration
    # - Can be migrated to workflow component in Phase E

    try:
        amplitude, phase = _run_inference_and_reconstruct(
            model=model,
            raw_data=raw_data,
            config=tf_inference_config,
            execution_config=execution_config,
            device=device,
            quiet=args.quiet,
        )

        # Save individual reconstructions (required by test contract)
        save_individual_reconstructions(amplitude, phase, args.output_dir)

        if not args.quiet:
            print(f"\nInference completed successfully!")
            print(f"Output artifacts saved to: {args.output_dir}")
            print(f"  - {args.output_dir / 'reconstructed_amplitude.png'}")
            print(f"  - {args.output_dir / 'reconstructed_phase.png'}")

        return 0

    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 2


def _run_inference_and_reconstruct(model, raw_data, config, execution_config, device, quiet=False):
    """
    Extract inference logic into testable helper function (Phase D.C).

    Args:
        model: Loaded Lightning module (eval mode)
        raw_data: RawData instance with test data
        config: TFInferenceConfig with n_groups, etc.
        execution_config: PyTorchExecutionConfig with device, batch size, etc.
        device: Torch device string ('cpu', 'cuda', 'mps')
        quiet: Suppress progress output (default: False)

    Returns:
        Tuple of (amplitude, phase) numpy arrays

    Notes:
        - Wraps existing simplified inference logic (lines 563-641)
        - Enforces DTYPE-001 (float32 for diffraction, complex64 for probe)
        - Handles shape permutations (H,W,N → N,H,W)
        - Averages across batch for single reconstruction
    """
    import torch
    import numpy as np

    # DTYPE ENFORCEMENT (Phase D1d): Cast to float32 per DATA-001
    diffraction = torch.from_numpy(raw_data.diff3d).to(device, dtype=torch.float32)
    probe = torch.from_numpy(raw_data.probeGuess).to(device, dtype=torch.complex64)

    # Handle different diffraction shapes (H, W, n) vs (n, H, W)
    if diffraction.ndim == 3 and diffraction.shape[-1] < diffraction.shape[0]:
        # Transpose from (H, W, n) to (n, H, W)
        diffraction = diffraction.permute(2, 0, 1)

    # Limit to n_groups
    diffraction = diffraction[:config.n_groups]

    # Add channel dimension if needed: (n, H, W) -> (n, 1, H, W)
    if diffraction.ndim == 3:
        diffraction = diffraction.unsqueeze(1)

    # Ensure probe is complex64
    if not torch.is_complex(probe):
        probe = probe.to(torch.complex64)

    # Add batch dimension to probe if needed
    if probe.ndim == 2:
        probe = probe.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)

    # Prepare dummy positions (needed for forward pass signature)
    batch_size = diffraction.shape[0]
    positions = torch.zeros((batch_size, 1, 1, 2), device=device)

    # Prepare scaling factors (simplified for Phase E2.C2)
    input_scale_factor = torch.ones((batch_size, 1, 1, 1), device=device)

    if not quiet:
        print(f"Running inference on {batch_size} images...")

    # Forward pass through model
    with torch.no_grad():
        reconstruction = model.forward_predict(
            diffraction,
            positions,
            probe,
            input_scale_factor
        )

    # Extract amplitude and phase
    reconstruction_cpu = reconstruction.cpu().numpy()

    # Average across batch for single reconstruction
    reconstruction_avg = np.mean(reconstruction_cpu, axis=0)

    # Remove channel dimension if present
    if reconstruction_avg.ndim == 3:
        reconstruction_avg = reconstruction_avg[0]

    result_amp = np.abs(reconstruction_avg)
    result_phase = np.angle(reconstruction_avg)

    if not quiet:
        print(f"Reconstruction shape: {reconstruction_avg.shape}")
        print(f"Amplitude range: [{result_amp.min():.4f}, {result_amp.max():.4f}]")
        print(f"Phase range: [{result_phase.min():.4f}, {result_phase.max():.4f}]")

    return result_amp, result_phase
```

---

## RawData Ownership Decision (Mirroring Training CLI Analysis)

**Problem:** Current CLI loads `RawData.from_file()` before inference (lines 550-561).

**Options:**

**Option A: CLI retains RawData loading** ✅ RECOMMENDED for Phase D.C
- ✅ PRO: Matches training CLI pattern (Phase D.B)
- ✅ PRO: Explicit CONFIG-001 ordering visible in CLI flow
- ✅ PRO: Minimal test churn (existing mocks unchanged)
- ❌ CON: CLI has business logic responsibility (data I/O)
- ❌ CON: Duplication if workflow also loads data

**Option B: Workflow handles RawData loading**
- ✅ PRO: CLI purely argument parsing + delegation
- ✅ PRO: Single data loading path (no duplication)
- ❌ CON: Requires updating `_reassemble_cdi_image_torch()` signature
- ❌ CON: Requires updating test mocks to patch workflow-internal calls

**Option C: Hybrid (accept RawData OR file paths)**
- ✅ PRO: Backward compatible with existing callers
- ✅ PRO: Flexibility for programmatic vs CLI usage
- ❌ CON: Signature complexity (`test_data: Union[RawData, Path]`)
- ❌ CON: Branching logic inside workflow

**DECISION (Phase D.C): Option A (CLI retains RawData loading) for Phase D**

**Rationale:**
1. Consistency with training CLI refactor (Phase D.B)
2. Minimal test churn (existing test mocks unchanged)
3. Explicit CONFIG-001 ordering visible in CLI flow (factory → data load)
4. Phase E can migrate to Option B after workflow signature stabilizes

**Implementation (Phase C3):**
```python
# In cli_main() (NEW PATH)
from ptycho.raw_data import RawData

# Load data AFTER factory invocation (CONFIG-001 already satisfied by factory)
raw_data = RawData.from_file(str(args.test_data))

amplitude, phase = _run_inference_and_reconstruct(
    model=model,
    raw_data=raw_data,
    config=tf_inference_config,
    execution_config=execution_config,
    device=device,
    quiet=args.quiet,
)
```

**Phase E Migration Path:** Update workflow component to accept file paths, remove CLI data loading, update test mocks.

---

## Accelerator & Execution Config Strategy

### Reuse Training CLI Patterns

**Accelerator Deprecation Warning:**
- Handled by `resolve_accelerator()` in `cli/shared.py` (no changes needed)
- Emits `DeprecationWarning` for `--device` usage

**Execution Config Validation:**
- Handled by `build_execution_config_from_args(args, mode='inference')`
- Validation (num_workers >= 0, inference_batch_size > 0) in `PyTorchExecutionConfig.__post_init__()`

**Quiet Mode Mapping:**
- `--quiet` maps to `enable_progress_bar=False`
- No `--disable_mlflow` flag for inference CLI (training-only legacy)

**Test Coverage:**
- Already covered by Phase D.B tests in `tests/torch/test_cli_shared.py`
- Inference-specific tests in `tests/torch/test_cli_inference_torch.py` (Phase C4.D1 GREEN tests)

---

## Inference Orchestration Refactor

### Current Simplified Inference (Lines 563-641)

**Purpose:** Minimal inference for CLI smoke testing (Phase E2.C2 contract).

**Logic:**
1. Load diffraction data from RawData
2. Handle shape permutations (H,W,N → N,H,W)
3. Prepare probe, positions, scaling factors
4. Forward pass through `model.forward_predict()`
5. Average across batch
6. Extract amplitude/phase
7. Save individual PNGs via `save_individual_reconstructions()`

**Problem:** Too long for thin wrapper (79 lines of business logic).

### Refactor Options

**Option 1: Keep inline** (current state)
- ✅ PRO: No signature changes
- ❌ CON: Violates thin wrapper principle (CLI has business logic)
- ❌ CON: Not independently testable

**Option 2: Extract to helper function `_run_inference_and_reconstruct()`** ✅ RECOMMENDED
- ✅ PRO: Thin wrapper delegates to helper
- ✅ PRO: Independently testable (unit tests can mock helper)
- ✅ PRO: Minimal test churn (existing integration test unchanged)
- ❌ CON: Helper still in inference.py (not in workflow component)

**Option 3: Delegate to workflow component `_reassemble_cdi_image_torch()`**
- ✅ PRO: CLI purely delegation
- ✅ PRO: Reuses production reassembly workflow
- ❌ CON: Overkill for Phase D (reassembly not required by test contract)
- ❌ CON: Requires updating test mocks to patch workflow internals

**DECISION (Phase D.C): Option 2 (extract to helper function) for Phase D.C C3**

**Rationale:**
1. Thin wrapper principle satisfied (CLI delegates to helper)
2. Independently testable (can mock helper for CLI tests)
3. Minimal test churn (integration test unchanged)
4. Phase E can migrate to full workflow component when reassembly required

**Implementation:** See `_run_inference_and_reconstruct()` in proposed `cli_main()` structure above.

---

## Open Questions & Design Choices

### Q1: Should inference CLI accept `--quiet` only (not `--disable_mlflow`)?

**Context:** Training CLI accepts both flags for backward compatibility.

**Decision:** Inference CLI accepts `--quiet` only.

**Rationale:**
- MLflow tracking never implemented for inference CLI
- No legacy users relying on `--disable_mlflow` for inference
- Cleaner interface (fewer deprecated flags)

---

### Q2: Should `_run_inference_and_reconstruct()` be in `inference.py` or `workflows/components.py`?

**Context:** Training CLI delegates to `run_cdi_example_torch()` in workflow component. Inference CLI currently has inline logic.

**Decision:** Keep in `inference.py` as private helper (`_run_inference_and_reconstruct()`) for Phase D.C.

**Rationale:**
- Test contract (Phase E2.C2) only requires PNG outputs, not full reassembly workflow
- Moving to workflow component prematurely adds unnecessary complexity
- Phase E can migrate when full production reassembly required

---

### Q3: Should bundle loading stay in CLI or move to helper?

**Context:** Lines 506-546 load bundle via `load_inference_bundle_torch()`.

**Decision:** Keep in CLI (thin wrapper delegates directly to factory-validated function).

**Rationale:**
- `load_inference_bundle_torch()` already handles CONFIG-001 ordering
- Factory validates bundle path before CLI reaches this point
- Moving to helper adds indirection without benefit

---

## Test Strategy (Phase C2 RED Coverage)

### Test File: `tests/torch/test_cli_inference_torch.py`

**Existing Tests (Phase C4.D1 GREEN):**
- `test_accelerator_flag_roundtrip` (✓ GREEN)
- `test_num_workers_flag_roundtrip` (✓ GREEN)
- `test_inference_batch_size_flag_roundtrip` (✓ GREEN)
- `test_multiple_execution_config_flags` (✓ GREEN)

**New Tests Required for Phase D.C C2 (RED):**

#### Test: `test_cli_delegates_to_helper_for_data_loading`
- Mock: `RawData.from_file()`
- Args: `--model_path <valid> --test_data <valid> --output_dir <tmp> --n_images 32`
- Expected: `RawData.from_file()` called exactly once with test_data path

#### Test: `test_cli_delegates_to_inference_helper`
- Mock: `_run_inference_and_reconstruct()`
- Args: Standard inference args
- Expected: Helper called with correct arguments (model, raw_data, config, device, quiet)

#### Test: `test_cli_calls_save_individual_reconstructions`
- Mock: `save_individual_reconstructions()`
- Args: Standard inference args
- Expected: Function called with (amplitude, phase, output_dir)

#### Test: `test_cli_validates_test_file_existence_before_factory`
- Setup: Test file does not exist
- Expected: CLI exits with code 1, error message "Test data file not found"

#### Test: `test_quiet_flag_suppresses_progress_output`
- Args: `--quiet` flag added
- Expected: No print statements for progress (capture stdout, assert empty)

---

### Test File: `tests/torch/test_cli_shared.py` (EXISTING from Phase D.B)

**Coverage Inherited from Training CLI:**
- `resolve_accelerator()` logic (5 tests, all GREEN)
- `build_execution_config_from_args(mode='training')` logic (9 tests, all GREEN)
- `validate_paths()` logic (6 tests, all GREEN)

**New Tests for Inference Mode (Phase D.C C2, RED if not already covered):**

#### Test: `test_build_execution_config_inference_mode_defaults`
- Args: `Namespace(accelerator='cpu', num_workers=0, inference_batch_size=None, quiet=False, device=None)`
- Mode: `'inference'`
- Expected: `PyTorchExecutionConfig(accelerator='cpu', num_workers=0, inference_batch_size=None, enable_progress_bar=True)`

#### Test: `test_build_execution_config_inference_mode_custom_batch_size`
- Args: `Namespace(accelerator='cpu', num_workers=0, inference_batch_size=64, quiet=False, device=None)`
- Mode: `'inference'`
- Expected: `PyTorchExecutionConfig(accelerator='cpu', num_workers=0, inference_batch_size=64, enable_progress_bar=True)`

#### Test: `test_build_execution_config_inference_mode_no_deterministic_warning`
- Args: `Namespace(accelerator='cpu', num_workers=4, ...)`
- Mode: `'inference'`
- Expected: No UserWarning emitted (deterministic warning is training-only)

---

## Expected RED Test Failures (Phase C2)

**After writing new tests but before implementing thin wrapper refactor:**

```
FAILED tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_cli_delegates_to_helper_for_data_loading - AssertionError: RawData.from_file not called (inline logic still executing)
FAILED tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_cli_delegates_to_inference_helper - AttributeError: module 'ptycho_torch.inference' has no attribute '_run_inference_and_reconstruct'
FAILED tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_cli_calls_save_individual_reconstructions - AssertionError: save_individual_reconstructions called directly (not after helper delegation)
FAILED tests/torch/test_cli_shared.py::TestBuildExecutionConfig::test_inference_mode_defaults - (already GREEN if Phase D.B implemented correctly)
```

**Expected RED Log Capture:** `pytest_cli_inference_thin_red.log` should show ~5-7 failures for missing helper function and incorrect delegation flow.

---

## Implementation Sequence (Phase D.C C3, deferred to next loop)

1. Extract `_run_inference_and_reconstruct()` helper function from inline logic (lines 563-641)
2. Refactor `cli_main()` to use `cli/shared.py` helpers:
   - Replace duplicate device mapping → `resolve_accelerator()`
   - Replace duplicate validation → `build_execution_config_from_args(args, mode='inference')`
   - Replace path checks → `validate_paths(train_file=None, test_file, output_dir)`
3. Update CLI to call extracted `_run_inference_and_reconstruct()` helper
4. Keep RawData loading in CLI (Option A)
5. Keep `save_individual_reconstructions()` call in CLI (output artifact generation)
6. Remove manual bundle discovery logic (trust factory validation)
7. Run `pytest tests/torch/test_cli_inference_torch.py -vv` → GREEN
8. Run `pytest tests/torch/test_cli_shared.py::TestBuildExecutionConfig -k inference -vv` → GREEN
9. Update `docs/workflows/pytorch.md` inference CLI examples with deprecation notices

---

## Artifacts & References

**This Blueprint References:**
- Training CLI blueprint: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/training_refactor.md`
- Baseline analysis: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/baseline.md`
- Design decisions: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/design_notes.md`
- Shared helpers: `ptycho_torch/cli/shared.py` (Phase D.B implementation)
- Spec: `specs/ptychodus_api_spec.md` §4.6 (persistence contract), §4.8 (backend selection), §7 (CLI flags)
- Workflow guide: `docs/workflows/pytorch.md` §12-13 (inference CLI usage)

**Phase C2 Artifacts (Next Step):**
- `tests/torch/test_cli_inference_torch.py` (EXTENDED: new tests for helper delegation)
- `tests/torch/test_cli_shared.py` (EXTENDED: inference mode tests if missing)
- `pytest_cli_inference_thin_red.log` (RED test output before implementation)

**Phase C3 Artifacts (Future):**
- `ptycho_torch/inference.py` (REFACTORED: thin wrapper + helper extraction)
- `pytest_cli_inference_thin_green.log` (GREEN test output after implementation)

---

## Success Criteria

Phase D.C is complete when:

1. ✅ **Blueprint Approved:** This document captures complete design (Phase C1)
2. ⏳ **RED Tests Written:** New tests in `test_cli_inference_torch.py` fail with expected errors (Phase C2, deferred)
3. ⏳ **RED Log Captured:** `pytest_cli_inference_thin_red.log` shows expected failures (Phase C2, deferred)
4. ⏳ **Helper Extracted:** `_run_inference_and_reconstruct()` function exists and passes tests (Phase C3, deferred)
5. ⏳ **CLI Refactored:** `inference.py` delegates to helpers, removes duplication (Phase C3, deferred)
6. ⏳ **GREEN Tests Verified:** Full regression suite passes (Phase C3, deferred)
7. ⏳ **Docs Updated:** `pytorch.md` reflects new CLI patterns + deprecation notices (Phase C4, deferred)

---

**Next Step:** Proceed to Phase C2 (write RED tests for thin wrapper delegation and helper extraction).
