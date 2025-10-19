# CLI Inventory — PyTorch Backend API Surface
**Initiative:** ADR-003-BACKEND-API
**Phase:** A — Architecture Carve-Out
**Artifact:** CLI Flag Inventory + TensorFlow Parity Analysis
**Timestamp:** 2025-10-19T225905Z
**Raw Command Output:** `logs/a1_cli_flags.txt`

---

## 1. PyTorch Training CLI (`ptycho_torch/train.py`)

### 1.1 New Interface (Phase E2.C1)
Entry point: `cli_main()` lines 366-520

| Flag | Source | Type | Required | Default | Target Config Field(s) | Flow Path | Notes / TODO |
|------|--------|------|----------|---------|------------------------|-----------|--------------|
| `--train_data_file` | L380 | str | YES | — | `DataConfig.N` (inferred via `_infer_probe_size` L96), `TFTrainingConfig.train_data_file` | CLI → `_infer_probe_size()` L468-473 → `DataConfig.N` L23; CLI → bridge override L561 → `train_data_file_path` (via KEY_MAPPINGS) | Probe size auto-inferred from `probeGuess.shape[0]` in NPZ metadata. Enforces existence check L431-436. |
| `--test_data_file` | L382 | str | NO | None | `TFTrainingConfig.test_data_file` | CLI → bridge override L562 → `test_data_file_path` (via KEY_MAPPINGS) | Optional validation data. Existence check L437-442 if provided. |
| `--output_dir` | L384 | str | YES | — | `TFTrainingConfig.output_dir` | CLI → bridge override L563 → `output_prefix` (via KEY_MAPPINGS L310) | Validated for write permissions L443-453. Parent directory auto-created. |
| `--max_epochs` | L386 | int | NO | 100 | `TFTrainingConfig.nepochs` | CLI → `TrainingConfig.epochs` L491 → bridge rename L187 → `nepochs` | **SEMANTIC MISMATCH:** TF uses `nepochs` internally (ptycho/config/config.py L94), PyTorch CLI uses `--max_epochs`. PyTorch `TrainingConfig.epochs` default is 50 (config_params.py L107). |
| `--n_images` | L388 | int | NO | 512 | `TFTrainingConfig.n_groups` | CLI → bridge override L564 → `n_groups` | **NAMING DIVERGENCE:** PyTorch uses `--n_images` (semantic: diffraction groups); TF CLI uses `--n_groups` (semantic: neighbor groups). TrainingConfig auto-renames `n_images`→`n_groups` in `__post_init__` (ptycho/config/config.py L116-119). |
| `--gridsize` | L390-391 | int | NO | 2 | `TFModelConfig.gridsize` | CLI → `DataConfig.grid_size` (tuple) L489-490 → bridge tuple→int extraction L101-109 → `gridsize` (int) | Validates square grid: `gridsize == DataConfig.grid_size[0]` must hold. Must be power-of-2 for UNet compatibility (unvalidated at CLI, enforced downstream). |
| `--batch_size` | L392 | int | NO | 4 | `TFTrainingConfig.batch_size` + `TrainingConfig.batch_size` | CLI → `TrainingConfig.batch_size` L492 + bridge L188 → `batch_size` | **POWER-OF-2 CONSTRAINT:** Lightning trainer uses this; must be power-of-2 for DDP (undocumented requirement). |
| `--device` | L394 | choice | NO | 'cpu' | `TrainingConfig.device` + `TrainingConfig.n_devices` (derived) | CLI → `TrainingConfig.device` L493 → Lightning Trainer device; also affects `n_devices` computation L493 (if device=='cuda', n_devices computed from CUDA availability) | **FEATURE GAP:** TensorFlow lacks explicit device CLI control; relies on environment (`CUDA_VISIBLE_DEVICES`). |
| `--disable_mlflow` | L396 | flag | NO | False | Runtime control (skips MLflow autolog L516-519) | CLI → boolean → conditional MLflow skip L516-519 | **BACKEND-SPECIFIC:** TensorFlow has no equivalent. Used in CI environments to avoid MLflow overhead. |

**New CLI Exit Criteria (L408-421):**
Must provide either modern interface (`--train_data_file`, `--output_dir`) XOR legacy interface (`--ptycho_dir`, `--config`). Raises `ValueError` if both or neither provided.

### 1.2 Legacy Interface
Entry point: `main()` lines 22-323, called from `cli_main()` L422-428

| Flag | Source | Required | Maps To | Notes |
|------|--------|----------|---------|-------|
| `--ptycho_dir` | L400 | YES (legacy mode) | Directory path passed to `main(ptycho_dir=...)` | Must contain `config.json` with training configuration. Mutually exclusive with new interface. |
| `--config` | L402 | YES (legacy mode) | Config override path passed to `main(config_path=...)` | Optional JSON override for parameters in `ptycho_dir/config.json`. |

Legacy interface delegates to `main()` which instantiates singleton configs from JSON and runs `train_full()` orchestration (L66-313). **Deprecated** per Phase E2; maintained for backward compatibility only.

---

## 2. PyTorch Inference CLI (`ptycho_torch/inference.py`)

### 2.1 New Interface (Phase E2.C2)
Entry point: `cli_main()` lines 293-572, mode-switched via sys.argv detection L574-576

| Flag | Source | Type | Required | Default | Target Config Field(s) | Flow Path | Notes / TODO |
|------|--------|------|----------|---------|------------------------|-----------|--------------|
| `--model_path` | L343-347 | str | YES | — | Checkpoint directory path | CLI → validation L394-429 → `find_torch_checkpoint()` → Lightning module load L438-457 | Expects directory containing `checkpoints/last.ckpt`, `wts.pt`, or `model.pt` (searched in that order L413-429). **SEMANTIC DIFFERENCE vs TF:** TF uses direct model file path; PyTorch expects training output directory structure. |
| `--test_data` | L349-353 | str | YES | — | NPZ file path | CLI → NPZ load L462-475 → field validation (requires `diffraction`, `probeGuess`, `xcoords`, `ycoords`) → dtype casting (float32, complex64) L494-495 | Enforces float32/complex64 dtypes per specs/data_contracts.md. |
| `--output_dir` | L355-359 | str | YES | — | PNG output directory | CLI → save paths L265-290 (`reconstructed_amplitude.png`, `reconstructed_phase.png`) | Saves 150 DPI PNG images. Directory auto-created if missing. |
| `--n_images` | L361-365 | int | NO | 32 | Batch limiting | CLI → min(len(coords), n_images) L504 | Limits number of scan positions processed. **DIVERGENCE:** TF inference uses this for gridsize interpretation (scripts/inference/inference.py L82-83); PyTorch uses purely for batching control. |
| `--device` | L367-372 | choice | NO | 'cpu' | Lightning Trainer device | CLI → `Trainer(accelerator=...)` L520-521 | Same semantics as training CLI. |
| `--quiet` | L374 | flag | NO | False | Logging suppression | CLI → six call sites (L383, L392, L400, L427, L434, L523) + `enable_progress_bar=False` L522 | **FEATURE GAP:** TensorFlow inference has no equivalent quiet mode. |

### 2.2 Legacy Interface (MLflow Inference)
Entry point: `load_and_predict()` lines 96-184, triggered when sys.argv does not match new CLI signature L574-600

| Flag | Source | Type | Required | Maps To | Notes |
|------|--------|------|----------|---------|-------|
| `--run_id` | L582 | str | YES | MLflow model lookup | Searches MLflow registry for matching run_id (L125-127). |
| `--infer_dir` | L583 | str | YES | Dataset directory | Directory containing test NPZ files (L149). |
| `--file_index` | L584 | int | NO (default 0) | `InferenceConfig.experiment_number` | Selects which file from `infer_dir` to process (L137). |
| `--config` | L585 | str | NO (default None) | Config override | Optional config file to override loaded model config (L129-133). |

**Deprecation Status:** Legacy interface retained for backward compatibility with MLflow-based workflows but not documented in modern Phase E2 user-facing docs.

---

## 3. Parity Gap Analysis: TensorFlow vs PyTorch CLI

### 3.1 Critical Semantic Mismatches

| Issue | TensorFlow Behavior | PyTorch Behavior | Impact | Recommendation |
|-------|---------------------|------------------|--------|----------------|
| **Epoch parameter naming** | `--epochs` (not exposed; config-only `nepochs` L94 ptycho/config/config.py) | `--max_epochs` (L386 ptycho_torch/train.py default 100) | **INCONSISTENCY:** Users switching between backends encounter different parameter names. PyTorch `TrainingConfig.epochs` default is 50; config bridge renames to `nepochs`. | Harmonize to `--nepochs` across both CLIs or document divergence. |
| **Activation function default** | `ModelConfig.amp_activation = 'sigmoid'` (L79 ptycho/config/config.py) | Hardcoded `'silu'` (L486 ptycho_torch/train.py) | **BEHAVIORAL DIVERGENCE:** Different activation functions produce different reconstruction quality. `silu` ≈ `swish` normalization applied in config_bridge L126-132. | Document or add CLI flag for PyTorch. |
| **Neighbor count** | TF inference hardcodes `K=4` (L222 scripts/inference/inference.py) | PyTorch `DataConfig.K=6` (L25 config_params.py) | **PARITY BREAK:** Grouped data cardinality differs between backends. | Expose `--neighbor_count` CLI flag in both stacks. |
| **Probe size** | Requires config override (ModelConfig.N L75 default 64) | Auto-inferred from `probeGuess.shape[0]` (L468-473 ptycho_torch/train.py) | **USABILITY:** PyTorch reduces config burden but may surprise users expecting explicit control. | Document auto-inference behavior; consider adding `--N` override flag. |
| **n_groups vs n_images** | TF CLI: `--n_groups` (L113-114 scripts/training/train.py) → internally becomes `n_images` (L134) | PyTorch CLI: `--n_images` (L388 ptycho_torch/train.py) → bridge renames to `n_groups` (L564) | **NAMING CONFUSION:** Inconsistent parameter names across stacks. TrainingConfig performs `n_images`→`n_groups` renaming in `__post_init__` (L116-119). | Standardize to `--n_groups` in Phase B/C. |

### 3.2 Feature Gaps (Missing in PyTorch Training CLI)

| Flag Name | TF Source | PyTorch Equivalent | Impact | Proposed Action |
|-----------|-----------|-------------------|--------|-----------------|
| `--n_subsample` | L84-86 scripts/inference/inference.py | None (supported in `DataConfig.n_subsample` L28 but not exposed) | **WORKFLOW LIMITATION:** Cannot independently control subsampling from CLI. | Add to Phase B/C CLI flags. |
| `--subsample_seed` | L87-88 scripts/inference/inference.py | None | **REPRODUCIBILITY GAP:** Cannot reproduce specific subsample runs from CLI. | Add to Phase B execution config. |
| `--sequential_sampling` | TrainingConfig L109 ptycho/config/config.py (default False) | None in training config or CLI | **DETERMINISM:** Cannot force sequential sampling for debugging. | Add to Phase B execution config. |
| `--phase_vmin` / `--phase_vmax` | L89-92 scripts/inference/inference.py | None (hardcoded plot defaults) | **VISUALIZATION CONTROL:** Cannot adjust phase color scale. | Lower priority; consider adding to inference CLI. |
| `--comparison_plot` | L79 scripts/inference/inference.py (flag required to enable) | None (comparison plots generated by default or via different code path in `load_and_predict`) | **BEHAVIORAL DIFFERENCE:** TF requires explicit flag; PyTorch auto-generates or uses legacy logic. | Document divergence. |
| `--visualize_probe` | Mentioned in TF legacy docstring (L20) but not modern path | None | **NOT IMPLEMENTED:** Both stacks lack modern probe visualization. | Out of scope. |

### 3.3 Feature Gaps (Missing in TensorFlow CLI)

| Flag Name | PyTorch Source | TF Equivalent | Impact | Notes |
|-----------|----------------|---------------|--------|-------|
| `--device` | L394 ptycho_torch/train.py + L367-372 inference.py | None (relies on environment `CUDA_VISIBLE_DEVICES`) | **USABILITY:** PyTorch allows explicit device selection; TF requires environment manipulation. | Consider backporting to TF. |
| `--gridsize` | L390-391 ptycho_torch/train.py | None (config-only ModelConfig.gridsize L76) | **CONVENIENCE:** PyTorch exposes commonly-changed parameter at CLI; TF requires YAML config. | Potential TF enhancement. |
| `--batch_size` | L392 ptycho_torch/train.py | None (config-only TrainingConfig.batch_size L93) | Same as gridsize | Same as gridsize |
| `--disable_mlflow` | L396 ptycho_torch/train.py | None | **CI OPTIMIZATION:** PyTorch allows disabling MLflow for faster CI runs. | TF uses different experiment tracking approach; not applicable. |
| `--quiet` | L374 ptycho_torch/inference.py | None | **LOGGING CONTROL:** PyTorch provides fine-grained output suppression. | Consider adding to TF. |
| Legacy `--ptycho_dir` / `--config` | L400-402 ptycho_torch/train.py | None (modern interface only) | **BACKWARD COMPATIBILITY:** PyTorch retains legacy JSON-based workflow; TF uses modern YAML-only. | Reflects different migration timelines. |

### 3.4 Naming Divergences Summary

| Concept | TensorFlow Terminology | PyTorch Terminology | Canonical Target (Phase B) |
|---------|------------------------|---------------------|----------------------------|
| Training duration | `nepochs` (config) | `epochs` (config) / `max_epochs` (CLI) | `nepochs` (spec alignment) |
| Data grouping parameter | `n_groups` (CLI/config) | `n_images` (CLI) → `n_groups` (config) | `n_groups` |
| Neighbor lookup | `neighbor_count` (config) | `K` (config) | `neighbor_count` (semantic clarity) |
| Activation function | `amp_activation` (config) | `amp_activation` (config, normalized) | `amp_activation` (with normalization) |
| Output directory | `output_dir` → `output_prefix` (KEY_MAPPINGS) | `output_dir` → `output_prefix` (KEY_MAPPINGS) | `output_dir` (modern) |

---

## 4. Implementation Status Notes

### 4.1 Currently Ignored/Stubbed Flags
None identified. All documented CLI flags are actively used in their respective workflows.

### 4.2 Hardcoded Values (Not CLI-Exposed)

**PyTorch Training (`ptycho_torch/train.py`):**
- **`nphotons`**: Default `1e5` (PyTorch `DataConfig` L20) overridden to `1e9` (TF default) in bridge L559 → config_params.py L20 vs ptycho/config/config.py default.
- **`mode`**: Hardcoded `'Unsupervised'` (L485) → no supervised option at CLI.
- **`learning_rate`**: Hardcoded `1e-3` (TrainingConfig L106) → no CLI override.
- **`fine_tune_gamma`**: Code exists (TrainingConfig L110) but `epochs_fine_tune` defaulted to 0 (L109) → fine-tuning disabled by default, no CLI flag.
- **`amp_activation`**: Hardcoded `'silu'` (L486) → no CLI selection.

**PyTorch Inference (`ptycho_torch/inference.py`):**
- **Dummy `positions` / `intensity_scale`**: L520-525 construct placeholders → not production-ready for multi-probe workflows (acceptable for Phase E2.C2 scope per docs/workflows/pytorch.md).
- **Phase averaging**: L544 uses `torch.mean()` → may create discontinuities for batched phase (consider circular mean in future phases).

### 4.3 Validation/Error Conditions

**Training CLI (`ptycho_torch/train.py`):**
- Path existence checks: L431-442 (`train_data_file`, `test_data_file`)
- Output directory writability: L443-453
- Mutual exclusion: L408-421 (new vs legacy interface)
- Square grid validation: L489-490 (implicit; assumes `DataConfig.grid_size[0] == grid_size[1]`)

**Inference CLI (`ptycho_torch/inference.py`):**
- NPZ field validation: L462-475 (requires `diffraction`, `probeGuess`, `xcoords`, `ycoords`)
- Dtype enforcement: L494-495 (float32/complex64 per specs/data_contracts.md)
- Checkpoint discovery: L413-429 (searches 3 candidates; raises FileNotFoundError if none found)
- torch/lightning import checks: L359-393 (raises RuntimeError per POLICY-001 if unavailable)

---

## 5. References

**Specification Sources:**
- `specs/ptychodus_api_spec.md` §4 (reconstructor lifecycle contract)
- `specs/data_contracts.md` §1 (NPZ format + dtype requirements)
- `docs/workflows/pytorch.md` §§5–12 (PyTorch workflow guidance + backend selection)

**Implementation Sources:**
- PyTorch Training: `ptycho_torch/train.py` (lines 366-520 new CLI, 22-323 legacy)
- PyTorch Inference: `ptycho_torch/inference.py` (lines 293-572 new CLI, 96-184 legacy MLflow)
- TensorFlow Training: `scripts/training/train.py` (lines 100-180)
- TensorFlow Inference: `scripts/inference/inference.py` (lines 60-230)
- Config Bridge: `ptycho_torch/config_bridge.py` (translation logic + field mappings)
- PyTorch Configs: `ptycho_torch/config_params.py` (singleton dataclasses)
- TensorFlow Configs: `ptycho/config/config.py` (canonical dataclasses + KEY_MAPPINGS)

**Test Coverage:**
- `tests/torch/test_config_bridge.py` (enforces field translations + parity)
- `tests/torch/test_integration_workflow_torch.py` (end-to-end CLI validation)

**Planning Artifacts:**
- `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` (CLI wiring decisions)
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/plan.md` (this inventory charter)

---

**Next Steps (Phase B):**
1. Design factory functions to centralize canonical config construction (eliminate hardcoded overrides scattered in train.py L485-559).
2. Define `PyTorchExecutionConfig` dataclass for backend-specific knobs (`device`, `strategy`, `disable_mlflow`, `quiet`).
3. Add missing CLI flags (`--n_subsample`, `--subsample_seed`, `--sequential_sampling`) per parity gap analysis.
4. Harmonize parameter naming (`--max_epochs` → `--nepochs`, `--n_images` → `--n_groups`).
5. Update CLI help text and `docs/workflows/pytorch.md` to document new factory-based architecture.
