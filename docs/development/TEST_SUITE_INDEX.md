# PtychoPINN Test Suite Index

This document provides a comprehensive index of the automated tests in the `tests/` directory. Its purpose is to make the test suite discoverable, explain the scope of each test module, and provide direct commands for running specific tests.

## How to Run Tests

- **Run all tests:** `python -m unittest discover tests/`
- **Run a specific file:** `python -m unittest tests.image.test_cropping`
- **Run a specific test class:** `python -m unittest tests.test_integration_workflow.TestFullWorkflow`

---

## Test Modules

### Core Library Tests (`tests/`)

| Test File | Purpose / Scope | Key Tests | Usage / Command | Notes |
| :--- | :--- | :--- | :--- | :--- |

| `test_baselines.py` | No module docstring found. | `test_build_model_always_creates_single_channel_output` | `python -m unittest tests.test_baselines` | — |
| `test_benchmark_throughput.py` | Unit and Integration Tests for Inference Throughput Benchmarking  This module tests the benchmarking infrastructure for measuring and optimizing PtychoPINN inference throughput. | `test_calculate_efficiency`, `test_calculate_latency`, `test_calculate_throughput`, `test_clear_timings`, `test_config_custom_values`, `test_config_initialization`, `test_export_results`, `test_find_optimal_batch_size_no_success`, `test_find_optimal_batch_size`, `test_func`, `test_generate_summary`, `test_get_statistics`, `test_load_model_and_data`, `test_measure_function`, `test_profile_cpu_memory`, `test_snapshot`, `test_timing_profiler_performance`, `test_warmup_inference` | `python -m unittest tests.test_benchmark_throughput` | — |
| `test_cli_args.py` | Unit tests for the ptycho.cli_args module. | `test_add_logging_arguments`, `test_console_level_choices`, `test_get_logging_config_custom_level`, `test_get_logging_config_defaults`, `test_get_logging_config_quiet`, `test_get_logging_config_verbose`, `test_quiet_verbose_mutually_exclusive` | `python -m unittest tests.test_cli_args` | — |
| `test_coordinate_grouping.py` | Comprehensive test suite for efficient coordinate grouping implementation.  This module tests the new "sample-then-group" strategy for coordinate grouping in the RawData class, ensuring correctness, performance, and edge case handling. | `test_backward_compatibility`, `test_different_seeds_produce_different_results`, `test_edge_case_k_less_than_c`, `test_edge_case_more_samples_than_points`, `test_edge_case_small_dataset`, `test_efficient_grouping_output_shape`, `test_efficient_grouping_spatial_coherence`, `test_efficient_grouping_valid_indices`, `test_existing_tests_still_pass`, `test_generate_grouped_data_gridsize_1`, `test_generate_grouped_data_integration`, `test_memory_efficiency`, `test_no_cache_files_created`, `test_performance_improvement`, `test_reproducibility_with_seed` | `python -m unittest tests.test_coordinate_grouping` | — |
| `test_generic_loader.py` | No module docstring found. | `test_generic_loader_roundtrip`, `test_generic_loader` | `python -m unittest tests.test_generic_loader` | — |
| `test_integration_baseline_gs2.py` | No module docstring found. | `test_baseline_gridsize2_end_to_end` | `python -m unittest tests.test_integration_baseline_gs2` | — |
| `test_integration_workflow.py` | Validates the full train → save → load → infer workflow using subprocesses, ensuring model artifacts persist across CLI entrypoints. | `test_train_save_load_infer_cycle` | `python -m unittest tests.test_integration_workflow` | Critical integration coverage for TensorFlow persistence. |
| `test_log_config.py` | Unit tests for the ptycho.log_config module. | `test_backward_compatibility`, `test_conflicting_flags_verbose_overrides`, `test_custom_console_level`, `test_default_setup_logging_creates_log_directory_and_file`, `test_quiet_flag_overrides_console_level`, `test_quiet_mode_disables_console`, `test_setup_logging_clears_existing_handlers`, `test_string_path_support`, `test_verbose_mode_enables_debug_console` | `python -m unittest tests.test_log_config` | — |
| `test_misc.py` | No module docstring found. | `test_memoize_simulated_data` | `python -m unittest tests.test_misc` | — |
| `test_nphotons_metadata_integration.py` | Comprehensive integration test for nphotons metadata system.  This test verifies the complete nphotons metadata workflow: 1. Simulation with different nphotons values saves metadata correctly 2. Training loads metadata and validates configurations  3. Inference uses correct nphotons from metadata 4. Parameter mismatch warnings work correctly 5. End-to-end workflow maintains nphotons consistency  The test follows the project's integration test patterns using subprocess calls to simulate real user workflows across separate processes. | `test_configuration_mismatch_warnings`, `test_end_to_end_workflow_consistency`, `test_metadata_backward_compatibility`, `test_metadata_persistence_single_nphotons`, `test_multiple_nphotons_metadata_consistency`, `test_training_with_mismatched_config_warns_but_continues` | `python -m unittest tests.test_nphotons_metadata_integration` | — |
| `test_oversampling.py` | Test automatic K choose C oversampling functionality. | `test_automatic_oversampling_triggers`, `test_gridsize_1_no_oversampling`, `test_oversampling_with_different_k_values`, `test_reproducibility_with_seed`, `test_standard_sampling_no_oversampling` | `python -m unittest tests.test_oversampling` | — |
| `test_projective_warp_xla.py` | Test cases for projective_warp_xla module. | `test_batch_processing`, `test_complex128_dtype`, `test_complex64_dtype`, `test_fill_modes`, `test_float32_dtype`, `test_float64_dtype`, `test_float64_with_translation`, `test_interpolation_modes`, `test_jit_compilation_float32`, `test_jit_compilation_float64`, `test_mixed_precision_translation`, `test_tfa_params_conversion` | `python -m unittest tests.test_projective_warp_xla` | — |
| `test_pytorch_tf_wrapper.py` | No module docstring found. | N/A | `python -m unittest tests.test_pytorch_tf_wrapper` | — |
| `test_raw_data_grouping.py` | Exhaustively checks the RawData sample-then-group implementation, covering shapes, bounds, locality, and oversized sampling requests. | `test_content_validity`, `test_edge_case_k_less_than_c`, `test_edge_case_more_samples_than_points`, `test_edge_case_small_dataset`, `test_memory_efficiency`, `test_output_shape`, `test_performance_improvement`, `test_reproducibility`, `test_uniform_sampling` | `python -m unittest tests.test_raw_data_grouping` | — |
| `test_run_baseline.py` | No module docstring found. | `test_prepare_baseline_data_flattens_gridsize2_data`, `test_prepare_baseline_data_passes_through_gridsize1` | `python -m unittest tests.test_run_baseline` | — |
| `test_scaling_regression.py` | Regression tests for scaling bugs in diffsim.py | `test_assertions_catch_invalid_intensity_scale`, `test_both_arrays_scaled_identically`, `test_different_nphotons_produce_proportional_scaling`, `test_intensity_scale_is_valid`, `test_phase_is_not_scaled`, `test_scaling_is_reversible`, `test_scaling_preserves_physics` | `python -m unittest tests.test_scaling_regression` | — |
| `test_sequential_sampling.py` | Test suite for sequential sampling functionality in coordinate grouping.  This module tests the sequential_sampling flag added to the RawData class to restore deterministic, sequential data subset selection capability. | `test_cli_argument_parsing`, `test_config_flag_exists`, `test_default_behavior_is_random`, `test_sequential_sampling_handles_edge_cases`, `test_sequential_sampling_is_deterministic`, `test_sequential_sampling_order`, `test_sequential_sampling_uses_first_n_points`, `test_sequential_sampling_with_gridsize_greater_than_1`, `test_sequential_sampling_with_seed_parameter`, `test_sequential_vs_random_coverage` | `python -m unittest tests.test_sequential_sampling` | — |
| `test_subsampling.py` | Unit tests for independent data subsampling functionality.  This module tests the new n_subsample parameter that enables independent control of data subsampling and neighbor grouping operations in PtychoPINN. | `test_different_seeds_produce_different_results`, `test_interaction_with_config_dataclass`, `test_legacy_n_images_behavior`, `test_n_subsample_overrides_n_images`, `test_no_subsample_uses_full_dataset`, `test_reproducible_subsampling_with_seed`, `test_sorted_indices_for_consistency`, `test_subsample_larger_than_dataset`, `test_subsample_with_n_subsample`, `test_subsample_zero_edge_case`, `test_y_patches_subsampled_consistently` | `python -m unittest tests.test_subsampling` | — |
| `test_tf_helper.py` | Unit tests for ptycho/tf_helper.py, focusing on patch reassembly logic.  This test suite validates the core functionality of the reassemble_position function by testing its fundamental properties without making assumptions about exact output sizes. | `test_basic_functionality`, `test_batch_translation`, `test_complex_tensor_translation`, `test_different_patch_values_blend`, `test_edge_cases`, `test_identical_patches_single_vs_double`, `test_integer_translation`, `test_perfect_overlap_averages_to_identity`, `test_subpixel_translation`, `test_translate_core_matches_addons`, `test_zero_translation` | `python -m unittest tests.test_tf_helper` | — |
| `test_tf_helper_edge_aware.py` | Edge-aware tests for translate functions that account for interpolation differences.  This test suite specifically handles the known differences between TFA and TF raw ops regarding edge interpolation while ensuring smooth patterns (as used in PtychoPINN) work correctly. | `test_batch_smooth_patterns`, `test_boundary_behavior`, `test_complex_smooth_translation`, `test_document_edge_differences`, `test_gaussian_probe_translation`, `test_smooth_object_translation`, `test_typical_probe_sizes` | `python -m unittest tests.test_tf_helper_edge_aware` | — |
| `test_utilities.py` | Test utilities for PtychoPINN tests. | N/A | `python -m unittest tests.test_utilities` | — |
| `test_workflow_components.py` | Unit tests for ptycho.workflows.components module. | `test_exception_propagation`, `test_load_valid_model_directory`, `test_missing_diffraction_model`, `test_missing_model_archive`, `test_nonexistent_directory`, `test_not_a_directory`, `test_path_conversion` | `python -m unittest tests.test_workflow_components` | — |

### Image Tests (`tests/image/`)

| Test File | Purpose / Scope | Key Tests | Usage / Command | Notes |
| :--- | :--- | :--- | :--- | :--- |

| `test_cropping.py` | Tests for the cropping module, focusing on the align_for_evaluation function. | `test_align_for_evaluation_bounding_box`, `test_align_for_evaluation_coordinates_format`, `test_align_for_evaluation_shapes`, `test_align_for_evaluation_with_squeeze`, `test_center_crop_exact_size` | `python -m unittest tests.image.test_cropping` | — |
| `test_registration.py` | Covers registration alignment helpers, including translation detection, shift application, and complex-valued data handling. | `test_apply_shift_and_crop_basic`, `test_apply_shift_and_crop_zero_offset`, `test_different_image_content`, `test_edge_case_maximum_shift`, `test_edge_case_single_pixel_shift`, `test_find_offset_known_shift_complex`, `test_find_offset_known_shift_real`, `test_input_validation_2d_requirement`, `test_input_validation_excessive_offset`, `test_input_validation_shape_matching`, `test_noise_robustness`, `test_register_and_align_convenience`, `test_registration_sign_verification`, `test_round_trip_registration`, `test_shift_and_crop_preserves_data_type` | `python -m unittest tests.image.test_registration` | — |

### Study Tests (`tests/study/`)

| Test File | Purpose / Scope | Key Tests | Usage / Command | Notes |
| :--- | :--- | :--- | :--- | :--- |

| `test_dose_overlap_design.py` | Tests study design configuration and validation for STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 initiative. Validates study parameters, overlap fractions, and spacing thresholds. | `test_study_design_constants`, `test_study_design_validation`, `test_study_design_to_dict` | `pytest tests/study/test_dose_overlap_design.py -v` | Validates study design dataclass and constants. |
| `test_dose_overlap_dataset_contract.py` | Tests DATA-001 contract enforcement for Phase C/D datasets. Validates dataset structure, dtypes, spacing constraints, and oversampling preconditions. | `test_validate_dataset_contract_happy_path`, `test_validate_dataset_contract_spacing_dense`, `test_validate_dataset_contract_oversampling_precondition_pass` | `pytest tests/study/test_dose_overlap_dataset_contract.py -v` | Enforces canonical NPZ format and study constraints. |
| `test_dose_overlap_generation.py` | Tests Phase C dataset generation pipeline for dose sweep. Validates simulation configuration, dataset orchestration, and output structure. | `test_build_simulation_plan`, `test_generate_dataset_pipeline_orchestration`, `test_generate_dataset_config_construction` | `pytest tests/study/test_dose_overlap_generation.py -v` | Tests dataset generation workflows (dose_1000, dose_10000, dose_100000). |
| `test_dose_overlap_overlap.py` | Tests Phase D overlap view filtering and metrics. Validates spacing matrix computation, acceptance mask generation, and metrics bundle emission. | `test_compute_spacing_matrix_basic`, `test_spacing_filter_parametrized`, `test_generate_overlap_views_metrics_manifest` | `pytest tests/study/test_dose_overlap_overlap.py -v` | Tests overlap filtering pipeline and metrics aggregation. Key selector: `pytest tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_metrics_manifest -vv` for metrics bundle validation. |

### Studies Tests (`tests/studies/`)

| Test File | Purpose / Scope | Key Tests | Usage / Command | Notes |
| :--- | :--- | :--- | :--- | :--- |

| `test_aggregate_nan_msssim.py` | Test script to verify NaN handling and MS-SSIM filtering in aggregate_and_plot_results.py  This script creates mock data directories with controlled test cases: 1. A trial with good MS-SSIM (0.8) 2. A trial with bad MS-SSIM (0.1)  3. A trial with NaN values  Then runs the aggregation script with different thresholds to verify behavior. | N/A | `python -m unittest tests.studies.test_aggregate_nan_msssim` | — |
| `test_filtering_order.py` | Enhanced test script to verify filtering order in aggregate_and_plot_results.py  This script creates mock data with specific MS-SSIM values to test if filtering happens before or after statistical aggregation.  Test scenario: - 4 trials with ms_ssim_phase values: 0.8, 0.7, 0.6, 0.1 - With threshold 0.3, the outlier (0.1) should be filtered out - Correct median after filtering: 0.7 (from [0.8, 0.7, 0.6]) - Incorrect median if filtering after aggregation: 0.65 (from [0.8, 0.7, 0.6, 0.1]) | N/A | `python -m unittest tests.studies.test_filtering_order` | — |
| `test_mean_vs_median.py` | Test script to verify that aggregate_and_plot_results.py now uses mean instead of median  This script creates mock data with specific values to test the statistical aggregation: - 3 trials with values: 10, 20, 30 - Expected mean: 20.0 - Expected 25th percentile: 15.0 - Expected 75th percentile: 25.0 - Expected median (old): 20.0 (same as mean in this case) | N/A | `python -m unittest tests.studies.test_mean_vs_median` | — |

### Tools Tests (`tests/tools/`)

| Test File | Purpose / Scope | Key Tests | Usage / Command | Notes |
| :--- | :--- | :--- | :--- | :--- |

| `test_generate_test_index.py` | Tests for the generate_test_index automation script. | `test_get_module_docstring_handles_missing_docstring`, `test_get_module_docstring_reads_existing_docstring`, `test_get_test_functions_lists_key_tests` | `python -m unittest tests.tools.test_generate_test_index` | — |
| `test_update_tool.py` | Test script for update_tool.py | `test_update_function`, `test_update_function` | `python -m unittest tests.tools.test_update_tool` | — |

### Torch Tests (`tests/torch/`)

| Test File | Purpose / Scope | Key Tests | Usage / Command | Notes |
| :--- | :--- | :--- | :--- | :--- |

| `test_api_deprecation.py` | Test suite for ptycho_torch.api deprecation warnings per ADR-003 Phase E.C1. Validates that legacy API entry points emit DeprecationWarning with migration guidance steering users toward factory-driven workflows documented in docs/workflows/pytorch.md. | `test_example_train_import_emits_deprecation_warning`, `test_api_package_import_is_idempotent` | `pytest tests/torch/test_api_deprecation.py -vv` | Validates legacy API deprecation messaging. Uses native pytest style. |
| `test_tf_helper.py` | No module docstring found. | `test_combine_complex`, `test_get_mask`, `test_placeholder_torch_functions` | `python -m unittest tests.torch.test_tf_helper` | — |

---
*This document can be automatically updated. Run ``python scripts/tools/generate_test_index.py`` to regenerate.*

