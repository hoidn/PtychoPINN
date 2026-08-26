"""PyTorch workflow orchestration facade — parity with ptycho/workflows/components.py.

This module is a spec-pinned re-exporting facade.  The implementation slabs
behind it were split by responsibility into sibling submodules
(``bundle_io``, ``containers``, ``dataloaders``, ``rect_s1s2``,
``lightning_service``, ``legacy``); every name below is re-exported unchanged so
``from ptycho_torch.workflows.components import ...`` and monkeypatch targets
resolve exactly as before the split.

Lifecycle (authored -> resolved payload -> sealed identity -> restored identity):

    authored config (TrainingConfig / InferenceConfig)
      -> resolved payload (TrainingPayload / InferencePayload)   [config_factory]
      -> sealed identity (ModelSpec)                             [application_factory]
      -> restored identity (strict bundle/checkpoint decode)     [bundle_io / checkpoint_decode]

The resolved payload is the single configuration currency at the four
consumption points: ``_train_with_lightning`` (training service), loader
construction, ``PtychoPINN_Lightning.__init__`` (module construction), and the
inference kernel (decoded bundle identity + explicit runtime argument).
"""

# Re-export the moved responsibility slabs (facade guarantee).
from .bundle_io import (
    _BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD,
    _BUNDLE_SCALING_METADATA,
    _decode_bundle_metadata,
    _decode_pinned_inference_bundle,
    _persist_bundle_scaling_metadata,
    _pinned_bundle_snapshot,
    _read_bundle_amplitude_physics_gain_record,
    _read_bundle_scaling_metadata,
    _reconstruct_inference_bundle_explicit,
    _strictly_reconstruct_bundle_model,
    load_inference_bundle_torch,
)
from .containers import (
    _adapt_container_for_ci,
    _attach_physics_scale,
    _canonicalize_ci_probe_modes,
    _get_container_tensor_required,
    _get_finalized_ci_statistics,
    _resolve_nphotons,
    attach_container_ci_fields,
    create_torch_data_container,
)
from .dataloaders import (
    _build_inference_dataloader,
    _build_lightning_dataloaders,
    _resolve_torch_training_seed,
)
from .rect_s1s2 import (
    _RECT_S1S2_IDENTITY_FIELD,
    _RectS1S2IndexedRows,
    _RectS1S2MaintainedCollation,
    _RectS1S2SelectedBatch,
    _RectS1S2SelectedDataset,
    _effective_dataloader_settings,
    _initialize_rect_s1s2,
    _initialize_rect_s1s2_unmanaged,
    _inspect_rect_s1s2_channels,
    _move_batch_to_device,
    _publish_training_summary_and_barrier,
    _rebuild_rect_s1s2_loader,
    _rect_s1s2_attach_identities,
    _rect_s1s2_batch_axes,
    _rect_s1s2_indexable_dataset,
    _rect_s1s2_training_loader,
    _rect_s1s2_verify_collated_identities,
    _write_training_summary_atomic,
)
from .lightning_service import (
    _CHECKPOINT_SELECTION_SCHEMA,
    _FinalModelSelectionCallback,
    _LossHistoryCallback,
    _MilestoneCheckpointCallback,
    _ServingModelCheckpoint,
    _ServingModelCheckpointMixin,
    _TrainingSummaryCallback,
    _checkpoint_artifact_path,
    _checkpoint_file_sha256,
    _checkpoint_score_value,
    _in_memory_checkpoint_selection,
    _publish_checkpoint_selection_and_barrier,
    _rank_shared_checkpoint_selection_token,
    _read_checkpoint_selection,
    _resolve_checkpoint_monitor,
    _train_with_lightning,
    _validate_training_execution_input,
    _write_checkpoint_selection_atomic,
)
from .legacy import (
    _reassemble_cdi_image_torch,
    _reassemble_cdi_image_torch_mmap,
    run_cdi_example_torch,
    train_cdi_model_torch,
)

__all__ = [
    '_BUNDLE_AMPLITUDE_PHYSICS_GAIN_RECORD', '_BUNDLE_SCALING_METADATA', '_decode_bundle_metadata', '_decode_pinned_inference_bundle', '_persist_bundle_scaling_metadata', '_pinned_bundle_snapshot', '_read_bundle_amplitude_physics_gain_record', '_read_bundle_scaling_metadata',
    '_reconstruct_inference_bundle_explicit', '_strictly_reconstruct_bundle_model', 'load_inference_bundle_torch', '_adapt_container_for_ci', '_attach_physics_scale', '_canonicalize_ci_probe_modes', '_get_container_tensor_required', '_get_finalized_ci_statistics',
    '_resolve_nphotons', 'attach_container_ci_fields', 'create_torch_data_container', '_build_inference_dataloader', '_build_lightning_dataloaders', '_resolve_torch_training_seed', '_RECT_S1S2_IDENTITY_FIELD', '_RectS1S2IndexedRows',
    '_RectS1S2MaintainedCollation', '_RectS1S2SelectedBatch', '_RectS1S2SelectedDataset', '_effective_dataloader_settings', '_initialize_rect_s1s2', '_initialize_rect_s1s2_unmanaged', '_inspect_rect_s1s2_channels', '_move_batch_to_device',
    '_publish_training_summary_and_barrier', '_rebuild_rect_s1s2_loader', '_rect_s1s2_attach_identities', '_rect_s1s2_batch_axes', '_rect_s1s2_indexable_dataset', '_rect_s1s2_training_loader', '_rect_s1s2_verify_collated_identities', '_write_training_summary_atomic',
    '_CHECKPOINT_SELECTION_SCHEMA', '_FinalModelSelectionCallback', '_LossHistoryCallback', '_MilestoneCheckpointCallback', '_ServingModelCheckpoint', '_ServingModelCheckpointMixin', '_TrainingSummaryCallback', '_checkpoint_artifact_path',
    '_checkpoint_file_sha256', '_checkpoint_score_value', '_in_memory_checkpoint_selection', '_publish_checkpoint_selection_and_barrier', '_rank_shared_checkpoint_selection_token', '_read_checkpoint_selection', '_resolve_checkpoint_monitor', '_train_with_lightning',
    '_validate_training_execution_input', '_write_checkpoint_selection_atomic', '_reassemble_cdi_image_torch', '_reassemble_cdi_image_torch_mmap', 'run_cdi_example_torch', 'train_cdi_model_torch',
]
