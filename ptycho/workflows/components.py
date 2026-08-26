"""High-level TensorFlow workflow orchestration facade.

Spec-pinned re-exporting facade.  The implementation slabs behind it were split
by responsibility into sibling submodules (``bundle_loading``, ``config_cli``,
``workflow_orchestration``); every name below is re-exported unchanged so
``from ptycho.workflows.components import ...`` and monkeypatch targets resolve
exactly as before the split.

Lifecycle (authored -> resolved payload -> sealed identity -> restored identity):

    authored config (TrainingConfig)
      -> resolved payload (TrainingPayload)                        [config_factory]
      -> sealed identity (ModelSpec)                               [application_factory]
      -> restored identity (strict bundle/checkpoint decode)       [bundle_loading]

The TensorFlow door delegates to the same orchestration entry points and return
structure as the PyTorch twin facade (specs/ptychodus_api_spec.md §4.8).
"""

# Re-export the moved responsibility slabs (facade guarantee).
from .bundle_loading import (
    DiffractionToObjectAdapter,
    load_inference_bundle,
    load_inference_bundle_explicit,
)
from .config_cli import (
    PUBLIC_TRAINING_INPUT_NAMES,
    _add_public_training_argument,
    _literal_argument_type,
    _public_training_argument_help,
    _unwrap_public_cli_type,
    add_public_training_config_arguments,
    load_data,
    load_yaml_config,
    parse_arguments,
    setup_configuration,
    update_config_from_dict,
)
from .workflow_orchestration import (
    _resolve_tensorflow_training_config,
    create_ptycho_data_container,
    reassemble_cdi_image,
    run_cdi_example,
    save_outputs,
    train_cdi_model,
)

__all__ = [
    'DiffractionToObjectAdapter',
    'load_inference_bundle',
    'load_inference_bundle_explicit',
    'PUBLIC_TRAINING_INPUT_NAMES',
    '_add_public_training_argument',
    '_literal_argument_type',
    '_public_training_argument_help',
    '_unwrap_public_cli_type',
    'add_public_training_config_arguments',
    'load_data',
    'load_yaml_config',
    'parse_arguments',
    'setup_configuration',
    'update_config_from_dict',
    '_resolve_tensorflow_training_config',
    'create_ptycho_data_container',
    'reassemble_cdi_image',
    'run_cdi_example',
    'save_outputs',
    'train_cdi_model',
]
