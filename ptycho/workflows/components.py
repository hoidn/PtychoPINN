"""High-level TensorFlow workflow orchestration facade.

Spec-pinned re-exporting facade. The implementation slabs behind it were
split by responsibility into sibling submodules (``bundle_loading``,
``config_cli``, ``workflow_orchestration``); every name below is re-exported
unchanged so ``from ptycho.workflows.components import ...`` and monkeypatch
targets resolve exactly as before the split.
"""
from .bundle_loading import (
    DiffractionToObjectAdapter,
    load_inference_bundle,
    load_inference_bundle_explicit,
)

from .config_cli import (
    update_config_from_dict,
    load_data,
    add_public_training_config_arguments,
    parse_arguments,
    load_yaml_config,
    _namespace_to_training_patch,
    setup_configuration,
)

from .workflow_orchestration import (
    _resolve_tensorflow_training_config,
    create_ptycho_data_container,
    train_cdi_model,
    reassemble_cdi_image,
    run_cdi_example,
    save_outputs,
)

__all__ = [
    'DiffractionToObjectAdapter',
    '_namespace_to_training_patch',
    '_resolve_tensorflow_training_config',
    'add_public_training_config_arguments',
    'create_ptycho_data_container',
    'load_data',
    'load_inference_bundle',
    'load_inference_bundle_explicit',
    'load_yaml_config',
    'parse_arguments',
    'reassemble_cdi_image',
    'run_cdi_example',
    'save_outputs',
    'setup_configuration',
    'train_cdi_model',
    'update_config_from_dict',
]
