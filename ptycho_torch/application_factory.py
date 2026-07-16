"""Application composition from the sealed Torch structural model identity."""

from __future__ import annotations

from collections.abc import Mapping

from ptycho_torch.config_bridge import to_model_config
from ptycho_torch.config_params import DataConfig, InferenceConfig, TrainingConfig
from ptycho_torch.model_spec import ModelSpec, derive_model_spec
from ptycho_torch.scaling_contract import validate_scale_contract


_APPLICATION_CONFIG_KEYS = frozenset(
    {
        "data_config",
        "model_config",
        "training_config",
        "inference_config",
    }
)
_COMPATIBILITY_CONFIG_KEYS = frozenset({"execution_config"})


def build_ptychopinn_from_configs(pt_configs: Mapping[str, object]):
    """Build through the single canonical-to-structural application route.

    ``execution_config`` is accepted only as a compatibility key for existing
    registry callers. Runtime orchestration cannot affect model identity.
    """
    if not isinstance(pt_configs, Mapping):
        raise TypeError("pt_configs must be a mapping")
    received = set(pt_configs)
    missing = _APPLICATION_CONFIG_KEYS - received
    unknown = received - _APPLICATION_CONFIG_KEYS - _COMPATIBILITY_CONFIG_KEYS
    if missing or unknown:
        raise ValueError(
            "application config keys are not exact; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    data_config = pt_configs["data_config"]
    model_config = pt_configs["model_config"]
    training_config = pt_configs["training_config"]
    inference_config = pt_configs["inference_config"]
    canonical_model = to_model_config(data_config, model_config)
    model_spec = derive_model_spec(canonical_model, model_config, data_config)
    return build_ptychopinn_application(
        model_spec,
        data_config,
        training_config,
        inference_config,
    )


def build_ptychopinn_application(
    model_spec: ModelSpec,
    data_config: DataConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
):
    """Compose model, scientific/data, training, and inference sections.

    Runtime execution remains with Trainer orchestration and cannot affect the
    module graph or its state-dict identity.
    """
    if not isinstance(model_spec, ModelSpec):
        raise TypeError("model_spec must be a ModelSpec")
    model_config = model_spec.to_model_config()
    if model_config.C_model != data_config.C or model_config.C_forward != data_config.C:
        raise ValueError(
            "ModelSpec channel joins conflict with data_config.C: "
            f"C_model={model_config.C_model}, C_forward={model_config.C_forward}, "
            f"data C={data_config.C}"
        )

    desired_loss = "Poisson" if training_config.torch_loss_mode == "poisson" else "MAE"
    if model_config.mode == "Supervised":
        if training_config.torch_loss_mode != "mae" or model_config.loss_function != "MAE":
            raise ValueError(
                "training torch_loss_mode and model loss_function conflict: "
                "Supervised construction requires torch_loss_mode='mae' and "
                "model loss_function='MAE'"
            )
    elif model_config.loss_function != desired_loss:
        raise ValueError(
            "training torch_loss_mode and model loss_function conflict: "
            f"torch_loss_mode={training_config.torch_loss_mode!r} requires "
            f"model loss_function={desired_loss!r}, got {model_config.loss_function!r}"
        )

    validate_scale_contract(data_config, model_config, training_config)

    from ptycho_torch.model import PtychoPINN_Lightning

    constructor_kwargs = dict(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        inference_config=inference_config,
        model_spec=model_spec.to_payload(),
    )
    if (
        model_spec.parity_scale_mode != "off"
        or model_spec.parity_fixed_delta != 0.0
        or model_spec.parity_init_scheme != "default"
    ):
        constructor_kwargs.update(
            parity_scale_mode=model_spec.parity_scale_mode,
            parity_fixed_delta=model_spec.parity_fixed_delta,
            parity_init_scheme=model_spec.parity_init_scheme,
        )
    module = PtychoPINN_Lightning(**constructor_kwargs)
    return module
