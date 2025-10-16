"""
Singleton-based configuration layer for the PyTorch implementation.

The TensorFlow code path uses typed dataclasses (`ptycho.config`) to pass settings into models.
For the PyTorch stack we instead expose three singletons — `ModelConfig`, `TrainingConfig`, and
`DataConfig` — that can be populated once per process and queried anywhere in the module graph.
This mirrors the legacy `ptycho.params.cfg` behaviour while remaining explicit about the expected
keys.

Usage pattern
-------------
    ```python
    from ptycho_torch.config_params import ModelConfig, model_config_default

    model_cfg = ModelConfig()
    model_cfg.set_settings(model_config_default)
    model_cfg.add("loss_function", "Poisson")
    ```
Access is read-only thereafter: calling `.get("loss_function")` in a downstream module returns the
same value until `.set_settings` is invoked again.

Integration points
------------------
- `ptycho_torch/model.py` reads runtime decisions (e.g., `object.big`, `n_filters_scale`, `offset`).
- `ptycho_torch/dset_loader_pt_mmap.py` inspects `DataConfig` to decide how to build memory-mapped
  tensors.
- `ptycho_torch/train.py` seeds all three singletons before constructing datasets or models.

Caveats
-------
- The singleton instances are process-global; re-seeding them midway through training is not safe.
- Keys are not validated. Keep the canonical defaults (`*_config_default`) synchronized with the
  data contracts in <doc-ref type="contract">specs/data_contracts.md</doc-ref> and update both when new
  parameters are introduced.
- When porting features from the TensorFlow path, map dataclass fields to these settings explicitly
  and document the mapping in the module-level docstrings.
"""

#Configuration singleton class

#Typical config values (need to be declared separately in __main__ or jupyter notebook)

#The way you use this is:
# 1. Defining a settings dictionary
# 2. Declare specific class: ModelConfig(), TrainingConfig(), DataConfig()
# 3. Call set_config or set_params methods using dictionary
# 4. Now the respective class can be accessed globally

#Configuration file for reference (this is not used within this file but used when configs are
#declared in other files)

data_config_default = {
    'nphotons': 1e5,
    'N': 128,
    'C': 4,
    'K': 6,
    'n_subsample': 10,
    'grid_size': (2,2),
    'probe_dir_get': True,
    'normalize': True
}

model_config_default = {
    'intensity_scale_trainable': False,
    'intensity_scale': 10000.0,
    'max_position_jitter': 10, #Random jitter for translation (helps make model more robust)
    'n_filters_scale': 2, #Shrinking factor for channels
    'intensity_scale': 15000.0, #General intensity scale guess, this can be trainable. Needs to be float
    'object.big': False, #True if need patch reassembly
    'probe.big': True, #True if need patch reassembly
    'offset': 4,
    'loss_function': 'MAE'
}

training_config_default = {
    'nll': True, #Negative log likelihood for loss function
    'device': 'cuda',
    'strategy': 'ddp'
}

class Settings:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Settings, cls).__new__(cls, *args, **kwargs)
            cls._instance.settings = {}
        
        return cls._instance
    
    def set_settings(self, settings):
        self.settings = settings

    def get(self, key, default_val = None):
        if key not in self.settings:
            return default_val
        else:
            return self.settings.get(key)
    def add(self, key, value):
        self.settings[key] = value


class TrainingConfig(Settings):
    _instance = None

class ModelConfig(Settings):
    _instance = None

class DataConfig(Settings):
    _instance = None

#Creating dataclasses 
