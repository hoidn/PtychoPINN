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
    'probe_dir_bool': True,
}

model_config_default = {
    'intensity_scale_trainable': True,
    'max_position_jitter': 10, #Random jitter for translation (helps make model more robust)
    'n_filters_scale': 2, #Shrinking factor for channels
    'intensity_scale': 15000.0, #General intensity scale guess, this can be trainable. Needs to be float
    'object.big': True, #True if need patch reassembly
    'probe.big': True, #True if need patch reassembly
    'offset': 4
}

training_config_default = {
    'nll': True #Negative log likelihood for loss function
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
# class Config:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
#             cls._instance.settings = {}
        
#         return cls._instance
    
#     def set_settings(self, settings):
#         self.settings = settings

#     def get(self, key, default_val = None):
#         if key not in self.settings:
#             return default_val
#         else:
#             return self.settings.get(key)
#     def add(self, key, value):
#         self.settings[key] = value

# class Params:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super(Params, cls).__new__(cls, *args, **kwargs)
#             cls._instance.settings = {}
        
#         return cls._instance
    
#     def set_settings(self, settings):
#         self.settings = settings

#     def get(self, key, default_val = None):
#         if key not in self.settings:
#             return default_val
#         else:
#             return self.settings.get(key)
#     def add(self, key, value):
#         self.settings[key] = value