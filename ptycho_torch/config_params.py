#Configuration singleton class

#Typical config values (need to be declared separately in __main__ or jupyter notebook)

#The way you use this is:
# 1. Defining a cfg or param dictionary
# 2. Declare Config() or Params() class
# 3. Call set_config or set_params methods using dictionary
# 4. Now Config() class or Params() class can be accessed globally

#Configuration file for reference (this is not used within this file but used when configs are
#declared in other files)
cfg = {'N': 64,
       'offset': 4,
       'gridsize': 2,
       'max_position_jitter': 10,
       'n_filters_scale': 2
    }

params = {


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

class Config(Settings):
    _instance = None

class Params(Settings):
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