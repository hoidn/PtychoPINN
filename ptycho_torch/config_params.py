#Configuration singleton class

#Typical config values (need to be declared separately in __main__ or jupyter notebook):

#Configuration file
cfg = {'N': 64,
       'offset': 4,
       'gridsize': 2,
       'max_position_jitter': 10
    }

params = {

    
}

class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
            cls._instance.config = {}
        
        return cls._instance
    
    def set_config(self, config):
        self.config = config

    def get(self, key):
        return self.config.get(key)
    
class Params:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Params, cls).__new__(cls, *args, **kwargs)
            cls._instance.params = {}
        
        return cls._instance
    
    def set_params(self, params):
        self.params = params

    def get(self, key):
        return self.params.get(key)