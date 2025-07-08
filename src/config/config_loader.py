import yaml

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split('.')
        val = self.cfg
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

# Example usage:
# config = Config('src/config/default.yaml')
# learning_rate = config.get('training.learning_rate')
