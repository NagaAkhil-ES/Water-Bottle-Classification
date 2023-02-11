import toml

from utils.dotdict import DotDict


def load_config(config_path):
    params = toml.load(config_path)
    params = DotDict(params)
    return params

def save_config(params, save_path):
    with open(save_path, "w") as f:
        toml.dump(params, f)