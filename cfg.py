import yaml
from easydict import EasyDict


def parse_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    
    return cfg
