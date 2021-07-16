import yaml
from easydict import EasyDict


def parse_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    
    cfg.lr = float(cfg.lr)
    cfg.wd = float(cfg.wd)
    cfg.grad_clip = float(cfg.grad_clip)
    cfg.fgm = float(cfg.fgm)
    
    return cfg
