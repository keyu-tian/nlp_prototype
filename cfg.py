import yaml
from easydict import EasyDict


def dfs(item):
    if isinstance(item, dict):
        for k, v in item.items():
            if isinstance(v, list):
                return item, k, v
            if isinstance(v, dict):
                return dfs(v)
    return None, None, None


def parse_cfg(cfg_path, world_size, rank, only_model=False):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    if only_model:
        return cfg.model
    
    dist_d, dist_k, dist_v = None, None, None
    for comp_k, comp in cfg.items():
        if comp_k != 'data' and isinstance(comp, dict):
            d, k, v = dfs(comp)
            if d is not None:
                dist_d, dist_k, dist_v = d, k, v
                break
    
    if dist_d is not None:
        assert len(dist_v) == world_size
        dist_d[dist_k] = dist_v[rank]
        cfg.train.descs = [f'[rk{rk:02d}: {dist_k}={dist_v[rk]}]' for rk in range(world_size)]
        cfg.train.descs_key = dist_k
        cfg.train.descs_val = dist_v[rank]
    else:
        cfg.train.descs = None
    
    return cfg
