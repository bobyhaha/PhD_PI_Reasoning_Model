from .common import ModelConfig, count_parameters, diversity_loss
from .transformer import StandardTransformer
from .hrm import HRM
from .trm import TRM
from .lg_prm import LibraryGatedPRM

def build_model(cfg):
    if cfg.model_type == 'transformer':
        return StandardTransformer(cfg)
    if cfg.model_type == 'hrm':
        return HRM(cfg)
    if cfg.model_type == 'trm':
        return TRM(cfg)
    if cfg.model_type == 'lg_prm':
        return LibraryGatedPRM(cfg)
    raise ValueError(f'Unknown model_type: {cfg.model_type}')
