from .common import ModelConfig, count_parameters, diversity_loss
from .transformer import StandardTransformer
from .hrm import HRM
from .trm import TRM
from .lg_prm import LibraryGatedPRM
from .trm_meta import TRM_V1, TRM_V2, TRM_V3
from .hrm_meta import HRM_V1, HRM_V2, HRM_V3
from .lg_prm_meta import LGP_V1, LGP_V2, LGP_V3

def build_model(cfg):
    if cfg.model_type == 'transformer':
        return StandardTransformer(cfg)
    if cfg.model_type == 'hrm':
        return HRM(cfg)
    if cfg.model_type == 'trm':
        return TRM(cfg)
    if cfg.model_type == 'lg_prm':
        return LibraryGatedPRM(cfg)
    if cfg.model_type == 'trm_v1':
        return TRM_V1(cfg)
    if cfg.model_type == 'trm_v2':
        return TRM_V2(cfg)
    if cfg.model_type == 'trm_v3':
        return TRM_V3(cfg)
    if cfg.model_type == 'hrm_v1':
        return HRM_V1(cfg)
    if cfg.model_type == 'hrm_v2':
        return HRM_V2(cfg)
    if cfg.model_type == 'hrm_v3':
        return HRM_V3(cfg)
    if cfg.model_type in {'lgp_v1', 'lg_prm_v1'}:
        return LGP_V1(cfg)
    if cfg.model_type in {'lgp_v2', 'lg_prm_v2'}:
        return LGP_V2(cfg)
    if cfg.model_type in {'lgp_v3', 'lg_prm_v3'}:
        return LGP_V3(cfg)
    raise ValueError(f'Unknown model_type: {cfg.model_type}')
