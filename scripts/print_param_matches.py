import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import ModelConfig
from param_match import find_width_for_params
for target in [100_000, 250_000, 500_000, 1_000_000]:
    print(f'target={target:,}')
    for mt in ['transformer', 'hrm', 'trm', 'lg_prm']:
        cfg, n = find_width_for_params(ModelConfig(model_type=mt), target)
        print(f'  {mt:12s} d={cfg.d_model:4d} heads={cfg.n_heads:2d} params={n:,}')
