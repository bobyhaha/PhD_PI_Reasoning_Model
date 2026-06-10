import argparse, copy, json, os, yaml
from dataclasses import replace
from models import ModelConfig
from tasks import TaskConfig
from train import train_from_config
from param_match import find_width_for_params

def load_base(path):
    raw = yaml.safe_load(open(path))
    return ModelConfig(**raw['model']), TaskConfig(**raw['task']), raw['train']

def variant_cfg(base, mt):
    c = replace(base, model_type=mt)
    if mt == 'transformer': return replace(c, n_layers=4)
    if mt == 'hrm': return replace(c, h_cycles=2, l_cycles=4, h_layers=1, l_layers=1)
    if mt == 'trm': return replace(c, trm_steps=8, trm_layers=2)
    if mt == 'lg_prm': return replace(c, lg_steps=4, n_explorers=8, pi_layers=2, use_library=True, forced_library=False, hard_library_gate=True)
    raise ValueError(mt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_config', default='configs/base.yaml')
    ap.add_argument('--target_params', type=int, default=250000)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--models', nargs='+', default=['transformer', 'hrm', 'trm', 'lg_prm'])
    ap.add_argument('--out_root', default='runs/compare')
    args = ap.parse_args()
    base, task, train = load_base(args.base_config)
    if args.epochs is not None: train['epochs'] = args.epochs
    os.makedirs(args.out_root, exist_ok=True); summary = []
    for mt in args.models:
        cfg, n = find_width_for_params(variant_cfg(base, mt), args.target_params)
        tc = copy.deepcopy(train); tc['out_dir'] = os.path.join(args.out_root, f'{mt}_{n}')
        metrics = train_from_config(cfg, task, tc)
        summary.append({'model_type': mt, 'n_params': n, 'd_model': cfg.d_model, 'n_heads': cfg.n_heads, 'final_metrics': metrics})
    json.dump(summary, open(os.path.join(args.out_root, 'summary.json'), 'w'), indent=2)
    print('\nSummary')
    for r in summary: print(r)

if __name__ == '__main__': main()
