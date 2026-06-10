import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import build_model, count_parameters
from train import load_config


def token_block_flops(seq_len, d_model, ff_mult):
    return (4 + 2 * ff_mult) * seq_len * d_model * d_model + 2 * seq_len * seq_len * d_model


def trm_flops(cfg, seq_len):
    core = cfg.trm_steps * cfg.trm_layers * token_block_flops(seq_len, cfg.d_model, cfg.dim_ff_mult)
    heads = seq_len * cfg.d_model * (cfg.vocab_size + cfg.num_classes)
    return core + heads


def hrm_flops(cfg, seq_len):
    low = cfg.h_cycles * cfg.l_cycles * cfg.l_layers * token_block_flops(seq_len, cfg.d_model, cfg.dim_ff_mult)
    high = cfg.h_cycles * cfg.h_layers * token_block_flops(seq_len, cfg.d_model, cfg.dim_ff_mult)
    heads = seq_len * cfg.d_model * (cfg.vocab_size + cfg.num_classes)
    return low + high + heads


def transformer_flops(cfg, seq_len):
    core = cfg.n_layers * token_block_flops(seq_len, cfg.d_model, cfg.dim_ff_mult)
    heads = seq_len * cfg.d_model * (cfg.vocab_size + cfg.num_classes)
    return core + heads


def lg_prm_flops(cfg, seq_len, active_library=1.0):
    d = cfg.d_model
    explorer_hidden = max(8, int(d * cfg.d_explorer_mult))
    explorer = cfg.n_explorers * (3 * d * explorer_hidden)
    gate = 2 * d * d + d
    rag = 2 * cfg.rag_library_size * d
    mlp_library = 2 * cfg.mlp_library_mult * d * d
    pi_in = 4 * d * d
    pi_layers = cfg.pi_layers * ((4 + 2 * cfg.dim_ff_mult) * d * d + 2 * d)
    pi_out = d * d
    step = explorer + gate + active_library * (rag + mlp_library) + pi_in + pi_layers + pi_out
    heads = seq_len * d * cfg.num_classes + d * cfg.num_classes
    return cfg.lg_steps * step + heads


def estimate_flops(cfg, seq_len):
    if cfg.model_type == "transformer":
        return transformer_flops(cfg, seq_len)
    if cfg.model_type == "trm":
        return trm_flops(cfg, seq_len)
    if cfg.model_type == "hrm":
        return hrm_flops(cfg, seq_len)
    if cfg.model_type == "lg_prm":
        active = 1.0 if cfg.forced_library else 0.5
        return lg_prm_flops(cfg, seq_len, active)
    raise ValueError(cfg.model_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args()
    for path in args.configs:
        model_cfg, task_cfg, _ = load_config(path)
        params = count_parameters(build_model(model_cfg))
        flops = estimate_flops(model_cfg, task_cfg.seq_len)
        print(f"{path}: model={model_cfg.model_type} params={params:,} est_forward_flops={flops / 1e6:.1f}M")


if __name__ == "__main__":
    main()
