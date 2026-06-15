import argparse
import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import yaml

from models import ModelConfig
from param_match import find_width_for_params
from tasks import TaskConfig
from train import train_from_config


DEFAULT_MODELS = [
    "trm",
    "trm_v1",
    "trm_v2",
    "trm_v3",
    "hrm",
    "hrm_v1",
    "hrm_v2",
    "hrm_v3",
    "lg_prm",
    "lg_prm_v1",
    "lg_prm_v2",
    "lg_prm_v3",
]


def load_base(path):
    raw = yaml.safe_load(open(path))
    return ModelConfig(**raw["model"]), TaskConfig(**raw["task"]), raw["train"]


def variant_cfg(base, model_type, preserve_base_arch=False):
    cfg = replace(base, model_type=model_type)
    if preserve_base_arch:
        return cfg

    if model_type == "transformer":
        return replace(cfg, n_layers=4)

    if model_type == "trm" or model_type.startswith("trm_v"):
        return replace(cfg, trm_steps=8, trm_layers=2)

    if model_type == "hrm" or model_type.startswith("hrm_v"):
        return replace(cfg, h_cycles=2, l_cycles=4, h_layers=1, l_layers=1)

    if model_type == "lg_prm" or model_type.startswith("lg_prm"):
        return replace(
            cfg,
            lg_steps=4,
            n_explorers=8,
            pi_layers=2,
            use_library=True,
            forced_library=False,
            hard_library_gate=True,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def matched_config(base, model_type, target_params, max_width, preserve_base_arch=False):
    widths = list(range(16, max_width + 1, 8))
    return find_width_for_params(
        variant_cfg(base, model_type, preserve_base_arch=preserve_base_arch),
        target_params,
        widths=widths,
    )


def run_one(
    model_type,
    base_config,
    target_params,
    epochs,
    out_root,
    max_width=512,
    polar_weight=0.0,
    lora_delta_weight=1e-5,
    seed=None,
    max_steps=None,
    eval_interval_steps=None,
    preserve_base_arch=False,
    dry_run=False,
):
    base, task, train = load_base(base_config)
    cfg, n_params = matched_config(
        base,
        model_type,
        target_params,
        max_width,
        preserve_base_arch=preserve_base_arch,
    )

    train_cfg = copy.deepcopy(train)
    train_cfg["epochs"] = epochs
    train_cfg["out_dir"] = os.path.join(out_root, f"{model_type}_{n_params}")
    train_cfg["polar_weight"] = polar_weight
    train_cfg["lora_delta_weight"] = lora_delta_weight
    if max_steps is not None:
        train_cfg["max_steps"] = max_steps
    if eval_interval_steps is not None:
        train_cfg["eval_interval_steps"] = eval_interval_steps
    if seed is not None:
        train_cfg["seed"] = seed

    row = {
        "model_type": model_type,
        "target_params": target_params,
        "n_params": n_params,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "epochs": epochs,
        "max_steps": max_steps,
        "eval_interval_steps": eval_interval_steps,
        "out_dir": train_cfg["out_dir"],
    }

    if dry_run:
        print(json.dumps(row, indent=2))
        return row

    metrics = train_from_config(cfg, task, train_cfg)
    row["final_metrics"] = metrics

    Path(out_root).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_root, f"{model_type}_{n_params}_summary.json"), "w") as f:
        json.dump(row, f, indent=2)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--target_params", type=int, default=500000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out_root", default="runs/full_meta_compare_500k")
    parser.add_argument("--max_width", type=int, default=512)
    parser.add_argument("--polar_weight", type=float, default=0.0)
    parser.add_argument("--lora_delta_weight", type=float, default=1e-5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--eval_interval_steps", type=int)
    parser.add_argument("--preserve_base_arch", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    row = run_one(
        model_type=args.model,
        base_config=args.base_config,
        target_params=args.target_params,
        epochs=args.epochs,
        out_root=args.out_root,
        max_width=args.max_width,
        polar_weight=args.polar_weight,
        lora_delta_weight=args.lora_delta_weight,
        seed=args.seed,
        max_steps=args.max_steps,
        eval_interval_steps=args.eval_interval_steps,
        preserve_base_arch=args.preserve_base_arch,
        dry_run=args.dry_run,
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
