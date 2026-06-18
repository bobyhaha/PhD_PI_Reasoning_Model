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
    "eqr",
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
        return replace(cfg, trm_steps=4, trm_layers=2)

    if model_type == "eqr":
        return replace(
            cfg,
            h_cycles=3,
            l_cycles=6,
            h_layers=0,
            l_layers=2,
            mlp_t=True,
            phd_lambda=0.95,
            phd_noise_scale=0.01,
            init_std=1.0,
            eqr_halt_max_steps=16,
            halt_exploration_prob=0.1,
            q_halt_weight=0.5,
        )

    if model_type == "hrm" or model_type.startswith("hrm_v"):
        return replace(cfg, h_cycles=2, l_cycles=3, h_layers=1, l_layers=1)

    if model_type == "lg_prm" or model_type.startswith("lg_prm"):
        return replace(cfg, lg_steps=4, n_explorers=8, pi_layers=2)

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
    batch_size=None,
    lr=None,
    weight_decay=None,
    state_max_rms=None,
    loss_type=None,
    q_halt_weight=None,
    lr_warmup_steps=None,
    lr_min_ratio=None,
    beta1=None,
    beta2=None,
    ema=None,
    ema_rate=None,
    lora_scale=None,
    lora_max=None,
    lora_warmup_steps=None,
    convergence_top_k=None,
    train_depth=None,
    train_init_noise_std=None,
    train_noise_scale=None,
    train_grad_last_only=False,
    eval_depth=None,
    eval_breadth=None,
    eval_grid=None,
    eval_max_batches=None,
    eval_residual_window=None,
    eval_init_noise_std=None,
    eval_noise_scale=None,
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
    if state_max_rms is not None:
        cfg = replace(cfg, state_max_rms=state_max_rms)
        n_params = matched_config(
            replace(base, state_max_rms=state_max_rms),
            model_type,
            target_params,
            max_width,
            preserve_base_arch=preserve_base_arch,
        )[1]

    train_cfg = copy.deepcopy(train)
    train_cfg["epochs"] = epochs
    train_cfg["out_dir"] = os.path.join(out_root, f"{model_type}_{n_params}")
    train_cfg["polar_weight"] = polar_weight
    train_cfg["lora_delta_weight"] = lora_delta_weight
    if max_steps is not None:
        train_cfg["max_steps"] = max_steps
    if eval_interval_steps is not None:
        train_cfg["eval_interval_steps"] = eval_interval_steps
    if batch_size is not None:
        train_cfg["batch_size"] = batch_size
    if lr is not None:
        train_cfg["lr"] = lr
    if weight_decay is not None:
        train_cfg["weight_decay"] = weight_decay
    if state_max_rms is not None:
        train_cfg["state_max_rms"] = state_max_rms
    if loss_type is not None:
        train_cfg["loss_type"] = loss_type
    if q_halt_weight is not None:
        train_cfg["q_halt_weight"] = q_halt_weight
    if lr_warmup_steps is not None:
        train_cfg["lr_warmup_steps"] = lr_warmup_steps
    if lr_min_ratio is not None:
        train_cfg["lr_min_ratio"] = lr_min_ratio
    if beta1 is not None:
        train_cfg["beta1"] = beta1
    if beta2 is not None:
        train_cfg["beta2"] = beta2
    if ema is not None:
        train_cfg["ema"] = ema
    if ema_rate is not None:
        train_cfg["ema_rate"] = ema_rate
    if lora_scale is not None:
        train_cfg["lora_scale"] = lora_scale
        cfg = replace(cfg, lora_scale=lora_scale)
    if lora_max is not None:
        train_cfg["lora_max"] = lora_max
        cfg = replace(cfg, lora_max=lora_max)
    if lora_warmup_steps is not None:
        train_cfg["lora_warmup_steps"] = lora_warmup_steps
    if convergence_top_k is not None:
        train_cfg["convergence_top_k"] = convergence_top_k
    if train_depth is not None:
        train_cfg["train_depth"] = train_depth
    if train_init_noise_std is not None:
        train_cfg["train_init_noise_std"] = train_init_noise_std
    if train_noise_scale is not None:
        train_cfg["train_noise_scale"] = train_noise_scale
    if train_grad_last_only:
        train_cfg["train_grad_last_only"] = True
    if eval_depth is not None:
        train_cfg["eval_depth"] = eval_depth
    if eval_breadth is not None:
        train_cfg["eval_breadth"] = eval_breadth
    if eval_grid is not None:
        train_cfg["eval_grid"] = eval_grid
    if eval_max_batches is not None:
        train_cfg["eval_max_batches"] = eval_max_batches
    if eval_residual_window is not None:
        train_cfg["eval_residual_window"] = eval_residual_window
    if eval_init_noise_std is not None:
        train_cfg["eval_init_noise_std"] = eval_init_noise_std
    if eval_noise_scale is not None:
        train_cfg["eval_noise_scale"] = eval_noise_scale
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
        "batch_size": train_cfg.get("batch_size"),
        "lr": train_cfg.get("lr"),
        "weight_decay": train_cfg.get("weight_decay"),
        "state_max_rms": train_cfg.get("state_max_rms"),
        "loss_type": train_cfg.get("loss_type", "cross_entropy"),
        "q_halt_weight": train_cfg.get("q_halt_weight"),
        "lr_warmup_steps": train_cfg.get("lr_warmup_steps"),
        "lr_min_ratio": train_cfg.get("lr_min_ratio"),
        "beta1": train_cfg.get("beta1"),
        "beta2": train_cfg.get("beta2"),
        "ema": train_cfg.get("ema"),
        "ema_rate": train_cfg.get("ema_rate"),
        "lora_scale": train_cfg.get("lora_scale"),
        "lora_max": train_cfg.get("lora_max"),
        "lora_warmup_steps": train_cfg.get("lora_warmup_steps"),
        "convergence_top_k": train_cfg.get("convergence_top_k"),
        "train_depth": train_cfg.get("train_depth"),
        "train_init_noise_std": train_cfg.get("train_init_noise_std"),
        "train_noise_scale": train_cfg.get("train_noise_scale"),
        "train_grad_last_only": train_cfg.get("train_grad_last_only"),
        "eval_depth": train_cfg.get("eval_depth"),
        "eval_breadth": train_cfg.get("eval_breadth"),
        "eval_grid": train_cfg.get("eval_grid"),
        "eval_max_batches": train_cfg.get("eval_max_batches"),
        "eval_residual_window": train_cfg.get("eval_residual_window", 3),
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
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--state_max_rms", type=float)
    parser.add_argument("--loss_type", choices=["cross_entropy", "stablemax_cross_entropy"])
    parser.add_argument("--q_halt_weight", type=float)
    parser.add_argument("--lr_warmup_steps", type=int)
    parser.add_argument("--lr_min_ratio", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_rate", type=float)
    parser.add_argument("--lora_scale", type=float)
    parser.add_argument("--lora_max", type=float)
    parser.add_argument("--lora_warmup_steps", type=int)
    parser.add_argument("--convergence_top_k", type=int)
    parser.add_argument("--train_depth", type=int)
    parser.add_argument("--train_init_noise_std", type=float)
    parser.add_argument("--train_noise_scale", type=float)
    parser.add_argument("--train_grad_last_only", action="store_true")
    parser.add_argument("--eval_depth", type=int)
    parser.add_argument("--eval_breadth", type=int)
    parser.add_argument("--eval_grid", type=str)
    parser.add_argument("--eval_max_batches", type=int)
    parser.add_argument("--eval_residual_window", type=int)
    parser.add_argument("--eval_init_noise_std", type=float)
    parser.add_argument("--eval_noise_scale", type=float)
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
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        state_max_rms=args.state_max_rms,
        loss_type=args.loss_type,
        q_halt_weight=args.q_halt_weight,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_min_ratio=args.lr_min_ratio,
        beta1=args.beta1,
        beta2=args.beta2,
        ema=args.ema if args.ema else None,
        ema_rate=args.ema_rate,
        lora_scale=args.lora_scale,
        lora_max=args.lora_max,
        lora_warmup_steps=args.lora_warmup_steps,
        convergence_top_k=args.convergence_top_k,
        train_depth=args.train_depth,
        train_init_noise_std=args.train_init_noise_std,
        train_noise_scale=args.train_noise_scale,
        train_grad_last_only=args.train_grad_last_only,
        eval_depth=args.eval_depth,
        eval_breadth=args.eval_breadth,
        eval_grid=args.eval_grid,
        eval_max_batches=args.eval_max_batches,
        eval_residual_window=args.eval_residual_window,
        eval_init_noise_std=args.eval_init_noise_std,
        eval_noise_scale=args.eval_noise_scale,
        preserve_base_arch=args.preserve_base_arch,
        dry_run=args.dry_run,
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
