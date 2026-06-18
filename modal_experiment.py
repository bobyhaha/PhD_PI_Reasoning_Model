import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "phd-pi-reasoning-h200"
PROJECT_ROOT = Path("/root/project")
RESULTS_ROOT = Path("/results")
DATA_ROOT = PROJECT_ROOT / "data"
VOLUME_NAME = "phd-pi-reasoning-results"
DATA_VOLUME_NAME = "phd-pi-reasoning-data"

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


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "tqdm", "pyyaml", "matplotlib")
    .add_local_file("train.py", str(PROJECT_ROOT / "train.py"))
    .add_local_file("tasks.py", str(PROJECT_ROOT / "tasks.py"))
    .add_local_file("param_match.py", str(PROJECT_ROOT / "param_match.py"))
    .add_local_file("compare.py", str(PROJECT_ROOT / "compare.py"))
    .add_local_file("experiment_runner.py", str(PROJECT_ROOT / "experiment_runner.py"))
    .add_local_dir("models", str(PROJECT_ROOT / "models"))
    .add_local_dir("configs", str(PROJECT_ROOT / "configs"))
)

app = modal.App(APP_NAME, image=image)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)


def _parse_models(models_csv):
    if not models_csv:
        return DEFAULT_MODELS
    return [m.strip() for m in models_csv.split(",") if m.strip()]


def _split_models(models, n_shards):
    return [models[i::n_shards] for i in range(n_shards)]


def _run_worker(
    gpu_id,
    models,
    run_dir,
    base_config,
    target_params,
    epochs,
    max_width,
    max_steps,
    eval_interval_steps,
    batch_size,
    lr,
    weight_decay,
    state_max_rms,
    loss_type,
    q_halt_weight,
    lr_warmup_steps,
    lr_min_ratio,
    beta1,
    beta2,
    ema,
    ema_rate,
    lora_scale,
    lora_max,
    lora_warmup_steps,
    convergence_top_k,
    train_depth,
    train_init_noise_std,
    train_noise_scale,
    train_grad_last_only,
    eval_depth,
    eval_breadth,
    eval_grid,
    eval_max_batches,
    eval_residual_window,
    eval_init_noise_std,
    eval_noise_scale,
    preserve_base_arch,
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    rows = []
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for model_type in models:
        log_path = logs_dir / f"{model_type}.log"
        cmd = [
            sys.executable,
            "experiment_runner.py",
            "--model",
            model_type,
            "--base_config",
            base_config,
            "--target_params",
            str(target_params),
            "--epochs",
            str(epochs),
            "--out_root",
            str(run_dir / "runs"),
            "--max_width",
            str(max_width),
            "--polar_weight",
            "0.0",
            "--lora_delta_weight",
            "1e-5",
        ]
        if max_steps is not None:
            cmd.extend(["--max_steps", str(max_steps)])
        if eval_interval_steps is not None:
            cmd.extend(["--eval_interval_steps", str(eval_interval_steps)])
        if batch_size is not None:
            cmd.extend(["--batch_size", str(batch_size)])
        if lr is not None:
            cmd.extend(["--lr", str(lr)])
        if weight_decay is not None:
            cmd.extend(["--weight_decay", str(weight_decay)])
        if state_max_rms is not None:
            cmd.extend(["--state_max_rms", str(state_max_rms)])
        if loss_type is not None:
            cmd.extend(["--loss_type", str(loss_type)])
        if q_halt_weight is not None:
            cmd.extend(["--q_halt_weight", str(q_halt_weight)])
        if lr_warmup_steps is not None:
            cmd.extend(["--lr_warmup_steps", str(lr_warmup_steps)])
        if lr_min_ratio is not None:
            cmd.extend(["--lr_min_ratio", str(lr_min_ratio)])
        if beta1 is not None:
            cmd.extend(["--beta1", str(beta1)])
        if beta2 is not None:
            cmd.extend(["--beta2", str(beta2)])
        if ema:
            cmd.append("--ema")
        if ema_rate is not None:
            cmd.extend(["--ema_rate", str(ema_rate)])
        if lora_scale is not None:
            cmd.extend(["--lora_scale", str(lora_scale)])
        if lora_max is not None:
            cmd.extend(["--lora_max", str(lora_max)])
        if lora_warmup_steps is not None:
            cmd.extend(["--lora_warmup_steps", str(lora_warmup_steps)])
        if convergence_top_k is not None:
            cmd.extend(["--convergence_top_k", str(convergence_top_k)])
        if train_depth is not None:
            cmd.extend(["--train_depth", str(train_depth)])
        if train_init_noise_std is not None:
            cmd.extend(["--train_init_noise_std", str(train_init_noise_std)])
        if train_noise_scale is not None:
            cmd.extend(["--train_noise_scale", str(train_noise_scale)])
        if train_grad_last_only:
            cmd.append("--train_grad_last_only")
        if eval_depth is not None:
            cmd.extend(["--eval_depth", str(eval_depth)])
        if eval_breadth is not None:
            cmd.extend(["--eval_breadth", str(eval_breadth)])
        if eval_grid is not None:
            cmd.extend(["--eval_grid", str(eval_grid)])
        if eval_max_batches is not None:
            cmd.extend(["--eval_max_batches", str(eval_max_batches)])
        if eval_residual_window is not None:
            cmd.extend(["--eval_residual_window", str(eval_residual_window)])
        if eval_init_noise_std is not None:
            cmd.extend(["--eval_init_noise_std", str(eval_init_noise_std)])
        if eval_noise_scale is not None:
            cmd.extend(["--eval_noise_scale", str(eval_noise_scale)])
        if preserve_base_arch:
            cmd.append("--preserve_base_arch")

        print(f"[gpu {gpu_id}] starting {model_type}: {' '.join(cmd)}", flush=True)
        with open(log_path, "w") as log_file:
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"{model_type} failed on GPU {gpu_id}; see {log_path}")

        summary_paths = sorted((run_dir / "runs").glob(f"{model_type}_*_summary.json"))
        if not summary_paths:
            raise FileNotFoundError(f"Missing per-model summary for {model_type}")
        with open(summary_paths[-1]) as f:
            rows.append(json.load(f))
        results_volume.commit()
        print(f"[gpu {gpu_id}] finished {model_type}", flush=True)

    return rows


@app.function(
    gpu="H200:2",
    timeout=24 * 60 * 60,
    volumes={str(RESULTS_ROOT): results_volume, str(DATA_ROOT): data_volume},
)
def run_compare(
    base_config: str = "configs/base.yaml",
    target_params: int = 500000,
    epochs: int = 30,
    run_name: str = "full_meta_compare_500k",
    models_csv: str = "",
    max_width: int = 512,
    max_steps: int | None = None,
    eval_interval_steps: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    state_max_rms: float | None = None,
    loss_type: str | None = None,
    q_halt_weight: float | None = None,
    lr_warmup_steps: int | None = None,
    lr_min_ratio: float | None = None,
    beta1: float | None = None,
    beta2: float | None = None,
    ema: bool = False,
    ema_rate: float | None = None,
    lora_scale: float | None = None,
    lora_max: float | None = None,
    lora_warmup_steps: int | None = None,
    convergence_top_k: int | None = None,
    train_depth: int | None = None,
    train_init_noise_std: float | None = None,
    train_noise_scale: float | None = None,
    train_grad_last_only: bool = False,
    eval_depth: int | None = None,
    eval_breadth: int | None = None,
    eval_grid: str | None = None,
    eval_max_batches: int | None = None,
    eval_residual_window: int | None = None,
    eval_init_noise_std: float | None = None,
    eval_noise_scale: float | None = None,
    preserve_base_arch: bool = False,
):
    import torch

    models = _parse_models(models_csv)
    run_dir = RESULTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(f"Expected 2 GPUs from gpu='H200:2', saw {n_gpus}")

    assignments = [models[::2], models[1::2]]
    print(f"Running {len(models)} models on {n_gpus} GPUs", flush=True)
    print(f"GPU 0 models: {assignments[0]}", flush=True)
    print(f"GPU 1 models: {assignments[1]}", flush=True)

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _run_worker,
                gpu_id,
                assigned,
                run_dir,
                base_config,
                target_params,
                epochs,
                max_width,
                max_steps,
                eval_interval_steps,
                batch_size,
                lr,
                weight_decay,
                state_max_rms,
                loss_type,
                q_halt_weight,
                lr_warmup_steps,
                lr_min_ratio,
                beta1,
                beta2,
                ema,
                ema_rate,
                lora_scale,
                lora_max,
                lora_warmup_steps,
                convergence_top_k,
                train_depth,
                train_init_noise_std,
                train_noise_scale,
                train_grad_last_only,
                eval_depth,
                eval_breadth,
                eval_grid,
                eval_max_batches,
                eval_residual_window,
                eval_init_noise_std,
                eval_noise_scale,
                preserve_base_arch,
            )
            for gpu_id, assigned in enumerate(assignments)
            if assigned
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())

    rows.sort(key=lambda r: DEFAULT_MODELS.index(r["model_type"]) if r["model_type"] in DEFAULT_MODELS else 999)
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)

    results_volume.commit()
    print(f"Saved summary to {summary_path}", flush=True)
    return rows


@app.function(
    gpu="H200",
    timeout=24 * 60 * 60,
    volumes={str(RESULTS_ROOT): results_volume, str(DATA_ROOT): data_volume},
)
def run_compare_1gpu(
    base_config: str = "configs/base.yaml",
    target_params: int = 500000,
    epochs: int = 30,
    run_name: str = "full_meta_compare_500k",
    models_csv: str = "",
    max_width: int = 512,
    max_steps: int | None = None,
    eval_interval_steps: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    state_max_rms: float | None = None,
    loss_type: str | None = None,
    q_halt_weight: float | None = None,
    lr_warmup_steps: int | None = None,
    lr_min_ratio: float | None = None,
    beta1: float | None = None,
    beta2: float | None = None,
    ema: bool = False,
    ema_rate: float | None = None,
    lora_scale: float | None = None,
    lora_max: float | None = None,
    lora_warmup_steps: int | None = None,
    convergence_top_k: int | None = None,
    train_depth: int | None = None,
    train_init_noise_std: float | None = None,
    train_noise_scale: float | None = None,
    train_grad_last_only: bool = False,
    eval_depth: int | None = None,
    eval_breadth: int | None = None,
    eval_grid: str | None = None,
    eval_max_batches: int | None = None,
    eval_residual_window: int | None = None,
    eval_init_noise_std: float | None = None,
    eval_noise_scale: float | None = None,
    preserve_base_arch: bool = False,
):
    models = _parse_models(models_csv)
    run_dir = RESULTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(models)} models on 1 H200", flush=True)
    print(f"GPU 0 models: {models}", flush=True)

    rows = _run_worker(
        0,
        models,
        run_dir,
        base_config,
        target_params,
        epochs,
        max_width,
        max_steps,
        eval_interval_steps,
        batch_size,
        lr,
        weight_decay,
        state_max_rms,
        loss_type,
        q_halt_weight,
        lr_warmup_steps,
        lr_min_ratio,
        beta1,
        beta2,
        ema,
        ema_rate,
        lora_scale,
        lora_max,
        lora_warmup_steps,
        convergence_top_k,
        train_depth,
        train_init_noise_std,
        train_noise_scale,
        train_grad_last_only,
        eval_depth,
        eval_breadth,
        eval_grid,
        eval_max_batches,
        eval_residual_window,
        eval_init_noise_std,
        eval_noise_scale,
        preserve_base_arch,
    )

    rows.sort(key=lambda r: DEFAULT_MODELS.index(r["model_type"]) if r["model_type"] in DEFAULT_MODELS else 999)
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)

    results_volume.commit()
    print(f"Saved summary to {summary_path}", flush=True)
    return rows


@app.local_entrypoint()
def main(
    base_config: str = "configs/base.yaml",
    target_params: int = 500000,
    epochs: int = 30,
    run_name: str = "full_meta_compare_500k",
    models: str = "",
    max_width: int = 512,
    max_steps: int | None = None,
    eval_interval_steps: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    state_max_rms: float | None = None,
    loss_type: str | None = None,
    q_halt_weight: float | None = None,
    lr_warmup_steps: int | None = None,
    lr_min_ratio: float | None = None,
    beta1: float | None = None,
    beta2: float | None = None,
    ema: bool = False,
    ema_rate: float | None = None,
    lora_scale: float | None = None,
    lora_max: float | None = None,
    lora_warmup_steps: int | None = None,
    convergence_top_k: int | None = None,
    train_depth: int | None = None,
    train_init_noise_std: float | None = None,
    train_noise_scale: float | None = None,
    train_grad_last_only: bool = False,
    eval_depth: int | None = None,
    eval_breadth: int | None = None,
    eval_grid: str | None = None,
    eval_max_batches: int | None = None,
    eval_residual_window: int | None = None,
    eval_init_noise_std: float | None = None,
    eval_noise_scale: float | None = None,
    preserve_base_arch: bool = False,
    gpu_count: int = 2,
    wait: bool = False,
):
    kwargs = dict(
        base_config=base_config,
        target_params=target_params,
        epochs=epochs,
        run_name=run_name,
        models_csv=models,
        max_width=max_width,
        max_steps=max_steps,
        eval_interval_steps=eval_interval_steps,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        state_max_rms=state_max_rms,
        loss_type=loss_type,
        q_halt_weight=q_halt_weight,
        lr_warmup_steps=lr_warmup_steps,
        lr_min_ratio=lr_min_ratio,
        beta1=beta1,
        beta2=beta2,
        ema=ema,
        ema_rate=ema_rate,
        lora_scale=lora_scale,
        lora_max=lora_max,
        lora_warmup_steps=lora_warmup_steps,
        convergence_top_k=convergence_top_k,
        train_depth=train_depth,
        train_init_noise_std=train_init_noise_std,
        train_noise_scale=train_noise_scale,
        train_grad_last_only=train_grad_last_only,
        eval_depth=eval_depth,
        eval_breadth=eval_breadth,
        eval_grid=eval_grid,
        eval_max_batches=eval_max_batches,
        eval_residual_window=eval_residual_window,
        eval_init_noise_std=eval_init_noise_std,
        eval_noise_scale=eval_noise_scale,
        preserve_base_arch=preserve_base_arch,
    )
    if gpu_count > 2:
        parsed_models = _parse_models(models)
        shards = _split_models(parsed_models, gpu_count)
        calls = []
        for shard_id, shard_models in enumerate(shards):
            if not shard_models:
                continue
            shard_kwargs = dict(kwargs)
            shard_kwargs["models_csv"] = ",".join(shard_models)
            shard_kwargs["run_name"] = f"{run_name}/shard_{shard_id:02d}"
            call = run_compare_1gpu.spawn(**shard_kwargs)
            calls.append((shard_id, shard_models, call))
            print(f"Spawned shard {shard_id:02d} on 1 H200: {shard_models}")
            print(f"  call: {call.object_id}")
            print(f"  dashboard: {call.get_dashboard_url()}")

        if not wait:
            print(f"Spawned {len(calls)} single-H200 calls for gpu_count={gpu_count}.")
            print("Use --wait to block locally and print combined JSON rows.")
            return

        rows = []
        for _, _, call in calls:
            rows.extend(call.get())
        rows.sort(key=lambda r: DEFAULT_MODELS.index(r["model_type"]) if r["model_type"] in DEFAULT_MODELS else 999)
        print(json.dumps(rows, indent=2))
        return

    runner = run_compare_1gpu if gpu_count == 1 else run_compare
    if not wait:
        call = runner.spawn(**kwargs)
        print(f"Spawned Modal call {call.object_id}")
        print(f"Dashboard: {call.get_dashboard_url()}")
        print("Use --wait to block locally and print the final JSON rows.")
        return

    rows = runner.remote(**kwargs)
    print(json.dumps(rows, indent=2))
