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
        print(f"[gpu {gpu_id}] finished {model_type}", flush=True)

    return rows


@app.function(
    gpu="H200:2",
    timeout=48 * 60 * 60,
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
    preserve_base_arch: bool = False,
):
    rows = run_compare.remote(
        base_config=base_config,
        target_params=target_params,
        epochs=epochs,
        run_name=run_name,
        models_csv=models,
        max_width=max_width,
        max_steps=max_steps,
        eval_interval_steps=eval_interval_steps,
        preserve_base_arch=preserve_base_arch,
    )
    print(json.dumps(rows, indent=2))
