#!/usr/bin/env bash
set -e
python train.py --config configs/smoke.yaml
python train.py --config configs/sudoku_smoke.yaml
python train.py --config configs/sudoku_lg_prm_smoke.yaml
python -m py_compile tasks.py train.py compare.py param_match.py models/*.py
python compare.py --base_config configs/base.yaml --target_params 250000 --epochs 5 --out_root runs/compare_250k
python compare.py --base_config configs/base.yaml --target_params 500000 --epochs 5 --out_root runs/compare_500k
python plot_results.py --runs_dir runs --out comparison.png
