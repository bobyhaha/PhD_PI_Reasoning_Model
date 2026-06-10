# Reasoning Model Comparison

Compares same-parameter versions of:
- `transformer`: standard Transformer encoder
- `hrm`: EqR/HRM-style high/low recurrent token-latent model
- `trm`: EqR-style single recurrent token-latent core
- `lg_prm`: your library-gated PI-PhD model with separate RAG and MLP libraries

Install:
```bash
pip install torch numpy tqdm pyyaml matplotlib
```

Smoke test:
```bash
python train.py --config configs/smoke.yaml
```

Sudoku smoke test using the released EqR dataset:
```bash
python train.py --config configs/sudoku_smoke.yaml
```

LG-PRM Sudoku smoke test:
```bash
python train.py --config configs/sudoku_lg_prm_smoke.yaml
```

Full Sudoku runs for an H100:
```bash
python train.py --config configs/sudoku_trm_h100.yaml
python train.py --config configs/sudoku_lg_prm_h100.yaml
```

Compare around 250k params:
```bash
python compare.py --base_config configs/base.yaml --target_params 250000 --epochs 5
```

Plot:
```bash
python plot_results.py --runs_dir runs
```

EqR/HRM-style `.npy` puzzle datasets can be used with token-level models by setting:
```yaml
task:
  task_name: puzzle_npy
  dataset_path: data/sudoku-extreme-1k-aug-1000
  puzzle_set: all
```

The EqR download stores Sudoku at `data/sudoku-extreme-1k-aug-1000` and uses `train/` plus `test/` splits. The local loader falls back from `val/` to `test/` for validation. For sequence labels, models train with token cross-entropy and log token/exact accuracy. Scalar tasks continue to use ordinary classification accuracy.
# PhD_PI_Reasoning_Model
