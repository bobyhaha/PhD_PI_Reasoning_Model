# Reasoning Model Comparison

Compares same-parameter versions of:
- `transformer`: standard Transformer encoder
- `hrm`: EqR/HRM-style high/low recurrent token-latent model
- `trm`: EqR-style single recurrent token-latent core
- `lg_prm`: your library-gated PI-PhD model with separate RAG and MLP libraries

Meta-model variants are available for HRM, TRM, and LG-PRM:
- V1 / PoLar-style program-of-layers: `hrm_v1`, `trm_v1`, `lg_prm_v1`
- V2 / per-loop LoRA hypernetwork: `hrm_v2`, `trm_v2`, `lg_prm_v2`
- V3 / combined PoLar + LoRA: `hrm_v3`, `trm_v3`, `lg_prm_v3`

The V1 controller predicts input-specific keep/skip gates for recurrent layer
slots, inspired by PoLar's program-of-layers view. The V2 controller predicts
low-rank LoRA deltas per reasoning loop from the input and current state. V3
uses both: the program decides which layer slots execute, and the hypernetwork
decides how executed slots are adapted.

Training objective for meta variants:
```text
loss = prediction_loss
     + retrieval_cost * library_gate_usage
     + diversity_weight * explorer_diversity
     - entropy_weight * library_entropy
     + polar_weight * polar_usage
     + lora_delta_weight * lora_delta_norm
```
`prediction_loss` is the primary objective. `polar_weight` optionally encourages
shorter programs, while `lora_delta_weight` keeps generated adapters small unless
the task loss rewards using them.

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
