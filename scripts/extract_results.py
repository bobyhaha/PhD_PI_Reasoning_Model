import argparse
import csv
import json
from pathlib import Path


METRIC_KEYS = [
    "val_loss",
    "val_acc",
    "val_token_acc",
    "val_exact_acc",
    "polar_usage",
    "lora_delta_norm",
    "eval_depth",
    "eval_breadth",
    "eval_depth_token_acc",
    "eval_depth_exact_acc",
    "eval_majority_token_acc",
    "eval_majority_exact_acc",
    "eval_convergence_token_acc",
    "eval_convergence_exact_acc",
    "eval_convergence_any_correct",
    "eval_convergence_top_k",
    "eval_avg_pass_rate",
    "eval_any_correct",
]


def load_jsonl_last(path: Path) -> dict:
    if not path.exists():
        return {}
    last = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def local_run_dir(run_root: Path, out_dir: str) -> Path:
    return run_root / "runs" / Path(out_dir).name


def summary_paths(run_root: Path) -> list[Path]:
    root_summary = run_root / "summary.json"
    if root_summary.exists():
        return [root_summary]
    paths = sorted(run_root.glob("shard_*/summary.json"))
    if paths:
        return paths
    raise FileNotFoundError(f"missing {root_summary} or shard_*/summary.json under {run_root}")


def collect_rows(run_root: Path) -> list[dict]:
    rows = []
    try:
        paths = summary_paths(run_root)
    except FileNotFoundError:
        paths = sorted(run_root.glob("shard_*/runs/*_summary.json"))
        if not paths:
            paths = sorted(run_root.glob("runs/*_summary.json"))
        if not paths:
            raise

    for summary_path in paths:
        shard_root = summary_path.parent
        summary = json.loads(summary_path.read_text())
        if isinstance(summary, dict):
            summary = [summary]
            if summary_path.parent.name == "runs":
                shard_root = summary_path.parent.parent
        for item in summary:
            metrics = dict(item.get("final_metrics") or {})
            metrics.update(load_jsonl_last(local_run_dir(shard_root, item["out_dir"]) / "metrics.jsonl"))
            row = {
                "model_type": item.get("model_type"),
                "n_params": item.get("n_params"),
                "d_model": item.get("d_model"),
                "n_heads": item.get("n_heads"),
                "epoch": metrics.get("epoch"),
                "step": metrics.get("step"),
                "shard": shard_root.name if shard_root != run_root else "",
            }
            for key in METRIC_KEYS:
                row[key] = metrics.get(key)
            for key, value in metrics.items():
                if key.startswith("frontier_"):
                    row[key] = value
            rows.append(row)
    rows.sort(key=lambda row: row["model_type"] or "")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", help="Downloaded Modal run directory, e.g. modal_results/sudoku_phd_pi_50k")
    parser.add_argument("--out", default="", help="CSV output path. Defaults to <run_root>/results.csv")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_path = Path(args.out) if args.out else run_root / "results.csv"
    rows = collect_rows(run_root)
    dynamic_keys = sorted({key for row in rows for key in row if key.startswith("frontier_")})
    fieldnames = ["model_type", "shard", "n_params", "d_model", "n_heads", "epoch", "step", *METRIC_KEYS, *dynamic_keys]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_path}")
    for row in rows:
        metrics = ", ".join(f"{k}={row[k]}" for k in METRIC_KEYS if row.get(k) is not None)
        print(f"{row['model_type']}: params={row['n_params']} step={row['step']} {metrics}")


if __name__ == "__main__":
    main()
