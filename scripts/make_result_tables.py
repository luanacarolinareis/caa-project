"""Aggregate per-seed evaluation JSONs into summary tables.

Supports both binary and three-class classification results.

Usage:
    python scripts/make_result_tables.py                     # binary (default)
    python scripts/make_result_tables.py --task three_class   # three-class
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

METRICS_DIR = Path("results/metrics")
MODELS = ["custom_cnn", "resnet18", "densenet121"]
SEEDS = [0, 1, 2]

BINARY_METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "auroc", "inference_time_ms_per_image"]
THREE_CLASS_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auroc", "inference_time_ms_per_image"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics.")
    parser.add_argument(
        "--task",
        choices=["binary", "three_class"],
        default="binary",
        help="Which task results to aggregate.",
    )
    return parser.parse_args()


def load_seed_metrics(task: str) -> list[dict]:
    rows = []
    metrics_list = THREE_CLASS_METRICS if task == "three_class" else BINARY_METRICS
    suffix = "_3class" if task == "three_class" else ""

    for model in MODELS:
        for seed in SEEDS:
            path = METRICS_DIR / f"{model}{suffix}_seed{seed}.json"
            if not path.exists():
                print(f"WARNING: missing {path}")
                continue
            with path.open() as f:
                data = json.load(f)
            row = {"model": model, "seed": seed}
            for m in metrics_list:
                row[m] = data.get(m)

            # For three-class, also grab per-class F1
            if task == "three_class" and "f1_per_class" in data:
                for cls_name, f1_val in data["f1_per_class"].items():
                    row[f"f1_{cls_name}"] = f1_val

            rows.append(row)
    return rows


def build_summary(rows: list[dict], task: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metrics_list = THREE_CLASS_METRICS if task == "three_class" else BINARY_METRICS

    # Add per-class F1 columns for three-class
    extra_cols = []
    if task == "three_class":
        for col in df.columns:
            if col.startswith("f1_") and col not in metrics_list and col != "f1_macro":
                extra_cols.append(col)

    numeric = metrics_list + sorted(extra_cols)
    # Filter to columns that exist
    numeric = [c for c in numeric if c in df.columns]

    agg = (
        df.groupby("model")[numeric]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reindex(MODELS)


def format_markdown(summary: pd.DataFrame, task: str) -> str:
    if task == "three_class":
        display_metrics = THREE_CLASS_METRICS
        header_map = {
            "accuracy": "Accuracy",
            "precision_macro": "Precision (macro)",
            "recall_macro": "Recall (macro)",
            "f1_macro": "F1 (macro)",
            "auroc": "AUROC (OvR)",
            "inference_time_ms_per_image": "Inference (ms/img)",
        }
    else:
        display_metrics = BINARY_METRICS
        header_map = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "specificity": "Specificity",
            "f1": "F1",
            "auroc": "AUROC",
            "inference_time_ms_per_image": "Inference (ms/img)",
        }

    cols = ["Model"]
    for m in display_metrics:
        cols.append(header_map.get(m, m))

    sep = ["---"] + ["---"] * len(display_metrics)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(sep) + " |"]

    for model in MODELS:
        if model not in summary.index:
            continue
        row_vals = [model]
        for m in display_metrics:
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"
            if mean_col in summary.columns:
                mean = summary.loc[model, mean_col]
                std = summary.loc[model, std_col]
                row_vals.append(f"{mean:.4f} ± {std:.4f}")
            else:
                row_vals.append("—")
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_seed_metrics(args.task)
    if not rows:
        print("No metric files found. Run evaluate_model.py first.")
        return

    suffix = "_3class" if args.task == "three_class" else ""

    df_all = pd.DataFrame(rows)
    csv_per_seed = METRICS_DIR / f"all_seeds{suffix}_results.csv"
    df_all.to_csv(csv_per_seed, index=False)
    print(f"Saved per-seed CSV: {csv_per_seed}")

    summary = build_summary(rows, args.task)

    csv_summary = METRICS_DIR / f"final{suffix}_results.csv"
    summary.to_csv(csv_summary)
    print(f"Saved summary CSV: {csv_summary}")

    md_table = format_markdown(summary, args.task)
    md_path = METRICS_DIR / f"final{suffix}_results.md"
    md_path.write_text(md_table, encoding="utf-8")
    print(f"Saved markdown table: {md_path}")

    print("\n" + md_table)


if __name__ == "__main__":
    main()
