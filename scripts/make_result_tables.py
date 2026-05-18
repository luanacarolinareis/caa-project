"""Aggregate per-seed evaluation JSONs into summary tables.

Usage:
    python scripts/make_result_tables.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

METRICS_DIR = Path("results/metrics")
MODELS = ["custom_cnn", "resnet18", "densenet121"]
SEEDS = [0, 1, 2]
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "auroc", "inference_time_ms_per_image"]


def load_seed_metrics() -> list[dict]:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            path = METRICS_DIR / f"{model}_seed{seed}.json"
            if not path.exists():
                print(f"WARNING: missing {path}")
                continue
            with path.open() as f:
                data = json.load(f)
            row = {"model": model, "seed": seed}
            for m in METRICS:
                row[m] = data.get(m)
            rows.append(row)
    return rows


def build_summary(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric = [m for m in METRICS if m != "inference_time_ms_per_image"] + ["inference_time_ms_per_image"]
    agg = (
        df.groupby("model")[numeric]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reindex(MODELS)


def format_markdown(summary: pd.DataFrame) -> str:
    display_metrics = ["accuracy", "precision", "recall", "specificity", "f1", "auroc", "inference_time_ms_per_image"]
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
        cols.append(header_map[m])

    sep = ["---"] + ["---"] * len(display_metrics)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(sep) + " |"]

    for model in MODELS:
        row_vals = [model]
        for m in display_metrics:
            mean = summary.loc[model, f"{m}_mean"]
            std = summary.loc[model, f"{m}_std"]
            row_vals.append(f"{mean:.4f} ± {std:.4f}")
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def main() -> None:
    rows = load_seed_metrics()
    if not rows:
        print("No metric files found. Run evaluate_model.py first.")
        return

    df_all = pd.DataFrame(rows)
    csv_per_seed = METRICS_DIR / "all_seeds_results.csv"
    df_all.to_csv(csv_per_seed, index=False)
    print(f"Saved per-seed CSV: {csv_per_seed}")

    summary = build_summary(rows)

    csv_summary = METRICS_DIR / "final_results.csv"
    summary.to_csv(csv_summary)
    print(f"Saved summary CSV: {csv_summary}")

    md_table = format_markdown(summary)
    md_path = METRICS_DIR / "final_results.md"
    md_path.write_text(md_table, encoding="utf-8")
    print(f"Saved markdown table: {md_path}")

    print("\n" + md_table)


if __name__ == "__main__":
    main()
