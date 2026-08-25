"""
Compares all models that have already been evaluated (i.e. have a
metrics.json under evaluate_outputs/<name>/), producing a side-by-side bar
chart and a CSV summary table in evaluate_outputs/comparison/.

Run this AFTER evaluating each model individually:
    python -m src.evaluate.evaluate --model bert
    python -m src.evaluate.evaluate --model roberta
    python -m src.evaluate.compare
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import COMPARISON_DIR, EVAL_PATHS, MODEL_CONFIGS

METRIC_KEYS = ['accuracy', 'precision', 'recall', 'f1', 'auc']


def load_available_metrics():
    rows = []
    missing = []
    for model_key in MODEL_CONFIGS:
        metrics_path = EVAL_PATHS[model_key]['metrics_json']
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            m['model_key'] = model_key
            rows.append(m)
        else:
            missing.append(model_key)
    return rows, missing


def plot_comparison(rows, out_path):
    labels = [r['model_key'] for r in rows]
    x = np.arange(len(METRIC_KEYS))
    width = 0.8 / len(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, r in enumerate(rows):
        values = [r[k] for k in METRIC_KEYS]
        offset = (i - (len(rows) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=labels[i])
        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in METRIC_KEYS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Test Set Metrics')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    rows, missing = load_available_metrics()

    if missing:
        print(f"Note: no metrics.json found yet for: {missing} "
              f"(run `python -m src.evaluate.evaluate --model {missing[0]}` first)")

    if len(rows) < 1:
        print("No evaluated models found. Nothing to compare.")
        return

    df = pd.DataFrame(rows)[['model_key', 'model'] + METRIC_KEYS + ['n_test_samples']]
    csv_path = COMPARISON_DIR / 'metrics_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved comparison table to {csv_path}")

    if len(rows) >= 2:
        png_path = COMPARISON_DIR / 'metrics_comparison.png'
        plot_comparison(rows, png_path)
        print(f"Saved comparison chart to {png_path}")
    else:
        print("Only one model evaluated so far — skipping bar chart "
              "(evaluate a second model to get a comparison plot).")


if __name__ == '__main__':
    main()
