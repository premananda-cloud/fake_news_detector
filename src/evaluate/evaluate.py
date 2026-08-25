"""
Evaluate a fine-tuned model (saved under models/<name>/) on the held-out
test set. Produces, in evaluate_outputs/<name>/:

    metrics.json               - accuracy/precision/recall/F1/AUC/loss
    classification_report.txt  - per-class sklearn report
    confusion_matrix.png
    roc_curve.png

Usage:
    python -m src.evaluate.evaluate --model bert
    python -m src.evaluate.evaluate --model roberta
"""

import argparse
import json

import matplotlib
matplotlib.use('Agg')  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import DATA_FILES, EVAL_PATHS, MODEL_CONFIGS, TRAIN_CONFIG

LABEL_NAMES = ['fake', 'real']  # label 0 = fake, label 1 = real


def load_model_and_tokenizer(save_dir):
    tokenizer = AutoTokenizer.from_pretrained(str(save_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(save_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def run_inference(model, tokenizer, device, texts, batch_size=64, max_length=128):
    """Returns (predicted_labels, probability_of_class_1) as numpy arrays."""
    all_preds, all_probs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=max_length,
            return_tensors='pt',
        ).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_probs)


def plot_confusion_matrix(y_true, y_pred, out_path, model_label):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
        cbar=False,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_label} — Confusion Matrix (test set)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, out_path, model_label):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_label} — ROC Curve (test set)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), required=True)
    args = parser.parse_args()

    model_key = args.model
    save_dir = MODEL_CONFIGS[model_key]['save_dir']
    paths = EVAL_PATHS[model_key]
    model_label = MODEL_CONFIGS[model_key]['model_name']

    print(f"Loading {model_label} from {save_dir} ...")
    model, tokenizer, device = load_model_and_tokenizer(save_dir)

    print("Loading test set ...")
    test_df = pd.read_csv(DATA_FILES['test'], sep='\t').dropna(subset=['text', 'label'])
    texts = test_df['text'].tolist()
    y_true = test_df['label'].astype(int).to_numpy()

    print(f"Running inference on {len(texts)} test examples ...")
    y_pred, y_prob = run_inference(
        model, tokenizer, device, texts,
        batch_size=TRAIN_CONFIG['eval_batch_size'],
        max_length=TRAIN_CONFIG['max_length'],
    )

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)

    metrics = {
        'model': model_label,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'n_test_samples': len(texts),
    }

    print(json.dumps(metrics, indent=2))
    with open(paths['metrics_json'], 'w') as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4)
    with open(paths['classification_report'], 'w') as f:
        f.write(f"Model: {model_label}\n\n{report}")
    print(report)

    plot_confusion_matrix(y_true, y_pred, paths['confusion_matrix_png'], model_label)
    plot_roc_curve(y_true, y_prob, paths['roc_curve_png'], model_label)

    print(f"Saved evaluation artifacts to {paths['dir']}")


if __name__ == '__main__':
    main()
