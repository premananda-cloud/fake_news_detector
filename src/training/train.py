"""
Standard full fine-tuning for fake news detection (no SPST / layer freezing).

Usage:
    python -m src.training.train --model bert
    python -m src.training.train --model roberta

Saves the best checkpoint (by validation F1) to models/<model>/, along with
tokenizer files and a results.json summary evaluated on the held-out test set.
"""

import argparse
import inspect
import json
import math

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Some transformers versions use `eval_strategy`, older ones `evaluation_strategy`.
# Detect which this install supports rather than hardcoding one.
_TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)
_EVAL_STRATEGY_KEY = 'eval_strategy' if 'eval_strategy' in _TA_PARAMS else 'evaluation_strategy'

from src.config import MODEL_CONFIGS, NUM_LABELS, TRAIN_CONFIG
from src.training.dataset import get_tokenized_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float('nan')  # only one class present in a batch, etc.

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), required=True,
                         help="Which model to fine-tune: 'bert' or 'roberta'")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    model_name = cfg['model_name']
    save_dir = cfg['save_dir']
    log_dir = cfg['log_dir']

    set_seed(TRAIN_CONFIG['random_seed'])

    print(f"Loading and tokenizing dataset for {model_name} ...")
    dataset, tokenizer = get_tokenized_dataset(model_name)

    print(f"Loading {model_name} for sequence classification ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )

    # Compute warmup_steps manually — works regardless of whether this
    # transformers version supports warmup_ratio directly.
    steps_per_epoch = math.ceil(len(dataset['train']) / TRAIN_CONFIG['batch_size'])
    total_steps = steps_per_epoch * TRAIN_CONFIG['epochs']
    warmup_steps = int(total_steps * TRAIN_CONFIG['warmup_ratio'])

    # NOTE: transformers v5.0 removed several TrainingArguments params
    # without deprecation, including logging_dir, warmup_ratio, and
    # overwrite_output_dir. We avoid all of those here. logging_steps still
    # controls console/log frequency; with report_to='none' there's no
    # separate TensorBoard dir needed anyway.
    ta_kwargs = dict(
        output_dir=str(log_dir / 'checkpoints'),
        save_strategy=TRAIN_CONFIG['save_strategy'],
        learning_rate=TRAIN_CONFIG['learning_rate'],
        per_device_train_batch_size=TRAIN_CONFIG['batch_size'],
        per_device_eval_batch_size=TRAIN_CONFIG['eval_batch_size'],
        num_train_epochs=TRAIN_CONFIG['epochs'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        warmup_steps=warmup_steps,
        max_grad_norm=TRAIN_CONFIG['max_grad_norm'],
        fp16=TRAIN_CONFIG['fp16'] and torch.cuda.is_available(),
        logging_steps=TRAIN_CONFIG['logging_steps'],
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        report_to='none',
    )
    ta_kwargs[_EVAL_STRATEGY_KEY] = TRAIN_CONFIG['eval_strategy']

    # Defensively drop any kwarg this install's TrainingArguments doesn't
    # actually accept, rather than trading one-error-at-a-time round trips.
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in _TA_PARAMS}

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=TRAIN_CONFIG['early_stopping_patience']
        )],
    )

    print("Starting training ...")
    trainer.train()

    print("Evaluating on held-out test set ...")
    test_metrics = trainer.evaluate(dataset['test'], metric_key_prefix='test')
    print(json.dumps(test_metrics, indent=2))

    print(f"Saving best model + tokenizer to {save_dir} ...")
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    with open(save_dir / 'results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Done. Model saved to: {save_dir}")


if __name__ == '__main__':
    main()
