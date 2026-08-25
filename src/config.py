"""
Configuration for standard full fine-tuning (no SPST / progressive unfreezing).

Usage:
    python -m src.training.train --model bert
    python -m src.training.train --model roberta
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATASETS_DIR = BASE_DIR / 'datasets' / 'datasets' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

DATA_FILES = {
    'train': DATASETS_DIR / 'unified_train.tsv',
    'val': DATASETS_DIR / 'unified_val.tsv',
    'test': DATASETS_DIR / 'unified_test.tsv',
}

# ============================================================================
# MODEL CHOICES
# ============================================================================
# Pick via --model {bert,roberta} on the CLI. Both trained the same way —
# standard full fine-tuning, all layers trainable from step 1.
MODEL_CONFIGS = {
    'bert': {
        'model_name': 'bert-base-uncased',
        'save_dir': MODELS_DIR / 'bert',
        'log_dir': LOGS_DIR / 'bert',
    },
    'roberta': {
        'model_name': 'roberta-base',
        'save_dir': MODELS_DIR / 'roberta',
        'log_dir': LOGS_DIR / 'roberta',
    },
}

NUM_LABELS = 2

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# RTX 4070 (12GB VRAM) comfortably fits batch_size=32 at max_length=128 for
# base models with fp16 — no gradient accumulation or freezing tricks needed.
TRAIN_CONFIG = {
    'max_length': 128,
    'batch_size': 32,
    'eval_batch_size': 64,
    'epochs': 4,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'fp16': True,
    'random_seed': 42,
    'early_stopping_patience': 2,  # stops if val F1 doesn't improve for N evals
    'logging_steps': 50,
    'eval_strategy': 'epoch',
    'save_strategy': 'epoch',
}

for d in [MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for cfg in MODEL_CONFIGS.values():
    cfg['save_dir'].mkdir(parents=True, exist_ok=True)
    cfg['log_dir'].mkdir(parents=True, exist_ok=True)

# ============================================================================
# EVALUATION OUTPUT PATHS
# ============================================================================
EVAL_OUTPUT_DIR = BASE_DIR / 'evaluate_outputs'
EVAL_PATHS = {
    name: {
        'dir': EVAL_OUTPUT_DIR / name,
        'metrics_json': EVAL_OUTPUT_DIR / name / 'metrics.json',
        'classification_report': EVAL_OUTPUT_DIR / name / 'classification_report.txt',
        'confusion_matrix_png': EVAL_OUTPUT_DIR / name / 'confusion_matrix.png',
        'roc_curve_png': EVAL_OUTPUT_DIR / name / 'roc_curve.png',
    }
    for name in MODEL_CONFIGS
}
COMPARISON_DIR = EVAL_OUTPUT_DIR / 'comparison'

for p in EVAL_PATHS.values():
    p['dir'].mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
