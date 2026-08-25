# Dataset & Model Download Guide

This guide provides commands to download the **dataset** and the **pre-trained model weights** used in this project.  
Both are hosted on Hugging Face Hub and can be retrieved with the `huggingface-cli` tool.

---

## Prerequisites

Make sure the Hugging Face CLI is installed and logged in (if required for private repos, but these are public):

```bash
pip install huggingface-hub
```

---

## 1. Download the Dataset

The dataset is available at [`kiliez/BERT_SPST_DATA`](https://huggingface.co/datasets/kiliez/BERT_SPST_DATA).  
It contains the unified train/val/test splits in TSV format.

**Command** (run from the project root):
```bash
hf download kiliez/BERT_SPST_DATA --repo-type dataset --local-dir datasets
```

> **Note:** The `--repo-type dataset` flag is required because this is a dataset repository.  
> The contents will be placed inside `datasets/`, matching the expected project structure (`datasets/datasets/processed/...`).

After download, you should have:
```
datasets/datasets/processed/unified_train.tsv
datasets/datasets/processed/unified_val.tsv
datasets/datasets/processed/unified_test.tsv
```
and the raw source files under `datasets/datasets/raw/`.

---

## 2. Download the Pre-trained Models

The trained models (BERT and RoBERTa) are stored at [`kiliez/fake_news_detector`](https://huggingface.co/kiliez/fake_news_detector).  
Each model directory includes the configuration, tokenizer, and model weights.

**Command** (run from the project root):
```bash
hf download kiliez/fake_news_detector --local-dir models
```

> This will create a `models/` folder containing `bert/` and `roberta/` subdirectories with all necessary files.

After download, you will have:
```
models/bert/config.json
models/bert/model.safetensors
models/bert/tokenizer.json
...
models/roberta/config.json
models/roberta/model.safetensors
...
```

---

## Optional: Verify the Downloads

- Check dataset files exist:
  ```bash
  ls datasets/datasets/processed/
  ```
- Check model files exist:
  ```bash
  ls models/bert/
  ls models/roberta/
  ```

---

## Training Your Own Models

If you prefer to train from scratch instead of using the pre-trained weights, the code in `src/training/train.py` is fully ready.  
Just ensure the dataset is downloaded (as above) and run:

```bash
python -m src.training.train --model bert
python -m src.training.train --model roberta
```

---

## Notes

- The dataset is about **~100 MB** and the models are **~1 GB** each (download times may vary).
- Both repositories are **public** – no authentication required.
- If you encounter any issues, refer to the main [README](README.md) for detailed project setup.
