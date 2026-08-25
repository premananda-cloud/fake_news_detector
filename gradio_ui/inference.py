"""
inference.py
-------------
Loads the fine-tuned BERT fake-news classifier and exposes:
  - predict(text)        -> label / confidence / probabilities
  - get_attention(text)   -> [{token, score}, ...] for the token cloud

Model resolution order (first that works wins):
  1. A local directory (default: <project_root>/model) that already
     contains a full HF-style checkpoint (config.json + weights).
  2. That same local directory containing ONLY `model.safetensors`
     (your current SPST output) — we build a bert-base-uncased
     classification head and load the safetensors weights into it.
  3. A HF Hub repo id (fallback), so this keeps working even before
     the local checkpoint is wired up.

Set FAKE_NEWS_MODEL_DIR / FAKE_NEWS_HUB_ID env vars to override.
"""

import os
from pathlib import Path

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, BertConfig

BASE_MODEL_NAME = "bert-base-uncased"
DEFAULT_HUB_FALLBACK = "kiliez/Bert_fn_detector"

# gradio_ui/ -> project root -> model/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "model"

MODEL_DIR = Path(os.environ.get("FAKE_NEWS_MODEL_DIR", DEFAULT_MODEL_DIR))
HUB_FALLBACK = os.environ.get("FAKE_NEWS_HUB_ID", DEFAULT_HUB_FALLBACK)

# label convention used throughout this project: 0 = fake, 1 = real
LABEL_FAKE, LABEL_REAL = 0, 1


class FakeNewsDetectorModel:
    """Thin wrapper: one load, reused across every request."""

    def __init__(self, max_length: int = 512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer, self.model, self.source = self._load()
        self.model.to(self.device)
        self.model.eval()

    # -- loading -----------------------------------------------------
    def _load(self):
        has_config = (MODEL_DIR / "config.json").exists()
        has_weights = (MODEL_DIR / "model.safetensors").exists() or (
            MODEL_DIR / "pytorch_model.bin"
        ).exists()

        # Case 1: fully-formed local HF checkpoint
        if has_config and has_weights:
            tokenizer = self._load_tokenizer(MODEL_DIR)
            model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR))
            return tokenizer, model, f"local:{MODEL_DIR} (full checkpoint)"

        # Case 2: bare weights file only (current SPST output) — build
        # the architecture fresh and load the state dict into it.
        if has_weights:
            tokenizer = self._load_tokenizer(MODEL_DIR)
            config = BertConfig.from_pretrained(BASE_MODEL_NAME, num_labels=2)
            model = BertForSequenceClassification(config)
            state_dict = self._read_weights(MODEL_DIR)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[inference] warning — missing keys when loading weights: {missing}")
            if unexpected:
                print(f"[inference] warning — unexpected keys when loading weights: {unexpected}")
            return tokenizer, model, f"local:{MODEL_DIR} (weights only, base config)"

        # Case 3: fall back to the published Hub checkpoint
        tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(
            HUB_FALLBACK, num_labels=2, ignore_mismatched_sizes=True
        )
        return tokenizer, model, f"hub:{HUB_FALLBACK} (local checkpoint not found)"

    @staticmethod
    def _load_tokenizer(model_dir: Path):
        # If a tokenizer was saved next to the weights, use it; otherwise
        # the base BERT vocab is identical to what SPST fine-tuning used.
        if (model_dir / "vocab.txt").exists() or (model_dir / "tokenizer.json").exists():
            return BertTokenizerFast.from_pretrained(str(model_dir))
        return BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)

    @staticmethod
    def _read_weights(model_dir: Path) -> dict:
        safet_path = model_dir / "model.safetensors"
        if safet_path.exists():
            from safetensors.torch import load_file
            return load_file(str(safet_path))
        bin_path = model_dir / "pytorch_model.bin"
        return torch.load(str(bin_path), map_location="cpu")

    # -- inference -----------------------------------------------------
    def _tokenize(self, text: str):
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, text: str) -> dict:
        inputs = self._tokenize(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()

        p_fake = probs[0][LABEL_FAKE].item()
        p_real = probs[0][LABEL_REAL].item()
        return {
            "is_fake": pred == LABEL_FAKE,
            "confidence": max(p_fake, p_real),
            "probability": {"fake": p_fake, "real": p_real},
        }

    def get_attention(self, text: str, top_n: int = 20, last_n_layers: int = 4) -> list[dict]:
        """
        Average CLS-token attention over the last `last_n_layers` layers
        (and all heads), map back to word-piece tokens, normalise to
        [0, 1] relative to the top token. Special tokens excluded.
        """
        inputs = self._tokenize(text)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        if not outputs.attentions:
            return []

        # outputs.attentions: tuple[num_layers] of (batch, heads, seq, seq)
        layers = outputs.attentions[-last_n_layers:]
        stacked = torch.stack(layers, dim=0)[:, 0, :, :, :]  # (layers, heads, seq, seq)
        avg_attention = stacked.mean(dim=(0, 1))              # (seq, seq)
        cls_attention = avg_attention[0]                       # attention FROM [CLS]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
        scores = cls_attention.cpu().tolist()

        special = {"[CLS]", "[SEP]", "[PAD]"}
        pairs = [(tok, s) for tok, s in zip(tokens, scores) if tok not in special]
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:top_n]

        if not pairs:
            return []
        max_score = pairs[0][1] or 1.0
        return [{"token": tok, "score": round(s / max_score, 4)} for tok, s in pairs]

    def analyze(self, text: str) -> dict:
        """Prediction + attention only — signals/explanation live in explain.py."""
        result = self.predict(text)
        result["attention"] = self.get_attention(text)
        return result


_detector: FakeNewsDetectorModel | None = None


def get_detector() -> FakeNewsDetectorModel:
    """Lazy singleton — model loads once, on first use."""
    global _detector
    if _detector is None:
        _detector = FakeNewsDetectorModel()
        print(f"[inference] model loaded from {_detector.source} on {_detector.device}")
    return _detector
