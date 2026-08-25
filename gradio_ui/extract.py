"""
extract.py
----------
Turns a URL or an uploaded file into plain article text.
Synchronous on purpose — Gradio callbacks run these directly.
"""

import io
import os

from newspaper import Article
from pdfminer.high_level import extract_text as pdf_extract_text


class ExtractionError(ValueError):
    """Raised for any user-facing extraction failure (bad URL, empty file, etc.)."""


def fetch_url_text(url: str) -> str:
    """Download + parse an article body from a URL via newspaper3k."""
    if not url or not url.strip():
        raise ExtractionError("Please enter a URL.")
    try:
        article = Article(url.strip())
        article.download()
        article.parse()
    except Exception as e:
        raise ExtractionError(f"Failed to fetch URL: {e}") from e

    text = (article.text or "").strip()
    if not text:
        raise ExtractionError("Could not extract article text from that URL.")
    return text


def extract_file_text(file_path: str) -> tuple[str, str]:
    """Extract text from an uploaded .txt or .pdf file (path on disk)."""
    if not file_path:
        raise ExtractionError("Please select a file to upload.")

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".pdf":
            with open(file_path, "rb") as f:
                text = pdf_extract_text(io.BytesIO(f.read())).strip()
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        else:
            raise ExtractionError("Only .txt and .pdf files are supported.")
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"Failed to process file: {e}") from e

    if not text:
        raise ExtractionError("Could not extract text from that file.")
    return text, filename


def validate_text(text: str) -> str:
    """Shared minimum-length guard used for all three input modes."""
    if not text or len(text.strip()) < 10:
        raise ExtractionError("Text is too short to analyse (minimum ~10 characters).")
    return text.strip()
