"""Text preprocessing for inference.

Self-contained copy of the training-time ``preprocess`` (originally
``preprocessing/clean.py``) so the deployed ``app/`` folder has no dependency
on modules outside itself. It MUST stay identical to training so the dashboard
mirrors the fine-tuned models exactly.
"""

from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords


def _ensure_stopwords() -> None:
    """Download the NLTK stop-words corpus on first use."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


_ensure_stopwords()

DOMAIN_STOPS = {"cambodia", "cambodian", "said", "told", "according"}
STOPS = set(stopwords.words("english")) | DOMAIN_STOPS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
NUM_RE = re.compile(r"\d+")
HTML_RE = re.compile(r"<[^>]+>")
SPLIT_RE = re.compile(r"[-/:]")


def preprocess(text: str) -> str:
    """Clean a raw article body into a tokenized, lower-cased string.

    Pipeline (in order):
      1. Lower-case the text
      2. Strip HTML tags, URLs, emails, digits, and ``- / :`` separators
      3. Drop tokens that aren't purely alphabetic or that match the English /
         domain stop-word set
    """
    if not text:
        return ""
    t = text.lower()
    t = HTML_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    t = NUM_RE.sub(" ", t)
    t = SPLIT_RE.sub(" ", t)
    tokens = [w for w in t.split() if w.isalpha() and w not in STOPS]
    return " ".join(tokens)
