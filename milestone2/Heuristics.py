# pure per-doc feature extraction
# gets 1. URL
# 2. 16 standard heuristics from Gopher paper

from __future__ import annotations
import math
import re
import string
from collections import Counter
from urllib.parse import urlparse

import tldextract
import nltk

try:
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words("english"))

# tldextract with a cached suffix list 
_TLD = tldextract.TLDExtract(cache_dir=str(__import__("pathlib").Path.home() / ".cache" / "tldextract"))

_PUNCT = set(string.punctuation)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s")
_TERMINAL_RE = re.compile(r'[.?!"]\s*$')

FEATURE_COLS = [
    "char_count", "word_count", "mean_word_length",
    "frac_alpha", "frac_digit", "frac_punctuation", "frac_uppercase",
    "frac_lines_terminal_punct", "frac_lines_bullet",
    "type_token_ratio", "stopword_fraction",
    "ngram_rep_2", "ngram_rep_3", "ngram_rep_4",
    "num_lines", "mean_line_length",
]

URL_COLS = ["url", "url_netloc", "url_registered_domain"]


def _parse_url(url: str) -> tuple[str, str]:
    if not url:
        return "", ""
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        netloc = ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    try:
        ex = _TLD(url)
        registered = ex.registered_domain.lower() if ex.registered_domain else netloc
    except Exception:
        registered = netloc
    return netloc, registered


def _ngram_repetition(words: list[str], n: int) -> float:
    """Fraction of words covered by n-grams that appear >= 2 times."""
    if len(words) < n:
        return float("nan")
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    # count tokens covered by repeated n-grams
    covered = sum(cnt * n for ng, cnt in counts.items() if cnt >= 2)
    return covered / len(words)


def extract_features(text: str, url: str) -> dict:
    out = {"url": url or ""}
    netloc, registered = _parse_url(url or "")
    out["url_netloc"] = netloc
    out["url_registered_domain"] = registered

    char_count = len(text)
    words = text.split()
    word_count = len(words)
    out["char_count"] = char_count
    out["word_count"] = word_count

    if char_count == 0 or word_count == 0:
        for k in FEATURE_COLS:
            if k not in ("char_count", "word_count"):
                out[k] = float("nan")
        out["num_lines"] = text.count("\n") + 1 if char_count > 0 else 0
        return out

    # char-level fractions
    n_alpha = n_digit = n_punct = n_upper = 0
    n_alpha_any_case = 0
    for ch in text:
        if ch.isalpha():
            n_alpha += 1
            n_alpha_any_case += 1
            if ch.isupper():
                n_upper += 1
        elif ch.isdigit():
            n_digit += 1
        elif ch in _PUNCT:
            n_punct += 1

    out["frac_alpha"] = n_alpha / char_count
    out["frac_digit"] = n_digit / char_count
    out["frac_punctuation"] = n_punct / char_count
    out["frac_uppercase"] = (n_upper / n_alpha_any_case) if n_alpha_any_case else float("nan")
    out["mean_word_length"] = sum(len(w) for w in words) / word_count

    # line-level
    lines = text.split("\n")
    num_lines = len(lines)
    out["num_lines"] = num_lines
    nonempty = [ln for ln in lines if ln.strip()]
    if nonempty:
        out["frac_lines_terminal_punct"] = sum(
            1 for ln in nonempty if _TERMINAL_RE.search(ln)
        ) / len(nonempty)
        out["frac_lines_bullet"] = sum(
            1 for ln in nonempty if _BULLET_RE.match(ln)
        ) / len(nonempty)
    else:
        out["frac_lines_terminal_punct"] = float("nan")
        out["frac_lines_bullet"] = float("nan")
    out["mean_line_length"] = sum(len(ln) for ln in lines) / num_lines

    # type/token and stopwords (case-insensitive)
    lower_words = [w.lower() for w in words]
    out["type_token_ratio"] = len(set(lower_words)) / word_count
    out["stopword_fraction"] = sum(1 for w in lower_words if w in _STOPWORDS) / word_count

    # n-gram repetition
    out["ngram_rep_2"] = _ngram_repetition(lower_words, 2)
    out["ngram_rep_3"] = _ngram_repetition(lower_words, 3)
    out["ngram_rep_4"] = _ngram_repetition(lower_words, 4)

    return out
