"""
Synthetic-data-inspired heuristic features (B2-B5) + readability (C).
Pure CPU, regex/string/textstat. One pass per doc, no models.

extract_synth_features(text) -> dict with SYNTH_FEATURE_COLS keys.
"""
from __future__ import annotations
import re
from collections import Counter

# ---------- sentence splitting ----------
# split on .!? optionally followed by quote/paren, then whitespace + capital OR newline.
# fallback: just count terminal punctuation.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=[A-Z"\'(\[])|\n+')

def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    return parts

# ---------- discourse / reasoning word lists ----------

_DISC_CAUSAL = [
    r"\bbecause\b", r"\btherefore\b", r"\bthus\b", r"\bhence\b",
    r"\bconsequently\b", r"\bas a result\b", r"\bsince\b",
    r"\bso that\b", r"\bdue to\b", r"\bowing to\b",
]
_DISC_CONTRASTIVE = [
    r"\bhowever\b", r"\balthough\b", r"\bnevertheless\b", r"\bdespite\b",
    r"\bwhereas\b", r"\bon the other hand\b",
    r"(?:^|[.!?]\s+|\n)\s*but\b", r"(?:^|[.!?]\s+|\n)\s*yet\b",
    r"\bin contrast\b", r"\bconversely\b",
]
_DISC_ELABORATIVE = [
    r"\bfurthermore\b", r"\bmoreover\b", r"\badditionally\b",
    r"\bspecifically\b", r"\bin particular\b", r"\bfor example\b",
    r"\bfor instance\b", r"\bnamely\b", r"\bthat is\b", r"\bin other words\b",
]
_DISC_TEMPORAL = [
    r"\bfirst\b", r"\bthen\b", r"\bnext\b", r"\bfinally\b",
    r"\bsubsequently\b", r"\bmeanwhile\b", r"\bafterward\b",
    r"\bpreviously\b", r"\bbefore this\b", r"\bafter this\b",
]
_EXPLANATION = [
    r"\bbecause\b", r"\bsince\b", r"\bthis means\b", r"\bthe reason is\b",
    r"\bthis is due to\b", r"\bwhich shows that\b", r"\bwhich means\b",
    r"\bwhich indicates\b", r"\bthis implies\b", r"\bthis suggests\b",
]

def _compile_union(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(patterns), flags=re.IGNORECASE | re.MULTILINE)

_RE_CAUSAL      = _compile_union(_DISC_CAUSAL)
_RE_CONTRASTIVE = _compile_union(_DISC_CONTRASTIVE)
_RE_ELABORATIVE = _compile_union(_DISC_ELABORATIVE)
_RE_TEMPORAL    = _compile_union(_DISC_TEMPORAL)
_RE_EXPLANATION = _compile_union(_EXPLANATION)

# ---------- self-containedness ----------
_PRONOUNS = r"\b(?:he|she|it|they|this|that|these|those|his|her|its|their)\b"
_FIRST_PERSON = r"\b(?:I|me|my|mine|we|us|our)\b"
_HEDGES = r"\b(?:might|could|perhaps|maybe|seems|arguably|possibly|apparently|likely|unlikely|probably)\b"
_RE_PRONOUN = re.compile(_PRONOUNS, flags=re.IGNORECASE)
_RE_FIRST = re.compile(_FIRST_PERSON)   # case-sensitive on purpose (I, me)
_RE_HEDGE = re.compile(_HEDGES, flags=re.IGNORECASE)

# ---------- structural ----------
_CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_CODE_TOKEN_RE = re.compile(
    r"\b(?:def|class|import|function|var|SELECT|CREATE)\b|[{}]|</?(?:div|p|span|a|li|ul|ol|h[1-6])\b"
)
_INDENT_CODE_RE = re.compile(r"^( {4,}|\t+).+", flags=re.MULTILINE)
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", flags=re.MULTILINE)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_TITLE_CASE_RE = re.compile(r"^(?:[A-Z][\w'-]*\b\s*){1,}$")  # short, title-like

# ---------- factual ----------
_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
# proper-noun heuristic: capitalized word NOT at start of sentence and NOT after newline.
# Approximate via tokenizing words and tracking sentence-start positions.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# ---------- column manifest ----------

SYNTH_FEATURE_COLS = [
    # B2 discourse / reasoning
    "discourse_causal", "discourse_contrastive", "discourse_elaborative",
    "discourse_temporal", "explanation_density", "question_density",
    # B3 self-containedness
    "pronoun_density", "first_person_fraction", "hedge_density",
    # B4 structural diversity
    "hapax_legomena_ratio", "header_density", "code_fraction",
    "avg_paragraph_length", "list_density",
    # B5 factual density (skip content_word_ratio = 1 - stopword_fraction)
    "numeric_density", "proper_noun_density",
    # C readability
    "flesch_kincaid_grade", "flesch_reading_ease",
]


def _proper_noun_count(text: str) -> int:
    """Capitalized words NOT at sentence start. Cheap heuristic for proper nouns."""
    # split into sentences once; for each, skip first word.
    sents = _split_sentences(text)
    n = 0
    for s in sents:
        words = _WORD_RE.findall(s)
        for w in words[1:]:
            if w[0].isupper():
                n += 1
    return n


def _hapax_ratio(words: list[str]) -> float:
    if not words:
        return float("nan")
    c = Counter(words)
    once = sum(1 for v in c.values() if v == 1)
    return once / len(words)


def _code_fraction(text: str, char_count: int) -> float:
    if char_count == 0:
        return float("nan")
    covered = 0
    for m in _CODE_FENCE_RE.finditer(text):
        covered += m.end() - m.start()
    # indent-based: only count if the line also contains code-like tokens
    for m in _INDENT_CODE_RE.finditer(text):
        ln = m.group(0)
        if _CODE_TOKEN_RE.search(ln):
            covered += len(ln)
    # cap at char_count
    return min(covered, char_count) / char_count


def _header_density(text: str) -> float:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return float("nan")
    n_hdr = 0
    for ln in lines:
        s = ln.strip()
        if len(s) >= 80:
            continue
        if s[-1:] in ".?!":
            continue
        if s.startswith("#") or _TITLE_CASE_RE.match(s):
            n_hdr += 1
    return n_hdr / len(lines)


def _avg_paragraph_length(text: str) -> float:
    paras = [p for p in _PARA_SPLIT_RE.split(text) if p.strip()]
    if not paras:
        return float("nan")
    return sum(len(p.split()) for p in paras) / len(paras)


def _list_density(text: str) -> float:
    lines = text.split("\n")
    if not lines:
        return float("nan")
    n = sum(1 for ln in lines if _LIST_LINE_RE.match(ln))
    return n / len(lines)


def _readability(text: str) -> tuple[float, float]:
    """Truncate to first 5000 words to keep textstat fast."""
    import textstat
    if not text or not text.strip():
        return float("nan"), float("nan")
    words = text.split()
    if len(words) > 5000:
        text = " ".join(words[:5000])
    try:
        fk = float(textstat.flesch_kincaid_grade(text))
    except Exception:
        fk = float("nan")
    try:
        fre = float(textstat.flesch_reading_ease(text))
    except Exception:
        fre = float("nan")
    return fk, fre


def extract_synth_features(text: str) -> dict:
    out: dict[str, float] = {}
    if not text:
        for k in SYNTH_FEATURE_COLS:
            out[k] = float("nan")
        return out

    sents = _split_sentences(text)
    n_sent = max(len(sents), 1)
    words = text.split()
    n_words = len(words)
    char_count = len(text)
    lower_words = [w.lower().strip(".,;:!?\"'()[]{}") for w in words]
    lower_words = [w for w in lower_words if w]

    # B2 discourse densities (matches per sentence)
    out["discourse_causal"]      = len(_RE_CAUSAL.findall(text))      / n_sent
    out["discourse_contrastive"] = len(_RE_CONTRASTIVE.findall(text)) / n_sent
    out["discourse_elaborative"] = len(_RE_ELABORATIVE.findall(text)) / n_sent
    out["discourse_temporal"]    = len(_RE_TEMPORAL.findall(text))    / n_sent
    out["explanation_density"]   = len(_RE_EXPLANATION.findall(text)) / n_sent
    if sents:
        out["question_density"] = sum(1 for s in sents if s.rstrip().endswith("?")) / n_sent
    else:
        out["question_density"] = float("nan")

    # B3 self-containedness
    out["pronoun_density"] = len(_RE_PRONOUN.findall(text)) / n_sent
    if sents:
        out["first_person_fraction"] = sum(1 for s in sents if _RE_FIRST.search(s)) / n_sent
    else:
        out["first_person_fraction"] = float("nan")
    out["hedge_density"] = len(_RE_HEDGE.findall(text)) / n_sent

    # B4 structural
    out["hapax_legomena_ratio"] = _hapax_ratio(lower_words)
    out["header_density"]       = _header_density(text)
    out["code_fraction"]        = _code_fraction(text, char_count)
    out["avg_paragraph_length"] = _avg_paragraph_length(text)
    out["list_density"]         = _list_density(text)

    # B5 factual
    out["numeric_density"]      = (len(_NUMERIC_RE.findall(text)) / max(n_words, 1)) * 1000.0
    out["proper_noun_density"]  = (_proper_noun_count(text) / max(n_words, 1)) * 1000.0

    # C readability
    fk, fre = _readability(text)
    out["flesch_kincaid_grade"] = fk
    out["flesch_reading_ease"]  = fre

    return out
