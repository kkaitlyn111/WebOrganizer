# hand-computed unit tests for heuristics.extract_features
from __future__ import annotations
import math

from heuristics import extract_features, FEATURE_COLS


def approx(a, b, tol=1e-6):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) < tol


def test_simple_sentence():
    text = "The cat sat on the mat."
    r = extract_features(text, "https://www.example.com/foo")
    # chars: 23, words: 6
    assert r["char_count"] == 23
    assert r["word_count"] == 6
    assert approx(r["mean_word_length"], (3 + 3 + 3 + 2 + 3 + 4) / 6)  # "mat." has trailing '.'? "mat." len=4
    assert r["url_netloc"] == "example.com"
    assert r["url_registered_domain"] == "example.com"
    # lowercase words for ttr: {the, cat, sat, on, mat.} -> "the" appears twice
    assert approx(r["type_token_ratio"], 5 / 6)
    # num_lines = 1
    assert r["num_lines"] == 1
    # terminal punct: sentence ends with "."
    assert approx(r["frac_lines_terminal_punct"], 1.0)
    # bullet lines: 0
    assert approx(r["frac_lines_bullet"], 0.0)
    # stopwords among {the, cat, sat, on, the, mat.}: "the","on","the" -> 3
    assert approx(r["stopword_fraction"], 3 / 6)


def test_empty():
    r = extract_features("", "http://foo.bar")
    assert r["char_count"] == 0
    assert r["word_count"] == 0
    for k in FEATURE_COLS:
        if k in ("char_count", "word_count", "num_lines"):
            continue
        assert math.isnan(r[k]), f"{k} should be NaN on empty doc"


def test_bullets_and_lines():
    text = "- first\n- second\n* third\n1. fourth\nplain"
    r = extract_features(text, "")
    assert r["num_lines"] == 5
    # 4 of 5 non-empty lines are bullets
    assert approx(r["frac_lines_bullet"], 4 / 5)


def test_all_caps_and_digits():
    text = "ABC 123 XYZ"
    r = extract_features(text, "")
    # chars=11; alpha=6 (A,B,C,X,Y,Z); digit=3; uppercase of alpha=6/6
    assert approx(r["frac_alpha"], 6 / 11)
    assert approx(r["frac_digit"], 3 / 11)
    assert approx(r["frac_uppercase"], 1.0)


def test_bigram_repetition():
    # "a b a b a b" -> bigrams: (a,b) x 3, (b,a) x 2. Both repeat.
    # covered = 3*2 + 2*2 = 10, words=6 -> 10/6
    text = "a b a b a b"
    r = extract_features(text, "")
    assert approx(r["ngram_rep_2"], 10 / 6)
    # trigrams: (a,b,a) x 2, (b,a,b) x 2. Both repeat.
    # covered = 2*3 + 2*3 = 12, words=6
    assert approx(r["ngram_rep_3"], 12 / 6)


def test_url_registered_domain():
    r = extract_features("hi", "https://blog.sub.example.co.uk/path?x=1")
    assert r["url_netloc"] == "blog.sub.example.co.uk"
    assert r["url_registered_domain"] == "example.co.uk"


def run():
    for fn in [
        test_simple_sentence, test_empty, test_bullets_and_lines,
        test_all_caps_and_digits, test_bigram_repetition,
        test_url_registered_domain,
    ]:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            raise


if __name__ == "__main__":
    run()
    print("\nAll heuristic unit tests passed.")
