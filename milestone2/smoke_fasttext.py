# DEBUG load each perplexity correlations fastText and score 10 docs
from __future__ import annotations
import json

import fasttext

from fasttext_filters import PC_FASTTEXT_MODELS, ensure_model
from config import doc_path
from _shared_io import iter_docs  # small helper, created below


# fastText needs newlines replaced with spaces
def clean(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()


def smoke_one(name: str, n_docs: int = 10):
    path = ensure_model(name)
    print(f"\n=== {name} ({path.name}) ===")
    model = fasttext.load_model(str(path))
    print(f"  labels = {model.get_labels()}")
    # pick shard 431 (first sampled)
    shard_docs = list(iter_docs(doc_path(431), max_docs=n_docs))
    POS = "__label__include"
    for i, obj in shard_docs:
        text = clean(obj.get("text", "") or "")
        labels, probs = model.predict(text, k=1)
        top_label, top_prob = labels[0], float(probs[0])
        p_include = top_prob if top_label == POS else 1.0 - top_prob
        print(f"  doc {i:>3}: top={top_label} top_p={top_prob:.4f}  p_include={p_include:.4f}  "
              f"(url={obj.get('url','')[:60]!r})")


if __name__ == "__main__":
    for name in PC_FASTTEXT_MODELS:
        smoke_one(name, n_docs=5)
