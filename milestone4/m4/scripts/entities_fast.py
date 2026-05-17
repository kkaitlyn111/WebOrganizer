"""Faster entity extraction using spaCy nlp.pipe with batch_size and n_process.

Usage:
    python entities_fast.py --shard 431
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from config import M2_DATA, REPO_ROOT  # noqa

import importlib.util as _ilu
def _load(name, p):
    spec = _ilu.spec_from_file_location(name, str(p))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m); return m
_m2cfg = _load("m2cfg", REPO_ROOT / "milestone2" / "config.py")
_m2io = _load("m2io", REPO_ROOT / "milestone2" / "_shared_io.py")
shard_stem = _m2cfg.shard_stem
doc_path = _m2cfg.doc_path
iter_docs = _m2io.iter_docs

OUT_DIR = M2_DATA / "entities"
ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "NORP", "FAC", "DATE", "QUANTITY",
}
MAX_CHARS = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = OUT_DIR / f"{shard_stem(args.shard)}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"already exists: {out_path}")
        return

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    t0 = time.time()
    texts = []
    word_counts = []
    for _i, obj in iter_docs(doc_path(args.shard)):
        text = (obj.get("text", "") or "")[:MAX_CHARS]
        word_counts.append(len(text.split()))
        texts.append(text)
    n_docs = len(texts)
    print(f"  loaded {n_docs} docs in {time.time()-t0:.1f}s, running spaCy...")

    t1 = time.time()
    dens = np.zeros(n_docs, dtype=np.float32)
    divr = np.zeros(n_docs, dtype=np.float32)
    persent = np.zeros(n_docs, dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=args.batch_size)):
        ents = [e for e in doc.ents if e.label_ in ENTITY_TYPES]
        n_men = len(ents)
        uniq = {e.text.lower() for e in ents}
        n_uniq = len(uniq)
        n_sents = max(sum(1 for _ in doc.sents), 1)
        wc = word_counts[i]
        dens[i] = n_uniq / (max(wc, 1) / 1000.0)
        divr[i] = (n_uniq / n_men) if n_men else 0.0
        persent[i] = n_uniq / n_sents
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_docs} ({(i+1)/(time.time()-t1):.1f} docs/s)")
    print(f"  spaCy done in {time.time()-t1:.1f}s, {n_docs/(time.time()-t1):.1f} docs/s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, entity_density=dens, entity_diversity=divr,
             entity_per_sentence=persent)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
