"""
B1 entity features via spaCy en_core_web_sm.
Runs over first 10_000 chars per doc. Writes one npz per shard with three columns:
    entity_density, entity_diversity, entity_per_sentence
Saved under milestone2/data/entities/.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "milestone2"))

from config import SAMPLED_SHARDS_JSON, DATA_ROOT, shard_stem, doc_path  # type: ignore
from _shared_io import iter_docs  # type: ignore

ENTITIES_DIR = DATA_ROOT / "entities"

ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "NORP", "FAC", "DATE", "QUANTITY",
}
MAX_CHARS = 10_000

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        # enable sentence boundary detection without parser (cheap)
        if "senter" not in _NLP.pipe_names and "sentencizer" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")
    return _NLP


def _features_for_doc(text: str, word_count_est: int) -> tuple[float, float, float]:
    nlp = _get_nlp()
    t = text[:MAX_CHARS]
    if not t.strip():
        return float("nan"), float("nan"), float("nan")
    doc = nlp(t)
    ents = [e for e in doc.ents if e.label_ in ENTITY_TYPES]
    n_mentions = len(ents)
    uniq_texts = {e.text.lower() for e in ents}
    n_unique = len(uniq_texts)
    n_sents = max(sum(1 for _ in doc.sents), 1)
    # word_count_est for the FULL doc (so density compares fairly)
    density = n_unique / (max(word_count_est, 1) / 1000.0)
    diversity = (n_unique / n_mentions) if n_mentions else 0.0
    per_sent = n_unique / n_sents
    return density, diversity, per_sent


def process_shard(shard_idx: int, out_dir: Path, overwrite: bool,
                   max_docs: int | None) -> tuple[int, int, float]:
    out_path = out_dir / f"{shard_stem(shard_idx)}.npz"
    if out_path.exists() and not overwrite:
        return shard_idx, -1, 0.0
    t0 = time.time()
    dens, divr, persent = [], [], []
    for _doc_idx, obj in iter_docs(doc_path(shard_idx), max_docs=max_docs):
        text = obj.get("text", "") or ""
        wc = len(text.split())
        a, b, c = _features_for_doc(text, wc)
        dens.append(a); divr.append(b); persent.append(c)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             entity_density=np.asarray(dens, dtype=np.float32),
             entity_diversity=np.asarray(divr, dtype=np.float32),
             entity_per_sentence=np.asarray(persent, dtype=np.float32))
    return shard_idx, len(dens), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=ENTITIES_DIR)
    args = ap.parse_args()

    if args.shards:
        shards = args.shards
    else:
        shards = json.loads(SAMPLED_SHARDS_JSON.read_text())["shards"]
    if args.max_shards:
        shards = shards[: args.max_shards]

    print(f"Processing {len(shards)} shards single-process "
          f"(spaCy is best parallelized via Slurm job array, not python multiproc).")
    total = 0; ttime = 0.0
    for s in tqdm(shards, desc="entities"):
        _, n, el = process_shard(s, args.output_dir, args.overwrite, args.max_docs)
        if n >= 0:
            total += n; ttime += el
            tqdm.write(f"  shard {s}: {n} docs in {el:.1f}s")
    if total:
        print(f"Total: {total:,} docs, {total / max(ttime, 1e-9):.1f} docs/sec")


if __name__ == "__main__":
    main()
