# perplexity correlations configs
# each entry = name -> (hf_repo_id, filename_in_repo)

from __future__ import annotations
from pathlib import Path

# patch fasttext's numpy-2 incompatibility before anyone imports it!!
import numpy as _np
import fasttext as _ft  # noqa: E402
import fasttext.FastText as _ftmod  # noqa: E402

def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error='strict'):
    def _check(entry):
        if entry.find("\n") != -1:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        return entry + "\n"
    if isinstance(text, list):
        text = [_check(e) for e in text]
        all_labels, all_probs = self.f.multilinePredict(text, k, threshold, on_unicode_error)
        return all_labels, _np.asarray(all_probs)
    predictions = self.f.predict(_check(text), k, threshold, on_unicode_error)
    if predictions:
        probs, labels = zip(*predictions)
    else:
        probs, labels = ([], ())
    return labels, _np.asarray(probs)
_ftmod._FastText.predict = _patched_predict

from huggingface_hub import hf_hub_download

from config import M2_ROOT

MODELS_DIR = M2_ROOT / "models"

PC_FASTTEXT_MODELS = {
    "arc_easy":   ("perplexity-correlations/fasttext-arc-easy-target",   "arc_easy_target.bin"),
    "piqa":       ("perplexity-correlations/fasttext-piqa-target",       "piqa_target.bin"),
    "sciq":       ("perplexity-correlations/fasttext-sciq-target",       "sciq_target.bin"),
    # using spanish variant for now
    "lambada_es": ("perplexity-correlations/fasttext-lambada-es-target", "lambada_es_target.bin"),
}


def ensure_model(name: str) -> Path:
    repo_id, fn = PC_FASTTEXT_MODELS[name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(repo_id=repo_id, filename=fn, local_dir=str(MODELS_DIR / name))
    return Path(local)
