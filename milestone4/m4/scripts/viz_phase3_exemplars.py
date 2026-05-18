"""For each top-6 EBM (no-PC) feature, fetch text exemplars at low/med/high values.

Outputs:
  figures/phase3/exemplars_<feature>.md   -- one markdown per feature
  figures/phase3/exemplars_summary.md     -- aggregated, easy to paste into report
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]   # .../milestone4/m4
REPO_ROOT = HERE.parent.parent               # repo root
sys.path.insert(0, str(REPO_ROOT / "milestone2"))
import importlib
m2cfg = importlib.import_module("config")
doc_path = m2cfg.doc_path

import zstandard as zstd

REPO = HERE.parent
DATA = HERE / "data"
OUT = HERE / "figures" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

# Auto-derive top-6 univariate features from the EBM-no-PC (nominal) model.
import joblib  # noqa: E402
_ebm = joblib.load(HERE / "data" / "models" / "ebm_interp.joblib")
_glob = _ebm.explain_global()
_gd = _glob.data()
_pairs = []
for _i, (_n, _imp) in enumerate(zip(_gd["names"], _gd["scores"])):
    if " & " in _n or _n in ("topic", "format"):
        continue
    _pairs.append((_n, _imp))
_pairs.sort(key=lambda kv: -kv[1])
FEATURES = [n for n, _ in _pairs[:6]]
print(f"Top-6 features from nominal EBM-no-PC: {FEATURES}")
# how many exemplars to pull per bucket
N_PER_BUCKET = 2
# fetch from at most this many distinct shards per feature to keep IO bounded
MAX_SHARDS_PER_FEATURE = 30
SNIPPET_CHARS = None   # None = full document (no truncation)


def iter_docs(path: Path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        reader = dctx.stream_reader(f)
        for i, line in enumerate(io.TextIOWrapper(reader, encoding="utf-8")):
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def fetch_texts(rows: pd.DataFrame) -> dict[tuple[int, int], dict]:
    """rows columns: __shard_idx, __doc_idx (others passthrough)."""
    out = {}
    for sh, sub in rows.groupby("__shard_idx"):
        target = set(int(x) for x in sub["__doc_idx"].tolist())
        p = doc_path(int(sh))
        if not p.exists():
            continue
        for di, doc in iter_docs(p):
            if di in target:
                out[(int(sh), int(di))] = doc
                target.remove(di)
                if not target:
                    break
    return out


def snippet(text: str, n=SNIPPET_CHARS) -> str:
    # Keep paragraph structure: only collapse runs of spaces/tabs, leave \n.
    t = "\n".join(" ".join(line.split()) for line in text.splitlines())
    if n is None or len(t) <= n:
        return t
    return t[:n] + "..."


def main():
    m = pd.read_parquet(DATA / "master.parquet")
    y_full = np.load(DATA / "document_quality_target.npy")
    val_shards = set(json.load(open(DATA / "val_shard_indices.json"))["val"])
    train_mask = (~m["__shard_idx"].isin(val_shards)).to_numpy()
    m_tr = m.loc[train_mask].reset_index(drop=True)
    m_tr["y"] = y_full
    ebm_pred = np.load(DATA / "ebm_interp_predicted_scores.npy")[train_mask]
    m_tr["ebm_pred"] = ebm_pred

    summary_lines = ["# EBM (no perplexity-corr) — exemplar documents per top feature\n"]
    summary_lines.append(
        "For each feature, we show a few documents drawn near the **low (10th-pct)**, "
        "**medium (50th-pct)**, and **high (90th-pct)** values of that feature, "
        "along with the doc's URL, the feature value, the per-doc target *y* "
        "(positive = lowers val_loss), and the EBM-no-PC predicted score. "
        "Full document text is shown verbatim (paragraph breaks preserved).\n")

    for feat in FEATURES:
        print(f"\n=== {feat} ===")
        f_per_md = [f"# Exemplars for `{feat}`\n"]
        col = m_tr[feat].astype(float)
        mask = col.notna() & m_tr["y"].notna()
        sub = m_tr.loc[mask, ["__shard_idx", "__doc_idx", "url",
                              feat, "y", "ebm_pred", "topic", "format"]].copy()
        lo, mid, hi = np.quantile(sub[feat], [0.10, 0.50, 0.90])
        # Pick rows near each quantile (smallest absolute distance) but
        # spread across shards to avoid grabbing 6 docs from the same file.
        picks = {}
        for label, q in [("low (10th-pct)", lo),
                         ("medium (50th-pct)", mid),
                         ("high (90th-pct)", hi)]:
            sub2 = sub.iloc[(sub[feat] - q).abs().argsort()].head(MAX_SHARDS_PER_FEATURE)
            # take first N from distinct shards
            kept = []
            seen_shards = set()
            for _, r in sub2.iterrows():
                if r["__shard_idx"] in seen_shards:
                    continue
                kept.append(r)
                seen_shards.add(r["__shard_idx"])
                if len(kept) >= N_PER_BUCKET:
                    break
            picks[label] = pd.DataFrame(kept)

        # Bulk-fetch texts for all picks of this feature
        all_rows = pd.concat(picks.values(), ignore_index=True)
        texts = fetch_texts(all_rows[["__shard_idx", "__doc_idx"]])

        summary_lines.append(f"\n## `{feat}`\n")
        for label, df in picks.items():
            f_per_md.append(f"\n## {label}  (target value ≈ "
                            f"{[lo, mid, hi][['low (10th-pct)', 'medium (50th-pct)', 'high (90th-pct)'].index(label)]:.3g})\n")
            summary_lines.append(f"\n### {label}\n")
            for _, r in df.iterrows():
                doc = texts.get((int(r["__shard_idx"]), int(r["__doc_idx"])))
                text = doc.get("text", "") if doc else "(text unavailable)"
                snip = snippet(text)
                line = (
                    f"\n---\n"
                    f"- **{feat} = {r[feat]:.3g}**, "
                    f"y = {r['y']:+.4f}, ebm_pred = {r['ebm_pred']:+.4f}, "
                    f"topic = {int(r['topic'])}, format = {int(r['format'])}\n"
                    f"- URL: `{r['url']}`\n"
                    f"- **Full document text:**\n\n"
                    f"```\n{snip}\n```\n"
                )
                f_per_md.append(line)
                summary_lines.append(line)

        (OUT / f"exemplars_{feat}.md").write_text("".join(f_per_md))
        print(f"wrote {OUT/('exemplars_'+feat+'.md')}")

    (OUT / "exemplars_summary.md").write_text("".join(summary_lines))
    print(f"\nwrote {OUT/'exemplars_summary.md'}")


if __name__ == "__main__":
    main()
