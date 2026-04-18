# to stream .jsonl.zst shards
from __future__ import annotations
import io
import json
from pathlib import Path

import zstandard as zstd


def iter_docs(path: Path, max_docs: int | None = None):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        stream = io.TextIOWrapper(reader, encoding="utf-8")
        for i, line in enumerate(stream):
            if max_docs is not None and i >= max_docs:
                return
            if not line.strip():
                continue
            yield i, json.loads(line)
