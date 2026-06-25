"""JSONL helpers — match the on-disk format the Oxford repo uses
(``datasets/*.jsonl`` with one JSON object per line)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            json.dump(row, f)
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
