"""
Pkg 3: Review record store for the feedback form [last updated 2026-10-23]

Reads review_data.jsonl (one record per line: {"id", "original", "variants"})
without loading it into memory: an index of line offsets is built once and
only the lines of a requested sample are read. That keeps the server flat at
a few MB whether the file has 9 records or the full UHRI export (~270k).

The index is keyed on the file's size + mtime, so replacing the file (e.g. a
new upload into the mounted bucket) is picked up on the next request without
a restart.

Usage:
    store = RecordStore(Path("review_data.jsonl"))
    store.count()                      -> number of records
    store.sample(80, 1234)             -> 80 records, deterministic for the seed
    store.sample(80, 1234, exclude=ids) -> same, drawn only from records whose
                                          id is not in ``ids`` (a device's
                                          already-submitted ones)
    store.sample(0)                    -> every record

Open Issues:
    - Sampling is uniform. Stratified draws (per theme / country) would need
      those columns to be carried through make_data first.

Resumption:
    serve.py owns the HTTP side; this module knows nothing about requests.
"""
from __future__ import annotations

import json
import random
import re
import threading
from pathlib import Path
from typing import Collection, Optional

_ID_PREFIX = re.compile(rb'^\s*\{\s*"id"\s*:\s*(-?\d+)')   # make_data writes "id" first


class RecordStore:
    """Lazy, sample-oriented reader for a JSONL file of review records.

    Args:
        path: The JSONL file. May not exist yet; ``exists()`` tells.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._stamp: Optional[tuple[int, int]] = None   # (size, mtime_ns) of the indexed file
        self._offsets: list[int] = []
        self._ids: list[int] = []                        # record id per line, same order

    def exists(self) -> bool:
        return self.path.is_file()

    @staticmethod
    def _id_of(line: bytes) -> int:
        m = _ID_PREFIX.match(line)
        return int(m.group(1)) if m else int(json.loads(line)["id"])

    def _index(self) -> list[int]:
        """Byte offsets of every non-blank line, rebuilt when the file changes."""
        stat = self.path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        with self._lock:
            if stamp != self._stamp:
                offsets: list[int] = []
                ids: list[int] = []
                with open(self.path, "rb") as f:
                    pos = 0
                    for line in f:
                        if line.strip():
                            offsets.append(pos)
                            ids.append(self._id_of(line))
                        pos += len(line)
                self._offsets, self._ids, self._stamp = offsets, ids, stamp
            return self._offsets

    def count(self) -> int:
        """Number of records in the file."""
        return len(self._index())

    def _read(self, offsets: list[int]) -> list[dict]:
        records = []
        with open(self.path, "rb") as f:
            for off in offsets:
                f.seek(off)
                records.append(json.loads(f.readline()))
        return records

    def remaining(self, exclude: Collection[int] = ()) -> int:
        """How many records are left once ``exclude`` ids are taken out."""
        self._index()
        if not exclude:
            return len(self._ids)
        exclude = set(exclude)
        return sum(1 for i in self._ids if i not in exclude)

    def sample(self, n: int, seed: Optional[int] = None,
               exclude: Collection[int] = ()) -> list[dict]:
        """Draw ``n`` records at random, in file order.

        Args:
            n: Sample size. 0, a negative number or anything >= the pool
                size returns the whole pool.
            seed: Makes the draw reproducible, so the same (n, seed, exclude)
                yields the same records as long as the file is unchanged.
            exclude: Record ids left out of the pool, e.g. the ones a device
                has already submitted.

        Returns:
            The sampled records, sorted by position in the file.
        """
        offsets = self._index()
        if exclude:
            exclude = set(exclude)
            pool = [off for off, rid in zip(offsets, self._ids) if rid not in exclude]
        else:
            pool = offsets
        if n <= 0 or n >= len(pool):
            chosen = pool
        else:
            picks = random.Random(seed).sample(range(len(pool)), n)
            chosen = [pool[i] for i in sorted(picks)]
        return self._read(chosen)

    def all(self) -> list[dict]:
        """Every record. Only for small files (the baked standalone page)."""
        return self._read(self._index())
