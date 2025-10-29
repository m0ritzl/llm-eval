from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple
import csv

@dataclass
class Segment:
    id: int
    source: str
    hypothesis: str
    reference: Optional[str] = None


def read_tsv(path: str, has_reference: bool = True, header: bool = False) -> List[Segment]:
    segments: List[Segment] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        idx = 0
        for row in reader:
            if idx == 0 and header:
                idx += 1
                continue
            if has_reference:
                if len(row) < 3:
                    raise ValueError("Expected at least 3 columns: source, reference, hypothesis")
                source, reference, hypothesis = row[0], row[1], row[2]
            else:
                if len(row) < 2:
                    raise ValueError("Expected at least 2 columns: source, hypothesis")
                source, hypothesis = row[0], row[1]
                reference = None
            segments.append(Segment(id=idx, source=source, reference=reference, hypothesis=hypothesis))
            idx += 1
    return segments


def iter_batches(items: List[Segment], batch_size: int) -> Iterable[List[Segment]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
