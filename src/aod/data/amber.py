"""AMBER discriminative loader for the official list annotations and keyed exports.

Maps official discriminative-hallucination to existence, three attribute
subtypes to attribute, and both relation spellings to relation. Generative
annotations are excluded. Query and annotation records are joined by ID.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


_DISCRIMINATIVE_TYPES = {"existence", "attribute", "relation"}


@dataclass(frozen=True)
class AMBERRecord:
    id: int
    image: str
    query: str
    answer: str  # "yes" | "no"
    typology: str  # "existence" | "attribute" | "relation"


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_amber_discriminative(
    query_path: str,
    annotations_path: str,
    *,
    typology: Optional[str] = None,
) -> List[AMBERRecord]:
    """Return AMBER discriminative samples, optionally filtered by typology."""
    queries = _load_json(query_path)
    if not isinstance(queries, list):
        raise ValueError(f"Expected a JSON list in {query_path}")
    annotations = _load_json(annotations_path)
    if isinstance(annotations, list):
        keyed = {str(row['id']): row for row in annotations}
        if len(keyed) != len(annotations):
            raise ValueError('Duplicate AMBER annotation IDs')
        annotations = keyed
    if not isinstance(annotations, dict):
        raise ValueError(f"Expected a JSON object in {annotations_path}")

    typology_filter = None
    if typology is not None:
        typology = typology.strip().lower()
        if typology not in _DISCRIMINATIVE_TYPES:
            raise ValueError(
                f"--amber_typology must be one of {sorted(_DISCRIMINATIVE_TYPES)} or omitted; got {typology!r}"
            )
        typology_filter = typology

    out: List[AMBERRecord] = []
    for rec in queries:
        sid = rec.get("id")
        if sid is None:
            continue
        # Annotations may be keyed by either str or int id; try str first, then
        # fall back to int (guarded — `int("foo")` would raise on non-numeric ids).
        ann = annotations.get(str(sid))
        if ann is None:
            try:
                ann = annotations.get(int(sid))
            except (TypeError, ValueError):
                ann = None
        if not isinstance(ann, dict):
            continue
        truth = str(ann.get("truth", "")).strip().lower()
        typ = str(ann.get("type", "")).strip().lower()
        if typ == 'discriminative-hallucination':
            typ = 'existence'
        elif typ.startswith('discriminative-attribute-'):
            typ = 'attribute'
        elif typ == 'discriminative-relation':
            typ = 'relation'
        if truth not in {"yes", "no"} or typ not in _DISCRIMINATIVE_TYPES:
            continue
        if typology_filter is not None and typ != typology_filter:
            continue
        out.append(
            AMBERRecord(
                id=int(sid),
                image=str(rec.get("image", "")),
                query=str(rec.get("query", "")),
                answer=truth,
                typology=typ,
            )
        )
    if not out:
        raise ValueError('No discriminative AMBER records found')
    if len({r.id for r in out}) != len(out):
        raise ValueError('Duplicate AMBER query IDs')
    return out


def amber_to_records(records: Iterable[AMBERRecord]) -> List[Dict[str, Any]]:
    """Flatten to the dict form consumed by the generic binary path."""
    return [
        {
            "id": rec.id,
            "image": rec.image,
            "question": rec.query,
            "answer": rec.answer,
            "gt_answer": "1" if rec.answer == "yes" else "0",
            "amber_type": rec.typology,
        }
        for rec in records
    ]
