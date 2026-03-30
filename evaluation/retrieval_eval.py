from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalItem:
    question: str
    expected_source_contains: str


def load_eval_items(path: Path) -> list[EvalItem]:
    """Load evaluation items from JSONL with keys:

    - question: str
    - expected_source_contains: str   (substring to match against doc metadata['source'])

    Example line:
    {"question":"How long do refunds take?","expected_source_contains":"returns_policy"}
    """

    items: list[EvalItem] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        items.append(
            EvalItem(
                question=obj["question"],
                expected_source_contains=obj["expected_source_contains"],
            )
        )
    return items


def recall_at_k(*, retriever, items: list[EvalItem], k: int = 4) -> float:
    hits = 0

    for item in items:
        docs = _retrieve(retriever, item.question)
        topk = docs[:k]
        if any(
            item.expected_source_contains.lower() in str((d.metadata or {}).get("source", "")).lower()
            for d in topk
        ):
            hits += 1

    return hits / max(1, len(items))


def _retrieve(retriever, query: str):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)
