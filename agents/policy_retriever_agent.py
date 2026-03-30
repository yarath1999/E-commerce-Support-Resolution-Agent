from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from retrieval.embeddings import get_embeddings
from retrieval.vectorstore import load_faiss_vectorstore


DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[1] / "retrieval" / "index" / "faiss"


@dataclass(frozen=True)
class PolicyChunk:
    excerpt: str
    metadata: dict[str, Any]
    citation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "excerpt": self.excerpt,
            "metadata": self.metadata,
            "citation": self.citation,
        }


class PolicyRetrieverAgent:
    """Retrieve raw policy chunks for a given ticket.

    Input:
    - issue_type
    - ticket_text
    - order_context

    Output:
    - top 5 relevant policy chunks

    Requirements:
    - include metadata
    - include citations (doc + chunk_id)
    - NO summarization
    - return raw excerpts
    """

    def __init__(self, *, index_dir: Path | None = None):
        self.index_dir = (index_dir or DEFAULT_INDEX_DIR).resolve()

    def retrieve(
        self,
        *,
        issue_type: str,
        ticket_text: str,
        order_context: Any | None = None,
        k: int = 5,
    ) -> list[PolicyChunk]:
        embeddings = get_embeddings()
        vectorstore = load_faiss_vectorstore(self.index_dir, embeddings)

        query = _build_query(issue_type=issue_type, ticket_text=ticket_text, order_context=order_context)

        # Use similarity_search to return Documents with raw page_content.
        docs = vectorstore.similarity_search(query, k=k)

        results: list[PolicyChunk] = []
        for doc in docs:
            metadata = dict(doc.metadata or {})
            source = metadata.get("source") or metadata.get("doc_id") or "unknown"
            chunk_id = metadata.get("chunk_id")
            citation = {
                "doc": source,
                "chunk_id": chunk_id,
            }
            results.append(
                PolicyChunk(
                    excerpt=doc.page_content,
                    metadata=metadata,
                    citation=citation,
                )
            )

        return results


def retrieve_policy_chunks(
    *,
    issue_type: str,
    ticket_text: str,
    order_context: Any | None = None,
    k: int = 5,
    index_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Functional wrapper returning plain dicts (JSON-serializable)."""

    agent = PolicyRetrieverAgent(index_dir=index_dir)
    return [c.to_dict() for c in agent.retrieve(issue_type=issue_type, ticket_text=ticket_text, order_context=order_context, k=k)]


def _build_query(*, issue_type: str, ticket_text: str, order_context: Any | None) -> str:
    ctx = _coerce_context(order_context)

    # Keep it simple and transparent: concatenate fields.
    # No summarization or interpretation; the vector search does the matching.
    parts: list[str] = []
    if issue_type:
        parts.append(f"issue_type: {issue_type}")
    if ticket_text:
        parts.append(f"ticket: {ticket_text}")
    if ctx:
        parts.append("order_context: " + json.dumps(ctx, ensure_ascii=False, sort_keys=True))

    return "\n".join(parts).strip()


def _coerce_context(order_context: Any | None) -> dict[str, Any]:
    if order_context is None:
        return {}

    if isinstance(order_context, Mapping):
        return dict(order_context)

    if isinstance(order_context, str):
        s = order_context.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            if isinstance(obj, Mapping):
                return dict(obj)
        except Exception:
            return {"raw": s}

    return {"raw": str(order_context)}
