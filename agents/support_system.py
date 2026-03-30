from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document

from agents.llm import format_answer_without_llm
from agents.router import RouteDecision, route_question


@dataclass
class SupportResponse:
    answer: str
    route: RouteDecision
    context_docs: list[Document]


def build_context_snippet(docs: list[Document], *, max_chars: int = 3500) -> str:
    parts: list[str] = []
    total = 0

    for i, doc in enumerate(docs, start=1):
        src = (doc.metadata or {}).get("source", "unknown")
        chunk = (doc.page_content or "").strip()
        if not chunk:
            continue

        header = f"[Source: {src} | Chunk {i}]\n"
        piece = header + chunk + "\n"

        if total + len(piece) > max_chars:
            remaining = max_chars - total
            if remaining > len(header) + 50:
                parts.append(piece[:remaining] + "\n")
            break

        parts.append(piece)
        total += len(piece)

    return "\n".join(parts).strip()


def answer_question(*, question: str, retriever, llm=None) -> SupportResponse:
    """Multi-agent-ish flow:
    1) Route agent decides intent
    2) Retrieval agent fetches top-k docs
    3) Answer agent generates (LLM) or falls back to a deterministic response
    """

    decision = route_question(question)

    # Route can influence retrieval later (metadata filters, per-index routing, etc.)
    docs = _retrieve(retriever, question)
    context = build_context_snippet(docs)

    if llm is None:
        return SupportResponse(
            answer=format_answer_without_llm(question, context),
            route=decision,
            context_docs=docs,
        )

    system_prompt = (
        "You are an e-commerce customer support agent. "
        "Use ONLY the provided context to answer. "
        "If the context doesn't contain the answer, say what is missing and ask one clarifying question. "
        "Be concise and action-oriented."
    )

    user_prompt = (
        f"Route: {decision.route}\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:" 
    )

    # Works for most LangChain chat models.
    msg = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    answer = getattr(msg, "content", None) or str(msg)

    return SupportResponse(answer=answer, route=decision, context_docs=docs)


def _retrieve(retriever, query: str):
    # Newer LangChain retrievers support `.invoke(query)`.
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)
