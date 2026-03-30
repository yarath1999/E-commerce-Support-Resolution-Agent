from __future__ import annotations

import os
from typing import Optional


def get_llm():
    """Return a chat model if configured, otherwise None.

    If `OPENAI_API_KEY` is present and `langchain-openai` is installed,
    this returns a `ChatOpenAI` instance.
    """

    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    return ChatOpenAI(model=model, temperature=temperature)


def format_answer_without_llm(question: str, context: str) -> str:
    context = (context or "").strip()

    if not context:
        return (
            "I couldn't find relevant policy/knowledge in your `data/` docs yet. "
            "Add FAQs/policies to `data/`, run `python main.py ingest`, then ask again."
        )

    return (
        "(LLM not configured) Based on the retrieved knowledge base, here is the most relevant info:\n\n"
        f"Question: {question.strip()}\n\n"
        f"Relevant context:\n{context}\n\n"
        "Next step: set `OPENAI_API_KEY` (or plug in another chat model) to generate a polished answer."
    )
