from __future__ import annotations

from langchain_core.retrievers import BaseRetriever


def get_retriever(vectorstore, *, k: int = 4) -> BaseRetriever:
    return vectorstore.as_retriever(search_kwargs={"k": k})
