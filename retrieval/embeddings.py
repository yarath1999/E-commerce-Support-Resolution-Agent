from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=4)
def _get_embeddings_cached(model_name: str):
    # Import from submodule to avoid importing chat model dependencies at package import time.
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def get_embeddings():
    """Return an embeddings model usable by LangChain.

    Defaults to a local SentenceTransformers model, so it runs without API keys.
    """

    model_name = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    return _get_embeddings_cached(model_name)
