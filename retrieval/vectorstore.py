from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS


_VECTORSTORE_CACHE: dict[tuple[str, str], FAISS] = {}


def _embeddings_cache_key(embeddings: Any) -> str:
    # Best-effort stable key so we don't mix indexes created with different embeddings.
    model_name = getattr(embeddings, "model_name", None)
    if isinstance(model_name, str) and model_name:
        return model_name
    return type(embeddings).__name__


def load_faiss_vectorstore(index_dir: Path, embeddings) -> FAISS:
    """Load a FAISS index that was previously saved with `save_local()`."""

    index_dir = index_dir.resolve()
    cache_key = (str(index_dir), _embeddings_cache_key(embeddings))
    cached = _VECTORSTORE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found: {index_dir}. Run: python main.py ingest"
        )

    # LangChain has varied the safety flag name across versions.
    try:
        vs = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except TypeError:
        vs = FAISS.load_local(str(index_dir), embeddings)

    _VECTORSTORE_CACHE[cache_key] = vs
    return vs
