from __future__ import annotations

from pathlib import Path
from collections import defaultdict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Add stable, per-source chunk identifiers for citations.
    counters: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        meta = chunk.metadata or {}
        source = str(meta.get("source", "unknown"))
        counters[source] += 1
        meta.setdefault("doc_id", source)
        meta.setdefault("chunk_id", counters[source])
        chunk.metadata = meta

    return chunks


def build_faiss_index(
    *,
    documents: list[Document],
    embeddings,
    index_dir: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> FAISS:
    """Build (or rebuild) a FAISS index under `index_dir`."""

    index_dir = index_dir.resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(str(index_dir))
    return vectorstore
