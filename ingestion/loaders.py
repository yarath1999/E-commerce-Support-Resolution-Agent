from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


SUPPORTED_TEXT_EXTS = {".txt", ".md"}
SUPPORTED_JSON_EXTS = {".json", ".jsonl"}


def iter_files(data_dir: Path) -> Iterable[Path]:
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in (SUPPORTED_TEXT_EXTS | SUPPORTED_JSON_EXTS):
            yield path


def load_documents(data_dir: Path) -> list[Document]:
    """Load raw documents from the `data/` folder.

    Supported:
    - .txt / .md (entire file as one document)
    - .json (object/list; best-effort stringify)
    - .jsonl (one JSON object per line; each becomes a document)
    """

    data_dir = data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: list[Document] = []

    for path in iter_files(data_dir):
        suffix = path.suffix.lower()
        policy_meta = infer_policy_metadata(path)
        if suffix in SUPPORTED_TEXT_EXTS:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path.relative_to(data_dir)),
                            "file_type": suffix.lstrip("."),
                            **policy_meta,
                        },
                    )
                )
            continue

        if suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore") or "null")
            if isinstance(obj, list):
                for i, item in enumerate(obj):
                    documents.append(
                        Document(
                            page_content=_json_item_to_text(item),
                            metadata={
                                "source": str(path.relative_to(data_dir)),
                                "file_type": "json",
                                "row": i,
                                **policy_meta,
                            },
                        )
                    )
            else:
                documents.append(
                    Document(
                        page_content=_json_item_to_text(obj),
                        metadata={
                            "source": str(path.relative_to(data_dir)),
                            "file_type": "json",
                            **policy_meta,
                        },
                    )
                )
            continue

        if suffix == ".jsonl":
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    item = {"text": line}
                documents.append(
                    Document(
                        page_content=_json_item_to_text(item),
                        metadata={
                            "source": str(path.relative_to(data_dir)),
                            "file_type": "jsonl",
                            "row": i,
                            **policy_meta,
                        },
                    )
                )
            continue

    return documents


def infer_policy_metadata(path: Path) -> dict[str, str]:
    """Infer required metadata fields from a file name.

    Returns keys:
    - policy_type
    - region
    - category

    Defaults are intentionally broad to keep ingestion robust.
    """

    name = path.name.lower()
    stem = path.stem.lower()
    import re

    tokens = {t for t in re.split(r"[^a-z0-9]+", stem) if t}

    policy_type = "unknown"
    region = "global"
    category = "general"

    if "marketplace" in name or "third" in name or "seller" in name:
        policy_type = "marketplace"
        category = "marketplace_vs_first_party"
    elif "regional" in name or "addendum" in name:
        policy_type = "addendum"
        category = "regional_differences"
    elif "first" in name or "core" in name:
        policy_type = "first_party"
        category = "core"

    # Region hints
    regions: list[str] = []
    if "us" in tokens or "usa" in tokens or "united" in tokens:
        regions.append("US")
    if "eu" in tokens or "europe" in tokens:
        regions.append("EU")
    if "india" in tokens:
        regions.append("India")
    if regions:
        region = ",".join(regions)

    return {
        "policy_type": policy_type,
        "region": region,
        "category": category,
    }


def _json_item_to_text(item: object) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "body", "message", "description"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return json.dumps(item, ensure_ascii=False, indent=2)
    return json.dumps(item, ensure_ascii=False)
