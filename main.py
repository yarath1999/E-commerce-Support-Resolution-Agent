from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
INDEX_DIR = PROJECT_DIR / "retrieval" / "index" / "faiss"
DEFAULT_EVAL_FILE = PROJECT_DIR / "evaluation" / "qa.jsonl"


def cmd_ingest() -> None:
    from ingestion.index_builder import build_faiss_index
    from ingestion.loaders import load_documents
    from retrieval.embeddings import get_embeddings

    docs = load_documents(DATA_DIR)
    if not docs:
        print(f"No supported documents found under {DATA_DIR}.")
        print("Add .txt/.md/.json/.jsonl files to data/ and re-run: python main.py ingest")
        return
    embeddings = get_embeddings()

    build_faiss_index(documents=docs, embeddings=embeddings, index_dir=INDEX_DIR)

    print(f"Loaded {len(docs)} raw documents from {DATA_DIR}")
    print(f"Built FAISS index at {INDEX_DIR}")


def cmd_ask(question: str, *, k: int) -> None:
    from agents.llm import get_llm
    from agents.support_system import answer_question
    from retrieval.embeddings import get_embeddings
    from retrieval.retriever import get_retriever
    from retrieval.vectorstore import load_faiss_vectorstore

    embeddings = get_embeddings()
    vectorstore = load_faiss_vectorstore(INDEX_DIR, embeddings)
    retriever = get_retriever(vectorstore, k=k)

    llm = get_llm()

    resp = answer_question(question=question, retriever=retriever, llm=llm)

    print(resp.answer)
    print(f"\n(route={resp.route.route}, rationale={resp.route.rationale}, retrieved={len(resp.context_docs)})")


def cmd_eval(eval_file: Path, *, k: int) -> None:
    from evaluation.retrieval_eval import load_eval_items, recall_at_k
    from retrieval.embeddings import get_embeddings
    from retrieval.retriever import get_retriever
    from retrieval.vectorstore import load_faiss_vectorstore

    if not eval_file.exists():
        raise FileNotFoundError(
            f"Eval file not found: {eval_file}. Create it as JSONL or pass --eval-file."
        )

    items = load_eval_items(eval_file)

    embeddings = get_embeddings()
    vectorstore = load_faiss_vectorstore(INDEX_DIR, embeddings)
    retriever = get_retriever(vectorstore, k=k)

    score = recall_at_k(retriever=retriever, items=items, k=k)
    print(f"recall@{k}: {score:.3f} ({len(items)} items)")


def cmd_triage(ticket_text: str, *, order_context_json: str | None, order_context_file: Path | None) -> None:
    from agents.triage_agent import triage_ticket

    order_context = None
    if order_context_file is not None:
        order_context = order_context_file.read_text(encoding="utf-8", errors="ignore")
    elif order_context_json is not None:
        order_context = order_context_json

    result = triage_ticket(ticket_text=ticket_text, order_context=order_context)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_pipeline(ticket_text: str, *, order_context_json: str | None, order_context_file: Path | None) -> None:
    from agents.main_pipeline import run_support_pipeline

    order_context = None
    if order_context_file is not None:
        order_context = order_context_file.read_text(encoding="utf-8", errors="ignore")
    elif order_context_json is not None:
        order_context = order_context_json

    result = run_support_pipeline(ticket_text=ticket_text, order_context=order_context)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_pipeline_eval(eval_file: Path) -> None:
    from evaluation.run_eval import run_pipeline_eval

    if not eval_file.exists():
        raise FileNotFoundError(
            f"Eval file not found: {eval_file}. Create it as JSONL or pass --eval-file."
        )

    result = run_pipeline_eval(dataset_path=eval_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-agent RAG system (LangChain + FAISS)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Build/rebuild the FAISS index from data/")
    ingest.set_defaults(func=lambda args: cmd_ingest())

    ask = sub.add_parser("ask", help="Ask a support question")
    ask.add_argument("question", type=str)
    ask.add_argument("--k", type=int, default=4, help="Top-k documents to retrieve")
    ask.set_defaults(func=lambda args: cmd_ask(args.question, k=args.k))

    ev = sub.add_parser("eval", help="Evaluate retrieval recall@k against evaluation/qa.jsonl")
    ev.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    ev.add_argument("--k", type=int, default=4)
    ev.set_defaults(func=lambda args: cmd_eval(args.eval_file, k=args.k))

    tri = sub.add_parser("triage", help="Classify a ticket into an issue type and ask clarifying questions")
    tri.add_argument("ticket", type=str, help="Ticket text")
    tri.add_argument(
        "--order-context-json",
        type=str,
        default=None,
        help="Order context as a JSON string (or any text).",
    )
    tri.add_argument(
        "--order-context-file",
        type=Path,
        default=None,
        help="Path to a JSON/text file containing order context.",
    )
    tri.set_defaults(
        func=lambda args: cmd_triage(
            args.ticket,
            order_context_json=args.order_context_json,
            order_context_file=args.order_context_file,
        )
    )

    pipe = sub.add_parser("pipeline", help="Run triage -> retrieve -> resolution -> compliance")
    pipe.add_argument("ticket", type=str, help="Ticket text")
    pipe.add_argument(
        "--order-context-json",
        type=str,
        default=None,
        help="Order context as a JSON string (or any text).",
    )
    pipe.add_argument(
        "--order-context-file",
        type=Path,
        default=None,
        help="Path to a JSON/text file containing order context.",
    )
    pipe.set_defaults(
        func=lambda args: cmd_pipeline(
            args.ticket,
            order_context_json=args.order_context_json,
            order_context_file=args.order_context_file,
        )
    )

    pe = sub.add_parser(
        "pipeline-eval",
        help="Evaluate end-to-end pipeline against evaluation/tickets.jsonl",
    )
    pe.add_argument(
        "--eval-file",
        type=Path,
        default=PROJECT_DIR / "evaluation" / "tickets.jsonl",
    )
    pe.set_defaults(func=lambda args: cmd_pipeline_eval(args.eval_file))

    return p


def main() -> None:
    load_dotenv(override=False)
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
