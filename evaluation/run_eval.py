from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.compliance_agent import ComplianceAgent
from agents.main_pipeline import INSUFFICIENT_POLICY_MESSAGE, run_support_pipeline


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "tickets.jsonl"


@dataclass
class Metrics:
    total: int = 0
    completed: int = 0
    abstained: int = 0

    citation_coverage: int = 0
    unsupported_claims: int = 0

    escalation_expected: int = 0
    escalation_correct: int = 0

    def to_dict(self) -> dict[str, Any]:
        def rate(num: int, den: int) -> float:
            return 0.0 if den == 0 else num / den

        return {
            "total": self.total,
            "completed": self.completed,
            "abstained": self.abstained,
            "citation_coverage_rate_completed": rate(self.citation_coverage, self.completed),
            "unsupported_claim_rate_completed": rate(self.unsupported_claims, self.completed),
            "escalation_accuracy": rate(self.escalation_correct, self.escalation_expected),
        }


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def parse_decision(resolution_text: str) -> str | None:
    # Extract from section 3 line: "- approve" etc.
    m = re.search(
        r"3\.\s*Decision.*?\n\s*-\s*(approve|deny|partial|escalate)\b",
        resolution_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    return m.group(1).lower()


def has_citations(policy_chunks: list[dict[str, Any]], resolution_text: str) -> bool:
    # Must have citations in chunks AND citations appear in resolution.
    any_chunk_citation = any(
        (c.get("citation") or {}).get("doc") and (c.get("citation") or {}).get("chunk_id") is not None
        for c in policy_chunks
    )
    any_in_text = bool(re.search(r"chunk_id\s*=\s*\d+", resolution_text))
    return any_chunk_citation and any_in_text


def run_pipeline_eval(*, dataset_path: Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    compliance = ComplianceAgent()

    metrics = Metrics(total=len(dataset))

    per_item: list[dict[str, Any]] = []

    for item in dataset:
        ticket_text = item["ticket_text"]
        order_context = item.get("order_context")
        expected = item.get("expected") or {}

        result = run_support_pipeline(ticket_text=ticket_text, order_context=order_context, top_k=5)

        status = result.get("status")
        resolution = result.get("resolution") or ""
        policy_chunks = result.get("policy_chunks") or []

        is_abstained = status == "ABSTAINED" or resolution.strip() == INSUFFICIENT_POLICY_MESSAGE
        if is_abstained:
            metrics.abstained += 1
        else:
            metrics.completed += 1

        # Citation coverage (only meaningful if completed)
        if not is_abstained and has_citations(policy_chunks, resolution):
            metrics.citation_coverage += 1

        # Unsupported claim rate (only meaningful if completed)
        if not is_abstained:
            verdict, findings = compliance.check_with_findings(
                resolution_draft=resolution,
                retrieved_evidence=policy_chunks,
            )
            if any(f.kind == "unsupported_claim" for f in findings):
                metrics.unsupported_claims += 1

        # Escalation correctness
        should_escalate = bool(expected.get("should_escalate"))
        if should_escalate:
            metrics.escalation_expected += 1
            decision = parse_decision(resolution)
            # Abstained counts as not escalated
            if (not is_abstained) and decision == "escalate":
                metrics.escalation_correct += 1

        per_item.append(
            {
                "id": item.get("id"),
                "bucket": item.get("bucket"),
                "status": status,
                "decision": parse_decision(resolution),
                "compliance": result.get("compliance"),
                "abstained": is_abstained,
            }
        )

    return {"metrics": metrics.to_dict(), "items": per_item}


def main() -> None:
    result = run_pipeline_eval(dataset_path=DEFAULT_DATASET_PATH)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
