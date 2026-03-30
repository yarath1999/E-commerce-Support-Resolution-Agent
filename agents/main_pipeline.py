from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from agents.compliance_agent import ComplianceAgent
from agents.policy_retriever_agent import retrieve_policy_chunks
from agents.resolution_writer_agent import ResolutionWriterAgent
from agents.triage_agent import triage_ticket


PipelineStatus = Literal[
    "NEEDS_INFO",
    "COMPLETED",
    "ABSTAINED",
]


INSUFFICIENT_POLICY_MESSAGE = "I don’t have sufficient policy information"
MIN_RETRIEVED_CHUNKS_THRESHOLD = 3


@dataclass(frozen=True)
class PipelineResult:
    status: PipelineStatus
    triage: dict[str, Any]
    policy_chunks: list[dict[str, Any]]
    resolution: str
    compliance: Literal["APPROVED", "REWRITE_REQUIRED"]
    compliance_attempts: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "triage": self.triage,
            "policy_chunks": self.policy_chunks,
            "resolution": self.resolution,
            "compliance": self.compliance,
            "compliance_attempts": self.compliance_attempts,
            "notes": self.notes,
        }


def run_support_pipeline(
    *,
    ticket_text: str,
    order_context: Any | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Main pipeline:

    1) Run Triage Agent
    2) If missing info → return questions
    3) Run Retriever
    4) Run Resolution Writer
    5) Run Compliance Agent

    If compliance fails:
    - regenerate (safe rewrite) or abstain

    Returns a structured dict.
    """

    notes: list[str] = []

    coerced_context = _coerce_order_context(order_context)

    triage = triage_ticket(ticket_text=ticket_text, order_context=coerced_context)

    missing_fields = triage.get("missing_fields") or []
    clarifying_questions = triage.get("clarifying_questions") or []

    issue_type = str(triage.get("issue_type") or "other")

    triage_confidence = triage.get("confidence")
    confidence = float(triage_confidence) if isinstance(triage_confidence, (int, float)) else 0.0

    # Step 2: early exit ONLY if issue is unclear OR confidence is very low OR ticket is highly ambiguous.
    issue_type_unclear = _issue_type_unclear(issue_type)
    highly_ambiguous = _is_highly_ambiguous(
        issue_type=issue_type,
        confidence=confidence,
        missing_fields=missing_fields,
        clarifying_questions=clarifying_questions,
        ticket_text=ticket_text,
    )

    if issue_type_unclear or confidence < 0.3 or highly_ambiguous:
        notes.append(
            "Early exit: triage requires clarification "
            f"(issue_type_unclear={issue_type_unclear}, confidence={confidence:.2f}, highly_ambiguous={highly_ambiguous})"
        )
        # Make sure missing fields are represented as clarifying questions in the output.
        triage_out = dict(triage)
        triage_out["clarifying_questions"] = _merge_questions(
            existing=triage_out.get("clarifying_questions"),
            derived=_missing_fields_to_questions(missing_fields),
        )
        return PipelineResult(
            status="NEEDS_INFO",
            triage=triage_out,
            policy_chunks=[],
            resolution="",
            compliance="REWRITE_REQUIRED",
            compliance_attempts=0,
            notes=notes,
        ).to_dict()

    # Otherwise, continue even if missing_fields exist; pass them through as clarifying questions.
    triage_for_writer = dict(triage)
    triage_for_writer["clarifying_questions"] = _merge_questions(
        existing=triage_for_writer.get("clarifying_questions"),
        derived=_missing_fields_to_questions(missing_fields),
    )

    # Step 3: retrieve evidence
    policy_chunks = retrieve_policy_chunks(
        issue_type=issue_type,
        ticket_text=ticket_text,
        order_context=coerced_context,
        k=top_k,
    )

    # Hallucination control: if we don't have enough evidence chunks, abstain.
    if len(policy_chunks) < MIN_RETRIEVED_CHUNKS_THRESHOLD:
        notes.append(
            f"Abstain: retrieved chunks below threshold ({len(policy_chunks)} < {MIN_RETRIEVED_CHUNKS_THRESHOLD})"
        )
        return PipelineResult(
            status="ABSTAINED",
            triage=triage_for_writer,
            policy_chunks=policy_chunks,
            resolution=INSUFFICIENT_POLICY_MESSAGE,
            compliance="REWRITE_REQUIRED",
            compliance_attempts=0,
            notes=notes,
        ).to_dict()

    # Hallucination control: if citations are missing, fail response.
    if not _has_any_citations(policy_chunks):
        notes.append("Fail: no citations in retrieved evidence")
        return PipelineResult(
            status="ABSTAINED",
            triage=triage_for_writer,
            policy_chunks=policy_chunks,
            resolution=INSUFFICIENT_POLICY_MESSAGE,
            compliance="REWRITE_REQUIRED",
            compliance_attempts=0,
            notes=notes,
        ).to_dict()

    # Hallucination control: if evidence doesn't appear relevant to the issue, abstain.
    if not _evidence_looks_relevant(issue_type=issue_type, policy_chunks=policy_chunks):
        notes.append("Abstain: retrieved evidence does not look relevant to issue_type")
        return PipelineResult(
            status="ABSTAINED",
            triage=triage_for_writer,
            policy_chunks=policy_chunks,
            resolution=INSUFFICIENT_POLICY_MESSAGE,
            compliance="REWRITE_REQUIRED",
            compliance_attempts=0,
            notes=notes,
        ).to_dict()

    writer = ResolutionWriterAgent()
    compliance = ComplianceAgent()

    # Step 4: generate draft
    draft = writer.write(
        issue_type=issue_type,
        ticket_text=ticket_text,
        order_context=coerced_context,
        retrieved_policy_chunks=policy_chunks,
        triage_result=triage_for_writer,
    ).to_text()

    # Step 5: compliance checks with safe rewrite
    attempts = 0
    verdict = "REWRITE_REQUIRED"

    for _ in range(2):
        attempts += 1
        verdict, findings = compliance.check_with_findings(
            resolution_draft=draft,
            retrieved_evidence=policy_chunks,
        )
        if verdict == "APPROVED":
            return PipelineResult(
                status="COMPLETED",
                triage=triage_for_writer,
                policy_chunks=policy_chunks,
                resolution=draft,
                compliance=verdict,
                compliance_attempts=attempts,
                notes=notes,
            ).to_dict()

        # Hallucination control: block output if unsupported claims or missing citations are detected.
        if any(f.kind in {"unsupported_claim", "missing_citations"} for f in findings):
            notes.append("Blocked: compliance detected unsupported claims or missing citations")
            return PipelineResult(
                status="ABSTAINED",
                triage=triage_for_writer,
                policy_chunks=policy_chunks,
                resolution=INSUFFICIENT_POLICY_MESSAGE,
                compliance="REWRITE_REQUIRED",
                compliance_attempts=attempts,
                notes=notes,
            ).to_dict()

        notes.append("Compliance failed; attempting safe rewrite")
        draft = _sanitize_draft(draft)

    # Abstain: produce an extremely conservative output with evidence blocks only.
    notes.append("Compliance still failing; abstaining")
    abstained = _abstain_resolution(
        issue_type=issue_type,
        triage=triage,
        policy_chunks=policy_chunks,
    )
    final_verdict = compliance.check(resolution_draft=abstained, retrieved_evidence=policy_chunks)

    return PipelineResult(
        status="ABSTAINED",
        triage=triage_for_writer,
        policy_chunks=policy_chunks,
        resolution=abstained,
        compliance=final_verdict,
        compliance_attempts=attempts + 1,
        notes=notes,
    ).to_dict()


def _has_any_citations(policy_chunks: list[Mapping[str, Any]]) -> bool:
    for ch in policy_chunks:
        c = ch.get("citation") if isinstance(ch, Mapping) else None
        if isinstance(c, Mapping) and c.get("doc") and c.get("chunk_id") is not None:
            return True
        meta = ch.get("metadata") if isinstance(ch, Mapping) else None
        if isinstance(meta, Mapping) and (meta.get("source") or meta.get("doc_id")) and meta.get("chunk_id") is not None:
            return True
    return False


def _evidence_looks_relevant(*, issue_type: str, policy_chunks: list[Mapping[str, Any]]) -> bool:
    keywords_by_type: dict[str, list[str]] = {
        "refund": ["refund", "return", "exchange", "final sale", "hygiene", "perishable", "opened"],
        "shipping": ["tracking", "delivery", "delivered", "lost", "delay", "non‑receipt", "non-receipt", "shipment"],
        "payment": ["payment", "charged", "authorization", "billing", "invoice", "chargeback"],
        "promo": ["coupon", "promo", "promotion", "discount", "code", "minimum"],
        "fraud": ["fraud", "unauthorized", "chargeback", "stolen", "scam"],
        "other": [],
    }

    kws = keywords_by_type.get(issue_type, [])
    if not kws:
        return True

    for ch in policy_chunks:
        excerpt = str(ch.get("excerpt") or "").lower()
        if any(k in excerpt for k in kws):
            return True

    return False


def _blocking_missing_fields(*, issue_type: str, missing_fields: Any) -> list[str]:
    if not isinstance(missing_fields, list):
        return []
    missing = [str(x) for x in missing_fields]

    # Define truly blocking fields per issue type.
    # Keep this minimal: retrieval + evidence-only writing can proceed with partial context.
    if issue_type == "shipping":
        # Need at least one way to locate shipment/order.
        must_have_any = {"order_id", "tracking_number"}
        if must_have_any.issubset(set(missing)):
            return ["order_id_or_tracking_number"]
        return []

    if issue_type == "refund":
        must_have = {"order_id"}
        if any(f in missing for f in must_have):
            return ["order_id"]
        return []

    if issue_type == "payment":
        must_have = {"order_id"}
        if any(f in missing for f in must_have):
            return ["order_id"]
        return []

    if issue_type == "promo":
        must_have = {"order_id", "coupon_code"}
        return [f for f in must_have if f in missing]

    if issue_type == "fraud":
        # Need at least one identifier.
        must_have_any = {"order_id", "account_email"}
        if must_have_any.issubset(set(missing)):
            return ["order_id_or_account_email"]
        return []

    return []


def _issue_type_unclear(issue_type: str) -> bool:
    it = (issue_type or "").strip().lower()
    # Triage uses a fixed taxonomy; treating "other" as unclear avoids applying the wrong policy.
    return it in {"", "unknown", "other"}


def _missing_fields_to_questions(missing_fields: Any) -> list[str]:
    if not isinstance(missing_fields, list):
        return []

    field_to_question: dict[str, str] = {
        "order_id": "What is your order number?",
        "order_number": "What is your order number?",
        "items": "Which item(s) are involved (product name or SKU) and what’s their condition (unused/opened/defective)?",
        "tracking_number": "Do you have the tracking number (or tracking link)?",
        "delivery_date": "When was the order delivered (or what is the latest tracking update date)?",
        "purchase_date": "What was the purchase date?",
        "item_condition": "What is the item condition (unused/opened/defective)?",
        "item_category": "What is the item category (e.g., perishable, hygiene, final sale)?",
        "return_status": "Have you already initiated a return (RMA), and what’s the current status?",
        "refund_status": "Has a refund already been initiated, and what’s the current refund status?",
        "payment_method": "What payment method did you use (card/PayPal/UPI/etc.)?",
        "transaction_id": "Do you have a transaction/authorization ID?",
        "coupon_code": "Which coupon/promo code did you try to use?",
        "region": "Which country/region was the order placed in?",
    }

    out: list[str] = []
    for f in missing_fields:
        key = str(f).strip()
        if not key:
            continue
        out.append(field_to_question.get(key, f"Can you provide: {key}?"))

    # Deduplicate preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for q in out:
        if q in seen:
            continue
        seen.add(q)
        deduped.append(q)
    return deduped


def _merge_questions(*, existing: Any, derived: list[str]) -> list[str]:
    out: list[str] = []

    # Prioritize questions derived from missing_fields so partial-info flows surface them.
    out.extend([str(x).strip() for x in (derived or []) if str(x).strip()])
    if isinstance(existing, list):
        out.extend([str(x).strip() for x in existing if str(x).strip()])

    deduped: list[str] = []
    seen: set[str] = set()
    for q in out:
        if q in seen:
            continue
        seen.add(q)
        deduped.append(q)
    return deduped


def _is_highly_ambiguous(
    *,
    issue_type: str,
    confidence: float,
    missing_fields: Any,
    clarifying_questions: Any,
    ticket_text: str,
) -> bool:
    # Keep this conservative: do not stop normal refund/shipping tickets.
    mf = missing_fields if isinstance(missing_fields, list) else []
    qs = clarifying_questions if isinstance(clarifying_questions, list) else []

    short_ticket = len((ticket_text or "").strip()) < 25
    many_unknowns = (len(mf) >= 3) or (len(qs) >= 3)
    lowish_conf = confidence < 0.45

    # Highly ambiguous if it's both low-signal and we still need lots of basic info,
    # or if issue type isn't confidently distinguished.
    if short_ticket and lowish_conf:
        return True

    if (issue_type or "").strip().lower() == "other" and confidence < 0.5 and (len(qs) > 0 or len(mf) > 0):
        return True

    if lowish_conf and many_unknowns:
        return True

    return False


def _sanitize_draft(draft: str) -> str:
    """Remove common sensitive info patterns without adding new claims."""

    text = draft

    # Emails
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)

    # Card-like long digit sequences
    text = re.sub(r"\b\d{13,19}\b", "[REDACTED_NUMBER]", text)

    # Phone-ish
    text = re.sub(
        r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b",
        "[REDACTED_PHONE]",
        text,
    )

    # Street address hints
    text = re.sub(
        r"\b\d+\s+[A-Za-z0-9.-]+\s+(street|st|road|rd|avenue|ave|blvd|lane|ln|drive|dr)\b",
        "[REDACTED_ADDRESS]",
        text,
        flags=re.IGNORECASE,
    )

    return text


def _coerce_order_context(order_context: Any | None) -> Any:
    """Coerce order context into a dict when possible.

    Accepts:
    - Mapping/dict -> dict
    - JSON string -> dict
    - Backslash-escaped JSON string -> dict
    - Loose object string: {order_id:A-100,tracking_number:1Z999}

    Returns either a dict or the original value if it can't be parsed.
    """

    if order_context is None:
        return None

    if isinstance(order_context, Mapping):
        return dict(order_context)

    if not isinstance(order_context, str):
        return order_context

    s = order_context.strip()
    if not s:
        return None

    # JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, Mapping):
            return dict(obj)
    except Exception:
        pass

    # Escaped JSON
    try:
        normalized = s.replace('\\"', '"')
        obj = json.loads(normalized)
        if isinstance(obj, Mapping):
            return dict(obj)
    except Exception:
        pass

    # Loose object
    loose = _parse_loose_object(s)
    if loose is not None:
        return loose

    return order_context


def _parse_loose_object(s: str) -> dict[str, Any] | None:
    text = s.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    # If it looks like proper JSON, let JSON parsing handle it.
    if '"' in text:
        return None

    inner = text[1:-1].strip()
    if not inner:
        return {}

    result: dict[str, Any] = {}
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            return None
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return None

        lowered = value.lower()
        if lowered in {"true", "false"}:
            coerced: Any = lowered == "true"
        elif lowered in {"null", "none"}:
            coerced = None
        else:
            # Attempt numeric coercion, but keep alphanumeric IDs as strings.
            if re.fullmatch(r"-?\d+", value):
                coerced = int(value)
            elif re.fullmatch(r"-?\d+\.\d+", value):
                coerced = float(value)
            else:
                coerced = value

        result[key] = coerced

    return result


def _abstain_resolution(*, issue_type: str, triage: Mapping[str, Any], policy_chunks: list[Mapping[str, Any]]) -> str:
    # Evidence-only abstention with required section headers.
    # No policy invention: decision remains escalate.

    lines: list[str] = []

    lines.append("1. Classification")
    lines.append(f"- issue_type: {issue_type}")
    if "confidence" in triage:
        lines.append(f"- triage_confidence: {triage.get('confidence')}")

    lines.append("")
    lines.append("2. Clarifying Questions")
    qs = triage.get("clarifying_questions")
    if isinstance(qs, list) and qs:
        for q in qs[:3]:
            lines.append(f"- {q}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("3. Decision (approve/deny/partial/escalate)")
    lines.append("- escalate")

    lines.append("")
    lines.append("4. Rationale")
    lines.append("- Compliance could not be satisfied for the generated draft; providing evidence-only excerpts.")

    lines.append("")
    lines.append("5. Citations")
    seen: set[tuple[str, Any]] = set()
    for ch in policy_chunks[:5]:
        c = ch.get("citation") or {}
        doc = c.get("doc")
        chunk_id = c.get("chunk_id")
        if not doc:
            meta = ch.get("metadata") or {}
            doc = meta.get("source") or meta.get("doc_id")
        key = (str(doc), chunk_id)
        if doc and key not in seen:
            seen.add(key)
            lines.append(f"- {doc} (chunk_id={chunk_id})")
    if len(seen) == 0:
        lines.append("- None")

    lines.append("")
    lines.append("6. Customer Response Draft")
    lines.append("Thanks for reaching out. We need a bit more information to proceed.")
    if isinstance(qs, list) and qs:
        lines.append("\nPlease confirm:")
        for q in qs[:3]:
            lines.append(f"- {q}")
    lines.append("\nRelevant policy excerpts:")
    for ch in policy_chunks[:5]:
        excerpt = str(ch.get("excerpt") or "").strip()
        c = ch.get("citation") or {}
        doc = c.get("doc")
        chunk_id = c.get("chunk_id")
        if not doc:
            meta = ch.get("metadata") or {}
            doc = meta.get("source") or meta.get("doc_id")
        lines.append("")
        lines.append(f"[{doc} | chunk_id={chunk_id}]")
        lines.append(excerpt)

    lines.append("")
    lines.append("7. Next Steps / Internal Notes")
    lines.append("- Route to a human reviewer if sensitive data or policy contradictions are present.")

    return "\n".join(lines).strip() + "\n"
