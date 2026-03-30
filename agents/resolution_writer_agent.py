from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


Decision = Literal["approve", "deny", "partial", "escalate"]


@dataclass(frozen=True)
class ResolutionOutput:
    classification: dict[str, Any]
    clarifying_questions: list[str]
    decision: Decision
    rationale: list[str]
    citations: list[dict[str, Any]]
    customer_response_draft: str
    next_steps_internal_notes: list[str]

    def to_text(self) -> str:
        lines: list[str] = []

        lines.append("1. Classification")
        for k, v in self.classification.items():
            lines.append(f"- {k}: {v}")

        lines.append("")
        lines.append("2. Clarifying Questions")
        if self.clarifying_questions:
            for q in self.clarifying_questions[:3]:
                lines.append(f"- {q}")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("3. Decision (approve/deny/partial/escalate)")
        lines.append(f"- {self.decision}")

        lines.append("")
        lines.append("4. Rationale")
        if self.rationale:
            for r in self.rationale:
                lines.append(f"- {r}")
        else:
            lines.append("- Insufficient policy evidence retrieved to justify a decision.")

        lines.append("")
        lines.append("5. Citations")
        if self.citations:
            for c in self.citations:
                doc = c.get("doc", "unknown")
                chunk_id = c.get("chunk_id")
                lines.append(f"- {doc} (chunk_id={chunk_id})")
        else:
            lines.append("- None")

        lines.append("")
        lines.append("6. Customer Response Draft")
        lines.append(self.customer_response_draft.strip() or "(No draft)")

        lines.append("")
        lines.append("7. Next Steps / Internal Notes")
        if self.next_steps_internal_notes:
            for n in self.next_steps_internal_notes:
                lines.append(f"- {n}")
        else:
            lines.append("- None")

        return "\n".join(lines).strip() + "\n"


class ResolutionWriterAgent:
    """Write a support resolution using ONLY retrieved evidence.

    STRICT RULES:
    - ONLY use retrieved evidence
    - DO NOT invent policy

    This agent is deliberately conservative: if required facts or policy excerpts
    are missing/unclear, it outputs Decision = escalate and asks clarifying questions.
    """

    def write(
        self,
        *,
        issue_type: str,
        ticket_text: str,
        order_context: Any | None,
        retrieved_policy_chunks: list[Mapping[str, Any]],
        triage_result: Mapping[str, Any] | None = None,
    ) -> ResolutionOutput:
        chunks = [dict(c) for c in (retrieved_policy_chunks or [])]
        citations = _collect_citations(chunks)

        classification = {
            "issue_type": issue_type,
        }
        if triage_result is not None:
            if "confidence" in triage_result:
                classification["triage_confidence"] = triage_result.get("confidence")
            if "issue_type" in triage_result and triage_result.get("issue_type"):
                classification["triage_issue_type"] = triage_result.get("issue_type")

        context = _coerce_context(order_context)

        clarifying_questions = _select_clarifying_questions(
            triage_result=triage_result,
            issue_type=issue_type,
            ticket_text=ticket_text,
            order_context=context,
            chunks=chunks,
        )

        needs_clarification = len(clarifying_questions) > 0

        decision, rationale, internal_notes = _decide_from_evidence(
            issue_type=issue_type,
            ticket_text=ticket_text,
            order_context=context,
            chunks=chunks,
            needs_clarification=needs_clarification,
        )

        customer_draft = _draft_customer_response(
            issue_type=issue_type,
            decision=decision,
            ticket_text=ticket_text,
            clarifying_questions=clarifying_questions,
            chunks=chunks,
            citations=citations,
        )

        return ResolutionOutput(
            classification=classification,
            clarifying_questions=clarifying_questions[:3],
            decision=decision,
            rationale=rationale,
            citations=citations,
            customer_response_draft=customer_draft,
            next_steps_internal_notes=internal_notes,
        )


def write_resolution(
    *,
    issue_type: str,
    ticket_text: str,
    order_context: Any | None,
    retrieved_policy_chunks: list[Mapping[str, Any]],
    triage_result: Mapping[str, Any] | None = None,
) -> str:
    """Convenience API returning formatted text output."""

    return ResolutionWriterAgent().write(
        issue_type=issue_type,
        ticket_text=ticket_text,
        order_context=order_context,
        retrieved_policy_chunks=retrieved_policy_chunks,
        triage_result=triage_result,
    ).to_text()


def _collect_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, Any]] = set()
    out: list[dict[str, Any]] = []

    for ch in chunks:
        citation = ch.get("citation") or {}
        doc = citation.get("doc")
        chunk_id = citation.get("chunk_id")
        if doc is None:
            # derive from metadata if present
            meta = ch.get("metadata") or {}
            doc = meta.get("source") or meta.get("doc_id")
        key = (str(doc), chunk_id)
        if doc is None:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append({"doc": doc, "chunk_id": chunk_id})

    return out


def _coerce_context(order_context: Any | None) -> dict[str, Any]:
    if order_context is None:
        return {}
    if isinstance(order_context, Mapping):
        return dict(order_context)
    if isinstance(order_context, str):
        s = order_context.strip()
        if not s:
            return {}
        return {"raw": s}
    return {"raw": str(order_context)}


def _select_clarifying_questions(
    *,
    triage_result: Mapping[str, Any] | None,
    issue_type: str,
    ticket_text: str,
    order_context: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> list[str]:
    # Prefer triage agent questions if provided.
    if triage_result is not None:
        qs = triage_result.get("clarifying_questions")
        if isinstance(qs, list) and qs:
            return [str(q) for q in qs][:3]

    # Otherwise: conservative set based on common missing facts needed to apply policy.
    qs: list[str] = []

    if not _truthy(order_context.get("order_id")) and not _truthy(order_context.get("order_number")):
        qs.append("What is your order number?")

    if issue_type == "shipping":
        if not _truthy(order_context.get("tracking_number")) and not _truthy(order_context.get("tracking")):
            qs.append("Do you have the tracking number (or tracking link)?")
        qs.append("What is the latest tracking status and when did it update?")

    if issue_type == "refund":
        qs.append("Which item(s) are you returning and what is their condition (unused/opened/defective)?")
        qs.append("Have you already initiated a return (RMA), and if so what’s the current status?")

    if issue_type == "promo":
        qs.append("Which coupon/promo code did you try to use, and what was in your cart?")
        qs.append("What was the cart subtotal before tax/shipping when you applied the code?")

    if issue_type == "payment":
        qs.append("What payment method did you use (card/PayPal/UPI/etc.)?")
        qs.append("Do you have a transaction/authorization ID or screenshot of the charge?")

    if issue_type == "fraud":
        qs.append("Do you recognize this order/charge, and have you already contacted your bank/payment provider?")
        qs.append("What email address is on the account/order?")

    # If we have evidence chunks but still can't apply policy, ask about the key condition often present in policy.
    if chunks and issue_type in {"shipping", "refund", "promo"}:
        qs.append("When did the issue occur (delivery date / purchase date), and when are you contacting us?")

    # Deduplicate preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for q in qs:
        q = q.strip()
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(q)

    return deduped[:3]


def _decide_from_evidence(
    *,
    issue_type: str,
    ticket_text: str,
    order_context: dict[str, Any],
    chunks: list[dict[str, Any]],
    needs_clarification: bool,
) -> tuple[Decision, list[str], list[str]]:
    # STRICT: If there is no evidence, never decide.
    if not chunks:
        return (
            "escalate",
            ["No policy chunks were retrieved; cannot justify a decision."],
            ["Re-run policy retrieval; verify FAISS index exists and ticket query is correct."],
        )

    # Conservative default: if key facts are missing, use "partial" (or "escalate" when appropriate)
    # while requesting the missing details. We avoid making firm commitments without facts.
    rationale: list[str] = []

    # Pull raw excerpts (evidence) and citations for rationale statements.
    excerpts = [str(ch.get("excerpt") or "") for ch in chunks]
    evidence_text = "\n\n".join(excerpts).lower()

    # Helper: add one or more direct evidence quotes (no summarization) to rationale.
    quotes = _extract_quotes_for_issue(issue_type=issue_type, chunks=chunks)
    if quotes:
        rationale.extend([f"Evidence excerpt: {q}" for q in quotes])
    else:
        rationale.append("Retrieved policy chunks do not contain a clear, directly applicable rule for this case.")

    # If we can conclusively deny/approve based on explicit policy and explicit facts, do it.
    # Otherwise escalate.
    internal_notes: list[str] = []

    normalized_issue = (issue_type or "").strip().lower()
    if needs_clarification:
        default_when_missing: Decision = "partial" if normalized_issue not in {"fraud", "other"} else "escalate"
        internal_notes.append("Proceed once missing details are confirmed; avoid firm commitments without the required facts.")
    else:
        default_when_missing = "escalate"

    if issue_type == "shipping":
        # We need the timing/track status to apply common "report within X hours" rules.
        has_tracking = _truthy(order_context.get("tracking_number")) or _truthy(order_context.get("tracking"))
        internal_notes.append("Check tracking events and delivery proof (GPS/photo/signature) if available.")
        if not has_tracking:
            internal_notes.append("Missing tracking number; request from customer or internal OMS.")
        return (default_when_missing, rationale, internal_notes)

    if issue_type == "refund":
        internal_notes.append("Verify item category (hygiene/perishable/final sale) and condition before approving return/refund.")
        return (default_when_missing, rationale, internal_notes)

    if issue_type == "promo":
        internal_notes.append("Verify promo terms: exclusions, minimum spend, stacking, and post-return subtotal effects.")
        return (default_when_missing, rationale, internal_notes)

    if issue_type == "payment":
        internal_notes.append("Validate duplicate charges vs pending authorizations; compare transaction IDs and settlement status.")
        # If policy evidence includes chargeback handling and ticket mentions chargeback, escalate.
        if "chargeback" in ticket_text.lower() or "chargeback" in evidence_text:
            internal_notes.append("If a chargeback is open, coordinate with disputes team; avoid conflicting refunds.")
        return (default_when_missing, rationale, internal_notes)

    if issue_type == "fraud":
        internal_notes.append("Follow account security playbook; verify identity before making account/order changes.")
        return ("escalate", rationale, internal_notes)

    return ("escalate", rationale, internal_notes)


def _extract_quotes_for_issue(*, issue_type: str, chunks: list[dict[str, Any]]) -> list[str]:
    # No summarization: return small raw excerpt lines that match issue keywords.
    # We keep it short so rationale stays readable.
    keywords_by_type = {
        "refund": ["refund", "return", "final sale", "hygiene", "perishable", "opened"],
        "shipping": ["lost", "delivered", "non‑receipt", "non-receipt", "tracking", "delay"],
        "payment": ["payment", "charged", "chargeback", "authorization", "billing"],
        "promo": ["coupon", "promo", "minimum", "stack", "discount"],
        "fraud": ["fraud", "unauthorized", "stolen", "chargeback"],
        "other": [],
    }

    kws = keywords_by_type.get(issue_type, [])
    if not kws:
        return []

    quotes: list[str] = []
    for ch in chunks:
        excerpt = str(ch.get("excerpt") or "")
        lines = [ln.strip() for ln in excerpt.splitlines() if ln.strip()]
        for ln in lines:
            if any(k in ln.lower() for k in kws):
                quotes.append(ln)
                if len(quotes) >= 5:
                    return quotes

    return quotes


def _draft_customer_response(
    *,
    issue_type: str,
    decision: Decision,
    ticket_text: str,
    clarifying_questions: list[str],
    chunks: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> str:
    # STRICT: avoid adding new policy; only:
    # - restate the customer’s issue
    # - ask clarifying questions
    # - include raw excerpts as evidence with citations

    lines: list[str] = []
    lines.append("Thanks for reaching out. I’m reviewing your request based on our current policy documentation.")
    lines.append("")
    lines.append(f"Issue type: {issue_type}")
    lines.append("")

    if clarifying_questions:
        lines.append("To proceed, please confirm:")
        for q in clarifying_questions[:3]:
            lines.append(f"- {q}")
        lines.append("")

    lines.append("Relevant policy excerpts:")
    for ch in chunks[:5]:
        excerpt = str(ch.get("excerpt") or "").strip()
        citation = ch.get("citation") or {}
        doc = citation.get("doc")
        chunk_id = citation.get("chunk_id")
        if not doc:
            meta = ch.get("metadata") or {}
            doc = meta.get("source") or meta.get("doc_id")
        lines.append("")
        lines.append(f"[{doc} | chunk_id={chunk_id}]")
        lines.append(excerpt)

    lines.append("")
    if decision in {"escalate", "partial"}:
        lines.append("Once I have the details above, I can confirm the correct outcome under the policy.")
        if decision == "partial":
            lines.append("In the meantime, I can start the process and verify eligibility once the missing details are confirmed.")
    else:
        lines.append(f"Decision: {decision}")

    return "\n".join(lines).strip()


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)
