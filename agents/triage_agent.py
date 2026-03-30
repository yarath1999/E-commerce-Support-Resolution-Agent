from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping


IssueType = Literal["refund", "shipping", "payment", "promo", "fraud", "other"]


@dataclass
class TriageResult:
    issue_type: IssueType
    confidence: float
    missing_fields: list[str]
    clarifying_questions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "confidence": float(round(self.confidence, 4)),
            "missing_fields": self.missing_fields,
            "clarifying_questions": self.clarifying_questions,
        }


class TriageAgent:
    """Deterministic triage for support tickets.

    Inputs:
    - ticket_text: free text
    - order_context: dict-like object (or JSON string) with any known order/account details

    Output:
    {
      "issue_type": "refund | shipping | payment | promo | fraud | other",
      "confidence": float,
      "missing_fields": [],
      "clarifying_questions": []
    }

    Rules:
    - detect ambiguity (low signal or close competing signals)
    - ask max 3 questions
    """

    def triage(self, *, ticket_text: str, order_context: Any | None = None) -> TriageResult:
        context = _coerce_context(order_context)
        ticket = (ticket_text or "").strip()
        norm = _normalize_text(ticket)

        scores = _score_issue_types(norm, context)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_type, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        confidence = _confidence_from_scores(top_score=top_score, second_score=second_score)
        ambiguous = _is_ambiguous(top_score=top_score, second_score=second_score)

        missing = _missing_fields(issue_type=top_type, order_context=context)
        questions = _clarifying_questions(
            issue_type=top_type,
            ambiguous=ambiguous,
            missing_fields=missing,
            ticket_text=ticket,
        )

        return TriageResult(
            issue_type=top_type,
            confidence=confidence,
            missing_fields=missing,
            clarifying_questions=questions,
        )


def triage_ticket(*, ticket_text: str, order_context: Any | None = None) -> dict[str, Any]:
    """Convenience functional API."""

    return TriageAgent().triage(ticket_text=ticket_text, order_context=order_context).to_dict()


def _coerce_context(order_context: Any | None) -> dict[str, Any]:
    if order_context is None:
        return {}

    if isinstance(order_context, Mapping):
        return dict(order_context)

    if isinstance(order_context, str):
        s = order_context.strip()
        if not s:
            return {}
        # Try JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, Mapping):
                return dict(obj)
        except Exception:
            # Some shells pass JSON with backslash-escaped quotes, e.g. {\"order_id\":\"123\"}
            # Try a light normalization before falling back.
            try:
                normalized = s.replace('\\"', '"')
                obj2 = json.loads(normalized)
                if isinstance(obj2, Mapping):
                    return dict(obj2)
            except Exception:
                pass

        # Try a "loose" dict format: {order_id:1234,payment_status:captured}
        loose = _parse_loose_object(s)
        if loose is not None:
            return loose
        # Fallback: store raw
        return {"raw": s}

    # Unknown type: best-effort repr
    return {"raw": str(order_context)}


def _parse_loose_object(s: str) -> dict[str, Any] | None:
    text = s.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
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

        # Coerce primitive values
        lowered = value.lower()
        if lowered in {"true", "false"}:
            coerced: Any = lowered == "true"
        elif lowered in {"null", "none"}:
            coerced = None
        else:
            # number?
            try:
                coerced = int(value)
            except Exception:
                try:
                    coerced = float(value)
                except Exception:
                    coerced = value

        result[key] = coerced

    return result


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _score_issue_types(ticket_norm: str, order_context: dict[str, Any]) -> dict[IssueType, float]:
    # Keyword buckets (kept small and high-signal on purpose)
    refund_kw = [
        "refund",
        "return",
        "exchange",
        "rma",
        "final sale",
        "hygiene",
        "perishable",
        "opened",
        "store credit",
    ]
    shipping_kw = [
        "tracking",
        "shipment",
        "shipping",
        "delivery",
        "delivered",
        "not delivered",
        "not received",
        "didn't receive",
        "did not receive",
        "never arrived",
        "where is my order",
        "late",
        "delay",
        "lost",
        "missing package",
        "wismo",
    ]
    payment_kw = [
        "payment",
        "charged twice",
        "charged 2x",
        "charged",
        "double charged",
        "declined",
        "authorization",
        "card",
        "paypal",
        "upi",
        "failed",
        "billing",
        "invoice",
    ]
    promo_kw = [
        "coupon",
        "promo",
        "promotion",
        "discount",
        "code",
        "voucher",
        "minimum spend",
        "stack",
    ]
    fraud_kw = [
        "fraud",
        "unauthorized",
        "chargeback",
        "stolen",
        "scam",
        "identity",
        "hacked",
        "account takeover",
    ]

    scores: dict[IssueType, float] = {
        "refund": _keyword_score(ticket_norm, refund_kw),
        "shipping": _keyword_score(ticket_norm, shipping_kw),
        "payment": _keyword_score(ticket_norm, payment_kw),
        "promo": _keyword_score(ticket_norm, promo_kw),
        "fraud": _keyword_score(ticket_norm, fraud_kw),
        "other": 0.2,  # baseline so we can fall back gracefully
    }

    # Context-based nudges
    if _truthy(order_context.get("tracking_number")):
        scores["shipping"] += 1.2
    if _truthy(order_context.get("carrier")):
        scores["shipping"] += 0.4

    if _truthy(order_context.get("refund_status")) or _truthy(order_context.get("return_status")):
        scores["refund"] += 0.6

    if _truthy(order_context.get("payment_status")) or _truthy(order_context.get("transaction_id")):
        scores["payment"] += 0.5

    if _truthy(order_context.get("coupon_code")) or _truthy(order_context.get("promo_code")):
        scores["promo"] += 0.5

    if _truthy(order_context.get("chargeback")) or _truthy(order_context.get("fraud_flag")):
        scores["fraud"] += 0.8

    # If nothing matched, prefer "other"
    if max(v for k, v in scores.items() if k != "other") < 0.8:
        scores["other"] += 0.8

    return scores


def _keyword_score(text: str, keywords: list[str]) -> float:
    score = 0.0
    for kw in keywords:
        if kw in text:
            score += 1.0
    # mild boost if explicit question words appear (often indicates a single topic)
    if any(w in text for w in ["how", "when", "where", "why", "what", "help"]):
        score += 0.1
    return score


def _confidence_from_scores(*, top_score: float, second_score: float) -> float:
    # Map to 0..1 without pretending probabilistic calibration.
    if top_score <= 0:
        return 0.0

    margin = max(0.0, top_score - second_score)
    # Higher absolute signal and higher margin => more confidence.
    raw = 0.35 + 0.12 * min(top_score, 6.0) + 0.18 * min(margin, 4.0)
    return float(max(0.0, min(0.99, raw)))


def _is_ambiguous(*, top_score: float, second_score: float) -> bool:
    # Ambiguity triggers:
    # - low evidence overall
    # - close competitor
    if top_score < 1.2:
        return True
    # Only treat a close margin as ambiguity if the runner-up also has meaningful evidence.
    if (top_score - second_score) < 0.8 and second_score >= 0.8:
        return True
    return False


def _missing_fields(*, issue_type: IssueType, order_context: dict[str, Any]) -> list[str]:
    required_by_type: dict[IssueType, list[str]] = {
        "shipping": ["order_id", "tracking_number", "delivery_address"],
        "refund": ["order_id", "items", "return_status"],
        "payment": ["order_id", "payment_method", "transaction_id"],
        "promo": ["order_id", "coupon_code", "cart_subtotal"],
        "fraud": ["order_id", "account_email", "chargeback"],
        "other": ["order_id"],
    }

    required = required_by_type[issue_type]
    missing: list[str] = []

    for field in required:
        if not _truthy(_get_field(order_context, field)):
            missing.append(field)

    return missing


def _get_field(ctx: dict[str, Any], field: str) -> Any:
    # allow aliases without adding complexity elsewhere
    aliases = {
        "order_id": ["order_id", "order_number", "id"],
        "tracking_number": ["tracking_number", "tracking", "tracking_id"],
        "delivery_address": ["delivery_address", "shipping_address", "address"],
        "items": ["items", "line_items", "skus"],
        "return_status": ["return_status", "refund_status", "rma_status"],
        "payment_method": ["payment_method", "card_brand", "method"],
        "transaction_id": ["transaction_id", "payment_id", "auth_id"],
        "coupon_code": ["coupon_code", "promo_code", "voucher"],
        "cart_subtotal": ["cart_subtotal", "subtotal", "amount"],
        "account_email": ["account_email", "email", "customer_email"],
        "chargeback": ["chargeback", "chargeback_status", "dispute"],
    }

    keys = aliases.get(field, [field])
    for k in keys:
        if k in ctx:
            return ctx.get(k)
    return None


def _clarifying_questions(
    *,
    issue_type: IssueType,
    ambiguous: bool,
    missing_fields: list[str],
    ticket_text: str,
) -> list[str]:
    questions: list[str] = []

    if ambiguous:
        questions.append(
            "To route this correctly, is your main issue about a refund/return, shipping/delivery, payment/charges, a promo/coupon, or suspected fraud?"
        )

    # Ask targeted questions based on missing fields (max 3 total).
    for field in missing_fields:
        if len(questions) >= 3:
            break
        q = _question_for_field(field=field, issue_type=issue_type)
        if q and q not in questions:
            questions.append(q)

    # If we still have room but the ticket seems vague, ask a single minimal clarifier.
    if len(questions) < 1 and len(ticket_text.strip()) < 25:
        questions.append("Could you share a bit more detail about what went wrong and what outcome you want?")

    return questions[:3]


def _question_for_field(*, field: str, issue_type: IssueType) -> str | None:
    if field == "order_id":
        return "What is your order number?"
    if field == "tracking_number":
        return "Do you have the tracking number (or tracking link) for this shipment?"
    if field == "delivery_address":
        return "Can you confirm the delivery city/ZIP/postal code (no full address needed)?"
    if field == "items":
        return "Which item(s) are involved (product name or SKU) and what’s their condition (unused/opened/defective)?"
    if field == "return_status":
        return "Have you already initiated a return (RMA), and if so what’s the current status?"
    if field == "payment_method":
        return "What payment method did you use (card/PayPal/UPI/etc.)?"
    if field == "transaction_id":
        return "Do you have a transaction/authorization ID or screenshot of the charge?"
    if field == "coupon_code":
        return "Which coupon/promo code did you try to use?"
    if field == "cart_subtotal":
        return "What was the cart subtotal before tax/shipping when you applied the promo?"
    if field == "account_email":
        return "What email address is on the account/order?"
    if field == "chargeback":
        return "Have you already filed a chargeback/dispute with your bank or payment provider?"

    # Type-specific fallback
    if issue_type == "shipping" and field == "delivery_date":
        return "What is the latest tracking status and the expected delivery date?"

    return None


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)
