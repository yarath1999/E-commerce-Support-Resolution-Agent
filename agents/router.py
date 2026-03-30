from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Route = Literal["orders", "returns", "shipping", "general"]


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    rationale: str


def route_question(question: str) -> RouteDecision:
    q = (question or "").lower()

    if any(w in q for w in ["refund", "return", "exchange", "rma", "chargeback"]):
        return RouteDecision(route="returns", rationale="Return/refund keywords")

    if any(w in q for w in ["where is my order", "wismo", "tracking", "shipment", "delivered"]):
        return RouteDecision(route="shipping", rationale="Shipping/tracking keywords")

    if any(w in q for w in ["order", "invoice", "receipt", "cancel", "modify", "payment"]):
        return RouteDecision(route="orders", rationale="Order/payment keywords")

    return RouteDecision(route="general", rationale="Fallback")
