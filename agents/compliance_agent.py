from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping


ComplianceVerdict = Literal["REWRITE_REQUIRED", "APPROVED"]


@dataclass(frozen=True)
class ComplianceFinding:
    kind: Literal[
        "unsupported_claim",
        "missing_citations",
        "policy_mismatch",
        "sensitive_info",
    ]
    message: str


class ComplianceAgent:
    """Checks a resolution draft for evidence compliance.

    Input:
    - resolution_draft: str
    - retrieved_evidence: list of policy chunks (each chunk should contain at least `excerpt` and `citation`)

    Checks:
    - unsupported claims (policy-like statements not supported by evidence excerpts)
    - missing citations
    - policy mismatch (citations in draft not present in retrieved evidence)
    - sensitive info

    Output:
    - "REWRITE_REQUIRED" if any issues
    - else "APPROVED"

    Notes:
    - This is a conservative, rule-based guardrail.
    - It is not a semantic entailment verifier.
    """

    def check(
        self,
        *,
        resolution_draft: str,
        retrieved_evidence: list[Mapping[str, Any]] | None,
    ) -> ComplianceVerdict:
        verdict, _findings = self.check_with_findings(
            resolution_draft=resolution_draft,
            retrieved_evidence=retrieved_evidence,
        )
        return verdict

    def check_with_findings(
        self,
        *,
        resolution_draft: str,
        retrieved_evidence: list[Mapping[str, Any]] | None,
    ) -> tuple[ComplianceVerdict, list[ComplianceFinding]]:
        draft = (resolution_draft or "").strip()
        evidence = [dict(e) for e in (retrieved_evidence or [])]
        findings: list[ComplianceFinding] = []

        if not draft:
            return "REWRITE_REQUIRED", [ComplianceFinding(kind="unsupported_claim", message="Empty draft")]

        evidence_text = "\n\n".join(str(e.get("excerpt") or "") for e in evidence).strip()
        evidence_norm = _normalize(evidence_text)

        # 1) Sensitive info
        sensitive = _find_sensitive_info(draft)
        if sensitive:
            findings.append(
                ComplianceFinding(
                    kind="sensitive_info",
                    message=f"Sensitive info detected: {', '.join(sorted(set(sensitive)))}",
                )
            )

        # 2) Citations present?
        cited_pairs = _extract_citations_from_draft(draft)
        evidence_pairs = _evidence_citation_pairs(evidence)

        # If the draft includes a Citations section but it's empty/None, flag.
        if _has_citations_header(draft) and not cited_pairs:
            findings.append(
                ComplianceFinding(
                    kind="missing_citations",
                    message="Citations section present but no citations found",
                )
            )

        # 3) Policy mismatch: cited chunks not in evidence
        if cited_pairs and evidence_pairs:
            missing = [p for p in cited_pairs if p not in evidence_pairs]
            if missing:
                findings.append(
                    ComplianceFinding(
                        kind="policy_mismatch",
                        message=f"Draft cites chunks not present in retrieved evidence: {missing}",
                    )
                )

        # 4) Missing citations for included evidence excerpts
        # If draft contains evidence-like excerpts but no citations at all, fail.
        if evidence_text and not cited_pairs and _looks_like_policy_text(draft):
            findings.append(
                ComplianceFinding(
                    kind="missing_citations",
                    message="Policy-like text present but no citations found",
                )
            )

        # 5) Unsupported claims: scan for policy-like sentences that are not supported by evidence
        # We allow:
        # - customer-service boilerplate
        # - clarifying questions
        # - internal next steps
        # We flag sentences that appear to make a policy commitment/rule but cannot be found in evidence.
        unsupported = _unsupported_policy_claims(draft, evidence_norm)
        for s in unsupported:
            findings.append(
                ComplianceFinding(
                    kind="unsupported_claim",
                    message=f"Unsupported policy-like claim: {s}",
                )
            )

        verdict: ComplianceVerdict = "APPROVED" if not findings else "REWRITE_REQUIRED"
        return verdict, findings


def check_compliance(*, resolution_draft: str, retrieved_evidence: list[Mapping[str, Any]] | None) -> ComplianceVerdict:
    """Functional convenience wrapper."""

    return ComplianceAgent().check(resolution_draft=resolution_draft, retrieved_evidence=retrieved_evidence)


def _normalize(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _has_citations_header(draft: str) -> bool:
    return "5. citations" in (draft or "").lower()


def _extract_citations_from_draft(draft: str) -> list[tuple[str, int | None]]:
    """Extract (doc, chunk_id) pairs from common formats.

    Supported patterns:
    - "- doc.txt (chunk_id=11)"
    - "[doc.txt | chunk_id=11]"
    """

    pairs: list[tuple[str, int | None]] = []

    # Bracket form: [doc | chunk_id=11]
    for m in re.finditer(r"\[([^\]|]+?)\s*\|\s*chunk_id\s*=\s*(\d+)\s*\]", draft):
        doc = m.group(1).strip()
        cid = int(m.group(2))
        pairs.append((doc, cid))

    # Bullet form: - doc (chunk_id=11)
    for m in re.finditer(r"-\s+([^\n\r]+?)\s*\(\s*chunk_id\s*=\s*(\d+)\s*\)", draft):
        doc = m.group(1).strip()
        cid = int(m.group(2))
        pairs.append((doc, cid))

    # Deduplicate preserving order
    out: list[tuple[str, int | None]] = []
    seen: set[tuple[str, int | None]] = set()
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    return out


def _evidence_citation_pairs(evidence: list[dict[str, Any]]) -> set[tuple[str, int | None]]:
    pairs: set[tuple[str, int | None]] = set()

    for e in evidence:
        citation = e.get("citation") or {}
        doc = citation.get("doc")
        chunk_id = citation.get("chunk_id")
        if not doc:
            meta = e.get("metadata") or {}
            doc = meta.get("source") or meta.get("doc_id")
        if doc is None:
            continue
        try:
            cid = int(chunk_id) if chunk_id is not None else None
        except Exception:
            cid = None
        pairs.add((str(doc), cid))

    return pairs


def _looks_like_policy_text(draft: str) -> bool:
    # Heuristic: contains language that implies a rule/requirement or a commitment.
    d = (draft or "").lower()
    return any(
        p in d
        for p in [
            "refund",
            "return",
            "eligible",
            "non-refundable",
            "non‑refundable",
            "within ",
            "business day",
            "policy",
            "required",
            "we may",
            "we will",
        ]
    )


def _unsupported_policy_claims(draft: str, evidence_norm: str) -> list[str]:
    # Split into sentences-ish units.
    # We intentionally keep this simple: each line is evaluated too.
    text = (draft or "").strip()
    if not text:
        return []

    candidates: list[str] = []

    # Remove policy excerpts blocks: lines directly under [doc | chunk_id=]
    # We still allow other statements, but this avoids flagging pasted evidence.
    lines = text.splitlines()
    filtered_lines: list[str] = []
    in_excerpt = False
    for ln in lines:
        if re.match(r"\[[^\]]+\|\s*chunk_id\s*=\s*\d+\s*\]", ln.strip()):
            in_excerpt = True
            filtered_lines.append(ln)
            continue
        # Excerpt blocks typically follow immediately and can be multi-line.
        # Heuristic: stop excerpt when we hit an empty line after being in excerpt.
        if in_excerpt and not ln.strip():
            in_excerpt = False
            filtered_lines.append(ln)
            continue
        if in_excerpt:
            # skip excerpt content from claim analysis
            continue
        filtered_lines.append(ln)

    filtered = "\n".join(filtered_lines)

    # Candidate lines: policy commitments and rules outside excerpt blocks.
    policy_patterns = [
        r"\bwe will\b",
        r"\bwe can\b",
        r"\byou (?:are|re) eligible\b",
        r"\byou may\b",
        r"\bnot eligible\b",
        r"\bnon[-‑]returnable\b",
        r"\bnon[-‑]refundable\b",
        r"\bwithin\s+\d+\s+(?:day|days|hours|business day|business days)\b",
        r"\brefund\b",
        r"\breturn\b",
        r"\bexchange\b",
        r"\bcancel\b",
        r"\bcancellation\b",
        r"\bchargeback\b",
        r"\bpolicy\b",
    ]

    for raw_line in filtered.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Skip classification/meta key-value lines (not policy claims)
        lowered = line.lower()
        if lowered.startswith("- issue_type:") or lowered.startswith("- triage_issue_type:") or lowered.startswith(
            "- triage_confidence:"
        ):
            continue
        if lowered.startswith("issue type:"):
            continue

        # Skip rationale quote markers (these are explicit citations/quotes, not new claims)
        if line.lower().startswith("- evidence excerpt:"):
            continue
        # Skip headings and section titles
        if re.match(r"^\d+\.", line) or line.endswith(":"):
            continue
        # Skip obvious questions
        if line.endswith("?"):
            continue
        # Skip internal notes markers
        if line.lower().startswith("- check ") or line.lower().startswith("- verify "):
            continue

        if any(re.search(p, line.lower()) for p in policy_patterns):
            candidates.append(line)

    unsupported: list[str] = []
    for c in candidates:
        c_norm = _normalize(c)
        # If a close-enough substring appears in evidence, accept.
        # Also accept non-policy boilerplate.
        if not c_norm:
            continue
        if "thanks for reaching out" in c_norm:
            continue
        if "i’m reviewing" in c_norm or "i'm reviewing" in c_norm:
            continue

        if c_norm in evidence_norm:
            continue

        # Permit statements that are explicitly conditional/requests (not policy commitments)
        if any(
            c_norm.startswith(p)
            for p in [
                "to proceed, please confirm",
                "once i have",
                "to route this correctly",
                "relevant policy excerpts",
            ]
        ):
            continue

        unsupported.append(c)

    return unsupported


def _find_sensitive_info(draft: str) -> list[str]:
    d = draft or ""
    hits: list[str] = []

    # Email addresses
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", d):
        hits.append("email")

    # Phone numbers (rough)
    if re.search(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b", d):
        hits.append("phone")

    # Card numbers (very rough; avoid false positives by requiring 13-19 digits)
    if re.search(r"\b\d{13,19}\b", d):
        hits.append("payment_card_number")

    # Full street address hints (heuristic)
    if re.search(r"\b\d+\s+[A-Za-z0-9.-]+\s+(street|st|road|rd|avenue|ave|blvd|lane|ln|drive|dr)\b", d, flags=re.IGNORECASE):
        hits.append("street_address")

    # Government ID hints
    if re.search(r"\b(ssn|social security|aadhaar|pan\s+number|passport)\b", d, flags=re.IGNORECASE):
        hits.append("government_id")

    return hits
