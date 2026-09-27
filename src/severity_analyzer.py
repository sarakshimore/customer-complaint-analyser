import re


HIGH_TERMS = [
    "fraud",
    "scam",
    "identity theft",
    "stolen identity",
    "stolen",
    "unauthorized transaction",
    "unauthorized transactions",
    "unauthorized payment",
    "unauthorized payments",
    "unauthorized account",
    "account wasn't mine",
    "account was not mine",
    "lawsuit",
    "legal action",
    "attorney",
    "lawyer",
    "court",
    "foreclosure",
    "eviction",
    "bankruptcy",
    "harassment",
    "threat",
    "threatened",
    "illegal",
    "wrongfully",
    "stolen money",
]


MEDIUM_TERMS = [
    "charged",
    "charge",
    "fee",
    "interest",
    "payment",
    "blocked",
    "closed",
    "denied",
    "incorrect",
    "error",
    "problem",
    "delay",
    "dispute",
    "refund",
    "refunded",
    "overcharged",
    "billing",
    "complaint",
]


# --------------------------------------------
# Resolution-based escalation
# --------------------------------------------

ESCALATION_PATTERNS = [
    r"\b\d+\s+times\b",
    r"\bmultiple\s+times\b",
    r"\bseveral\s+times\b",
    r"\brepeatedly\b",
    r"\bagain\s+and\s+again\b",
    r"\bno\s+response\b",
    r"\bno\s+one\s+helped\b",
    r"\bnobody\s+has\s+helped\b",
    r"\bnobody\s+helped\b",
    r"\bstill\s+haven'?t\b",
    r"\bstill\s+not\b",
    r"\bnot\s+resolved\b",
    r"\bnot\s+been\s+resolved\b",
    r"\bunresolved\b",
    r"\bno\s+resolution\b",
    r"\bwithout\s+resolution\b",
    r"\bmanager\b",
    r"\bsupervisor\b",
]


# --------------------------------------------
# Risk-based escalation
# --------------------------------------------

RISK_ESCALATION_TERMS = [
    "fraud",
    "scam",
    "identity theft",
    "stolen identity",
    "unauthorized transaction",
    "unauthorized transactions",
    "unauthorized payment",
    "unauthorized payments",
    "unauthorized account",
    "account wasn't mine",
    "account was not mine",
    "stolen money",
    "lawsuit",
    "legal action",
    "attorney",
    "lawyer",
    "court",
    "foreclosure",
    "eviction",
    "bankruptcy",
]


def calculate_severity(text):

    text = text.lower().strip()

    score = 0

    # --------------------------------------------
    # High-severity indicators
    # --------------------------------------------

    for term in HIGH_TERMS:

        if term in text:
            score += 3

    # --------------------------------------------
    # Medium-severity indicators
    # --------------------------------------------

    for term in MEDIUM_TERMS:

        if term in text:
            score += 1

    # --------------------------------------------
    # Resolution-based escalation
    # --------------------------------------------

    resolution_escalation = False

    for pattern in ESCALATION_PATTERNS:

        if re.search(pattern, text):

            score += 3
            resolution_escalation = True
            break

    # --------------------------------------------
    # Risk-based escalation
    # --------------------------------------------

    risk_escalation = False

    for term in RISK_ESCALATION_TERMS:

        if term in text:

            risk_escalation = True
            break

    # --------------------------------------------
    # Financial impact
    # --------------------------------------------

    if (
        "$" in text
        or "dollar" in text
        or "dollars" in text
    ):

        score += 1

    # --------------------------------------------
    # Duration / unresolved issue
    # --------------------------------------------

    if re.search(
        r"\b\d+\s+(day|days|week|weeks|month|months)\b",
        text
    ):

        score += 1

    # --------------------------------------------
    # Final severity
    # --------------------------------------------

    if score >= 6:

        severity = "Critical"

    elif score >= 4:

        severity = "High"

    elif score >= 1:

        severity = "Medium"

    else:

        severity = "Low"

    # --------------------------------------------
    # Final escalation decision
    # --------------------------------------------

    escalation_detected = (
        risk_escalation
        or resolution_escalation
    )

    return {
        "severity": severity,
        "score": score,
        "escalation_detected": escalation_detected
    }