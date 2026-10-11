"""Decision agent extracted from the Project 6 V6.1 notebook."""
from .state import SupportState

def decision_agent_v3(state: SupportState) -> dict:
    """
    Decision Agent V3

    Escalation rules:

    1. High priority -> escalate
    2. Low category confidence -> escalate
    3. Weak RAG category agreement -> escalate
    4. Low retrieval confidence -> escalate
    5. Otherwise -> resolve

    The escalation reason is built only from
    conditions that are actually true.
    """

    category_confidence = state.get(
        "category_confidence",
        0.0
    )

    retrieval_confidence = state.get(
        "retrieval_confidence",
        0.0
    )

    rag_category_agreement = state.get(
        "rag_category_agreement",
        0.0
    )

    priority = state.get(
        "priority",
        ""
    )

    sentiment = state.get(
        "sentiment",
        ""
    )

    reasons = []

    # --------------------------------------
    # RULE 1 — HIGH PRIORITY
    # --------------------------------------

    if priority.lower() == "high":
        reasons.append(
            "High priority ticket"
        )

    # --------------------------------------
    # RULE 2 — LOW CATEGORY CONFIDENCE
    # --------------------------------------

    if category_confidence < 0.50:
        reasons.append(
            f"Low classifier confidence "
            f"({category_confidence:.2f})"
        )

    # --------------------------------------
    # RULE 3 — WEAK RAG AGREEMENT
    # --------------------------------------

    if rag_category_agreement < 0.60:
        reasons.append(
            f"Weak RAG category agreement "
            f"({rag_category_agreement:.2f})"
        )

    # --------------------------------------
    # RULE 4 — LOW RETRIEVAL CONFIDENCE
    # --------------------------------------

    if retrieval_confidence < 0.60:
        reasons.append(
            f"Low retrieval confidence "
            f"({retrieval_confidence:.2f})"
        )

    # --------------------------------------
    # FINAL DECISION
    # --------------------------------------

    if reasons:

        escalation_required = True

        escalation_reason = (
            "; ".join(reasons)
        )

    else:

        escalation_required = False

        escalation_reason = (
            "Classification and RAG evidence "
            "are sufficiently consistent; "
            "no immediate human escalation "
            "required."
        )

    return {
        "escalation_required":
            escalation_required,

        "escalation_reason":
            escalation_reason
    }
