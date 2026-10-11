"""Escalation agent extracted from the Project 6 V6.1 notebook."""
from .state import SupportState

def escalation_agent_v3(state: SupportState) -> dict:
    """
    Escalation Agent V3

    Preserves:
    - escalation_required
    - escalation_reason

    Generates:
    - escalation_note
    """

    ticket_text = state.get(
        "ticket_text",
        ""
    )

    category = state.get(
        "category",
        ""
    )

    priority = state.get(
        "priority",
        ""
    )

    sentiment = state.get(
        "sentiment",
        ""
    )

    response = state.get(
        "response",
        ""
    )

    escalation_required = state.get(
        "escalation_required",
        False
    )

    escalation_reason = state.get(
        "escalation_reason",
        "Escalation required"
    )

    # --------------------------------------
    # Human handoff note
    # --------------------------------------

    escalation_note = f"""
ESCALATION REQUIRED

Customer Issue: {ticket_text}

Category: {category}
Priority: {priority}
Sentiment: {sentiment}

Reason: {escalation_reason}

AI Generated Response:
{response}

Action: Human support agent should review and take appropriate action.
""".strip()

    return {
        "escalation_required":
            escalation_required,

        "escalation_reason":
            escalation_reason,

        "escalation_note":
            escalation_note
    }
