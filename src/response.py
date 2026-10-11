"""Response agent extracted from the Project 6 V6.1 notebook."""

from .state import SupportState


def response_agent_v2(state: SupportState) -> dict:
    """
    Response Agent V2

    Uses retrieved historical resolutions when relevant.
    Avoids presenting unrelated resolutions as answers.
    """

    ticket = state["ticket_text"]
    category = state["category"]
    priority = state["priority"]
    sentiment = state["sentiment"]
    retrieved_docs = state["retrieved_docs"]

    ticket_lower = ticket.lower()

    # --------------------------------------
    # No retrieved documents
    # --------------------------------------

    if not retrieved_docs:
        return {
            "response": (
                "We could not find enough relevant information "
                "to resolve your request. Your case should be "
                "reviewed by a support agent."
            ),
            "escalation_required": True
        }

    # --------------------------------------
    # Identify email-update requests
    # --------------------------------------

    email_update_request = (
        ("email" in ticket_lower or "e-mail" in ticket_lower)
        and any(
            word in ticket_lower
            for word in [
                "update",
                "change",
                "modify",
                "edit"
            ]
        )
    )

    # --------------------------------------
    # Use a relevant, safe response
    # --------------------------------------

    if email_update_request:
        resolution = (
            "Please sign in to your account and open Profile "
            "or Account Settings. Look for the email address "
            "option and follow the instructions to update and "
            "verify your new email address. If you cannot find "
            "this option, contact customer support for assistance."
        )

    else:
        # --------------------------------------
        # Extract historical resolution
        # --------------------------------------

        top_doc = retrieved_docs[0]
        document_text = top_doc.get("text", "")

        resolution = ""

        if "Resolution:" in document_text:
            resolution = document_text.split(
                "Resolution:", 1
            )[1]

            if "Category:" in resolution:
                resolution = resolution.split(
                    "Category:", 1
                )[0]

            resolution = resolution.strip()

        # --------------------------------------
        # Fallback
        # --------------------------------------

        if not resolution:
            resolution = (
                "Our support team will review your issue "
                "and assist you with the appropriate resolution."
            )

    # --------------------------------------
    # Customer response
    # --------------------------------------

    response = (
        f"Thank you for contacting us regarding your "
        f"{category.lower()}.\n\n"
        f"We understand that you reported: "
        f"\"{ticket}\"\n\n"
        f"Recommended next steps:\n{resolution}\n\n"
        f"We apologize for the inconvenience and appreciate "
        f"your patience."
    )

    # --------------------------------------
    # Escalation
    # --------------------------------------

    escalation_required = (
        priority.lower() == "high"
        or sentiment.lower() in [
            "negative",
            "very negative"
        ]
    )

    return {
        "response": response,
        "escalation_required": escalation_required
    }