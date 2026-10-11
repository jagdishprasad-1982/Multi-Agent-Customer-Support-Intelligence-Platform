"""Streamlit interface for the Project 6 support platform."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.workflow import process_ticket


st.set_page_config(
    page_title="Customer Support Intelligence",
    page_icon="🎧",
    layout="wide",
)

st.title("Customer Support Intelligence Platform")
st.caption(
    "Multi-agent ticket classification, retrieval, response generation, "
    "and escalation."
)

with st.sidebar:
    st.header("About")
    st.write(
        "Submit a support ticket to run it through the Project 6 "
        "LangGraph workflow."
    )
    st.info(
        "Predictions and confidence scores are model outputs. "
        "Review proposed resolutions before taking customer-facing action."
    )

st.subheader("Submit a customer ticket")

ticket_text = st.text_area(
    "Describe the customer's issue",
    placeholder=(
        "Example: My order has not arrived and it is 5 days late."
    ),
    height=130,
)

if st.button("Process Ticket", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a customer ticket first.")
    else:
        with st.spinner("Processing ticket through the agent workflow..."):
            try:
                result = process_ticket(ticket_text.strip())
                st.session_state["last_ticket_result"] = result
                st.session_state["last_ticket_text"] = ticket_text.strip()

            except Exception as exc:
                st.error(
                    f"Ticket processing failed: {type(exc).__name__}: {exc}"
                )

result = st.session_state.get("last_ticket_result")

if result:
    st.divider()
    st.subheader("Ticket Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Category", result.get("category", "Unknown"))
    col2.metric("Priority", result.get("priority", "Unknown"))
    col3.metric("Sentiment", result.get("sentiment", "Unknown"))

    st.subheader("Model Confidence")

    confidence_fields = [
        ("Category", "category_confidence"),
        ("Priority", "priority_confidence"),
        ("Sentiment", "sentiment_confidence"),
        ("Retrieval", "retrieval_confidence"),
    ]

    confidence_cols = st.columns(4)

    for col, (label, key) in zip(confidence_cols, confidence_fields):
        value = result.get(key)

        if isinstance(value, (int, float)):
            col.metric(label, f"{value:.1%}")
        else:
            col.metric(label, "N/A")

    st.caption(
        "Confidence values reflect the system's scores, not guaranteed "
        "probabilities of correctness."
    )

    st.subheader("Suggested Response")
    response = result.get("response", "No response was generated.")

    st.write(response)

    st.subheader("Escalation")

    if result.get("escalation_required"):
        st.error("Escalation required")
    else:
        st.success("No escalation requested by the workflow")

    reason = result.get("escalation_reason")
    note = result.get("escalation_note")

    if reason:
        st.write("**Reason:**", reason)

    if note:
        st.write("**Escalation note:**")
        st.write(note)

    with st.expander("Retrieved documents"):
        docs = result.get("retrieved_docs", [])

        if not docs:
            st.info("No documents were returned.")
        else:
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Document {i}**")

                if isinstance(doc, dict):
                    st.write("Category:", doc.get("category", "N/A"))
                    st.write("Source:", doc.get("source", "N/A"))
                    st.write("Score:", doc.get("final_score", "N/A"))
                    st.write(doc.get("text", str(doc)))
                else:
                    st.write(str(doc))

                if i < len(docs):
                    st.divider()

    with st.expander("Technical workflow output"):
        st.json(
            {
                key: value
                for key, value in result.items()
                if key != "retrieved_docs"
            }
        )

# --------------------------------------------------
# HUMAN FEEDBACK / LEARNING LOOP
# --------------------------------------------------

from src.learning import record_feedback, load_feedback

st.divider()
st.subheader("Human Review & Learning Feedback")

st.write(
    "Record a human-reviewed resolution to build a reusable "
    "resolution history for future analysis."
)

last_ticket = st.session_state.get("last_ticket_text", "")
last_result = st.session_state.get("last_ticket_result", {})

with st.form("resolution_feedback_form"):
    feedback_ticket = st.text_area(
        "Customer ticket",
        value=last_ticket,
        key="feedback_ticket",
    )

    predicted_category = st.text_input(
        "Predicted category",
        value=str(last_result.get("category", "")),
    )

    predicted_priority = st.text_input(
        "Predicted priority",
        value=str(last_result.get("priority", "")),
    )

    suggested_response = st.text_area(
        "AI suggested response",
        value=str(last_result.get("response", "")),
    )

    human_resolution = st.text_area(
        "Human-reviewed resolution",
        placeholder="Enter the confirmed resolution.",
    )

    corrected_category = st.text_input(
        "Corrected category (optional)",
        value="",
    )

    feedback_note = st.text_input(
        "Reviewer feedback (optional)",
        value="",
    )

    feedback_submitted = st.form_submit_button("Save Human Feedback")

if feedback_submitted:
    try:
        saved_file = record_feedback(
            ticket_text=feedback_ticket,
            predicted_category=predicted_category,
            predicted_priority=predicted_priority,
            suggested_response=suggested_response,
            human_resolution=human_resolution,
            corrected_category=corrected_category,
            feedback=feedback_note,
        )

        st.success("Human feedback saved successfully.")
        st.caption(f"Stored in: {saved_file}")

    except Exception as exc:
        st.error(f"Could not save feedback: {exc}")

with st.expander("Feedback history"):
    feedback_records = load_feedback()
    st.write("Saved records:", len(feedback_records))

    if feedback_records:
        st.dataframe(feedback_records, use_container_width=True)
    else:
        st.info("No feedback records have been saved yet.")
