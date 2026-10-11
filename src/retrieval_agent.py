"""Retrieval workflow agent extracted from V6.1."""

from .state import SupportState
from .retrieval import hybrid_retrieve

def retrieval_agent_v4(state: SupportState) -> dict:
    """
    Retrieval Agent V4

    Uses the existing RAG V2.1 hybrid retriever:

        hybrid_retrieve(query, predicted_category)

    Adds:
        - retrieval_confidence
        - rag_category_agreement
    """

    query = state["ticket_text"]
    predicted_category = state["category"]

    # --------------------------------------
    # Existing working hybrid retriever
    # --------------------------------------

    retrieved_docs = hybrid_retrieve(
        query,
        predicted_category
    )

    # --------------------------------------
    # Retrieval confidence
    # --------------------------------------

    if retrieved_docs:
        retrieval_confidence = float(
            retrieved_docs[0].get(
                "final_score",
                retrieved_docs[0].get(
                    "semantic_score",
                    0.0
                )
            )
        )
    else:
        retrieval_confidence = 0.0

    # --------------------------------------
    # RAG category agreement
    # --------------------------------------

    if retrieved_docs:

        matching_documents = sum(
            1
            for doc in retrieved_docs
            if doc.get("category")
            == predicted_category
        )

        rag_category_agreement = (
            matching_documents /
            len(retrieved_docs)
        )

    else:
        rag_category_agreement = 0.0

    return {
        "retrieved_docs": retrieved_docs,
        "retrieval_confidence": retrieval_confidence,
        "rag_category_agreement": rag_category_agreement
    }
