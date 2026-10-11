"""Shared state schema for Project 6."""

from typing import TypedDict, List, Dict, Any

class SupportState(TypedDict, total=False):

    # --------------------------------------
    # INPUT
    # --------------------------------------
    ticket_text: str

    # --------------------------------------
    # CLASSIFICATION
    # --------------------------------------
    category: str
    priority: str
    sentiment: str

    # --------------------------------------
    # CLASSIFICATION CONFIDENCE
    # --------------------------------------
    category_confidence: float
    priority_confidence: float
    sentiment_confidence: float

    # --------------------------------------
    # RAG / RETRIEVAL
    # --------------------------------------
    retrieved_docs: List[Dict[str, Any]]
    retrieval_confidence: float
    rag_category_agreement: float

    # --------------------------------------
    # CUSTOMER RESPONSE
    # --------------------------------------
    response: str

    # --------------------------------------
    # DECISION
    # --------------------------------------
    escalation_required: bool
    escalation_reason: str

    # --------------------------------------
    # HUMAN ESCALATION
    # --------------------------------------
    escalation_note: str
