"""Reusable runtime for the Project 6 V6.1 support workflow."""

from pathlib import Path
import pickle

import faiss
import joblib
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

from .state import SupportState
from . import classification as classification_module
from . import retrieval as retrieval_module

from .classification import classification_agent_v3
from .retrieval import hybrid_retrieve
from .retrieval_agent import retrieval_agent_v4
from .response import response_agent_v2
from .decision import decision_agent_v3
from .escalation import escalation_agent_v3


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
VECTOR_DIR = PROJECT_ROOT / "vector_store"


# --------------------------------------------------
# 2. LOAD CLASSIFICATION ARTIFACTS
# --------------------------------------------------

tfidf_group = joblib.load(MODELS_DIR / "tfidf_group.joblib")
category_model_group = joblib.load(
    MODELS_DIR / "category_model_group.joblib"
)

tfidf_priority = joblib.load(MODELS_DIR / "tfidf_priority.joblib")
priority_model = joblib.load(MODELS_DIR / "priority_model.joblib")

tfidf_sentiment = joblib.load(MODELS_DIR / "tfidf_sentiment.joblib")
sentiment_model_v3 = joblib.load(
    MODELS_DIR / "sentiment_model_v3.joblib"
)


# --------------------------------------------------
# 3. LOAD EMBEDDING MODEL
# --------------------------------------------------

embedding_dir = VECTOR_DIR / "embedding_model"
embedding_model = SentenceTransformer(str(embedding_dir))


# --------------------------------------------------
# 4. LOAD FAISS INDEXES AND METADATA
# --------------------------------------------------

rag_index = faiss.read_index(str(VECTOR_DIR / "rag_index.faiss"))

metadata_path = VECTOR_DIR / "retrieval_metadata.pkl"

with metadata_path.open("rb") as file:
    retrieval_metadata = pickle.load(file)

rag_documents = retrieval_metadata["rag_documents"]
category_doc_indices = retrieval_metadata["category_doc_indices"]

category_indexes = {}

category_index_dir = VECTOR_DIR / "category_indexes"

for item in retrieval_metadata["saved_categories"]:
    category = item["category"]
    filename = item["filename"]

    index_path = category_index_dir / filename

    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing category index: {index_path}"
        )

    category_indexes[category] = faiss.read_index(str(index_path))


# --------------------------------------------------
# 5. CONNECT AGENT DEPENDENCIES
# --------------------------------------------------

classification_agent_v3.__globals__.update({
    "SupportState": SupportState,
    "tfidf_group": tfidf_group,
    "category_model_group": category_model_group,
    "tfidf_priority": tfidf_priority,
    "priority_model": priority_model,
    "tfidf_sentiment": tfidf_sentiment,
    "sentiment_model_v3": sentiment_model_v3,
})

hybrid_retrieve.__globals__.update({
    "faiss": faiss,
    "embedding_model": embedding_model,
    "rag_index": rag_index,
    "rag_documents": rag_documents,
    "category_indexes": category_indexes,
    "category_doc_indices": category_doc_indices,
})


# --------------------------------------------------
# 6. COMPILE THE ORIGINAL V6.1 GRAPH
# --------------------------------------------------

graph_definition_path = PROJECT_ROOT / "src" / "graph_definition.txt"

if not graph_definition_path.exists():
    raise FileNotFoundError(
        f"Graph definition not found: {graph_definition_path}"
    )

graph_source = graph_definition_path.read_text(encoding="utf-8")

runtime_namespace = {
    "StateGraph": StateGraph,
    "START": START,
    "END": END,
    "SupportState": SupportState,
    "classification_agent_v3": classification_agent_v3,
    "retrieval_agent_v4": retrieval_agent_v4,
    "response_agent_v2": response_agent_v2,
    "decision_agent_v3": decision_agent_v3,
    "escalation_agent_v3": escalation_agent_v3,
}

# Make the shared state and agents available to the graph definition.
runtime_namespace["SupportState"] = SupportState

exec(
    compile(graph_source, str(graph_definition_path), "exec"),
    runtime_namespace,
    runtime_namespace,
)

support_graph_v61 = runtime_namespace["support_graph_v61"]


# --------------------------------------------------
# 7. PUBLIC ENTRY POINT
# --------------------------------------------------

def process_ticket(ticket_text: str) -> dict:
    """Run a customer support ticket through the V6.1 graph."""

    if not isinstance(ticket_text, str) or not ticket_text.strip():
        raise ValueError("Please enter a non-empty support ticket.")

    return support_graph_v61.invoke({
        "ticket_text": ticket_text.strip()
    })


def get_support_graph():
    """Return the compiled LangGraph workflow."""

    return support_graph_v61
