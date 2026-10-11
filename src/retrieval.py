"""RAG V2.1 hybrid retrieval function extracted from the Project 6 notebook."""
from .state import SupportState

from typing import Any

def hybrid_retrieve(
    query,
    predicted_category,
    global_k=5,
    category_k=5,
    final_k=5
):
    """
    RAG V2.1 Hybrid Retrieval

    Combines:
    1. Global FAISS retrieval
    2. Predicted-category FAISS retrieval
    3. Category-aware reranking
    """

    # --------------------------------------
    # 1. Create query embedding
    # --------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    candidates = {}

    # --------------------------------------
    # 2. GLOBAL FAISS RETRIEVAL
    # --------------------------------------

    global_scores, global_indices = rag_index.search(
        query_embedding,
        global_k
    )

    for score, idx in zip(
        global_scores[0],
        global_indices[0]
    ):

        doc = rag_documents[idx]

        candidates[idx] = {
            "document_index": idx,
            "semantic_score": float(score),
            "category": doc["category"],
            "source": doc["source"],
            "document_id": doc["id"],
            "text": doc["text"]
        }

    # --------------------------------------
    # 3. CATEGORY-SPECIFIC RETRIEVAL
    # --------------------------------------

    if predicted_category in category_indexes:

        category_index = category_indexes[
            predicted_category
        ]

        category_indices = category_doc_indices[
            predicted_category
        ]

        category_scores, local_indices = (
            category_index.search(
                query_embedding,
                category_k
            )
        )

        for score, local_idx in zip(
            category_scores[0],
            local_indices[0]
        ):

            original_idx = category_indices[
                local_idx
            ]

            doc = rag_documents[original_idx]

            # Add or update candidate
            if original_idx not in candidates:

                candidates[original_idx] = {
                    "document_index": original_idx,
                    "semantic_score": float(score),
                    "category": doc["category"],
                    "source": doc["source"],
                    "document_id": doc["id"],
                    "text": doc["text"]
                }

            else:

                # Keep the stronger semantic score
                candidates[original_idx][
                    "semantic_score"
                ] = max(
                    candidates[original_idx][
                        "semantic_score"
                    ],
                    float(score)
                )

    # --------------------------------------
    # 4. RERANK
    # --------------------------------------

    for doc in candidates.values():

        category_bonus = (
            0.05
            if doc["category"] == predicted_category
            else 0.0
        )

        source_bonus = (
            0.02
            if doc["source"] == "FAQ"
            else 0.0
        )

        doc["category_bonus"] = category_bonus
        doc["source_bonus"] = source_bonus

        doc["final_score"] = (
            doc["semantic_score"]
            + category_bonus
            + source_bonus
        )

    # --------------------------------------
    # 5. Sort candidates
    # --------------------------------------

    ranked = sorted(
        candidates.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )

    # --------------------------------------
    # 6. Final Top-K
    # --------------------------------------

    results = ranked[:final_k]

    for rank, doc in enumerate(
        results,
        start=1
    ):
        doc["rank"] = rank

    return results
