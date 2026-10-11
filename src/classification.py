"""Classification agent extracted from the Project 6 V6.1 notebook."""
from .state import SupportState

def classification_agent_v3(state: SupportState) -> dict:
    """
    Classification Agent V3

    Predicts category, priority and sentiment,
    and exposes model confidence using the
    maximum class probability.
    """

    ticket_text = state["ticket_text"]

    # --------------------------------------
    # CATEGORY
    # --------------------------------------

    category_vector = tfidf_group.transform([ticket_text])

    category_prediction = category_model_group.predict(
        category_vector
    )[0]

    category_probabilities = (
        category_model_group.predict_proba(category_vector)[0]
    )

    category_confidence = float(
        category_probabilities.max()
    )

    # --------------------------------------
    # PRIORITY
    # --------------------------------------

    priority_vector = tfidf_priority.transform(
        [ticket_text]
    )

    priority_prediction = priority_model.predict(
        priority_vector
    )[0]

    priority_probabilities = (
        priority_model.predict_proba(priority_vector)[0]
    )

    priority_confidence = float(
        priority_probabilities.max()
    )

    # --------------------------------------
    # SENTIMENT
    # --------------------------------------

    sentiment_vector = tfidf_sentiment.transform(
        [ticket_text]
    )

    sentiment_prediction = sentiment_model_v3.predict(
    sentiment_vector
    )[0]

    sentiment_probabilities = (
        sentiment_model_v3.predict_proba(sentiment_vector)[0]
    )

    sentiment_confidence = float(
        sentiment_probabilities.max()
    )

    return {
        "category": category_prediction,
        "priority": priority_prediction,
        "sentiment": sentiment_prediction,

        "category_confidence": category_confidence,
        "priority_confidence": priority_confidence,
        "sentiment_confidence": sentiment_confidence
    }
