"""Minimal human-feedback learning loop for Project 6."""

import csv
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"

FEEDBACK_FILE = DATABASE_DIR / "resolution_feedback.csv"

FIELDS = [
    "timestamp",
    "ticket_text",
    "predicted_category",
    "predicted_priority",
    "suggested_response",
    "human_resolution",
    "corrected_category",
    "feedback",
]


def record_feedback(
    ticket_text: str,
    predicted_category: str,
    predicted_priority: str,
    suggested_response: str,
    human_resolution: str,
    corrected_category: str = "",
    feedback: str = "",
) -> str:
    """Store a human-reviewed resolution for future analysis."""

    if not ticket_text.strip():
        raise ValueError("Ticket text cannot be empty.")

    if not human_resolution.strip():
        raise ValueError("Human-reviewed resolution cannot be empty.")

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = FEEDBACK_FILE.exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticket_text": ticket_text.strip(),
        "predicted_category": predicted_category,
        "predicted_priority": predicted_priority,
        "suggested_response": suggested_response,
        "human_resolution": human_resolution.strip(),
        "corrected_category": corrected_category,
        "feedback": feedback,
    }

    with FEEDBACK_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        if not file_exists or FEEDBACK_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)

    return str(FEEDBACK_FILE)


def load_feedback() -> list:
    """Load previously recorded feedback entries."""

    if not FEEDBACK_FILE.exists():
        return []

    with FEEDBACK_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))
