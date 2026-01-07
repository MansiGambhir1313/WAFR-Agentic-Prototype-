"""
Storage module for WAFR HITL workflow.

Provides:
- ReviewStorage: Abstract base class for review session persistence
- InMemoryReviewStorage: In-memory implementation for development/testing
- FileReviewStorage: File-based implementation for persistence
"""

from storage.review_storage import (
    ReviewStorage,
    InMemoryReviewStorage,
    FileReviewStorage,
    create_review_storage,
)

__all__ = [
    "ReviewStorage",
    "InMemoryReviewStorage",
    "FileReviewStorage",
    "create_review_storage",
]

