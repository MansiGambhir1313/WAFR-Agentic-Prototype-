"""
Multi-Agent System for Automated WAFR Processing

This module provides the core agents and orchestration for WAFR assessments,
including the HITL (Human-in-the-Loop) workflow for AI-generated answer validation.

Key Components:
- WafrOrchestrator: Main pipeline coordinator
- AnswerSynthesisAgent: AI answer generation for gap questions
- ReviewOrchestrator: HITL review workflow management
- ReviewStorage: Persistence for review sessions
"""
__version__ = "1.1.0"

# Core agents
from wafr.agents.answer_synthesis_agent import (
    AnswerSynthesisAgent,
    create_answer_synthesis_agent,
)

# HITL components
from wafr.agents.review_orchestrator import (
    ReviewOrchestrator,
    ReviewSession,
    create_review_orchestrator,
)

# Configuration
from wafr.agents.config import (
    settings,
    hitl_settings,
    HITL_AUTO_APPROVE_THRESHOLD,
    HITL_QUICK_REVIEW_THRESHOLD,
    HITL_MIN_AUTHENTICITY_SCORE,
)

# Error classes
from wafr.agents.errors import (
    WAFRAgentError,
    SynthesisError,
    ReviewError,
    ValidationError,
    FinalizationError,
    SessionNotFoundError,
    ReviewItemNotFoundError,
)

__all__ = [
    # Version
    "__version__",
    # Core agents
    "AnswerSynthesisAgent",
    "create_answer_synthesis_agent",
    # HITL
    "ReviewOrchestrator",
    "ReviewSession",
    "create_review_orchestrator",
    # Config
    "settings",
    "hitl_settings",
    "HITL_AUTO_APPROVE_THRESHOLD",
    "HITL_QUICK_REVIEW_THRESHOLD",
    "HITL_MIN_AUTHENTICITY_SCORE",
    # Errors
    "WAFRAgentError",
    "SynthesisError",
    "ReviewError",
    "ValidationError",
    "FinalizationError",
    "SessionNotFoundError",
    "ReviewItemNotFoundError",
]
