"""
Gap Detection Agent - Identifies unanswered WAFR questions.

Uses Strands framework to detect gaps in WAFR coverage and prioritize
which questions need answers most urgently.
"""

import logging
from typing import Any

from strands import Agent, tool

from agents.config import DEFAULT_MODEL_ID
from agents.model_config import get_strands_model
from agents.wafr_context import get_wafr_context_summary, load_wafr_schema

logger = logging.getLogger(__name__)


# =============================================================================
# Priority Scoring Constants
# =============================================================================

CRITICALITY_SCORES: dict[str, int] = {
    "critical": 40,
    "high": 30,
    "medium": 20,
    "low": 10,
}

# Weight multipliers for priority calculation
PILLAR_COVERAGE_WEIGHT = 0.3
HRI_INDICATOR_WEIGHT = 20
BEST_PRACTICE_WEIGHT = 2
BEST_PRACTICE_MAX = 10
MAX_PRIORITY_SCORE = 100.0


# =============================================================================
# System Prompt
# =============================================================================

GAP_DETECTION_BASE_PROMPT = """
You are an expert in AWS Well-Architected Framework Reviews (WAFR). Your job is to 
identify gaps - questions that haven't been adequately answered based on the 
transcript and current answers.

GAP DETECTION PROCESS:
1. Compare answered questions against complete WAFR schema
2. Identify unanswered questions (gaps)
3. Calculate priority score for each gap based on:
   - Question criticality (critical=40, high=30, medium=20, low=10)
   - Pillar coverage (lower coverage = higher priority)
   - High Risk Issue (HRI) indicators (presence increases priority)
   - Best practice count (more practices = higher priority)

PRIORITY CALCULATION:
- Criticality weight: 40 points max
- Pillar coverage weight: 30 points max (inverse: lower coverage = higher priority)
- HRI indicator weight: 20 points max
- Best practice count: 10 points max
- Total: 0-100 (higher = more urgent)

GAP IDENTIFICATION:
- A gap is a WAFR question that has NO answer or only partial answer
- Check transcript for context hints (keywords, related services)
- Prioritize critical and high-criticality questions
- Focus on pillars with low coverage

Use identify_gap() to record each gap with priority score.
"""


def get_gap_detection_system_prompt(wafr_schema: dict[str, Any] | None = None) -> str:
    """
    Generate enhanced system prompt with WAFR context.
    
    Args:
        wafr_schema: Optional WAFR schema for additional context
        
    Returns:
        Complete system prompt string
    """
    if not wafr_schema:
        return GAP_DETECTION_BASE_PROMPT

    wafr_context = get_wafr_context_summary(wafr_schema)
    return f"{GAP_DETECTION_BASE_PROMPT}\n\n{wafr_context}\n\nUse this WAFR context to identify all gaps comprehensively."


# =============================================================================
# Tools
# =============================================================================

@tool
def calculate_priority_score(
    question_id: str,
    criticality: str,
    pillar_coverage: float,
    has_hri_indicators: bool,
    best_practice_count: int,
) -> float:
    """
    Calculate priority score for a gap (0-100). Higher score = more urgent.
    
    Args:
        question_id: Question identifier
        criticality: "critical", "high", "medium", or "low"
        pillar_coverage: Current coverage percentage for pillar (0-100)
        has_hri_indicators: Whether question has HRI indicators
        best_practice_count: Number of best practices for question
        
    Returns:
        Priority score (0-100)
    """
    score = 0.0

    # Criticality weight (40 points max)
    score += CRITICALITY_SCORES.get(criticality, CRITICALITY_SCORES["medium"])

    # Pillar coverage weight (30 points max) - lower coverage = higher priority
    score += (100 - pillar_coverage) * PILLAR_COVERAGE_WEIGHT

    # HRI indicator weight (20 points max)
    if has_hri_indicators:
        score += HRI_INDICATOR_WEIGHT

    # Best practice count (10 points max)
    score += min(best_practice_count * BEST_PRACTICE_WEIGHT, BEST_PRACTICE_MAX)

    return min(score, MAX_PRIORITY_SCORE)


@tool
def identify_gap(
    question_id: str,
    question_text: str,
    pillar: str,
    criticality: str,
    priority_score: float,
    context_hint: str | None = None,
) -> dict[str, Any]:
    """
    Record an identified gap.
    
    Args:
        question_id: Question identifier
        question_text: Full question text
        pillar: Pillar ID
        criticality: Criticality level
        priority_score: Calculated priority score
        context_hint: Optional context from transcript
        
    Returns:
        Gap dictionary with all metadata
    """
    return {
        "question_id": question_id,
        "question_text": question_text,
        "pillar": pillar,
        "criticality": criticality,
        "priority_score": priority_score,
        "context_hint": context_hint,
        "status": "pending",
    }


# =============================================================================
# Gap Detection Agent
# =============================================================================

class GapDetectionAgent:
    """Agent that detects gaps in WAFR coverage."""

    def __init__(
        self,
        wafr_schema: dict[str, Any] | None = None,
        lens_context: dict[str, Any] | None = None,
    ):
        """
        Initialize Gap Detection Agent.
        
        Args:
            wafr_schema: Complete WAFR question schema
            lens_context: Optional lens context for multi-lens support
        """
        if wafr_schema is None:
            wafr_schema = load_wafr_schema()

        self.wafr_schema = wafr_schema or self._load_default_schema()
        self.lens_context = lens_context or {}
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent | None:
        """Create and configure Strands agent with tools."""
        system_prompt = get_gap_detection_system_prompt(self.wafr_schema)

        try:
            model = get_strands_model(DEFAULT_MODEL_ID)
            agent = Agent(
                system_prompt=system_prompt,
                name="GapDetectionAgent",
                **({"model": model} if model else {}),
            )
            self._register_tools(agent)
            return agent

        except Exception as e:
            logger.warning("Strands Agent initialization issue: %s, using direct Bedrock", e)
            return None

    def _register_tools(self, agent: Agent) -> None:
        """Register tools with agent, trying available methods."""
        tools = [calculate_priority_score, identify_gap]

        for method_name in ("add_tool", "register_tool"):
            if not hasattr(agent, method_name):
                continue

            try:
                for t in tools:
                    getattr(agent, method_name)(t)
                return
            except Exception as e:
                logger.warning("Could not add tools via %s: %s", method_name, e)

    def process(
        self,
        answered_questions: list[str],
        pillar_coverage: dict[str, float],
        session_id: str,
        transcript: str | None = None,
    ) -> dict[str, Any]:
        """
        Identify gaps in WAFR coverage.
        
        Args:
            answered_questions: List of question IDs that have answers
            pillar_coverage: Dict mapping pillar IDs to coverage percentages
            session_id: Session identifier
            transcript: Optional transcript for context hints
            
        Returns:
            Dictionary with identified gaps sorted by priority
        """
        logger.info("GapDetectionAgent: Detecting gaps for session %s", session_id)

        all_questions = self._get_all_questions()
        answered_set = set(answered_questions)
        gaps = []

        for question in all_questions:
            question_id = question["id"]

            if question_id in answered_set:
                continue

            # This is a gap - calculate priority and create gap record
            gap = self._create_gap_record(question, pillar_coverage, transcript)
            gaps.append(gap)

        # Sort by priority (highest first)
        gaps.sort(key=lambda x: x["priority_score"], reverse=True)

        return {
            "session_id": session_id,
            "total_gaps": len(gaps),
            "gaps": gaps,
            "agent": "gap_detection",
        }

    def _create_gap_record(
        self,
        question: dict[str, Any],
        pillar_coverage: dict[str, float],
        transcript: str | None,
    ) -> dict[str, Any]:
        """
        Create a gap record for an unanswered question.
        
        Args:
            question: Question data from schema
            pillar_coverage: Current coverage by pillar
            transcript: Optional transcript for context hints
            
        Returns:
            Gap record with priority score
        """
        pillar = question.get("pillar_id", "UNKNOWN")
        criticality = question.get("criticality", "medium")
        has_hri = len(question.get("hri_indicators", [])) > 0
        bp_count = len(question.get("best_practices", []))
        coverage = pillar_coverage.get(pillar, 0.0)

        priority_score = calculate_priority_score(
            question_id=question["id"],
            criticality=criticality,
            pillar_coverage=coverage,
            has_hri_indicators=has_hri,
            best_practice_count=bp_count,
        )

        context_hint = self._find_context_hint(question, transcript) if transcript else None

        gap = identify_gap(
            question_id=question["id"],
            question_text=question["text"],
            pillar=pillar,
            criticality=criticality,
            priority_score=priority_score,
            context_hint=context_hint,
        )

        gap["question_data"] = question
        return gap

    def _get_all_questions(self) -> list[dict[str, Any]]:
        """
        Get all WAFR questions from schema and lens context.
        
        Returns:
            List of all questions with pillar and lens metadata
        """
        all_questions = []

        # Get standard WAFR questions
        all_questions.extend(self._get_schema_questions())

        # Add lens-specific questions
        all_questions.extend(self._get_lens_questions())

        return all_questions

    def _get_schema_questions(self) -> list[dict[str, Any]]:
        """Extract questions from WAFR schema."""
        questions = []

        if not self.wafr_schema or "pillars" not in self.wafr_schema:
            return questions

        for pillar in self.wafr_schema["pillars"]:
            pillar_id = pillar.get("id", "UNKNOWN")

            for question in pillar.get("questions", []):
                question["pillar_id"] = pillar_id
                question["lens_alias"] = "wellarchitected"
                questions.append(question)

        return questions

    def _get_lens_questions(self) -> list[dict[str, Any]]:
        """Extract questions from lens context."""
        questions = []

        if not self.lens_context or not self.lens_context.get("all_questions"):
            return questions

        for lens_q in self.lens_context["all_questions"]:
            # Skip standard WAFR questions (already included)
            if lens_q.get("lens_alias") == "wellarchitected":
                continue

            question_dict = {
                "id": lens_q.get("question_id", ""),
                "text": lens_q.get("question_title", ""),
                "pillar_id": lens_q.get("pillar_id", "UNKNOWN"),
                "criticality": "medium",
                "keywords": lens_q.get("best_practices", [])[:5],
                "best_practices": lens_q.get("best_practices", []),
                "hri_indicators": [],
                "lens_alias": lens_q.get("lens_alias", ""),
                "lens_name": lens_q.get("lens_name", ""),
            }
            questions.append(question_dict)

        return questions

    def _find_context_hint(
        self,
        question: dict[str, Any],
        transcript: str,
    ) -> str | None:
        """
        Find related context in transcript for a question.
        
        Uses keyword matching from question schema to identify
        relevant discussion in the transcript.
        
        Args:
            question: Question with keywords and related services
            transcript: Full transcript text
            
        Returns:
            Context hint string or None if no matches found
        """
        if not transcript or not question:
            return None

        keywords = question.get("keywords", [])
        related_services = question.get("related_services", [])
        search_terms = keywords + related_services

        if not search_terms:
            return None

        transcript_lower = transcript.lower()
        found_terms = [
            term for term in search_terms
            if term.lower() in transcript_lower
        ]

        if not found_terms:
            return None

        return f"Related terms found in transcript: {', '.join(found_terms[:3])}"

    def _load_default_schema(self) -> dict[str, Any]:
        """
        Load default WAFR schema structure.
        
        Returns:
            Empty schema structure as fallback
        """
        return {"pillars": []}


# =============================================================================
# Factory Function
# =============================================================================

def create_gap_detection_agent(
    wafr_schema: dict[str, Any] | None = None,
    lens_context: dict[str, Any] | None = None,
) -> GapDetectionAgent:
    """
    Factory function to create Gap Detection Agent.
    
    Args:
        wafr_schema: Optional WAFR schema
        lens_context: Optional lens context for multi-lens support
        
    Returns:
        Configured GapDetectionAgent instance
    """
    return GapDetectionAgent(wafr_schema=wafr_schema, lens_context=lens_context)