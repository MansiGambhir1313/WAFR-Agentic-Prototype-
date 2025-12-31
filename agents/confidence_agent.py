"""
Confidence Agent - Validates evidence and assigns confidence scores.

Anti-hallucination validation using Strands framework. Ensures WAFR answers
are properly supported by transcript evidence.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from strands import Agent, tool

from agents.config import DEFAULT_MODEL_ID
from agents.model_config import get_strands_model
from agents.utils import batch_process, extract_json_from_text, retry_with_backoff
from agents.wafr_context import get_question_context, load_wafr_schema

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Similarity thresholds
SIMILARITY_EXACT = 1.0
SIMILARITY_CASE_INSENSITIVE = 0.95
SIMILARITY_THRESHOLD = 0.6  # Minimum for acceptance
SIMILARITY_PHRASE_CAP = 0.9

# Confidence thresholds
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.5
CONFIDENCE_LOW_REVIEW = 0.4
CONFIDENCE_CLARIFICATION = 0.3

# Processing limits
MAX_TRANSCRIPT_LENGTH = 15000
BATCH_SIZE = 5
MAX_WORKERS = 3
BATCH_TIMEOUT = 120.0
MIN_PHRASE_LENGTH = 3


# =============================================================================
# Data Structures
# =============================================================================

class ConfidenceLevel(Enum):
    """Confidence level categories."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MatchType(Enum):
    """Evidence match types."""
    EXACT = "exact"
    CASE_INSENSITIVE = "case_insensitive"
    FUZZY = "fuzzy"
    PHRASE = "phrase"
    NONE = "none"


@dataclass
class EvidenceMatch:
    """Result of evidence verification."""
    verified: bool
    match_type: MatchType
    position: int = -1
    similarity: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "match_type": self.match_type.value,
            "position": self.position,
            "similarity": self.similarity,
            "confidence": self.confidence,
        }


@dataclass
class ValidationResult:
    """Result of answer validation."""
    validation_passed: bool
    confidence_score: float
    confidence_level: str
    evidence_verified: bool
    issues: list[str] = field(default_factory=list)
    requires_clarification: bool = False
    mapping_id: str | None = None
    pillar: str | None = None
    question_id: str | None = None
    clarification_request: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_passed": self.validation_passed,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level,
            "evidence_verified": self.evidence_verified,
            "issues": self.issues,
            "requires_clarification": self.requires_clarification,
            "mapping_id": self.mapping_id,
            "pillar": self.pillar,
            "question_id": self.question_id,
            "clarification_request": self.clarification_request,
        }


# =============================================================================
# System Prompt
# =============================================================================

CONFIDENCE_SYSTEM_PROMPT = """
You are a rigorous fact-checker for WAFR (AWS Well-Architected Framework Review) assessments.
Your role is to validate that each answer is supported by transcript evidence and prevent hallucinations.

VALIDATION CRITERIA:
1. Evidence Verification: Does the quote appear in the transcript (verbatim or >60% similarity)?
2. Answer Support: Does the evidence actually support the answer?
3. Interpretation Accuracy: Is the interpretation accurate and not overstated?
4. Assumption Check: Are there unsupported assumptions?
5. WAFR Alignment: Does the answer align with WAFR best practices?

CONFIDENCE SCORING:
- HIGH (0.75-1.0): Evidence found verbatim/high similarity (>80%), answer directly reflects transcript
- MEDIUM (0.5-0.74): Evidence found (60-80% similarity), reasonable interpretation of client's words
- LOW (0.0-0.49): Evidence NOT found (<60% similarity), significant inference without support

VALIDATION RULES:
- ACCEPT: Verified evidence (>=60% similarity) that reflects client's actual words
- ACCEPT: Partial answers based on real transcript content
- REJECT: Evidence not found in transcript
- REJECT: Claims not supported by what client said

OUTPUT FORMAT:
- validation_passed: true if evidence verified AND confidence >= 0.5
- confidence_score: 0.0-1.0 numeric score
- confidence_level: "high", "medium", or "low"
- evidence_verified: true if quote found in transcript
- issues: List of concerns
- requires_clarification: true if answer needs more information
"""


# =============================================================================
# Tools
# =============================================================================

@tool
def verify_evidence_in_transcript(evidence_quote: str, transcript: str) -> dict:
    """
    Verify if evidence quote exists in transcript.
    
    Args:
        evidence_quote: Claimed evidence quote
        transcript: Full transcript text
        
    Returns:
        Verification result with match type and confidence
    """
    result = _verify_evidence(evidence_quote.strip(), transcript)
    return result.to_dict()


def _verify_evidence(evidence: str, transcript: str) -> EvidenceMatch:
    """Internal evidence verification logic."""
    if not evidence:
        return EvidenceMatch(verified=False, match_type=MatchType.NONE)

    # Strategy 1: Exact match
    if evidence in transcript:
        return EvidenceMatch(
            verified=True,
            match_type=MatchType.EXACT,
            position=transcript.find(evidence),
            similarity=SIMILARITY_EXACT,
            confidence=SIMILARITY_EXACT,
        )

    evidence_lower = evidence.lower()
    transcript_lower = transcript.lower()

    # Strategy 2: Case-insensitive match
    if evidence_lower in transcript_lower:
        return EvidenceMatch(
            verified=True,
            match_type=MatchType.CASE_INSENSITIVE,
            position=transcript_lower.find(evidence_lower),
            similarity=SIMILARITY_CASE_INSENSITIVE,
            confidence=SIMILARITY_CASE_INSENSITIVE,
        )

    # Strategy 3: Fuzzy word matching
    fuzzy_result = _fuzzy_word_match(evidence_lower, transcript_lower)
    if fuzzy_result.verified:
        return fuzzy_result

    # Strategy 4: Key phrase matching
    return _phrase_match(evidence_lower, transcript_lower)


def _fuzzy_word_match(evidence: str, transcript: str) -> EvidenceMatch:
    """Check for word overlap in sliding window."""
    evidence_words = set(evidence.split())
    transcript_words = transcript.split()

    if not evidence_words:
        return EvidenceMatch(verified=False, match_type=MatchType.NONE)

    window_size = len(evidence_words)
    max_matches = 0
    best_position = -1

    for i in range(len(transcript_words) - window_size + 1):
        window = set(transcript_words[i : i + window_size])
        matches = len(evidence_words & window)
        if matches > max_matches:
            max_matches = matches
            best_position = i

    similarity = max_matches / len(evidence_words)

    if similarity >= SIMILARITY_THRESHOLD:
        return EvidenceMatch(
            verified=True,
            match_type=MatchType.FUZZY,
            position=best_position,
            similarity=similarity,
            confidence=similarity,
        )

    return EvidenceMatch(verified=False, match_type=MatchType.NONE)


def _phrase_match(evidence: str, transcript: str) -> EvidenceMatch:
    """Check for consecutive word phrases."""
    words = evidence.split()
    if len(words) < MIN_PHRASE_LENGTH:
        return EvidenceMatch(verified=False, match_type=MatchType.NONE)

    matched_words = 0
    first_position = -1

    for i in range(len(words) - MIN_PHRASE_LENGTH + 1):
        phrase = " ".join(words[i : i + MIN_PHRASE_LENGTH])
        if phrase in transcript:
            matched_words += MIN_PHRASE_LENGTH
            if first_position == -1:
                first_position = transcript.find(phrase)

    if matched_words == 0:
        return EvidenceMatch(verified=False, match_type=MatchType.NONE)

    similarity = min(matched_words / len(words), SIMILARITY_PHRASE_CAP)

    if similarity >= SIMILARITY_THRESHOLD:
        return EvidenceMatch(
            verified=True,
            match_type=MatchType.PHRASE,
            position=first_position,
            similarity=similarity,
            confidence=similarity,
        )

    return EvidenceMatch(verified=False, match_type=MatchType.NONE)


@tool
def validate_answer(
    answer: str,
    evidence: str,
    transcript: str,
    validation_passed: bool,
    confidence_score: float,
    confidence_level: str,
    evidence_verified: bool,
    issues: list[str] | None = None,
    requires_clarification: bool = False,
) -> dict:
    """
    Record validation result for an answer.
    
    Args:
        answer: The answer being validated
        evidence: Evidence quote
        transcript: Full transcript
        validation_passed: Whether validation passed
        confidence_score: Score 0.0-1.0
        confidence_level: "high", "medium", or "low"
        evidence_verified: Whether evidence found in transcript
        issues: List of concerns
        requires_clarification: Whether clarification needed
        
    Returns:
        Validation result dictionary
    """
    return {
        "validation_passed": validation_passed,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "evidence_verified": evidence_verified,
        "issues": issues or [],
        "requires_clarification": requires_clarification,
    }


# =============================================================================
# Confidence Agent
# =============================================================================

class ConfidenceAgent:
    """Agent that validates evidence and assigns confidence scores."""

    def __init__(self, wafr_schema: dict[str, Any] | None = None):
        """
        Initialize Confidence Agent.
        
        Args:
            wafr_schema: Optional WAFR schema for context
        """
        self.wafr_schema = wafr_schema or load_wafr_schema()
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent | None:
        """Create Strands agent with tools."""
        try:
            model = get_strands_model(DEFAULT_MODEL_ID)
            agent = Agent(
                system_prompt=CONFIDENCE_SYSTEM_PROMPT,
                name="ConfidenceAgent",
                **({"model": model} if model else {}),
            )
            self._register_tools(agent)
            return agent
        except Exception as e:
            logger.warning("Strands Agent init failed: %s, using direct Bedrock", e)
            return None

    def _register_tools(self, agent: Agent) -> None:
        """Register tools with agent, trying multiple methods."""
        tools = [verify_evidence_in_transcript, validate_answer]
        
        for method_name in ("add_tool", "register_tool"):
            if hasattr(agent, method_name):
                for t in tools:
                    try:
                        getattr(agent, method_name)(t)
                    except Exception as e:
                        logger.debug("Tool registration via %s failed: %s", method_name, e)
                return

    def process(
        self,
        mappings: list[dict],
        transcript: str,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Validate mappings and assign confidence scores.
        
        Args:
            mappings: List of WAFR answer mappings
            transcript: Full transcript text
            session_id: Session identifier
            
        Returns:
            Dictionary with validation results
        """
        logger.info("ConfidenceAgent: Validating %d mappings for session %s", len(mappings), session_id)

        if not mappings:
            return self._empty_result(session_id)

        transcript_sample = transcript[:MAX_TRANSCRIPT_LENGTH]
        validations = self._process_mappings(mappings, transcript_sample)
        
        return self._build_result(session_id, mappings, validations)

    def _empty_result(self, session_id: str) -> dict[str, Any]:
        """Return empty result structure."""
        return {
            "session_id": session_id,
            "summary": self._aggregate_confidence([]),
            "approved_answers": [],
            "review_needed": [],
            "clarification_needed": [],
            "all_validations": [],
            "clarification_requests": [],
            "agent": "confidence",
        }

    def _process_mappings(
        self,
        mappings: list[dict],
        transcript: str,
    ) -> list[dict]:
        """Process all mappings, using batch processing for large sets."""
        processor = lambda m: self._validate_single_mapping(m, transcript)

        if len(mappings) > BATCH_SIZE:
            return batch_process(
                mappings,
                processor,
                batch_size=BATCH_SIZE,
                max_workers=MAX_WORKERS,
                timeout=BATCH_TIMEOUT,
            )

        return [processor(m) for m in mappings]

    def _validate_single_mapping(self, mapping: dict, transcript: str) -> dict:
        """Validate a single mapping against transcript."""
        answer = mapping.get("answer_content", "")
        evidence = mapping.get("evidence_quote", "")
        question_id = mapping.get("question_id")

        question_context = self._get_question_context(question_id)
        prompt = self._build_validation_prompt(answer, evidence, question_id, question_context, transcript)

        try:
            response = self._call_agent_with_retry(prompt)
            return self._parse_validation(response, mapping)
        except Exception as e:
            logger.error("Error validating mapping: %s", e)
            return self._default_validation(mapping)

    def _get_question_context(self, question_id: str | None) -> str:
        """Get WAFR question context if available."""
        if not question_id or not self.wafr_schema:
            return ""

        context = get_question_context(question_id, self.wafr_schema)
        return f"\n\nWAFR QUESTION CONTEXT:\n{context}\n" if context else ""

    def _build_validation_prompt(
        self,
        answer: str,
        evidence: str,
        question_id: str | None,
        question_context: str,
        transcript: str,
    ) -> str:
        """Build the validation prompt."""
        return f"""
Validate this WAFR answer against the transcript:

<answer>{answer}</answer>
<claimed_evidence>{evidence}</claimed_evidence>
<question_id>{question_id or 'UNKNOWN'}</question_id>
{question_context}
<original_transcript>{transcript}</original_transcript>

VALIDATION STEPS:
1. Use verify_evidence_in_transcript() to check if evidence quote exists
2. Verify the evidence actually supports the answer
3. Check if interpretation is accurate (not overstated)
4. Assess alignment with WAFR best practices
5. Use validate_answer() to record your result

PRIORITY:
1. Verify evidence exists (similarity >= 0.6)
2. Check answer reflects what client said
3. Assess interpretation reasonableness
4. Score confidence based on evidence quality
"""

    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def _call_agent_with_retry(self, prompt: str) -> Any:
        """Call agent with retry logic."""
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        return self.agent(prompt)

    def _parse_validation(self, response: Any, mapping: dict) -> dict:
        """Parse validation response into structured result."""
        validation = self._extract_validation_dict(response)
        
        evidence_verified = validation.get("evidence_verified", False)
        confidence_score = validation.get("confidence_score", 0.0)

        # Set defaults if missing
        if "validation_passed" not in validation:
            validation["validation_passed"] = evidence_verified and confidence_score >= CONFIDENCE_LOW_REVIEW

        if "confidence_score" not in validation:
            validation["confidence_score"] = 0.5 if evidence_verified else 0.0

        if "confidence_level" not in validation:
            validation["confidence_level"] = self._score_to_level(validation["confidence_score"])

        # Add mapping context
        validation["mapping_id"] = mapping.get("id")
        validation["pillar"] = mapping.get("pillar")
        validation["question_id"] = mapping.get("question_id")

        return validation

    def _extract_validation_dict(self, response: Any) -> dict:
        """Extract validation dictionary from response."""
        if isinstance(response, dict):
            return response

        try:
            parsed = extract_json_from_text(str(response), strict=False)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.debug("JSON extraction failed: %s", e)

        return {}

    def _default_validation(self, mapping: dict) -> dict:
        """Create default validation for errors."""
        return {
            "validation_passed": False,
            "confidence_score": 0.3,
            "confidence_level": ConfidenceLevel.LOW.value,
            "evidence_verified": False,
            "issues": ["Could not validate - processing error"],
            "requires_clarification": True,
            "mapping_id": mapping.get("id"),
            "pillar": mapping.get("pillar"),
            "question_id": mapping.get("question_id"),
        }

    def _score_to_level(self, score: float) -> str:
        """Convert confidence score to level."""
        if score >= CONFIDENCE_HIGH:
            return ConfidenceLevel.HIGH.value
        if score >= CONFIDENCE_MEDIUM:
            return ConfidenceLevel.MEDIUM.value
        return ConfidenceLevel.LOW.value

    def _build_result(
        self,
        session_id: str,
        mappings: list[dict],
        validations: list[dict],
    ) -> dict[str, Any]:
        """Build final result with categorized validations."""
        approved = []
        review_needed = []
        clarification_needed = []

        for v in validations:
            evidence_verified = v.get("evidence_verified", False)
            score = v.get("confidence_score", 0)
            level = v.get("confidence_level")

            if self._is_approved(evidence_verified, level, score):
                approved.append(v)
            elif self._needs_review(evidence_verified, score):
                review_needed.append(v)
            else:
                mapping = self._find_mapping(mappings, v.get("question_id"))
                v["clarification_request"] = self._generate_clarification(v, mapping)
                clarification_needed.append(v)

        return {
            "session_id": session_id,
            "summary": self._aggregate_confidence(validations),
            "approved_answers": approved,
            "review_needed": review_needed,
            "clarification_needed": clarification_needed,
            "all_validations": validations,
            "clarification_requests": [
                v["clarification_request"]
                for v in clarification_needed
                if v.get("clarification_request")
            ],
            "agent": "confidence",
        }

    def _is_approved(self, evidence_verified: bool, level: str, score: float) -> bool:
        """Check if validation is auto-approved."""
        if not evidence_verified:
            return False
        return level == "high" or (level == "medium" and score >= CONFIDENCE_MEDIUM)

    def _needs_review(self, evidence_verified: bool, score: float) -> bool:
        """Check if validation needs manual review."""
        return (
            evidence_verified
            and CONFIDENCE_LOW_REVIEW <= score < CONFIDENCE_MEDIUM
        )

    def _find_mapping(self, mappings: list[dict], question_id: str | None) -> dict:
        """Find mapping by question ID."""
        if not question_id:
            return {}
        return next((m for m in mappings if m.get("question_id") == question_id), {})

    def _generate_clarification(self, validation: dict, mapping: dict) -> dict[str, Any]:
        """Generate clarification request for incomplete validation."""
        question_id = validation.get("question_id") or mapping.get("question_id", "UNKNOWN")
        question_text = mapping.get("question_text", "Unknown question")
        evidence_verified = validation.get("evidence_verified", False)
        confidence_score = validation.get("confidence_score", 0)
        evidence_quote = mapping.get("evidence_quote", "")
        answer_content = mapping.get("answer_content", "")

        understood = []
        if evidence_verified and evidence_quote:
            understood.append(f'Found in discussion: "{evidence_quote[:150]}..."')
        if answer_content:
            understood.append(f"We understand: {answer_content[:200]}")

        missing = []
        if not evidence_verified:
            missing.append("No clear evidence found in transcript")
        if confidence_score < CONFIDENCE_LOW_REVIEW:
            missing.append("Information is partial or unclear")

        return {
            "question_id": question_id,
            "question_text": question_text,
            "understood_parts": understood,
            "missing_parts": missing,
            "evidence_verified": evidence_verified,
            "confidence_score": confidence_score,
            "clarification_text": self._format_clarification_text(
                question_id, question_text, understood, missing
            ),
        }

    def _format_clarification_text(
        self,
        question_id: str,
        question_text: str,
        understood: list[str],
        missing: list[str],
    ) -> str:
        """Format clarification request as readable text."""
        understood_text = "\n".join(f"- {p}" for p in understood) if understood else "- Limited information available."
        missing_text = "\n".join(f"- {p}" for p in missing) if missing else "- Please provide more details."

        return f"""# Clarification Request for {question_id}

## What We Understood
{understood_text}

## What We Need
{missing_text}

## Question
**{question_text}**

## Your Input Needed
Please provide specific examples, methodology details, or relevant implementation information.
"""

    def _aggregate_confidence(self, validations: list[dict]) -> dict[str, Any]:
        """Aggregate validation results into summary."""
        total = len(validations)
        
        if total == 0:
            return {
                "total_answers": 0,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "average_score": 0.0,
                "auto_approved": 0,
                "needs_review": 0,
                "needs_clarification": 0,
                "overall_readiness": "review_required",
            }

        counts = {"high": 0, "medium": 0, "low": 0}
        for v in validations:
            level = v.get("confidence_level", "low")
            counts[level] = counts.get(level, 0) + 1

        scores = [v.get("confidence_score", 0.0) for v in validations]
        avg_score = sum(scores) / total

        return {
            "total_answers": total,
            "high_confidence": counts["high"],
            "medium_confidence": counts["medium"],
            "low_confidence": counts["low"],
            "average_score": round(avg_score, 3),
            "auto_approved": counts["high"],
            "needs_review": counts["medium"],
            "needs_clarification": counts["low"],
            "overall_readiness": "ready" if counts["low"] == 0 else "review_required",
        }


# =============================================================================
# Factory
# =============================================================================

def create_confidence_agent(wafr_schema: dict[str, Any] | None = None) -> ConfidenceAgent:
    """Factory function to create Confidence Agent."""
    return ConfidenceAgent(wafr_schema)