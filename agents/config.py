"""
Configuration for WAFR Agent System.

This module centralizes all configuration settings for the WAFR agent pipeline,
including model selection, temperature settings, scoring weights, and thresholds.

Usage:
    from agents.config import settings, ModelConfig
    
    model_id = settings.model.model_id
    region = settings.model.region
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final


# =============================================================================
# Bedrock Model Configuration
# =============================================================================

class BedrockModel(Enum):
    """
    Available Bedrock Claude models.
    
    Models with inference profile (cross-region, add "us." prefix):
    - CLAUDE_3_7_SONNET: Recommended for production
    - CLAUDE_SONNET_4: Latest Claude 4 release
    - CLAUDE_SONNET_4_5: Newest Claude 4.5
    
    Legacy models (direct access, no prefix):
    - CLAUDE_3_HAIKU: Budget option, faster responses
    - CLAUDE_3_SONNET: Fallback option
    """
    # Cross-region inference models (recommended)
    CLAUDE_3_7_SONNET = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_SONNET_4 = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_SONNET_4_5 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_3_5_SONNET_V2 = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_5_SONNET = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_3_5_HAIKU = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_3_HAIKU_PROFILE = "us.anthropic.claude-3-haiku-20240307-v1:0"
    
    # Legacy direct-access models (fallback)
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"


class Grade(Enum):
    """Grade levels with associated thresholds."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class Priority(Enum):
    """Priority levels for issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass(frozen=True)
class ModelSettings:
    """Bedrock model configuration."""
    model_id: str
    region: str
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> ModelSettings:
        """Create settings from environment variables."""
        return cls(
            model_id=os.getenv("BEDROCK_MODEL_ID", BedrockModel.CLAUDE_3_7_SONNET.value),
            region=os.getenv("AWS_REGION", "us-east-1"),
            max_tokens=int(os.getenv("BEDROCK_MAX_TOKENS", "4096")),
        )


@dataclass(frozen=True)
class AgentTemperatures:
    """
    Temperature settings for each agent.
    
    Lower temperatures = more deterministic/factual
    Higher temperatures = more creative/varied
    """
    understanding: float = 0.1  # Low: factual extraction
    mapping: float = 0.2       # Low-medium: structured mapping
    confidence: float = 0.1    # Very low: consistent validation
    scoring: float = 0.2       # Low-medium: numerical assessment
    report: float = 0.3        # Medium: narrative generation


@dataclass(frozen=True)
class ScoringWeights:
    """
    Weights for overall scoring calculation.
    
    Must sum to 1.0.
    """
    confidence: float = 0.4
    completeness: float = 0.3
    compliance: float = 0.3

    def __post_init__(self):
        total = self.confidence + self.completeness + self.compliance
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class GradeThresholds:
    """
    Score thresholds for letter grades.
    
    Scores >= threshold receive that grade.
    """
    A: int = 90
    B: int = 80
    C: int = 70
    D: int = 60
    # F: Below D threshold

    def get_grade(self, score: float) -> Grade:
        """Convert numeric score to letter grade."""
        if score >= self.A:
            return Grade.A
        if score >= self.B:
            return Grade.B
        if score >= self.C:
            return Grade.C
        if score >= self.D:
            return Grade.D
        return Grade.F


@dataclass(frozen=True)
class PriorityWeights:
    """Weights for priority calculation."""
    
    # Criticality impact on priority score
    criticality: dict[str, int] = None
    
    # Grade impact on priority score
    grade: dict[str, int] = None

    def __post_init__(self):
        # Use object.__setattr__ for frozen dataclass
        if self.criticality is None:
            object.__setattr__(self, "criticality", {
                Priority.CRITICAL.value: 40,
                Priority.HIGH.value: 30,
                Priority.MEDIUM.value: 20,
                Priority.LOW.value: 10,
            })
        if self.grade is None:
            object.__setattr__(self, "grade", {
                Grade.F.value: -30,
                Grade.D.value: -15,
                Grade.C.value: 0,
                Grade.B.value: 15,
                Grade.A.value: 30,
            })


@dataclass(frozen=True)
class ProcessingSettings:
    """Settings for text processing."""
    max_transcript_segment: int = 5000
    max_tokens: int = 4096


@dataclass(frozen=True)
class PathSettings:
    """File path configuration."""
    schema_path: Path

    @classmethod
    def from_env(cls) -> PathSettings:
        """Create path settings from environment or defaults."""
        default_path = Path(__file__).parent.parent / "schemas" / "wafr-schema.json"
        schema_path = Path(os.getenv("WAFR_SCHEMA_PATH", str(default_path)))
        return cls(schema_path=schema_path)


# =============================================================================
# Main Settings Container
# =============================================================================

@dataclass(frozen=True)
class Settings:
    """
    Central configuration container for WAFR Agent System.
    
    Access settings via the global `settings` instance:
        from agents.config import settings
        
        model_id = settings.model.model_id
        temp = settings.temperatures.confidence
    """
    model: ModelSettings
    temperatures: AgentTemperatures
    scoring: ScoringWeights
    grades: GradeThresholds
    priority: PriorityWeights
    processing: ProcessingSettings
    paths: PathSettings

    @classmethod
    def load(cls) -> Settings:
        """Load settings from environment and defaults."""
        return cls(
            model=ModelSettings.from_env(),
            temperatures=AgentTemperatures(),
            scoring=ScoringWeights(),
            grades=GradeThresholds(),
            priority=PriorityWeights(),
            processing=ProcessingSettings(),
            paths=PathSettings.from_env(),
        )


# =============================================================================
# Global Settings Instance
# =============================================================================

settings: Final[Settings] = Settings.load()


# =============================================================================
# Backward Compatibility Exports
# =============================================================================
# These maintain compatibility with existing code that imports individual constants.

DEFAULT_MODEL_ID: Final[str] = settings.model.model_id
BEDROCK_REGION: Final[str] = settings.model.region

UNDERSTANDING_AGENT_TEMPERATURE: Final[float] = settings.temperatures.understanding
MAPPING_AGENT_TEMPERATURE: Final[float] = settings.temperatures.mapping
CONFIDENCE_AGENT_TEMPERATURE: Final[float] = settings.temperatures.confidence
SCORING_AGENT_TEMPERATURE: Final[float] = settings.temperatures.scoring
REPORT_AGENT_TEMPERATURE: Final[float] = settings.temperatures.report

MAX_TRANSCRIPT_SEGMENT_LENGTH: Final[int] = settings.processing.max_transcript_segment
MAX_TOKENS_DEFAULT: Final[int] = settings.processing.max_tokens

CONFIDENCE_WEIGHT: Final[float] = settings.scoring.confidence
COMPLETENESS_WEIGHT: Final[float] = settings.scoring.completeness
COMPLIANCE_WEIGHT: Final[float] = settings.scoring.compliance

GRADE_A_THRESHOLD: Final[int] = settings.grades.A
GRADE_B_THRESHOLD: Final[int] = settings.grades.B
GRADE_C_THRESHOLD: Final[int] = settings.grades.C
GRADE_D_THRESHOLD: Final[int] = settings.grades.D

PRIORITY_CRITICALITY_WEIGHTS: Final[dict[str, int]] = settings.priority.criticality
PRIORITY_GRADE_WEIGHTS: Final[dict[str, int]] = settings.priority.grade

WAFR_SCHEMA_PATH: Final[str] = str(settings.paths.schema_path)


# =============================================================================
# Utility Functions
# =============================================================================

def get_grade(score: float) -> Grade:
    """
    Convert numeric score to letter grade.
    
    Args:
        score: Numeric score (0-100)
        
    Returns:
        Grade enum value
    """
    return settings.grades.get_grade(score)


def get_model_id(model: BedrockModel | str | None = None) -> str:
    """
    Get model ID string, with fallback to default.
    
    Args:
        model: BedrockModel enum, model ID string, or None for default
        
    Returns:
        Model ID string
    """
    if model is None:
        return settings.model.model_id
    if isinstance(model, BedrockModel):
        return model.value
    return model


def validate_config() -> list[str]:
    """
    Validate configuration settings.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check schema path exists
    if not settings.paths.schema_path.exists():
        errors.append(f"WAFR schema not found: {settings.paths.schema_path}")
    
    # Check temperature ranges
    for name, temp in [
        ("understanding", settings.temperatures.understanding),
        ("mapping", settings.temperatures.mapping),
        ("confidence", settings.temperatures.confidence),
        ("scoring", settings.temperatures.scoring),
        ("report", settings.temperatures.report),
    ]:
        if not 0.0 <= temp <= 1.0:
            errors.append(f"Temperature {name} out of range: {temp}")
    
    return errors