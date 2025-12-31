"""
Lens Manager - Fetches and manages AWS Well-Architected Lenses.

Provides functionality to:
- Fetch lens catalog from AWS API
- Cache lens definitions locally
- Provide lens question context for agents
- Auto-detect relevant lenses from transcript content
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LensPillar:
    """Represents a pillar within a lens."""
    pillar_id: str
    pillar_name: str
    questions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LensQuestion:
    """Represents a question within a lens pillar."""
    question_id: str
    question_title: str
    question_description: str
    pillar_id: str
    choices: list[dict[str, Any]] = field(default_factory=list)
    best_practices: list[str] = field(default_factory=list)
    improvement_plan: list[str] = field(default_factory=list)
    helpful_resources: list[str] = field(default_factory=list)
    risk: str = "UNANSWERED"


@dataclass
class LensDefinition:
    """Complete lens definition with all questions and metadata."""
    lens_alias: str
    lens_arn: str
    lens_name: str
    lens_version: str
    description: str
    owner: str
    pillars: dict[str, LensPillar] = field(default_factory=dict)
    questions: dict[str, LensQuestion] = field(default_factory=dict)
    question_count: int = 0
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    def is_stale(self, ttl_hours: int = 24) -> bool:
        """Check if the lens data needs refreshing."""
        return datetime.utcnow() - self.fetched_at > timedelta(hours=ttl_hours)


# =============================================================================
# Constants
# =============================================================================

# Pillar ID to display name mapping
PILLAR_DISPLAY_NAMES: dict[str, str] = {
    "operationalExcellence": "Operational Excellence",
    "security": "Security",
    "reliability": "Reliability",
    "performanceEfficiency": "Performance Efficiency",
    "costOptimization": "Cost Optimization",
    "sustainability": "Sustainability",
    "tenantIsolation": "Tenant Isolation",
}

# Known AWS official lens aliases (discovered via API)
# These are the actual aliases from AWS API responses
OFFICIAL_LENSES: dict[str, str] = {
    "wellarchitected": "AWS Well-Architected Framework",
    "serverless": "Serverless Applications Lens",
    "softwareasaservice": "SaaS Lens",
    "saas": "SaaS Lens",
    "machinelearning": "Machine Learning Lens",
    "machine-learning": "Machine Learning Lens",
    "genai": "Generative AI Lens",
    "generative-ai": "Generative AI Lens",
    "dataanalytics": "Data Analytics Lens",
    "data-analytics": "Data Analytics Lens",
    "containerbuild": "Container Build Lens",
    "containers": "Container Build Lens",
    "iot": "IoT Lens",
    "sap": "SAP Lens",
    "financialservices": "Financial Services Industry Lens",
    "financial-services": "Financial Services Industry Lens",
    "healthcare": "Healthcare Industry Lens",
    "government": "Government Lens",
    "migration": "Migration Lens",
    "devops": "DevOps Lens",
    "connectedmobility": "Connected Mobility Lens",
    "connected-mobility": "Connected Mobility Lens",
    "mavaluecreation": "Mergers and Acquisitions Lens",
    "ma": "Mergers and Acquisitions Lens",
}

# Alias mapping: user-friendly -> AWS actual alias
ALIAS_MAPPING: dict[str, str] = {
    "generative-ai": "genai",
    "machine-learning": "machinelearning",
    "data-analytics": "dataanalytics",
    "financial-services": "financialservices",
    "connected-mobility": "connectedmobility",
    "saas": "softwareasaservice",
    "containers": "containerbuild",
    "ma": "mavaluecreation",
}

# Reverse mapping for display purposes
REVERSE_ALIAS_MAPPING: dict[str, str] = {
    "genai": "generative-ai",
}

# Keywords and patterns for lens detection
LENS_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "generative-ai": {
        "keywords": [
            "generative ai", "genai", "llm", "large language model", "foundation model",
            "bedrock", "anthropic", "claude", "gpt", "openai", "rag", "retrieval augmented",
            "vector database", "embedding", "prompt engineering", "agentic", "agent",
            "strands", "tool use", "function calling", "chain of thought", "few shot",
        ],
        "services": ["bedrock", "sagemaker", "opensearch", "kendra", "neptune"],
        "patterns": ["rag", "agent", "prompt", "embedding", "vector"],
    },
    "machine-learning": {
        "keywords": [
            "machine learning", "ml", "model training", "inference", "sagemaker",
            "training data", "feature engineering", "model deployment", "mlops",
            "model versioning", "experiment tracking", "feature store", "model registry",
        ],
        "services": ["sagemaker", "s3", "glue", "redshift", "athena"],
        "patterns": ["training", "inference", "model", "feature", "ml pipeline"],
    },
    "serverless": {
        "keywords": [
            "serverless", "lambda", "api gateway", "step functions", "eventbridge",
            "dynamodb", "s3", "cloudfront", "no servers", "function as a service",
            "faas", "event-driven", "stateless",
        ],
        "services": ["lambda", "api gateway", "step functions", "eventbridge", "dynamodb"],
        "patterns": ["lambda", "serverless", "event-driven", "stateless"],
    },
    "saas": {
        "keywords": [
            "saas", "multi-tenant", "tenant isolation", "tenant onboarding",
            "tenant tiering", "per-tenant", "tenant data", "subscription",
            "tenant management", "tenant provisioning",
        ],
        "services": ["cognito", "dynamodb", "s3", "lambda"],
        "patterns": ["tenant", "multi-tenant", "isolation", "onboarding"],
    },
    "data-analytics": {
        "keywords": [
            "data analytics", "data lake", "data warehouse", "etl", "elt",
            "redshift", "athena", "glue", "emr", "quicksight", "data pipeline",
            "analytics", "business intelligence", "bi", "data processing",
        ],
        "services": ["redshift", "athena", "glue", "emr", "quicksight", "kinesis"],
        "patterns": ["data lake", "data warehouse", "etl", "analytics"],
    },
    "containers": {
        "keywords": [
            "container", "docker", "kubernetes", "eks", "ecs", "fargate",
            "container orchestration", "pod", "service mesh", "istio",
            "container registry", "ecr",
        ],
        "services": ["eks", "ecs", "fargate", "ecr", "app mesh"],
        "patterns": ["container", "kubernetes", "docker", "pod"],
    },
    "iot": {
        "keywords": [
            "iot", "internet of things", "device", "sensor", "iot core",
            "greengrass", "iot analytics", "device management", "edge computing",
        ],
        "services": ["iot core", "greengrass", "iot analytics", "kinesis"],
        "patterns": ["iot", "device", "sensor", "edge"],
    },
    "responsible-ai": {
        "keywords": [
            "responsible ai", "ai ethics", "bias", "fairness", "transparency",
            "explainability", "model interpretability", "ai governance",
            "ethical ai", "ai safety", "harmful content", "content moderation",
        ],
        "services": ["bedrock", "sagemaker", "comprehend"],
        "patterns": ["ethics", "bias", "fairness", "transparency", "explainability"],
    },
}


# =============================================================================
# Lens Manager
# =============================================================================

class LensManager:
    """
    Manages AWS Well-Architected Lenses.
    
    Capabilities:
    - Fetches lens catalog from AWS API
    - Caches lens definitions locally
    - Provides lens question context for agents
    - Auto-detects relevant lenses from transcript content
    """

    # Class-level constants for backward compatibility
    OFFICIAL_LENSES = OFFICIAL_LENSES
    ALIAS_MAPPING = ALIAS_MAPPING
    REVERSE_ALIAS_MAPPING = REVERSE_ALIAS_MAPPING

    @classmethod
    def normalize_lens_alias(cls, alias: str) -> str:
        """
        Normalize lens alias from user-friendly format to AWS actual alias.
        
        Args:
            alias: User-provided lens alias
            
        Returns:
            Normalized AWS lens alias (always returns genai for generative-ai)
        """
        normalized = ALIAS_MAPPING.get(alias.lower(), alias.lower())

        # Ensure genai is used (not generative-ai)
        if normalized == "generative-ai":
            normalized = "genai"

        return normalized

    def __init__(
        self,
        aws_region: str = "us-east-1",
        cache_dir: str = "./knowledge_base/lenses",
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize LensManager.
        
        Args:
            aws_region: AWS region for API calls
            cache_dir: Directory for caching lens definitions
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.aws_region = aws_region
        self.cache_dir = Path(cache_dir)
        self.cache_ttl_hours = cache_ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._lens_cache: dict[str, LensDefinition] = {}

        # Initialize boto3 client lazily
        self._wa_client: Any = None

    @property
    def wa_client(self) -> Any:
        """Lazy initialization of WA Tool client."""
        if self._wa_client is None:
            try:
                self._wa_client = boto3.client(
                    "wellarchitected",
                    region_name=self.aws_region,
                )
            except Exception as e:
                logger.warning("Could not initialize WA Tool client: %s", e)
                self._wa_client = None
        return self._wa_client

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def list_available_lenses(self, lens_type: str = "AWS_OFFICIAL") -> list[dict[str, str]]:
        """
        List all available lenses from the Lens Catalog.
        
        Args:
            lens_type: "AWS_OFFICIAL", "CUSTOM_SHARED", "CUSTOM_SELF", or "ALL"
            
        Returns:
            List of lens summaries with alias, name, and description
        """
        lenses = []

        # Try to fetch from API
        if self.wa_client:
            try:
                lenses = self._fetch_lenses_from_api(lens_type)
                if lenses:
                    logger.info("Found %d %s lenses from API", len(lenses), lens_type)
                    return lenses
            except ClientError as e:
                logger.warning("Failed to list lenses from API: %s", e)

        # Fallback to known lenses
        logger.info("Using fallback list of known lenses")
        return self._get_fallback_lenses()

    def get_lens(
        self,
        lens_alias: str,
        force_refresh: bool = False,
    ) -> LensDefinition | None:
        """
        Get complete lens definition with all questions.
        
        Args:
            lens_alias: Lens alias (e.g., "serverless", "generative-ai")
            force_refresh: Force refresh from API even if cached
            
        Returns:
            LensDefinition with all questions and metadata
        """
        # Check in-memory cache first
        if lens_alias in self._lens_cache and not force_refresh:
            lens = self._lens_cache[lens_alias]
            if not lens.is_stale(self.cache_ttl_hours):
                return lens

        # Check file cache
        cache_file = self.cache_dir / f"{lens_alias}.json"
        if cache_file.exists() and not force_refresh:
            try:
                lens = self._load_from_cache(cache_file)
                if lens and not lens.is_stale(self.cache_ttl_hours):
                    self._lens_cache[lens_alias] = lens
                    return lens
            except Exception as e:
                logger.warning("Failed to load cache for %s: %s", lens_alias, e)

        # Try to fetch from API
        if self.wa_client:
            lens = self._fetch_lens_from_api(lens_alias)
            if lens:
                self._lens_cache[lens_alias] = lens
                self._save_to_cache(lens, cache_file)
                return lens

        # Fallback to schema registry
        logger.info("Falling back to schema registry for %s", lens_alias)
        return self._get_lens_from_schema(lens_alias)

    def get_lens_context_for_agents(self, lens_aliases: list[str]) -> dict[str, Any]:
        """
        Generate context data for agents based on selected lenses.
        
        Args:
            lens_aliases: List of lens aliases to include
            
        Returns:
            Combined context with all questions from selected lenses
        """
        context: dict[str, Any] = {
            "lenses": {},
            "all_questions": [],
            "pillar_summaries": {},
            "question_count": 0,
        }

        for alias in lens_aliases:
            lens = self.get_lens(alias)
            if not lens:
                logger.warning("Could not fetch lens: %s", alias)
                continue

            self._add_lens_to_context(context, alias, lens)

        context["question_count"] = len(context["all_questions"])
        return context

    def detect_relevant_lenses(
        self,
        transcript: str,
        min_confidence: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Automatically detect relevant lenses based on transcript content.
        
        Args:
            transcript: Transcript text to analyze
            min_confidence: Minimum confidence score to include a lens (0.0-1.0)
            
        Returns:
            List of detected lenses with confidence scores, sorted by confidence
        """
        transcript_lower = transcript.lower()
        detected_lenses = []

        for lens_alias, patterns in LENS_KEYWORDS.items():
            result = self._calculate_lens_confidence(lens_alias, patterns, transcript_lower)
            if result["confidence"] >= min_confidence:
                detected_lenses.append(result)

        # Always include wellarchitected as base
        detected_lenses.append({
            "lens_alias": "wellarchitected",
            "confidence": 1.0,
            "matches": ["base framework"],
            "match_count": 1,
        })

        # Sort by confidence (descending)
        detected_lenses.sort(key=lambda x: x["confidence"], reverse=True)

        logger.info("Detected %d relevant lenses from transcript", len(detected_lenses))
        for lens in detected_lenses:
            logger.info(
                "  - %s: %.2f%% confidence (%d matches)",
                lens["lens_alias"],
                lens["confidence"] * 100,
                lens["match_count"],
            )

        return detected_lenses

    def auto_select_lenses(
        self,
        transcript: str,
        max_lenses: int = 3,
    ) -> list[str]:
        """
        Automatically select the most relevant lenses based on transcript.
        
        Args:
            transcript: Transcript text to analyze
            max_lenses: Maximum number of lenses to select (excluding wellarchitected)
            
        Returns:
            List of lens aliases to use
        """
        detected = self.detect_relevant_lenses(transcript, min_confidence=0.2)

        # Always include wellarchitected
        selected = ["wellarchitected"]

        # Add top lenses (excluding wellarchitected)
        other_lenses = [l for l in detected if l["lens_alias"] != "wellarchitected"]
        for lens in other_lenses[:max_lenses]:
            if lens["confidence"] >= 0.3:
                selected.append(lens["lens_alias"])

        logger.info("Auto-selected lenses: %s", ", ".join(selected))
        return selected

    # -------------------------------------------------------------------------
    # Private Methods - API Interaction
    # -------------------------------------------------------------------------

    def _fetch_lenses_from_api(self, lens_type: str) -> list[dict[str, str]]:
        """Fetch lens list from AWS API."""
        lenses = []
        next_token = None

        # Handle lens type filtering
        lens_types = ["AWS_OFFICIAL", "CUSTOM_SHARED", "CUSTOM_SELF"] if lens_type == "ALL" else [lens_type]

        while True:
            for lt in lens_types:
                params: dict[str, Any] = {
                    "LensStatus": "PUBLISHED",
                    "MaxResults": 50,
                    "LensType": lt,
                }

                if next_token:
                    params["NextToken"] = next_token

                response = self.wa_client.list_lenses(**params)

                for lens in response.get("LensSummaries", []):
                    lens_data = self._parse_lens_summary(lens, lt)
                    if lens_data:
                        lenses.append(lens_data)

                next_token = response.get("NextToken")
                if next_token:
                    break

            if not next_token:
                break

        return lenses

    def _parse_lens_summary(
        self,
        lens: dict[str, Any],
        lens_type: str,
    ) -> dict[str, str] | None:
        """Parse lens summary from API response."""
        alias = lens.get("LensAlias")
        if not alias:
            alias = self._extract_alias_from_arn(lens.get("LensArn", ""))

        if not alias:
            return None

        return {
            "alias": alias,
            "arn": lens.get("LensArn"),
            "name": lens.get("LensName"),
            "description": lens.get("Description", ""),
            "version": lens.get("LensVersion"),
            "status": lens.get("LensStatus"),
            "owner": lens.get("Owner", "AWS"),
            "type": lens.get("LensType", lens_type),
        }

    def _extract_alias_from_arn(self, arn: str) -> str | None:
        """
        Extract lens alias from ARN.
        
        Format: arn:aws:wellarchitected::aws:lens/{alias}
        """
        if not arn:
            return None

        try:
            parts = arn.split("/")
            if len(parts) > 1:
                return parts[-1]
        except Exception:
            pass

        return None

    def _fetch_lens_from_api(self, lens_alias: str) -> LensDefinition | None:
        """Fetch lens definition from AWS WA Tool API."""
        logger.info("Fetching lens from API: %s", lens_alias)

        try:
            lens_response = self.wa_client.get_lens(LensAlias=lens_alias)
            lens_data = lens_response.get("Lens", {})

            lens_def = LensDefinition(
                lens_alias=lens_alias,
                lens_arn=lens_data.get("LensArn", ""),
                lens_name=lens_data.get("Name", ""),
                lens_version=lens_data.get("LensVersion", ""),
                description=lens_data.get("Description", ""),
                owner=lens_data.get("Owner", "AWS"),
            )

            logger.info("Lens metadata fetched: %s", lens_def.lens_name)
            return lens_def

        except ClientError as e:
            logger.warning("Failed to fetch lens %s from API: %s", lens_alias, e)
            return None

    def _get_fallback_lenses(self) -> list[dict[str, str]]:
        """Get fallback list of known lenses."""
        return [
            {
                "alias": alias,
                "name": name,
                "owner": "AWS",
                "description": f"AWS Well-Architected {name}",
                "status": "PUBLISHED",
                "type": "AWS_OFFICIAL",
            }
            for alias, name in OFFICIAL_LENSES.items()
        ]

    # -------------------------------------------------------------------------
    # Private Methods - Schema Processing
    # -------------------------------------------------------------------------

    def _get_lens_from_schema(self, lens_alias: str) -> LensDefinition | None:
        """Get lens from schema registry as fallback."""
        from .lens_schema import get_lens_schema

        schema = get_lens_schema(lens_alias)
        if not schema:
            logger.warning("Could not fetch lens: %s", lens_alias)
            return None

        lens = self._create_lens_from_schema(schema)
        if lens:
            self._lens_cache[lens_alias] = lens

        return lens

    def _create_lens_from_schema(self, schema: dict[str, Any]) -> LensDefinition | None:
        """Create lens definition from schema registry."""
        try:
            lens_def = LensDefinition(
                lens_alias=schema["lens_alias"],
                lens_arn=f"arn:aws:wellarchitected::lens/{schema['lens_alias']}",
                lens_name=schema["lens_name"],
                lens_version=schema.get("version", "1.0"),
                description=schema.get("description", ""),
                owner="AWS",
            )

            self._populate_lens_from_schema(lens_def, schema)

            lens_def.question_count = len(lens_def.questions)
            logger.info(
                "Created lens from schema: %s (%d questions)",
                lens_def.lens_name,
                lens_def.question_count,
            )
            return lens_def

        except Exception as e:
            logger.error("Failed to create lens from schema: %s", e)
            return None

    def _populate_lens_from_schema(
        self,
        lens_def: LensDefinition,
        schema: dict[str, Any],
    ) -> None:
        """Populate lens definition with pillars and questions from schema."""
        for pillar_id, pillar_data in schema.get("pillars", {}).items():
            pillar = LensPillar(
                pillar_id=pillar_id,
                pillar_name=pillar_data.get("name", pillar_id),
            )

            focus_areas = pillar_data.get("focus_areas", [])
            key_questions = pillar_data.get("key_questions", [])

            for i, (focus_area, question_text) in enumerate(zip(
                focus_areas,
                key_questions if key_questions else focus_areas,
            )):
                question_id = f"{pillar_id}-{i + 1:02d}"
                question = LensQuestion(
                    question_id=question_id,
                    question_title=question_text if isinstance(question_text, str) else focus_area,
                    question_description=focus_area,
                    pillar_id=pillar_id,
                    best_practices=[focus_area],
                )

                lens_def.questions[question_id] = question
                pillar.questions.append({
                    "QuestionId": question_id,
                    "QuestionTitle": question.question_title,
                    "QuestionDescription": question.question_description,
                })

            lens_def.pillars[pillar_id] = pillar

    def _get_pillar_name(self, pillar_id: str) -> str:
        """Convert pillar ID to display name."""
        return PILLAR_DISPLAY_NAMES.get(pillar_id, pillar_id)

    # -------------------------------------------------------------------------
    # Private Methods - Context Building
    # -------------------------------------------------------------------------

    def _add_lens_to_context(
        self,
        context: dict[str, Any],
        alias: str,
        lens: LensDefinition,
    ) -> None:
        """Add lens data to context dictionary."""
        context["lenses"][alias] = {
            "name": lens.lens_name,
            "description": lens.description,
            "version": lens.lens_version,
            "question_count": lens.question_count,
        }

        # Add questions with lens context
        for qid, question in lens.questions.items():
            q_context = {
                "lens_alias": alias,
                "lens_name": lens.lens_name,
                "question_id": qid,
                "question_title": question.question_title,
                "question_description": question.question_description,
                "pillar_id": question.pillar_id,
                "pillar_name": self._get_pillar_name(question.pillar_id),
                "best_practices": question.best_practices,
                "choices": question.choices,
            }
            context["all_questions"].append(q_context)

        # Build pillar summaries
        for pid, pillar in lens.pillars.items():
            key = f"{alias}:{pid}"
            context["pillar_summaries"][key] = {
                "lens": alias,
                "pillar_id": pid,
                "pillar_name": pillar.pillar_name,
                "question_count": len(pillar.questions),
            }

    # -------------------------------------------------------------------------
    # Private Methods - Lens Detection
    # -------------------------------------------------------------------------

    def _calculate_lens_confidence(
        self,
        lens_alias: str,
        patterns: dict[str, list[str]],
        transcript_lower: str,
    ) -> dict[str, Any]:
        """Calculate confidence score for a lens based on transcript matches."""
        score = 0.0
        matches = []

        # Check keywords
        for keyword in patterns["keywords"]:
            if keyword in transcript_lower:
                score += 0.15
                matches.append(keyword)

        # Check services
        for service in patterns["services"]:
            if service in transcript_lower:
                score += 0.2
                matches.append(service)

        # Check patterns (case-insensitive word matching)
        for pattern in patterns["patterns"]:
            pattern_words = pattern.split()
            if all(word in transcript_lower for word in pattern_words):
                score += 0.1
                matches.append(pattern)

        # Normalize score (cap at 1.0)
        confidence = min(score, 1.0)

        # Boost confidence if multiple matches
        if len(matches) > 5:
            confidence = min(confidence * 1.2, 1.0)

        return {
            "lens_alias": lens_alias,
            "confidence": round(confidence, 2),
            "matches": matches[:10],
            "match_count": len(matches),
        }

    # -------------------------------------------------------------------------
    # Private Methods - Caching
    # -------------------------------------------------------------------------

    def _save_to_cache(self, lens: LensDefinition, cache_file: Path) -> None:
        """Save lens definition to file cache."""
        try:
            data = self._serialize_lens(lens)

            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info("Cached lens %s to %s", lens.lens_alias, cache_file)

        except Exception as e:
            logger.error("Failed to cache lens: %s", e)

    def _serialize_lens(self, lens: LensDefinition) -> dict[str, Any]:
        """Serialize lens definition to dictionary for caching."""
        return {
            "lens_alias": lens.lens_alias,
            "lens_arn": lens.lens_arn,
            "lens_name": lens.lens_name,
            "lens_version": lens.lens_version,
            "description": lens.description,
            "owner": lens.owner,
            "question_count": lens.question_count,
            "fetched_at": lens.fetched_at.isoformat(),
            "pillars": {
                pid: {
                    "pillar_id": p.pillar_id,
                    "pillar_name": p.pillar_name,
                    "questions": p.questions,
                }
                for pid, p in lens.pillars.items()
            },
            "questions": {
                qid: {
                    "question_id": q.question_id,
                    "question_title": q.question_title,
                    "question_description": q.question_description,
                    "pillar_id": q.pillar_id,
                    "choices": q.choices,
                    "best_practices": q.best_practices,
                    "improvement_plan": q.improvement_plan,
                    "helpful_resources": q.helpful_resources,
                }
                for qid, q in lens.questions.items()
            },
        }

    def _load_from_cache(self, cache_file: Path) -> LensDefinition | None:
        """Load lens definition from file cache."""
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            return self._deserialize_lens(data)

        except Exception as e:
            logger.error("Failed to load cache: %s", e)
            return None

    def _deserialize_lens(self, data: dict[str, Any]) -> LensDefinition:
        """Deserialize lens definition from cached dictionary."""
        lens = LensDefinition(
            lens_alias=data["lens_alias"],
            lens_arn=data["lens_arn"],
            lens_name=data["lens_name"],
            lens_version=data["lens_version"],
            description=data["description"],
            owner=data["owner"],
            question_count=data["question_count"],
            fetched_at=datetime.fromisoformat(data["fetched_at"]),
        )

        # Reconstruct pillars
        for pid, p_data in data.get("pillars", {}).items():
            lens.pillars[pid] = LensPillar(
                pillar_id=p_data["pillar_id"],
                pillar_name=p_data["pillar_name"],
                questions=p_data["questions"],
            )

        # Reconstruct questions
        for qid, q_data in data.get("questions", {}).items():
            lens.questions[qid] = LensQuestion(
                question_id=q_data["question_id"],
                question_title=q_data["question_title"],
                question_description=q_data["question_description"],
                pillar_id=q_data["pillar_id"],
                choices=q_data.get("choices", []),
                best_practices=q_data.get("best_practices", []),
                improvement_plan=q_data.get("improvement_plan", []),
                helpful_resources=q_data.get("helpful_resources", []),
            )

        return lens


# =============================================================================
# Factory Function
# =============================================================================

def create_lens_manager(**kwargs: Any) -> LensManager:
    """
    Factory function for creating lens manager.
    
    Args:
        **kwargs: Arguments to pass to LensManager constructor
        
    Returns:
        Configured LensManager instance
    """
    return LensManager(**kwargs)