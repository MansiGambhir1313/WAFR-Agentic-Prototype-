"""
Answer Synthesis Agent - Generates intelligent answers for gap questions.

Uses LLM reasoning to synthesize answers for unanswered WAFR questions based on
transcript context, extracted insights, and AWS best practices.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from agents.config import BEDROCK_REGION, DEFAULT_MODEL_ID
from agents.utils import retry_with_backoff, extract_json_from_text
from agents.wafr_context import get_question_context, load_wafr_schema
from models.synthesized_answer import (
    SynthesizedAnswer,
    SynthesisMethod,
    EvidenceQuote,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

SYNTHESIS_TEMPERATURE = 0.4  # Balanced creativity and consistency
SYNTHESIS_MAX_TOKENS = 2500
MAX_EVIDENCE_QUOTES = 3
MAX_RELATED_INSIGHTS = 5
MAX_RELATED_ANSWERS = 3
SYNTHESIS_BATCH_SIZE = 5  # Process gaps in batches
SYNTHESIS_TIMEOUT = 120.0  # Timeout for synthesis calls (2 minutes)

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.50
LOW_CONFIDENCE_THRESHOLD = 0.25


# =============================================================================
# System Prompt
# =============================================================================

SYNTHESIS_BASE_PROMPT = """
You are an AWS Well-Architected Framework expert. Your task is to GENERATE an intelligent answer for a WAFR question by COMBINING transcript insights with logical inference.

CRITICAL INSTRUCTION: You MUST use BOTH transcript insights (evidence) AND logical inference (reasoning) together to generate comprehensive answers.

SYNTHESIS STRATEGY:
1. START with transcript insights - extract all relevant evidence from the transcript and insights
2. APPLY logical inference - use your reasoning capabilities to extend insights to answer the question
3. COMBINE both - merge evidence-based facts with inferential reasoning to create complete answers
4. Use AWS best practices as guidance when inference is needed
5. Clearly distinguish what comes from transcript vs what is inferred

SYNTHESIS PROCESS:
Step 1: EVIDENCE GATHERING
- Identify all relevant insights from transcript
- Extract direct evidence quotes
- Note any patterns or architectural decisions mentioned

Step 2: LOGICAL INFERENCE
- Apply logical reasoning based on the evidence
- Infer reasonable conclusions from the patterns identified
- Use AWS Well-Architected best practices to guide inference
- Consider workload type, industry patterns, and architecture patterns

Step 3: SYNTHESIS
- Combine evidence with inference to create comprehensive answer
- Use evidence to ground the answer in reality
- Use inference to fill gaps and provide complete coverage
- Ensure the answer is both evidence-backed and logically sound

IMPORTANT: 
- ALWAYS combine insights with inference - never use only one
- Use transcript insights as the foundation
- Use inference to extend and complete the answer
- Clearly indicate what is evidence-based vs inferred in your reasoning chain

CONFIDENCE LEVELS:
- HIGH (0.75-1.0): Strong evidence in transcript + solid inference supports answer
- MEDIUM (0.50-0.74): Some evidence + reasonable inference
- LOW (0.25-0.49): Minimal evidence + inference based on patterns
- VERY_LOW (0.0-0.24): Best practice default with minimal context

OUTPUT FORMAT:
You must return ONLY valid JSON in this exact structure:
{
  "synthesized_answer": "Your detailed answer combining evidence from transcript insights with logical inference. Be comprehensive and specific.",
  "reasoning_chain": [
    "Step 1: Evidence from transcript insights (what was found)",
    "Step 2: Logical inference applied (how insights are extended)",
    "Step 3: Combination of evidence + inference (how they work together)",
    "Step 4: Final synthesized answer (complete answer)"
  ],
  "assumptions": [
    "Assumption 1 with justification",
    "Assumption 2 with justification"
  ],
  "confidence": 0.0-1.0,
  "confidence_justification": "Why this confidence level - reference both evidence strength AND inference quality",
  "evidence_quotes": [
    {"text": "relevant quote from transcript", "location": "source", "relevance": "why relevant"}
  ],
  "requires_attention": [
    "Specific point that human reviewer should verify",
    "Another point needing confirmation"
  ],
  "synthesis_method": "EVIDENCE_BASED or INFERENCE or BEST_PRACTICE"
}
"""


def get_synthesis_system_prompt(wafr_schema: Optional[Dict] = None) -> str:
    """
    Generate enhanced system prompt with WAFR context.
    
    Args:
        wafr_schema: Optional WAFR schema for additional context
        
    Returns:
        Complete system prompt string
    """
    base_prompt = SYNTHESIS_BASE_PROMPT
    
    if wafr_schema:
        from agents.wafr_context import get_wafr_context_summary
        wafr_context = get_wafr_context_summary(wafr_schema)
        return f"{base_prompt}\n\n{wafr_context}\n\nUse this WAFR context to guide answer synthesis."
    
    return base_prompt


# =============================================================================
# Answer Synthesis Agent
# =============================================================================

class AnswerSynthesisAgent:
    """
    Generates intelligent answers for gap questions using LLM reasoning.
    
    The agent NEVER asks the customer to provide answers directly.
    Instead, it synthesizes answers using:
    1. Transcript context and extracted insights
    2. WAFR best practices and guidance
    3. Industry/workload type inference
    4. LLM reasoning and domain knowledge
    """
    
    def __init__(
        self,
        wafr_schema: Optional[Dict] = None,
        lens_context: Optional[Dict] = None,
        region_name: str = BEDROCK_REGION,
    ):
        """
        Initialize Answer Synthesis Agent.
        
        Args:
            wafr_schema: Optional WAFR schema for context
            lens_context: Optional lens context for multi-lens support
            region_name: AWS region for Bedrock client
        """
        if wafr_schema is None:
            wafr_schema = load_wafr_schema()
        
        self.wafr_schema = wafr_schema
        self.lens_context = lens_context or {}
        self.region_name = region_name
        self.model_id = DEFAULT_MODEL_ID
        self._bedrock_client = None
        
        logger.info("AnswerSynthesisAgent initialized")
    
    @property
    def bedrock(self) -> Any:
        """Lazily initialize Bedrock client on first use."""
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client(
                "bedrock-runtime", region_name=self.region_name
            )
        return self._bedrock_client
    
    def synthesize_gaps(
        self,
        gaps: List[Dict],
        transcript: str,
        insights: List[Dict],
        validated_answers: List[Dict],
        session_id: str,
    ) -> List[SynthesizedAnswer]:
        """
        Generate answers for ALL gap questions.
        
        Args:
            gaps: List of unanswered WAFR questions (from gap detection)
            transcript: Original workshop transcript
            insights: Extracted architecture insights
            validated_answers: Already answered questions for context
            session_id: Session identifier
            
        Returns:
            List of synthesized answers ready for human review
        """
        logger.info(f"Synthesizing answers for {len(gaps)} gaps (session: {session_id})")
        
        synthesized = []
        
        # Sort gaps by criticality (HIGH first, then by priority score)
        sorted_gaps = sorted(
            gaps,
            key=lambda g: (
                {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(
                    g.get("criticality", "MEDIUM").upper(), 2
                ),
                -g.get("priority_score", 0),  # Negative for descending
            ),
        )
        
        # Process gaps in batches
        for i in range(0, len(sorted_gaps), SYNTHESIS_BATCH_SIZE):
            batch = sorted_gaps[i : i + SYNTHESIS_BATCH_SIZE]
            logger.info(f"Processing batch {i // SYNTHESIS_BATCH_SIZE + 1} ({len(batch)} gaps)")
            
            for gap in batch:
                try:
                    answer = self._synthesize_single_answer(
                        gap=gap,
                        transcript=transcript,
                        insights=insights,
                        validated_answers=validated_answers,
                        other_synthesized=synthesized,  # Use already synthesized for context
                    )
                    synthesized.append(answer)
                    logger.info(
                        f"Synthesized answer for {gap['question_id']} "
                        f"(confidence: {answer.confidence:.2f}, method: {answer.synthesis_method.value})"
                    )
                except Exception as e:
                    logger.error(f"Failed to synthesize {gap.get('question_id', 'unknown')}: {e}")
                    # Create low-confidence fallback answer
                    synthesized.append(self._create_fallback_answer(gap, str(e)))
        
        logger.info(f"Successfully synthesized {len(synthesized)} answers")
        return synthesized
    
    def _synthesize_single_answer(
        self,
        gap: Dict,
        transcript: str,
        insights: List[Dict],
        validated_answers: List[Dict],
        other_synthesized: List[SynthesizedAnswer],
    ) -> SynthesizedAnswer:
        """Generate answer for a single gap question."""
        question_id = gap.get("question_id", "")
        question_text = gap.get("question_text", "")
        pillar = gap.get("pillar", "")
        criticality = gap.get("criticality", "MEDIUM")
        
        # Build rich context for synthesis
        context = self._build_synthesis_context(
            gap, transcript, insights, validated_answers, other_synthesized
        )
        
        # Get WAFR best practice guidance for this question
        best_practice_guidance = get_question_context(question_id, self.wafr_schema) or ""
        
        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(gap, context, best_practice_guidance)
        
        # Invoke LLM
        response_text = self._invoke_synthesis_model(prompt)
        
        # Parse response
        return self._parse_synthesis_response(response_text, gap)
    
    def _build_synthesis_context(
        self,
        gap: Dict,
        transcript: str,
        insights: List[Dict],
        validated_answers: List[Dict],
        other_synthesized: List[SynthesizedAnswer],
    ) -> Dict[str, Any]:
        """Build comprehensive context for answer synthesis using Claude reasoning."""
        pillar = gap.get("pillar", "")
        question_id = gap.get("question_id", "")
        
        # Use Claude to intelligently build context (with fallback to keyword matching)
        try:
            # Extract relevant transcript sections using Claude
            relevant_transcript = self._extract_relevant_sections_claude(transcript, gap)
        except Exception as e:
            logger.warning(f"Claude transcript extraction failed, using fallback: {e}")
            relevant_transcript = self._extract_relevant_sections_fallback(transcript, gap)
        
        try:
            # Find related insights using Claude semantic understanding
            related_insights = self._find_related_insights_claude(insights, gap, pillar)[:MAX_RELATED_INSIGHTS]
        except Exception as e:
            logger.warning(f"Claude insight matching failed, using fallback: {e}")
            related_insights = self._find_related_insights_fallback(insights, gap, pillar)[:MAX_RELATED_INSIGHTS]
        
        try:
            # Find related answered questions using Claude
            related_answers = self._find_related_answers_claude(validated_answers, gap, pillar)[:MAX_RELATED_ANSWERS]
        except Exception as e:
            logger.warning(f"Claude answer matching failed, using fallback: {e}")
            related_answers = self._find_related_answers_fallback(validated_answers, gap, pillar)[:MAX_RELATED_ANSWERS]
        
        try:
            # Infer workload characteristics using Claude reasoning
            workload_profile = self._infer_workload_profile_claude(transcript, insights, validated_answers)
        except Exception as e:
            logger.warning(f"Claude workload inference failed, using fallback: {e}")
            workload_profile = self._infer_workload_profile_fallback(insights, validated_answers)
        
        return {
            "relevant_transcript": relevant_transcript,
            "related_insights": related_insights,
            "related_answers": related_answers,
            "workload_profile": workload_profile,
            "pillar": pillar,
            "transcript": transcript,  # Store full transcript for insight+inference combination
        }
    
    def _combine_insights_with_inference(
        self,
        insights: List[Dict],
        transcript: str,
        question: Dict,
        workload_profile: Dict,
    ) -> Dict[str, Any]:
        """
        Combine transcript insights with Claude's inference capabilities.
        
        This method ensures Claude uses BOTH insights (evidence) AND inference
        (reasoning) together to answer questions comprehensively.
        
        Args:
            insights: Extracted architecture insights from transcript
            transcript: Original workshop transcript
            question: WAFR question to answer
            workload_profile: Inferred workload characteristics
            
        Returns:
            Combined context dictionary with insights and inference
        """
        question_text = question.get("question_text", "")
        question_id = question.get("question_id", "")
        pillar = question.get("pillar", "")
        
        # Prepare insights summary
        insights_summary = []
        for i, insight in enumerate(insights[:15]):  # Top 15 most relevant
            insights_summary.append({
                "type": insight.get("insight_type", ""),
                "content": insight.get("content", "")[:200],
                "quote": insight.get("transcript_quote", "")[:150],
                "pillar": insight.get("pillar", ""),
            })
        
        prompt = f"""You are analyzing a WAFR question and need to combine transcript insights with logical inference to generate a comprehensive answer.

QUESTION:
{question_text}

Question ID: {question_id}
Pillar: {pillar}

TRANSCRIPT INSIGHTS (Evidence from transcript):
{json.dumps(insights_summary, indent=2)[:2000]}

TRANSCRIPT EXCERPT (first 3000 chars):
{transcript[:3000]}

WORKLOAD PROFILE:
{json.dumps(workload_profile, indent=2)}

TASK:
Analyze the transcript insights and use logical inference to determine how they relate to this question. Return a JSON object with:

{{
  "evidence_summary": "Summary of relevant evidence found in transcript insights",
  "inference_applied": "Logical inferences you can make from the evidence",
  "combined_approach": "How evidence and inference work together to answer the question",
  "key_insights_used": [0, 2, 5, ...],  // indices of insights most relevant
  "inference_steps": [
    "Inference step 1 based on evidence",
    "Inference step 2 extending the evidence",
    "Inference step 3 completing the answer"
  ]
}}

Return ONLY the JSON, no other text."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1500,
                    "temperature": 0.3,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            
            result = json.loads(response["body"].read())
            response_text = result.get("content", [{}])[0].get("text", "")
            
            # Parse JSON response
            parsed = extract_json_from_text(response_text)
            if isinstance(parsed, dict):
                # Get insights used
                insight_indices = parsed.get("key_insights_used", [])
                insights_used = [
                    insights[i] for i in insight_indices
                    if 0 <= i < len(insights)
                ]
                
                return {
                    "evidence_summary": parsed.get("evidence_summary", ""),
                    "inference_applied": parsed.get("inference_applied", ""),
                    "combined_approach": parsed.get("combined_approach", ""),
                    "insights_used": insights_used,
                    "inference_steps": parsed.get("inference_steps", []),
                }
            
            # Fallback if parsing fails
            return {
                "evidence_summary": "Transcript insights analyzed",
                "inference_applied": "Logical inference applied",
                "combined_approach": "Combining evidence with inference",
                "insights_used": insights[:5],
                "inference_steps": ["Applying inference to extend evidence"],
            }
            
        except Exception as e:
            logger.warning(f"Failed to combine insights with inference: {e}. Using direct insights.")
            # Fallback: use insights directly
            return {
                "evidence_summary": "Using transcript insights directly",
                "inference_applied": "Basic inference from insights",
                "combined_approach": "Direct use of insights with minimal inference",
                "insights_used": insights[:5],
                "inference_steps": ["Using insights as evidence"],
            }
    
    def _build_synthesis_prompt(
        self, gap: Dict, context: Dict, best_practice_guidance: str
    ) -> str:
        """Build the LLM prompt for answer synthesis using insights + inference."""
        question_id = gap.get("question_id", "")
        question_text = gap.get("question_text", "")
        pillar = gap.get("pillar", "")
        criticality = gap.get("criticality", "MEDIUM")
        
        # Get combined insights + inference context
        insights = context.get("related_insights", [])
        transcript = context.get("transcript", "")  # We'll need to pass this
        workload_profile = context.get("workload_profile", {})
        
        # Combine insights with inference (if we have transcript)
        combined_context = None
        if transcript and insights:
            try:
                combined_context = self._combine_insights_with_inference(
                    insights=insights,
                    transcript=transcript,
                    question=gap,
                    workload_profile=workload_profile,
                )
            except Exception as e:
                logger.warning(f"Failed to combine insights with inference: {e}")
        
        # Format context for prompt
        transcript_excerpt = context.get("relevant_transcript", "")[:2000]  # Limit length
        insights_json = json.dumps(insights, indent=2)[:1500]
        answers_json = json.dumps(context.get("related_answers", []), indent=2)[:1000]
        workload_json = json.dumps(workload_profile, indent=2)
        
        # Build combined context section if available
        combined_section = ""
        if combined_context:
            combined_section = f"""
### Evidence + Inference Combination
**Evidence Summary**: {combined_context.get("evidence_summary", "")}
**Inference Applied**: {combined_context.get("inference_applied", "")}
**Combined Approach**: {combined_context.get("combined_approach", "")}
**Inference Steps**:
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(combined_context.get("inference_steps", [])))}
"""
        
        prompt = f"""You are an AWS Well-Architected Framework expert. Your task is to GENERATE an intelligent answer by COMBINING transcript insights (evidence) with logical inference (reasoning).

CRITICAL: You MUST use BOTH insights from transcript AND logical inference together. Start with evidence, then apply inference to extend and complete the answer.

## WAFR Question
- **Question ID**: {question_id}
- **Pillar**: {pillar}
- **Question**: {question_text}
- **Criticality**: {criticality}

## AWS Best Practice Guidance
{best_practice_guidance[:1500] if best_practice_guidance else "Standard AWS Well-Architected Framework best practices apply."}

## Available Context

### Relevant Transcript Excerpts
{transcript_excerpt if transcript_excerpt else "No directly relevant transcript excerpts found."}

### Related Insights Extracted (EVIDENCE)
{insights_json if insights_json else "[]"}

{combined_section}

### Related Answered Questions (for reference)
{answers_json if answers_json else "[]"}

### Inferred Workload Profile
{workload_json}

## Your Task

Generate a comprehensive answer by COMBINING insights (evidence) with inference (reasoning). Follow this EXACT JSON structure:

{{
  "synthesized_answer": "Your detailed answer combining evidence from insights with logical inference. Be comprehensive and specific.",
  "reasoning_chain": [
    "Step 1: Evidence from transcript insights (what insights tell us)",
    "Step 2: Logical inference applied (how we extend the evidence)",
    "Step 3: Combination of evidence + inference (how they work together)",
    "Step 4: Final synthesized answer (complete answer)"
  ],
  "assumptions": [
    "Assumption 1 with justification",
    "Assumption 2 with justification"
  ],
  "confidence": 0.0-1.0,
  "confidence_justification": "Why this confidence level - reference BOTH evidence strength AND inference quality",
  "evidence_quotes": [
    {{"text": "relevant quote from transcript", "location": "source", "relevance": "why relevant"}}
  ],
  "requires_attention": [
    "Specific point that human reviewer should verify",
    "Another point needing confirmation"
  ],
  "synthesis_method": "EVIDENCE_BASED or INFERENCE or BEST_PRACTICE"
}}

CRITICAL RULES:
- ALWAYS combine insights (evidence) with inference (reasoning) - use BOTH
- Start with what insights tell us (evidence)
- Apply logical inference to extend and complete the answer
- Be EXPLICIT about what comes from transcript vs what is inferred
- Align with AWS Well-Architected best practices
- Be specific - mention actual AWS services and configurations
- The answer will be reviewed by a human - make their job easier

Generate the JSON response now:"""
        
        return prompt
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def _invoke_synthesis_model(self, prompt: str) -> str:
        """Invoke Bedrock model for answer synthesis with timeout protection."""
        from agents.utils import timeout_wrapper
        
        system_prompt = get_synthesis_system_prompt(self.wafr_schema)
        
        # Wrap the Bedrock call with timeout protection
        def _call_bedrock():
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": SYNTHESIS_MAX_TOKENS,
                    "temperature": SYNTHESIS_TEMPERATURE,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            
            result = json.loads(response["body"].read())
            return result.get("content", [{}])[0].get("text", "")
        
        try:
            return timeout_wrapper(_call_bedrock, SYNTHESIS_TIMEOUT)
            
        except TimeoutError as e:
            logger.warning(f"Answer synthesis timed out after {SYNTHESIS_TIMEOUT}s: {e}")
            raise
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            
            if error_code in [
                "UnrecognizedClientException",
                "InvalidClientTokenId",
                "InvalidUserID.NotFound",
            ]:
                logger.error(
                    f"Bedrock synthesis failed due to invalid credentials: {error_msg}"
                )
                from agents.utils import validate_aws_credentials
                _, cred_error_msg = validate_aws_credentials()
                logger.error(f"\n{cred_error_msg}")
                raise
            elif error_code == "ExpiredToken":
                logger.error(f"Bedrock synthesis failed due to expired token: {error_msg}")
                raise
            else:
                logger.error(f"Bedrock synthesis failed: {error_code} - {error_msg}")
                raise
        except Exception as e:
            logger.error(f"Bedrock synthesis failed: {e}")
            raise
    
    def _parse_synthesis_response(
        self, response_text: str, gap: Dict
    ) -> SynthesizedAnswer:
        """Parse LLM response into SynthesizedAnswer object."""
        try:
            # Extract JSON from response
            parsed = extract_json_from_text(response_text)
            
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a valid JSON object")
            
            # Extract fields
            synthesized_answer = parsed.get("synthesized_answer", "")
            reasoning_chain = parsed.get("reasoning_chain", [])
            assumptions = parsed.get("assumptions", [])
            confidence = float(parsed.get("confidence", 0.5))
            confidence_justification = parsed.get("confidence_justification", "")
            evidence_quotes_data = parsed.get("evidence_quotes", [])
            requires_attention = parsed.get("requires_attention", [])
            synthesis_method_str = parsed.get("synthesis_method", "INFERENCE")
            
            # Determine synthesis method
            try:
                synthesis_method = SynthesisMethod(synthesis_method_str)
            except ValueError:
                # Infer method from confidence
                if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                    synthesis_method = SynthesisMethod.EVIDENCE_BASED
                elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
                    synthesis_method = SynthesisMethod.INFERENCE
                elif confidence >= LOW_CONFIDENCE_THRESHOLD:
                    synthesis_method = SynthesisMethod.BEST_PRACTICE
                else:
                    synthesis_method = SynthesisMethod.FALLBACK
            
            # Convert evidence quotes
            evidence_quotes = []
            for eq_data in evidence_quotes_data[:MAX_EVIDENCE_QUOTES]:
                evidence_quotes.append(
                    EvidenceQuote(
                        text=eq_data.get("text", ""),
                        location=eq_data.get("location", ""),
                        relevance=eq_data.get("relevance", ""),
                    )
                )
            
            # Create SynthesizedAnswer object
            return SynthesizedAnswer(
                question_id=gap.get("question_id", ""),
                pillar=gap.get("pillar", ""),
                question_text=gap.get("question_text", ""),
                synthesized_answer=synthesized_answer,
                criticality=gap.get("criticality", "MEDIUM"),
                reasoning_chain=reasoning_chain,
                assumptions=assumptions,
                confidence=confidence,
                confidence_justification=confidence_justification,
                synthesis_method=synthesis_method,
                evidence_quotes=evidence_quotes,
                related_insights=[],  # Will be filled if needed
                requires_attention=requires_attention,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse synthesis response: {e}. Creating fallback answer.")
            return self._create_fallback_answer(gap, f"Parse error: {str(e)}")
    
    def _create_fallback_answer(self, gap: Dict, error: str) -> SynthesizedAnswer:
        """Create a low-confidence fallback answer when synthesis fails."""
        pillar = gap.get("pillar", "")
        
        return SynthesizedAnswer(
            question_id=gap.get("question_id", ""),
            pillar=pillar,
            question_text=gap.get("question_text", ""),
            synthesized_answer=(
                f"Unable to synthesize a confident answer for this question. "
                f"Following AWS Well-Architected Framework best practices for {pillar} pillar is recommended. "
                f"Human input required to provide accurate answer."
            ),
            criticality=gap.get("criticality", "MEDIUM"),
            reasoning_chain=["Synthesis failed", f"Error: {error}"],
            assumptions=["No assumptions could be made due to synthesis failure"],
            confidence=0.1,
            confidence_justification="Synthesis failed - requires human input",
            synthesis_method=SynthesisMethod.FALLBACK,
            evidence_quotes=[],
            related_insights=[],
            requires_attention=["Full human answer required - synthesis failed"],
        )
    
    def _extract_relevant_sections_claude(self, transcript: str, gap: Dict) -> str:
        """Extract relevant transcript sections using Claude semantic understanding."""
        if not transcript or not gap:
            return ""
        
        question_text = gap.get("question_text", "")
        question_id = gap.get("question_id", "")
        pillar = gap.get("pillar", "")
        
        # Use Claude to semantically find relevant sections
        prompt = f"""You are analyzing a workshop transcript to find sections relevant to a specific AWS Well-Architected Framework question.

QUESTION:
{question_text}

Question ID: {question_id}
Pillar: {pillar}

TRANSCRIPT (first 8000 chars):
{transcript[:8000]}

TASK:
Identify the most relevant sections from the transcript that relate to this question, even if not directly addressing it. Consider:
- Semantic relevance (concepts, services, patterns mentioned)
- Contextual relevance (discussions that inform the answer)
- Indirect relevance (related topics that provide context)

Return ONLY a JSON object with this structure:
{{
  "relevant_sections": [
    {{
      "text": "exact quote from transcript",
      "line_number": approximate line number or "unknown",
      "relevance_explanation": "why this section is relevant to the question"
    }}
  ]
}}

Limit to the top 5 most relevant sections. Return ONLY the JSON, no other text."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1500,
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            
            result = json.loads(response["body"].read())
            response_text = result.get("content", [{}])[0].get("text", "")
            
            # Parse JSON response
            parsed = extract_json_from_text(response_text)
            if isinstance(parsed, dict) and "relevant_sections" in parsed:
                sections = parsed["relevant_sections"]
                formatted_sections = []
                for section in sections[:5]:  # Limit to 5
                    text = section.get("text", "")[:300]
                    line_num = section.get("line_number", "unknown")
                    relevance = section.get("relevance_explanation", "")
                    formatted_sections.append(f"Line {line_num}: {text}\n  (Relevance: {relevance})")
                return "\n\n".join(formatted_sections)[:2000]
            
            return ""
            
        except Exception as e:
            logger.warning(f"Claude transcript extraction failed: {e}")
            raise
    
    def _extract_relevant_sections_fallback(self, transcript: str, gap: Dict) -> str:
        """Fallback: Extract relevant transcript sections using keyword matching."""
        if not transcript or not gap:
            return ""
        
        question_text = gap.get("question_text", "").lower()
        question_id = gap.get("question_id", "")
        pillar = gap.get("pillar", "")
        context_hint = gap.get("context_hint", "")
        
        # Extract keywords from question
        keywords = []
        if question_data := gap.get("question_data"):
            keywords.extend(question_data.get("keywords", []))
        
        # Add pillar-related keywords
        pillar_keywords = {
            "SEC": ["security", "encrypt", "identity", "access", "permission"],
            "REL": ["reliability", "fault", "recover", "backup", "disaster"],
            "OPS": ["operational", "monitor", "automate", "process", "workload"],
            "PERF": ["performance", "efficient", "optimize", "scal", "resource"],
            "COST": ["cost", "price", "budget", "optimize", "spending"],
            "SUS": ["sustain", "carbon", "energy", "efficient", "environment"],
        }
        keywords.extend(pillar_keywords.get(pillar, []))
        
        # Search for relevant sections (simple keyword matching)
        lines = transcript.split("\n")
        relevant_lines = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Check if line contains any keywords or context hint
            if any(kw.lower() in line_lower for kw in keywords if kw):
                relevant_lines.append(f"Line {i+1}: {line[:200]}")
            elif context_hint and context_hint.lower() in line_lower:
                relevant_lines.append(f"Line {i+1}: {line[:200]}")
        
        # Return first 500 chars of relevant sections
        result = "\n".join(relevant_lines[:10])  # Limit to 10 matches
        return result[:2000]  # Limit total length
    
    def _find_related_insights_claude(
        self, insights: List[Dict], gap: Dict, pillar: str
    ) -> List[Dict]:
        """Find related insights using Claude semantic understanding."""
        if not insights:
            return []
        
        question_text = gap.get("question_text", "")
        question_id = gap.get("question_id", "")
        
        # Prepare insights summary (limit to first 20 for context)
        insights_summary = []
        for i, insight in enumerate(insights[:20]):
            insights_summary.append({
                "index": i,
                "type": insight.get("insight_type", ""),
                "content": insight.get("content", "")[:200],
                "pillar": insight.get("pillar", ""),
            })
        
        prompt = f"""You are analyzing architecture insights to find those most relevant to a specific AWS Well-Architected Framework question.

QUESTION:
{question_text}

Question ID: {question_id}
Pillar: {pillar}

AVAILABLE INSIGHTS:
{json.dumps(insights_summary, indent=2)}

TASK:
Identify which insights are semantically relevant to this question. Consider:
- Conceptual relevance (related topics, patterns, services)
- Contextual relevance (insights that inform the answer)
- Pillar alignment (insights from the same pillar)

Return ONLY a JSON object with this structure:
{{
  "relevant_insight_indices": [0, 2, 5, ...],
  "reasoning": "brief explanation of why these insights are relevant"
}}

List the indices (0-based) of the top {MAX_RELATED_INSIGHTS} most relevant insights. Return ONLY the JSON, no other text."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            
            result = json.loads(response["body"].read())
            response_text = result.get("content", [{}])[0].get("text", "")
            
            # Parse JSON response
            parsed = extract_json_from_text(response_text)
            if isinstance(parsed, dict) and "relevant_insight_indices" in parsed:
                indices = parsed["relevant_insight_indices"][:MAX_RELATED_INSIGHTS]
                # Return insights at those indices (if within range)
                related = [
                    insights[i] for i in indices
                    if 0 <= i < len(insights)
                ]
                return related
            
            return []
            
        except Exception as e:
            logger.warning(f"Claude insight matching failed: {e}")
            raise
    
    def _find_related_insights_fallback(
        self, insights: List[Dict], gap: Dict, pillar: str
    ) -> List[Dict]:
        """Fallback: Find related insights using keyword matching."""
        question_text = gap.get("question_text", "").lower()
        
        # Find insights from same pillar first
        pillar_insights = [i for i in insights if i.get("pillar") == pillar]
        
        # Then find by keyword overlap
        question_words = set(question_text.split())
        scored_insights = []
        
        for insight in insights:
            insight_text = str(insight.get("content", "")).lower()
            insight_words = set(insight_text.split())
            overlap = question_words.intersection(insight_words)
            score = len(overlap)
            
            # Boost score if same pillar
            if insight.get("pillar") == pillar:
                score += 2
            
            scored_insights.append((score, insight))
        
        # Sort by score and return top insights
        scored_insights.sort(key=lambda x: x[0], reverse=True)
        return [insight for _, insight in scored_insights[:MAX_RELATED_INSIGHTS]]
    
    def _find_related_answers_claude(
        self, validated_answers: List[Dict], gap: Dict, pillar: str
    ) -> List[Dict]:
        """Find related answered questions using Claude semantic understanding."""
        if not validated_answers:
            return []
        
        question_text = gap.get("question_text", "")
        question_id = gap.get("question_id", "")
        
        # Prepare answers summary (limit to first 15 for context)
        answers_summary = []
        for i, answer in enumerate(validated_answers[:15]):
            answers_summary.append({
                "index": i,
                "question_id": answer.get("question_id", ""),
                "question_text": answer.get("question_text", "")[:150],
                "pillar": answer.get("pillar", ""),
                "answer_summary": answer.get("answer_content", "")[:200],
            })
        
        prompt = f"""You are analyzing answered WAFR questions to find those most relevant to a specific unanswered question.

UNANSWERED QUESTION:
{question_text}

Question ID: {question_id}
Pillar: {pillar}

AVAILABLE ANSWERED QUESTIONS:
{json.dumps(answers_summary, indent=2)}

TASK:
Identify which answered questions are semantically relevant to the unanswered question. Consider:
- Conceptual similarity (related topics, patterns)
- Contextual relevance (answers that inform how to answer the question)
- Pillar alignment (questions from the same pillar)

Return ONLY a JSON object with this structure:
{{
  "relevant_answer_indices": [0, 2, 4, ...],
  "reasoning": "brief explanation of why these answers are relevant"
}}

List the indices (0-based) of the top {MAX_RELATED_ANSWERS} most relevant answers. Return ONLY the JSON, no other text."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            
            result = json.loads(response["body"].read())
            response_text = result.get("content", [{}])[0].get("text", "")
            
            # Parse JSON response
            parsed = extract_json_from_text(response_text)
            if isinstance(parsed, dict) and "relevant_answer_indices" in parsed:
                indices = parsed["relevant_answer_indices"][:MAX_RELATED_ANSWERS]
                # Return answers at those indices (if within range)
                related = [
                    validated_answers[i] for i in indices
                    if 0 <= i < len(validated_answers)
                ]
                return related
            
            return []
            
        except Exception as e:
            logger.warning(f"Claude answer matching failed: {e}")
            raise
    
    def _find_related_answers_fallback(
        self, validated_answers: List[Dict], gap: Dict, pillar: str
    ) -> List[Dict]:
        """Fallback: Find related answered questions using pillar matching."""
        # Return answers from same pillar
        return [
            a for a in validated_answers
            if a.get("pillar") == pillar
        ][:MAX_RELATED_ANSWERS]
    
    def _infer_workload_profile_claude(
        self, transcript: str, insights: List[Dict], validated_answers: List[Dict]
    ) -> Dict[str, Any]:
        """Infer workload characteristics using Claude reasoning."""
        # Prepare context summary
        insights_summary = "\n".join([
            f"- {ins.get('insight_type', 'unknown')}: {ins.get('content', '')[:150]}"
            for ins in insights[:15]
        ])
        
        answers_summary = "\n".join([
            f"- {ans.get('question_id', 'unknown')}: {ans.get('answer_content', '')[:150]}"
            for ans in validated_answers[:10]
        ])
        
        prompt = f"""You are analyzing a workload architecture to infer its characteristics.

TRANSCRIPT (first 3000 chars):
{transcript[:3000]}

KEY INSIGHTS:
{insights_summary[:1000]}

KEY ANSWERS:
{answers_summary[:800]}

TASK:
Analyze this workload and infer its characteristics. Return ONLY a JSON object:

{{
  "type": "serverless or containerized or compute-based or hybrid or unknown",
  "scale": "enterprise or medium or startup or unknown",
  "industry": "healthcare or finance or retail or tech or unknown (if identifiable)",
  "aws_services": ["service1", "service2", ...],
  "security_posture": "high or standard or basic",
  "compliance_requirements": ["requirement1", "requirement2", ...] or []
}}

Be specific and evidence-based. If information is not available, use "unknown" or empty arrays.

Return ONLY the JSON, no other text."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            
            result = json.loads(response["body"].read())
            response_text = result.get("content", [{}])[0].get("text", "")
            
            # Parse JSON response
            parsed = extract_json_from_text(response_text)
            if isinstance(parsed, dict):
                # Ensure all required fields with defaults
                profile = {
                    "type": parsed.get("type", "unknown"),
                    "scale": parsed.get("scale", "unknown"),
                    "industry": parsed.get("industry", "unknown"),
                    "aws_services": parsed.get("aws_services", []),
                    "security_posture": parsed.get("security_posture", "standard"),
                    "compliance_requirements": parsed.get("compliance_requirements", []),
                }
                return profile
            
            # Fallback to default
            return {
                "type": "unknown",
                "scale": "unknown",
                "industry": "unknown",
                "aws_services": [],
                "security_posture": "standard",
                "compliance_requirements": [],
            }
            
        except Exception as e:
            logger.warning(f"Claude workload inference failed: {e}")
            raise
    
    def _infer_workload_profile_fallback(
        self, insights: List[Dict], validated_answers: List[Dict]
    ) -> Dict[str, Any]:
        """Fallback: Infer workload characteristics using keyword matching."""
        profile = {
            "type": "unknown",
            "scale": "unknown",
            "industry": "unknown",
            "aws_services": [],
            "security_posture": "standard",
            "compliance_requirements": [],
        }
        
        # Extract AWS services mentioned
        services = set()
        for insight in insights:
            if insight.get("insight_type") == "service":
                content = insight.get("content", "")
                # Simple service name extraction
                for service in [
                    "Lambda",
                    "EC2",
                    "S3",
                    "RDS",
                    "DynamoDB",
                    "API Gateway",
                    "CloudWatch",
                    "IAM",
                    "EKS",
                    "ECS",
                    "SageMaker",
                    "Bedrock",
                    "Neptune",
                    "Amplify",
                ]:
                    if service.lower() in content.lower():
                        services.add(service)
        
        profile["aws_services"] = list(services)
        
        # Infer workload type from services
        if "Lambda" in services or "API Gateway" in services:
            profile["type"] = "serverless"
        elif "EKS" in services or "ECS" in services:
            profile["type"] = "containerized"
        elif "EC2" in services:
            profile["type"] = "compute-based"
        
        # Infer scale from discussion
        all_text = " ".join(
            [
                str(i.get("content", ""))
                for i in insights + validated_answers
            ]
        ).lower()
        
        if any(word in all_text for word in ["enterprise", "large-scale", "millions"]):
            profile["scale"] = "enterprise"
        elif any(word in all_text for word in ["startup", "small", "mvp"]):
            profile["scale"] = "startup"
        else:
            profile["scale"] = "medium"
        
        return profile


    # =========================================================================
    # Re-synthesis for HITL (Human-in-the-Loop)
    # =========================================================================
    
    def re_synthesize_with_feedback(
        self,
        original: SynthesizedAnswer,
        feedback: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SynthesizedAnswer:
        """
        Re-synthesize an answer incorporating human feedback.
        
        Called when a reviewer rejects an answer with feedback. The agent
        uses the feedback to improve the answer quality.
        
        Args:
            original: Original synthesized answer that was rejected
            feedback: Human reviewer's feedback explaining rejection
            context: Optional additional context for re-synthesis
            
        Returns:
            New SynthesizedAnswer incorporating the feedback
        """
        logger.info(f"Re-synthesizing answer for {original.question_id} with feedback")
        
        prompt = f"""You previously generated this WAFR answer that was REJECTED by a human reviewer. Your task is to generate an IMPROVED answer based on their feedback.

## Original Question
- **Question ID**: {original.question_id}
- **Pillar**: {original.pillar}
- **Question**: {original.question_text}
- **Criticality**: {original.criticality}

## Your Original Answer (REJECTED)
{original.synthesized_answer}

## Original Reasoning Chain
{json.dumps(original.reasoning_chain, indent=2)}

## Original Assumptions
{json.dumps(original.assumptions, indent=2)}

## Original Confidence
{original.confidence:.2f} - {original.confidence_justification}

## Human Reviewer Feedback
**IMPORTANT - Address this feedback in your revised answer:**
{feedback}

## Your Task
Generate an IMPROVED answer that specifically addresses the reviewer's feedback. Follow the same JSON format:

{{
  "synthesized_answer": "Your IMPROVED answer addressing the feedback. Be more specific and accurate.",
  "reasoning_chain": [
    "Step 1: How feedback was incorporated",
    "Step 2: What evidence supports the revision",
    "Step 3: How assumptions were adjusted",
    "Step 4: Final improved answer rationale"
  ],
  "assumptions": [
    "Revised assumption 1 (adjusted based on feedback)",
    "Revised assumption 2"
  ],
  "confidence": 0.0-1.0,
  "confidence_justification": "Updated confidence with reference to feedback and revisions",
  "evidence_quotes": [
    {{"text": "relevant quote", "location": "source", "relevance": "why relevant"}}
  ],
  "requires_attention": [
    "Remaining items for human verification"
  ],
  "synthesis_method": "EVIDENCE_BASED or INFERENCE or BEST_PRACTICE"
}}

Key requirements:
1. DIRECTLY address the feedback - don't repeat the same mistakes
2. Be more careful about assumptions
3. Increase specificity where the reviewer indicated issues
4. If feedback mentions specific facts, incorporate them
5. Adjust confidence level appropriately

Generate the improved JSON response now:"""

        try:
            response_text = self._invoke_synthesis_model(prompt)
            
            # Parse response with original gap info
            gap = {
                "question_id": original.question_id,
                "pillar": original.pillar,
                "question_text": original.question_text,
                "criticality": original.criticality,
            }
            
            new_answer = self._parse_synthesis_response(response_text, gap)
            logger.info(
                f"Re-synthesis complete for {original.question_id} "
                f"(new confidence: {new_answer.confidence:.2f})"
            )
            return new_answer
            
        except Exception as e:
            logger.error(f"Re-synthesis failed for {original.question_id}: {e}")
            # Return modified original with lower confidence
            return SynthesizedAnswer(
                question_id=original.question_id,
                pillar=original.pillar,
                question_text=original.question_text,
                synthesized_answer=(
                    f"{original.synthesized_answer}\n\n"
                    f"[Re-synthesis attempted but failed. Reviewer feedback: {feedback}]"
                ),
                criticality=original.criticality,
                reasoning_chain=original.reasoning_chain + [f"Re-synthesis failed: {str(e)}"],
                assumptions=original.assumptions,
                confidence=max(0.1, original.confidence - 0.2),
                confidence_justification=(
                    f"Reduced due to re-synthesis failure. "
                    f"Original: {original.confidence_justification}"
                ),
                synthesis_method=SynthesisMethod.FALLBACK,
                evidence_quotes=original.evidence_quotes,
                related_insights=original.related_insights,
                requires_attention=original.requires_attention + [
                    "Re-synthesis failed - manual review required"
                ],
            )
    
    def synthesize_batch(
        self,
        gaps: List[Dict],
        transcript: str,
        insights: List[Dict],
        validated_answers: List[Dict],
    ) -> List[SynthesizedAnswer]:
        """
        Synthesize multiple answers in a single LLM call for efficiency.
        
        This is more efficient than single-question synthesis when processing
        multiple gaps that share similar context.
        
        Args:
            gaps: List of gap questions to synthesize (max 5 recommended)
            transcript: Workshop transcript
            insights: Extracted insights
            validated_answers: Already answered questions
            
        Returns:
            List of synthesized answers
        """
        if len(gaps) == 0:
            return []
        
        if len(gaps) > 5:
            logger.warning(f"Batch too large ({len(gaps)}). Splitting into smaller batches.")
            results = []
            for i in range(0, len(gaps), 5):
                batch_result = self.synthesize_batch(
                    gaps[i:i+5], transcript, insights, validated_answers
                )
                results.extend(batch_result)
            return results
        
        # Build workload profile once for the batch
        try:
            workload_profile = self._infer_workload_profile_claude(
                transcript, insights, validated_answers
            )
        except Exception:
            workload_profile = self._infer_workload_profile_fallback(insights, validated_answers)
        
        # Prepare batch context
        questions_text = "\n\n".join([
            f"### Question {i+1}\n"
            f"- **ID**: {gap.get('question_id')}\n"
            f"- **Pillar**: {gap.get('pillar')}\n"
            f"- **Text**: {gap.get('question_text')}\n"
            f"- **Criticality**: {gap.get('criticality', 'MEDIUM')}"
            for i, gap in enumerate(gaps)
        ])
        
        # Build batch prompt
        prompt = f"""You are an AWS Well-Architected Framework expert. Generate answers for {len(gaps)} WAFR questions using the transcript context and logical inference.

## Questions to Answer
{questions_text}

## Transcript Context (first 5000 chars)
{transcript[:5000]}

## Key Insights (first 10)
{json.dumps([{"type": i.get("insight_type", ""), "content": i.get("content", "")[:200]} for i in insights[:10]], indent=2)}

## Workload Profile
{json.dumps(workload_profile, indent=2)}

## Your Task
Generate answers for ALL {len(gaps)} questions. Return a JSON array with one object per question:

[
  {{
    "question_id": "exact question ID",
    "synthesized_answer": "comprehensive answer text",
    "reasoning_chain": ["step 1", "step 2", "step 3", "step 4"],
    "assumptions": ["assumption 1", "assumption 2"],
    "confidence": 0.0-1.0,
    "confidence_justification": "why this confidence",
    "evidence_quotes": [{{"text": "quote", "location": "source", "relevance": "why"}}],
    "requires_attention": ["item 1"],
    "synthesis_method": "EVIDENCE_BASED or INFERENCE or BEST_PRACTICE"
  }},
  ... (one object per question)
]

Return ONLY the JSON array, no other text."""

        try:
            response_text = self._invoke_synthesis_model(prompt)
            
            # Parse batch response
            parsed = extract_json_from_text(response_text)
            
            if not isinstance(parsed, list):
                # If not a list, wrap in list
                if isinstance(parsed, dict):
                    parsed = [parsed]
                else:
                    raise ValueError("Response is not a valid JSON array")
            
            # Convert to SynthesizedAnswer objects
            results = []
            for i, gap in enumerate(gaps):
                if i < len(parsed):
                    answer_data = parsed[i]
                    # Merge gap info with answer data
                    merged = {
                        "question_id": gap.get("question_id"),
                        "pillar": gap.get("pillar"),
                        "question_text": gap.get("question_text"),
                        "criticality": gap.get("criticality", "MEDIUM"),
                        **answer_data,
                    }
                    results.append(self._parse_synthesis_response(json.dumps(merged), gap))
                else:
                    # Not enough answers returned - create fallback
                    results.append(self._create_fallback_answer(gap, "Batch synthesis missing answer"))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch synthesis failed: {e}. Falling back to individual synthesis.")
            # Fallback to individual synthesis
            return [
                self._create_fallback_answer(gap, f"Batch failed: {str(e)}")
                for gap in gaps
            ]


# =============================================================================
# Factory Function
# =============================================================================

def create_answer_synthesis_agent(
    wafr_schema: Optional[Dict] = None,
    lens_context: Optional[Dict] = None,
) -> AnswerSynthesisAgent:
    """
    Factory function to create Answer Synthesis Agent.
    
    Args:
        wafr_schema: Optional WAFR schema for context
        lens_context: Optional lens context for multi-lens support
        
    Returns:
        Configured AnswerSynthesisAgent instance
    """
    return AnswerSynthesisAgent(
        wafr_schema=wafr_schema, lens_context=lens_context
    )

