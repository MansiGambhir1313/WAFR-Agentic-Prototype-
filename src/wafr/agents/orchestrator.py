"""
Agent Orchestrator - Coordinates multi-agent workflow.

Uses Strands framework for agent coordination to process WAFR assessments
through a pipeline of specialized agents.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Optional strands import (not used directly, but may be needed by agents)
try:
    from strands import Agent
except ImportError:
    Agent = None

from wafr.agents.answer_synthesis_agent import create_answer_synthesis_agent
from wafr.agents.confidence_agent import create_confidence_agent
from wafr.agents.gap_detection_agent import create_gap_detection_agent
from wafr.agents.input_processor import (
    InputProcessor,
    InputType,
    create_input_processor,
)
from wafr.agents.mapping_agent import create_mapping_agent
from wafr.agents.pdf_processor import create_pdf_processor
from wafr.agents.prompt_generator_agent import create_prompt_generator_agent
from wafr.agents.report_agent import create_report_agent
from wafr.agents.scoring_agent import create_scoring_agent
from wafr.agents.understanding_agent import create_understanding_agent
from wafr.agents.utils import cache_result
from wafr.agents.wa_tool_agent import WAToolAgent

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_ENVIRONMENT = "PRODUCTION"
DEFAULT_LENS_ALIAS = "wellarchitected"

SCHEMA_CACHE_TTL_SECONDS = 3600.0
MAX_GAPS_TO_PROCESS = 10
MIN_CONFIDENCE_THRESHOLD = 0.4
MIN_LENS_DETECTION_CONFIDENCE = 0.2
MAX_AUTO_SELECT_LENSES = 3

# User choice constants
CHOICE_MANUAL = "manual"
CHOICE_PROCEED = "proceed"
CHOICE_CONSOLE = "console"
CHOICE_SKIP = "skip"

# Status constants
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_ERRORS = "completed_with_errors"
STATUS_ERROR = "error"
STATUS_EXISTING = "existing"
STATUS_CREATED = "created"

# Confidence levels
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

logger = logging.getLogger(__name__)

# Schema cache (module-level for persistence across instances)
_schema_cache: Dict[str, Any] = {}


# -----------------------------------------------------------------------------
# Main Orchestrator Class
# -----------------------------------------------------------------------------


class WafrOrchestrator:
    """Orchestrates the multi-agent WAFR processing pipeline."""

    def __init__(
        self,
        wafr_schema: Optional[Dict] = None,
        input_processor: Optional[InputProcessor] = None,
        lens_context: Optional[Dict] = None,
    ):
        """
        Initialize orchestrator with all agents.

        Args:
            wafr_schema: Complete WAFR schema (loaded from file/database).
            input_processor: Optional input processor for file handling.
            lens_context: Optional lens context for multi-lens support.
        """
        if wafr_schema is None:
            from wafr.agents.wafr_context import load_wafr_schema
            wafr_schema = load_wafr_schema()

        self.wafr_schema = wafr_schema
        self.lens_context = lens_context or {}
        self.logger = logger

        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """Initialize all processing agents with schema and lens context."""
        # Core processing agents
        self.understanding_agent = create_understanding_agent(
            self.wafr_schema,
            lens_context=self.lens_context,
        )
        self.mapping_agent = create_mapping_agent(
            self.wafr_schema,
            lens_context=self.lens_context,
        )
        self.confidence_agent = create_confidence_agent(self.wafr_schema)
        self.gap_detection_agent = create_gap_detection_agent(
            self.wafr_schema,
            lens_context=self.lens_context,
        )
        self.answer_synthesis_agent = create_answer_synthesis_agent(
            self.wafr_schema,
            lens_context=self.lens_context,
        )
        self.prompt_generator_agent = create_prompt_generator_agent(self.wafr_schema)
        self.scoring_agent = create_scoring_agent(self.wafr_schema)
        self.report_agent = create_report_agent(
            self.wafr_schema,
            lens_context=self.lens_context,
        )

        # Tool agents
        self.wa_tool_agent = WAToolAgent()
        self.pdf_processor = create_pdf_processor()
        self.input_processor = create_input_processor()

    # -------------------------------------------------------------------------
    # Main Processing Methods
    # -------------------------------------------------------------------------

    def process_transcript(
        self,
        transcript: str,
        session_id: str,
        generate_report: bool = True,
        create_wa_workload: bool = False,
        client_name: Optional[str] = None,
        environment: str = DEFAULT_ENVIRONMENT,
        existing_workload_id: Optional[str] = None,
        pdf_files: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, str, Optional[Dict]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process transcript through complete agent pipeline.

        Args:
            transcript: Workshop transcript text.
            session_id: Session identifier.
            generate_report: Whether to generate final report.
            create_wa_workload: Whether to create WA Tool workload.
            client_name: Client/company name for WA Tool workload.
            environment: Environment type (PRODUCTION, PREPRODUCTION, DEVELOPMENT).
            existing_workload_id: Existing WA Tool workload ID to use.
            pdf_files: Optional list of PDF file paths to include in analysis.
            progress_callback: Optional callback for progress updates.

        Returns:
            Complete assessment results dictionary.
        """
        start_time = time.time()
        self.logger.info(f"Orchestrator: Starting processing for session {session_id}")

        results = self._create_initial_results(session_id)

        try:
            # Step 0: Process PDFs if provided
            enhanced_transcript, pdf_content = self._process_pdf_files(
                transcript, pdf_files, results, progress_callback
            )

            # Step 1-6: Core pipeline processing
            insights = self._step_extract_insights(
                enhanced_transcript, pdf_content, pdf_files,
                session_id, results, progress_callback
            )

            mappings = self._step_map_insights(
                insights, session_id, results, progress_callback
            )

            validated_answers = self._step_validate_confidence(
                mappings, transcript, session_id, results, progress_callback
            )

            gap_result = self._step_detect_gaps(
                validated_answers, transcript, session_id, results, progress_callback
            )

            # Step 6: Generate AI answers for gaps (NEW - HITL workflow)
            synthesized_answers = self._step_synthesize_gap_answers(
                gap_result, transcript, insights, validated_answers,
                session_id, results, progress_callback
            )

            # Step 6.5: Auto-populate answers using synthesized answers (NEW)
            all_answers = self._step_auto_populate_answers(
                validated_answers, synthesized_answers, session_id,
                results, progress_callback
            )

            self._step_generate_prompts(
                gap_result, results, progress_callback
            )

            # Use all_answers (includes synthesized) for scoring
            self._step_score_answers(
                all_answers, session_id, results, progress_callback
            )

            # Step 7: Report generation (optional)
            if generate_report:
                # Use all_answers (includes synthesized) for report generation
                self._step_generate_report(
                    all_answers, gap_result, results,
                    session_id, progress_callback
                )
            else:
                results["steps"]["report"] = None

            # Step 8: WA Tool integration (optional)
            if create_wa_workload and (client_name or existing_workload_id):
                # Use all_answers (includes synthesized) for WA Tool integration
                self._step_wa_tool_integration(
                    insights, mappings, all_answers, transcript,
                    session_id, client_name, environment, existing_workload_id,
                    generate_report, results, progress_callback
                )
            else:
                results["steps"]["wa_workload"] = None

            # Finalize results (use all_answers to include synthesized answers)
            self._finalize_results(
                results, insights, mappings, all_answers,
                gap_result, start_time
            )

            return results

        except Exception as e:
            self.logger.error(f"Orchestrator processing failed: {e}", exc_info=True)
            results["status"] = STATUS_ERROR
            results["error"] = str(e)
            results["processing_time"]["total"] = round(time.time() - start_time, 2)
            return results

    def process_file(
        self,
        file_path: str,
        session_id: Optional[str] = None,
        generate_report: bool = True,
        create_wa_workload: bool = False,
        client_name: Optional[str] = None,
        environment: str = DEFAULT_ENVIRONMENT,
        existing_workload_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, str, Optional[Dict]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process any supported input file through the pipeline.

        Supports text, PDF, video, and audio files.

        Args:
            file_path: Path to input file.
            session_id: Session identifier (auto-generated if not provided).
            generate_report: Whether to generate final report.
            create_wa_workload: Whether to create WA Tool workload.
            client_name: Client/company name for WA Tool workload.
            environment: Environment type.
            existing_workload_id: Existing WA Tool workload ID to use.
            progress_callback: Optional callback for progress updates.

        Returns:
            Complete assessment results dictionary.
        """
        if session_id is None:
            session_id = self._generate_session_id("file")

        if progress_callback:
            progress_callback("input_processing", f"Processing input file: {file_path}")

        try:
            processed_input = self.input_processor.process(file_path)

            # Auto-detect lenses if not already set
            self._auto_detect_lenses(processed_input.content)

            input_metadata = self._build_input_metadata(processed_input)

            results = self.process_transcript(
                transcript=processed_input.content,
                session_id=session_id,
                generate_report=generate_report,
                create_wa_workload=create_wa_workload,
                client_name=client_name,
                environment=environment,
                existing_workload_id=existing_workload_id,
                progress_callback=progress_callback,
            )

            results["input_metadata"] = input_metadata

            if processed_input.errors:
                results.setdefault("errors", []).extend(processed_input.errors)

            return results

        except Exception as e:
            self.logger.error(f"Input processing failed: {e}", exc_info=True)
            raise

    def process_user_answer(
        self,
        session_id: str,
        question_id: str,
        answer: str,
        wafr_schema: Dict,
    ) -> Dict[str, Any]:
        """
        Process user-provided answer for a gap question.

        Args:
            session_id: Session identifier.
            question_id: Question ID.
            answer: User's answer text.
            wafr_schema: WAFR schema.

        Returns:
            Processing result with score.
        """
        self.logger.info(f"Processing user answer for question {question_id}")

        question_data = self._get_question_data(question_id, wafr_schema)
        if not question_data:
            return {"error": f"Question {question_id} not found"}

        answer_dict = {
            "question_id": question_id,
            "question_text": question_data.get("text", ""),
            "pillar": question_data.get("pillar_id", "UNKNOWN"),
            "answer_content": answer,
            "evidence_quotes": [],
            "source": "user_input",
        }

        scoring_result = self.scoring_agent.process(
            answers=[answer_dict],
            wafr_schema=wafr_schema,
            session_id=session_id,
        )

        scored_answers = scoring_result.get("scored_answers", [])
        scores = scored_answers[0] if scored_answers else {}

        return {
            "question_id": question_id,
            "answer": answer,
            "scores": scores,
            "status": "scored",
        }

    # -------------------------------------------------------------------------
    # Pipeline Step Methods
    # -------------------------------------------------------------------------

    def _step_extract_insights(
        self,
        enhanced_transcript: str,
        pdf_content: Optional[Dict],
        pdf_files: Optional[List[str]],
        session_id: str,
        results: Dict,
        progress_callback: Optional[Callable],
    ) -> List[Dict]:
        """Step 1: Extract insights from transcript and PDF content."""
        step_start = time.time()
        self.logger.info("Step 1: Extracting insights from transcript and PDF content")

        if progress_callback:
            progress_callback("understanding", "Extracting insights from transcript and PDFs...")

        insights = []
        try:
            insights_result = self.understanding_agent.process(enhanced_transcript, session_id)
            results["steps"]["understanding"] = insights_result
            insights = insights_result.get("insights", [])

            # Add PDF diagram insight if images were found
            if pdf_content and pdf_content.get("images"):
                insights.append(self._create_pdf_diagram_insight(pdf_content, pdf_files))

            if not insights:
                self.logger.warning("No insights extracted, continuing with empty list")

            self.logger.info(f"Extracted {len(insights)} insights")

        except Exception as e:
            self.logger.error(f"Understanding agent failed: {e}", exc_info=True)
            results["steps"]["understanding"] = {
                "error": str(e),
                "insights": [],
                "agent": "understanding",
            }

        results["processing_time"]["understanding"] = round(time.time() - step_start, 2)
        return insights

    def _step_map_insights(
        self,
        insights: List[Dict],
        session_id: str,
        results: Dict,
        progress_callback: Optional[Callable],
    ) -> List[Dict]:
        """Step 2: Map insights to WAFR questions."""
        step_start = time.time()
        self.logger.info("Step 2: Mapping insights to WAFR questions")

        if progress_callback:
            progress_callback("mapping", "Mapping insights to WAFR questions...")

        mappings = []
        try:
            if insights:
                mapping_result = self.mapping_agent.process(insights, session_id)
                results["steps"]["mapping"] = mapping_result
                mappings = mapping_result.get("mappings", [])
            else:
                results["steps"]["mapping"] = {
                    "session_id": session_id,
                    "total_mappings": 0,
                    "mappings": [],
                    "pillar_coverage": {},
                    "agent": "mapping",
                }

        except Exception as e:
            self.logger.error(f"Mapping agent failed: {e}", exc_info=True)
            results["steps"]["mapping"] = {"error": str(e), "mappings": []}

        results["processing_time"]["mapping"] = round(time.time() - step_start, 2)
        return mappings

    def _step_validate_confidence(
        self,
        mappings: List[Dict],
        transcript: str,
        session_id: str,
        results: Dict,
        progress_callback: Optional[Callable],
    ) -> List[Dict]:
        """Step 3: Validate evidence and confidence scores."""
        step_start = time.time()
        self.logger.info("Step 3: Validating evidence and confidence")

        if progress_callback:
            progress_callback("confidence", "Validating evidence and confidence scores...")

        validated_answers = []
        try:
            if mappings:
                confidence_result = self.confidence_agent.process(
                    mappings, transcript, session_id
                )
                results["steps"]["confidence"] = confidence_result
            else:
                confidence_result = {
                    "session_id": session_id,
                    "summary": {"average_score": 0, "total_answers": 0},
                    "all_validations": [],
                    "agent": "confidence",
                }
                results["steps"]["confidence"] = confidence_result

            validated_answers = self._extract_validated_answers(mappings, confidence_result)
            
            # Fallback: If confidence validation failed or returned empty, use mappings directly
            if not validated_answers and mappings:
                self.logger.warning("Confidence validation returned no results, using mappings as fallback")
                validated_answers = [
                    {
                        **m,
                        "confidence_score": m.get("relevance_score", 0.6),
                        "confidence_level": "medium",
                        "evidence_verified": True,
                        "validation_passed": True,
                        "source": "mapping_fallback",
                    }
                    for m in mappings
                ]

        except Exception as e:
            self.logger.error(f"Confidence agent failed: {e}", exc_info=True)
            results["errors"].append({"step": "confidence", "error": str(e)})
            results["steps"]["confidence"] = {
                "error": str(e),
                "all_validations": [],
                "agent": "confidence",
            }
            
            # Fallback: Use mappings directly if confidence agent completely fails
            if mappings:
                self.logger.warning("Using mappings as fallback after confidence agent failure")
                validated_answers = [
                    {
                        **m,
                        "confidence_score": m.get("relevance_score", 0.6),
                        "confidence_level": "medium",
                        "evidence_verified": True,
                        "validation_passed": True,
                        "source": "error_fallback",
                    }
                    for m in mappings
                ]
            else:
                validated_answers = []

        results["processing_time"]["confidence"] = round(time.time() - step_start, 2)
        return validated_answers

    def _step_detect_gaps(
        self,
        validated_answers: List[Dict],
        transcript: str,
        session_id: str,
        results: Dict,
        progress_callback: Optional[Callable],
    ) -> Dict:
        """Step 4: Detect gaps in WAFR coverage."""
        step_start = time.time()
        self.logger.info("Step 4: Detecting gaps")

        if progress_callback:
            progress_callback("gap_detection", "Detecting gaps in WAFR coverage...")

        gap_result = {"gaps": []}
        try:
            answered_questions = [
                a.get("question_id")
                for a in validated_answers
                if a.get("question_id")
            ]

            pillar_coverage = results["steps"].get("mapping", {}).get("pillar_coverage", {})
            normalized_coverage = self._normalize_pillar_coverage(pillar_coverage)

            gap_result = self.gap_detection_agent.process(
                answered_questions=answered_questions,
                pillar_coverage=normalized_coverage,
                session_id=session_id,
                transcript=transcript,
            )
            results["steps"]["gap_detection"] = gap_result

        except Exception as e:
            self.logger.error(f"Gap detection agent failed: {e}", exc_info=True)
            results["steps"]["gap_detection"] = {"error": str(e), "gaps": []}

        results["processing_time"]["gap_detection"] = round(time.time() - step_start, 2)
        return gap_result

    def _step_synthesize_gap_answers(
        self,
        gap_result: Dict,
        transcript: str,
        insights: List[Dict],
        validated_answers: List[Dict],
        session_id: str,
        results: Dict,
        progress_callback: Optional[Callable],
    ) -> List[Dict]:
        """Step 6: Generate AI answers for all gap questions."""
        step_start = time.time()
        self.logger.info("Step 6: Synthesizing answers for gap questions")

        if progress_callback:
            progress_callback("answer_synthesis", "Generating AI answers for gap questions...")

        synthesized_answers = []
        try:
            gaps = gap_result.get("gaps", [])
            
            if gaps:
                # Convert SynthesizedAnswer objects to dicts for storage
                synthesized_objects = self.answer_synthesis_agent.synthesize_gaps(
                    gaps=gaps,
                    transcript=transcript,
                    insights=insights,
                    validated_answers=validated_answers,
                    session_id=session_id,
                )
                
                # Convert to dict format for storage
                synthesized_answers = [sa.to_dict() for sa in synthesized_objects]
                
                results["steps"]["answer_synthesis"] = {
                    "session_id": session_id,
                    "total_synthesized": len(synthesized_answers),
                    "synthesized_answers": synthesized_answers,
                    "agent": "answer_synthesis",
                }
                
                self.logger.info(f"Synthesized {len(synthesized_answers)} answers for gap questions")
            else:
                results["steps"]["answer_synthesis"] = {
                    "session_id": session_id,
                    "total_synthesized": 0,
                    "synthesized_answers": [],
                    "agent": "answer_synthesis",
                }
                self.logger.info("No gaps to synthesize answers for")

        except Exception as e:
            self.logger.error(f"Answer synthesis agent failed: {e}", exc_info=True)
            results["steps"]["answer_synthesis"] = {
                "error": str(e),
                "synthesized_answers": [],
                "agent": "answer_synthesis",
            }

        results["processing_time"]["answer_synthesis"] = round(time.time() - step_start, 2)
        return synthesized_answers
    
    def _step_auto_populate_answers(
        self,
        validated_answers: List[Dict],
        synthesized_answers: List[Dict],
        session_id: str,
        results: Dict,
        progress_callback: Optional[Callable],
    ) -> List[Dict]:
        """
        Step 6.5: Automatically merge synthesized answers with validated answers.
        
        This step automatically answers questions using the synthesized answers,
        merging them with validated answers from the transcript.
        
        Args:
            validated_answers: Answers extracted from transcript
            synthesized_answers: Answers synthesized for gaps
            session_id: Session identifier
            results: Results dictionary
            progress_callback: Optional progress callback
            
        Returns:
            Combined list of all answers (validated + synthesized)
        """
        step_start = time.time()
        self.logger.info("Step 6.5: Auto-populating answers from synthesized answers")

        if progress_callback:
            progress_callback("auto_populate", "Auto-populating answers from synthesis...")

        all_answers = validated_answers  # Default to validated answers only
        try:
            # Merge synthesized answers into validated answers
            all_answers = self._merge_synthesized_answers(
                validated_answers, synthesized_answers
            )
            
            results["steps"]["auto_populate"] = {
                "session_id": session_id,
                "validated_count": len(validated_answers),
                "synthesized_count": len(synthesized_answers),
                "total_count": len(all_answers),
                "merged": True,
            }
            
            self.logger.info(
                f"Merged {len(synthesized_answers)} synthesized answers with "
                f"{len(validated_answers)} validated answers = {len(all_answers)} total"
            )
            
        except Exception as e:
            self.logger.error(f"Auto-populate answers failed: {e}", exc_info=True)
            results["steps"]["auto_populate"] = {
                "error": str(e),
                "merged": False,
            }
            # Keep all_answers as validated_answers (already set as default)

        results["processing_time"]["auto_populate"] = round(time.time() - step_start, 2)
        return all_answers
    
    def _merge_synthesized_answers(
        self,
        validated_answers: List[Dict],
        synthesized_answers: List[Dict],
    ) -> List[Dict]:
        """
        Merge synthesized answers with validated answers.
        
        Strategy:
        - Validated answers from transcript are kept as-is
        - Synthesized answers are converted to validated answer format
        - No duplicates (synthesized answers override if same question_id exists)
        
        Args:
            validated_answers: Answers from transcript
            synthesized_answers: AI-synthesized answers for gaps
            
        Returns:
            Merged list of all answers
        """
        # Create a map of validated answers by question_id
        validated_map = {
            ans.get("question_id", ""): ans
            for ans in validated_answers
        }
        
        # Convert synthesized answers to validated answer format
        for synth_ans in synthesized_answers:
            question_id = synth_ans.get("question_id", "")
            if not question_id:
                continue
            
            # Convert synthesized answer to validated answer format
            validated_format = {
                "question_id": question_id,
                "question_text": synth_ans.get("question_text", ""),
                "pillar": synth_ans.get("pillar", ""),
                "answer_content": synth_ans.get("synthesized_answer", ""),
                "source": "AI_SYNTHESIZED",
                "confidence": synth_ans.get("confidence", 0.5),
                "synthesis_method": synth_ans.get("synthesis_method", "INFERENCE"),
                "reasoning_chain": synth_ans.get("reasoning_chain", []),
                "assumptions": synth_ans.get("assumptions", []),
                "evidence_quotes": [
                    {
                        "text": eq.get("text", ""),
                        "location": eq.get("location", ""),
                        "relevance": eq.get("relevance", ""),
                    }
                    for eq in synth_ans.get("evidence_quotes", [])
                ],
                "requires_attention": synth_ans.get("requires_attention", []),
                "confidence_justification": synth_ans.get("confidence_justification", ""),
            }
            
            # Add to map (synthesized answers don't override validated answers)
            # Only add if question_id not already in validated answers
            if question_id not in validated_map:
                validated_map[question_id] = validated_format
        
        # Return all answers as a list
        return list(validated_map.values())

    def _step_generate_prompts(
        self,
        gap_result: Dict,
        results: Dict,
        progress_callback: Optional[Callable],
    ) -> None:
        """Step 5: Generate smart prompts for gaps."""
        step_start = time.time()
        self.logger.info("Step 5: Generating smart prompts for gaps")

        if progress_callback:
            progress_callback("prompt_generator", "Generating smart prompts for gaps...")

        gap_prompts = []
        try:
            gaps = gap_result.get("gaps", [])[:MAX_GAPS_TO_PROCESS]

            for gap in gaps:
                try:
                    question_data = gap.get("question_data", {})
                    if question_data:
                        prompt = self.prompt_generator_agent.process(gap, question_data)
                        gap_prompts.append(prompt)
                except Exception as e:
                    self.logger.warning(f"Error generating prompt for gap: {e}")

            results["steps"]["gap_prompts"] = gap_prompts

        except Exception as e:
            self.logger.error(f"Prompt generator failed: {e}", exc_info=True)
            results["steps"]["gap_prompts"] = []

        results["processing_time"]["prompt_generator"] = round(time.time() - step_start, 2)

    def _step_score_answers(
        self,
        validated_answers: List[Dict],
        session_id: str,
        results: Dict,
        progress_callback: Optional[Callable],
    ) -> None:
        """Step 6: Score and rank answers."""
        step_start = time.time()
        self.logger.info("Step 6: Scoring and ranking answers")

        if progress_callback:
            progress_callback("scoring", "Scoring and ranking answers...")

        try:
            if validated_answers:
                scoring_result = self.scoring_agent.process(
                    answers=validated_answers,
                    wafr_schema=self.wafr_schema,
                    session_id=session_id,
                )
                results["steps"]["scoring"] = scoring_result
            else:
                results["steps"]["scoring"] = {
                    "session_id": session_id,
                    "total_answers": 0,
                    "scored_answers": [],
                    "review_queues": {},
                    "agent": "scoring",
                }

        except Exception as e:
            self.logger.error(f"Scoring agent failed: {e}", exc_info=True)
            results["steps"]["scoring"] = {"error": str(e)}

        results["processing_time"]["scoring"] = round(time.time() - step_start, 2)

    def _step_generate_report(
        self,
        validated_answers: List[Dict],
        gap_result: Dict,
        results: Dict,
        session_id: str,
        progress_callback: Optional[Callable],
    ) -> None:
        """Step 7: Generate comprehensive report."""
        step_start = time.time()
        self.logger.info("Step 7: Generating report")

        if progress_callback:
            progress_callback("report", "Generating comprehensive report...")

        try:
            pillar_coverage = results["steps"].get("mapping", {}).get("pillar_coverage", {})

            assessment_data = {
                "answers": validated_answers,
                "scores": results["steps"].get("confidence", {}).get("summary", {}),
                "gaps": gap_result.get("gaps", []),
                "pillar_coverage": pillar_coverage,
            }

            report_result = self.report_agent.process(assessment_data, session_id)

            if isinstance(report_result, dict):
                report_result = self._sanitize_dict_recursive(report_result)

            results["steps"]["report"] = report_result

        except Exception as e:
            self.logger.error(f"Report agent failed: {e}", exc_info=True)
            error_msg = self._sanitize_error_message(str(e))
            results["steps"]["report"] = {"error": error_msg}

        results["processing_time"]["report"] = round(time.time() - step_start, 2)

    def _step_wa_tool_integration(
        self,
        insights: List[Dict],
        mappings: List[Dict],
        validated_answers: List[Dict],
        transcript: str,
        session_id: str,
        client_name: Optional[str],
        environment: str,
        existing_workload_id: Optional[str],
        generate_report: bool,
        results: Dict,
        progress_callback: Optional[Callable],
    ) -> None:
        """Step 8: WA Tool workload integration."""
        step_start = time.time()
        self.logger.info("Step 8: WA Tool workload integration")

        if progress_callback:
            progress_callback("wa_tool", "Processing WA Tool workload...")

        try:
            # Include synthesized answers in transcript_analysis if available
            synthesized_answers = results["steps"].get("answer_synthesis", {}).get("synthesized_answers", [])
            
            transcript_analysis = {
                "session_id": session_id,
                "insights": insights,
                "mappings": mappings,
                "validated_answers": validated_answers,  # Already includes synthesized answers
                "synthesized_answers": synthesized_answers,  # Also include separately for reference
                "confidence": results["steps"].get("confidence", {}),
                "lens_context": self.lens_context,  # Include detected lens context for workload creation
            }

            # Get or create workload
            workload_id = self._get_or_create_workload(
                existing_workload_id, client_name, environment,
                transcript_analysis, results
            )

            # Populate answers and generate report
            if workload_id:
                self._populate_workload_answers(
                    workload_id, transcript_analysis, transcript,
                    generate_report, client_name, session_id, results
                )

            self.logger.info(f"WA Tool workload processed: {workload_id}")

        except Exception as e:
            self.logger.error(f"WA Tool integration failed: {e}", exc_info=True)
            results["steps"]["wa_workload"] = {"error": str(e)}

        results["processing_time"]["wa_tool"] = round(time.time() - step_start, 2)

    # -------------------------------------------------------------------------
    # Helper Methods - PDF Processing
    # -------------------------------------------------------------------------

    def _process_pdf_files(
        self,
        transcript: str,
        pdf_files: Optional[List[str]],
        results: Dict,
        progress_callback: Optional[Callable],
    ) -> tuple[str, Optional[Dict]]:
        """
        Process PDF files and merge with transcript.

        Returns:
            Tuple of (enhanced_transcript, pdf_content).
        """
        if not pdf_files:
            return transcript, None

        step_start = time.time()
        self.logger.info(f"Step 0: Processing {len(pdf_files)} PDF file(s)")

        if progress_callback:
            progress_callback("pdf_processing", f"Processing {len(pdf_files)} PDF file(s)...")

        try:
            pdf_result = self.pdf_processor.process_multiple_pdfs(pdf_files)

            results["pdf_processing"] = {
                "num_pdfs": pdf_result.get("num_pdfs", 0),
                "text_extracted": len(pdf_result.get("text", "")),
                "images_found": len(pdf_result.get("images", [])),
                "tables_found": len(pdf_result.get("tables", [])),
                "errors": pdf_result.get("errors", []),
                "status": pdf_result.get("processing_status", "unknown"),
            }

            pdf_text = pdf_result.get("text", "")
            if pdf_text:
                enhanced_transcript = self._merge_transcript_with_pdf(transcript, pdf_text)
                self.logger.info(
                    f"Merged {len(pdf_text)} characters from PDF(s) with transcript"
                )
            else:
                self.logger.warning("No text extracted from PDF files")
                enhanced_transcript = transcript

            results["processing_time"]["pdf_processing"] = round(time.time() - step_start, 2)
            return enhanced_transcript, pdf_result

        except Exception as e:
            error_msg = f"PDF processing failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            results["errors"].append(error_msg)
            results["pdf_processing"] = {"status": "error", "error": error_msg}
            return transcript, None

    def _merge_transcript_with_pdf(self, transcript: str, pdf_text: str) -> str:
        """Merge PDF text with transcript content."""
        return f"""
=== WORKSHOP TRANSCRIPT ===
{transcript}

=== PDF DOCUMENTATION ===
{pdf_text}
"""

    def _create_pdf_diagram_insight(
        self,
        pdf_content: Dict,
        pdf_files: Optional[List[str]],
    ) -> Dict:
        """Create an insight entry for PDF diagrams."""
        num_images = len(pdf_content["images"])
        return {
            "insight_type": "pdf_diagram",
            "content": (
                f"PDF contains {num_images} diagram(s) or image(s) "
                "that may contain architecture information"
            ),
            "source": "pdf_processing",
            "metadata": {
                "num_images": num_images,
                "pdf_files": pdf_files,
            },
        }

    # -------------------------------------------------------------------------
    # Helper Methods - WA Tool Integration
    # -------------------------------------------------------------------------

    def _get_or_create_workload(
        self,
        existing_workload_id: Optional[str],
        client_name: Optional[str],
        environment: str,
        transcript_analysis: Dict,
        results: Dict,
    ) -> Optional[str]:
        """Get existing workload or create a new one."""
        if existing_workload_id:
            self.logger.info(f"Using existing workload: {existing_workload_id}")
            workload = self.wa_tool_agent.wa_client.get_workload(existing_workload_id)
            results["steps"]["wa_workload"] = {
                "workload_id": existing_workload_id,
                "workload_arn": workload.get("Workload", {}).get("WorkloadArn"),
                "status": STATUS_EXISTING,
            }
            return existing_workload_id

        self.logger.info("Creating new WA Tool workload...")
        workload = self.wa_tool_agent.create_workload_from_transcript(
            transcript_analysis=transcript_analysis,
            client_name=client_name,
            environment=environment,
        )
        workload_id = workload.get("WorkloadId")
        results["steps"]["wa_workload"] = {
            "workload_id": workload_id,
            "workload_arn": workload.get("WorkloadArn"),
            "status": STATUS_CREATED,
        }
        return workload_id

    def _populate_workload_answers(
        self,
        workload_id: str,
        transcript_analysis: Dict,
        transcript: str,
        generate_report: bool,
        client_name: Optional[str],
        session_id: str,
        results: Dict,
    ) -> None:
        """Populate answers in WA Tool workload and generate report."""
        logger.info("Auto-filling all WAFR questions from transcript...")

        populate_result = self.wa_tool_agent.populate_answers_from_analysis(
            workload_id=workload_id,
            transcript_analysis=transcript_analysis,
            transcript=transcript,
            mapping_agent=self.mapping_agent,
            lens_context=self.lens_context,
        )
        results["steps"]["wa_workload"]["answers_populated"] = populate_result

        updated_count = populate_result.get("updated_answers", 0)
        total_count = populate_result.get("total_questions", 0)
        skipped_count = populate_result.get("skipped_answers", 0)

        logger.info(f"Auto-filled {updated_count} out of {total_count} questions")

        # Handle remaining questions if needed
        if skipped_count > 0 and generate_report:
            self._handle_remaining_questions(
                workload_id, updated_count, total_count, skipped_count, results
            )

        # Create milestone first (needed to get HRIs)
        if generate_report:
            self._create_milestone_and_report(
                workload_id, client_name, session_id, results
            )
            
            # Validate HRIs using Claude (after milestone creation, HRIs are available)
            self._validate_workload_hris(
                workload_id, transcript, transcript_analysis, results, session_id
            )

    def _handle_remaining_questions(
        self,
        workload_id: str,
        updated_count: int,
        total_count: int,
        skipped_count: int,
        results: Dict,
    ) -> None:
        """Handle remaining unanswered questions."""
        user_choice = self._prompt_user_for_remaining_questions(
            updated_count=updated_count,
            total_count=total_count,
            skipped_count=skipped_count,
            workload_id=workload_id,
        )

        if user_choice == CHOICE_MANUAL:
            logger.info("Starting manual question answering...")
            manual_result = self._manual_answer_questions(
                workload_id=workload_id,
                skipped_count=skipped_count,
            )
            results["steps"]["wa_workload"]["manual_answers"] = manual_result
            logger.info(
                f"Manually answered {manual_result.get('updated', 0)} additional questions"
            )

    def _create_milestone_and_report(
        self,
        workload_id: str,
        client_name: Optional[str],
        session_id: str,
        results: Dict,
    ) -> None:
        """Create milestone and generate official WAFR report."""
        logger.info("Creating milestone and generating official WAFR report...")

        report_filename = f"wafr_report_{workload_id}_{session_id[:8]}.pdf"
        milestone_name = f"WAFR Assessment - {client_name or 'Workload'}"

        review_result = self.wa_tool_agent.create_milestone_and_review(
            workload_id=workload_id,
            milestone_name=milestone_name,
            save_report_path=report_filename,
        )

        results["steps"]["wa_workload"]["review"] = review_result
        results["steps"]["wa_workload"]["report_file"] = report_filename
        logger.info(f"Official WAFR report generated: {report_filename}")

    def _validate_workload_hris(
        self,
        workload_id: str,
        transcript: str,
        transcript_analysis: Dict,
        results: Dict,
        session_id: str
    ) -> None:
        """
        Validate High-Risk Issues using Claude to identify only tangible, actionable HRIs.
        
        Args:
            workload_id: Workload ID
            transcript: Original transcript
            transcript_analysis: Transcript analysis results
            results: Results dict to store validation results
            session_id: Session ID
        """
        try:
            self.logger.info("Validating High-Risk Issues using Claude...")
            
            # Get potential HRIs from WA Tool (after milestone creation)
            # Get milestone number from review result
            review_result = results.get("steps", {}).get("wa_workload", {}).get("review", {})
            milestone_number = review_result.get("milestone_number")
            
            potential_hris = self.wa_tool_agent.get_high_risk_issues(
                workload_id, 
                milestone_number=milestone_number
            )
            
            if not potential_hris:
                self.logger.info("No High-Risk Issues found in workload")
                results["steps"]["hri_validation"] = {
                    "validated_hris": [],
                    "filtered_hris": [],
                    "total_potential": 0,
                    "total_validated": 0,
                    "total_filtered": 0
                }
                return
            
            # Get confidence results for context
            confidence_results = results.get("steps", {}).get("confidence", {})
            
            # Validate HRIs using Claude
            validation_result = self._validate_hris_with_claude(
                potential_hris=potential_hris,
                transcript=transcript,
                transcript_analysis=transcript_analysis,
                confidence_results=confidence_results
            )
            
            # Store validation results
            results["steps"]["hri_validation"] = validation_result
            
            validated_count = validation_result.get("total_validated", 0)
            filtered_count = validation_result.get("total_filtered", 0)
            total_potential = validation_result.get("total_potential", 0)
            
            self.logger.info(
                f"HRI Validation complete: {validated_count} tangible HRIs validated, "
                f"{filtered_count} filtered out (false positives/non-tangible)"
            )
            
            if filtered_count > 0:
                self.logger.info(
                    f"  Filtered {filtered_count}/{total_potential} HRIs as non-tangible "
                    f"({filtered_count/total_potential*100:.0f}% reduction)"
                )
            
        except Exception as e:
            self.logger.error(f"Error validating HRIs: {e}", exc_info=True)
            # Don't fail the pipeline if HRI validation fails
            results["steps"]["hri_validation"] = {
                "error": str(e),
                "validated_hris": [],
                "filtered_hris": []
            }
    
    def _validate_hris_with_claude(
        self,
        potential_hris: List[Dict],
        transcript: str,
        transcript_analysis: Dict,
        confidence_results: Dict
    ) -> Dict[str, Any]:
        """
        Validate High-Risk Issues using Claude to identify only tangible, actionable HRIs.
        
        Args:
            potential_hris: List of potential HRIs from WA Tool
            transcript: Original transcript text
            transcript_analysis: Transcript analysis results
            confidence_results: Confidence validation results
            
        Returns:
            Dict with validated_hris, filtered_hris, and counts
        """
        import json
        import re
        import boto3
        from agents.config import BEDROCK_REGION, DEFAULT_MODEL_ID
        
        try:
            # Prepare HRI data for validation
            hri_data = []
            for hri in potential_hris:
                hri_data.append({
                    "question_id": hri.get("question_id", ""),
                    "question_title": hri.get("question_title", ""),
                    "question_description": hri.get("question_description", ""),
                    "selected_choices": hri.get("selected_choices", []),
                    "answer_notes": hri.get("notes", ""),
                    "risk_level": hri.get("risk", "HIGH"),
                    "pillar": hri.get("pillar", "UNKNOWN")
                })
            
            # Get relevant insights and mappings
            insights = transcript_analysis.get("insights", [])
            mappings = transcript_analysis.get("mappings", [])
            
            # Build context
            insights_text = "\n".join([
                f"- {insight.get('content', '')[:200]}"
                for insight in insights[:20]
            ])
            
            mappings_text = "\n".join([
                f"- Q: {m.get('question_id', '')} - {m.get('answer_content', '')[:200]}"
                for m in mappings[:20]
            ])
            
            # Create validation prompt
            prompt = f"""You are an AWS Well-Architected Framework expert validating High-Risk Issues (HRIs).

CRITICAL: Only identify TANGIBLE, ACTIONABLE HRIs that represent real, significant risks.

TANGIBLE HRI CRITERIA:
1. REAL RISK: Represents an actual, significant risk to the workload (not theoretical or minor)
2. EVIDENCE-BASED: Has clear evidence in transcript that supports the risk
3. ACTIONABLE: Can be addressed with specific, concrete actions
4. IMPACT: Would have meaningful negative impact if not addressed
5. NOT FALSE POSITIVE: Not a misunderstanding, missing context, or overly cautious assessment

NON-TANGIBLE HRIs TO FILTER OUT:
- Missing information where transcript doesn't provide enough context (not a real risk)
- Theoretical risks without evidence in transcript
- Minor gaps that don't represent significant risks
- Issues that are already addressed but not explicitly stated
- Overly cautious assessments without real evidence
- Questions where transcript is neutral/unclear (not a risk, just lack of information)

TRANSCRIPT CONTEXT:
{transcript[:6000]}

RELEVANT INSIGHTS:
{insights_text}

RELEVANT MAPPINGS:
{mappings_text}

POTENTIAL HRIs TO VALIDATE:
{json.dumps(hri_data, indent=2)}

TASK: For each potential HRI, determine if it is a TANGIBLE, ACTIONABLE HRI.

For EACH HRI, evaluate:
1. Is there EVIDENCE in transcript that supports this as a real risk?
2. Is this a SIGNIFICANT risk (not minor or theoretical)?
3. Is this ACTIONABLE (can be addressed with specific actions)?
4. Is this a FALSE POSITIVE (misunderstanding, missing context, overly cautious)?

Return ONLY a JSON array with validation results:
[
  {{
    "question_id": "question_id_1",
    "is_tangible_hri": true/false,
    "justification": "Clear explanation of why this is or is not a tangible HRI. If tangible, explain the real risk and evidence. If not, explain why it's a false positive.",
    "risk_severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NOT_TANGIBLE",
    "evidence_in_transcript": "Specific quotes or references from transcript that support or refute this HRI",
    "actionable": true/false,
    "recommended_actions": ["action 1", "action 2"] if tangible, [] if not
  }}
]

STRICT RULES:
- Only mark as tangible if there is CLEAR EVIDENCE in transcript
- Filter out false positives aggressively
- Be conservative - better to filter out a borderline case than report a false positive
- Only report HRIs that represent REAL, SIGNIFICANT risks
- If transcript doesn't provide enough context, it's NOT a tangible HRI (it's just missing information)

Return ONLY the JSON array, no other text."""

            # Use bedrock runtime
            bedrock_runtime = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
            model_id = DEFAULT_MODEL_ID
            
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "temperature": 0.2,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body.get('content', [])
            if content:
                text = content[0].get('text', '')
                validations = self._parse_hri_validation_response(text)
                
                # Separate validated and filtered HRIs
                validated_hris = []
                filtered_hris = []
                
                for validation in validations:
                    question_id = validation.get("question_id")
                    is_tangible = validation.get("is_tangible_hri", False)
                    
                    # Find original HRI
                    original_hri = next(
                        (hri for hri in potential_hris if hri.get("question_id") == question_id),
                        None
                    )
                    
                    if original_hri:
                        # Add validation metadata
                        validated_hri = {
                            **original_hri,
                            "hri_validation": {
                                "is_tangible": is_tangible,
                                "justification": validation.get("justification", ""),
                                "risk_severity": validation.get("risk_severity", "UNKNOWN"),
                                "evidence_in_transcript": validation.get("evidence_in_transcript", ""),
                                "actionable": validation.get("actionable", False),
                                "recommended_actions": validation.get("recommended_actions", [])
                            }
                        }
                        
                        if is_tangible:
                            validated_hris.append(validated_hri)
                        else:
                            filtered_hris.append(validated_hri)
                
                return {
                    "validated_hris": validated_hris,
                    "filtered_hris": filtered_hris,
                    "total_potential": len(potential_hris),
                    "total_validated": len(validated_hris),
                    "total_filtered": len(filtered_hris)
                }
            
            # Fallback: if validation fails, return all as potential (conservative)
            self.logger.warning("HRI validation failed, returning all potential HRIs")
            return {
                "validated_hris": potential_hris,
                "filtered_hris": [],
                "total_potential": len(potential_hris),
                "total_validated": len(potential_hris),
                "total_filtered": 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in HRI validation: {str(e)}", exc_info=True)
            # Fallback: return all as potential (conservative)
            return {
                "validated_hris": potential_hris,
                "filtered_hris": [],
                "total_potential": len(potential_hris),
                "total_validated": len(potential_hris),
                "total_filtered": 0
            }
    
    def _parse_hri_validation_response(self, text: str) -> List[Dict]:
        """Parse Claude's HRI validation response."""
        import json
        import re
        
        if not text:
            return []
        
        try:
            # Try to find JSON array
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Try direct JSON parse
            return json.loads(text)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse HRI validation response as JSON")
            # Try to extract JSON from markdown code blocks
            cleaned = re.sub(r'```json\s*', '', text)
            cleaned = re.sub(r'```\s*', '', cleaned)
            try:
                return json.loads(cleaned)
            except:
                self.logger.error("Could not parse HRI validation response")
                return []

    # -------------------------------------------------------------------------
    # Helper Methods - Answer Validation
    # -------------------------------------------------------------------------

    def _extract_validated_answers(
        self,
        mappings: List[Dict],
        confidence_result: Dict,
    ) -> List[Dict]:
        """Extract validated answers from mappings and confidence results."""
        validation_map = self._build_validation_map(confidence_result)

        answers = []
        for mapping in mappings:
            question_id = mapping.get("question_id")
            validation = validation_map.get(question_id, {})

            if self._should_accept_answer(validation):
                answer = self._build_validated_answer(mapping, validation)
                answers.append(answer)

        return answers

    def _build_validation_map(self, confidence_result: Dict) -> Dict[str, Dict]:
        """Build a map of question_id to validation data."""
        validations = confidence_result.get("all_validations", [])
        approved = confidence_result.get("approved_answers", [])
        review_needed = confidence_result.get("review_needed", [])

        validation_map = {
            v.get("question_id"): v
            for v in validations
            if v.get("question_id")
        }

        # Include approved and review_needed answers
        for answer in approved + review_needed:
            question_id = answer.get("question_id")
            if question_id and question_id not in validation_map:
                validation_map[question_id] = answer

        return validation_map

    def _should_accept_answer(self, validation: Dict) -> bool:
        """Determine if an answer should be accepted based on validation."""
        evidence_verified = validation.get("evidence_verified", False)
        confidence_score = validation.get("confidence_score", 0)
        confidence_level = validation.get("confidence_level", CONFIDENCE_LOW)
        validation_passed = validation.get("validation_passed", False)

        # Accept if:
        # 1. Evidence verified AND confidence >= threshold
        # 2. OR validation explicitly passed
        # 3. OR high/medium confidence with verified evidence
        return (
            (evidence_verified and confidence_score >= MIN_CONFIDENCE_THRESHOLD)
            or validation_passed
            or (confidence_level in [CONFIDENCE_HIGH, CONFIDENCE_MEDIUM] and evidence_verified)
        )

    def _build_validated_answer(self, mapping: Dict, validation: Dict) -> Dict:
        """Build a validated answer dictionary."""
        return {
            "question_id": mapping.get("question_id"),
            "question_text": mapping.get("question_text", ""),
            "pillar": mapping.get("pillar", "UNKNOWN"),
            "answer_content": mapping.get("answer_content", ""),
            "evidence_quotes": [mapping.get("evidence_quote", "")],
            "source": "transcript_direct",
            "confidence_score": validation.get("confidence_score", 0),
            "confidence_level": validation.get("confidence_level", CONFIDENCE_LOW),
            "evidence_verified": validation.get("evidence_verified", False),
            "validation_passed": validation.get("validation_passed", False),
        }

    # -------------------------------------------------------------------------
    # Helper Methods - User Interaction
    # -------------------------------------------------------------------------

    def _prompt_user_for_remaining_questions(
        self,
        updated_count: int,
        total_count: int,
        skipped_count: int,
        workload_id: str,
    ) -> str:
        """
        Prompt user to choose how to handle remaining questions.

        Returns:
            'manual' or 'proceed'.
        """
        coverage_pct = (updated_count / total_count * 100) if total_count else 0

        self._print_question_summary(updated_count, skipped_count, total_count, coverage_pct)

        while True:
            try:
                choice = input(
                    "\n  Enter your choice (1/2 or 'manual'/'proceed'): "
                ).strip().lower()

                if choice in ["1", "manual", "m"]:
                    return CHOICE_MANUAL
                elif choice in ["2", "proceed", "p", ""]:
                    return CHOICE_PROCEED
                else:
                    print("  ❌ Invalid choice. Please enter 1, 2, 'manual', or 'proceed'")

            except (EOFError, KeyboardInterrupt):
                print("\n  ⚠️  Interrupted. Proceeding with current answers...")
                return CHOICE_PROCEED

    def _print_question_summary(
        self,
        updated_count: int,
        skipped_count: int,
        total_count: int,
        coverage_pct: float,
    ) -> None:
        """Print the question auto-fill summary."""
        print("\n" + "=" * 70)
        print("  QUESTION AUTO-FILL SUMMARY")
        print("=" * 70)
        print(f"  ✅ Auto-filled: {updated_count} questions")
        print(f"  ⏭️  Skipped: {skipped_count} questions")
        print(f"  📊 Total: {total_count} questions")
        print(f"  📈 Coverage: {coverage_pct:.1f}%")
        print("=" * 70)
        print("\n  The transcript may not contain answers to all questions.")
        print("  You have two options:\n")
        print("  1. MANUALLY ANSWER remaining questions")
        print("     → Answer the skipped questions interactively")
        print("     → Then generate the complete report")
        print("\n  2. PROCEED WITH CURRENT ANSWERS")
        print("     → Generate report with auto-filled questions only")
        print("     → Skipped questions will remain unanswered")
        print("\n" + "-" * 70)

    def _manual_answer_questions(
        self,
        workload_id: str,
        skipped_count: int,
    ) -> Dict[str, Any]:
        """
        Interactive interface for manually answering remaining questions.

        Returns:
            Dict with updated count and details.
        """
        self._print_manual_answer_header()

        while True:
            try:
                method = input(
                    "\n  Choose method (1/2 or 'console'/'skip'): "
                ).strip().lower()

                if method in ["1", "interactive", "cli"]:
                    return self._interactive_answer_questions(workload_id)
                elif method in ["2", "console", "c"]:
                    return self._handle_console_answering(workload_id)
                elif method in ["skip", "s", ""]:
                    print("  ⏭️  Skipping manual answers. Proceeding with current state...")
                    return {"updated": 0, "method": "skipped"}
                else:
                    print("  ❌ Invalid choice. Please enter 1, 2, 'console', or 'skip'")

            except (EOFError, KeyboardInterrupt):
                print("\n  ⚠️  Interrupted. Skipping manual answers...")
                return {"updated": 0, "method": "interrupted"}

    def _print_manual_answer_header(self) -> None:
        """Print the manual answering header."""
        print("\n" + "=" * 70)
        print("  MANUAL QUESTION ANSWERING")
        print("=" * 70)
        print("  You can answer questions via:")
        print("  1. Interactive CLI (answer questions one by one)")
        print("  2. AWS Console (then continue)")
        print("\n" + "-" * 70)

    def _handle_console_answering(self, workload_id: str) -> Dict[str, Any]:
        """Handle answering via AWS Console."""
        console_url = (
            f"https://console.aws.amazon.com/wellarchitected/home"
            f"?#/workloads/{workload_id}"
        )
        print(f"\n  📋 Please answer questions in AWS Console:")
        print(f"     {console_url}")
        input("\n  Press Enter when you've finished answering questions in the console...")
        return {
            "updated": 0,
            "method": CHOICE_CONSOLE,
            "note": "User answered via AWS Console",
        }

    def _interactive_answer_questions(self, workload_id: str) -> Dict[str, Any]:
        """
        Interactive CLI to answer questions one by one.

        Returns:
            Dict with updated count.
        """
        print("\n" + "=" * 70)
        print("  INTERACTIVE QUESTION ANSWERING")
        print("=" * 70)
        print("  Answer questions one by one. Type 'skip' to skip a question.")
        print("  Type 'done' when finished.\n")

        try:
            unanswered_questions = self._get_unanswered_questions(workload_id)

            if not unanswered_questions:
                print("  ✅ All questions are already answered!")
                return {
                    "updated": 0,
                    "method": "interactive",
                    "note": "All questions already answered",
                }

            print(f"  Found {len(unanswered_questions)} unanswered questions.\n")
            return self._process_unanswered_questions(workload_id, unanswered_questions)

        except Exception as e:
            logger.error(f"Error in interactive answering: {e}")
            return {"updated": 0, "method": "interactive", "error": str(e)}

    def _get_unanswered_questions(self, workload_id: str) -> List[Dict]:
        """Get list of unanswered questions for a workload."""
        all_questions = self.wa_tool_agent._get_all_questions(
            workload_id=workload_id,
            lens_alias=DEFAULT_LENS_ALIAS,
        )

        answered_question_ids = set()

        for question_summary in all_questions:
            question_id = question_summary.get("QuestionId")
            if not question_id:
                continue

            try:
                answer_details = self.wa_tool_agent.wa_client.get_answer(
                    workload_id=workload_id,
                    lens_alias=DEFAULT_LENS_ALIAS,
                    question_id=question_id,
                )
                selected_choices = answer_details.get("Answer", {}).get("SelectedChoices", [])
                if selected_choices:
                    answered_question_ids.add(question_id)
            except Exception:
                pass

        return [
            q for q in all_questions
            if q.get("QuestionId") not in answered_question_ids
        ]

    def _process_unanswered_questions(
        self,
        workload_id: str,
        unanswered_questions: List[Dict],
    ) -> Dict[str, Any]:
        """Process unanswered questions interactively."""
        updated_count = 0
        total_questions = len(unanswered_questions)

        for idx, question_summary in enumerate(unanswered_questions, 1):
            result = self._process_single_question(
                workload_id, question_summary, idx, total_questions, updated_count
            )

            if result == "done":
                break
            elif result == "updated":
                updated_count += 1
            elif result == "interrupted":
                return {
                    "updated": updated_count,
                    "method": "interactive",
                    "interrupted": True,
                }

        print(f"\n  ✅ Finished answering questions. Updated {updated_count} questions.")
        return {"updated": updated_count, "method": "interactive"}

    def _process_single_question(
        self,
        workload_id: str,
        question_summary: Dict,
        idx: int,
        total: int,
        current_count: int,
    ) -> str:
        """
        Process a single question interactively.

        Returns:
            'updated', 'skipped', 'done', or 'interrupted'.
        """
        question_id = question_summary.get("QuestionId")
        question_title = question_summary.get("QuestionTitle", "")
        pillar_name = question_summary.get("PillarName", "Unknown")

        print(f"\n  [{idx}/{total}] {pillar_name}")
        print(f"  Question: {question_title}")
        print(f"  ID: {question_id}")
        print("-" * 70)

        try:
            answer_details = self.wa_tool_agent.wa_client.get_answer(
                workload_id=workload_id,
                lens_alias=DEFAULT_LENS_ALIAS,
                question_id=question_id,
            )
            choices = answer_details.get("Answer", {}).get("Choices", [])

            self._print_choices(choices)

            return self._get_user_answer_choice(
                workload_id, question_id, choices, current_count
            )

        except Exception as e:
            logger.warning(f"Error getting question details for {question_id}: {e}")
            return "skipped"

    def _print_choices(self, choices: List[Dict]) -> None:
        """Print available choices for a question."""
        print("\n  Available Choices:")
        for i, choice in enumerate(choices, 1):
            choice_id = choice.get("ChoiceId", "")
            title = choice.get("Title", "")
            desc = choice.get("Description", "")
            print(f"    {i}. [{choice_id}] {title}")
            if desc:
                print(f"       {desc[:80]}...")

    def _get_user_answer_choice(
        self,
        workload_id: str,
        question_id: str,
        choices: List[Dict],
        current_count: int,
    ) -> str:
        """Get user's answer choice and update the workload."""
        while True:
            try:
                user_input = input(
                    "\n  Enter choice number(s) (comma-separated) or 'skip': "
                ).strip()

                if user_input.lower() in ["skip", "s", ""]:
                    print("  ⏭️  Skipped")
                    return "skipped"

                if user_input.lower() == "done":
                    print(f"\n  ✅ Finished. Updated {current_count} questions.")
                    return "done"

                choice_numbers = self._parse_choice_numbers(user_input)

                if not choice_numbers:
                    print("  ❌ Invalid input. Please enter choice number(s) or 'skip'")
                    continue

                if not self._validate_choice_numbers(choice_numbers, len(choices)):
                    print(f"  ❌ Invalid choice number(s). Please enter 1-{len(choices)}")
                    continue

                selected_choice_ids = [
                    choices[n - 1].get("ChoiceId")
                    for n in choice_numbers
                ]

                notes = input("  Enter notes (optional, press Enter to skip): ").strip()

                self.wa_tool_agent.wa_client.update_answer(
                    workload_id=workload_id,
                    lens_alias=DEFAULT_LENS_ALIAS,
                    question_id=question_id,
                    selected_choices=selected_choice_ids,
                    notes=notes or "",
                    is_applicable=True,
                )

                print(f"  ✅ Updated answer for {question_id}")
                return "updated"

            except (EOFError, KeyboardInterrupt):
                print(f"\n  ⚠️  Interrupted. Updated {current_count} questions so far.")
                return "interrupted"
            except ValueError:
                print("  ❌ Invalid input. Please enter numbers or 'skip'")
            except Exception as e:
                print(f"  ❌ Error updating answer: {e}")

    def _parse_choice_numbers(self, user_input: str) -> List[int]:
        """Parse comma-separated choice numbers from user input."""
        return [
            int(x.strip())
            for x in user_input.split(",")
            if x.strip().isdigit()
        ]

    def _validate_choice_numbers(
        self,
        choice_numbers: List[int],
        num_choices: int,
    ) -> bool:
        """Validate that all choice numbers are within valid range."""
        return all(1 <= n <= num_choices for n in choice_numbers)

    # -------------------------------------------------------------------------
    # Helper Methods - Utilities
    # -------------------------------------------------------------------------

    def _create_initial_results(self, session_id: str) -> Dict[str, Any]:
        """Create initial results dictionary."""
        return {
            "session_id": session_id,
            "status": STATUS_PROCESSING,
            "steps": {},
            "processing_time": {},
            "errors": [],
            "pdf_processing": {},
        }

    def _finalize_results(
        self,
        results: Dict,
        insights: List,
        mappings: List,
        validated_answers: List,
        gap_result: Dict,
        start_time: float,
    ) -> None:
        """Finalize results with status and summary."""
        if results.get("errors"):
            results["status"] = STATUS_COMPLETED_WITH_ERRORS
            self.logger.warning(f"Processing completed with {len(results['errors'])} errors")
        else:
            results["status"] = STATUS_COMPLETED

        results["processing_time"]["total"] = round(time.time() - start_time, 2)

        # Build summary with safe defaults
        confidence_summary = results["steps"].get("confidence", {}).get("summary", {})
        gaps = gap_result.get("gaps", []) if isinstance(gap_result, dict) else []

        results["summary"] = {
            "total_insights": len(insights) if insights else 0,
            "total_mappings": len(mappings) if mappings else 0,
            "total_answers": len(validated_answers) if validated_answers else 0,
            "total_gaps": len(gaps),
            "confidence_score": confidence_summary.get("average_score", 0) if confidence_summary else 0,
        }

    def _normalize_pillar_coverage(self, pillar_coverage: Dict) -> Dict[str, float]:
        """Normalize pillar coverage to simple float percentages."""
        return {
            k: v.get("coverage_pct", 0) if isinstance(v, dict) else 0
            for k, v in pillar_coverage.items()
        }

    def _generate_session_id(self, prefix: str) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"{prefix}-{timestamp}-{unique_id}"

    def _auto_detect_lenses(self, content: str) -> None:
        """Auto-detect relevant lenses from content."""
        if self.lens_context or not content:
            return

        try:
            from agents.lens_manager import create_lens_manager

            aws_region = getattr(self, "aws_region", DEFAULT_AWS_REGION)
            lens_manager = create_lens_manager(aws_region=aws_region)

            detected = lens_manager.detect_relevant_lenses(
                content,
                min_confidence=MIN_LENS_DETECTION_CONFIDENCE,
            )

            if detected:
                self.logger.info(
                    f"Auto-detected {len(detected)} relevant lenses from file content"
                )
                selected_lenses = lens_manager.auto_select_lenses(
                    content,
                    max_lenses=MAX_AUTO_SELECT_LENSES,
                )
                self.lens_context = lens_manager.get_lens_context_for_agents(selected_lenses)
                self.logger.info(f"Auto-selected lenses: {', '.join(selected_lenses)}")

        except Exception as e:
            self.logger.warning(f"Failed to auto-detect lenses: {e}")

    def _build_input_metadata(self, processed_input) -> Dict[str, Any]:
        """Build input metadata dictionary from processed input."""
        return {
            "source_type": processed_input.input_type.value,
            "source_file": processed_input.source_file,
            "word_count": processed_input.word_count,
            "extraction_confidence": processed_input.confidence,
            "processing_metadata": processed_input.metadata,
        }

    def _get_question_data(
        self,
        question_id: str,
        wafr_schema: Dict,
    ) -> Optional[Dict]:
        """Get question data from schema by question ID."""
        if not wafr_schema or "pillars" not in wafr_schema:
            return None

        for pillar in wafr_schema["pillars"]:
            for question in pillar.get("questions", []):
                if question.get("id") == question_id:
                    question["pillar_id"] = pillar.get("id", "UNKNOWN")
                    return question

        return None

    def _sanitize_dict_recursive(self, data: Any) -> Any:
        """Recursively sanitize dictionary values for Unicode safety."""
        if isinstance(data, dict):
            return {k: self._sanitize_dict_recursive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_dict_recursive(item) for item in data]
        elif isinstance(data, str):
            return self._sanitize_string(data)
        return data

    def _sanitize_string(self, text: str) -> str:
        """Sanitize a string for Unicode safety."""
        try:
            text.encode("utf-8")
            return text
        except UnicodeEncodeError:
            return text.encode("utf-8", errors="replace").decode("utf-8")

    def _sanitize_error_message(self, error_msg: str) -> str:
        """Sanitize error message for Unicode safety."""
        try:
            error_msg.encode("utf-8")
            return error_msg
        except UnicodeEncodeError:
            return error_msg.encode("utf-8", errors="replace").decode("utf-8")


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def _load_schema_from_file(schema_path: str) -> Dict[str, Any]:
    """Load schema from file with caching."""
    def load() -> Dict[str, Any]:
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading schema from {schema_path}: {e}")
            return {"pillars": []}

    cache_key = f"schema_{schema_path}"
    return cache_result(_schema_cache, cache_key, load, ttl=SCHEMA_CACHE_TTL_SECONDS)


def create_orchestrator(
    wafr_schema: Optional[Dict] = None,
    input_processor: Optional[InputProcessor] = None,
    aws_region: str = DEFAULT_AWS_REGION,
    lens_context: Optional[Dict] = None,
) -> WafrOrchestrator:
    """
    Factory function to create WAFR orchestrator.

    Args:
        wafr_schema: Optional WAFR schema (loads from AWS API first, then file).
        input_processor: Optional input processor for file handling.
        aws_region: AWS region for services like Textract.
        lens_context: Optional lens context for multi-lens support.

    Returns:
        Configured WafrOrchestrator instance.
    """
    if wafr_schema is None:
        from agents.wafr_context import load_wafr_schema, get_schema_source

        wafr_schema = load_wafr_schema(use_aws_api=True)
        source = get_schema_source()

        if source == "aws_api":
            logger.info("Using official AWS Well-Architected Framework schema from AWS API")
        elif source == "file":
            logger.info("Using WAFR schema from file (AWS API not available)")
        else:
            logger.warning("WAFR schema not found, using empty schema")
            wafr_schema = {"pillars": []}

    return WafrOrchestrator(
        wafr_schema,
        input_processor=input_processor,
        lens_context=lens_context,
    )