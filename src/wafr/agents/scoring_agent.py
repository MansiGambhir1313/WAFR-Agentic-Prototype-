"""
Answer Scoring and Ranking Agent - Multi-dimensional scoring with grade assignment
Uses Strands framework
"""
import json
import logging
from typing import Any, Dict, List, Optional

from strands import Agent, tool

from wafr.agents.config import DEFAULT_MODEL_ID
from wafr.agents.model_config import get_strands_model
from wafr.agents.utils import (
    batch_process,
    extract_json_from_text,
    retry_with_backoff,
)
from wafr.agents.wafr_context import get_question_context, load_wafr_schema

logger = logging.getLogger(__name__)


def get_scoring_system_prompt(wafr_schema: Optional[Dict[str, Any]] = None) -> str:
    """Generate enhanced system prompt with WAFR context."""
    base_prompt = """
You are an expert WAFR (AWS Well-Architected Framework Review) evaluator. You score answers on multiple dimensions using WAFR best practices.

SCORING DIMENSIONS:

1. CONFIDENCE (40% weight): Evidence quality and verification
   - Evidence citations present and verifiable
   - Evidence verified in transcript
   - Source reliability and accuracy
   - No unsupported claims or assumptions

2. COMPLETENESS (30% weight): How well answer addresses the WAFR question
   - Best practices from WAFR schema addressed
   - Answer specificity and detail
   - AWS service mentions with context
   - Coverage of question intent

3. COMPLIANCE (30% weight): Alignment with WAFR best practices
   - Adherence to recommended best practices
   - Anti-pattern penalties (if present)
   - HRI (High-Risk Issue) indicators (negative impact)
   - Recommended AWS services mentioned (positive)
   - Alignment with WAFR pillar principles

GRADE ASSIGNMENT:
- A (90-100): Excellent, fully compliant with WAFR best practices
- B (80-89): Good, minor gaps, mostly aligned with best practices
- C (70-79): Adequate, some improvements needed, partial compliance
- D (60-69): Needs significant work, missing key best practices
- F (<60): Critical gaps, non-compliant, or missing essential practices

SCORING PROCESS:
1. Review the answer against WAFR question best practices
2. Assess evidence quality and verification
3. Evaluate completeness against question requirements
4. Check compliance with WAFR best practices
5. Calculate composite score using weighted average
6. Assign grade based on composite score
7. Identify best practices met and missing
8. Flag any HRI indicators

Be thorough and fair in your evaluation. Use WAFR best practices as the standard.
"""
    
    return base_prompt


@tool
def calculate_composite_score(
    confidence_score: float,
    completeness_score: float,
    compliance_score: float
) -> Dict:
    """
    Calculate composite score from three dimensions.
    
    Args:
        confidence_score: Confidence score (0-100)
        completeness_score: Completeness score (0-100)
        compliance_score: Compliance score (0-100)
        
    Returns:
        Composite score and grade
    """
    # Weighted average
    composite = (
        confidence_score * 0.4 +
        completeness_score * 0.3 +
        compliance_score * 0.3
    )
    
    # Assign grade
    if composite >= 90:
        grade = 'A'
    elif composite >= 80:
        grade = 'B'
    elif composite >= 70:
        grade = 'C'
    elif composite >= 60:
        grade = 'D'
    else:
        grade = 'F'
    
    return {
        'composite_score': round(composite, 2),
        'grade': grade,
        'confidence': round(confidence_score, 2),
        'completeness': round(completeness_score, 2),
        'compliance': round(compliance_score, 2)
    }


@tool
def assess_answer(
    answer: str,
    question_id: str,
    best_practices: List[Dict],
    evidence_quotes: List[str],
    source: str
) -> Dict:
    """
    Assess an answer and return scores.
    
    Args:
        answer: Answer content
        question_id: Question identifier
        best_practices: List of best practices for question
        evidence_quotes: List of evidence quotes
        source: Answer source (transcript_direct, user_input, etc.)
        
    Returns:
        Assessment with scores
    """
    # This would be enhanced by LLM reasoning
    # For now, provide structure
    return {
        'question_id': question_id,
        'answer': answer,
        'best_practices_met': [],
        'best_practices_missing': [],
        'hri_indicators': [],
        'improvement_suggestions': []
    }


@tool
def calculate_rank_priority(
    question_id: str,
    grade: str,
    hri_indicators: List[str],
    criticality: str,
    source: str
) -> int:
    """
    Calculate review priority (1 = highest priority).
    Lower number = needs attention first.
    
    Args:
        question_id: Question identifier
        grade: Answer grade (A-F)
        hri_indicators: List of HRI indicators
        criticality: Question criticality
        source: Answer source
        
    Returns:
        Priority rank (1-100)
    """
    priority = 100
    
    # Criticality adjustment
    criticality_adjustments = {
        'critical': -40,
        'high': -20,
        'medium': 0,
        'low': 20
    }
    priority += criticality_adjustments.get(criticality, 0)
    
    # Grade adjustment
    grade_adjustments = {
        'F': -30,
        'D': -15,
        'C': 0,
        'B': 15,
        'A': 30
    }
    priority += grade_adjustments.get(grade, 0)
    
    # HRI adjustment
    if hri_indicators:
        priority -= 25
    
    # Source adjustment
    if source == 'not_answered':
        priority -= 35
    elif source == 'transcript_inferred':
        priority -= 10
    
    return max(1, priority)


class ScoringAgent:
    """Agent that scores and ranks WAFR answers."""
    
    def __init__(self, wafr_schema: Optional[Dict] = None):
        """
        Initialize Scoring Agent.
        
        Args:
            wafr_schema: Optional WAFR schema for context
        """
        if wafr_schema is None:
            wafr_schema = load_wafr_schema()
        
        self.wafr_schema = wafr_schema
        system_prompt = get_scoring_system_prompt(wafr_schema)
        
        try:
            model = get_strands_model(DEFAULT_MODEL_ID)
            agent_kwargs = {
                'system_prompt': system_prompt,
                'name': 'ScoringAgent'
            }
            if model:
                agent_kwargs['model'] = model
            
            self.agent = Agent(**agent_kwargs)
            # Try to add tools if method exists
            try:
                try:
                    self.agent.add_tool(calculate_composite_score)
                    self.agent.add_tool(assess_answer)
                    self.agent.add_tool(calculate_rank_priority)
                except AttributeError:
                    try:
                        self.agent.register_tool(calculate_composite_score)
                        self.agent.register_tool(assess_answer)
                        self.agent.register_tool(calculate_rank_priority)
                    except AttributeError:
                        pass  # Tools may be auto-detected
            except Exception as e:
                logger.warning(f"Could not add tools to scoring agent: {e}")
        except Exception as e:
            logger.warning(f"Strands Agent initialization issue: {e}, using direct Bedrock")
            self.agent = None
    
    def process(
        self,
        answers: List[Dict],
        wafr_schema: Dict,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Score and rank all answers.
        
        Args:
            answers: List of answer dictionaries
            wafr_schema: WAFR schema with best practices
            session_id: Session identifier
            
        Returns:
            Scored and ranked answers
        """
        logger.info(f"ScoringAgent: Scoring {len(answers)} answers for session {session_id}")
        
        if not answers:
            return {
                'session_id': session_id,
                'total_answers': 0,
                'scored_answers': [],
                'review_queues': self._organize_review_queues([]),
                'agent': 'scoring'
            }
        
        # Process answers in batches
        def process_answer(answer: Dict) -> Dict:
            question_id = answer.get('question_id')
            question_data = self._get_question_data(question_id, wafr_schema)
            
            if not question_data:
                logger.warning(f"Question data not found for {question_id}")
                return None
            
            # Get detailed question context
            question_context = get_question_context(question_id, self.wafr_schema)
            context_section = ""
            if question_context:
                context_section = f"\n\nWAFR QUESTION CONTEXT:\n{question_context}\n"
            
            best_practices = question_data.get('best_practices', [])
            bp_text = "\n".join([f"  - {bp.get('text', '')}" for bp in best_practices])
            
            hri_indicators = question_data.get('hri_indicators', [])
            hri_text = ""
            if hri_indicators:
                hri_text = f"\n\nHIGH-RISK ISSUE INDICATORS (if present, reduce score):\n" + "\n".join([f"  - {hri}" for hri in hri_indicators])
            
            # Use agent to score answer
            prompt = f"""
            Score this WAFR answer using WAFR best practices:
            
            Question ID: {question_id}
            Question: {answer.get('question_text', '')}
            Answer: {answer.get('answer_content', '')}
            Evidence Quotes: {answer.get('evidence_quotes', [])}
            Source: {answer.get('source', 'unknown')}
            Confidence Score: {answer.get('confidence_score', 0)}
            {context_section}
            
            BEST PRACTICES FOR THIS QUESTION:
            {bp_text}
            {hri_text}
            
            SCORING INSTRUCTIONS:
            1. Use assess_answer() to analyze the answer against best practices
            2. Evaluate CONFIDENCE (40%): Evidence quality, verification, reliability
            3. Evaluate COMPLETENESS (30%): How well it addresses the question and best practices
            4. Evaluate COMPLIANCE (30%): Alignment with WAFR best practices, HRI indicators
            5. Use calculate_composite_score() to get final grade
            6. Identify which best practices are met and which are missing
            7. Flag any HRI indicators present in the answer
            
            Return complete scoring breakdown with all three dimension scores and final grade.
            """
            
            try:
                response = self._call_agent_with_retry(prompt)
                scores = self._parse_scoring(response, answer, question_data)
                return scores
            except Exception as e:
                logger.error(f"Error scoring answer: {str(e)}")
                return self._create_default_scores_for_answer(answer, question_data)
        
        # Process in batches if many answers
        scored_answers = []
        if len(answers) > 5:
            results = batch_process(
                answers,
                process_answer,
                batch_size=5,
                max_workers=3,
                timeout=120.0
            )
            scored_answers = [r for r in results if r is not None]
        else:
            for answer in answers:
                result = process_answer(answer)
                if result:
                    scored_answers.append(result)
        
        # Calculate priorities and organize into queues
        for scored in scored_answers:
            priority = calculate_rank_priority(
                question_id=scored['question_id'],
                grade=scored['grade'],
                hri_indicators=scored.get('hri_indicators', []),
                criticality=scored.get('criticality', 'medium'),
                source=scored.get('source', 'unknown')
            )
            scored['rank_priority'] = priority
        
        # Sort by priority
        scored_answers.sort(key=lambda x: x.get('rank_priority', 100))
        
        # Organize into review queues
        queues = self._organize_review_queues(scored_answers)
        
        return {
            'session_id': session_id,
            'total_answers': len(scored_answers),
            'scored_answers': scored_answers,
            'review_queues': queues,
            'agent': 'scoring'
        }
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def _call_agent_with_retry(self, prompt: str) -> Any:
        """Call agent with retry logic."""
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        return self.agent(prompt)
    
    def _get_question_data(self, question_id: str, wafr_schema: Dict) -> Optional[Dict]:
        """Get question data from schema."""
        if not wafr_schema or 'pillars' not in wafr_schema:
            return None
        
        for pillar in wafr_schema['pillars']:
            for question in pillar.get('questions', []):
                if question.get('id') == question_id:
                    return question
        
        return None
    
    def _parse_scoring(self, response: Any, answer: Dict, question_data: Dict) -> Dict:
        """Parse scoring response from agent with improved JSON extraction."""
        try:
            if isinstance(response, dict):
                scores = response
            elif isinstance(response, str):
                parsed = extract_json_from_text(response, strict=False)
                if parsed and isinstance(parsed, dict):
                    scores = parsed
                else:
                    scores = self._create_default_scores()
            else:
                parsed = extract_json_from_text(str(response), strict=False)
                if parsed and isinstance(parsed, dict):
                    scores = parsed
                else:
                    scores = self._create_default_scores()
        except Exception as e:
            logger.error(f"Error parsing scoring: {str(e)}")
            scores = self._create_default_scores()
        
        # Merge with answer data
        scores.update({
            'question_id': answer.get('question_id'),
            'question_text': answer.get('question_text'),
            'pillar': answer.get('pillar'),
            'answer_content': answer.get('answer_content'),
            'evidence_quotes': answer.get('evidence_quotes', []),
            'source': answer.get('source', 'unknown'),
            'criticality': question_data.get('criticality', 'medium')
        })
        
        # Ensure scores exist
        if 'composite_score' not in scores:
            scores['composite_score'] = 70.0
        if 'grade' not in scores:
            # Calculate grade from composite score
            composite = scores.get('composite_score', 70.0)
            if composite >= 90:
                scores['grade'] = 'A'
            elif composite >= 80:
                scores['grade'] = 'B'
            elif composite >= 70:
                scores['grade'] = 'C'
            elif composite >= 60:
                scores['grade'] = 'D'
            else:
                scores['grade'] = 'F'
        
        return scores
    
    def _create_default_scores_for_answer(self, answer: Dict, question_data: Dict) -> Dict:
        """Create default scores for an answer."""
        scores = self._create_default_scores()
        scores.update({
            'question_id': answer.get('question_id'),
            'question_text': answer.get('question_text'),
            'pillar': answer.get('pillar'),
            'answer_content': answer.get('answer_content'),
            'evidence_quotes': answer.get('evidence_quotes', []),
            'source': answer.get('source', 'unknown'),
            'criticality': question_data.get('criticality', 'medium')
        })
        return scores
    
    def _create_default_scores(self) -> Dict:
        """Create default scoring structure."""
        return {
            'confidence_score': 50.0,
            'completeness_score': 50.0,
            'compliance_score': 50.0,
            'composite_score': 50.0,
            'grade': 'F',
            'best_practices_met': [],
            'best_practices_missing': [],
            'hri_indicators': []
        }
    
    def _organize_review_queues(self, scored_answers: List[Dict]) -> Dict:
        """Organize answers into review queues."""
        critical = [a for a in scored_answers if a['grade'] == 'F' or a.get('hri_indicators')]
        needs_improvement = [a for a in scored_answers if a['grade'] == 'D' and not a.get('hri_indicators')]
        suggested_review = [a for a in scored_answers if a['grade'] == 'C']
        auto_approved = [a for a in scored_answers if a['grade'] in ['A', 'B']]
        
        return {
            'critical_review': critical,
            'needs_improvement': needs_improvement,
            'suggested_review': suggested_review,
            'auto_approved': auto_approved,
            'summary': {
                'total': len(scored_answers),
                'critical': len(critical),
                'with_hri': len([a for a in scored_answers if a.get('hri_indicators')]),
                'auto_approved_pct': round(len(auto_approved) / len(scored_answers) * 100, 1) if scored_answers else 0
            }
        }


def create_scoring_agent(wafr_schema: Optional[Dict] = None) -> ScoringAgent:
    """
    Factory function to create Scoring Agent.
    
    Args:
        wafr_schema: Optional WAFR schema for context
    """
    return ScoringAgent(wafr_schema)

