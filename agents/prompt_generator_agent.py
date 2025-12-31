"""
Smart Prompt Generator Agent - Generates context-aware prompts for gaps
Uses Strands framework
"""
import json
import logging
from typing import Any, Dict, List, Optional

from strands import Agent, tool

from agents.config import DEFAULT_MODEL_ID
from agents.model_config import get_strands_model
from agents.wafr_context import load_wafr_schema

logger = logging.getLogger(__name__)


def get_prompt_generator_system_prompt(wafr_schema: Optional[Dict[str, Any]] = None) -> str:
    """Generate enhanced system prompt with WAFR context."""
    base_prompt = """
You are an expert at generating intelligent, context-aware prompts to help users
answer WAFR (AWS Well-Architected Framework Review) questions.

PROMPT GENERATION PRINCIPLES:
1. Be clear and specific about what's needed for the WAFR question
2. Include relevant hints based on WAFR best practices from the schema
3. Provide example good answers based on best practice examples
4. Reference any related discussion from transcript (context hints)
5. Offer quick-select options for common answer patterns
6. Make prompts actionable and user-friendly
7. Align with WAFR pillar principles and best practices
8. For partial evidence: Show what we understood and ask for clarification
9. For incomplete knowledge: Request specific information needed
10. Always frame requests professionally and helpfully

PROMPT STRUCTURE:
- Question text (from WAFR schema)
- Pillar and criticality information
- Hints based on best practices (top 3)
- Context from transcript (if available)
- Example good answer (from best practices)
- Quick-select options (common answer patterns)

Use generate_smart_prompt() to create comprehensive prompts that guide users
to provide complete, WAFR-aligned answers.
"""
    
    return base_prompt


@tool
def generate_smart_prompt(
    question_id: str,
    question_text: str,
    pillar: str,
    criticality: str,
    hints: List[str],
    example_answer: str = None,
    context_hint: str = None,
    quick_options: List[str] = None
) -> Dict:
    """
    Generate a smart prompt for a gap question.
    
    Args:
        question_id: Question identifier
        question_text: Full question text
        pillar: Pillar name
        criticality: Criticality level
        hints: List of hints based on best practices
        example_answer: Example good answer
        context_hint: Context from transcript if available
        quick_options: Quick-select answer options
        
    Returns:
        Prompt dictionary
    """
    return {
        "question_id": question_id,
        "question_text": question_text,
        "pillar": pillar,
        "criticality": criticality,
        "hints": hints,
        "example_answer": example_answer,
        "context_hint": context_hint,
        "quick_options": quick_options or [],
        "prompt_text": _format_prompt_text(
            question_text, pillar, hints, example_answer, context_hint
        )
    }


def _format_prompt_text(
    question_text: str,
    pillar: str,
    hints: List[str],
    example_answer: str = None,
    context_hint: str = None
) -> str:
    """Format prompt text from components."""
    lines = [
        f"**{pillar} Question:**",
        question_text,
        "",
        "**Hints:**"
    ]
    
    for i, hint in enumerate(hints[:3], 1):
        lines.append(f"{i}. {hint}")
    
    if context_hint:
        lines.append("")
        lines.append(f"**Context:** {context_hint}")
    
    if example_answer:
        lines.append("")
        lines.append("**Example Answer:**")
        lines.append(example_answer)
    
    return "\n".join(lines)


class PromptGeneratorAgent:
    """Agent that generates smart prompts for gap questions."""
    
    def __init__(self, wafr_schema: Optional[Dict] = None):
        """
        Initialize Prompt Generator Agent.
        
        Args:
            wafr_schema: Optional WAFR schema for context
        """
        if wafr_schema is None:
            wafr_schema = load_wafr_schema()
        
        self.wafr_schema = wafr_schema
        system_prompt = get_prompt_generator_system_prompt(wafr_schema)
        
        try:
            model = get_strands_model(DEFAULT_MODEL_ID)
            agent_kwargs = {
                'system_prompt': system_prompt,
                'name': 'PromptGeneratorAgent'
            }
            if model:
                agent_kwargs['model'] = model
            
            self.agent = Agent(**agent_kwargs)
            # Try to add tool if method exists
            try:
                try:
                    self.agent.add_tool(generate_smart_prompt)
                except AttributeError:
                    try:
                        self.agent.register_tool(generate_smart_prompt)
                    except AttributeError:
                        pass  # Tools may be auto-detected
            except Exception as e:
                logger.warning(f"Could not add tools to prompt generator agent: {e}")
        except Exception as e:
            logger.warning(f"Strands Agent initialization issue: {e}, using direct Bedrock")
            self.agent = None
    
    def process(self, gap: Dict, wafr_question: Dict) -> Dict[str, Any]:
        """
        Generate smart prompt for a gap.
        
        Args:
            gap: Gap dictionary from gap detection
            wafr_question: Full WAFR question schema data
            
        Returns:
            Generated prompt dictionary
        """
        logger.info(f"PromptGeneratorAgent: Generating prompt for question {gap['question_id']}")
        
        # Extract hints from best practices
        hints = [
            bp.get('text', '') 
            for bp in wafr_question.get('best_practices', [])[:3]
        ]
        
        # Get example answer
        example_answer = None
        best_practices = wafr_question.get('best_practices', [])
        if best_practices and len(best_practices) > 0:
            example_answer = best_practices[0].get('example_good_answer')
        
        # Generate quick options based on best practices
        quick_options = [
            bp.get('text', '') 
            for bp in best_practices[:5]
            if bp.get('text')
        ]
        
        # Use agent to generate prompt
        prompt = f"""
        Generate a smart prompt for this WAFR question gap:
        
        Question: {gap['question_text']}
        Pillar: {gap['pillar']}
        Criticality: {gap['criticality']}
        
        Best Practices Hints: {hints}
        Context: {gap.get('context_hint', 'None')}
        
        Use generate_smart_prompt() to create the prompt with all components.
        Make it user-friendly and actionable.
        """
        
        response = self.agent(prompt)
        
        # Parse response
        if isinstance(response, dict) and 'question_id' in response:
            return response
        
        # Fallback to manual construction
        return generate_smart_prompt(
            question_id=gap['question_id'],
            question_text=gap['question_text'],
            pillar=gap['pillar'],
            criticality=gap['criticality'],
            hints=hints,
            example_answer=example_answer,
            context_hint=gap.get('context_hint'),
            quick_options=quick_options
        )


def create_prompt_generator_agent(wafr_schema: Optional[Dict] = None) -> PromptGeneratorAgent:
    """
    Factory function to create Prompt Generator Agent.
    
    Args:
        wafr_schema: Optional WAFR schema for context
    """
    return PromptGeneratorAgent(wafr_schema)

