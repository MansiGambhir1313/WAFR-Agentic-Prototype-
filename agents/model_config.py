"""
Model configuration for Strands agents
"""
import logging
from typing import Optional, Any

from agents.config import DEFAULT_MODEL_ID, BEDROCK_REGION

logger = logging.getLogger(__name__)


def get_strands_model(model_id: Optional[str] = None) -> Any:
    """
    Get configured model for Strands Agent.
    
    Note: Claude 3.7 Sonnet works directly with ConverseStream API.
    If Strands fails, agents will fall back to direct Bedrock invoke_model API.
    
    Args:
        model_id: Optional model ID (defaults to Claude 3.7 Sonnet)
        
    Returns:
        Model instance for Strands Agent, or None to use fallback
    """
    if model_id is None:
        model_id = DEFAULT_MODEL_ID
    
    try:
        # Try to import and create BedrockModel from Strands
        from strands.models.bedrock import BedrockModel
        
        model = BedrockModel(
            model_id=model_id
        )
        return model
    except ImportError:
        # If BedrockModel not available, try alternative import
        try:
            from strands.models import BedrockModel
            model = BedrockModel(
                model_id=model_id
            )
            return model
        except (ImportError, Exception) as e:
            # Fallback: return None and let Strands use default
            # Strands might auto-detect or use environment variables
            # If Strands fails, agents will use direct Bedrock invoke_model API
            logger.warning(f"Could not create BedrockModel explicitly: {e}. Will use direct Bedrock API fallback.")
            return None

