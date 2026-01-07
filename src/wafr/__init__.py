"""
WAFR - Well-Architected Framework Review System

A multi-agent AI system for automated AWS Well-Architected Framework assessments.
"""

__version__ = "1.0.0"

# Core exports
from wafr.agents.orchestrator import create_orchestrator, WafrOrchestrator

# AG-UI exports (if available)
try:
    from wafr.ag_ui.orchestrator_integration import create_agui_orchestrator
    __all__ = [
        "create_orchestrator",
        "WafrOrchestrator",
        "create_agui_orchestrator",
    ]
except ImportError:
    __all__ = [
        "create_orchestrator",
        "WafrOrchestrator",
    ]

