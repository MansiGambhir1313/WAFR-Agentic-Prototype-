"""
AG-UI Integration for WAFR Orchestrator

This module provides AG-UI event emission integration for the WAFR orchestrator,
adding real-time event streaming throughout the pipeline execution.

Usage:
    from ag_ui.orchestrator_integration import create_agui_orchestrator
    
    orchestrator = create_agui_orchestrator()
    emitter = orchestrator.emitter
    
    # Process with AG-UI events
    results = await orchestrator.process_transcript_with_agui(
        transcript=transcript,
        session_id=session_id,
    )
"""

from typing import Any, Dict, List, Optional, Callable
import asyncio
import logging
import uuid

from wafr.ag_ui.emitter import WAFREventEmitter
from wafr.ag_ui.core import (
    WAFRTool,
    get_wafr_tool,
    WAFRMessage,
)
from wafr.ag_ui.events import (
    WAFRPipelineStep,
    SynthesisProgress,
)

logger = logging.getLogger(__name__)


class AGUIOrchestratorWrapper:
    """
    Wrapper for WafrOrchestrator that adds AG-UI event emission.
    
    This wrapper enhances the orchestrator with AG-UI events while
    maintaining backward compatibility with the existing orchestrator API.
    """
    
    def __init__(
        self,
        orchestrator,
        emitter: Optional[WAFREventEmitter] = None,
        thread_id: Optional[str] = None,
    ):
        """
        Initialize AG-UI orchestrator wrapper.
        
        Args:
            orchestrator: Base WafrOrchestrator instance
            emitter: Optional WAFREventEmitter (created if not provided)
            thread_id: Thread/session ID for emitter
        """
        self.orchestrator = orchestrator
        self.thread_id = thread_id or str(uuid.uuid4())
        
        if emitter is None:
            self.emitter = WAFREventEmitter(thread_id=self.thread_id)
        else:
            self.emitter = emitter
        
        logger.info(f"AG-UI orchestrator wrapper initialized: thread={self.thread_id}")
    
    async def process_transcript_with_agui(
        self,
        transcript: str,
        session_id: str,
        generate_report: bool = True,
        create_wa_workload: bool = False,
        client_name: Optional[str] = None,
        environment: str = "PRODUCTION",
        existing_workload_id: Optional[str] = None,
        pdf_files: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, str, Optional[Dict]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process transcript with full AG-UI event streaming.
        
        This is an async wrapper around the sync orchestrator.process_transcript,
        adding AG-UI events throughout the pipeline.
        """
        # Start run
        await self.emitter.run_started()
        await self.emitter.state_snapshot()
        
        try:
            # Create enhanced progress callback that also emits AG-UI events
            async def agui_progress_callback(step: str, message: str, data: Optional[Dict] = None):
                """Progress callback that emits AG-UI events."""
                if progress_callback:
                    progress_callback(step, message, data)
                
                # Emit step events
                step_name = step.lower().replace(" ", "_")
                await self.emitter.step_started(step_name, metadata=data or {})
                
                # Emit text message for progress
                msg_id = f"msg-{step_name}-{uuid.uuid4().hex[:8]}"
                await self.emitter.text_message_start(msg_id, role="assistant")
                await self.emitter.text_message_content(msg_id, f"[{step}] {message}")
                await self.emitter.text_message_end(msg_id)
            
            # Run orchestrator in executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.orchestrator.process_transcript(
                    transcript=transcript,
                    session_id=session_id,
                    generate_report=generate_report,
                    create_wa_workload=create_wa_workload,
                    client_name=client_name,
                    environment=environment,
                    existing_workload_id=existing_workload_id,
                    pdf_files=pdf_files,
                    # Orchestrator sometimes calls progress_callback(step, message)
                    # and sometimes progress_callback(step, message, data). Accept both.
                    progress_callback=lambda s, m, d=None: asyncio.run_coroutine_threadsafe(
                        agui_progress_callback(s, m, d), loop
                    ).result(),
                )
            )
            
            # Emit final state and completion
            await self.emitter.state_snapshot()
            await self.emitter.run_finished()
            
            return results
            
        except Exception as e:
            logger.error(f"AG-UI orchestrator error: {e}", exc_info=True)
            await self.emitter.run_error(str(e), code="ORCHESTRATOR_ERROR")
            raise
    
    async def _emit_agent_tool_call(
        self,
        agent_type: str,
        tool_name: str,
        args: Dict[str, Any],
        result: Any = None,
    ):
        """
        Emit tool call events for an agent operation.
        
        Args:
            agent_type: Type of agent (understanding, mapping, etc.)
            tool_name: Name of the tool/agent
            args: Tool arguments
            result: Tool result (if available)
        """
        tool_call_id = f"tool-{agent_type}-{uuid.uuid4().hex[:8]}"
        
        # Get tool definition
        tool = get_wafr_tool(agent_type)
        if tool:
            tool_name = tool.name
        
        # Emit tool call start
        await self.emitter.tool_call_start(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
        
        # Emit tool call args (streamed)
        args_json = str(args)[:500]  # Truncate for display
        await self.emitter.tool_call_args(tool_call_id, args_json)
        
        # Emit tool call end with result
        result_str = str(result)[:1000] if result else None
        await self.emitter.tool_call_end(tool_call_id, result_str)
    
    async def _emit_step_with_tool_calls(
        self,
        step_name: str,
        agent_type: str,
        operation: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a step with AG-UI tool call events.
        
        Args:
            step_name: Name of the step
            agent_type: Type of agent being used
            operation: Function to execute
            *args, **kwargs: Arguments for operation
        
        Returns:
            Result from operation
        """
        # Emit step started
        await self.emitter.step_started(step_name)
        
        # Emit tool call start
        await self._emit_agent_tool_call(
            agent_type=agent_type,
            tool_name=step_name,
            args={"step": step_name, "args_count": len(args)},
        )
        
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: operation(*args, **kwargs))
            
            # Emit step finished
            await self.emitter.step_finished(
                step_name,
                result={"status": "success", "has_result": result is not None},
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Step {step_name} error: {e}", exc_info=True)
            await self.emitter.step_finished(
                step_name,
                result={"status": "error", "error": str(e)},
            )
            raise
    
    async def _emit_synthesis_progress(
        self,
        current: int,
        total: int,
        question_id: str = "",
        pillar: str = "",
    ):
        """Emit synthesis progress event."""
        progress = SynthesisProgress(
            current=current,
            total=total,
            question_id=question_id,
            pillar=pillar,
        )
        await self.emitter.synthesis_progress(progress)
    
    # Delegate other methods to base orchestrator
    def __getattr__(self, name):
        """Delegate attribute access to base orchestrator."""
        return getattr(self.orchestrator, name)


def create_agui_orchestrator(
    orchestrator=None,
    emitter: Optional[WAFREventEmitter] = None,
    thread_id: Optional[str] = None,
) -> AGUIOrchestratorWrapper:
    """
    Create AG-UI enabled orchestrator.
    
    Args:
        orchestrator: Base orchestrator (created if not provided)
        emitter: Optional event emitter
        thread_id: Optional thread ID
    
    Returns:
        AGUIOrchestratorWrapper instance
    """
    if orchestrator is None:
        from agents.orchestrator import create_orchestrator
        orchestrator = create_orchestrator()
    
    return AGUIOrchestratorWrapper(
        orchestrator=orchestrator,
        emitter=emitter,
        thread_id=thread_id,
    )


__all__ = [
    "AGUIOrchestratorWrapper",
    "create_agui_orchestrator",
]

