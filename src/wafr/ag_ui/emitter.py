"""
AG-UI Event Emitter for WAFR Pipeline.

Provides WAFREventEmitter class that streams AG-UI compliant events
for real-time frontend updates during WAFR pipeline execution.

Features:
- All 16 standard AG-UI event types
- Custom HITL events for review workflow
- SSE-compatible streaming
- Async event queue
- Heartbeat support

Usage:
    emitter = WAFREventEmitter(thread_id="session-123", run_id="run-456")
    
    # Emit events
    await emitter.run_started()
    await emitter.step_started("understanding")
    await emitter.text_message_content("msg-1", "Processing insights...")
    await emitter.step_finished("understanding", {"count": 15})
    await emitter.run_finished()
    
    # Stream to client
    async for event_data in emitter.stream_events():
        yield event_data  # SSE format
"""

from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
import asyncio
import json
import uuid
import logging

from wafr.ag_ui.events import (
    HITLEventType,
    HITLEvents,
    ReviewQueueSummary,
    SynthesisProgress,
    ReviewDecisionData,
    ValidationStatus,
    WAFRPipelineStep,
)
from wafr.ag_ui.state import WAFRState
from wafr.ag_ui.core import (
    AG_UI_AVAILABLE,
    RunStartedEvent as OfficialRunStartedEvent,
    RunFinishedEvent as OfficialRunFinishedEvent,
    RunErrorEvent as OfficialRunErrorEvent,
    StepStartedEvent as OfficialStepStartedEvent,
    StepFinishedEvent as OfficialStepFinishedEvent,
    TextMessageStartEvent as OfficialTextMessageStartEvent,
    TextMessageContentEvent as OfficialTextMessageContentEvent,
    TextMessageEndEvent as OfficialTextMessageEndEvent,
    ToolCallStartEvent as OfficialToolCallStartEvent,
    ToolCallArgsEvent as OfficialToolCallArgsEvent,
    ToolCallEndEvent as OfficialToolCallEndEvent,
    StateSnapshotEvent as OfficialStateSnapshotEvent,
    StateDeltaEvent as OfficialStateDeltaEvent,
    MessagesSnapshotEvent as OfficialMessagesSnapshotEvent,
    RawEvent as OfficialRawEvent,
    CustomEvent as OfficialCustomEvent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# AG-UI Event Type Enum
# =============================================================================

class EventType:
    """Standard AG-UI event types."""
    
    # Lifecycle events
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    
    # Text message events
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    
    # Tool call events
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    
    # State management events
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"
    
    # Special events
    RAW = "RAW"
    CUSTOM = "CUSTOM"


# =============================================================================
# Base Event Class
# =============================================================================

@dataclass
class BaseEvent:
    """Base class for AG-UI events."""
    
    type: str
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# =============================================================================
# AG-UI Event Classes
# =============================================================================

@dataclass
class RunStartedEvent(BaseEvent):
    """RUN_STARTED event."""
    
    type: str = EventType.RUN_STARTED
    thread_id: str = ""
    run_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "threadId": self.thread_id,
            "runId": self.run_id,
        }


@dataclass
class RunFinishedEvent(BaseEvent):
    """RUN_FINISHED event."""
    
    type: str = EventType.RUN_FINISHED
    thread_id: str = ""
    run_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "threadId": self.thread_id,
            "runId": self.run_id,
        }


@dataclass
class RunErrorEvent(BaseEvent):
    """RUN_ERROR event."""
    
    type: str = EventType.RUN_ERROR
    thread_id: str = ""
    run_id: str = ""
    message: str = ""
    code: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            **super().to_dict(),
            "threadId": self.thread_id,
            "runId": self.run_id,
            "message": self.message,
        }
        if self.code:
            result["code"] = self.code
        return result


@dataclass
class StepStartedEvent(BaseEvent):
    """STEP_STARTED event."""
    
    type: str = EventType.STEP_STARTED
    step_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "stepName": self.step_name,
            "metadata": self.metadata,
        }


@dataclass
class StepFinishedEvent(BaseEvent):
    """STEP_FINISHED event."""
    
    type: str = EventType.STEP_FINISHED
    step_name: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "stepName": self.step_name,
            "result": self.result,
        }


@dataclass
class TextMessageStartEvent(BaseEvent):
    """TEXT_MESSAGE_START event."""
    
    type: str = EventType.TEXT_MESSAGE_START
    message_id: str = ""
    role: str = "assistant"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "messageId": self.message_id,
            "role": self.role,
        }


@dataclass
class TextMessageContentEvent(BaseEvent):
    """TEXT_MESSAGE_CONTENT event."""
    
    type: str = EventType.TEXT_MESSAGE_CONTENT
    message_id: str = ""
    delta: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "messageId": self.message_id,
            "delta": self.delta,
        }


@dataclass
class TextMessageEndEvent(BaseEvent):
    """TEXT_MESSAGE_END event."""
    
    type: str = EventType.TEXT_MESSAGE_END
    message_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "messageId": self.message_id,
        }


@dataclass
class ToolCallStartEvent(BaseEvent):
    """TOOL_CALL_START event."""
    
    type: str = EventType.TOOL_CALL_START
    tool_call_id: str = ""
    tool_name: str = ""
    parent_message_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            **super().to_dict(),
            "toolCallId": self.tool_call_id,
            "toolName": self.tool_name,
        }
        if self.parent_message_id:
            result["parentMessageId"] = self.parent_message_id
        return result


@dataclass
class ToolCallArgsEvent(BaseEvent):
    """TOOL_CALL_ARGS event."""
    
    type: str = EventType.TOOL_CALL_ARGS
    tool_call_id: str = ""
    delta: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "toolCallId": self.tool_call_id,
            "delta": self.delta,
        }


@dataclass
class ToolCallEndEvent(BaseEvent):
    """TOOL_CALL_END event."""
    
    type: str = EventType.TOOL_CALL_END
    tool_call_id: str = ""
    result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result_dict = {
            **super().to_dict(),
            "toolCallId": self.tool_call_id,
        }
        if self.result:
            result_dict["result"] = self.result
        return result_dict


@dataclass
class StateSnapshotEvent(BaseEvent):
    """STATE_SNAPSHOT event."""
    
    type: str = EventType.STATE_SNAPSHOT
    snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "snapshot": self.snapshot,
        }


@dataclass
class StateDeltaEvent(BaseEvent):
    """STATE_DELTA event."""
    
    type: str = EventType.STATE_DELTA
    delta: Union[Dict[str, Any], List[Dict[str, Any]]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "delta": self.delta,
        }


@dataclass
class MessagesSnapshotEvent(BaseEvent):
    """MESSAGES_SNAPSHOT event."""
    
    type: str = EventType.MESSAGES_SNAPSHOT
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "messages": self.messages,
        }


@dataclass
class RawEvent(BaseEvent):
    """RAW event."""
    
    type: str = EventType.RAW
    data: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "data": self.data,
        }


@dataclass
class CustomEvent(BaseEvent):
    """CUSTOM event."""
    
    type: str = EventType.CUSTOM
    name: str = ""
    value: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "name": self.name,
            "value": self.value,
        }


# =============================================================================
# WAFR Event Emitter
# =============================================================================

class WAFREventEmitter:
    """
    AG-UI compliant event emitter for WAFR pipeline.
    
    Streams events to connected clients via SSE or WebSocket.
    Supports all 16 standard AG-UI event types plus custom HITL events.
    
    Thread-safe async event queue with heartbeat support.
    """
    
    def __init__(
        self,
        thread_id: str,
        run_id: Optional[str] = None,
        heartbeat_interval: float = 30.0,
    ):
        """
        Initialize event emitter.
        
        Args:
            thread_id: Thread/session identifier
            run_id: Run identifier (auto-generated if not provided)
            heartbeat_interval: Seconds between heartbeats (default 30)
        """
        self.thread_id = thread_id
        self.run_id = run_id or str(uuid.uuid4())
        self.heartbeat_interval = heartbeat_interval
        
        self.event_queue: asyncio.Queue[BaseEvent] = asyncio.Queue()
        self._started = False
        self._finished = False
        self._error: Optional[str] = None
        
        # State management
        self.state = WAFRState(session_id=thread_id)
        
        # Event listeners
        self._listeners: List[Callable[[BaseEvent], None]] = []
        
        logger.info(f"WAFREventEmitter initialized: thread={thread_id}, run={self.run_id}")
    
    # =========================================================================
    # Event Emission
    # =========================================================================
    
    async def emit(self, event: BaseEvent) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event: Event to emit
        """
        event.timestamp = datetime.utcnow().timestamp()
        await self.event_queue.put(event)
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")
        
        logger.debug(f"Event emitted: {event.type}")
    
    def add_listener(self, listener: Callable[[BaseEvent], None]) -> None:
        """Add event listener."""
        self._listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[BaseEvent], None]) -> None:
        """Remove event listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    # =========================================================================
    # Lifecycle Events
    # =========================================================================
    
    async def run_started(self) -> None:
        """Emit RUN_STARTED event."""
        self._started = True
        await self.emit(RunStartedEvent(
            thread_id=self.thread_id,
            run_id=self.run_id,
        ))
        logger.info(f"Run started: {self.run_id}")
    
    async def run_finished(self) -> None:
        """Emit RUN_FINISHED event."""
        self._finished = True
        await self.emit(RunFinishedEvent(
            thread_id=self.thread_id,
            run_id=self.run_id,
        ))
        logger.info(f"Run finished: {self.run_id}")
    
    async def run_error(self, error: str, code: Optional[str] = None) -> None:
        """Emit RUN_ERROR event."""
        self._finished = True
        self._error = error
        await self.emit(RunErrorEvent(
            thread_id=self.thread_id,
            run_id=self.run_id,
            message=error,
            code=code,
        ))
        logger.error(f"Run error: {error}")
    
    async def step_started(
        self,
        step_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit STEP_STARTED event."""
        await self.emit(StepStartedEvent(
            step_name=step_name,
            metadata=metadata or {},
        ))
        
        # Update state
        deltas = self.state.update_step(step_name)
        await self.state_delta(deltas)
        
        logger.info(f"Step started: {step_name}")
    
    async def step_finished(
        self,
        step_name: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit STEP_FINISHED event."""
        await self.emit(StepFinishedEvent(
            step_name=step_name,
            result=result or {},
        ))
        
        # Update state
        delta = self.state.complete_step(step_name)
        await self.state_delta([delta])
        
        logger.info(f"Step finished: {step_name}")
    
    # =========================================================================
    # Text Message Events
    # =========================================================================
    
    async def text_message_start(
        self,
        message_id: str,
        role: str = "assistant",
    ) -> None:
        """Emit TEXT_MESSAGE_START event."""
        await self.emit(TextMessageStartEvent(
            message_id=message_id,
            role=role,
        ))
    
    async def text_message_content(
        self,
        message_id: str,
        delta: str,
    ) -> None:
        """Emit TEXT_MESSAGE_CONTENT event."""
        await self.emit(TextMessageContentEvent(
            message_id=message_id,
            delta=delta,
        ))
    
    async def text_message_end(self, message_id: str) -> None:
        """Emit TEXT_MESSAGE_END event."""
        await self.emit(TextMessageEndEvent(
            message_id=message_id,
        ))
    
    async def text_message_stream(
        self,
        message_id: str,
        content_generator: AsyncIterator[str],
        role: str = "assistant",
    ) -> str:
        """
        Stream text message content.
        
        Args:
            message_id: Message identifier
            content_generator: Async generator yielding content chunks
            role: Message role (default: "assistant")
        
        Returns:
            Complete message content
        """
        await self.text_message_start(message_id, role)
        
        full_content = ""
        async for chunk in content_generator:
            full_content += chunk
            await self.text_message_content(message_id, chunk)
        
        await self.text_message_end(message_id)
        
        return full_content
    
    # =========================================================================
    # Tool Call Events
    # =========================================================================
    
    async def tool_call_start(
        self,
        tool_call_id: str,
        tool_name: str,
        parent_message_id: Optional[str] = None,
    ) -> None:
        """Emit TOOL_CALL_START event."""
        await self.emit(ToolCallStartEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            parent_message_id=parent_message_id,
        ))
    
    async def tool_call_args(
        self,
        tool_call_id: str,
        delta: str,
    ) -> None:
        """Emit TOOL_CALL_ARGS event."""
        await self.emit(ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            delta=delta,
        ))
    
    async def tool_call_end(
        self,
        tool_call_id: str,
        result: Optional[str] = None,
    ) -> None:
        """Emit TOOL_CALL_END event."""
        await self.emit(ToolCallEndEvent(
            tool_call_id=tool_call_id,
            result=result,
        ))
    
    # =========================================================================
    # State Management Events
    # =========================================================================
    
    async def state_snapshot(self, snapshot: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit STATE_SNAPSHOT event.
        
        Args:
            snapshot: State snapshot (uses current state if not provided)
        """
        if snapshot is None:
            snapshot = self.state.to_snapshot()
        
        await self.emit(StateSnapshotEvent(snapshot=snapshot))
    
    async def state_delta(
        self,
        delta: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> None:
        """
        Emit STATE_DELTA event.
        
        Args:
            delta: JSON Patch operations
        """
        await self.emit(StateDeltaEvent(delta=delta))
    
    async def messages_snapshot(self, messages: List[Dict[str, Any]]) -> None:
        """Emit MESSAGES_SNAPSHOT event."""
        await self.emit(MessagesSnapshotEvent(messages=messages))
    
    # =========================================================================
    # Special Events
    # =========================================================================
    
    async def raw_event(self, data: Any) -> None:
        """Emit RAW event."""
        await self.emit(RawEvent(data=data))
    
    async def custom_event(self, name: str, value: Dict[str, Any]) -> None:
        """Emit CUSTOM event."""
        await self.emit(CustomEvent(name=name, value=value))
    
    # =========================================================================
    # HITL-Specific Events
    # =========================================================================
    
    async def review_required(
        self,
        session_id: str,
        queue_summary: ReviewQueueSummary,
    ) -> None:
        """
        Signal that human review is required.
        
        Emits STATE_SNAPSHOT with current state and CUSTOM review_required event.
        """
        # Update state
        self.state.update_review_state(
            session_id=session_id,
            pending=queue_summary.pending_count,
            approved=queue_summary.approved_count,
            modified=queue_summary.modified_count,
            rejected=queue_summary.rejected_count,
        )
        self.state.update_review_confidence_counts(
            high=queue_summary.high_confidence_count,
            medium=queue_summary.medium_confidence_count,
            low=queue_summary.low_confidence_count,
        )
        
        # Emit snapshot
        await self.state_snapshot()
        
        # Emit custom event
        payload = HITLEvents.review_required(session_id, queue_summary)
        await self.custom_event(HITLEventType.REVIEW_REQUIRED.value, payload)
        
        logger.info(f"Review required: {queue_summary.pending_count} items pending")
    
    async def synthesis_progress(self, progress: SynthesisProgress) -> None:
        """Report synthesis progress."""
        payload = HITLEvents.synthesis_progress(progress)
        await self.custom_event(HITLEventType.SYNTHESIS_PROGRESS.value, payload)
        
        # Update state
        delta = self.state.set_synthesized_count(progress.current)
        await self.state_delta([delta])
    
    async def review_decision(self, decision_data: ReviewDecisionData) -> None:
        """Report human review decision."""
        payload = HITLEvents.review_decision(decision_data)
        await self.custom_event(HITLEventType.REVIEW_DECISION.value, payload)
        
        logger.info(f"Review decision: {decision_data.decision} for {decision_data.question_id}")
    
    async def batch_approve_completed(
        self,
        session_id: str,
        approved_count: int,
        remaining_count: int,
    ) -> None:
        """Report batch approval completion."""
        payload = HITLEvents.batch_approve_completed(
            session_id, approved_count, remaining_count
        )
        await self.custom_event(HITLEventType.BATCH_APPROVE_COMPLETED.value, payload)
        
        # Update state
        deltas = self.state.update_review_state(
            session_id=session_id,
            pending=remaining_count,
            approved=self.state.review.approved_count + approved_count,
        )
        await self.state_delta(deltas)
    
    async def validation_status(self, status: ValidationStatus) -> None:
        """Report validation status."""
        payload = HITLEvents.validation_status(status)
        event_type = (
            HITLEventType.VALIDATION_PASSED.value 
            if status.can_finalize 
            else HITLEventType.VALIDATION_FAILED.value
        )
        await self.custom_event(event_type, payload)
        
        # Update state
        delta = self.state.set_authenticity_score(status.authenticity_score)
        await self.state_delta([delta])
    
    async def session_finalized(
        self,
        session_id: str,
        authenticity_score: float,
        total_items: int,
        approved: int,
        modified: int,
    ) -> None:
        """Report session finalization."""
        from wafr.ag_ui.state import SessionStatus
        
        payload = HITLEvents.session_finalized(
            session_id, authenticity_score, total_items, approved, modified
        )
        await self.custom_event(HITLEventType.SESSION_FINALIZED.value, payload)
        
        # Update state
        delta = self.state.update_status(SessionStatus.FINALIZED)
        await self.state_delta([delta])
        
        logger.info(f"Session finalized: {session_id}")
    
    async def authenticity_score_update(
        self,
        session_id: str,
        score: float,
        breakdown: Dict[str, Any],
    ) -> None:
        """Report authenticity score update."""
        payload = HITLEvents.authenticity_score_update(session_id, score, breakdown)
        await self.custom_event(HITLEventType.AUTHENTICITY_SCORE_UPDATE.value, payload)
        
        # Update state
        delta = self.state.set_authenticity_score(score)
        await self.state_delta([delta])
    
    # =========================================================================
    # Streaming
    # =========================================================================
    
    async def stream_events(self) -> AsyncIterator[str]:
        """
        Yield events as SSE-formatted strings.
        
        Format: "data: {json}\n\n"
        
        Yields:
            SSE-formatted event strings
        """
        while not self._finished or not self.event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=self.heartbeat_interval,
                )
                yield f"data: {json.dumps(event.to_dict())}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat
                yield ": heartbeat\n\n"
            except Exception as e:
                logger.error(f"Error streaming event: {e}")
                yield f"data: {json.dumps({'type': 'ERROR', 'message': str(e)})}\n\n"
                break
    
    def stream_events_sync(self) -> AsyncIterator[str]:
        """
        Synchronous wrapper for stream_events.
        
        For use in non-async contexts.
        """
        loop = asyncio.get_event_loop()
        while not self._finished or not self.event_queue.empty():
            try:
                event = loop.run_until_complete(
                    asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=self.heartbeat_interval,
                    )
                )
                yield f"data: {json.dumps(event.to_dict())}\n\n"
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def is_started(self) -> bool:
        """Check if run has started."""
        return self._started
    
    @property
    def is_finished(self) -> bool:
        """Check if run has finished."""
        return self._finished
    
    @property
    def has_error(self) -> bool:
        """Check if run has error."""
        return self._error is not None
    
    @property
    def error_message(self) -> Optional[str]:
        """Get error message if any."""
        return self._error

