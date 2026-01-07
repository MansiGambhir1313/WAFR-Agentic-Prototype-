# AG-UI Implementation Plan for WAFR Project

## Overview

This document outlines the implementation plan for integrating the [Agent User Interaction Protocol (AG-UI)](https://docs.ag-ui.com) into the WAFR HITL pipeline. AG-UI is a lightweight, event-driven protocol that enables seamless communication between AI agents and frontend applications.

---

## 1. AG-UI Protocol Summary

### 1.1 Core Concepts

Based on the [AG-UI Core Architecture](https://docs.ag-ui.com/concepts/architecture):

| Concept | Description |
|---------|-------------|
| **Event-Driven** | 16 standardized event types for real-time streaming |
| **Bidirectional** | Agents accept input from users for collaborative workflows |
| **Transport Agnostic** | Supports SSE, WebSockets, webhooks |
| **Flexible** | Events don't need exact format match - just AG-UI compatible |

### 1.2 Event Categories

From the [Events documentation](https://docs.ag-ui.com/concepts/events):

```
┌─────────────────────────────────────────────────────────────────┐
│                    AG-UI EVENT TYPES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LIFECYCLE EVENTS:                                              │
│  ├── RUN_STARTED      - Agent run begins                       │
│  ├── RUN_FINISHED     - Agent run completes successfully       │
│  ├── RUN_ERROR        - Agent run fails                        │
│  ├── STEP_STARTED     - Processing step begins                 │
│  └── STEP_FINISHED    - Processing step completes              │
│                                                                 │
│  TEXT MESSAGE EVENTS:                                           │
│  ├── TEXT_MESSAGE_START   - Message stream begins              │
│  ├── TEXT_MESSAGE_CONTENT - Message chunk received             │
│  └── TEXT_MESSAGE_END     - Message stream completes           │
│                                                                 │
│  TOOL CALL EVENTS:                                              │
│  ├── TOOL_CALL_START  - Tool execution begins                  │
│  ├── TOOL_CALL_ARGS   - Tool arguments streamed                │
│  └── TOOL_CALL_END    - Tool execution completes               │
│                                                                 │
│  STATE MANAGEMENT EVENTS:                                       │
│  ├── STATE_SNAPSHOT   - Complete state at point in time        │
│  ├── STATE_DELTA      - Incremental changes (JSON Patch)       │
│  └── MESSAGES_SNAPSHOT - Complete conversation history         │
│                                                                 │
│  SPECIAL EVENTS:                                                │
│  ├── RAW              - Raw/unprocessed data                   │
│  └── CUSTOM           - Custom application events              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Python SDK

Install: `pip install ag-ui-protocol`

```python
from ag_ui.core import (
    RunAgentInput,
    Message,
    Context,
    Tool,
    State,
    # Events
    RunStartedEvent,
    RunFinishedEvent,
    StepStartedEvent,
    StepFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    StateSnapshotEvent,
    StateDeltaEvent,
)
```

---

## 2. WAFR-AG-UI Mapping

### 2.1 Pipeline Steps to AG-UI Events

| WAFR Pipeline Step | AG-UI Events |
|--------------------|--------------|
| Session Start | `RUN_STARTED` |
| PDF Processing | `STEP_STARTED("pdf_processing")` → `STEP_FINISHED` |
| Understanding Agent | `STEP_STARTED("understanding")` → `TEXT_MESSAGE_*` → `STEP_FINISHED` |
| Mapping Agent | `STEP_STARTED("mapping")` → `STATE_DELTA` → `STEP_FINISHED` |
| Confidence Agent | `STEP_STARTED("confidence")` → `STATE_DELTA` → `STEP_FINISHED` |
| Gap Detection | `STEP_STARTED("gap_detection")` → `STATE_DELTA` → `STEP_FINISHED` |
| Answer Synthesis | `STEP_STARTED("synthesis")` → `TEXT_MESSAGE_*` → `STEP_FINISHED` |
| **HITL Checkpoint** | `STATE_SNAPSHOT` (review queue) + `CUSTOM` (review_required) |
| Review Decision | `CUSTOM` (review_decision) → `STATE_DELTA` |
| Scoring Agent | `STEP_STARTED("scoring")` → `STATE_DELTA` → `STEP_FINISHED` |
| Report Generation | `STEP_STARTED("report")` → `STEP_FINISHED` |
| Session Complete | `RUN_FINISHED` |

### 2.2 HITL-Specific Events

```python
# Custom HITL events
HITL_EVENTS = {
    "REVIEW_REQUIRED": "hitl.review_required",      # Pause for human review
    "REVIEW_QUEUE_UPDATE": "hitl.queue_update",     # Queue changed
    "REVIEW_DECISION": "hitl.decision",             # Human decision received
    "BATCH_APPROVE": "hitl.batch_approve",          # Batch approval
    "SYNTHESIS_PROGRESS": "hitl.synthesis_progress", # Synthesis progress
    "VALIDATION_STATUS": "hitl.validation_status",   # Finalization check
}
```

---

## 3. Implementation Architecture

### 3.1 New Components

```
wafr-prototype/
├── ag_ui/
│   ├── __init__.py
│   ├── events.py          # AG-UI event definitions
│   ├── emitter.py         # Event emitter for streaming
│   ├── state.py           # State management
│   ├── middleware.py      # AG-UI middleware adapter
│   └── server.py          # SSE server endpoint
├── agents/
│   ├── orchestrator.py    # ENHANCED with AG-UI events
│   └── ...
└── ...
```

### 3.2 Event Emitter Class

```python
# ag_ui/emitter.py

from typing import AsyncIterator, Optional, Dict, Any
from datetime import datetime
import asyncio
import json

from ag_ui.core import (
    BaseEvent,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    StepStartedEvent,
    StepFinishedEvent,
    TextMessageStartEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    StateSnapshotEvent,
    StateDeltaEvent,
    CustomEvent,
)


class WAFREventEmitter:
    """
    AG-UI compliant event emitter for WAFR pipeline.
    
    Streams events to connected clients via SSE or WebSocket.
    """
    
    def __init__(self, thread_id: str, run_id: str):
        self.thread_id = thread_id
        self.run_id = run_id
        self.event_queue: asyncio.Queue[BaseEvent] = asyncio.Queue()
        self._started = False
        self._finished = False
    
    async def emit(self, event: BaseEvent) -> None:
        """Emit an event to all subscribers."""
        event.timestamp = datetime.utcnow().timestamp()
        await self.event_queue.put(event)
    
    async def run_started(self) -> None:
        """Emit RUN_STARTED event."""
        self._started = True
        await self.emit(RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=self.thread_id,
            run_id=self.run_id,
        ))
    
    async def run_finished(self) -> None:
        """Emit RUN_FINISHED event."""
        self._finished = True
        await self.emit(RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=self.thread_id,
            run_id=self.run_id,
        ))
    
    async def run_error(self, error: str, code: Optional[str] = None) -> None:
        """Emit RUN_ERROR event."""
        self._finished = True
        await self.emit(RunErrorEvent(
            type=EventType.RUN_ERROR,
            thread_id=self.thread_id,
            run_id=self.run_id,
            message=error,
            code=code,
        ))
    
    async def step_started(self, step_name: str, metadata: Optional[Dict] = None) -> None:
        """Emit STEP_STARTED event."""
        await self.emit(StepStartedEvent(
            type=EventType.STEP_STARTED,
            step_name=step_name,
            metadata=metadata or {},
        ))
    
    async def step_finished(self, step_name: str, result: Optional[Dict] = None) -> None:
        """Emit STEP_FINISHED event."""
        await self.emit(StepFinishedEvent(
            type=EventType.STEP_FINISHED,
            step_name=step_name,
            result=result or {},
        ))
    
    async def text_message_stream(
        self,
        message_id: str,
        content_generator: AsyncIterator[str],
        role: str = "assistant",
    ) -> None:
        """Stream text message content."""
        await self.emit(TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=message_id,
            role=role,
        ))
        
        full_content = ""
        async for chunk in content_generator:
            full_content += chunk
            await self.emit(TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=message_id,
                delta=chunk,
            ))
        
        await self.emit(TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=message_id,
        ))
    
    async def state_snapshot(self, state: Dict[str, Any]) -> None:
        """Emit complete state snapshot."""
        await self.emit(StateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT,
            snapshot=state,
        ))
    
    async def state_delta(self, delta: Dict[str, Any]) -> None:
        """Emit state delta (JSON Patch format)."""
        await self.emit(StateDeltaEvent(
            type=EventType.STATE_DELTA,
            delta=delta,
        ))
    
    async def custom_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit custom HITL event."""
        await self.emit(CustomEvent(
            type=EventType.CUSTOM,
            name=event_type,
            value=data,
        ))
    
    # HITL-specific convenience methods
    
    async def review_required(self, session_id: str, queue_summary: Dict) -> None:
        """Signal that human review is required."""
        await self.custom_event("hitl.review_required", {
            "session_id": session_id,
            "queue_summary": queue_summary,
        })
    
    async def synthesis_progress(self, current: int, total: int, question_id: str) -> None:
        """Report synthesis progress."""
        await self.custom_event("hitl.synthesis_progress", {
            "current": current,
            "total": total,
            "question_id": question_id,
            "percentage": round(current / total * 100, 1),
        })
    
    async def review_decision(self, review_id: str, decision: str, question_id: str) -> None:
        """Report human review decision."""
        await self.custom_event("hitl.decision", {
            "review_id": review_id,
            "decision": decision,
            "question_id": question_id,
        })
    
    async def stream_events(self) -> AsyncIterator[str]:
        """Yield events as SSE-formatted strings."""
        while not self._finished or not self.event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=30.0  # Heartbeat every 30s
                )
                yield f"data: {event.model_dump_json()}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat
                yield f": heartbeat\n\n"
```

### 3.3 State Management

```python
# ag_ui/state.py

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class WAFRState:
    """
    AG-UI compatible state for WAFR pipeline.
    
    Tracks the complete state of a WAFR assessment session,
    enabling efficient state sync with frontends.
    """
    
    # Session info
    session_id: str = ""
    status: str = "initialized"  # initialized, processing, review, finalized
    
    # Pipeline progress
    current_step: str = ""
    completed_steps: List[str] = field(default_factory=list)
    
    # Content
    transcript_loaded: bool = False
    insights_count: int = 0
    questions_answered: int = 0
    questions_total: int = 0
    gaps_count: int = 0
    
    # HITL Review
    review_session_id: Optional[str] = None
    review_queue: Dict[str, Any] = field(default_factory=dict)
    pending_reviews: int = 0
    approved_reviews: int = 0
    
    # Scores
    authenticity_score: float = 0.0
    pillar_coverage: Dict[str, float] = field(default_factory=dict)
    
    def to_snapshot(self) -> Dict[str, Any]:
        """Convert to STATE_SNAPSHOT format."""
        return {
            "session": {
                "id": self.session_id,
                "status": self.status,
            },
            "pipeline": {
                "current_step": self.current_step,
                "completed_steps": self.completed_steps,
            },
            "content": {
                "transcript_loaded": self.transcript_loaded,
                "insights_count": self.insights_count,
                "questions_answered": self.questions_answered,
                "questions_total": self.questions_total,
                "gaps_count": self.gaps_count,
            },
            "review": {
                "session_id": self.review_session_id,
                "pending": self.pending_reviews,
                "approved": self.approved_reviews,
                "queue_summary": self.review_queue,
            },
            "scores": {
                "authenticity": self.authenticity_score,
                "pillar_coverage": self.pillar_coverage,
            },
        }
    
    def create_delta(self, path: str, value: Any, op: str = "replace") -> Dict:
        """Create JSON Patch delta for STATE_DELTA event."""
        return {
            "op": op,
            "path": path,
            "value": value,
        }
```

### 3.4 Enhanced Orchestrator

```python
# Modifications to agents/orchestrator.py

class WafrOrchestrator:
    """Enhanced orchestrator with AG-UI event emission."""
    
    def __init__(self, ..., emitter: Optional[WAFREventEmitter] = None):
        self.emitter = emitter
        # ... existing init
    
    async def process_transcript_with_events(
        self,
        transcript_path: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """Process transcript with AG-UI event streaming."""
        
        if self.emitter:
            await self.emitter.run_started()
        
        state = WAFRState(session_id=session_id)
        
        try:
            # Step 1: PDF Processing
            if self.emitter:
                await self.emitter.step_started("pdf_processing")
                await self.emitter.state_delta(state.create_delta(
                    "/pipeline/current_step", "pdf_processing"
                ))
            
            transcript = self._step_process_pdf(transcript_path)
            state.transcript_loaded = True
            
            if self.emitter:
                await self.emitter.step_finished("pdf_processing", {
                    "transcript_length": len(transcript)
                })
            
            # Step 2: Understanding
            if self.emitter:
                await self.emitter.step_started("understanding")
            
            insights = self._step_understanding(transcript)
            state.insights_count = len(insights)
            
            if self.emitter:
                await self.emitter.step_finished("understanding", {
                    "insights_count": len(insights)
                })
                await self.emitter.state_delta(state.create_delta(
                    "/content/insights_count", len(insights)
                ))
            
            # ... continue for each step
            
            # Step 6: Answer Synthesis with progress
            if self.emitter:
                await self.emitter.step_started("synthesis")
            
            for i, gap in enumerate(gaps):
                if self.emitter:
                    await self.emitter.synthesis_progress(i + 1, len(gaps), gap["question_id"])
                
                synthesized = await self._synthesize_single(gap, transcript, insights)
                # ...
            
            if self.emitter:
                await self.emitter.step_finished("synthesis", {
                    "synthesized_count": len(synthesized_answers)
                })
            
            # HITL Checkpoint
            if self.emitter:
                review_session = self.review_orchestrator.create_review_session(synthesized_answers)
                queue = self.review_orchestrator.get_batch_review_queue(review_session.session_id)
                
                await self.emitter.state_snapshot(state.to_snapshot())
                await self.emitter.review_required(
                    session_id=review_session.session_id,
                    queue_summary={
                        "total": len(synthesized_answers),
                        "high_confidence": len(queue["high_confidence"]),
                        "medium_confidence": len(queue["medium_confidence"]),
                        "low_confidence": len(queue["low_confidence"]),
                    }
                )
            
            # ... continue after review
            
            if self.emitter:
                await self.emitter.run_finished()
            
            return results
            
        except Exception as e:
            if self.emitter:
                await self.emitter.run_error(str(e))
            raise
```

---

## 4. SSE Server Endpoint

### 4.1 FastAPI SSE Endpoint

```python
# ag_ui/server.py

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio

from ag_ui.emitter import WAFREventEmitter
from agents.orchestrator import WafrOrchestrator


app = FastAPI()


@app.post("/api/wafr/run")
async def run_wafr_agent(request: Request):
    """
    AG-UI compatible endpoint for running WAFR assessment.
    
    Returns SSE stream of events.
    """
    body = await request.json()
    
    thread_id = body.get("threadId", str(uuid.uuid4()))
    run_id = body.get("runId", str(uuid.uuid4()))
    transcript_path = body.get("transcriptPath")
    
    # Create event emitter
    emitter = WAFREventEmitter(thread_id=thread_id, run_id=run_id)
    
    # Create orchestrator with emitter
    orchestrator = WafrOrchestrator(emitter=emitter)
    
    async def event_generator():
        # Start processing in background
        task = asyncio.create_task(
            orchestrator.process_transcript_with_events(
                transcript_path=transcript_path,
                session_id=thread_id,
            )
        )
        
        # Stream events
        async for event_data in emitter.stream_events():
            yield event_data
        
        # Wait for processing to complete
        await task
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/wafr/review/{session_id}/decision")
async def submit_review_decision(session_id: str, request: Request):
    """Submit review decision during HITL checkpoint."""
    body = await request.json()
    
    # Process decision
    result = await review_orchestrator.submit_review(
        session_id=session_id,
        review_id=body["reviewId"],
        decision=body["decision"],
        reviewer_id=body["reviewerId"],
        modified_answer=body.get("modifiedAnswer"),
        feedback=body.get("feedback"),
    )
    
    return {"status": "success", "result": result.to_dict()}
```

---

## 5. Implementation Phases

### Phase 1: Core AG-UI Infrastructure (Week 1)

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Core Infrastructure                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tasks:                                                         │
│  [ ] Install ag-ui-protocol package                            │
│  [ ] Create ag_ui/ module structure                            │
│  [ ] Implement WAFREventEmitter class                          │
│  [ ] Implement WAFRState class                                 │
│  [ ] Create custom HITL event types                            │
│  [ ] Write unit tests for event emission                       │
│                                                                 │
│  Deliverables:                                                  │
│  • Working event emitter with all AG-UI event types            │
│  • State management with snapshot/delta support                │
│  • Custom HITL events for review workflow                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Orchestrator Integration (Week 2)

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Orchestrator Integration                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tasks:                                                         │
│  [ ] Add emitter parameter to WafrOrchestrator                 │
│  [ ] Add event emission to each pipeline step                  │
│  [ ] Implement async processing with events                    │
│  [ ] Add synthesis progress streaming                          │
│  [ ] Implement HITL checkpoint events                          │
│  [ ] Wire review decisions to state updates                    │
│                                                                 │
│  Deliverables:                                                  │
│  • Full pipeline with event emission                           │
│  • Real-time progress tracking                                 │
│  • HITL pause/resume via events                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: SSE Server (Week 3)

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: SSE Server Endpoint                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tasks:                                                         │
│  [ ] Create FastAPI SSE endpoint                               │
│  [ ] Implement streaming response                              │
│  [ ] Add heartbeat for connection keep-alive                   │
│  [ ] Create review decision endpoints                          │
│  [ ] Add batch approval endpoint                               │
│  [ ] Implement finalization endpoint                           │
│                                                                 │
│  Deliverables:                                                  │
│  • Working SSE endpoint for event streaming                    │
│  • REST endpoints for HITL interactions                        │
│  • Connection management                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 4: Testing & Documentation (Week 4)

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: Testing & Documentation                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tasks:                                                         │
│  [ ] Integration tests for event streaming                     │
│  [ ] E2E tests with mock frontend                              │
│  [ ] Performance testing (event latency)                       │
│  [ ] API documentation (OpenAPI)                               │
│  [ ] AG-UI compatibility validation                            │
│  [ ] Example client implementation                             │
│                                                                 │
│  Deliverables:                                                  │
│  • Comprehensive test coverage                                 │
│  • API documentation                                           │
│  • Sample client code                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Event Flow Diagrams

### 6.1 Complete Pipeline Flow

```
Client                    Server                    Agent Pipeline
  │                         │                            │
  │──POST /api/wafr/run────>│                            │
  │                         │                            │
  │<──SSE: RUN_STARTED──────│                            │
  │                         │──Start Processing──────────>│
  │                         │                            │
  │<──SSE: STEP_STARTED─────│<─────pdf_processing────────│
  │    (pdf_processing)     │                            │
  │<──SSE: STEP_FINISHED────│                            │
  │                         │                            │
  │<──SSE: STEP_STARTED─────│<─────understanding─────────│
  │<──SSE: TEXT_MESSAGE_*───│<─────insights streaming────│
  │<──SSE: STATE_DELTA──────│                            │
  │<──SSE: STEP_FINISHED────│                            │
  │                         │                            │
  │    ... more steps ...   │                            │
  │                         │                            │
  │<──SSE: STEP_STARTED─────│<─────synthesis─────────────│
  │<──SSE: CUSTOM───────────│<─────synthesis_progress────│
  │    (synthesis_progress) │                            │
  │<──SSE: STATE_DELTA──────│                            │
  │<──SSE: STEP_FINISHED────│                            │
  │                         │                            │
  │<──SSE: STATE_SNAPSHOT───│<─────HITL Checkpoint───────│
  │<──SSE: CUSTOM───────────│                            │
  │    (review_required)    │                            │
  │                         │                            │
  │                    [PAUSE - Waiting for human review]│
  │                         │                            │
```

### 6.2 HITL Review Flow

```
Client                    Server                    Review System
  │                         │                            │
  │──GET /review/queue──────>│                            │
  │<──{high, med, low}──────│                            │
  │                         │                            │
  │──POST /review/batch─────>│                            │
  │  (auto-approve high)    │──batch_approve────────────>│
  │<──SSE: CUSTOM───────────│<─────approved──────────────│
  │    (batch_approved)     │                            │
  │<──SSE: STATE_DELTA──────│                            │
  │                         │                            │
  │──POST /review/decision──>│                            │
  │  (approve item)         │──submit_review────────────>│
  │<──SSE: CUSTOM───────────│<─────decision──────────────│
  │    (review_decision)    │                            │
  │<──SSE: STATE_DELTA──────│                            │
  │                         │                            │
  │──POST /review/finalize──>│                            │
  │                         │──finalize_session─────────>│
  │<──SSE: CUSTOM───────────│<─────finalized─────────────│
  │    (session_finalized)  │                            │
  │                         │                            │
  │<──SSE: STEP_STARTED─────│<─────scoring───────────────│
  │    ... continues ...    │                            │
  │                         │                            │
  │<──SSE: RUN_FINISHED─────│                            │
  │                         │                            │
```

---

## 7. File Structure After Implementation

```
wafr-prototype/
├── ag_ui/
│   ├── __init__.py
│   ├── emitter.py          # WAFREventEmitter class
│   ├── state.py            # WAFRState class
│   ├── events.py           # Custom event definitions
│   ├── server.py           # FastAPI SSE endpoints
│   └── middleware.py       # Optional: AG-UI middleware
├── agents/
│   ├── orchestrator.py     # ENHANCED with AG-UI
│   ├── answer_synthesis_agent.py  # ENHANCED with progress
│   ├── review_orchestrator.py     # ENHANCED with events
│   └── ...
├── storage/
│   └── ...
├── models/
│   └── ...
├── tests/
│   ├── test_ag_ui_events.py       # NEW
│   ├── test_ag_ui_streaming.py    # NEW
│   └── ...
└── requirements.txt        # + ag-ui-protocol
```

---

## 8. Benefits of AG-UI Integration

| Benefit | Description |
|---------|-------------|
| **Real-time Progress** | Frontend shows live pipeline progress |
| **Efficient Updates** | STATE_DELTA minimizes data transfer |
| **Standard Protocol** | Compatible with any AG-UI client |
| **HITL Awareness** | Custom events for review workflow |
| **Error Handling** | Structured error events |
| **Debuggability** | Event stream is inspectable |
| **Scalability** | SSE works with load balancers |

---

## 9. Dependencies

Add to `requirements.txt`:

```
ag-ui-protocol>=0.1.0
fastapi>=0.100.0
sse-starlette>=1.6.0
uvicorn>=0.23.0
```

---

## 10. Success Criteria

- [ ] All 16 AG-UI event types implemented
- [ ] HITL custom events working
- [ ] SSE streaming stable (no connection drops)
- [ ] State sync efficient (< 100ms latency)
- [ ] 100% test coverage for event emission
- [ ] Compatible with standard AG-UI clients
- [ ] Documentation complete

---

## References

- [AG-UI Core Architecture](https://docs.ag-ui.com/concepts/architecture)
- [AG-UI Events](https://docs.ag-ui.com/concepts/events)
- [AG-UI Python SDK](https://docs.ag-ui.com/sdk/python/core/overview)
- [AG-UI State Management](https://docs.ag-ui.com/concepts/state)
- [AG-UI Server Quickstart](https://docs.ag-ui.com/quickstart/server)
- [GitHub: ag-ui-protocol/ag-ui](https://github.com/ag-ui-protocol/ag-ui)

