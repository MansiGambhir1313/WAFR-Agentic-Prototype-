# AG-UI Integration Guide for WAFR

This guide documents the complete AG-UI (Agent User Interaction Protocol) integration in the WAFR project.

## Overview

The WAFR project now includes comprehensive AG-UI integration, enabling real-time event streaming, tool call tracking, message streaming, and state management throughout the pipeline execution.

**Official AG-UI Documentation**: https://docs.ag-ui.com/sdk/python/core/overview

---

## Installation

### 1. Install AG-UI Protocol SDK

```bash
pip install ag-ui-protocol
```

The package is also listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```python
from ag_ui.core import RunAgentInput, Message, Context, Tool, State
print("AG-UI SDK imported successfully!")
```

---

## Architecture

### Components

1. **`ag_ui/core.py`** - Official SDK wrapper and WAFR adapters
   - Wraps official `ag-ui-protocol` SDK types
   - Provides WAFR-specific adapters (WAFRRunAgentInput, WAFRMessage, etc.)
   - Defines WAFR agent tools registry

2. **`ag_ui/emitter.py`** - Event emitter (enhanced with official SDK)
   - Emits all 16 standard AG-UI event types
   - Supports custom HITL events
   - SSE-compatible streaming

3. **`ag_ui/state.py`** - State management
   - WAFRState class for complete session state
   - JSON Patch deltas for incremental updates
   - State snapshots for initial sync

4. **`ag_ui/orchestrator_integration.py`** - Orchestrator wrapper
   - AGUIOrchestratorWrapper adds AG-UI events to pipeline
   - Tool call events for each agent
   - Message streaming for agent responses

5. **`ag_ui/server.py`** - FastAPI SSE server
   - REST endpoints for WAFR processing
   - SSE event streaming
   - WebSocket support (optional)

---

## Event Types

### Standard AG-UI Events (16 types)

#### Lifecycle Events
- `RUN_STARTED` - Pipeline execution begins
- `RUN_FINISHED` - Pipeline execution completes
- `RUN_ERROR` - Pipeline execution fails
- `STEP_STARTED` - Individual step begins
- `STEP_FINISHED` - Individual step completes

#### Text Message Events
- `TEXT_MESSAGE_START` - Message stream begins
- `TEXT_MESSAGE_CONTENT` - Message chunk received
- `TEXT_MESSAGE_END` - Message stream completes

#### Tool Call Events
- `TOOL_CALL_START` - Agent/tool execution begins
- `TOOL_CALL_ARGS` - Tool arguments streamed
- `TOOL_CALL_END` - Tool execution completes

#### State Management Events
- `STATE_SNAPSHOT` - Complete state at point in time
- `STATE_DELTA` - Incremental changes (JSON Patch)
- `MESSAGES_SNAPSHOT` - Complete conversation history

#### Special Events
- `RAW` - Raw/unprocessed data
- `CUSTOM` - Custom application events

### Custom HITL Events

- `hitl.review_required` - Human review needed
- `hitl.synthesis_progress` - Answer synthesis progress
- `hitl.review_decision` - Human review decision
- `hitl.batch_approve_completed` - Batch approval done
- `hitl.validation_status` - Finalization validation
- `hitl.session_finalized` - Session finalized

---

## Usage Examples

### 1. Basic Usage with AG-UI

```python
import asyncio
from ag_ui.orchestrator_integration import create_agui_orchestrator

async def main():
    # Create AG-UI enabled orchestrator
    orchestrator = create_agui_orchestrator(thread_id="session-123")
    emitter = orchestrator.emitter
    
    # Process transcript with AG-UI events
    results = await orchestrator.process_transcript_with_agui(
        transcript=transcript_text,
        session_id="session-123",
        generate_report=True,
    )
    
    # Stream events (for SSE/WebSocket)
    async for event_data in emitter.stream_events():
        print(event_data)  # SSE format: "data: {...}\n\n"

asyncio.run(main())
```

### 2. Using Standalone Script

```bash
# Run with default transcript
python run_wafr_with_agui.py

# Run with custom transcript
python run_wafr_with_agui.py --transcript transcript.txt

# Save events to file
python run_wafr_with_agui.py --transcript transcript.txt --output-events events.jsonl

# With WA Tool integration
python run_wafr_with_agui.py --wa-tool --client-name "My Client"
```

### 3. Using FastAPI Server

```bash
# Start server
uvicorn ag_ui.server:app --reload --port 8000

# Or use the router in your existing FastAPI app
from ag_ui.server import router
app.include_router(router)
```

**Endpoints:**
- `POST /api/wafr/run` - Run WAFR assessment with SSE streaming
- `GET /api/wafr/session/{session_id}/state` - Get session state
- `POST /api/wafr/review/{session_id}/decision` - Submit review decision
- `POST /api/wafr/review/{session_id}/batch-approve` - Batch approve items
- `POST /api/wafr/review/{session_id}/finalize` - Finalize session

### 4. Event Collection and Analysis

```python
from ag_ui.emitter import WAFREventEmitter

class EventCollector:
    def __init__(self):
        self.events = []
    
    def collect(self, event):
        self.events.append(event.to_dict())

emitter = WAFREventEmitter(thread_id="session-123")
collector = EventCollector()
emitter.add_listener(collector.collect)

# Process pipeline...
# Events are automatically collected
print(f"Collected {len(collector.events)} events")
```

---

## Pipeline Step Mapping

| WAFR Step | AG-UI Events |
|-----------|--------------|
| Session Start | `RUN_STARTED` → `STATE_SNAPSHOT` |
| PDF Processing | `STEP_STARTED` → `TOOL_CALL_*` → `STEP_FINISHED` |
| Understanding | `STEP_STARTED` → `TOOL_CALL_*` → `TEXT_MESSAGE_*` → `STEP_FINISHED` |
| Mapping | `STEP_STARTED` → `TOOL_CALL_*` → `STATE_DELTA` → `STEP_FINISHED` |
| Confidence | `STEP_STARTED` → `TOOL_CALL_*` → `STATE_DELTA` → `STEP_FINISHED` |
| Gap Detection | `STEP_STARTED` → `TOOL_CALL_*` → `STATE_DELTA` → `STEP_FINISHED` |
| Answer Synthesis | `STEP_STARTED` → `TOOL_CALL_*` → `hitl.synthesis_progress` → `STEP_FINISHED` |
| HITL Review | `STATE_SNAPSHOT` → `hitl.review_required` → `hitl.review_decision` |
| Scoring | `STEP_STARTED` → `TOOL_CALL_*` → `STATE_DELTA` → `STEP_FINISHED` |
| Report Generation | `STEP_STARTED` → `TOOL_CALL_*` → `STEP_FINISHED` |
| WA Tool Integration | `STEP_STARTED` → `TOOL_CALL_*` → `STEP_FINISHED` |
| Session Complete | `STATE_SNAPSHOT` → `RUN_FINISHED` |

---

## WAFR Agent Tools

Each WAFR agent is registered as an AG-UI Tool:

1. **understanding_agent** - Extracts insights from transcripts
2. **mapping_agent** - Maps insights to WAFR questions
3. **confidence_agent** - Validates evidence and assigns confidence
4. **gap_detection_agent** - Identifies gaps in coverage
5. **answer_synthesis_agent** - Synthesizes answers for gaps
6. **scoring_agent** - Scores and ranks answers
7. **report_agent** - Generates PDF reports
8. **wa_tool_agent** - Integrates with AWS WA Tool API

Access tools:

```python
from ag_ui.core import get_wafr_tool, get_all_wafr_tools

# Get specific tool
tool = get_wafr_tool("understanding")
print(tool.name, tool.description)

# Get all tools
for tool in get_all_wafr_tools():
    print(f"{tool.name}: {tool.description}")
```

---

## State Management

### State Structure

```json
{
  "session": {
    "id": "session-123",
    "status": "processing",
    "started_at": "2026-01-07T12:00:00Z",
    "updated_at": "2026-01-07T12:05:00Z"
  },
  "pipeline": {
    "current_step": "understanding",
    "completed_steps": ["pdf_processing"],
    "total_steps": 10,
    "progress_percentage": 10.0
  },
  "content": {
    "transcript_loaded": true,
    "transcript_length": 2481,
    "insights_count": 12,
    "questions_answered": 22,
    "questions_total": 51,
    "gaps_count": 38,
    "synthesized_count": 38,
    "coverage_percentage": 43.1
  },
  "review": {
    "session_id": "session-123",
    "status": "pending",
    "pending_count": 5,
    "approved_count": 33,
    "review_progress": 86.8
  },
  "scores": {
    "authenticity_score": 85.5,
    "overall_score": 82.3,
    "pillar_scores": {
      "SEC": 88.0,
      "REL": 85.0,
      "PERF": 80.0
    }
  },
  "report": {
    "generated": true,
    "file_path": "wafr_report_123.pdf",
    "format": "pdf"
  }
}
```

### State Updates

State updates use JSON Patch format:

```python
# Single delta
delta = {
    "op": "replace",
    "path": "/content/insights_count",
    "value": 15
}

# Multiple deltas
deltas = [
    {"op": "replace", "path": "/pipeline/current_step", "value": "mapping"},
    {"op": "replace", "path": "/content/insights_count", "value": 15}
]
```

---

## Frontend Integration

### SSE (Server-Sent Events)

```javascript
const eventSource = new EventSource('/api/wafr/run', {
    method: 'POST',
    body: JSON.stringify({
        transcript: transcriptText,
        session_id: 'session-123'
    })
});

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Event:', data.type, data);
    
    switch(data.type) {
        case 'RUN_STARTED':
            console.log('Pipeline started');
            break;
        case 'STEP_STARTED':
            console.log('Step:', data.stepName);
            break;
        case 'STATE_DELTA':
            updateState(data.delta);
            break;
        case 'RUN_FINISHED':
            console.log('Pipeline completed');
            eventSource.close();
            break;
    }
};
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/wafr/session-123');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleAGUIEvent(data);
};
```

---

## Testing

### Run Tests

```bash
# Run AG-UI specific tests
pytest tests/test_ag_ui.py -v

# Run with event collection
python run_wafr_with_agui.py --transcript test_transcript.txt --output-events test_events.jsonl
```

### Verify Events

```python
import json

# Load events from JSONL
events = []
with open('test_events.jsonl', 'r') as f:
    for line in f:
        events.append(json.loads(line))

# Analyze events
event_types = {}
for event in events:
    event_type = event.get('type', 'UNKNOWN')
    event_types[event_type] = event_types.get(event_type, 0) + 1

print("Event Summary:")
for event_type, count in sorted(event_types.items()):
    print(f"  {event_type}: {count}")
```

---

## Troubleshooting

### AG-UI SDK Not Found

```
ImportError: cannot import name 'RunAgentInput' from 'ag_ui.core'
```

**Solution**: Install the official SDK:
```bash
pip install ag-ui-protocol
```

### Events Not Streaming

Check that:
1. Emitter is properly initialized
2. Events are being emitted (check logs)
3. SSE/WebSocket connection is active
4. CORS is configured correctly

### State Not Updating

Ensure:
1. State deltas are being created correctly
2. JSON Patch format is valid
3. Paths match state structure

---

## References

- **Official AG-UI Documentation**: https://docs.ag-ui.com
- **Python SDK**: https://docs.ag-ui.com/sdk/python/core/overview
- **Events Reference**: https://docs.ag-ui.com/sdk/python/core/events
- **Types Reference**: https://docs.ag-ui.com/sdk/python/core/types

---

## Summary

The WAFR project now has complete AG-UI integration, providing:

✅ All 16 standard AG-UI event types  
✅ Custom HITL events for review workflow  
✅ Tool call tracking for all agents  
✅ Message streaming for agent responses  
✅ Complete state management with snapshots and deltas  
✅ FastAPI SSE server for real-time updates  
✅ WebSocket support for bidirectional communication  
✅ Comprehensive examples and documentation  

This enables real-time frontend updates, debugging, monitoring, and integration with AG-UI compatible tools and frameworks.

