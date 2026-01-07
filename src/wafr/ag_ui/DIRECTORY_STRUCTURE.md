# AG-UI Directory Structure

This document provides an overview of all files in the `ag_ui/` directory.

## Directory Contents

```
ag_ui/
├── __init__.py                    # Package initialization (29 exports)
├── core.py                        # Official SDK wrapper & WAFR adapters
├── events.py                      # Custom HITL event definitions
├── emitter.py                     # AG-UI event emitter (16 standard + custom)
├── state.py                       # State management (WAFRState)
├── orchestrator_integration.py    # Orchestrator wrapper with AG-UI
├── server.py                      # FastAPI SSE server
├── README.md                      # Module documentation
└── DIRECTORY_STRUCTURE.md         # This file
```

## File Descriptions

### `__init__.py`
- **Purpose**: Package initialization and public API exports
- **Exports**: 29 items including all core components
- **Version**: 1.0.0
- **Usage**: `from ag_ui import WAFREventEmitter, WAFRState, create_agui_orchestrator`

### `core.py`
- **Purpose**: Official AG-UI SDK wrapper and WAFR-specific adapters
- **Key Components**:
  - Official SDK type imports (RunAgentInput, Message, Context, Tool, State)
  - WAFR adapters (WAFRRunAgentInput, WAFRMessage, WAFRContext, WAFRTool)
  - WAFR agent tools registry (8 agents: understanding, mapping, confidence, etc.)
  - Graceful fallback if SDK not installed
- **Usage**: `from ag_ui.core import get_wafr_tool, WAFR_AGENT_TOOLS`

### `events.py`
- **Purpose**: Custom event definitions for WAFR/HITL workflow
- **Key Components**:
  - HITLEventType enum (review_required, synthesis_progress, etc.)
  - Event data classes (ReviewQueueSummary, SynthesisProgress, ReviewDecisionData, ValidationStatus)
  - HITLEvents factory class
  - WAFRPipelineStep enum
- **Usage**: `from ag_ui.events import HITLEvents, SynthesisProgress`

### `emitter.py`
- **Purpose**: AG-UI compliant event emitter for streaming events
- **Key Components**:
  - WAFREventEmitter class
  - All 16 standard AG-UI event types
  - Custom HITL event support
  - SSE-compatible streaming
  - Async event queue with heartbeat
- **Usage**: `from ag_ui.emitter import WAFREventEmitter`

### `state.py`
- **Purpose**: Complete state management for WAFR assessment sessions
- **Key Components**:
  - WAFRState class with nested components
  - SessionInfo, PipelineProgress, ContentState, ReviewState, ScoreState, ReportState
  - JSON Patch operations for state deltas
  - State snapshots and incremental updates
- **Usage**: `from ag_ui.state import WAFRState, SessionStatus`

### `orchestrator_integration.py`
- **Purpose**: Orchestrator wrapper that adds AG-UI event emission
- **Key Components**:
  - AGUIOrchestratorWrapper class
  - Tool call events for each agent
  - Message streaming for agent responses
  - Step-by-step event emission
  - Async processing support
- **Usage**: `from ag_ui.orchestrator_integration import create_agui_orchestrator`

### `server.py`
- **Purpose**: FastAPI server for AG-UI event streaming
- **Key Components**:
  - FastAPI application with SSE endpoints
  - REST API for WAFR processing
  - Review decision endpoints
  - WebSocket support (optional)
  - Session state management
- **Usage**: `uvicorn ag_ui.server:app --reload --port 8000`

### `README.md`
- **Purpose**: Module documentation and usage guide
- **Content**: Overview, usage examples, module structure, dependencies

## Module Dependencies

### Internal Dependencies
```
ag_ui/
├── __init__.py
│   ├── imports from: core, events, state, emitter, orchestrator_integration
│
├── core.py
│   └── (standalone, wraps external SDK)
│
├── events.py
│   └── (standalone)
│
├── emitter.py
│   ├── imports from: events, state, core
│
├── state.py
│   └── (standalone)
│
├── orchestrator_integration.py
│   ├── imports from: emitter, core, events
│
└── server.py
    ├── imports from: emitter, events, state, orchestrator_integration
```

### External Dependencies
- `ag-ui-protocol>=0.1.0` - Official AG-UI SDK (optional, with fallback)
- `fastapi>=0.100.0` - Web framework (for server.py)
- `sse-starlette>=1.6.0` - SSE support (for server.py)
- `websockets>=11.0.0` - WebSocket support (for server.py)

## Import Patterns

### Basic Import
```python
from ag_ui import WAFREventEmitter, WAFRState, HITLEvents
```

### Full Import
```python
from ag_ui import (
    # Core SDK
    WAFRRunAgentInput,
    WAFRMessage,
    WAFRTool,
    get_wafr_tool,
    
    # Events
    HITLEventType,
    HITLEvents,
    SynthesisProgress,
    
    # State
    WAFRState,
    SessionStatus,
    
    # Emitter
    WAFREventEmitter,
    
    # Orchestrator
    create_agui_orchestrator,
)
```

### Module-Specific Imports
```python
from ag_ui.core import get_all_wafr_tools, WAFR_AGENT_TOOLS
from ag_ui.events import ReviewQueueSummary, ValidationStatus
from ag_ui.state import JSONPatch, PatchOp
from ag_ui.orchestrator_integration import AGUIOrchestratorWrapper
```

## File Sizes (Approximate)

- `__init__.py`: ~100 lines
- `core.py`: ~400 lines
- `events.py`: ~380 lines
- `emitter.py`: ~870 lines
- `state.py`: ~590 lines
- `orchestrator_integration.py`: ~200 lines
- `server.py`: ~630 lines
- `README.md`: ~150 lines

**Total**: ~3,320 lines of code + documentation

## Testing

All AG-UI components are tested in:
- `tests/test_ag_ui.py` - Unit tests for events, state, emitter, server

## Related Files (Outside ag_ui/)

- `run_wafr_with_agui.py` - Standalone script using AG-UI
- `AG_UI_INTEGRATION_GUIDE.md` - Comprehensive integration guide
- `AG_UI_IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `tests/test_ag_ui.py` - Unit tests

## Status

✅ All AG-UI related files are properly organized in `ag_ui/` directory  
✅ All modules are properly exported via `__init__.py`  
✅ Documentation is complete  
✅ Dependencies are clearly defined  
✅ Module structure is clean and maintainable  

