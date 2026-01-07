# WAFR AgentCore Integration Plan

## Executive Summary

This document outlines the comprehensive plan to migrate the WAFR (Well-Architected Framework Review) multi-agent system from a local execution model to **AWS Bedrock AgentCore**, enabling serverless deployment, persistent memory, tool gateway integration, and enterprise-grade observability.

---

## Current Architecture Analysis

### Current WAFR System Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  CLI/File Input │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   WafrOrchestrator (Local)          │
│   • Coordinates agent pipeline      │
│   • Manages workflow                │
│   • Handles errors                  │
└────────┬────────────────────────────┘
         │
         ├──► Understanding Agent
         ├──► Mapping Agent
         ├──► Confidence Agent
         ├──► Gap Detection Agent
         ├──► Answer Synthesis Agent (HITL)
         ├──► Review Orchestrator (HITL)
         ├──► Scoring Agent
         ├──► Report Agent
         └──► WA Tool Agent
         │
         ▼
┌─────────────────┐
│  Local Output   │
│  • JSON results │
│  • PDF reports  │
└─────────────────┘
```

### Current Technology Stack

| Component | Technology | Status |
|-----------|-----------|--------|
| **Agent Framework** | Strands Agents | ✅ Already compatible |
| **LLM** | Amazon Bedrock (Claude) | ✅ Already compatible |
| **Orchestration** | Custom WafrOrchestrator | ⚠️ Needs AgentCore wrapper |
| **Memory** | In-memory/File-based | ⚠️ Needs AgentCore Memory |
| **Storage** | File-based (JSON) | ⚠️ Needs AgentCore integration |
| **Deployment** | Local Python script | ⚠️ Needs AgentCore Runtime |
| **Observability** | Basic logging | ⚠️ Needs AgentCore Observability |
| **Tools** | Direct function calls | ⚠️ Needs AgentCore Gateway |

---

## AgentCore Integration Architecture

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              AGENTCORE-ENABLED ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Client Layer                                    │
│  • CLI / SDK / REST API / WebSocket                         │
└─────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         AWS Bedrock AgentCore Runtime                        │
│  • Serverless hosting                                        │
│  • Auto-scaling                                              │
│  • Session isolation                                          │
│  • JWT authentication                                         │
└─────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         WAFR AgentCore App (agentcore_agent.py)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  @app.entrypoint                                       │  │
│  │  def invoke(payload) -> Dict                          │  │
│  │      • Receives request                               │  │
│  │      • Extracts actor_id, session_id                  │  │
│  │      • Routes to WafrOrchestrator                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                       │                                     │
│                       ▼                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  WafrOrchestrator (Enhanced)                           │  │
│  │  • AgentCore Memory integration                       │  │
│  │  • AgentCore Gateway for tools                        │  │
│  │  • AgentCore Identity for credentials                 │  │
│  │  • AgentCore Observability hooks                      │  │
│  └─────────────────────┬─────────────────────────────────┘  │
│                        │                                     │
│        ┌───────────────┼───────────────┐                    │
│        │               │               │                    │
│        ▼               ▼               ▼                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │ Agents   │   │ Memory   │   │ Gateway  │               │
│  │ (Strands)│   │ Client   │   │ Client   │               │
│  └──────────┘   └──────────┘   └──────────┘               │
└─────────────────────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ AgentCore    │ │ AgentCore    │ │ AgentCore    │
│ Memory       │ │ Gateway      │ │ Identity     │
│ (DynamoDB)   │ │ (MCP Server) │ │ (Secrets)    │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## Integration Components

### 1. AgentCore Runtime Integration

**File:** `agentcore_agent.py` (NEW)

**Purpose:** Main entrypoint for AgentCore Runtime

**Key Features:**
- Wraps WafrOrchestrator in AgentCore app
- Handles AgentCore request format
- Manages session context
- Integrates with AgentCore Memory
- Exposes health check endpoint

**Implementation:**
```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agents.orchestrator import create_orchestrator
from bedrock_agentcore.memory import MemoryClient
from typing import Dict, Any

app = BedrockAgentCoreApp()
orchestrator = None  # Lazy initialization

@app.entrypoint
def invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint for WAFR agent.
    
    Expected payload:
    {
        "prompt": "Process this transcript...",
        "transcript": "...",  # Optional if in memory
        "actor_id": "user-123",
        "session_id": "session-456",
        "action": "process_transcript" | "review_answer" | "generate_report"
    }
    """
    global orchestrator
    
    # Initialize orchestrator if needed
    if orchestrator is None:
        orchestrator = create_orchestrator()
    
    # Extract context
    actor_id = payload.get("actor_id", "default-user")
    session_id = payload.get("session_id", "default-session")
    action = payload.get("action", "process_transcript")
    
    # Route to appropriate handler
    if action == "process_transcript":
        return handle_process_transcript(payload, orchestrator, actor_id, session_id)
    elif action == "review_answer":
        return handle_review_answer(payload, orchestrator, actor_id, session_id)
    elif action == "generate_report":
        return handle_generate_report(payload, orchestrator, actor_id, session_id)
    else:
        return {"error": f"Unknown action: {action}"}

@app.health_check
def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "wafr-agent",
        "version": "2.0.0"
    }
```

---

### 2. AgentCore Memory Integration

**File:** `agents/memory_manager.py` (NEW)

**Purpose:** Manages conversation context using AgentCore Memory

**Key Features:**
- Stores transcript processing history
- Retrieves previous assessment context
- Maintains user preferences
- Tracks review decisions

**Implementation:**
```python
from bedrock_agentcore.memory import MemoryClient
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class WAFRMemoryManager:
    """Manages WAFR-specific memory operations"""
    
    def __init__(self, memory_id: str, region: str = "us-east-1"):
        self.memory_id = memory_id
        self.client = MemoryClient(region_name=region)
        logger.info(f"WAFR Memory Manager initialized: {memory_id}")
    
    def store_assessment_context(
        self,
        actor_id: str,
        session_id: str,
        transcript: str,
        insights: List[Dict],
        mappings: List[Dict],
        gaps: List[Dict]
    ) -> Dict[str, Any]:
        """Store assessment context for later retrieval"""
        messages = [
            {
                "role": "system",
                "content": f"WAFR Assessment Context for session {session_id}"
            },
            {
                "role": "user",
                "content": f"Transcript: {transcript[:1000]}..."
            },
            {
                "role": "assistant",
                "content": f"Insights: {len(insights)} extracted. "
                          f"Mappings: {len(mappings)} found. "
                          f"Gaps: {len(gaps)} identified."
            }
        ]
        
        event = self.client.create_event(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=messages
        )
        
        return event
    
    def retrieve_assessment_context(
        self,
        actor_id: str,
        session_id: str,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve previous assessment context"""
        context = self.client.query_memory(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            query=query or "previous WAFR assessment context"
        )
        
        return context
    
    def store_review_decision(
        self,
        actor_id: str,
        session_id: str,
        question_id: str,
        decision: str,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """Store human review decision for learning"""
        messages = [
            {
                "role": "user",
                "content": f"Review decision for {question_id}: {decision}"
            },
            {
                "role": "assistant",
                "content": f"Decision: {decision}. Feedback: {feedback or 'None'}"
            }
        ]
        
        event = self.client.create_event(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=f"{session_id}-reviews",
            messages=messages
        )
        
        return event
```

**Integration Points:**
- `agents/orchestrator.py`: Add memory retrieval at start
- `agents/review_orchestrator.py`: Store review decisions
- `agents/answer_synthesis_agent.py`: Use context for better synthesis

---

### 3. AgentCore Gateway Integration

**File:** `agents/gateway_tools.py` (NEW)

**Purpose:** Expose WAFR agents as Gateway tools

**Key Features:**
- Register agents as MCP tools
- Enable tool discovery
- Support tool chaining
- Handle authentication

**Implementation:**
```python
from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class WAFRGatewayManager:
    """Manages Gateway tool registration"""
    
    def __init__(self, gateway_id: str, region: str = "us-east-1"):
        self.gateway_id = gateway_id
        self.client = GatewayClient(region_name=region)
        logger.info(f"WAFR Gateway Manager initialized: {gateway_id}")
    
    def register_wafr_tools(self) -> List[Dict[str, Any]]:
        """Register WAFR agents as Gateway tools"""
        
        tools = [
            {
                "name": "extract_insights",
                "description": "Extract architecture insights from transcript",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "transcript": {"type": "string"},
                        "session_id": {"type": "string"}
                    },
                    "required": ["transcript", "session_id"]
                }
            },
            {
                "name": "map_to_wafr",
                "description": "Map insights to WAFR questions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "insights": {"type": "array"},
                        "session_id": {"type": "string"}
                    },
                    "required": ["insights", "session_id"]
                }
            },
            {
                "name": "synthesize_answer",
                "description": "Synthesize answer for gap question",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question_id": {"type": "string"},
                        "context": {"type": "object"},
                        "session_id": {"type": "string"}
                    },
                    "required": ["question_id", "context", "session_id"]
                }
            },
            {
                "name": "generate_report",
                "description": "Generate WAFR assessment report",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "format": {"type": "string", "enum": ["pdf", "json"]}
                    },
                    "required": ["session_id"]
                }
            }
        ]
        
        # Register each tool
        registered_tools = []
        for tool in tools:
            result = self.client.register_tool(
                gateway_id=self.gateway_id,
                tool_definition=tool
            )
            registered_tools.append(result)
            logger.info(f"Registered tool: {tool['name']}")
        
        return registered_tools
```

**Integration Points:**
- `agents/orchestrator.py`: Use Gateway tools instead of direct calls
- `agents/answer_synthesis_agent.py`: Expose as Gateway tool
- `agents/report_agent.py`: Expose report generation as tool

---

### 4. AgentCore Identity Integration

**File:** `agents/identity_manager.py` (NEW)

**Purpose:** Manage secure credential access

**Key Features:**
- Store AWS credentials securely
- Manage OAuth tokens for external APIs
- Provide just-enough access
- Support credential rotation

**Implementation:**
```python
from bedrock_agentcore.services.identity import IdentityClient
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class WAFRIdentityManager:
    """Manages identity and credentials for WAFR agent"""
    
    def __init__(self, identity_arn: str, region: str = "us-east-1"):
        self.identity_arn = identity_arn
        self.client = IdentityClient(region)
        logger.info(f"WAFR Identity Manager initialized: {identity_arn}")
    
    def get_aws_credentials(
        self,
        actor_id: str,
        required_permissions: List[str]
    ) -> Dict[str, str]:
        """Get AWS credentials with required permissions"""
        credentials = self.client.get_credentials(
            identity_arn=self.identity_arn,
            actor_id=actor_id,
            permissions=required_permissions
        )
        
        return {
            "access_key_id": credentials["access_key_id"],
            "secret_access_key": credentials["secret_access_key"],
            "session_token": credentials.get("session_token")
        }
    
    def store_oauth_token(
        self,
        actor_id: str,
        service: str,
        token: str,
        refresh_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Store OAuth token for external service"""
        result = self.client.store_token(
            identity_arn=self.identity_arn,
            actor_id=actor_id,
            service=service,
            token=token,
            refresh_token=refresh_token
        )
        
        return result
```

**Integration Points:**
- `agents/wa_tool_agent.py`: Use Identity for AWS credentials
- `agents/report_agent.py`: Use Identity for S3 access
- Any external API integrations

---

### 5. AgentCore Observability Integration

**File:** `agents/observability_hooks.py` (NEW)

**Purpose:** Add observability hooks throughout pipeline

**Key Features:**
- Trace agent execution
- Log metrics (latency, token usage)
- Track errors
- Monitor performance

**Implementation:**
```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import logging
from typing import Dict, Any
from functools import wraps
import time

logger = logging.getLogger(__name__)

# Initialize tracing
tracer = trace.get_tracer(__name__)

def trace_agent_execution(agent_name: str):
    """Decorator to trace agent execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(f"{agent_name}.{func.__name__}"):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Log metrics
                    logger.info(f"{agent_name} execution: {duration:.2f}s")
                    
                    return result
                except Exception as e:
                    logger.error(f"{agent_name} error: {str(e)}")
                    raise
        return wrapper
    return decorator

class WAFRObservability:
    """Manages observability for WAFR pipeline"""
    
    @staticmethod
    def log_agent_start(agent_name: str, session_id: str):
        """Log agent start"""
        logger.info(f"[{session_id}] Starting {agent_name}")
    
    @staticmethod
    def log_agent_complete(agent_name: str, session_id: str, duration: float):
        """Log agent completion"""
        logger.info(f"[{session_id}] Completed {agent_name} in {duration:.2f}s")
    
    @staticmethod
    def log_agent_error(agent_name: str, session_id: str, error: Exception):
        """Log agent error"""
        logger.error(f"[{session_id}] {agent_name} error: {str(error)}")
```

**Integration Points:**
- All agent `process()` methods
- `agents/orchestrator.py`: Trace pipeline execution
- `agents/review_orchestrator.py`: Track review metrics

---

## Migration Steps

### Phase 1: Foundation Setup (Week 1)

**Goal:** Set up AgentCore infrastructure and basic integration

**Tasks:**

1. **Install AgentCore Dependencies**
   ```bash
   pip install bedrock-agentcore bedrock-agentcore-starter-toolkit
   ```

2. **Create AgentCore Configuration**
   - Create `.bedrock_agentcore.yaml`
   - Configure runtime settings
   - Set up environment variables

3. **Create Memory Resource**
   ```bash
   python setup_agentcore.py --setup-memory
   ```

4. **Create Gateway Resource**
   ```bash
   python setup_agentcore.py --setup-gateway
   ```

5. **Create Identity Resource**
   ```bash
   python setup_agentcore.py --setup-identity
   ```

6. **Create `agentcore_agent.py`**
   - Basic entrypoint wrapper
   - Health check endpoint
   - Simple routing

**Deliverables:**
- ✅ AgentCore infrastructure created
- ✅ Basic entrypoint working
- ✅ Can deploy to Runtime

---

### Phase 2: Memory Integration (Week 2)

**Goal:** Integrate AgentCore Memory for context persistence

**Tasks:**

1. **Create `agents/memory_manager.py`**
   - Implement WAFRMemoryManager
   - Add context storage methods
   - Add context retrieval methods

2. **Update `agents/orchestrator.py`**
   - Initialize MemoryManager
   - Retrieve context at start
   - Store context after processing

3. **Update `agents/review_orchestrator.py`**
   - Store review decisions in memory
   - Retrieve review history

4. **Update `agents/answer_synthesis_agent.py`**
   - Use memory context for synthesis
   - Learn from previous reviews

**Deliverables:**
- ✅ Memory integration complete
- ✅ Context persists across sessions
- ✅ Review history available

---

### Phase 3: Gateway Integration (Week 3)

**Goal:** Expose WAFR agents as Gateway tools

**Tasks:**

1. **Create `agents/gateway_tools.py`**
   - Implement WAFRGatewayManager
   - Define tool schemas
   - Register tools

2. **Update Agent Implementations**
   - Make agents callable via Gateway
   - Add tool wrappers
   - Handle Gateway requests

3. **Update `agents/orchestrator.py`**
   - Use Gateway tools instead of direct calls
   - Support tool chaining

**Deliverables:**
- ✅ Agents exposed as Gateway tools
- ✅ Tool discovery working
- ✅ Can chain tools together

---

### Phase 4: Identity Integration (Week 4)

**Goal:** Secure credential management

**Tasks:**

1. **Create `agents/identity_manager.py`**
   - Implement WAFRIdentityManager
   - Add credential retrieval
   - Add token storage

2. **Update `agents/wa_tool_agent.py`**
   - Use Identity for AWS credentials
   - Remove hardcoded credentials

3. **Update `agents/report_agent.py`**
   - Use Identity for S3 access
   - Secure file storage

**Deliverables:**
- ✅ Secure credential management
- ✅ No hardcoded credentials
- ✅ Just-enough access working

---

### Phase 5: Observability Integration (Week 5)

**Goal:** Add comprehensive observability

**Tasks:**

1. **Create `agents/observability_hooks.py`**
   - Implement tracing decorators
   - Add metrics logging
   - Add error tracking

2. **Update All Agents**
   - Add observability decorators
   - Log execution metrics
   - Track errors

3. **Create CloudWatch Dashboards**
   - Agent execution metrics
   - Error rates
   - Token usage

**Deliverables:**
- ✅ Full observability
- ✅ CloudWatch dashboards
- ✅ Error tracking

---

### Phase 6: Deployment & Testing (Week 6)

**Goal:** Deploy to production and test

**Tasks:**

1. **Create Deployment Scripts**
   - `deploy_agentcore.py`
   - Docker configuration
   - CI/CD pipeline

2. **Test Locally**
   - Test with AgentCore Runtime locally
   - Verify all integrations
   - Performance testing

3. **Deploy to AWS**
   - Deploy to AgentCore Runtime
   - Configure auto-scaling
   - Set up monitoring

4. **End-to-End Testing**
   - Test complete pipeline
   - Test memory persistence
   - Test Gateway tools
   - Test Identity

**Deliverables:**
- ✅ Deployed to AgentCore Runtime
- ✅ All tests passing
- ✅ Production-ready

---

## File Structure Changes

### New Files to Create

```
WAFR prototype - Copy/
├── agentcore_agent.py              # NEW: Main AgentCore entrypoint
├── setup_agentcore.py              # NEW: Infrastructure setup
├── deploy_agentcore.py             # NEW: Deployment script
├── .bedrock_agentcore.yaml         # NEW: AgentCore config
├── agentcore_config.json           # NEW: Infrastructure config
│
├── agents/
│   ├── memory_manager.py           # NEW: AgentCore Memory integration
│   ├── gateway_tools.py            # NEW: Gateway tool registration
│   ├── identity_manager.py         # NEW: Identity management
│   ├── observability_hooks.py      # NEW: Observability integration
│   │
│   ├── orchestrator.py             # MODIFY: Add Memory/Gateway integration
│   ├── review_orchestrator.py      # MODIFY: Add Memory integration
│   ├── answer_synthesis_agent.py   # MODIFY: Use Memory context
│   ├── wa_tool_agent.py            # MODIFY: Use Identity
│   └── report_agent.py             # MODIFY: Use Identity, Gateway
│
└── requirements.txt                # MODIFY: Add AgentCore dependencies
```

---

## Configuration Changes

### New Environment Variables

```bash
# AgentCore Configuration
AGENTCORE_MEMORY_ID=mem-xxxxx
AGENTCORE_GATEWAY_ID=gw-xxxxx
AGENTCORE_IDENTITY_ARN=arn:aws:bedrock-agentcore:...
AGENTCORE_REGION=us-east-1

# Runtime Configuration
AGENTCORE_RUNTIME_ARN=arn:aws:bedrock-agentcore:...
AGENTCORE_EXECUTION_ROLE_ARN=arn:aws:iam::...

# Feature Flags
ENABLE_AGENTCORE_MEMORY=true
ENABLE_AGENTCORE_GATEWAY=true
ENABLE_AGENTCORE_IDENTITY=true
ENABLE_AGENTCORE_OBSERVABILITY=true
```

### Updated `config.yaml`

```yaml
# AgentCore Integration
agentcore:
  enabled: true
  memory:
    enabled: true
    memory_id: ${AGENTCORE_MEMORY_ID}
    strategies:
      - summaryMemoryStrategy
      - semanticMemoryStrategy
  gateway:
    enabled: true
    gateway_id: ${AGENTCORE_GATEWAY_ID}
  identity:
    enabled: true
    identity_arn: ${AGENTCORE_IDENTITY_ARN}
  observability:
    enabled: true
    tracing: true
    metrics: true
```

---

## Code Changes Summary

### 1. Orchestrator Changes

**File:** `agents/orchestrator.py`

**Changes:**
```python
# ADD: Memory Manager initialization
from agents.memory_manager import WAFRMemoryManager

class WafrOrchestrator:
    def __init__(self, ...):
        # ... existing code ...
        
        # ADD: Initialize Memory Manager
        if os.getenv("ENABLE_AGENTCORE_MEMORY") == "true":
            memory_id = os.getenv("AGENTCORE_MEMORY_ID")
            self.memory_manager = WAFRMemoryManager(memory_id)
        else:
            self.memory_manager = None
    
    def process_transcript(self, ...):
        # ADD: Retrieve context from memory
        if self.memory_manager:
            context = self.memory_manager.retrieve_assessment_context(
                actor_id=actor_id,
                session_id=session_id
            )
            # Use context to inform processing
        
        # ... existing processing ...
        
        # ADD: Store context in memory
        if self.memory_manager:
            self.memory_manager.store_assessment_context(
                actor_id=actor_id,
                session_id=session_id,
                transcript=transcript,
                insights=insights,
                mappings=mappings,
                gaps=gaps
            )
```

### 2. Review Orchestrator Changes

**File:** `agents/review_orchestrator.py`

**Changes:**
```python
# ADD: Memory integration for review decisions
from agents.memory_manager import WAFRMemoryManager

class ReviewOrchestrator:
    def __init__(self, ...):
        # ... existing code ...
        
        # ADD: Memory Manager
        if os.getenv("ENABLE_AGENTCORE_MEMORY") == "true":
            memory_id = os.getenv("AGENTCORE_MEMORY_ID")
            self.memory_manager = WAFRMemoryManager(memory_id)
    
    async def submit_review(self, ...):
        # ... existing review logic ...
        
        # ADD: Store review decision in memory
        if self.memory_manager:
            self.memory_manager.store_review_decision(
                actor_id=reviewer_id,
                session_id=session_id,
                question_id=item.question_id,
                decision=decision.value,
                feedback=feedback
            )
```

### 3. Answer Synthesis Agent Changes

**File:** `agents/answer_synthesis_agent.py`

**Changes:**
```python
# ADD: Use memory context for better synthesis
from agents.memory_manager import WAFRMemoryManager

class AnswerSynthesisAgent:
    def __init__(self, ...):
        # ... existing code ...
        
        # ADD: Memory Manager
        if os.getenv("ENABLE_AGENTCORE_MEMORY") == "true":
            memory_id = os.getenv("AGENTCORE_MEMORY_ID")
            self.memory_manager = WAFRMemoryManager(memory_id)
    
    def synthesize_gaps(self, ...):
        # ADD: Retrieve previous synthesis context
        if self.memory_manager:
            context = self.memory_manager.retrieve_assessment_context(
                actor_id=actor_id,
                session_id=session_id,
                query="previous answer synthesis patterns"
            )
            # Use context to improve synthesis
```

### 4. WA Tool Agent Changes

**File:** `agents/wa_tool_agent.py`

**Changes:**
```python
# ADD: Use Identity for AWS credentials
from agents.identity_manager import WAFRIdentityManager

class WAToolAgent:
    def __init__(self, ...):
        # ... existing code ...
        
        # ADD: Identity Manager
        if os.getenv("ENABLE_AGENTCORE_IDENTITY") == "true":
            identity_arn = os.getenv("AGENTCORE_IDENTITY_ARN")
            self.identity_manager = WAFRIdentityManager(identity_arn)
    
    def create_workload(self, ...):
        # REPLACE: Hardcoded credentials with Identity
        if self.identity_manager:
            credentials = self.identity_manager.get_aws_credentials(
                actor_id=actor_id,
                required_permissions=["wellarchitected:CreateWorkload"]
            )
            # Use credentials for AWS API calls
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_agentcore_integration.py` (NEW)

```python
import pytest
from agentcore_agent import invoke, health
from agents.memory_manager import WAFRMemoryManager
from agents.gateway_tools import WAFRGatewayManager

def test_agentcore_entrypoint():
    """Test AgentCore entrypoint"""
    payload = {
        "prompt": "Process this transcript",
        "transcript": "Test transcript",
        "actor_id": "test-user",
        "session_id": "test-session",
        "action": "process_transcript"
    }
    
    result = invoke(payload)
    assert "result" in result
    assert result["status"] == "completed"

def test_memory_integration():
    """Test Memory integration"""
    memory_manager = WAFRMemoryManager(memory_id="test-memory")
    
    # Store context
    event = memory_manager.store_assessment_context(
        actor_id="test-user",
        session_id="test-session",
        transcript="test",
        insights=[],
        mappings=[],
        gaps=[]
    )
    
    assert event is not None
    
    # Retrieve context
    context = memory_manager.retrieve_assessment_context(
        actor_id="test-user",
        session_id="test-session"
    )
    
    assert context is not None

def test_gateway_tools():
    """Test Gateway tool registration"""
    gateway_manager = WAFRGatewayManager(gateway_id="test-gateway")
    tools = gateway_manager.register_wafr_tools()
    
    assert len(tools) > 0
    assert any(tool["name"] == "extract_insights" for tool in tools)
```

### Integration Tests

**File:** `tests/test_agentcore_e2e.py` (NEW)

```python
import pytest
from agentcore_agent import invoke

def test_end_to_end_pipeline():
    """Test complete pipeline with AgentCore"""
    payload = {
        "prompt": "Process this transcript",
        "transcript": "Full transcript text...",
        "actor_id": "test-user",
        "session_id": "test-session",
        "action": "process_transcript"
    }
    
    result = invoke(payload)
    
    # Verify all steps completed
    assert result["status"] == "completed"
    assert "steps" in result
    assert "understanding" in result["steps"]
    assert "mapping" in result["steps"]
    assert "synthesis" in result["steps"]
    
    # Verify memory was used
    assert "memory_context" in result or "context_retrieved" in result
```

---

## Deployment Guide

### Step 1: Setup Infrastructure

```bash
# Install dependencies
pip install -r requirements.txt

# Setup AgentCore infrastructure
python setup_agentcore.py --setup-all

# This creates:
# - Memory resource
# - Gateway resource
# - Identity resource
# - Saves config to agentcore_config.json
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Update with values from agentcore_config.json
# AGENTCORE_MEMORY_ID=mem-xxxxx
# AGENTCORE_GATEWAY_ID=gw-xxxxx
# AGENTCORE_IDENTITY_ARN=arn:aws:...
```

### Step 3: Test Locally

```bash
# Test AgentCore Runtime locally
python agentcore_agent.py

# In another terminal, test invocation
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Process this transcript",
    "transcript": "Test transcript",
    "actor_id": "test-user",
    "session_id": "test-session",
    "action": "process_transcript"
  }'
```

### Step 4: Deploy to AWS

```bash
# Configure AgentCore
agentcore configure -e agentcore_agent.py

# Deploy
agentcore launch

# Test deployed agent
agentcore invoke '{
  "prompt": "Process this transcript",
  "transcript": "Full transcript...",
  "actor_id": "user-123",
  "session_id": "session-456",
  "action": "process_transcript"
}'
```

---

## Benefits of AgentCore Integration

### 1. Serverless Deployment
- ✅ Auto-scaling
- ✅ No infrastructure management
- ✅ Pay-per-use pricing
- ✅ High availability

### 2. Persistent Memory
- ✅ Context across sessions
- ✅ Learning from reviews
- ✅ Personalization
- ✅ Long-term knowledge

### 3. Tool Gateway
- ✅ Expose agents as tools
- ✅ Tool discovery
- ✅ Multi-agent collaboration
- ✅ Standardized interfaces

### 4. Secure Identity
- ✅ No hardcoded credentials
- ✅ Just-enough access
- ✅ Credential rotation
- ✅ Audit trail

### 5. Observability
- ✅ CloudWatch integration
- ✅ Distributed tracing
- ✅ Performance metrics
- ✅ Error tracking

---

## Migration Checklist

### Pre-Migration
- [ ] Review current architecture
- [ ] Identify integration points
- [ ] Plan testing strategy
- [ ] Set up AgentCore account

### Phase 1: Foundation
- [ ] Install AgentCore dependencies
- [ ] Create infrastructure resources
- [ ] Create basic entrypoint
- [ ] Test local deployment

### Phase 2: Memory
- [ ] Implement MemoryManager
- [ ] Integrate with Orchestrator
- [ ] Integrate with ReviewOrchestrator
- [ ] Test memory persistence

### Phase 3: Gateway
- [ ] Implement GatewayManager
- [ ] Register tools
- [ ] Update agent calls
- [ ] Test tool discovery

### Phase 4: Identity
- [ ] Implement IdentityManager
- [ ] Update WA Tool Agent
- [ ] Update Report Agent
- [ ] Test credential access

### Phase 5: Observability
- [ ] Add tracing hooks
- [ ] Add metrics logging
- [ ] Create dashboards
- [ ] Test observability

### Phase 6: Deployment
- [ ] Deploy to Runtime
- [ ] End-to-end testing
- [ ] Performance testing
- [ ] Production rollout

---

## Risk Mitigation

### Risk 1: Breaking Changes
**Mitigation:**
- Feature flags for gradual rollout
- Backward compatibility mode
- Comprehensive testing

### Risk 2: Performance Impact
**Mitigation:**
- Performance benchmarking
- Caching strategies
- Async operations

### Risk 3: Cost Increase
**Mitigation:**
- Monitor usage
- Optimize memory strategies
- Use appropriate instance sizes

### Risk 4: Migration Complexity
**Mitigation:**
- Phased approach
- Detailed documentation
- Rollback plan

---

## Success Metrics

### Technical Metrics
- ✅ Deployment success rate > 99%
- ✅ Average response time < 5s
- ✅ Memory retrieval time < 500ms
- ✅ Gateway tool latency < 200ms

### Business Metrics
- ✅ Zero downtime migration
- ✅ Cost reduction (vs. self-hosted)
- ✅ Improved scalability
- ✅ Better observability

---

## Next Steps

1. **Review this plan** with the team
2. **Prioritize phases** based on business needs
3. **Set up AgentCore account** and resources
4. **Start Phase 1** implementation
5. **Iterate** based on feedback

---

## Questions & Support

For questions about this integration plan:
- Review AgentCore documentation
- Check AWS Bedrock AgentCore samples
- Consult with AWS solutions architects

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-15  
**Status:** Ready for Implementation

