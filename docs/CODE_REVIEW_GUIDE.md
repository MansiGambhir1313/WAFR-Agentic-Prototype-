# WAFR Agent System - Code Review Guide

**Prepared for:** Code Review Meeting  
**Date:** January 6, 2026  
**System:** WAFR (Well-Architected Framework Review) Agent System  
**Author:** ML / NeuralEDGE

> **Note:** This document references the current project structure. See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for detailed system architecture and [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for file organization.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Core Components](#3-core-components)
4. [Agent Modules](#4-agent-modules)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [HITL Enhancement Plan](#6-hitl-enhancement-plan)
7. [AG-UI Event Protocol](#7-ag-ui-event-protocol)
8. [Code Quality Assessment](#8-code-quality-assessment)
9. [Improvement Recommendations](#9-improvement-recommendations)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. Executive Summary

### What is WAFR Agent System?

The WAFR Agent System is an **AI-powered automation tool** that conducts AWS Well-Architected Framework Reviews by:

1. **Processing workshop transcripts** to extract architecture insights
2. **Mapping insights** to WAFR questions automatically
3. **Generating answers** with confidence scores
4. **Producing official reports** and syncing to AWS WA Tool

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Agent Modules | 18+ |
| WAFR Pillars Covered | 6 (OPS, SEC, REL, PERF, COST, SUS) |
| Multi-Lens Support | Yes (GenAI, ML, Serverless, etc.) |
| Output Formats | PDF, AWS WA Tool, JSON |
| Confidence Threshold | 0.7 (strict quality control) |
| Expected Coverage | 30-70% (quality over quantity) |

### Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                          │
├─────────────────────────────────────────────────────────────┤
│  Framework      │ Strands Agent Framework                   │
│  LLM            │ AWS Bedrock (Claude 3.7 Sonnet)           │
│  AWS Services   │ WA Tool, Bedrock, S3                      │
│  Language       │ Python 3.11+                              │
│  Data Models    │ Pydantic + Dataclasses                    │
│  PDF Processing │ PyMuPDF (fitz)                            │
│  Event Streaming│ AG-UI Protocol (SSE/WebSocket)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WAFR AGENT SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   INPUTS     │
                              ├──────────────┤
                              │ • Transcript │
                              │ • PDF Docs   │
                              │ • Config     │
                              └──────┬───────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ORCHESTRATOR                                      │
│              (src/wafr/agents/orchestrator.py)                               │
│                                                                              │
│  Coordinates all agents, manages state, handles errors                       │
│  • HRI Validation (Claude-based filtering)                                   │
│  • Lens Detection (automatic from transcript)                               │
│  • Strict Quality Control (confidence >= 0.7)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  PROCESSING   │          │   ANALYSIS    │          │    OUTPUT     │
│    LAYER      │          │    LAYER      │          │    LAYER      │
├───────────────┤          ├───────────────┤          ├───────────────┤
│• PDF Processor│          │• Understanding│          │• Report Agent │
│• Input Process│          │• Mapping      │          │• WA Tool Agent│
│• Lens Manager │          │• Confidence   │          │• WA Tool Client│
│               │          │• Gap Detection│          │               │
│               │          │• Answer Synth │          │               │
│               │          │• Prompt Gen   │          │               │
└───────────────┘          └───────────────┘          └───────────────┘
        │                            │                            │
        └────────────────────────────┼────────────────────────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │   OUTPUTS    │
                              ├──────────────┤
                              │ • PDF Report │
                              │ • WA Workload│
                              │ • JSON Data  │
                              │ • Logs       │
                              └──────────────┘
```

### Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODULE DEPENDENCIES                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                    src/wafr/agents/config.py
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
        model_config.py    base_agent.py   wafr_context.py
                    │             │             │
                    └──────┬──────┴──────┬──────┘
                           │             │
                           ▼             ▼
                    ┌──────────────────────────┐
                    │      ALL AGENTS          │
                    │  (inherit patterns)      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      ORCHESTRATOR        │
                    │   (coordinates all)      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            scripts/run_wafr_full.py   ag_ui/server.py
```

---

## 3. Core Components

### 3.1 Configuration Module (`src/wafr/agents/config.py`)

**Purpose:** Central configuration hub for all settings.

**Key Settings:**
- **AWS Settings**: BEDROCK_REGION, DEFAULT_MODEL_ID, WA_TOOL_REGION
- **Processing Settings**: CHUNK_SIZE (5000), MAX_PARALLEL_WORKERS (6), RETRY_ATTEMPTS (3)
- **Confidence Thresholds**: HIGH (0.75), MEDIUM (0.50), LOW (0.25)
- **Strict Quality Control**: CONFIDENCE_THRESHOLD (0.7), expected coverage 30-70%
- **Lens Mappings**: LENS_ALIASES for normalization

**Key Points for Review:**
- All magic numbers centralized
- Environment-specific overrides supported
- Lens alias normalization handled here
- Strict quality control settings

---

### 3.2 Model Configuration (`src/wafr/agents/model_config.py`)

**Purpose:** LLM model initialization and management.

**Core Function:**
```python
def get_strands_model(model_id: str, max_tokens: int = 8192) -> Model:
    """
    Initialize Strands-compatible model with Bedrock.
    
    Flow:
    1. Validate model_id
    2. Create Bedrock client
    3. Wrap in Strands Model with max_tokens
    4. Return configured model
    """
```

**Key Features:**
- Default max_tokens: 8192 (increased for complex tasks)
- Graceful fallback if max_tokens not supported
- Model validation and error handling

---

### 3.3 WAFR Context Loader (`src/wafr/agents/wafr_context.py`)

**Purpose:** Load and provide WAFR schema to all agents.

**Loading Strategy:**
1. Try AWS API (create temp workload, fetch questions, delete workload)
2. Fallback to local file (`data/knowledge_base/` or `data/schemas/`)
3. Cache for reuse (TTL: 3600 seconds)

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `load_wafr_schema()` | Load schema from AWS API or file |
| `get_wafr_context_summary()` | Generate text summary for agents |
| `get_question_context()` | Get details for specific question |
| `get_pillar_questions_summary()` | Get all questions in a pillar |
| `refresh_aws_schema_cache()` | Force refresh cached schema |

---

### 3.4 Lens Manager (`src/wafr/agents/lens_manager.py`)

**Purpose:** Handle multi-lens WAFR reviews (GenAI, ML, Serverless, etc.)

**Lens Detection Flow:**
1. Analyze transcript for keywords/services
2. Calculate relevance scores per lens
3. Select top 3 lenses (plus base wellarchitected)
4. Validate lens access in AWS account
5. Return: active_lenses, skipped_lenses

**Supported Lenses:**

| Lens | Alias | Focus Area |
|------|-------|------------|
| Well-Architected | wellarchitected | Core framework (always included) |
| Generative AI | genai | LLMs, RAG, embeddings, Bedrock |
| Machine Learning | ml | SageMaker, ML pipelines |
| Serverless | serverless | Lambda, API Gateway, event-driven |
| SaaS | saas | Multi-tenancy, tenant isolation |
| Data Analytics | data-analytics | Data lakes, Redshift, Athena |
| Containers | containers | EKS, ECS, Docker, Kubernetes |

---

## 4. Agent Modules

### 4.1 Agent Architecture Pattern

All agents follow a consistent pattern:

```python
class SomeAgent:
    """Standard agent pattern used across the system."""
    
    def __init__(self, wafr_schema, lens_context):
        # 1. Store schema and context
        self.wafr_schema = wafr_schema
        self.lens_context = lens_context
        
        # 2. Initialize Strands agent
        self.agent = Agent(
            system_prompt=get_system_prompt(),
            name='SomeAgent',
            max_tokens=8192  # Increased for complex tasks
        )
        
        # 3. Setup Bedrock fallback
        self.bedrock_client = boto3.client('bedrock-runtime')
    
    def process(self, inputs, session_id) -> Dict:
        """Main entry point - returns structured results."""
        # Processing logic with error handling
        return {'session_id': session_id, 'results': [...]}
    
    def _call_bedrock_direct(self, prompt) -> str:
        """Fallback when Strands fails."""
        # Direct API call with retry logic
```

---

### 4.2 Understanding Agent (`src/wafr/agents/understanding_agent.py`)

**Purpose:** First agent - extracts architecture insights from transcripts.

**Processing Flow:**
1. Segment transcript into 5000-char chunks
2. Process chunks in parallel (ThreadPoolExecutor, 6 workers)
3. For each chunk: Call Claude with extraction prompt
4. Parse JSON response (4 fallback strategies)
5. Extract: services, decisions, constraints, risks, quotes
6. Deduplicate and merge insights

**Output Structure:**
```python
{
    "insight_type": "service|decision|constraint|risk",
    "content": "Uses Lambda for API",
    "transcript_quote": "We use Lambda...",
    "confidence": 1.0,
    "lens_relevance": ["serverless"]
}
```

**Multi-Strategy JSON Parsing:**
- Strategy 1: Direct JSON parse
- Strategy 2: Extract from dict/markdown
- Strategy 3: Regex extraction
- Strategy 4: Individual object extraction + combine

---

### 4.3 Mapping Agent (`src/wafr/agents/mapping_agent.py`)

**Purpose:** Maps extracted insights to specific WAFR questions.

**Matching Criteria:**
- Keyword overlap
- Pillar alignment
- Semantic similarity
- AWS service relevance

**Output Structure:**
```python
{
    "question_id": "SEC_02",
    "pillar_id": "SEC",
    "answer_content": "Based on discussion...",
    "source_insights": ["INS_001", "INS_003"],
    "mapping_confidence": 0.85
}
```

---

### 4.4 Confidence Agent (`src/wafr/agents/confidence_agent.py`)

**Purpose:** Validates evidence quality and assigns confidence scores.

**Confidence Calculation:**
```python
confidence = (
    direct_quote_presence * 0.40 +
    answer_specificity * 0.25 +
    multiple_source_support * 0.20 +
    aws_service_alignment * 0.15
)
```

**Confidence Levels:**
- HIGH (0.75-1.0): Strong evidence
- MEDIUM (0.50-0.74): Moderate evidence
- LOW (0.25-0.49): Weak evidence

**Key Features:**
- Batch processing with timeout (90s)
- Graceful degradation on timeout
- max_tokens increased to 8192
- Optimized prompt truncation

---

### 4.5 Gap Detection Agent (`src/wafr/agents/gap_detection_agent.py`)

**Purpose:** Identifies unanswered WAFR questions.

**Gap Types:**
- NOT_ADDRESSED: No answer at all
- LOW_CONFIDENCE: Answer exists but confidence < threshold
- PARTIAL: Answer incomplete

**Criticality Assignment:**
- HIGH: Security, Reliability core questions
- MEDIUM: Performance, Cost core questions
- LOW: Nice-to-have coverage

---

### 4.6 Answer Synthesis Agent (`src/wafr/agents/answer_synthesis_agent.py`)

**Purpose:** Generates AI answers for gap questions using evidence + inference.

**Synthesis Strategy:**
1. Gather context:
   - Transcript excerpts (priority: highest)
   - Related insights
   - Related answered questions
   - WAFR best practices
   - Inferred workload profile

2. AI Synthesis:
   - Combine evidence + inference
   - Generate reasoning chain
   - Identify assumptions
   - Assign confidence

3. Output:
   - Synthesized answer
   - Reasoning chain
   - Assumptions list
   - Confidence score
   - Requires attention flags

**Confidence Formula:**
```python
confidence = (
    evidence_strength * 0.40 +
    assumption_factor * 0.25 +
    context_richness * 0.20 +
    best_practice_alignment * 0.15
)
```

---

### 4.7 WA Tool Agent (`src/wafr/agents/wa_tool_agent.py`)

**Purpose:** AWS Well-Architected Tool integration with optimized batching.

**Optimization Strategy:**
1. **Preprocess Transcript**: ONE AI call to extract facts organized by pillar
2. **Get All Questions**: Fetch from WA Tool API
3. **Process Pillars in Parallel**: ThreadPoolExecutor (6 workers)
   - ONE batch AI call per pillar
   - Answer all questions in pillar together
4. **Create Milestone & Report**: Generate official PDF

**Performance Impact:**
- Traditional: 200 questions × 1 call = 200 API calls (~7 min)
- Optimized: 1 preprocessing + 6 pillar batches = 7 calls (~32 sec)
- **Speedup: ~12x faster**

**Key Features:**
- Strict confidence threshold (0.7)
- HRI validation (Claude-based filtering)
- Honest assessment (no forced answers)
- Expected coverage: 30-70%

---

### 4.8 Orchestrator (`src/wafr/agents/orchestrator.py`)

**Purpose:** Main coordinator - runs the entire pipeline.

**Pipeline Steps:**
1. PDF Processing (optional)
2. Understanding Agent - Extract insights
3. Mapping Agent - Map to WAFR questions
4. Confidence Agent - Validate evidence
5. Gap Detection Agent - Identify gaps
6. Answer Synthesis Agent - Generate AI answers
7. Auto-Populate - Merge validated + synthesized
8. Prompt Generator Agent - Generate prompts
9. Scoring Agent - Grade answers
10. Report Agent (optional) - Generate PDF
11. WA Tool Agent (optional) - Create workload + PDF

**Key Features:**
- HRI Validation: Filters non-tangible HRIs using Claude
- Lens Detection: Automatic from transcript
- Strict Quality Control: Confidence >= 0.7
- Error Handling: Graceful degradation at each step

---

## 5. Data Flow Diagrams

### 5.1 Complete System Data Flow

```
Transcript Input
    │
    ▼
[Understanding Agent] ──► Insights: [{type, content, quote, confidence}]
    │
    ▼
[Mapping Agent] ──► Mappings: [{question_id, answer_content, confidence}]
    │
    ▼
[Confidence Agent] ──► Validated: [{question_id, answer, confidence_score}]
    │
    ▼
[Gap Detection Agent] ──► Gaps: [{question_id, gap_type, criticality}]
    │
    ▼
[Answer Synthesis Agent] ──► Synthesized: [{question_id, answer, reasoning}]
    │
    ▼
[Auto-Populate] ──► All Answers: {validated + synthesized}
    │
    ▼
[Scoring Agent] ──► Scored: [{question_id, score, risk_level}]
    │
    ├──► [Report Agent] ──► PDF Report (output/reports/)
    └──► [WA Tool Agent] ──► AWS Workload + Official PDF
```

---

## 6. HITL Enhancement Plan

### 6.1 HITL Pipeline Overview

The system includes Human-in-the-Loop (HITL) workflow for answer validation:

1. **Answer Synthesis**: AI generates answers for gaps
2. **Checkpoint 1**: Human validation
   - High Confidence (≥0.75): Auto-approve eligible
   - Medium Confidence: Quick review
   - Low Confidence (<0.50): Detailed review
3. **Re-synthesis**: Rejected answers re-synthesized with feedback (max 2 attempts)
4. **Checkpoint 2**: Final approval before report generation

---

## 7. AG-UI Event Protocol

### 7.1 Event Architecture

**Location:** `src/wafr/ag_ui/`

**Components:**
- `emitter.py`: Event emission
- `state.py`: State management
- `orchestrator_integration.py`: Orchestrator wrapper
- `server.py`: FastAPI SSE server

**Event Types:**
- Lifecycle: RUN_STARTED, RUN_FINISHED, RUN_ERROR
- Streaming: TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END
- State: STATE_SNAPSHOT, STATE_DELTA
- HITL: SYNTHESIS_STARTED, CHECKPOINT_REACHED, AWAITING_HUMAN_INPUT

---

## 8. Code Quality Assessment

### 8.1 Strengths

✅ **Architecture**
- Clean separation of concerns
- Consistent agent pattern
- Centralized configuration
- Multi-lens support built-in

✅ **Performance**
- Parallel processing (ThreadPoolExecutor)
- Batch API calls (12x speedup)
- Caching (WAFR schema, lens context)
- Connection pooling

✅ **Reliability**
- Multi-strategy JSON parsing (4 fallback methods)
- Graceful degradation (Strands → Bedrock fallback)
- Retry with exponential backoff
- Comprehensive error handling

✅ **Quality Control**
- Strict confidence threshold (0.7)
- HRI validation (Claude-based)
- Honest assessment (no forced answers)
- Expected coverage: 30-70%

---

## 9. Improvement Recommendations

### 9.1 High Priority

1. **Standardize Data Models**: Use Pydantic throughout
2. **Add Storage Abstraction**: Replace in-memory with database
3. **Align Async/Sync Patterns**: Async throughout for HITL
4. **Add Input Validation**: Pydantic validation at entry points

### 9.2 Medium Priority

5. **Add Comprehensive Error Types**: Custom exception hierarchy
6. **Add Configuration Validation**: Pydantic config models
7. **Add Metrics and Monitoring**: Prometheus metrics
8. **Add Batch Processing for Synthesis**: Similar to WA Tool Agent

---

## 10. Quick Reference Cheat Sheet

### 10.1 Agent Summary Table

| Agent | Input | Output | Key Function | File Path |
|-------|-------|--------|--------------|-----------|
| Understanding | Transcript | Insights | Extract architecture info | `src/wafr/agents/understanding_agent.py` |
| Mapping | Insights + Schema | Q&A Mappings | Match to WAFR questions | `src/wafr/agents/mapping_agent.py` |
| Confidence | Mappings | Validated Answers | Score evidence quality | `src/wafr/agents/confidence_agent.py` |
| Gap Detection | Answers + Schema | Gaps | Find unanswered questions | `src/wafr/agents/gap_detection_agent.py` |
| Answer Synthesis | Gaps + Context | Synthesized Answers | AI-generate answers | `src/wafr/agents/answer_synthesis_agent.py` |
| Scoring | All Answers | Scored Answers | Grade answer quality | `src/wafr/agents/scoring_agent.py` |
| Report | All Data | PDF | Generate report | `src/wafr/agents/report_agent.py` |
| WA Tool | All Data | AWS Workload | Sync to AWS | `src/wafr/agents/wa_tool_agent.py` |

### 10.2 Key Data Structures

```python
# Insight (from Understanding Agent)
{
    "insight_type": "service|decision|constraint|risk",
    "content": "Description of insight",
    "transcript_quote": "Exact quote from transcript",
    "confidence": 1.0,
    "lens_relevance": ["genai", "serverless"]
}

# Synthesized Answer (from Answer Synthesis Agent)
{
    "question_id": "SEC_02",
    "synthesized_answer": "AI-generated answer...",
    "reasoning_chain": ["Step 1...", "Step 2..."],
    "assumptions": ["Assumption 1", "Assumption 2"],
    "confidence": 0.65,
    "synthesis_method": "INFERENCE",
    "requires_attention": ["Verify MFA policy"]
}
```

### 10.3 Configuration Quick Reference

```python
# Key Configuration Values (src/wafr/agents/config.py)
BEDROCK_REGION = "us-east-1"
DEFAULT_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
CHUNK_SIZE = 5000  # Characters per transcript segment
MAX_PARALLEL_WORKERS = 6  # Pillar processing threads

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.7  # Strict quality control
HIGH_CONFIDENCE = 0.75   # Auto-approve eligible
MEDIUM_CONFIDENCE = 0.50  # Quick review
LOW_CONFIDENCE = 0.25    # Detailed review

# Expected Coverage: 30-70% (quality over quantity)
```

### 10.4 Common CLI Commands

```bash
# Run WAFR analysis (from project root)
python scripts/run_wafr_full.py --wa-tool --client-name "ClientName"

# With custom transcript
python scripts/run_wafr_full.py \
    --transcript data/transcripts/my_transcript.txt \
    --wa-tool \
    --client-name "Client Name"

# Without PDF generation
python scripts/run_wafr_full.py --no-report

# Save AG-UI events
python scripts/run_wafr_full.py \
    --wa-tool \
    --client-name "Client" \
    --output-events output/events.jsonl
```

### 10.5 Troubleshooting Quick Fixes

| Issue | Solution |
|-------|----------|
| Strands init fails | Falls back to direct Bedrock |
| JSON parsing fails | Uses 4 fallback strategies |
| Lens access denied | Retries with accessible lenses |
| WA Tool timeout | Exponential backoff (1s, 2s, 4s, 8s, 16s) |
| Synthesis fails | Creates low-confidence fallback |
| Too many HRIs | HRI validation filters non-tangible (60-80% reduction) |

---

## Appendix: File Structure

```
WAFR-prototype/
├── src/wafr/                    # Main WAFR package
│   ├── agents/                  # AI agent implementations
│   │   ├── orchestrator.py      # Main pipeline coordinator
│   │   ├── understanding_agent.py
│   │   ├── mapping_agent.py
│   │   ├── confidence_agent.py
│   │   ├── gap_detection_agent.py
│   │   ├── answer_synthesis_agent.py
│   │   ├── scoring_agent.py
│   │   ├── report_agent.py
│   │   ├── wa_tool_agent.py
│   │   ├── wa_tool_client.py
│   │   ├── config.py
│   │   ├── model_config.py
│   │   ├── wafr_context.py
│   │   └── ...
│   ├── ag_ui/                   # AG-UI integration
│   │   ├── emitter.py
│   │   ├── state.py
│   │   ├── orchestrator_integration.py
│   │   └── server.py
│   ├── models/                  # Data models
│   │   ├── synthesized_answer.py
│   │   └── review_item.py
│   └── storage/                 # Storage layer
│       └── review_storage.py
├── scripts/                     # Executable scripts
│   ├── run_wafr_full.py        # Main runner script
│   ├── generate_pdf_from_results.py
│   └── ...
├── docs/                        # Documentation
│   ├── SYSTEM_DESIGN.md        # System architecture
│   ├── CODE_REVIEW_GUIDE.md    # This file
│   └── PROJECT_STRUCTURE.md    # File organization
├── data/                        # Data files
│   ├── knowledge_base/         # Knowledge base JSON files
│   ├── schemas/                # JSON schemas
│   └── transcripts/            # Sample transcripts
├── output/                      # Generated outputs
│   ├── logs/                   # Log files
│   ├── reports/                # PDF reports
│   └── results/                # JSON results
└── config/                      # Configuration files
    ├── requirements.txt
    └── setup.py
```

---

## Related Documentation

- **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)**: Comprehensive system architecture and design
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Project file organization guide
- **[README.md](../README.md)**: Project overview and quick start

---

## Document Version

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.0 | 2026-01-06 | ML | Updated for new project structure (src/wafr/) |
| 1.0 | 2026-01-06 | ML | Initial creation for code review |

---

*End of Document*

