# WAFR System Design - Brief Overview

**Version:** 1.0  
**Date:** January 2026  
**System:** Well-Architected Framework Review (WAFR) Agent System

> **Related Documentation:**
> - [CODE_REVIEW_GUIDE.md](CODE_REVIEW_GUIDE.md) - Detailed code review guide with agent specifications
> - [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Project file organization and structure

---

## 1. Executive Summary

The WAFR system is an **AI-powered multi-agent system** that automates AWS Well-Architected Framework Reviews by analyzing workshop transcripts and generating comprehensive assessments. The system uses Amazon Bedrock (Claude Sonnet) to process natural language through a coordinated pipeline of specialized agents.

### Key Capabilities

- **Automated Transcript Analysis**: Extracts architecture insights from workshop transcripts
- **Intelligent Question Mapping**: Maps insights to WAFR questions automatically
- **Evidence Validation**: Validates evidence quality with confidence scoring
- **Gap Detection**: Identifies unanswered questions
- **AI Answer Synthesis**: Generates answers for gaps using inference + evidence
- **HRI Validation**: Filters non-tangible high-risk issues using Claude
- **Multi-Lens Support**: Supports GenAI, Serverless, ML, and other specialized lenses
- **AWS Integration**: Creates workloads and generates official PDF reports via WA Tool

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WAFR SYSTEM ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

INPUT LAYER
    │
    ├── Transcript (text)
    ├── PDF Documents (optional)
    └── Configuration
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                                  │
│              (WafrOrchestrator)                                  │
│  • Coordinates pipeline execution                                │
│  • Manages state and error handling                              │
│  • Handles HITL checkpoints                                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ├──► Understanding Agent ────► Extract insights from transcript
    │
    ├──► Mapping Agent ──────────► Map insights to WAFR questions
    │
    ├──► Confidence Agent ───────► Validate evidence & assign scores
    │
    ├──► Gap Detection Agent ────► Identify unanswered questions
    │
    ├──► Answer Synthesis Agent ─► Generate AI answers for gaps
    │
    ├──► Scoring Agent ──────────► Grade all answers
    │
    ├──► Report Agent ───────────► Generate PDF report
    │
    └──► WA Tool Agent ──────────► Create workload & sync to AWS
    │
    ▼
OUTPUT LAYER
    │
    ├── PDF Report (output/reports/)
    ├── JSON Results (output/results/)
    ├── AWS WA Tool Workload
    └── Logs (output/logs/)
```

### 2.2 Component Organization

```
src/wafr/
├── agents/              # Core AI agents
│   ├── orchestrator.py  # Main pipeline coordinator
│   ├── understanding_agent.py
│   ├── mapping_agent.py
│   ├── confidence_agent.py
│   ├── gap_detection_agent.py
│   ├── answer_synthesis_agent.py
│   ├── scoring_agent.py
│   ├── report_agent.py
│   ├── wa_tool_agent.py
│   └── ...
├── ag_ui/               # AG-UI event streaming
│   ├── emitter.py
│   ├── state.py
│   └── orchestrator_integration.py
├── models/              # Data models
│   ├── synthesized_answer.py
│   └── review_item.py
└── storage/             # Storage abstraction
    └── review_storage.py
```

---

## 3. Core Processing Pipeline

### 3.1 Pipeline Flow

```
Step 0: PDF Processing (Optional)
    ├── Extract text from PDF attachments
    └── Combine with transcript

Step 1: Understanding Agent
    ├── Segment transcript (5000 char chunks)
    ├── Process in parallel (ThreadPoolExecutor)
    ├── Extract insights (services, decisions, risks, best practices)
    └── Output: List of structured insights

Step 2: Mapping Agent
    ├── For each insight, find matching WAFR questions
    ├── Generate answer draft from insight
    └── Output: Question-answer mappings with confidence

Step 3: Confidence Agent
    ├── Validate evidence quality (direct quotes, specificity, etc.)
    ├── Calculate confidence scores (0.0-1.0)
    └── Output: Validated answers with confidence levels

Step 4: Gap Detection Agent
    ├── Compare all questions vs answered questions
    ├── Identify gaps (NOT_ADDRESSED, LOW_CONFIDENCE, PARTIAL)
    └── Output: Gap list with criticality

Step 5: Answer Synthesis Agent (HITL)
    ├── Generate AI answers for gaps
    ├── Combine transcript evidence + logical inference
    ├── Assign confidence scores
    └── Output: Synthesized answers with reasoning chains

Step 6: Auto-Populate
    ├── Merge validated + synthesized answers
    └── Output: Complete answer set

Step 7: Scoring Agent
    ├── Grade all answers
    └── Output: Scored answers with risk levels

Step 8: Report Agent 
    ├── Generate PDF report
    └── Output: PDF file (output/reports/)

Step 9: WA Tool Agent 
    ├── Create workload in AWS
    ├── Populate answers (optimized batching)
    ├── HRI Validation (filter non-tangible HRIs)
    ├── Create milestone
    └── Generate official PDF report
```

### 3.2 Key Processing Logic

#### Understanding Agent Logic

```python
# Parallel Processing Strategy
1. Segment transcript into 5000-char chunks
2. Process chunks in parallel (ThreadPoolExecutor)
3. For each chunk:
   - Call Claude with extraction prompt
   - Parse JSON response (4 fallback strategies)
   - Extract: services, decisions, constraints, risks, quotes
4. Deduplicate and merge insights
5. Return structured insight list
```

#### Mapping Agent Logic

```python
# Matching Strategy
For each insight:
  1. Find matching WAFR questions using:
     - Keyword overlap
     - Pillar alignment
     - Semantic similarity
     - AWS service relevance
  2. Generate answer draft from insight
  3. Calculate mapping confidence (0.0-1.0)
  4. Return question-answer mappings
```

#### Confidence Agent Logic

```python
# Confidence Calculation
confidence = (
    direct_quote_presence * 0.40 +
    answer_specificity * 0.25 +
    multiple_source_support * 0.20 +
    aws_service_alignment * 0.15
)

# Confidence Levels
HIGH:   0.75-1.0  (Strong evidence)
MEDIUM: 0.50-0.74 (Moderate evidence)
LOW:    0.25-0.49 (Weak evidence)
```

#### Answer Synthesis Logic

```python
# Synthesis Strategy
1. Gather context:
   - Transcript excerpts
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
```

#### WA Tool Agent Optimization

```python
# Batch Processing Strategy
1. Preprocess transcript (ONE AI call):
   - Extract facts organized by pillar
   - Result: {SEC: [facts], REL: [facts], ...}

2. Process pillars in parallel (ThreadPoolExecutor):
   - For each pillar: ONE batch AI call
   - Answer all questions in pillar together
   - Result: 6 API calls instead of 200+

3. Speedup: ~12x faster than individual calls
```

---

## 4. Key Concepts & Algorithms

### 4.1 Lens Detection

**Purpose**: Automatically identify relevant AWS Well-Architected Lenses from transcript

**Algorithm**:
```python
1. Analyze transcript for keywords/services:
   - GenAI: "LLM", "RAG", "Bedrock", "embeddings"
   - Serverless: "Lambda", "API Gateway", "event-driven"
   - ML: "SageMaker", "ML pipeline", "model training"

2. Calculate relevance scores per lens

3. Select top 3 lenses (plus base wellarchitected)

4. Validate lens access in AWS account

5. Return: active_lenses, skipped_lenses
```

### 4.2 HRI Validation

**Purpose**: Filter non-tangible high-risk issues using Claude validation

**Algorithm**:
```python
1. Extract potential HRIs from WA Tool

2. For each HRI, validate with Claude:
   - Check for clear evidence in transcript
   - Verify it's actionable (not theoretical)
   - Confirm it's a real risk (not false positive)

3. Filter out non-tangible HRIs

4. Return: validated HRIs only

Result: 60-80% reduction in HRI count
```

### 4.3 Confidence Threshold Management

**Current Strategy**: Strict quality control

```python
# Confidence Threshold: 0.7 (strict)
# Expected Coverage: 30-70%

Logic:
- Only answer questions with confidence >= 0.7
- Skip questions with insufficient evidence
- Honest assessment (no forced answers)
- Focus on quality over quantity
```

### 4.4 Multi-Strategy JSON Parsing

**Purpose**: Robust parsing of LLM JSON responses

**Strategy Chain**:
```python
1. Direct JSON parse
   └─► Success? Return
       └─► Fail? Try Strategy 2

2. Extract from dict/markdown
   └─► Success? Return
       └─► Fail? Try Strategy 3

3. Regex extraction
   └─► Success? Return
       └─► Fail? Try Strategy 4

4. Individual object extraction
   └─► Combine and return
```

---

## 5. Data Flow

### 5.1 End-to-End Data Flow

```
Transcript Input
    │
    ▼
[Understanding Agent]
    │
    └─► Insights: [{type, content, quote, confidence, lens_relevance}]
    │
    ▼
[Mapping Agent]
    │
    └─► Mappings: [{question_id, answer_content, mapping_confidence}]
    │
    ▼
[Confidence Agent]
    │
    └─► Validated Answers: [{question_id, answer, confidence_score}]
    │
    ▼
[Gap Detection Agent]
    │
    └─► Gaps: [{question_id, gap_type, criticality}]
    │
    ▼
[Answer Synthesis Agent]
    │
    └─► Synthesized: [{question_id, answer, reasoning_chain, confidence}]
    │
    ▼
[Auto-Populate]
    │
    └─► All Answers: {validated + synthesized}
    │
    ▼
[Scoring Agent]
    │
    └─► Scored Answers: [{question_id, score, risk_level}]
    │
    ▼
[Report Agent] ──► PDF Report
[WA Tool Agent] ──► AWS Workload + Official PDF
```

### 5.2 State Management

```python
# Session State
{
    "session_id": "wafr-abc123",
    "status": "processing|completed|error",
    "steps": {
        "understanding": {...},
        "mapping": {...},
        "confidence": {...},
        ...
    },
    "outputs": {
        "report_path": "output/reports/...",
        "workload_id": "...",
        "console_url": "..."
    }
}
```

---

## 6. Integration Points

### 6.1 AWS Services

**Amazon Bedrock**:
- Model: Claude 3.7 Sonnet
- Usage: All agent LLM calls
- Region: Configurable (default: us-east-1)

**AWS Well-Architected Tool**:
- Create workloads
- Populate answers
- Generate official PDF reports
- List lenses and questions

**AWS S3** (Optional):
- Store PDF attachments
- Store generated reports

### 6.2 AG-UI Integration

**Purpose**: Real-time event streaming for frontend applications

**Event Types**:
- `RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR`
- `STEP_STARTED`, `STEP_FINISHED`
- `TEXT_MESSAGE_CONTENT` (streaming updates)
- `STATE_SNAPSHOT`, `STATE_DELTA`
- `HITL` events (review checkpoints)

**Transport**: Server-Sent Events (SSE) or WebSocket

---

## 7. Error Handling & Resilience

### 7.1 Error Handling Strategy

```python
# Multi-Level Error Handling

1. Agent Level:
   - Try-catch around agent calls
   - Return empty/default results on failure
   - Log errors with context

2. Orchestrator Level:
   - Graceful degradation
   - Continue pipeline with partial results
   - Mark steps as "error" but continue

3. System Level:
   - Retry with exponential backoff
   - Fallback mechanisms (Strands → Bedrock direct)
   - Timeout management (90s default)
```

### 7.2 Resilience Features

- **JSON Parsing**: 4 fallback strategies
- **Model Failures**: Strands → Bedrock direct fallback
- **Timeout Handling**: Graceful degradation with defaults
- **Lens Access Errors**: Retry with accessible lenses only
- **WA Tool Timeouts**: Exponential backoff (1s, 2s, 4s, 8s, 16s)

---

## 8. Performance Optimizations

### 8.1 Parallel Processing

```python
# Understanding Agent
- Process transcript chunks in parallel
- ThreadPoolExecutor with 6 workers

# WA Tool Agent
- Process pillars in parallel
- ThreadPoolExecutor with 6 workers
- Batch answer questions per pillar
```

### 8.2 Caching

```python
# WAFR Schema Cache
- Cache duration: 3600 seconds
- Module-level cache for persistence

# Question Cache (WA Tool)
- Cache question details per workload
- Key: (workload_id, lens_alias, question_id)
```

### 8.3 Batch Processing

```python
# WA Tool Answer Population
Traditional: 200 questions × 1 call = 200 API calls (~7 min)
Optimized:   1 preprocessing + 6 pillar batches = 7 calls (~32 sec)
Speedup:     ~12x faster
```

---

## 9. Configuration & Deployment

### 9.1 Key Configuration

```python
# src/wafr/agents/config.py

BEDROCK_REGION = "us-east-1"
DEFAULT_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
CHUNK_SIZE = 5000  # Transcript segmentation
MAX_PARALLEL_WORKERS = 6  # Parallel processing
CONFIDENCE_THRESHOLD = 0.7  # Strict quality control
```

### 9.2 File Structure

```
WAFR-prototype/
├── src/wafr/          # Source code
│   ├── agents/        # AI agents
│   ├── ag_ui/         # AG-UI integration
│   ├── models/        # Data models
│   └── storage/       # Storage layer
├── scripts/            # Executable scripts
├── docs/              # Documentation
├── data/              # Data files
│   ├── knowledge_base/
│   ├── schemas/
│   └── transcripts/
├── output/            # Generated outputs
│   ├── logs/
│   ├── reports/
│   └── results/
└── config/            # Configuration
```

### 9.3 Running the System

```bash
# Main entry point
python scripts/run_wafr_full.py \
    --wa-tool \
    --client-name "Client Name"

# With custom transcript
python scripts/run_wafr_full.py \
    --transcript data/transcripts/my_transcript.txt \
    --wa-tool \
    --client-name "Client Name"
```

---

## 10. Key Design Decisions

### 10.1 Multi-Agent Architecture

**Decision**: Use specialized agents instead of monolithic system

**Rationale**:
- Single Responsibility Principle
- Easier testing and maintenance
- Better error isolation
- Modular extensibility

### 10.2 Confidence-Based Quality Control

**Decision**: Strict confidence threshold (0.7) with 30-70% expected coverage

**Rationale**:
- Quality over quantity
- Honest assessment (no forced answers)
- Reduces false positives
- Focuses on actionable insights

### 10.3 HRI Validation

**Decision**: Use Claude to validate HRIs after extraction

**Rationale**:
- Filters non-tangible HRIs
- Reduces false positives
- Ensures only actionable risks are reported
- 60-80% reduction in HRI count

### 10.4 Batch Processing in WA Tool

**Decision**: Batch answer questions per pillar instead of individual calls

**Rationale**:
- 12x performance improvement
- Reduces API costs
- Faster user experience
- Maintains answer quality

---

## 11. Future Enhancements

### 11.1 Planned Features

- **Distributed Processing**: Support for distributed agent execution
- **Advanced Analytics**: Review analytics and confidence calibration
- **Plugin Architecture**: Support for custom agent plugins
- **Enhanced HITL**: More sophisticated review workflows

### 11.2 Scalability Improvements

- **Database Storage**: Replace in-memory storage with DynamoDB
- **Async Throughout**: Full async/await support
- **Caching Layer**: Redis for session and schema caching
- **API Gateway**: RESTful API for external integrations

---

## 12. Summary

The WAFR system is a **sophisticated multi-agent AI system** that automates AWS Well-Architected Framework Reviews through:

1. **Intelligent Analysis**: Extracts insights from natural language transcripts
2. **Quality Control**: Strict confidence thresholds ensure high-quality outputs
3. **Optimization**: Batch processing and parallel execution for performance
4. **Resilience**: Multi-level error handling and graceful degradation
5. **Integration**: Seamless AWS WA Tool integration with official PDF generation
6. **Validation**: HRI validation ensures only tangible risks are reported

The system is designed for **production use** with comprehensive error handling, performance optimizations, and maintainable architecture.

---

## 13. Related Documentation

### 13.1 Code Review Guide

For detailed code review information, see **[CODE_REVIEW_GUIDE.md](CODE_REVIEW_GUIDE.md)** which includes:

- Detailed agent module specifications
- Complete data flow diagrams
- HITL enhancement plan details
- AG-UI event protocol documentation
- Code quality assessment
- Improvement recommendations
- Quick reference cheat sheet

### 13.2 Project Structure

For file organization details, see **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** which includes:

- Complete directory structure
- Package organization
- Import patterns
- Migration notes

---

**Document Version**: 1.0  
**Last Updated**: January 2026

**Related Documents:**
- [CODE_REVIEW_GUIDE.md](CODE_REVIEW_GUIDE.md) - Comprehensive code review guide
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Project organization guide
