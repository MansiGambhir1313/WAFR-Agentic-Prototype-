# WAFR System Design Document

## Overview

The WAFR (Well-Architected Framework Review) system is an AI-powered multi-agent system that automatically analyzes workshop transcripts and generates comprehensive AWS Well-Architected Framework assessments. The system uses Amazon Bedrock (Claude Sonnet) to process transcripts through a pipeline of specialized AI agents.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Transcript    │
│   Input File    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      Orchestrator                    │
│  (WafrOrchestrator)                  │
│  - Coordinates agent pipeline        │
│  - Manages workflow                  │
│  - Handles errors gracefully         │
└────────┬─────────────────────────────┘
         │
         ├──► Understanding Agent ────► Extract insights
         │
         ├──► Mapping Agent ──────────► Map to WAFR questions
         │
         ├──► Confidence Agent ───────► Validate evidence
         │
         ├──► Gap Detection Agent ────► Find unanswered questions
         │
         ├──► Prompt Generator ───────► Generate follow-up prompts
         │
         ├──► Scoring Agent ───────────► Grade answers
         │
         ├──► Report Agent ────────────► Generate PDF report
         │
         └──► WA Tool Agent ──────────► Integrate with AWS WA Tool
```


### Core Components

#### 1. Orchestrator (`agents/orchestrator.py`)

**Purpose**: Central coordinator that manages the multi-agent pipeline

**Responsibilities**:
- Initialize all agents with WAFR schema context
- Execute agents in the correct sequence
- Pass data between agents
- Handle errors gracefully (graceful degradation)
- Track processing time and status
- Generate final results summary

**Key Methods**:
- `process_transcript()`: Main entry point for processing
- `_extract_validated_answers()`: Extract validated answers from mappings
- `_prompt_user_for_remaining_questions()`: Interactive prompt for gaps
- `_manual_answer_questions()`: Manual question answering interface

#### 2. Understanding Agent (`agents/understanding_agent.py`)

**Purpose**: Extract architecture-relevant insights from transcripts

**Process**:
1. Segment long transcripts into processable chunks (5000 chars)
2. Process segments in parallel batches
3. Extract insights using AI (Claude Sonnet)
4. Validate and deduplicate insights

**Insight Types**:
- **Decision**: Architecture choices, technology selections
- **Service**: AWS services mentioned with usage context
- **Constraint**: Requirements, limitations, compliance needs
- **Risk**: Security, availability, scalability concerns

**Output**: List of structured insights with transcript quotes

#### 3. Mapping Agent (`agents/mapping_agent.py`)

**Purpose**: Map extracted insights to specific WAFR questions

**Process**:
1. For each insight, find relevant WAFR questions
2. Calculate relevance scores (0.0-1.0)
3. Determine answer coverage (complete/partial)
4. Synthesize answer content from transcript
5. Extract evidence quotes

**Mapping Criteria**:
- Keyword matching
- Question intent alignment
- Pillar relevance (OPS, SEC, REL, PERF, COST, SUS)
- Criticality weighting

**Output**: Mappings with relevance scores and answer content

#### 4. Confidence Agent (`agents/confidence_agent.py`)

**Purpose**: Validate evidence and prevent hallucinations

**Validation Process**:
1. Verify evidence quotes exist in transcript (verbatim or similar)
2. Check if evidence supports the answer
3. Assess interpretation accuracy
4. Identify unsupported assumptions
5. Assign confidence scores (HIGH/MEDIUM/LOW)

**Confidence Levels**:
- **HIGH (0.75-1.0)**: Quote found verbatim, direct support
- **MEDIUM (0.5-0.74)**: Quote found with similarity, reasonable interpretation
- **LOW (0.0-0.49)**: Quote not found or significant inference required

**Output**: Validated answers with confidence scores

#### 5. Gap Detection Agent (`agents/gap_detection_agent.py`)

**Purpose**: Identify unanswered WAFR questions

**Process**:
1. Compare answered questions to all WAFR questions
2. Identify gaps (unanswered questions)
3. Prioritize gaps by criticality
4. Calculate pillar coverage percentages

**Output**: List of gaps with priority scores

#### 6. Prompt Generator Agent (`agents/prompt_generator_agent.py`)

**Purpose**: Generate smart prompts for unanswered questions

**Process**:
1. Analyze gap question context
2. Generate contextual prompts based on transcript
3. Create actionable follow-up questions

**Output**: Smart prompts for each gap

#### 7. Scoring Agent (`agents/scoring_agent.py`)

**Purpose**: Grade answers on multiple dimensions

**Scoring Dimensions**:
- **Confidence (40% weight)**: Evidence quality and verification
- **Completeness (30% weight)**: How well answer addresses question
- **Compliance (30% weight)**: Alignment with WAFR best practices

**Grade Assignment**:
- **A (90-100)**: Excellent, fully compliant
- **B (80-89)**: Good, minor gaps
- **C (70-79)**: Adequate, some improvements needed
- **D (60-69)**: Needs significant work
- **F (<60)**: Critical gaps, non-compliant

**Output**: Scored answers with letter grades

#### 8. Report Agent (`agents/report_agent.py`)

**Purpose**: Generate comprehensive PDF reports

**Report Sections**:
1. **Executive Summary**: Overall health, key findings, immediate actions
2. **Pillar-by-Pillar Analysis**: Current state, strengths, gaps for each pillar
3. **High-Risk Issues (HRIs)**: Critical issues requiring attention
4. **90-Day Remediation Roadmap**: Phased action plan
5. **Appendix**: Evidence citations, confidence scores

**Output**: Professional PDF report

#### 9. WA Tool Agent (`agents/wa_tool_agent.py`)

**Purpose**: Integrate with AWS Well-Architected Tool API

**Capabilities**:
- Create workloads in WA Tool
- Auto-populate answers from transcript analysis
- Create milestones and generate official reports
- Interactive question answering

**Output**: Workload ID, populated answers, official WA Tool report

## Data Flow

### Processing Pipeline

```
1. Transcript Input
   ↓
2. Understanding Agent
   - Extract insights
   - Output: List of insights
   ↓
3. Mapping Agent
   - Map insights to WAFR questions
   - Output: List of mappings
   ↓
4. Confidence Agent
   - Validate evidence
   - Output: Validated answers
   ↓
5. Gap Detection Agent
   - Find unanswered questions
   - Output: List of gaps
   ↓
6. Prompt Generator Agent
   - Generate prompts for gaps
   - Output: Smart prompts
   ↓
7. Scoring Agent
   - Grade answers
   - Output: Scored answers
   ↓
8. Report Agent (optional)
   - Generate PDF report
   - Output: PDF file
   ↓
9. WA Tool Agent (optional)
   - Create workload
   - Populate answers
   - Generate official report
   - Output: Workload ID, report
```

## Technology Stack

### AI/ML
- **Amazon Bedrock**: AI service for agent processing
- **Claude Sonnet 3.7**: Primary AI model (`us.anthropic.claude-3-7-sonnet-20250219-v1:0`)
- **Strands Framework**: Agent orchestration framework

### AWS Services
- **AWS Well-Architected Tool API**: Workload management and reporting
- **S3**: Report storage (optional)
- **CloudWatch**: Logging and monitoring (if deployed)

### Python Libraries
- **boto3**: AWS SDK for Python
- **json**: JSON processing
- **concurrent.futures**: Parallel processing
- **logging**: Logging framework

## Design Patterns

### 1. Multi-Agent Pattern
- Specialized agents for specific tasks
- Agents work independently but coordinate through orchestrator
- Each agent has a single responsibility

### 2. Pipeline Pattern
- Sequential processing through agent pipeline
- Data flows from one agent to the next
- Each stage transforms the data

### 3. Graceful Degradation
- If one agent fails, others continue
- Errors are logged but don't stop the pipeline
- Partial results are still useful

### 4. Singleton Pattern
- Orchestrator reused across invocations (Lambda)
- Reduces initialization overhead

### 5. Factory Pattern
- `create_orchestrator()`: Factory function for orchestrator
- `create_*_agent()`: Factory functions for each agent

### 6. Retry Pattern
- `retry_with_backoff()`: Automatic retry with exponential backoff
- Handles transient failures (network, API throttling)

## Error Handling

### Strategy
- **Try-catch blocks**: Each agent step wrapped in try-catch
- **Error logging**: All errors logged with context
- **Error collection**: Non-fatal errors collected in results
- **Status tracking**: Final status reflects errors (`completed`, `completed_with_errors`, `error`)

### Error Types
- **Agent failures**: Individual agent errors don't stop pipeline
- **Validation failures**: Invalid data skipped with warnings
- **API failures**: Retried with backoff, fallback to direct Bedrock calls
- **Timeout errors**: Handled with retry logic

## Performance Optimizations

### 1. Parallel Processing
- Transcript segments processed in parallel batches
- Multiple insights mapped simultaneously
- Uses `ThreadPoolExecutor` for concurrency

### 2. Caching
- WAFR schema cached (1 hour TTL)
- Question details cached to reduce API calls
- Orchestrator singleton pattern

### 3. Batching
- Insights processed in batches (5-10 at a time)
- Reduces API call overhead
- Balances speed and resource usage

### 4. Smart Segmentation
- Transcripts split at natural boundaries (sentences, paragraphs)
- Prevents cutting off context
- Optimal segment size (5000 characters)

## Configuration

### Key Configuration Files

**`agents/config.py`**:
- Model ID: `us.anthropic.claude-3-7-sonnet-20250219-v1:0`
- Region: `us-east-1` (default)
- Temperature settings per agent
- Scoring weights and thresholds

**`agents/wafr_context.py`**:
- WAFR schema loading
- Question context generation
- Pillar summaries

## Security Considerations

### 1. AWS Credentials
- Uses AWS SDK default credential chain
- Supports IAM roles, environment variables, credentials file

### 2. Data Privacy
- Transcripts processed in memory
- No persistent storage of sensitive data (unless explicitly saved)
- Reports can be stored in S3 with appropriate permissions

### 3. API Security
- WA Tool API uses IAM authentication
- Bedrock API uses IAM authentication
- No hardcoded credentials

## Scalability

### Current Design
- **Local execution**: Runs on single machine
- **Parallel processing**: Uses threading for concurrent operations
- **Stateless agents**: Agents don't maintain state between runs

### Future Scalability Options
- **Distributed processing**: Deploy agents as separate services
- **Queue-based**: Use SQS for task distribution
- **Serverless**: Deploy as Lambda functions (removed in cleanup)
- **Containerization**: Docker containers for agents

## Monitoring and Observability

### Logging
- Structured logging with Python `logging` module
- Log levels: INFO, WARNING, ERROR, DEBUG
- Logs include session IDs, agent names, processing times

### Metrics (if deployed)
- Processing time per agent
- Success/failure rates
- Answer counts and coverage percentages

## Testing Strategy

### Unit Tests
- Individual agent testing
- Utility function testing
- Mock AWS services

### Integration Tests
- End-to-end pipeline testing
- Real transcript processing
- WA Tool integration testing

### Test Files
- `test_e2e_wafr.py`: End-to-end tests
- Test payloads and expected results

## Deployment Options

### 1. Local Execution
- **Entry point**: `run_wafr.py`
- **Usage**: `python run_wafr.py transcript.txt`
- **Best for**: Development, testing, small-scale usage

### 2. AWS Lambda (Removed)
- Previously deployed as serverless function
- Removed in cleanup

## Future Enhancements

### Potential Improvements
1. **Real-time processing**: WebSocket updates during processing
2. **Multi-language support**: Process transcripts in multiple languages
3. **Custom lenses**: Support for custom WAFR lenses
4. **Advanced analytics**: Trend analysis across multiple reviews
5. **Collaboration features**: Multi-user review and approval workflow
6. **Integration**: Connect with other AWS services (Security Hub, Config)

## Dependencies

### Core Dependencies
- `strands`: Agent framework
- `boto3`: AWS SDK
- Standard Python library (json, logging, concurrent.futures, etc.)

### Optional Dependencies
- PDF generation libraries (for reports)

## File Structure

```
wafr-prototype/
├── agents/              # Agent implementations
│   ├── orchestrator.py  # Main coordinator
│   ├── understanding_agent.py
│   ├── mapping_agent.py
│   ├── confidence_agent.py
│   ├── gap_detection_agent.py
│   ├── prompt_generator_agent.py
│   ├── scoring_agent.py
│   ├── report_agent.py
│   ├── wa_tool_agent.py
│   ├── config.py        # Configuration
│   ├── utils.py         # Utility functions
│   └── wafr_context.py  # WAFR schema loader
├── knowledge_base/       # WAFR knowledge base
│   └── wafr-schema.json # WAFR questions schema
├── schemas/             # Data schemas
├── run_wafr.py         # CLI entry point
└── README.md           # Project documentation
```

## Conclusion

The WAFR system is designed as a modular, extensible multi-agent system that automates the analysis of workshop transcripts and generates comprehensive AWS Well-Architected Framework assessments. The architecture emphasizes:

- **Modularity**: Each agent has a single, well-defined responsibility
- **Reliability**: Graceful error handling and retry mechanisms
- **Performance**: Parallel processing and caching optimizations
- **Extensibility**: Easy to add new agents or modify existing ones
- **Maintainability**: Clear separation of concerns and comprehensive logging

The system can be deployed locally for development and testing, or integrated into larger workflows for production use.

