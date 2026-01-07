# WAFR Prototype - Cursor AI Context Guide

## Project Overview

**WAFR (Well-Architected Framework Review) Prototype** is an AI-powered multi-agent system that automatically analyzes AWS Well-Architected Framework workshop transcripts using specialized AI agents powered by Amazon Bedrock (Claude Sonnet).

## Architecture Pattern

**Multi-Agent System with Sequential Pipeline Orchestration**

- **Orchestrator** (`agents/orchestrator.py`) coordinates all agents
- **Specialized Agents** perform focused tasks in sequence
- **Context flows** through the pipeline (transcript → insights → mappings → answers → report)
- **Graceful degradation** - continues with partial results on errors

## Key Components

### Core Agents (in execution order)

1. **Understanding Agent** (`agents/understanding_agent.py`)
   - Extracts architecture insights from transcripts
   - Uses Claude Sonnet with temperature 0.1 (factual)
   - Outputs: Decisions, Services, Constraints, Risks

2. **Mapping Agent** (`agents/mapping_agent.py`)
   - Maps insights to WAFR questions
   - Calculates relevance scores
   - Synthesizes answers from transcript
   - Uses Claude Sonnet with temperature 0.2

3. **Confidence Agent** (`agents/confidence_agent.py`)
   - Validates evidence (anti-hallucination)
   - Verifies verbatim quotes in transcript
   - Assigns confidence scores (0.0-1.0)
   - Uses Claude Sonnet with temperature 0.1
   - **Timeout**: 30 seconds per validation

4. **Gap Detection Agent** (`agents/gap_detection_agent.py`)
   - Identifies unanswered WAFR questions
   - Prioritizes gaps by criticality
   - Outputs prioritized gap list

5. **Answer Synthesis Agent** (`agents/answer_synthesis_agent.py`) ⭐ **NEW**
   - Generates AI answers for gap questions
   - Uses transcript + insights + inference
   - Includes reasoning chains and assumptions
   - **Timeout**: 120 seconds per synthesis
   - **Continues on errors** - creates fallback answers

6. **Scoring Agent** (`agents/scoring_agent.py`)
   - Multi-dimensional scoring (confidence, completeness, compliance)
   - Assigns letter grades (A-F)
   - Uses Claude Sonnet with temperature 0.2

7. **Report Agent** (`agents/report_agent.py`)
   - Generates AWS-compatible PDF reports
   - Uses Claude Sonnet with temperature 0.3

8. **WA Tool Agent** (`agents/wa_tool_agent.py`)
   - Integrates with AWS Well-Architected Tool API
   - Creates workloads, populates answers, generates reviews

### Supporting Components

- **Lens Manager** (`agents/lens_manager.py`)
  - Manages AWS Well-Architected Lenses (GenAI, Serverless, SaaS, etc.)
  - **Uses Claude for intelligent lens detection** (replaces keyword matching)
  - Caches lens definitions locally
  - Auto-detects relevant lenses from transcript

- **Input Processor** (`agents/input_processor.py`)
  - Handles PDF, video, audio, and text inputs
  - Routes to appropriate processors

- **PDF Processor** (`agents/pdf_processor.py`)
  - Extracts text, images, tables from PDFs
  - Uses Amazon Textract or local OCR

- **Video Processor** (`agents/video_processor.py`)
  - Transcribes video/audio files
  - Uses Amazon Transcribe or Whisper

## Important Patterns & Conventions

### 1. Agent Initialization

All agents follow this pattern:
```python
def create_agent_name(wafr_schema: Optional[Dict] = None, 
                      lens_context: Optional[Dict] = None) -> AgentName:
    """Factory function for creating agent."""
    return AgentName(wafr_schema=wafr_schema, lens_context=lens_context)
```

### 2. Bedrock Model Invocation

Agents use this pattern for Bedrock calls:
```python
response = self.bedrock.invoke_model(
    modelId=self.model_id,
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
    }),
)
```

### 3. Timeout Protection

Critical operations use `timeout_wrapper` from `agents/utils.py`:
```python
from agents.utils import timeout_wrapper
result = timeout_wrapper(lambda: some_operation(), timeout_seconds=120.0)
```

### 4. Error Handling

- **Individual failures don't stop the pipeline**
- **Answer Synthesis** creates fallback answers on errors
- **Orchestrator** continues with partial results
- All errors are logged but don't crash the system

### 5. Data Flow

```
transcript → insights → mappings → validated_answers → gaps → 
synthesized_answers → all_answers (merged) → scoring → report
```

**Important**: `all_answers` includes both validated (from transcript) and synthesized (AI-generated) answers. Always use `all_answers` in later steps, not just `validated_answers`.

### 6. Configuration

- **Model ID**: From `agents/config.py` - defaults to Claude 3.7 Sonnet
- **Temperatures**: Defined per agent in `agents/config.py`
- **Timeouts**: Defined per agent (30s for confidence, 120s for synthesis)
- **Batch Sizes**: Defined per agent for parallel processing

## Key Files & Their Purposes

### Entry Points
- `run_wafr.py` - Main CLI entry point with full feature set
- `agents/main.py` - Alternative entry point (simpler)

### Core Orchestration
- `agents/orchestrator.py` - Main coordinator (1720 lines)
  - `process_transcript()` - Main processing method
  - `_step_*()` methods for each pipeline step
  - **Uses `all_answers` (not `validated_answers`) for scoring/report/WA Tool**

### Agent Implementations
- `agents/understanding_agent.py` - Insight extraction
- `agents/mapping_agent.py` - Question mapping
- `agents/confidence_agent.py` - Evidence validation
- `agents/gap_detection_agent.py` - Gap identification
- `agents/answer_synthesis_agent.py` - AI answer generation ⭐
- `agents/scoring_agent.py` - Scoring & grading
- `agents/report_agent.py` - PDF report generation
- `agents/wa_tool_agent.py` - AWS WA Tool integration

### Utilities
- `agents/utils.py` - Shared utilities, timeout_wrapper, batch processing
- `agents/config.py` - Configuration settings, model IDs, temperatures
- `agents/base_agent.py` - Base class for agents (Bedrock client)

### Lens Management
- `agents/lens_manager.py` - Lens catalog, caching, **Claude-based detection**
- `agents/lens_schema.py` - Lens schema definitions

### Input Processing
- `agents/input_processor.py` - Input type detection and routing
- `agents/pdf_processor.py` - PDF text/image/table extraction
- `agents/video_processor.py` - Video/audio transcription

## Important Constants

### Timeouts
- `BATCH_TIMEOUT = 30.0` (confidence agent)
- `SYNTHESIS_TIMEOUT = 120.0` (answer synthesis agent)
- `BATCH_TIMEOUT = 120.0` (mapping agent)

### Batch Sizes
- `SYNTHESIS_BATCH_SIZE = 5` (answer synthesis)
- `BATCH_SIZE = 2` (confidence agent)
- Varies by agent

### Model Settings
- Default: `us.anthropic.claude-3-7-sonnet-20250219-v1:0`
- Region: `us-east-1` (configurable)
- Max tokens: Varies by agent (typically 4096)

## Common Issues & Solutions

### Issue: Process stops mid-execution
**Causes**:
- Timeout on Bedrock calls (add timeout_wrapper)
- Memory issues with large transcripts
- AWS credential expiration
- Network connectivity issues

**Solutions**:
- All agents now have timeout protection
- Answer synthesis continues on errors
- Check AWS credentials before running

### Issue: Synthesized answers not appearing in report
**Fix Applied**: Orchestrator now uses `all_answers` (merged) instead of `validated_answers` for:
- Scoring step
- Report generation
- WA Tool integration
- Final results

### Issue: Lens detection not working
**Fix Applied**: Replaced keyword-based detection with Claude-based intelligent detection in `lens_manager.py`

## Code Style & Best Practices

1. **Type Hints**: Use type hints for all function parameters and returns
2. **Docstrings**: All public methods have docstrings
3. **Logging**: Use `logger` from `logging` module (not print)
4. **Error Messages**: Include context in error messages
5. **Constants**: Define at module level, use UPPER_CASE
6. **Factory Functions**: Use `create_*` pattern for agent creation

## Testing & Debugging

- Check `debug.log` for detailed logs
- Use `--output` flag to save results to JSON
- Check `results["steps"]` for individual agent outputs
- Look for `"error"` keys in step results

## AWS Integration

### Required Services
- **Bedrock**: For AI model inference
- **Well-Architected Tool**: For workload creation (optional)
- **Textract**: For PDF OCR (optional)
- **Transcribe**: For audio transcription (optional)

### Credentials
- Uses default AWS credential chain
- Check with `check_aws_credentials.py`
- Region defaults to `us-east-1`

## Recent Changes

1. **Claude-based Lens Detection** - Replaced keyword matching with intelligent Claude analysis
2. **Answer Synthesis Integration** - Fixed orchestrator to use `all_answers` throughout pipeline
3. **Timeout Protection** - Added timeout_wrapper to answer synthesis agent
4. **Error Resilience** - Answer synthesis continues on errors with fallback answers

## When Modifying Code

### Adding a New Agent
1. Create agent class in `agents/agent_name.py`
2. Inherit from `BaseAgent` or implement Bedrock client
3. Add factory function `create_agent_name()`
4. Add step method in orchestrator: `_step_agent_name()`
5. Integrate into `process_transcript()` pipeline
6. Update this cursor.md file

### Modifying Pipeline Flow
1. Update `process_transcript()` in orchestrator
2. Ensure data flows correctly between steps
3. Use `all_answers` (not `validated_answers`) for steps after synthesis
4. Add error handling that allows continuation

### Adding Timeout Protection
```python
from agents.utils import timeout_wrapper
result = timeout_wrapper(lambda: operation(), timeout_seconds=120.0)
```

### Adding Retry Logic
```python
from agents.utils import retry_with_backoff

@retry_with_backoff(max_retries=3, initial_delay=1.0)
def operation():
    # Your code here
```

## Project Structure

```
WAFR prototype - Copy/
├── agents/                    # All agent implementations
│   ├── orchestrator.py       # Main coordinator (1720 lines)
│   ├── understanding_agent.py
│   ├── mapping_agent.py
│   ├── confidence_agent.py
│   ├── gap_detection_agent.py
│   ├── answer_synthesis_agent.py  # ⭐ AI answer generation
│   ├── scoring_agent.py
│   ├── report_agent.py
│   ├── wa_tool_agent.py
│   ├── lens_manager.py       # ⭐ Claude-based lens detection
│   ├── input_processor.py
│   ├── pdf_processor.py
│   ├── video_processor.py
│   ├── utils.py              # Shared utilities
│   ├── config.py             # Configuration
│   └── base_agent.py         # Base agent class
├── knowledge_base/           # WAFR knowledge base
│   └── lenses/              # Cached lens definitions
├── schemas/                  # WAFR schema definitions
│   └── wafr-schema.json
├── run_wafr.py              # Main CLI entry point
├── README.md                 # Project overview
├── SYSTEM_DESIGN.md          # Detailed design doc
└── cursor.md                 # This file
```

## Quick Reference

### Run WAFR Analysis
```bash
python run_wafr.py transcript.txt --output results.json
```

### With WA Tool Integration
```bash
python run_wafr.py transcript.txt --wa-tool --client-name "Client Name"
```

### With Specific Lenses
```bash
python run_wafr.py transcript.txt --lenses generative-ai serverless
```

### List Available Lenses
```bash
python run_wafr.py --list-lenses
```

## Notes for AI Assistants

- **Always use `all_answers`** (not `validated_answers`) after answer synthesis step
- **Answer synthesis continues on errors** - creates fallback answers
- **Timeout protection is critical** - use `timeout_wrapper` for Bedrock calls
- **Lens detection uses Claude** - not keyword matching anymore
- **Orchestrator coordinates everything** - check there first for pipeline issues
- **Error handling is graceful** - system continues with partial results
- **All agents use Bedrock Claude Sonnet** - configured in `config.py`

