# WAFR Agentic Prototype - AI Architecture

## Overview

An **AI-powered multi-agent system** that automatically analyzes AWS Well-Architected Framework Review (WAFR) workshop transcripts using specialized AI agents powered by Amazon Bedrock (Claude Sonnet). The system processes natural language transcripts through a coordinated pipeline of intelligent agents to extract insights, map to WAFR questions, validate evidence, and generate comprehensive assessments.

## AI Architecture

### Multi-Agent System Design

The system uses a **coordinated multi-agent architecture** where specialized AI agents work together in a sequential pipeline, each performing a specific cognitive task:

```
┌─────────────────────────────────────────────────────────┐
│              WAFR Orchestrator (Coordinator)            │
│  • Coordinates agent pipeline                           │
│  • Manages workflow and data flow                       │
│  • Handles errors with graceful degradation            │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Understanding│ │   Mapping    │ │  Confidence  │
│    Agent     │ │    Agent     │ │    Agent     │
│              │ │              │ │              │
│ Extracts     │ │ Maps to      │ │ Validates    │
│ insights     │ │ WAFR Qs      │ │ evidence     │
└──────────────┘ └──────────────┘ └──────────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Gap Detection│ │   Scoring    │ │    Report    │
│    Agent     │ │    Agent     │ │    Agent     │
│              │ │              │ │              │
│ Identifies   │ │ Grades &     │ │ Generates    │
│ gaps         │ │ ranks        │ │ PDF report   │
└──────────────┘ └──────────────┘ └──────────────┘
```

### Core AI Agents

#### 1. **Understanding Agent** (`agents/understanding_agent.py`)
**AI Model**: Claude Sonnet via Bedrock  
**Purpose**: Semantic extraction of architecture insights from transcripts

- **Process**: Segment transcripts → Parallel batch processing → AI-powered extraction
- **Output Types**:
  - **Decisions**: Architecture choices, technology selections
  - **Services**: AWS services with usage context
  - **Constraints**: Requirements, limitations, compliance needs
  - **Risks**: Security, availability, scalability concerns
- **Features**: Verbatim quote extraction, speaker attribution, deduplication

#### 2. **Mapping Agent** (`agents/mapping_agent.py`)
**AI Model**: Claude Sonnet via Bedrock  
**Purpose**: Intelligent mapping of insights to WAFR questions

- **Process**: Relevance scoring → Question matching → Answer synthesis
- **Mapping Strategy**:
  - Keyword matching with semantic understanding
  - Question intent alignment
  - Pillar relevance (OPS, SEC, REL, PERF, COST, SUS)
  - Criticality-weighted scoring
- **Output**: Mapped questions with relevance scores, answer content, evidence quotes

#### 3. **Confidence Agent** (`agents/confidence_agent.py`)
**AI Model**: Claude Sonnet via Bedrock  
**Purpose**: Anti-hallucination validation and evidence verification

- **Validation Criteria**:
  - Evidence quote verification (verbatim matching)
  - Answer completeness assessment
  - Contextual relevance check
  - Hallucination detection
- **Scoring**: Confidence scores (0.0-1.0) with high/medium/low classification
- **Output**: Validated answers with confidence metrics

#### 4. **Gap Detection Agent** (`agents/gap_detection_agent.py`)
**AI Model**: Claude Sonnet via Bedrock  
**Purpose**: Identify unanswered WAFR questions and prioritize gaps

- **Process**: Coverage analysis → Priority scoring → Gap identification
- **Priority Factors**:
  - Question criticality (critical/high/medium/low)
  - Pillar coverage gaps
  - Business impact assessment
- **Output**: Prioritized list of unanswered questions with priority scores

#### 5. **Prompt Generator Agent** (`agents/prompt_generator_agent.py`)
**AI Model**: Claude Sonnet via Bedrock  
**Purpose**: Generate context-aware prompts for filling gaps

- **Features**: 
  - Context-aware question prompts
  - Best practice hints from WAFR schema
  - Example answers based on patterns
  - Multi-format output (text, structured)

#### 6. **Scoring Agent** (`agents/scoring_agent.py`)
**AI Model**: Claude Sonnet via Bedrock  
**Purpose**: Multi-dimensional scoring and grade assignment

- **Scoring Dimensions**:
  - **Confidence** (40% weight): Evidence quality
  - **Completeness** (30% weight): Answer thoroughness
  - **Compliance** (30% weight): WAFR best practice alignment
- **Grading**: Letter grades (A-F) based on score thresholds
- **Output**: Scored answers with grades and detailed metrics

#### 7. **Report Agent** (`agents/report_agent.py`)
**AI Model**: Claude Sonnet via Bedrock  
**Purpose**: Generate comprehensive WAFR assessment reports

- **Output Format**: AWS Well-Architected Tool-compatible PDF reports
- **Content**: Executive summary, pillar analysis, recommendations, risk assessment
- **Structure**: Official AWS report format and terminology

#### 8. **WA Tool Agent** (`agents/wa_tool_agent.py`)
**Purpose**: Integration with AWS Well-Architected Tool API

- **Features**:
  - Autonomous workload creation
  - Automatic answer population from transcript analysis
  - Multi-lens support (Well-Architected, Generative AI, Serverless, etc.)
  - Milestone and review generation

### AI Model Configuration

**Primary Model**: Amazon Bedrock - Claude Sonnet 3.7  
**Fallback Model**: Claude 3.5 Sonnet / Claude 3 Haiku

**Model Settings**:
- **Understanding Agent**: Temperature 0.1 (factual extraction)
- **Mapping Agent**: Temperature 0.2 (structured mapping)
- **Confidence Agent**: Temperature 0.1 (consistent validation)
- **Scoring Agent**: Temperature 0.2 (numerical assessment)
- **Report Agent**: Temperature 0.3 (narrative generation)

### Architecture Features

#### 1. **Strands Framework Integration**
- Uses Strands framework for agent orchestration
- Tool-based agent capabilities
- Fallback to direct Bedrock API if Strands unavailable

#### 2. **Knowledge Base Integration**
- WAFR schema from AWS API (official questions)
- Local knowledge base for best practices
- Multi-lens support (Well-Architected, GenAI, Serverless, SaaS, etc.)

#### 3. **Input Processing Pipeline**
- **PDF Processing**: Text extraction, OCR, image/table extraction
- **Video/Audio Processing**: Transcription via Amazon Transcribe or Whisper
- **Text Processing**: Direct transcript processing

#### 4. **Error Handling & Resilience**
- Graceful degradation (continues on partial failures)
- Retry logic with exponential backoff
- Circuit breaker pattern for API failures
- Comprehensive error logging

#### 5. **Performance Optimizations**
- Parallel batch processing for segments
- Caching of schema and processed segments
- Lazy loading of processors
- Connection pooling for AWS services

## Technology Stack

- **AI/ML**: Amazon Bedrock (Claude Sonnet), Strands Framework
- **Language**: Python 3.10+
- **AWS Services**: 
  - Bedrock (AI models)
  - Well-Architected Tool API
  - Textract (PDF OCR)
  - Transcribe (audio transcription)
- **Libraries**: boto3, pdfplumber, PyPDF2, Pillow, pytesseract

## Project Structure

```
WAFR-Agentic-Prototype/
├── agents/                      # AI Agent implementations
│   ├── orchestrator.py         # Multi-agent coordinator
│   ├── understanding_agent.py  # Insight extraction agent
│   ├── mapping_agent.py        # Question mapping agent
│   ├── confidence_agent.py     # Evidence validation agent
│   ├── gap_detection_agent.py  # Gap identification agent
│   ├── scoring_agent.py        # Scoring & grading agent
│   ├── report_agent.py         # Report generation agent
│   ├── wa_tool_agent.py        # AWS WA Tool integration
│   ├── input_processor.py      # File processing pipeline
│   ├── pdf_processor.py        # PDF extraction
│   ├── video_processor.py      # Video/audio transcription
│   ├── utils.py                # Shared utilities
│   └── config.py               # Configuration
├── knowledge_base/             # WAFR knowledge base
│   ├── lenses/                # Lens schemas
│   └── *.json                 # Knowledge base files
├── schemas/                    # WAFR schema definitions
├── run_wafr.py                # Main entry point
└── setup.py                   # Package setup
```

## Key AI Capabilities

1. **Semantic Understanding**: Extracts meaningful insights from natural language transcripts
2. **Intelligent Mapping**: Maps unstructured content to structured WAFR questions
3. **Evidence Validation**: Validates answers with anti-hallucination checks
4. **Gap Analysis**: Identifies missing information with priority scoring
5. **Automated Scoring**: Multi-dimensional assessment with letter grades
6. **Report Generation**: Creates AWS-compatible WAFR reports
7. **Multi-Lens Support**: Handles specialized lenses (GenAI, Serverless, etc.)
8. **Autonomous Operation**: Fully automated from transcript to report

## Usage

```bash
# Process a transcript file
python run_wafr.py transcript.txt

# Process a PDF document
python run_wafr.py architecture_doc.pdf --wa-tool --client-name "Acme Corp"

# Process with multiple lenses
python run_wafr.py transcript.txt --lenses generative-ai serverless
```

## Design Principles

1. **Agent Specialization**: Each agent has a focused, well-defined responsibility
2. **Coordinated Pipeline**: Orchestrator manages sequential agent execution
3. **Context Preservation**: WAFR schema context flows through all agents
4. **Graceful Degradation**: System continues with partial results on errors
5. **Evidence-Based**: All answers require transcript evidence with quotes
6. **Anti-Hallucination**: Confidence agent validates against source material

---

**Built with**: Amazon Bedrock, Claude Sonnet, Strands Framework  
**Architecture Pattern**: Multi-Agent System with Sequential Pipeline Orchestration
