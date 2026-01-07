# WAFR - Well-Architected Framework Review System

An AI-powered multi-agent system for automated AWS Well-Architected Framework assessments.

## Overview

WAFR automatically analyzes workshop transcripts using specialized AI agents powered by Amazon Bedrock (Claude Sonnet). The system processes natural language transcripts through a coordinated pipeline to extract insights, map to WAFR questions, validate evidence, and generate comprehensive assessments with PDF reports.

## Features

- ✅ **Multi-Agent AI Pipeline**: Specialized agents for understanding, mapping, confidence validation, gap detection, and more
- ✅ **AG-UI Integration**: Real-time event streaming for frontend applications
- ✅ **Strict Quality Control**: Confidence threshold >= 0.7, honest assessment
- ✅ **HRI Validation**: Claude-based validation to filter non-tangible HRIs
- ✅ **Automatic Lens Detection**: Identifies relevant AWS Well-Architected Lenses from transcript
- ✅ **WA Tool Integration**: Creates workloads and generates official PDF reports
- ✅ **Enhanced Question Answering**: Intelligent inference with evidence validation

## Quick Start

### Prerequisites

- Python 3.10+
- AWS credentials configured (for Bedrock and Well-Architected Tool)
- Required packages (see `config/requirements.txt`)

### Installation

```bash
# Install dependencies
pip install -r config/requirements.txt

# Optional: Install AG-UI protocol for event streaming
pip install ag-ui-protocol
```

### Run the Pipeline

**Windows:**
```batch
scripts\run_wafr_full.bat "My Client Name"
```

**Linux/Mac:**
```bash
chmod +x scripts/run_wafr_full.sh
./scripts/run_wafr_full.sh "My Client Name"
```

**Python (any platform):**
```bash
python scripts/run_wafr_full.py --wa-tool --client-name "My Client Name"
```

### With Custom Transcript

```bash
python scripts/run_wafr_full.py \
    --transcript data/transcripts/my_transcript.txt \
    --wa-tool \
    --client-name "Client Name"
```

## Project Structure

```
WAFR-prototype/
├── src/wafr/          # Source code (agents, ag_ui, models, storage)
├── scripts/           # Executable scripts
├── docs/              # Documentation
├── data/              # Data files (knowledge_base, schemas, lenses, transcripts)
├── output/            # Generated outputs (logs, reports, results)
└── config/            # Configuration files
```

See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed structure documentation.

## Documentation

- **[SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)**: System architecture and design
- **[CODE_REVIEW_GUIDE.md](docs/CODE_REVIEW_GUIDE.md)**: Comprehensive code review guide
- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)**: Project organization guide
- **[AG_UI_INTEGRATION_GUIDE.md](docs/AG_UI_INTEGRATION_GUIDE.md)**: AG-UI integration documentation

## Architecture

The system uses a **coordinated multi-agent architecture**:

```
Orchestrator
    ├── Understanding Agent (extracts insights)
    ├── Mapping Agent (maps to WAFR questions)
    ├── Confidence Agent (validates evidence)
    ├── Gap Detection Agent (identifies gaps)
    ├── Answer Synthesis Agent (synthesizes answers)
    ├── Scoring Agent (grades and ranks)
    ├── WA Tool Agent (creates workload, generates PDF)
    └── Report Agent (generates reports)
```

Each agent is specialized for a specific task and uses Claude Sonnet via Amazon Bedrock.

### System Architecture Flow

```mermaid
graph TB
    Start([Start: Input Transcript]) --> PDF[PDF Processing<br/>Optional]
    PDF --> Understanding[Understanding Agent<br/>Extract Insights]
    Understanding --> Mapping[Mapping Agent<br/>Map to WAFR Questions]
    Mapping --> Confidence[Confidence Agent<br/>Validate Evidence]
    Confidence --> Gap[Gap Detection Agent<br/>Identify Gaps]
    Gap --> Synthesis[Answer Synthesis Agent<br/>Generate AI Answers]
    Synthesis --> AutoPop[Auto-Populate<br/>Merge Answers]
    AutoPop --> Scoring[Scoring Agent<br/>Grade Answers]
    Scoring --> Report{Generate<br/>Report?}
    Report -->|Yes| ReportGen[Report Agent<br/>Generate PDF]
    Report -->|No| WATool{Create WA<br/>Workload?}
    ReportGen --> WATool
    WATool -->|Yes| WAToolAgent[WA Tool Agent<br/>Create Workload & PDF]
    WATool -->|No| End([End: Results])
    WAToolAgent --> End
    
    style Start fill:#90EE90
    style End fill:#90EE90
    style Understanding fill:#87CEEB
    style Mapping fill:#87CEEB
    style Confidence fill:#87CEEB
    style Gap fill:#87CEEB
    style Synthesis fill:#87CEEB
    style Scoring fill:#87CEEB
    style ReportGen fill:#FFD700
    style WAToolAgent fill:#FFD700
```

### Detailed Processing Pipeline

```mermaid
flowchart TD
    A[Input: Transcript + PDFs] --> B[Step 0: PDF Processing]
    B --> C[Step 1: Understanding Agent]
    C --> C1[Segment Transcript]
    C1 --> C2[Parallel Processing]
    C2 --> C3[Extract Insights]
    C3 --> D[Step 2: Mapping Agent]
    D --> D1[Map Insights to Questions]
    D1 --> D2[Generate Answer Drafts]
    D2 --> E[Step 3: Confidence Agent]
    E --> E1[Validate Evidence]
    E1 --> E2[Calculate Confidence Scores]
    E2 --> F[Step 4: Gap Detection]
    F --> F1[Compare All vs Answered]
    F1 --> F2[Identify Gaps]
    F2 --> G[Step 5: Answer Synthesis]
    G --> G1[Generate AI Answers]
    G1 --> G2[Assign Confidence]
    G2 --> H[Step 6: Auto-Populate]
    H --> H1[Merge Validated + Synthesized]
    H1 --> I[Step 7: Scoring Agent]
    I --> I1[Grade All Answers]
    I1 --> I2[Calculate Risk Levels]
    I2 --> J{Options}
    J -->|Report| K[Step 8: Report Agent]
    J -->|WA Tool| L[Step 9: WA Tool Agent]
    K --> M[Generate PDF Report]
    L --> N[Create Workload]
    N --> O[Populate Answers]
    O --> P[Generate Official PDF]
    M --> Q[Output: Results]
    P --> Q
    
    style A fill:#E6F3FF
    style Q fill:#90EE90
    style C fill:#FFE6E6
    style D fill:#FFE6E6
    style E fill:#FFE6E6
    style F fill:#FFE6E6
    style G fill:#FFE6E6
    style I fill:#FFE6E6
    style K fill:#FFF4E6
    style L fill:#FFF4E6
```

### Agent Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant Understanding
    participant Mapping
    participant Confidence
    participant GapDetection
    participant Synthesis
    participant Scoring
    participant WATool
    
    User->>Orchestrator: Submit Transcript
    Orchestrator->>Understanding: Extract Insights
    Understanding-->>Orchestrator: Insights List
    Orchestrator->>Mapping: Map to Questions
    Mapping-->>Orchestrator: Question-Answer Mappings
    Orchestrator->>Confidence: Validate Evidence
    Confidence-->>Orchestrator: Validated Answers
    Orchestrator->>GapDetection: Find Gaps
    GapDetection-->>Orchestrator: Gap List
    Orchestrator->>Synthesis: Generate Answers
    Synthesis-->>Orchestrator: Synthesized Answers
    Orchestrator->>Scoring: Grade All Answers
    Scoring-->>Orchestrator: Scored Answers
    Orchestrator->>WATool: Create Workload
    WATool-->>Orchestrator: Workload ID
    Orchestrator->>WATool: Populate Answers
    WATool-->>Orchestrator: Success
    Orchestrator->>WATool: Generate PDF
    WATool-->>Orchestrator: PDF Report
    Orchestrator-->>User: Complete Results
```

### Data Flow Diagram

```mermaid
graph LR
    subgraph Input
        T[Transcript]
        P[PDFs]
    end
    
    subgraph Processing
        U[Understanding<br/>Agent]
        M[Mapping<br/>Agent]
        C[Confidence<br/>Agent]
        G[Gap<br/>Detection]
        S[Synthesis<br/>Agent]
    end
    
    subgraph Output
        R[PDF Report]
        J[JSON Results]
        W[WA Tool<br/>Workload]
    end
    
    T --> U
    P --> U
    U -->|Insights| M
    M -->|Mappings| C
    C -->|Validated| G
    G -->|Gaps| S
    S -->|Synthesized| R
    S -->|Synthesized| J
    S -->|Synthesized| W
    
    style Input fill:#E6F3FF
    style Processing fill:#FFE6E6
    style Output fill:#90EE90
```

### Lens Detection Flow

```mermaid
flowchart TD
    A[Transcript Input] --> B[Lens Detection Agent]
    B --> C{Detect Lenses}
    C -->|GenAI Keywords| D[Generative AI Lens]
    C -->|Serverless Keywords| E[Serverless Lens]
    C -->|ML Keywords| F[Machine Learning Lens]
    C -->|Default| G[Well-Architected Lens]
    D --> H[Load Lens Questions]
    E --> H
    F --> H
    G --> H
    H --> I[Process with<br/>Lens-Specific Context]
    
    style A fill:#E6F3FF
    style B fill:#FFE6E6
    style H fill:#90EE90
    style I fill:#90EE90
```

### HRI Validation Flow

```mermaid
flowchart TD
    A[All Answers Scored] --> B[Extract High-Risk Issues]
    B --> C[For Each HRI]
    C --> D[Claude Validation]
    D --> E{Is HRI Tangible?}
    E -->|Yes| F[Keep HRI<br/>Evidence-Backed]
    E -->|No| G[Filter Out<br/>False Positive]
    F --> H[Final HRI List]
    G --> H
    H --> I[Generate Report<br/>with Validated HRIs]
    
    style A fill:#E6F3FF
    style D fill:#FFE6E6
    style F fill:#90EE90
    style G fill:#FFB6C1
    style I fill:#90EE90
```

### Quick Reference: Decision Points

```mermaid
flowchart TD
    Start([Start Processing]) --> Input{Input Type?}
    Input -->|File| FileProc[Process File]
    Input -->|Transcript| TranscriptProc[Process Transcript]
    FileProc --> DetectLens{Lens Detection}
    TranscriptProc --> DetectLens
    DetectLens --> Pipeline[Run Pipeline]
    Pipeline --> GapCheck{Gaps Found?}
    GapCheck -->|Yes| Synthesis[Answer Synthesis]
    GapCheck -->|No| Scoring
    Synthesis --> Scoring[Scoring Agent]
    Scoring --> ReportQ{Generate Report?}
    ReportQ -->|Yes| Report[Report Agent]
    ReportQ -->|No| WAToolQ{Create WA Workload?}
    Report --> WAToolQ
    WAToolQ -->|Yes| WATool[WA Tool Agent]
    WAToolQ -->|No| End([End])
    WATool --> End
    
    style Start fill:#90EE90
    style End fill:#90EE90
    style DetectLens fill:#FFD700
    style GapCheck fill:#FFD700
    style ReportQ fill:#FFD700
    style WAToolQ fill:#FFD700
```

## Configuration

### Environment Variables

- `AWS_REGION`: AWS region (default: us-east-1)
- `BEDROCK_MODEL_ID`: Bedrock model ID (default: anthropic.claude-sonnet-3-5-20241022-v2:0)
- `WA_TOOL_ENABLED`: Enable WA Tool integration (default: true)

### AWS Credentials

Configure AWS credentials using one of:
- AWS CLI: `aws configure`
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- IAM roles (if running on EC2)

## Output

The system generates:

- **PDF Reports**: Official AWS Well-Architected Framework reports
  - Location: `output/reports/wafr_report_{workload_id}_{session_id}.pdf`
  
- **JSON Results**: Complete assessment results
  - Location: `output/results/wafr_results_{session_id}.json`
  
- **Logs**: Detailed execution logs
  - Location: `output/logs/wafr_{timestamp}.log`

## Development

### Project Structure

The project follows a clean, professional structure:

- **`src/wafr/`**: Main package with all source code
- **`scripts/`**: Executable scripts
- **`docs/`**: Documentation
- **`data/`**: Data files
- **`output/`**: Generated outputs
- **`config/`**: Configuration files

### Adding New Agents

1. Create agent file in `src/wafr/agents/`
2. Implement agent class following existing patterns
3. Register in orchestrator (`src/wafr/agents/orchestrator.py`)
4. Update imports and exports

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_specific.py
```

## License

Copyright (c) 2024

## Support

For issues, questions, or contributions, please refer to the documentation in the `docs/` directory.

