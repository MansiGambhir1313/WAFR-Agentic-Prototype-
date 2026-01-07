# WAFR Project Structure

This document describes the professional, clean code structure of the WAFR (Well-Architected Framework Review) project.

## Directory Structure

```
WAFR-prototype/
├── src/                          # Source code
│   └── wafr/                    # Main WAFR package
│       ├── __init__.py          # Package initialization
│       ├── agents/              # AI agent implementations
│       │   ├── __init__.py
│       │   ├── orchestrator.py  # Main orchestrator
│       │   ├── understanding_agent.py
│       │   ├── mapping_agent.py
│       │   ├── confidence_agent.py
│       │   ├── gap_detection_agent.py
│       │   ├── answer_synthesis_agent.py
│       │   ├── scoring_agent.py
│       │   ├── report_agent.py
│       │   ├── wa_tool_agent.py
│       │   └── ...              # Other agent files
│       ├── ag_ui/               # AG-UI integration
│       │   ├── __init__.py
│       │   ├── core.py
│       │   ├── events.py
│       │   ├── emitter.py
│       │   ├── state.py
│       │   ├── orchestrator_integration.py
│       │   └── server.py
│       ├── models/              # Data models
│       │   ├── __init__.py
│       │   ├── review_item.py
│       │   ├── synthesized_answer.py
│       │   └── validation_record.py
│       └── storage/            # Storage layer
│           ├── __init__.py
│           └── review_storage.py
│
├── scripts/                     # Executable scripts
│   ├── run_wafr_full.py        # Main runner script
│   ├── run_wafr_full.bat       # Windows batch script
│   ├── run_wafr_full.sh        # Linux/Mac shell script
│   ├── generate_pdf_from_results.py
│   ├── list_all_lenses.py
│   └── list_workloads.py
│
├── docs/                        # Documentation
│   ├── RUN_WAFR_FULL.md
│   ├── AG_UI_INTEGRATION_GUIDE.md
│   ├── SYSTEM_DESIGN.md
│   └── ...                     # Other documentation files
│
├── data/                        # Data files
│   ├── knowledge_base/         # Knowledge base JSON files
│   ├── schemas/                # JSON schemas
│   ├── lenses/                 # Lens configuration files
│   └── transcripts/            # Sample transcripts
│
├── output/                      # Generated outputs
│   ├── logs/                   # Log files
│   ├── reports/                # PDF reports
│   └── results/                # JSON results
│
├── config/                      # Configuration files
│   ├── requirements.txt
│   ├── setup.py
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tests/                       # Test files
│   └── ...
│
├── README.md                    # Main project README
├── PROJECT_STRUCTURE.md        # This file
└── .gitignore                  # Git ignore rules
```

## Package Organization

### Source Code (`src/wafr/`)

All source code is organized under the `wafr` package:

- **`agents/`**: Contains all AI agent implementations
  - Each agent is a separate module
  - `orchestrator.py` coordinates the multi-agent pipeline
  
- **`ag_ui/`**: AG-UI integration for real-time event streaming
  - Follows official AG-UI protocol
  - Provides event emission and state management
  
- **`models/`**: Data models and schemas
  - Pydantic models for type safety
  - Validation records and review items
  
- **`storage/`**: Storage abstraction layer
  - Review storage implementations
  - Future: Database integration

### Scripts (`scripts/`)

All executable scripts are in the `scripts/` directory:

- **`run_wafr_full.py`**: Main entry point for running the complete pipeline
- Helper scripts for utilities (list lenses, workloads, etc.)
- Platform-specific launchers (`.bat`, `.sh`)

### Documentation (`docs/`)

All documentation is centralized:

- User guides
- Integration guides
- Architecture documentation
- API references

### Data (`data/`)

All data files are organized by type:

- **`knowledge_base/`**: JSON knowledge bases for agents
- **`schemas/`**: JSON schemas for validation
- **`lenses/`**: AWS Well-Architected lens configurations
- **`transcripts/`**: Sample workshop transcripts

### Output (`output/`)

All generated files are organized:

- **`logs/`**: Application logs
- **`reports/`**: Generated PDF reports
- **`results/`**: JSON result files

### Configuration (`config/`)

Configuration and deployment files:

- Dependencies (`requirements.txt`)
- Package setup (`setup.py`)
- Docker configuration

## Import Patterns

### From Scripts

Scripts should add `src` to the path and import from `wafr`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wafr.agents.orchestrator import create_orchestrator
from wafr.ag_ui.orchestrator_integration import create_agui_orchestrator
```

### Within Package

Within the `wafr` package, use absolute imports:

```python
from wafr.agents.orchestrator import create_orchestrator
from wafr.agents.wa_tool_agent import WAToolAgent
from wafr.ag_ui.emitter import WAFREventEmitter
```

## Running the System

### From Project Root

```bash
# Windows
scripts\run_wafr_full.bat "Client Name"

# Linux/Mac
./scripts/run_wafr_full.sh "Client Name"

# Python (any platform)
python scripts/run_wafr_full.py --wa-tool --client-name "Client Name"
```

### From Scripts Directory

```bash
cd scripts
python run_wafr_full.py --wa-tool --client-name "Client Name"
```

## Benefits of This Structure

1. **Separation of Concerns**: Source code, scripts, docs, and data are clearly separated
2. **Professional Standard**: Follows Python packaging best practices
3. **Maintainability**: Easy to find and update files
4. **Scalability**: Easy to add new modules, agents, or features
5. **Clean Imports**: Clear import paths, no circular dependencies
6. **Organized Outputs**: All generated files in one place
7. **Version Control**: Easy to ignore output/ and venv/ directories

## Migration Notes

If you have existing code that imports from the old structure:

- `from agents.X import Y` → `from wafr.agents.X import Y`
- `from ag_ui.X import Y` → `from wafr.ag_ui.X import Y`
- `from models.X import Y` → `from wafr.models.X import Y`

Update script paths:
- Old: `python run_wafr_full.py` (from root)
- New: `python scripts/run_wafr_full.py` (from root) or `python run_wafr_full.py` (from scripts/)

Update data paths:
- Old: `knowledge_base/file.json`
- New: `data/knowledge_base/file.json`

Update output paths:
- Old: `logs/file.log`, `wafr_results.json`
- New: `output/logs/file.log`, `output/results/wafr_results.json`

