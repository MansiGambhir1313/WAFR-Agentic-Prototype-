# Automated WAFR System - End-to-End Implementation

Automated AWS Well-Architected Framework Review system with multi-agent AI processing.

## Architecture

- **Frontend**: React + TypeScript with WebSocket real-time updates
- **Backend**: AWS Lambda (Python) + API Gateway + WebSocket API
- **AI**: Amazon Bedrock (Claude Sonnet) for multi-agent processing
- **Storage**: S3 (files/transcripts/reports), DynamoDB (WAFR data)
- **Orchestration**: Step Functions for workflow management
- **Knowledge Base**: OpenSearch for semantic search (optional)

## Quick Start

### Prerequisites

```bash
# AWS CLI v2
aws --version

# Node.js 18+
node --version

# Python 3.11+
python --version

# AWS CDK v2
npm install -g aws-cdk
cdk --version
```

### Setup

```bash
# Install dependencies
cd infrastructure && npm install
cd ../frontend && npm install

# Configure AWS credentials
aws configure

# Bootstrap CDK (first time only)
cd infrastructure
cdk bootstrap aws://ACCOUNT_ID/REGION

# Deploy infrastructure
cdk deploy --all
```

### Development

```bash
# Backend development
cd lambdas/agent-orchestrator
pip install -r requirements.txt
python -m pytest tests/

# Frontend development
cd frontend
npm run dev
```

## Project Structure

```
wafr-automated/
├── infrastructure/          # CDK infrastructure code
├── lambdas/                 # Lambda function handlers
├── agents/                  # Agent implementations and prompts
├── frontend/                # React frontend application
├── schemas/                 # Data schemas and WAFR questions
├── scripts/                 # Utility scripts
└── tests/                   # Test suites
```

## Deployment

See [DEPLOYMENT.md](./docs/DEPLOYMENT.md) for detailed deployment instructions.

## Documentation

- [System Architecture](./docs/ARCHITECTURE.md)
- [API Reference](./docs/API.md)
- [Agent System](./docs/AGENTS.md)

