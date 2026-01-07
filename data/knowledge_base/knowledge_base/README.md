# WAFR Knowledge Base

Semantic search knowledge base for WAFR documentation using AWS S3Vectors.

## Overview

This knowledge base system:
1. **Stores** content as vectors in AWS S3Vectors for semantic search
2. **Provides** semantic search capabilities for agents to frame questions based on customer use cases

## Architecture

```
┌─────────────────┐
│  Embedding      │
│    Service      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   S3Vectors     │
│  Knowledge Base │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent Query    │
│    Interface    │
└─────────────────┘
```

## Setup

### 1. Prerequisites

```bash
pip install beautifulsoup4 requests boto3
```

### 2. AWS Configuration

Ensure AWS credentials are configured:
```bash
aws configure
```

Required permissions:
- `s3vectors:CreateVectorBucket`
- `s3vectors:CreateIndex`
- `s3vectors:PutVectors`
- `s3vectors:QueryVectors`
- `bedrock:InvokeModel` (for embeddings)

### 3. Environment Variables (Optional)

```bash
export WAFR_KB_BUCKET=wafr-knowledge-base
export WAFR_KB_INDEX=wafr-docs-index
export AWS_REGION=us-east-1
export EMBEDDING_PROVIDER=bedrock  # or 'openai'
```

### 4. Load Knowledge Base Data

Data should be loaded into S3Vectors using the appropriate scripts or tools.

## Usage

### Query Knowledge Base

```python
from knowledge_base.kb_query import get_knowledge_base

kb = get_knowledge_base()

# Search for best practices
results = kb.get_wafr_best_practices('SEC', 'SEC_01')

# Get implementation guidance
results = kb.get_implementation_guidance('multi-AZ deployment')

# General search
results = kb.search('How to design for failure?', top_k=5)
```

### Integration with Agents

```python
from knowledge_base.kb_query import get_knowledge_base

kb = get_knowledge_base()

# In your agent code
def frame_question(customer_use_case: str):
    # Get relevant WAFR guidance
    guidance = kb.search(f"WAFR best practices for {customer_use_case}", top_k=3)
    
    # Use guidance to frame question
    # ...
```

## Components

### `s3vectors_client.py`
- S3Vectors client wrapper
- Handles bucket/index creation
- Stores and queries vectors

### `embedding_service.py`
- Generates embeddings using AWS Bedrock or OpenAI
- Supports batch processing

### `kb_query.py`
- High-level query interface
- Convenience methods for common queries

## Data Sources

Knowledge base data is loaded from pre-processed sources and stored in S3Vectors.

## Notes

- Embeddings use Amazon Titan Embeddings (1536 dimensions)
- Content is chunked into ~1000 character segments with 200 character overlap
- Vector keys are generated using MD5 hashes
- Metadata includes source URL, title, and chunk information

## Troubleshooting

### Error: NoSuchBucket
- Ensure bucket name is correct
- Check AWS credentials and region

### Error: Embedding generation failed
- Verify Bedrock access and model availability
- Check `amazon.titan-embed-text-v1` is available in your region

### Error: Rate limiting
- Use batch processing for embeddings
- Reduce batch size if needed
