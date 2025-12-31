"""
WAFR Knowledge Base Module
Provides semantic search capabilities using AWS S3Vectors
"""
from knowledge_base.s3vectors_client import S3VectorsKnowledgeBase
from knowledge_base.embedding_service import EmbeddingService

__all__ = [
    'S3VectorsKnowledgeBase',
    'EmbeddingService',
]


