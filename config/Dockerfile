FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY agents/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agents/ ./agents/
COPY schemas/ ./schemas/ 2>/dev/null || true
COPY knowledge_base/ ./knowledge_base/ 2>/dev/null || true
COPY run_wafr.py .
COPY test_e2e_wafr.py .

# Set environment variables
ENV AWS_REGION=us-east-1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create directories
RUN mkdir -p /app/reports /app/transcripts

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "from agents.orchestrator import create_orchestrator; create_orchestrator()" || exit 1

# Default command
CMD ["python", "run_wafr.py", "--help"]

