# WAFR Prototype Project - Complete Code Explanation
## For Non-Technical Readers

This document explains every file in the WAFR (Well-Architected Framework Review) project in simple, easy-to-understand language.

---

## What This Project Does (Overview)

Imagine you're an AWS consultant who just finished a workshop with a client. During the workshop, you discussed their cloud architecture, security practices, costs, and more. You recorded everything in a transcript (like meeting notes).

**This project automatically:**
1. Reads that transcript
2. Extracts important information (like "they use S3 for storage" or "they have security concerns")
3. Maps that information to AWS Well-Architected Framework questions
4. Creates answers to those questions automatically
5. Generates a professional assessment report
6. Optionally creates a workload in AWS Well-Architected Tool

Think of it as an AI assistant that does the tedious work of analyzing workshop transcripts and creating assessment reports.

---

## Project Structure Overview

The project is organized like a restaurant kitchen:
- **Main files** (`run_wafr.py`, `lambda_handler.py`) = The front door where orders come in
- **Orchestrator** (`orchestrator.py`) = The head chef who coordinates everything
- **Agents** (various agent files) = Specialized cooks (one for understanding, one for mapping, etc.)
- **Config files** = Recipe books with settings
- **Knowledge base** = Reference materials

---

## File-by-File Explanation

### 1. `run_wafr.py` - The Command-Line Interface

**What it does:** This is the main entry point when you run the program from your computer's command line (terminal).

**Line-by-line explanation:**

```python
"""
WAFR Orchestrator - Terminal Interface
Run WAFR analysis directly from command line
"""
```
- Lines 1-4: This is a comment (documentation) explaining what the file does. It's like a label on a box.

```python
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
```
- Lines 5-9: These lines import (bring in) tools from Python's standard library:
  - `sys`: System-specific functions (like exiting the program)
  - `json`: For working with JSON data (a common data format)
  - `argparse`: For reading command-line arguments (like file names you type)
  - `Path`: For working with file paths
  - `datetime`: For working with dates and times

```python
sys.path.insert(0, str(Path(__file__).parent))
```
- Line 12: This tells Python where to find other project files. `__file__` is the current file, `.parent` is the folder it's in.

```python
from agents.orchestrator import create_orchestrator
```
- Line 14: This imports the `create_orchestrator` function from the orchestrator file. Think of it as getting a tool from another toolbox.

```python
def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")
```
- Lines 17-22: This is a function (a reusable piece of code) that prints a nice-looking header. It prints 70 equal signs, then the text, then 70 more equal signs. The `\n` means "new line."

```python
def print_section(title):
    """Print a section header."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}\n")
```
- Lines 24-29: Similar function but uses dashes instead of equal signs for section headers.

```python
def print_results(results):
    """Print formatted results."""
    # Summary
    print_section("📊 PROCESSING SUMMARY")
    summary = results.get('summary', {})
    print(f"  Status: {results.get('status', 'unknown')}")
    print(f"  Total Insights: {summary.get('total_insights', 0)}")
    # ... more print statements
```
- Lines 31-136: This function takes the results and prints them in a nice, organized way. It shows:
  - Processing summary (how many insights, mappings, answers, gaps)
  - Agent results (what each agent did)
  - Top insights, mappings, and gaps
  - AWS Well-Architected Tool results if available

```python
def main():
    parser = argparse.ArgumentParser(
        description='WAFR Orchestrator - Terminal Interface',
        # ... more configuration
    )
```
- Lines 138-207: The `main()` function is where the program starts. It:
  1. Creates an argument parser (to understand command-line options)
  2. Defines what options are available (like `--wa-tool`, `--client-name`)
  3. Reads the transcript file
  4. Creates the orchestrator
  5. Processes the transcript
  6. Prints results
  7. Saves results to a file if requested

**Key parts:**
- `parser.add_argument('transcript', ...)`: Expects a transcript file path
- `parser.add_argument('--wa-tool', ...)`: Optional flag to enable AWS Well-Architected Tool integration
- `parser.add_argument('--client-name', ...)`: Optional client name
- `parser.add_argument('--output', ...)`: Optional output file path

```python
if args.transcript == '-':
    transcript = sys.stdin.read()
else:
    transcript_path = Path(args.transcript)
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = f.read()
```
- Lines 217-227: This reads the transcript. If you pass `-`, it reads from standard input (like piping). Otherwise, it reads from the file you specified.

```python
orchestrator = create_orchestrator()
results = orchestrator.process_transcript(
    transcript=transcript,
    session_id=session_id,
    generate_report=not args.no_report,
    create_wa_workload=args.wa_tool,
    client_name=args.client_name,
    environment=args.environment,
    existing_workload_id=args.workload_id
)
```
- Lines 240-261: This creates the orchestrator (the coordinator) and tells it to process the transcript. It passes all the options you specified.

```python
if __name__ == '__main__':
    main()
```
- Lines 297-298: This is Python's way of saying "if someone runs this file directly (not importing it), run the main() function."

---

### 2. `lambda_handler.py` - AWS Lambda Function Entry Point

**What it does:** This is the entry point when the code runs on AWS Lambda (a serverless computing service). It's similar to `run_wafr.py` but designed for cloud execution.

**Key differences from `run_wafr.py`:**
- It receives events from AWS API Gateway (web requests)
- It returns HTTP responses
- It handles errors in a web-friendly way

**Line-by-line explanation:**

```python
# Fix OpenTelemetry issues in Lambda FIRST
import os
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['OTEL_PYTHON_CONTEXT'] = 'contextvars_context'
```
- Lines 5-8: This disables OpenTelemetry (a monitoring tool) that can cause issues in Lambda. It's like turning off a feature that's causing problems.

```python
def get_orchestrator():
    """Get or create orchestrator instance (singleton pattern)"""
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initializing WAFR orchestrator...")
        _orchestrator = create_orchestrator()
        logger.info("Orchestrator initialized successfully")
    return _orchestrator
```
- Lines 32-39: This function uses a "singleton pattern" - it creates the orchestrator once and reuses it. This is important in Lambda because the same container can handle multiple requests, and we don't want to recreate everything each time.

```python
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for WAFR processing.
    ...
    """
```
- Lines 42-68: This is the main Lambda function. AWS calls this when a request comes in.

**The function:**
1. Extracts data from the event (the web request)
2. Validates that a transcript was provided
3. Gets the orchestrator
4. Processes the transcript
5. Returns a formatted HTTP response

```python
if isinstance(event.get('body'), str):
    body = json.loads(event['body'])
else:
    body = event.get('body', event)
```
- Lines 71-76: This handles two cases:
  - If the body is a string (from API Gateway), it parses it as JSON
  - If it's already a dictionary, it uses it directly

```python
transcript = body.get('transcript', '')
if not transcript:
    return {
        'statusCode': 400,
        'body': json.dumps({
            'error': 'transcript is required',
            ...
        })
    }
```
- Lines 78-86: This checks if a transcript was provided. If not, it returns an error response (HTTP 400 = Bad Request).

```python
results = orchestrator.process_transcript(
    transcript=transcript,
    session_id=session_id,
    generate_report=generate_report,
    create_wa_workload=create_wa_workload,
    client_name=client_name,
    environment=environment,
    existing_workload_id=existing_workload_id
)
```
- Lines 102-110: Same as in `run_wafr.py` - processes the transcript.

```python
return {
    'statusCode': 200,
    'headers': {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
    },
    'body': json.dumps(response_body, default=str)
}
```
- Lines 153-160: Returns a successful HTTP response. The headers tell the browser it's JSON data and allows cross-origin requests (CORS).

```python
def health_check_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Health check endpoint for Lambda.
    ...
    """
```
- Lines 178-218: A simple health check function that returns "healthy" if the system is working. Useful for monitoring.

---

### 3. `agents/orchestrator.py` - The Coordinator

**What it does:** This is the "head chef" that coordinates all the specialized agents. It runs them in the right order and passes data between them.

**Key concepts:**
- **Orchestrator**: The coordinator
- **Agents**: Specialized workers (Understanding Agent, Mapping Agent, etc.)
- **Pipeline**: A series of steps that process the transcript

**Line-by-line explanation:**

```python
class WafrOrchestrator:
    """
    Orchestrates the multi-agent WAFR processing pipeline.
    """
```
- Lines 39-42: This defines a class (a blueprint for creating objects). The orchestrator is an object that coordinates everything.

```python
def __init__(self, wafr_schema: Optional[Dict] = None):
    """
    Initialize orchestrator with all agents.
    """
    if wafr_schema is None:
        from agents.wafr_context import load_wafr_schema
        wafr_schema = load_wafr_schema()
    
    self.wafr_schema = wafr_schema
    
    # Initialize all agents with WAFR schema context
    self.understanding_agent = create_understanding_agent(wafr_schema)
    self.mapping_agent = create_mapping_agent(wafr_schema)
    self.confidence_agent = create_confidence_agent(wafr_schema)
    # ... more agents
```
- Lines 44-70: The `__init__` method (constructor) runs when you create an orchestrator. It:
  1. Loads the WAFR schema (the list of questions)
  2. Creates all the specialized agents
  3. Stores them so they can be used later

**The Processing Pipeline:**

```python
def process_transcript(
    self,
    transcript: str,
    session_id: str,
    generate_report: bool = True,
    create_wa_workload: bool = False,
    # ... more parameters
) -> Dict[str, Any]:
```
- Lines 72-97: This is the main processing function. It takes a transcript and processes it through all the agents.

**Step 1: Understanding Agent**
```python
# Step 1: Understanding Agent - Extract insights
step_start = time.time()
self.logger.info("Step 1: Extracting insights from transcript")
insights_result = self.understanding_agent.process(transcript, session_id)
results['steps']['understanding'] = insights_result
insights = insights_result.get('insights', [])
```
- Lines 110-121: The Understanding Agent reads the transcript and extracts important information (insights). Like highlighting important sentences in a document.

**Step 2: Mapping Agent**
```python
# Step 2: Mapping Agent - Map to WAFR questions
mapping_result = self.mapping_agent.process(insights, session_id)
results['steps']['mapping'] = mapping_result
mappings = mapping_result.get('mappings', [])
```
- Lines 138-148: The Mapping Agent takes those insights and matches them to specific WAFR questions. Like matching "we use S3" to the question "How do you store data?"

**Step 3: Confidence Agent**
```python
# Step 3: Confidence Agent - Validate evidence
confidence_result = self.confidence_agent.process(mappings, transcript, session_id)
```
- Lines 166-187: The Confidence Agent checks if the answers are reliable. It verifies that the evidence (quotes from transcript) actually supports the answers.

**Step 4: Gap Detection**
```python
# Step 4: Gap Detection - Identify missing questions
gap_result = self.gap_detection_agent.process(
    answered_questions=answered_questions,
    pillar_coverage=pillar_coverage,
    session_id=session_id,
    transcript=transcript
)
```
- Lines 200-218: The Gap Detection Agent finds questions that weren't answered. Like checking which questions on a test you didn't answer.

**Step 5: Prompt Generator**
```python
# Step 5: Generate smart prompts for gaps
for gap in gaps:
    prompt = self.prompt_generator_agent.process(gap, question_data)
    gap_prompts.append(prompt)
```
- Lines 224-246: For unanswered questions, this agent creates smart prompts to ask the client later.

**Step 6: Scoring Agent**
```python
# Step 6: Scoring Agent - Score and rank answers
scoring_result = self.scoring_agent.process(
    answers=validated_answers,
    wafr_schema=self.wafr_schema,
    session_id=session_id
)
```
- Lines 248-273: The Scoring Agent grades the answers. Like giving a grade (A, B, C, D, F) to each answer.

**Step 7: Report Generation**
```python
# Step 7: Report Generation (optional)
if generate_report:
    report_result = self.report_agent.process(assessment_data, session_id)
    results['steps']['report'] = report_result
```
- Lines 275-304: The Report Agent creates a final PDF report with all the findings.

**Step 8: WA Tool Integration**
```python
# Step 8: WA Tool Integration (optional)
if create_wa_workload:
    workload = self.wa_tool_agent.create_workload_from_transcript(...)
    # Populate answers in AWS Well-Architected Tool
    populate_result = self.wa_tool_agent.populate_answers_from_analysis(...)
```
- Lines 306-406: If requested, this creates a workload in AWS Well-Architected Tool and fills in the answers automatically.

**Error Handling:**
```python
try:
    # ... process step
except Exception as e:
    self.logger.error(f"Understanding agent failed: {str(e)}")
    results['steps']['understanding'] = {
        'error': str(e),
        'insights': [],
        'agent': 'understanding'
    }
    insights = []
```
- Throughout the code, each step is wrapped in try/except blocks. If one agent fails, the system continues with the others (graceful degradation).

---

### 4. `agents/base_agent.py` - Base Class for Agents

**What it does:** This is a template/blueprint that other agents can inherit from. It provides common functionality like calling AWS Bedrock (the AI service).

**Note:** The actual agents use the Strands framework, but this base class shows the pattern.

**Key concepts:**
- **Base class**: A template that other classes can extend
- **Inheritance**: When a class uses another class as a starting point
- **Abstract method**: A method that must be implemented by child classes

**Line-by-line explanation:**

```python
class BaseAgent(ABC):
    """Base class for all WAFR agents."""
```
- Lines 14-15: This defines a base class. `ABC` means "Abstract Base Class" - it can't be used directly, only extended.

```python
def __init__(self, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
    """
    Initialize base agent with Bedrock client.
    """
    self.model_id = model_id
    self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    self.logger = logger
```
- Lines 17-26: The constructor sets up:
  - The AI model to use (Claude Sonnet)
  - A connection to AWS Bedrock (the AI service)
  - A logger for recording what happens

```python
def invoke_model(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    **kwargs
) -> Dict[str, Any]:
```
- Lines 28-48: This method calls the AI model. It:
  - Takes a prompt (the question/instruction)
  - Optionally takes a system prompt (instructions for how the AI should behave)
  - Sends it to Bedrock
  - Returns the AI's response

```python
messages = []
if system_prompt:
    messages.append({
        'role': 'system',
        'content': system_prompt
    })
messages.append({
    'role': 'user',
    'content': prompt
})
```
- Lines 49-60: This builds the message structure. AI models understand "system" messages (instructions) and "user" messages (the actual question).

```python
response = self.bedrock.invoke_model(
    modelId=self.model_id,
    body=json.dumps({
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': max_tokens,
        'temperature': temperature,
        'messages': messages,
        **kwargs
    })
)
```
- Lines 63-72: This actually calls AWS Bedrock. It sends the messages and configuration.

```python
@abstractmethod
def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process input data and return results.
    """
    pass
```
- Lines 111-122: This is an abstract method. Child classes MUST implement this. It's like saying "every agent must have a process method, but each implements it differently."

---

### 5. `agents/understanding_agent.py` - The Insight Extractor

**What it does:** This agent reads the transcript and extracts important information (insights). It's like a smart highlighter that finds the important parts.

**Key concepts:**
- **Insight**: A piece of important information extracted from the transcript
- **Insight types**: decision, service, constraint, risk
- **Transcript quote**: The exact words from the transcript that support the insight

**Line-by-line explanation:**

```python
def get_understanding_system_prompt(wafr_schema: Optional[Dict] = None) -> str:
    """Generate enhanced system prompt with WAFR context."""
    base_prompt = """
You are an expert AWS Solutions Architect analyzing Well-Architected Framework Review (WAFR) workshop transcripts to extract architecture-relevant information.

Your task is to extract:
1. Architecture decisions discussed (what was decided, why)
2. AWS services mentioned (with context on how they're used)
3. Constraints and requirements stated (performance, cost, compliance, security)
4. Risks or concerns raised (security, availability, scalability, reliability)
5. Best practices mentioned or implemented
6. Gaps or missing capabilities identified
...
```
- Lines 25-65: This function creates the instructions for the AI. It tells the AI:
  - What role to play (AWS Solutions Architect)
  - What to look for (decisions, services, constraints, risks)
  - How to format the output
  - What rules to follow

```python
@tool
def extract_insights(
    transcript: str,
    insight_type: str,
    content: str,
    transcript_quote: str,
    speaker: str = None,
    timestamp: str = None
) -> Dict:
```
- Lines 74-104: This is a "tool" function that the AI can call. The `@tool` decorator makes it available to the AI agent. It structures an insight with all the required fields.

```python
class UnderstandingAgent:
    """Agent that extracts architecture insights from transcripts."""
    
    def __init__(self, wafr_schema: Optional[Dict] = None):
        # Load WAFR schema if not provided
        if wafr_schema is None:
            wafr_schema = load_wafr_schema()
        
        self.wafr_schema = wafr_schema
        system_prompt = get_understanding_system_prompt(wafr_schema)
        
        # Create Strands agent
        model = get_strands_model(DEFAULT_MODEL_ID)
        self.agent = Agent(
            system_prompt=system_prompt,
            name='UnderstandingAgent',
            model=model
        )
```
- Lines 123-149: The constructor:
  1. Loads the WAFR schema
  2. Creates the system prompt (instructions for the AI)
  3. Creates a Strands agent (the AI worker)

```python
def process(self, transcript: str, session_id: str) -> Dict[str, Any]:
    """
    Process transcript and extract insights.
    """
    # Segment transcript if too long
    segments_data = smart_segment_transcript(transcript, max_segment_length=5000)
    segments = [seg['text'] for seg in segments_data]
```
- Lines 172-187: The main processing function:
  1. Splits long transcripts into smaller chunks (AI models have token limits)
  2. Processes each segment

```python
def process_segment(segment_data: Dict) -> List[Dict]:
    idx = segment_data['index']
    segment = segment_data['text']
    
    prompt = f"""Extract architecture insights from this transcript segment. Return ONLY a valid JSON array.
    
TRANSCRIPT SEGMENT:
{segment[:4000]}

EXTRACT:
- Decisions: Technology choices, design patterns, architectural decisions
- Services: AWS services (EC2, S3, RDS, Lambda, etc.) with usage context
- Constraints: Requirements, limitations, compliance, SLAs, security
- Risks: Security, availability, scalability, cost concerns
...
```
- Lines 190-220: For each segment, it:
  1. Creates a prompt asking the AI to extract insights
  2. Calls the AI
  3. Parses the response
  4. Validates the insights

```python
# Process segments in batches
if len(segments_data) > 1:
    results = batch_process(
        segments_data,
        process_segment,
        batch_size=3,
        max_workers=3,
        timeout=120.0
    )
    all_insights = [insight for result in results for insight in result]
```
- Lines 256-267: If there are multiple segments, it processes them in parallel (at the same time) for speed.

```python
# Deduplicate insights
all_insights = deduplicate_insights(all_insights)
```
- Line 273: Removes duplicate insights (same information found multiple times).

```python
def _fallback_extraction(self, transcript: str, session_id: str) -> List[Dict]:
    """
    Fallback extraction method when main agent returns no insights.
    Uses simple keyword-based extraction to find at least some information.
    """
    insights = []
    
    # Look for AWS services mentioned
    aws_services = [
        'EC2', 'S3', 'RDS', 'Lambda', 'CloudWatch', 'X-Ray', 'IAM', 
        'VPC', 'Route 53', 'Auto Scaling', 'ElastiCache', 'EKS', 
        ...
    ]
```
- Lines 397-454: If the AI fails to extract insights, this fallback method uses simple keyword matching to find at least some information.

---

### 6. `agents/mapping_agent.py` - The Question Mapper

**What it does:** This agent takes the insights from the Understanding Agent and maps them to specific WAFR questions. Like matching answers to questions on a test.

**Key concepts:**
- **Mapping**: Connecting an insight to a WAFR question
- **Relevance score**: How well the insight answers the question (0.0 to 1.0)
- **Answer coverage**: Whether it fully or partially answers the question
- **Pillar**: One of the 6 WAFR pillars (Operational Excellence, Security, etc.)

**Line-by-line explanation:**

```python
def get_mapping_system_prompt(wafr_schema: Optional[Dict] = None) -> str:
    """Generate enhanced system prompt with WAFR context."""
    base_prompt = """
You are a WAFR (AWS Well-Architected Framework Review) expert with deep knowledge of all six pillars and their questions.

Your task is to map architecture insights from workshop transcripts to specific WAFR questions.

THE 6 WAFR PILLARS:
1. Operational Excellence (OPS) - Running and monitoring systems to deliver business value
2. Security (SEC) - Protecting information and assets
3. Reliability (REL) - Recovering from failures and meeting demand
4. Performance Efficiency (PERF) - Using resources efficiently
5. Cost Optimization (COST) - Managing costs effectively
6. Sustainability (SUS) - Minimizing environmental impact
...
```
- Lines 24-64: Creates instructions for the AI, explaining:
  - What the 6 pillars are
  - How to map insights to questions
  - What information to include in each mapping

```python
def process(self, insights: List[Dict], session_id: str) -> Dict[str, Any]:
    """
    Map insights to WAFR questions.
    """
    if not insights:
        return {
            'session_id': session_id,
            'total_mappings': 0,
            'mappings': [],
            'pillar_coverage': {},
            'agent': 'mapping'
        }
```
- Lines 168-188: The main processing function. If there are no insights, it returns an empty result.

```python
def process_insight(insight: Dict) -> List[Dict]:
    # Get question context for better mapping
    question_contexts = []
    if self.wafr_schema:
        for pillar in self.wafr_schema.get('pillars', []):
            for question in pillar.get('questions', []):
                keywords = question.get('keywords', [])
                if any(kw.lower() in insight.get('content', '').lower() or 
                       kw.lower() in insight.get('transcript_quote', '').lower() 
                       for kw in keywords):
                    q_context = get_question_context(question.get('id'), self.wafr_schema)
                    if q_context:
                        question_contexts.append(q_context[:500])
```
- Lines 191-203: For each insight, it finds relevant questions by matching keywords. This helps the AI understand which questions might be relevant.

```python
prompt = f"""You are a WAFR expert mapping architecture insights to AWS Well-Architected Framework questions.

ARCHITECTURE INSIGHT TO MAP:
Type: {insight.get('insight_type', 'unknown')}
Content: {insight.get('content', '')}
Evidence Quote: {insight.get('transcript_quote', '')}

AVAILABLE WAFR QUESTIONS:
{questions_json}

TASK: Map this insight to relevant WAFR questions. Return ONLY a valid JSON array of mappings.
...
```
- Lines 231-265: Creates a prompt that:
  1. Shows the insight to map
  2. Lists available WAFR questions
  3. Asks the AI to create mappings
  4. Specifies the output format

```python
# Calculate pillar coverage
pillar_coverage = {}
for mapping in all_mappings:
    pillar = mapping.get('pillar', 'UNKNOWN')
    if pillar not in pillar_coverage:
        pillar_coverage[pillar] = {
            'total_mappings': 0,
            'questions_addressed': set()
        }
    pillar_coverage[pillar]['total_mappings'] += 1
    pillar_coverage[pillar]['questions_addressed'].add(mapping.get('question_id'))
```
- Lines 314-334: After mapping, it calculates how many questions were answered for each pillar. This shows coverage (like "we answered 15 out of 20 Security questions").

---

### 7. `agents/config.py` - Configuration Settings

**What it does:** This file stores all the configuration settings (like a settings menu). It's a central place to change how the system behaves.

**Line-by-line explanation:**

```python
DEFAULT_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0"  # Cross-region inference profile
)
```
- Lines 22-25: Sets the default AI model. `os.getenv()` checks if an environment variable is set, otherwise uses the default value.

```python
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
```
- Line 26: Sets the AWS region where Bedrock is located.

```python
UNDERSTANDING_AGENT_TEMPERATURE = 0.1  # Low for factual extraction
MAPPING_AGENT_TEMPERATURE = 0.2
CONFIDENCE_AGENT_TEMPERATURE = 0.1  # Very low for consistent validation
```
- Lines 29-32: Temperature controls how "creative" the AI is. Lower = more consistent/factual, Higher = more creative. Understanding and Confidence agents need low temperature for accuracy.

```python
MAX_TRANSCRIPT_SEGMENT_LENGTH = 5000
MAX_TOKENS_DEFAULT = 4096
```
- Lines 36-37: Limits:
  - Maximum characters per transcript segment
  - Maximum tokens (words/characters) the AI can generate

```python
GRADE_A_THRESHOLD = 90
GRADE_B_THRESHOLD = 80
GRADE_C_THRESHOLD = 70
GRADE_D_THRESHOLD = 60
```
- Lines 45-48: Score thresholds for grading answers (like a grading scale).

---

### 8. `agents/wafr_context.py` - WAFR Knowledge Loader

**What it does:** This file loads and provides WAFR schema information (the list of questions) to agents.

**Line-by-line explanation:**

```python
def load_wafr_schema(schema_path: Optional[str] = None) -> Dict:
    """Load WAFR schema from file."""
    if schema_path is None:
        possible_paths = [
            'knowledge_base/wafr-schema.json',
            'schemas/wafr-schema.json',
            # ... more paths
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                schema_path = path
                break
```
- Lines 13-26: Tries to find the WAFR schema file in common locations.

```python
def get_wafr_context_summary(wafr_schema: Dict) -> str:
    """Generate a comprehensive WAFR context summary for agents."""
    context_parts = [
        "AWS WELL-ARCHITECTED FRAMEWORK REVIEW (WAFR) CONTEXT",
        "=" * 70,
        "",
        f"Schema Version: {wafr_schema.get('version', 'unknown')}",
        ...
    ]
    
    for pillar in wafr_schema.get('pillars', []):
        pillar_id = pillar.get('id', 'UNKNOWN')
        pillar_name = pillar.get('name', 'Unknown')
        description = pillar.get('description', '')
        questions = pillar.get('questions', [])
        
        context_parts.append(f"{pillar_id} - {pillar_name}")
        context_parts.append(f"  Description: {description}")
        context_parts.append(f"  Questions: {len(questions)}")
```
- Lines 38-75: Creates a summary of the WAFR schema to give context to AI agents. It lists all pillars and their questions.

```python
def get_question_context(question_id: str, wafr_schema: Dict) -> Optional[str]:
    """Get detailed context for a specific question."""
    for pillar in wafr_schema.get('pillars', []):
        for question in pillar.get('questions', []):
            if question.get('id') == question_id:
                context_parts = [
                    f"QUESTION: {question.get('text', '')}",
                    f"Pillar: {pillar.get('name', '')} ({pillar.get('id', '')})",
                    f"Criticality: {question.get('criticality', 'medium')}",
                    ...
                ]
```
- Lines 78-121: Gets detailed information about a specific question, including:
  - The question text
  - Which pillar it belongs to
  - How critical it is
  - Best practices
  - Related AWS services

---

## How It All Works Together

1. **User runs the program** (`run_wafr.py` or `lambda_handler.py`)
2. **Orchestrator is created** and initializes all agents
3. **Understanding Agent** reads the transcript and extracts insights
4. **Mapping Agent** matches insights to WAFR questions
5. **Confidence Agent** validates the answers
6. **Gap Detection Agent** finds unanswered questions
7. **Prompt Generator** creates prompts for gaps
8. **Scoring Agent** grades the answers
9. **Report Agent** generates a PDF report
10. **WA Tool Agent** (optional) creates a workload in AWS

Each agent uses AWS Bedrock (Claude AI) to do its specialized task. The orchestrator coordinates everything and handles errors gracefully.

---

### 9. `agents/confidence_agent.py` - The Validator

**What it does:** This agent checks if the answers are reliable and supported by evidence. It prevents "hallucinations" (when AI makes things up).

**Key concepts:**
- **Evidence verification**: Checking if quotes actually exist in the transcript
- **Confidence score**: A number (0.0 to 1.0) indicating how reliable an answer is
- **Validation**: The process of checking if something is correct

**Line-by-line explanation:**

```python
def get_confidence_system_prompt(wafr_schema: Optional[Dict] = None) -> str:
    """Generate enhanced system prompt with WAFR context."""
    base_prompt = """
You are a rigorous fact-checker and evidence validator for WAFR assessments. 

Your critical role is to validate that each WAFR answer is properly supported by transcript evidence and prevent hallucinations.

VALIDATION CRITERIA:
1. Evidence Verification: Does the evidence quote appear VERBATIM or with high similarity in the transcript?
2. Answer Support: Does the evidence actually support the answer given?
3. Interpretation Accuracy: Is the interpretation accurate and not overstated?
...
```
- Lines 22-81: Creates instructions for the AI validator. It tells the AI to:
  - Check if evidence quotes exist in the transcript
  - Verify that answers are supported by evidence
  - Assign confidence scores (HIGH, MEDIUM, LOW)
  - Reject answers without evidence

```python
@tool
def verify_evidence_in_transcript(evidence_quote: str, transcript: str) -> Dict:
    """
    Verify if evidence quote exists in transcript.
    """
    evidence_clean = evidence_quote.strip()
    transcript_lower = transcript.lower()
    evidence_lower = evidence_clean.lower()
    
    # Check for exact match
    if evidence_lower in transcript_lower:
        return {
            'found': True,
            'similarity': 1.0,
            'match_type': 'exact'
        }
```
- Lines 86-100: This tool function checks if a quote exists in the transcript. It:
  1. Cleans the quote (removes extra spaces)
  2. Converts to lowercase for comparison
  3. Checks for exact match
  4. Returns whether it was found and how similar

**Confidence Levels:**
- **HIGH (0.75-1.0)**: Quote found verbatim, answer directly reflects transcript
- **MEDIUM (0.5-0.74)**: Quote found with some similarity, reasonable interpretation
- **LOW (0.0-0.49)**: Quote not found or very different, significant inference required

---

### 10. `agents/scoring_agent.py` - The Grader

**What it does:** This agent grades each answer on multiple dimensions and assigns a letter grade (A, B, C, D, F).

**Key concepts:**
- **Multi-dimensional scoring**: Grading on multiple criteria (confidence, completeness, compliance)
- **Weighted average**: Different criteria have different importance
- **Grade assignment**: Converting scores to letter grades

**Line-by-line explanation:**

```python
def get_scoring_system_prompt(wafr_schema: Optional[Dict] = None) -> str:
    """Generate enhanced system prompt with WAFR context."""
    base_prompt = """
You are an expert WAFR evaluator. You score answers on multiple dimensions using WAFR best practices.

SCORING DIMENSIONS:

1. CONFIDENCE (40% weight): Evidence quality and verification
   - Evidence citations present and verifiable
   - Evidence verified in transcript
   - Source reliability and accuracy
   - No unsupported claims or assumptions

2. COMPLETENESS (30% weight): How well answer addresses the WAFR question
   - Best practices from WAFR schema addressed
   - Answer specificity and detail
   - AWS service mentions with context
   - Coverage of question intent

3. COMPLIANCE (30% weight): Alignment with WAFR best practices
   - Adherence to recommended best practices
   - Anti-pattern penalties (if present)
   - HRI (High-Risk Issue) indicators (negative impact)
   - Recommended AWS services mentioned (positive)
...
```
- Lines 22-66: Creates instructions explaining:
  - Three scoring dimensions (Confidence 40%, Completeness 30%, Compliance 30%)
  - What each dimension measures
  - How to assign grades (A=90-100, B=80-89, etc.)

```python
@tool
def calculate_composite_score(
    confidence_score: float,
    completeness_score: float,
    compliance_score: float
) -> Dict:
    """
    Calculate composite score from three dimensions.
    """
    # Weighted average
    composite = (
        confidence_score * 0.4 +
        completeness_score * 0.3 +
        compliance_score * 0.3
    )
    
    # Assign grade
    if composite >= 90:
        grade = 'A'
    elif composite >= 80:
        grade = 'B'
    elif composite >= 70:
        grade = 'C'
    elif composite >= 60:
        grade = 'D'
    else:
        grade = 'F'
```
- Lines 71-100: This tool calculates the final score:
  1. Multiplies each dimension by its weight (40%, 30%, 30%)
  2. Adds them together
  3. Assigns a letter grade based on the total

**Example:**
- Confidence: 85 (85 × 0.4 = 34)
- Completeness: 90 (90 × 0.3 = 27)
- Compliance: 80 (80 × 0.3 = 24)
- **Total: 85** → **Grade: B**

---

### 11. `agents/report_agent.py` - The Report Writer

**What it does:** This agent generates a comprehensive PDF report with all findings, recommendations, and analysis.

**Key concepts:**
- **Executive summary**: High-level overview for executives
- **Pillar analysis**: Detailed analysis for each of the 6 WAFR pillars
- **Remediation roadmap**: Action plan to fix issues

**Line-by-line explanation:**

```python
def get_report_system_prompt(wafr_schema: Optional[Dict] = None) -> str:
    """Generate enhanced system prompt with WAFR context."""
    base_prompt = """
You are an expert AWS Solutions Architect generating professional 
Well-Architected Framework Review (WAFR) reports.

Your reports must include:
1. Executive Summary - Overall health, key findings, immediate action items
2. Pillar-by-Pillar Analysis - Current state, strengths, gaps, evidence for each of the 6 pillars
3. High-Risk Issues (HRIs) - Critical issues requiring immediate attention
4. 90-Day Remediation Roadmap - Phased action plan (Days 1-30, 31-60, 61-90)
5. Appendix - Evidence citations, confidence scores, clarifications needed
...
```
- Lines 19-48: Creates instructions for report generation. The report must include:
  - Executive summary (for leadership)
  - Detailed pillar analysis (for technical teams)
  - High-risk issues (urgent problems)
  - Remediation roadmap (action plan)
  - Appendix (supporting details)

```python
@tool
def generate_executive_summary(
    overall_health: str,
    key_findings: List[str],
    immediate_actions: List[str],
    confidence_summary: Dict
) -> Dict:
    """
    Generate executive summary section.
    """
    return {
        'overall_health': overall_health,
        'key_findings': key_findings,
        'immediate_actions': immediate_actions,
        'confidence_summary': confidence_summary,
        'generated_at': datetime.utcnow().isoformat()
    }
```
- Lines 57-82: This tool creates the executive summary section. It's a high-level overview that executives can read quickly.

```python
@tool
def generate_pillar_analysis(
    pillar_id: str,
    pillar_name: str,
    current_state: str,
    strengths: List[str],
    gaps: List[str],
    evidence_citations: List[str],
    score: float
) -> Dict:
```
- Lines 85-100: This tool creates detailed analysis for each pillar, including:
  - Current state assessment
  - Strengths (what's working well)
  - Gaps (what needs improvement)
  - Evidence (quotes from transcript)
  - Score and grade

---

### 12. `agents/utils.py` - Helper Functions

**What it does:** This file contains utility functions used by all agents. Think of it as a toolbox with common tools.

**Key functions:**

**1. Retry with Backoff**
```python
def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except all_exceptions as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        delay *= backoff_factor  # Double the delay each time
```
- Lines 15-69: This decorator automatically retries failed operations:
  - If a function fails, it waits and tries again
  - Each retry waits longer (exponential backoff: 1s, 2s, 4s, 8s...)
  - Useful for network calls that might temporarily fail

**2. JSON Extraction**
```python
def extract_json_from_text(text: str, strict: bool = False) -> Dict:
    """
    Extract JSON from text, handling markdown code blocks and plain JSON.
    """
    # Strategy 1: Try to extract JSON from markdown code blocks
    json_patterns = [
        (r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL),  # JSON object in code block
        (r'```(?:json)?\s*(\[.*?\])\s*```', re.DOTALL),   # JSON array in code block
    ]
    
    for pattern, flags in json_patterns:
        matches = re.finditer(pattern, text, flags)
        for match in matches:
            try:
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError:
                continue
```
- Lines 72-142: This function extracts JSON from text. AI responses sometimes include JSON wrapped in markdown code blocks (like ` ```json {...} ``` `). This function:
  1. Tries multiple strategies to find JSON
  2. Handles markdown code blocks
  3. Handles plain JSON
  4. Returns the parsed JSON

**3. Batch Processing**
```python
def batch_process(
    items: List[Any],
    process_func: Callable,
    batch_size: int = 5,
    max_workers: int = 3,
    timeout: float = 120.0
) -> List[Any]:
    """
    Process items in parallel batches.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            future = executor.submit(process_batch, batch, process_func)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                batch_result = future.result(timeout=timeout)
                results.extend(batch_result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
```
- This function processes multiple items in parallel (at the same time) for speed. Instead of processing 100 items one by one (slow), it processes them in batches of 5 simultaneously (fast).

---

### 13. `agents/wa_tool_agent.py` - AWS Well-Architected Tool Integration

**What it does:** This agent integrates with AWS Well-Architected Tool API. It creates workloads and automatically fills in answers.

**Key concepts:**
- **Workload**: A representation of a client's system in AWS Well-Architected Tool
- **Lens**: A set of questions (like "Well-Architected Framework")
- **Answer population**: Automatically filling in answers from transcript analysis

**Line-by-line explanation:**

```python
class WAToolAgent:
    """Agent for managing WA Tool operations based on transcript analysis."""
    
    def __init__(self, region: str = None):
        """
        Initialize WA Tool Agent.
        """
        self.region = region or BEDROCK_REGION
        self.wa_client = WellArchitectedToolClient(region=self.region)
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=self.region)
        self.model_id = DEFAULT_MODEL_ID
```
- Lines 20-40: The constructor sets up:
  - AWS region
  - WA Tool client (for API calls)
  - Bedrock client (for AI calls)
  - AI model to use

```python
def create_workload_from_transcript(
    self,
    transcript_analysis: Dict,
    client_name: str,
    environment: str = 'PRODUCTION',
    aws_regions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a WA Tool workload from transcript analysis.
    """
    # Extract insights from transcript analysis
    insights = transcript_analysis.get('insights', [])
    session_id = transcript_analysis.get('session_id', 'unknown')
    
    # Create workload description from insights
    description = f"WAFR review for {client_name} (Session: {session_id})"
    
    workload = self.wa_client.create_workload(
        workload_name=workload_name,
        description=description,
        environment=environment,
        aws_regions=aws_regions or [self.region],
        lenses=['wellarchitected'],
        tags={
            'Source': 'Agentic_WAFR_System',
            'Client': client_name,
            'SessionId': session_id
        }
    )
```
- Lines 42-110: This function creates a new workload in AWS Well-Architected Tool:
  1. Extracts insights from transcript analysis
  2. Creates a description
  3. Calls the WA Tool API to create the workload
  4. Returns the workload details

```python
def populate_answers_from_analysis(
    self,
    workload_id: str,
    transcript_analysis: Dict,
    transcript: str,
    lens_alias: str = 'wellarchitected',
    mapping_agent=None
) -> Dict[str, Any]:
    """
    Populate ALL WA Tool answers from transcript analysis.
    """
    # Step 1: Get ALL questions from the lens
    all_questions = self._get_all_questions(workload_id, lens_alias)
    
    # Step 2: For each question, find answer from transcript
    # Step 3: Update answer in WA Tool
    # Step 4: Return summary
```
- Lines 112-149: This function automatically fills in answers:
  1. Gets all questions from the workload
  2. For each question, finds the answer from the transcript (using AI)
  3. Updates the answer in WA Tool
  4. Returns a summary of how many were filled

**Why this is useful:** Instead of manually answering 100+ questions in the AWS console, this agent does it automatically based on the transcript.

---

### 14. `agents/gap_detection_agent.py` - The Gap Finder

**What it does:** This agent finds questions that weren't answered in the transcript. These are "gaps" that need to be addressed.

**Key concepts:**
- **Gap**: A question that wasn't answered
- **Pillar coverage**: How many questions were answered per pillar
- **Priority score**: How important it is to answer a gap

**How it works:**
1. Gets list of all WAFR questions
2. Compares with answered questions
3. Identifies unanswered questions
4. Prioritizes them (critical questions first)
5. Returns list of gaps

---

### 15. `agents/prompt_generator_agent.py` - The Question Generator

**What it does:** For unanswered questions (gaps), this agent creates smart prompts to ask the client later.

**Example:**
- Gap: Question about disaster recovery wasn't answered
- Generated prompt: "Based on your architecture discussion, can you tell us about your disaster recovery strategy? Specifically, how do you handle data backups and failover procedures?"

---

### 16. `agents/model_config.py` - AI Model Configuration

**What it does:** Configures which AI model to use and how to connect to it.

**Key concepts:**
- **Model ID**: The specific AI model (like Claude Sonnet)
- **Strands framework**: A framework for building AI agents
- **Model wrapper**: Code that connects to the AI service

---

### 17. `agents/unicode_safe.py` - Text Sanitization

**What it does:** Handles special characters and Unicode issues that can break the system.

**Why it's needed:** Sometimes transcripts contain special characters (like emojis, accented letters) that can cause errors. This file cleans them up.

---

### 18. `agents/wafr_context.py` - WAFR Knowledge Loader

**What it does:** Loads the WAFR schema (list of questions) and provides context to agents.

**Already explained in detail above (section 8).**

---

### 19. `agents/main.py` - Alternative Entry Point

**What it does:** An alternative way to run the system (simpler than `run_wafr.py`).

**Differences from `run_wafr.py`:**
- Simpler interface
- Fewer options
- Good for basic usage

---

### 20. Configuration Files

**`agents/config.py`**: Already explained (section 7)

**`lambda-env.json`**: Environment variables for Lambda deployment

**`requirements.txt`**: List of Python packages needed

---

### 21. Knowledge Base Files

**`knowledge_base/wafr-schema.json`**: The complete list of WAFR questions organized by pillars.

**Structure:**
```json
{
  "version": "1.0",
  "pillars": [
    {
      "id": "OPS",
      "name": "Operational Excellence",
      "description": "...",
      "questions": [
        {
          "id": "OPS_01",
          "text": "How do you understand the health of your workload?",
          "keywords": ["monitoring", "health", "observability"],
          "criticality": "high",
          "best_practices": [...]
        }
      ]
    }
  ]
}
```

---

## Complete Workflow Example

Let's trace through what happens when you run the system:

1. **User runs:** `python run_wafr.py transcript.txt --wa-tool --client-name "Acme Corp"`

2. **`run_wafr.py` reads the transcript file**

3. **Creates orchestrator** which initializes all agents:
   - Understanding Agent
   - Mapping Agent
   - Confidence Agent
   - Gap Detection Agent
   - Prompt Generator Agent
   - Scoring Agent
   - Report Agent
   - WA Tool Agent

4. **Understanding Agent processes transcript:**
   - Splits into segments
   - Calls AI for each segment
   - Extracts insights: "They use S3 for storage", "They have security concerns about encryption"
   - Returns list of insights

5. **Mapping Agent processes insights:**
   - For each insight, finds relevant WAFR questions
   - Creates mappings: Insight "S3 storage" → Question "How do you store data?"
   - Returns list of mappings

6. **Confidence Agent validates mappings:**
   - Checks if evidence quotes exist in transcript
   - Assigns confidence scores
   - Filters out low-confidence answers
   - Returns validated answers

7. **Gap Detection Agent finds gaps:**
   - Compares answered questions to all questions
   - Finds unanswered questions
   - Prioritizes them
   - Returns list of gaps

8. **Prompt Generator creates prompts for gaps**

9. **Scoring Agent grades answers:**
   - Scores each answer (Confidence, Completeness, Compliance)
   - Assigns letter grades (A, B, C, D, F)
   - Returns scored answers

10. **Report Agent generates PDF:**
    - Creates executive summary
    - Creates pillar-by-pillar analysis
    - Creates remediation roadmap
    - Saves PDF file

11. **WA Tool Agent (if enabled):**
    - Creates workload in AWS
    - Auto-fills answers from transcript
    - Creates milestone and generates official report

12. **Results are returned and displayed**

---

## Summary

This project is an AI-powered system that:
- Reads workshop transcripts
- Extracts important information using AI
- Maps it to AWS Well-Architected Framework questions
- Validates and scores the answers
- Generates professional reports
- Optionally integrates with AWS Well-Architected Tool

**Key Technologies:**
- **Python**: Programming language
- **AWS Bedrock**: AI service (Claude Sonnet)
- **Strands Framework**: Framework for building AI agents
- **AWS Well-Architected Tool API**: AWS service for WAFR reviews
- **boto3**: Python library for AWS services

**Architecture Pattern:**
- **Multi-Agent System**: Specialized agents work together
- **Pipeline Processing**: Data flows through agents in sequence
- **Graceful Degradation**: If one agent fails, others continue
- **Parallel Processing**: Multiple items processed simultaneously for speed

It's designed to automate the tedious work of analyzing transcripts and creating assessment reports, saving consultants hours of manual work while ensuring accuracy and consistency.

