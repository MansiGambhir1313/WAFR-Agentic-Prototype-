"""
Main entry point for WAFR Agent System.

Provides CLI interface for processing transcripts through the
multi-agent WAFR assessment pipeline.
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from wafr.agents.orchestrator import create_orchestrator


# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# File Loading Functions
# =============================================================================

def load_transcript(file_path: str) -> str:
    """
    Load transcript from file.
    
    Args:
        file_path: Path to transcript file
        
    Returns:
        Transcript content as string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_wafr_schema(schema_path: str | None = None) -> dict[str, Any]:
    """
    Load WAFR schema from file.
    
    Args:
        schema_path: Optional path to schema file. If None, uses default location.
        
    Returns:
        WAFR schema dictionary
    """
    if schema_path is None:
        schema_path = Path(__file__).parent.parent / "schemas" / "wafr-schema.json"

    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="WAFR Multi-Agent System")

    parser.add_argument(
        "transcript",
        type=str,
        help="Path to transcript file",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to WAFR schema JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation",
    )

    return parser.parse_args()


def save_results(results: dict[str, Any], output_path: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary to save
        output_path: Path to output file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to: %s", output_path)


def print_results(results: dict[str, Any]) -> None:
    """
    Print results to stdout.
    
    Args:
        results: Results dictionary to print
    """
    print(json.dumps(results, indent=2, ensure_ascii=False))


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_arguments()

    # Generate session ID if not provided
    session_id = args.session_id or str(uuid.uuid4())

    logger.info("Starting WAFR processing for session: %s", session_id)

    try:
        # Load transcript
        logger.info("Loading transcript from: %s", args.transcript)
        transcript = load_transcript(args.transcript)

        # Load WAFR schema
        logger.info("Loading WAFR schema")
        wafr_schema = load_wafr_schema(args.schema)

        # Create orchestrator
        logger.info("Initializing agent orchestrator")
        orchestrator = create_orchestrator(wafr_schema)

        # Process transcript
        logger.info("Processing transcript through agent pipeline")
        results = orchestrator.process_transcript(
            transcript=transcript,
            session_id=session_id,
            generate_report=not args.no_report,
        )

        # Output results
        if args.output:
            save_results(results, args.output)
        else:
            print_results(results)

        # Check status and return
        if results.get("status") == "completed":
            logger.info("Processing completed successfully")
            summary = results.get("summary", {})
            logger.info("Summary: %s", summary)
            return 0

        logger.error("Processing failed: %s", results.get("error", "Unknown error"))
        return 1

    except Exception as e:
        logger.error("Error: %s", str(e), exc_info=True)
        return 1


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())