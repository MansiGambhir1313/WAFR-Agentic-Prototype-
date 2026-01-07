"""
Generate PDF Report from Existing WAFR Results

This script takes an existing WAFR results JSON file and generates a PDF report
using AWS Well-Architected Tool integration.

Usage:
    python generate_pdf_from_results.py --results wafr_results_wafr-4d2f1c57.json --client-name "Client Name"
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from wafr.agents.orchestrator import create_orchestrator
    from wafr.agents.wa_tool_agent import WAToolAgent
except ImportError as e:
    logger.error(f"Failed to import WAFR components: {e}")
    sys.exit(1)


def load_results(results_file: str) -> Dict[str, Any]:
    """Load WAFR results from JSON file."""
    results_path = Path(results_file)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_file}")
        sys.exit(1)
    
    logger.info(f"Loading results from: {results_file}")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def generate_pdf_from_results(
    results_file: str,
    client_name: str,
    environment: str = "PRODUCTION",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate PDF report from existing WAFR results."""
    
    # Load results
    results = load_results(results_file)
    
    # Extract data from results
    session_id = session_id or results.get("session_id", "unknown")
    insights = results.get("steps", {}).get("understanding", {}).get("insights", [])
    mappings = results.get("steps", {}).get("mapping", {}).get("mappings", [])
    all_answers = results.get("steps", {}).get("auto_populate", {}).get("all_answers", [])
    
    if not all_answers:
        # Try to get from other sources
        validated = results.get("steps", {}).get("confidence", {}).get("validated_answers", [])
        synthesized = results.get("steps", {}).get("answer_synthesis", {}).get("synthesized_answers", [])
        all_answers = validated + synthesized
    
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Found {len(insights)} insights")
    logger.info(f"Found {len(mappings)} mappings")
    logger.info(f"Found {len(all_answers)} answers")
    
    # Create orchestrator to get WA Tool agent
    logger.info("Initializing orchestrator...")
    orchestrator = create_orchestrator()
    
    # Prepare transcript analysis structure
    transcript_analysis = {
        "session_id": session_id,
        "insights": insights,
        "mappings": mappings,
        "answers": all_answers
    }
    
    # Create WA Tool agent
    wa_agent = orchestrator.wa_tool_agent
    
    # Create workload
    logger.info(f"Creating WA Tool workload for client: {client_name}")
    workload_result = wa_agent.create_workload_from_transcript(
        transcript_analysis=transcript_analysis,
        client_name=client_name,
        environment=environment
    )
    
    workload_id = workload_result.get("WorkloadId") or workload_result.get("workload_id")
    if not workload_id:
        logger.error(f"Failed to create workload. Result: {workload_result}")
        return {"status": "error", "error": "Failed to create workload"}
    
    logger.info(f"Created workload: {workload_id}")
    
    # Get transcript from results (if available) or use a placeholder
    transcript = results.get("transcript", "")
    if not transcript:
        # Reconstruct transcript from insights
        transcript = "\n".join([
            f"{insight.get('speaker', 'Unknown')}: {insight.get('transcript_quote', insight.get('content', ''))}"
            for insight in insights[:20]  # Use first 20 insights
        ])
    
    # Populate answers
    logger.info("Populating answers in workload...")
    populate_result = wa_agent.populate_answers_from_analysis(
        workload_id=workload_id,
        transcript_analysis=transcript_analysis,
        transcript=transcript,
        lens_alias='wellarchitected',
        mapping_agent=orchestrator.mapping_agent,
        lens_context=orchestrator.lens_context
    )
    
    updated_count = populate_result.get("updated_answers", 0)
    total_count = populate_result.get("total_questions", 0)
    logger.info(f"Populated {updated_count} out of {total_count} questions")
    
    # Generate PDF report
    logger.info("Generating PDF report...")
    pdf_filename = f"wafr_report_{workload_id}_{session_id}.pdf"
    
    milestone_result = wa_agent.create_milestone_and_review(
        workload_id=workload_id,
        milestone_name=f"WAFR_Review_{session_id}",
        save_report_path=pdf_filename
    )
    
    if milestone_result.get("report_pdf_path"):
        logger.info(f"PDF report generated: {milestone_result['report_pdf_path']}")
    else:
        logger.warning("PDF report path not found in result")
    
    return {
        "status": "completed",
        "workload_id": workload_id,
        "milestone_number": milestone_result.get("milestone_number"),
        "pdf_path": milestone_result.get("report_pdf_path", pdf_filename),
        "answers_populated": updated_count,
        "total_questions": total_count
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate PDF report from existing WAFR results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate PDF from results file
  python generate_pdf_from_results.py --results wafr_results_wafr-4d2f1c57.json --client-name "My Client"
  
  # Specify environment
  python generate_pdf_from_results.py --results results.json --client-name "Client" --environment "PRODUCTION"
        """
    )
    
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to WAFR results JSON file"
    )
    
    parser.add_argument(
        "--client-name",
        type=str,
        required=True,
        help="Client name for WA Tool workload"
    )
    
    parser.add_argument(
        "--environment",
        type=str,
        default="PRODUCTION",
        help="Environment type (default: PRODUCTION)"
    )
    
    parser.add_argument(
        "--session-id",
        type=str,
        help="Session ID (default: from results file)"
    )
    
    args = parser.parse_args()
    
    try:
        result = generate_pdf_from_results(
            results_file=args.results,
            client_name=args.client_name,
            environment=args.environment,
            session_id=args.session_id
        )
        
        if result.get("status") == "completed":
            logger.info("")
            logger.info("=" * 70)
            logger.info("PDF Report Generation Complete")
            logger.info("=" * 70)
            logger.info(f"Workload ID: {result.get('workload_id')}")
            logger.info(f"PDF Path: {result.get('pdf_path')}")
            logger.info(f"Answers Populated: {result.get('answers_populated')}/{result.get('total_questions')}")
            logger.info("=" * 70)
            sys.exit(0)
        else:
            logger.error(f"Failed to generate PDF: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n\nPDF generation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

