"""
Complete WAFR Pipeline Runner with All Latest Features

This script runs the complete WAFR pipeline with:
- AG-UI event streaming for real-time updates
- Strict quality control (confidence >= 0.7)
- HRI validation using Claude (filters non-tangible HRIs)
- Enhanced question answering (honest assessment, no forced coverage)
- Automatic lens detection from transcript
- WA Tool integration for PDF generation
- Comprehensive logging and progress tracking

Usage:
    python run_wafr_full.py --wa-tool --client-name "My Client"
    python run_wafr_full.py --transcript transcript.txt --wa-tool --client-name "Client"
"""

import sys
import os
import json
import uuid
import logging
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
log_dir = Path(__file__).parent.parent / "output" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f'wafr_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from wafr.ag_ui.orchestrator_integration import create_agui_orchestrator
    from wafr.ag_ui.emitter import WAFREventEmitter
    from wafr.ag_ui.core import get_all_wafr_tools
except ImportError as e:
    logger.warning(f"AG-UI components not available: {e}")
    logger.warning("Continuing without AG-UI (install with: pip install ag-ui-protocol)")
    create_agui_orchestrator = None
    WAFREventEmitter = None

from wafr.agents.orchestrator import create_orchestrator


# Default transcript for testing
DEFAULT_TRANSCRIPT = """
AWS Well-Architected Framework Review Workshop

Date: 2026-01-15
Participants: John (Solutions Architect), Sarah (DevOps Lead), Mike (Security Engineer)

John: We're building a serverless application using AWS Lambda for the API layer.
The frontend is hosted on S3 with CloudFront for CDN distribution. We're using
DynamoDB for our database and API Gateway for REST endpoints.

Sarah: We have CloudWatch Logs set up for monitoring, but we don't have automated
alerts configured yet. Our deployment is manual through the AWS Console.

Mike: Security-wise, we're using IAM roles for Lambda functions, but we haven't
implemented encryption at rest for DynamoDB. We also don't have a disaster recovery plan.

John: Cost optimization is a concern. We're using on-demand pricing for everything,
but we should look into reserved capacity for DynamoDB.

Sarah: We're planning to add auto-scaling for Lambda, but it's not implemented yet.
We also need to set up automated backups for our data.
"""


async def run_complete_wafr(
    transcript: str,
    session_id: Optional[str] = None,
    generate_report: bool = True,
    create_wa_workload: bool = False,
    client_name: Optional[str] = None,
    output_events: Optional[str] = None,
    use_agui: bool = True
) -> Dict[str, Any]:
    """
    Run complete WAFR pipeline with all features.
    
    Args:
        transcript: Workshop transcript text
        session_id: Optional session ID
        generate_report: Whether to generate PDF report
        create_wa_workload: Whether to create WA Tool workload
        client_name: Client name for WA Tool
        output_events: Optional path to save AG-UI events
        use_agui: Whether to use AG-UI integration
        
    Returns:
        Complete results dictionary
    """
    if session_id is None:
        session_id = f"wafr-{uuid.uuid4().hex[:8]}"
    
    logger.info("=" * 70)
    logger.info("WAFR Complete Pipeline - All Features Enabled")
    logger.info("=" * 70)
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Transcript Length: {len(transcript)} characters")
    logger.info(f"Report Generation: {'Enabled' if generate_report else 'Disabled'}")
    logger.info(f"WA Tool Integration: {'Enabled' if create_wa_workload else 'Disabled'}")
    logger.info(f"Strict Quality Control: Enabled (confidence >= 0.7, expected coverage: 30-70%)")
    logger.info(f"HRI Validation: Enabled (Claude-based, filters non-tangible HRIs)")
    logger.info(f"Lens Detection: Enabled (automatic from transcript)")
    logger.info(f"AG-UI Integration: {'Enabled' if use_agui and create_agui_orchestrator else 'Disabled'}")
    logger.info("=" * 70)
    logger.info("")
    
    # Create event collector
    event_collector = AGUIEventCollector()
    
    try:
        # Create orchestrator
        if use_agui and create_agui_orchestrator:
            logger.info("Creating AG-UI enabled orchestrator...")
            agui_orchestrator = create_agui_orchestrator(thread_id=session_id)
            emitter = agui_orchestrator.emitter
            
            # Add event listener
            def collect_event(event):
                event_dict = event.to_dict() if hasattr(event, 'to_dict') else event
                event_collector.collect(event_dict)
            
            emitter.add_listener(collect_event)
            
            # Log available tools
            logger.info("Available WAFR Agent Tools:")
            for tool in get_all_wafr_tools():
                logger.info(f"  - {tool.name}: {tool.description[:60]}...")
            logger.info("")
            
            # Run pipeline with AG-UI
            logger.info("Starting WAFR pipeline with AG-UI event streaming...")
            logger.info("")
            
            results = await agui_orchestrator.process_transcript_with_agui(
                transcript=transcript,
                session_id=session_id,
                generate_report=generate_report,
                create_wa_workload=create_wa_workload,
                client_name=client_name,
            )
        else:
            logger.info("Creating standard orchestrator...")
            orchestrator = create_orchestrator()
            
            # Run pipeline
            logger.info("Starting WAFR pipeline...")
            logger.info("")
            
            results = orchestrator.process_transcript(
                transcript=transcript,
                session_id=session_id,
                generate_report=generate_report,
                create_wa_workload=create_wa_workload,
                client_name=client_name,
                environment='PRODUCTION',
                progress_callback=lambda step, message, data=None: logger.info(f"Step: {step} - {message}")
            )
        
        # Print event summary if AG-UI was used
        if use_agui and create_agui_orchestrator:
            logger.info("")
            event_collector.print_summary()
            
            # Save events if requested
            if output_events:
                event_collector.save_to_file(output_events)
        
        # Print results summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("WAFR Processing Complete")
        logger.info("=" * 70)
        logger.info(f"Status: {results.get('status', 'unknown')}")
        logger.info(f"Session ID: {session_id}")
        
        if "steps" in results:
            logger.info("")
            logger.info("Pipeline Steps:")
            for step_name, step_result in results["steps"].items():
                if step_result is not None:
                    if isinstance(step_result, dict):
                        status = "OK" if step_result.get("status") != "error" else "ERROR"
                        
                        # Step-specific field reading for accurate counts
                        count = 0
                        display_text = ""
                        
                        if step_name == "understanding":
                            count = len(step_result.get("insights", [])) or step_result.get("insights_count", 0) or step_result.get("count", 0)
                            display_text = f"{count} items"
                        elif step_name == "mapping":
                            count = len(step_result.get("mappings", [])) or step_result.get("mappings_count", 0) or step_result.get("count", 0)
                            display_text = f"{count} items"
                        elif step_name == "confidence":
                            summary = step_result.get("summary", {})
                            count = summary.get("total_answers", 0) or len(step_result.get("all_validations", []))
                            display_text = f"{count} validations"
                        elif step_name == "gap_detection":
                            count = len(step_result.get("gaps", [])) or step_result.get("gaps_count", 0) or step_result.get("count", 0)
                            display_text = f"{count} gaps"
                        elif step_name == "answer_synthesis":
                            count = step_result.get("total_synthesized", 0) or len(step_result.get("synthesized_answers", []))
                            display_text = f"{count} synthesized"
                        elif step_name == "auto_populate":
                            count = step_result.get("total_count", 0) or (step_result.get("validated_count", 0) + step_result.get("synthesized_count", 0))
                            validated = step_result.get("validated_count", 0)
                            synthesized = step_result.get("synthesized_count", 0)
                            if validated or synthesized:
                                display_text = f"{count} total ({validated} validated + {synthesized} synthesized)"
                            else:
                                display_text = f"{count} items"
                        elif step_name == "scoring":
                            count = step_result.get("scored_count", 0) or len(step_result.get("scored_answers", [])) or step_result.get("count", 0)
                            display_text = f"{count} scored"
                        elif step_name == "wa_workload":
                            workload_id = step_result.get("workload_id")
                            if workload_id:
                                answers_info = step_result.get("answers_populated", {})
                                updated = answers_info.get("updated_answers", 0)
                                total = answers_info.get("total_questions", 0)
                                skipped = answers_info.get("skipped_answers", 0)
                                if total > 0:
                                    display_text = f"workload {workload_id[:8]}... ({updated}/{total} answered, {skipped} skipped)"
                                else:
                                    display_text = f"workload {workload_id[:8]}... (created)"
                            else:
                                display_text = "not created"
                        elif step_name == "report":
                            report_path = step_result.get("report_path") or step_result.get("pdf_path")
                            if report_path:
                                display_text = f"generated: {report_path}"
                            else:
                                display_text = "not generated"
                        elif step_name == "gap_prompts":
                            display_text = "completed"
                        elif step_name == "hri_validation":
                            validated = step_result.get("total_validated", 0)
                            filtered = step_result.get("total_filtered", 0)
                            total = step_result.get("total_potential", 0)
                            if total > 0:
                                display_text = f"{validated} tangible ({filtered} filtered, {total} total)"
                            else:
                                display_text = "no HRIs found"
                        else:
                            count = (
                                step_result.get("count", 0) or
                                len(step_result.get("items", [])) or
                                0
                            )
                            display_text = f"{count} items" if count > 0 else "completed"
                        
                        logger.info(f"  [{status}] {step_name:20s} - {display_text}")
                    else:
                        logger.info(f"  [OK] {step_name:20s} - completed")
        
        # WA Tool results - show accurate information
        if create_wa_workload and "steps" in results:
            wa_workload = results["steps"].get("wa_workload")
            if wa_workload and isinstance(wa_workload, dict):
                answers_populated = wa_workload.get("answers_populated", {})
                if answers_populated:
                    updated = answers_populated.get("updated_answers", 0)
                    total = answers_populated.get("total_questions", 0)
                    skipped = answers_populated.get("skipped_answers", 0)
                    failed = answers_populated.get("failed_answers", 0)
                    coverage = (updated / total * 100) if total > 0 else 0
                    
                    logger.info("")
                    logger.info("WA Tool Integration Results:")
                    logger.info(f"  Total Questions: {total}")
                    logger.info(f"  Answered: {updated} ({coverage:.1f}%)")
                    logger.info(f"  Skipped: {skipped} (insufficient evidence)")
                    if failed > 0:
                        logger.info(f"  Failed: {failed}")
                    logger.info(f"  Confidence Threshold: >= 0.7 (strict quality control)")
                    
                    if wa_workload.get("workload_id"):
                        logger.info(f"  Workload ID: {wa_workload.get('workload_id')}")
                    
                    if wa_workload.get("report_file"):
                        logger.info(f"  PDF Report: {wa_workload.get('report_file')}")
        
        # HRI Validation results
        if "steps" in results and "hri_validation" in results["steps"]:
            hri_validation = results["steps"]["hri_validation"]
            if hri_validation and isinstance(hri_validation, dict):
                validated = hri_validation.get("total_validated", 0)
                filtered = hri_validation.get("total_filtered", 0)
                total = hri_validation.get("total_potential", 0)
                
                if total > 0:
                    logger.info("")
                    logger.info("HRI Validation Results:")
                    logger.info(f"  Potential HRIs: {total}")
                    logger.info(f"  Tangible HRIs: {validated} (validated with evidence)")
                    logger.info(f"  Filtered Out: {filtered} (non-tangible/false positives)")
                    if filtered > 0:
                        reduction = (filtered / total * 100)
                        logger.info(f"  Reduction: {reduction:.0f}% (only tangible HRIs reported)")
        
        # Save results
        results_dir = Path(__file__).parent.parent / "output" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"wafr_results_{session_id}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("")
        logger.info(f"Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error running WAFR pipeline: {e}", exc_info=True)
        if use_agui and create_agui_orchestrator:
            try:
                agui_orchestrator = create_agui_orchestrator(thread_id=session_id)
                await agui_orchestrator.emitter.run_error(str(e), code="PIPELINE_ERROR")
            except:
                pass
        raise


class AGUIEventCollector:
    """Collects AG-UI events for analysis."""
    
    def __init__(self):
        self.events = []
        self.event_counts = {}
    
    def collect(self, event):
        """Collect an event."""
        event_dict = event.to_dict() if hasattr(event, 'to_dict') else event
        self.events.append(event_dict)
        event_type = event_dict.get('type', 'UNKNOWN')
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
    
    def save_to_file(self, filepath: str):
        """Save events to JSONL file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for event in self.events:
                f.write(json.dumps(event) + '\n')
        logger.info(f"Saved {len(self.events)} events to {filepath}")
    
    def print_summary(self):
        """Print event summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("AG-UI Event Summary")
        logger.info("=" * 70)
        logger.info(f"Total Events: {len(self.events)}")
        logger.info("")
        logger.info("Event Types:")
        for event_type, count in sorted(self.event_counts.items()):
            logger.info(f"  {event_type:30s} {count:4d}")
        logger.info("=" * 70)


def load_transcript(transcript_path: Optional[str] = None) -> str:
    """Load transcript from file or use default."""
    if transcript_path and Path(transcript_path).exists():
        logger.info(f"Loading transcript from: {transcript_path}")
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif transcript_path:
        logger.warning(f"Transcript file not found: {transcript_path}, using default")
    return DEFAULT_TRANSCRIPT


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete WAFR pipeline with all latest features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  - AG-UI event streaming for real-time updates
  - Strict quality control (confidence >= 0.7)
  - HRI validation using Claude (filters non-tangible HRIs)
  - Enhanced question answering (honest assessment)
  - Automatic lens detection from transcript
  - WA Tool integration for PDF generation

Examples:
  # Run with default transcript and WA Tool (recommended)
  python run_wafr_full.py --wa-tool --client-name "My Client"
  
  # Run with custom transcript
  python run_wafr_full.py --transcript transcript.txt --wa-tool --client-name "Client"
  
  # Run without PDF generation
  python run_wafr_full.py --no-report
  
  # Save AG-UI events
  python run_wafr_full.py --wa-tool --client-name "Client" --output-events events.jsonl
  
  # Disable AG-UI
  python run_wafr_full.py --wa-tool --client-name "Client" --no-agui
        """
    )
    
    parser.add_argument(
        "--transcript",
        type=str,
        help="Path to transcript file (default: uses built-in transcript)"
    )
    
    parser.add_argument(
        "--session-id",
        type=str,
        help="Custom session ID (default: auto-generated)"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )
    
    parser.add_argument(
        "--wa-tool",
        action="store_true",
        help="Enable WA Tool integration (creates workload and generates PDF)"
    )
    
    parser.add_argument(
        "--client-name",
        type=str,
        help="Client name for WA Tool workload (required if --wa-tool is used)"
    )
    
    parser.add_argument(
        "--output-events",
        type=str,
        help="Path to save AG-UI events as JSONL file"
    )
    
    parser.add_argument(
        "--no-agui",
        action="store_true",
        help="Disable AG-UI integration (use standard orchestrator)"
    )
    
    args = parser.parse_args()
    
    # Load transcript
    transcript = load_transcript(args.transcript)
    
    # Validate WA Tool arguments
    if args.wa_tool and not args.client_name:
        logger.warning("WA Tool integration requires --client-name")
        logger.warning("Continuing without WA Tool integration...")
        create_wa_workload = False
        client_name = None
    else:
        create_wa_workload = args.wa_tool
        client_name = args.client_name
    
    # Run pipeline
    try:
        results = asyncio.run(run_complete_wafr(
            transcript=transcript,
            session_id=args.session_id,
            generate_report=not args.no_report,
            create_wa_workload=create_wa_workload,
            client_name=client_name,
            output_events=args.output_events,
            use_agui=not args.no_agui
        ))
        
        # Exit with appropriate code
        if results.get("status") == "completed":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

