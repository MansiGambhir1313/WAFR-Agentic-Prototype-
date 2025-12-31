"""
WAFR Orchestrator - Terminal Interface
Run WAFR analysis directly from command line
"""
import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.input_processor import InputType, create_input_processor
from agents.orchestrator import create_orchestrator


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(title: str) -> None:
    """Print a section header."""
    # Use ASCII-safe characters for Windows compatibility
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}\n")


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted results."""
    # Summary
    print_section("PROCESSING SUMMARY")
    summary = results.get('summary', {})
    print(f"  Status: {results.get('status', 'unknown')}")
    print(f"  Total Insights: {summary.get('total_insights', 0)}")
    print(f"  Total Mappings: {summary.get('total_mappings', 0)}")
    print(f"  Total Answers: {summary.get('total_answers', 0)}")
    print(f"  Total Gaps: {summary.get('total_gaps', 0)}")
    print(f"  Confidence Score: {summary.get('confidence_score', 0):.2%}")
    print(f"  Processing Time: {results.get('processing_time', {}).get('total', 0):.2f}s")
    
    # PDF Processing Summary
    pdf_processing = results.get('pdf_processing', {})
    if pdf_processing:
        print_section("PDF PROCESSING SUMMARY")
        print(f"  PDFs Processed: {pdf_processing.get('num_pdfs', 0)}")
        print(f"  Text Extracted: {pdf_processing.get('text_extracted', 0)} characters")
        print(f"  Images Found: {pdf_processing.get('images_found', 0)}")
        print(f"  Tables Found: {pdf_processing.get('tables_found', 0)}")
        print(f"  Status: {pdf_processing.get('status', 'unknown')}")
        if pdf_processing.get('errors'):
            print(f"  Errors: {len(pdf_processing['errors'])}")
            for error in pdf_processing['errors'][:3]:  # Show first 3 errors
                print(f"    - {error[:60]}")
    
    # Agent Steps
    print_section("AGENT RESULTS")
    steps = results.get('steps', {})
    for step_name, step_data in steps.items():
        if step_data is None:
            print(f"  ⚪ {step_name}: Not executed")
        elif isinstance(step_data, dict):
            if 'error' in step_data:
                print(f"  ERROR {step_name}: {step_data['error'][:60]}")
            else:
                # Get counts based on step type
                if step_name == 'understanding':
                    count = len(step_data.get('insights', []))
                    print(f"  OK {step_name}: {count} insights extracted")
                elif step_name == 'mapping':
                    count = len(step_data.get('mappings', []))
                    print(f"  OK {step_name}: {count} mappings created")
                elif step_name == 'confidence':
                    count = len(step_data.get('all_validations', []))
                    print(f"  OK {step_name}: {count} validations completed")
                elif step_name == 'gap_detection':
                    count = len(step_data.get('gaps', []))
                    print(f"  OK {step_name}: {count} gaps identified")
                elif step_name == 'scoring':
                    count = len(step_data.get('scored_answers', []))
                    print(f"  OK {step_name}: {count} answers scored")
                elif step_name == 'report':
                    print(f"  OK {step_name}: Report generated")
                elif step_name == 'wa_workload':
                    workload_id = step_data.get('workload_id')
                    if workload_id:
                        print(f"  OK {step_name}: Workload {workload_id} created")
                        answers = step_data.get('answers_populated', {}).get('updated_answers', 0)
                        print(f"     └─ {answers} questions auto-filled")
                    else:
                        print(f"  ⚪ {step_name}: Not executed")
                else:
                    print(f"  OK {step_name}: Complete")
        else:
            print(f"  WARNING {step_name}: {type(step_data).__name__}")
    
    # Insights (top 5)
    insights = steps.get('understanding', {}).get('insights', [])
    if insights:
        print_section("TOP INSIGHTS")
        for i, insight in enumerate(insights[:5], 1):
            print(f"  {i}. [{insight.get('insight_type', 'unknown')}] {insight.get('content', '')[:80]}")
    
    # Mappings (top 5)
    mappings = steps.get('mapping', {}).get('mappings', [])
    if mappings:
        print_section("TOP MAPPINGS")
        for i, mapping in enumerate(mappings[:5], 1):
            q_id = mapping.get('question_id', 'UNKNOWN')
            pillar = mapping.get('pillar', 'UNKNOWN')
            print(f"  {i}. [{pillar}] {q_id}: {mapping.get('question_text', '')[:60]}")
            print(f"     └─ Relevance: {mapping.get('relevance_score', 0):.2%}")
    
    # Gaps (top 5)
    gaps = steps.get('gap_detection', {}).get('gaps', [])
    if gaps:
        print_section("TOP GAPS")
        for i, gap in enumerate(gaps[:5], 1):
            q_id = gap.get('question_id', 'UNKNOWN')
            priority = gap.get('priority_score', 0)
            print(f"  {i}. {q_id}: Priority {priority:.2f}")
            if gap.get('question_text'):
                print(f"     └─ {gap.get('question_text', '')[:70]}")
    
    # WA Tool Results
    wa_workload = steps.get('wa_workload')
    if wa_workload and wa_workload.get('workload_id'):
        print_section("AWS WELL-ARCHITECTED TOOL")
        print(f"  Workload ID: {wa_workload.get('workload_id')}")
        answers_pop = wa_workload.get('answers_populated', {})
        print(f"  Questions Auto-filled: {answers_pop.get('updated_answers', 0)} / {answers_pop.get('total_questions', 0)}")
        
        review = wa_workload.get('review', {})
        risk_counts = review.get('risk_counts', {})
        if risk_counts:
            print(f"  Risk Summary:")
            print(f"    High Risk: {risk_counts.get('HIGH', 0)}")
            print(f"    Medium Risk: {risk_counts.get('MEDIUM', 0)}")
            print(f"    Low Risk: {risk_counts.get('LOW', 0)}")
        
        console_url = review.get('console_url')
        if console_url:
            print(f"  Console URL: {console_url}")
        
        report_file = wa_workload.get('report_file')
        if report_file:
            print(f"  Report File: {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='WAFR Orchestrator - Terminal Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a transcript file
  python run_wafr.py transcript.txt

  # Process a PDF file directly
  python run_wafr.py architecture_doc.pdf

  # Process a video file with Amazon Transcribe (requires S3 bucket)
  python run_wafr.py workshop_recording.mp4 --s3-bucket my-bucket --wa-tool --client-name "Acme Corp"

  # Process a video file with local Whisper (no AWS required)
  python run_wafr.py meeting.mp4 --use-whisper --wa-tool --client-name "Acme Corp"

  # Process audio file
  python run_wafr.py interview.mp3 --s3-bucket my-bucket

  # Process transcript with additional PDF files
  python run_wafr.py transcript.txt --pdfs diagram.pdf workflow.pdf

  # Process all PDFs in a directory (as additional files)
  python run_wafr.py transcript.txt --pdf-dir ./documents

  # Process PDF with WA Tool integration
  python run_wafr.py architecture_doc.pdf --wa-tool --client-name "Acme Corp"

  # Process PDF without Textract (use local OCR only)
  python run_wafr.py scanned_doc.pdf --no-textract

  # Save results to file
  python run_wafr.py transcript.txt --output results.json

  # Process from stdin
  cat transcript.txt | python run_wafr.py -
        """
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input file (transcript, PDF, video, audio, or "-" for stdin). If not provided, will prompt for file.'
    )
    
    parser.add_argument(
        '--lenses', '-l',
        nargs='+',
        default=['wellarchitected'],
        help='Lenses to apply (e.g., generative-ai serverless saas). Default: wellarchitected'
    )
    
    parser.add_argument(
        '--list-lenses',
        action='store_true',
        help='List all available lenses and exit'
    )
    
    parser.add_argument(
        '--lens-info',
        type=str,
        help='Show detailed information for a specific lens and exit'
    )
    
    parser.add_argument(
        '--wa-tool',
        action='store_true',
        help='Enable AWS Well-Architected Tool integration'
    )
    
    parser.add_argument(
        '--client-name',
        type=str,
        help='Client name for WA Tool workload (required if --wa-tool)'
    )
    
    parser.add_argument(
        '--environment',
        type=str,
        default='PRODUCTION',
        choices=['PRODUCTION', 'PREPRODUCTION', 'DEVELOPMENT'],
        help='Environment type for WA Tool (default: PRODUCTION)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )
    
    parser.add_argument(
        '--session-id',
        type=str,
        help='Custom session ID (default: auto-generated)'
    )
    
    parser.add_argument(
        '--workload-id',
        type=str,
        help='Use existing AWS Well-Architected Tool workload ID (skips workload creation)'
    )
    
    parser.add_argument(
        '--pdfs',
        nargs='+',
        help='Additional PDF files to include in analysis (diagrams, workflows, documentation). Use this when input_file is a transcript.'
    )
    
    parser.add_argument(
        '--pdf-dir',
        type=str,
        help='Directory containing PDF files to process (processes all PDFs in directory). Use this when input_file is a transcript.'
    )
    
    parser.add_argument(
        '--no-textract',
        action='store_true',
        help='Disable Amazon Textract for scanned PDFs (use local OCR only)'
    )
    
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR completely (skip scanned PDF processing)'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='us-east-1',
        help='AWS region for services like Textract (default: us-east-1)'
    )
    
    parser.add_argument(
        '--use-whisper',
        action='store_true',
        help='Use local Whisper model instead of Amazon Transcribe for video/audio transcription'
    )
    
    parser.add_argument(
        '--no-diarization',
        action='store_true',
        help='Disable speaker diarization for video/audio (faster processing)'
    )
    
    parser.add_argument(
        '--s3-bucket',
        type=str,
        help='S3 bucket for video/audio transcription (required for Amazon Transcribe)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default='en-US',
        help='Language code for transcription (default: en-US). Examples: en-US, es-ES, fr-FR'
    )
    
    args = parser.parse_args()
    
    # Validate WA Tool args
    if args.wa_tool and not args.client_name and not args.workload_id:
        print("Error: --client-name is required when using --wa-tool (unless --workload-id is provided)")
        sys.exit(1)
    
    # Get input file
    if not args.input_file:
        args.input_file = input("Enter input file path (or '-' for stdin): ").strip()
        if not args.input_file:
            print("Error: No input file provided")
            sys.exit(1)
    
    print_header("WAFR ORCHESTRATOR - TERMINAL INTERFACE")
    
    input_path = Path(args.input_file)
    
    # Detect input type
    input_processor = create_input_processor(
        aws_region=args.region,
        use_textract=not args.no_textract,
        ocr_fallback=not args.no_ocr,
        use_transcribe=not args.use_whisper,
        enable_diarization=not args.no_diarization,
        transcribe_language=args.language,
        s3_bucket=args.s3_bucket
    )
    
    # Check if input is a file or stdin
    if args.input_file == '-':
        print("Reading transcript from stdin...")
        transcript = sys.stdin.read()
        input_type = InputType.TEXT
        use_file_processing = False
    else:
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input_file}")
            sys.exit(1)
        
        input_type = input_processor.detect_input_type(str(input_path))
        print(f"Detected input type: {input_type.value}")
        
        # If it's a PDF, use file processing; if text, read directly for backward compatibility
        if input_type == InputType.PDF:
            use_file_processing = True
            transcript = None
        else:
            use_file_processing = False
            print(f"Reading {input_type.value} file from: {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
    
    # Validate transcript length (if we have it)
    if transcript and len(transcript.strip()) < 50:
        print("Error: Transcript too short (minimum 50 characters)")
        sys.exit(1)
    
    if transcript:
        print(f"   Transcript length: {len(transcript)} characters")
    
    # Process PDF files if provided
    pdf_files = []
    if args.pdfs:
        pdf_files.extend(args.pdfs)
    
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.exists() or not pdf_dir.is_dir():
            print(f"Error: PDF directory not found: {args.pdf_dir}")
            sys.exit(1)
        
        # Find all PDF files in directory
        dir_pdfs = list(pdf_dir.glob("*.pdf"))
        pdf_files.extend([str(p) for p in dir_pdfs])
        print(f"Found {len(dir_pdfs)} PDF file(s) in directory")
    
    # Validate PDF files
    if pdf_files:
        print(f"Processing {len(pdf_files)} PDF file(s)...")
        valid_pdfs = []
        for pdf_path in pdf_files:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                print(f"   Warning: PDF file not found: {pdf_path}")
            elif not pdf_file.suffix.lower() == '.pdf':
                print(f"   Warning: Not a PDF file: {pdf_path}")
            else:
                valid_pdfs.append(str(pdf_file))
                print(f"   OK: {pdf_file.name}")
        
        pdf_files = valid_pdfs
        if not pdf_files:
            print("   Warning: No valid PDF files found, continuing with transcript only")
    
    # Generate session ID
    session_id = args.session_id or f"cli-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"   Session ID: {session_id}")
    
    # Initialize lens manager and get lens context
    lens_context = None
    selected_lenses = args.lenses if args.lenses else None
    
    # Auto-detect lenses if not specified
    # For file processing, we'll detect after processing the file
    # For transcript processing, detect now
    if not selected_lenses:
        content_for_detection = None
        
        if transcript:
            content_for_detection = transcript
        elif use_file_processing and input_path.exists():
            # For file processing, we'll do a quick preview for detection
            # This is a lightweight check - just read first part of file
            try:
                if input_type == InputType.PDF:
                    # For PDFs, we'll detect after processing
                    content_for_detection = None
                else:
                    # For text files, read first 5000 chars for detection
                    with open(input_path, 'r', encoding='utf-8') as f:
                        content_for_detection = f.read(5000)
            except Exception:
                content_for_detection = None
        
        if content_for_detection:
            print(f"\nAuto-detecting relevant lenses from content...")
            try:
                from agents.lens_manager import create_lens_manager
                lens_manager = create_lens_manager(aws_region=args.region)
                detected = lens_manager.detect_relevant_lenses(content_for_detection, min_confidence=0.2)
                
                if detected:
                    print(f"   Detected {len(detected)} relevant lens(es):")
                    for lens_info in detected[:5]:  # Show top 5
                        print(f"     - {lens_info['lens_alias']}: {lens_info['confidence']:.0%} confidence ({lens_info['match_count']} matches)")
                    
                    # Auto-select top lenses
                    selected_lenses = lens_manager.auto_select_lenses(content_for_detection, max_lenses=3)
                    print(f"\n   Auto-selected lenses: {', '.join(selected_lenses)}")
            except Exception as e:
                print(f"   Warning: Could not auto-detect lenses: {e}")
                selected_lenses = ['wellarchitected']
    
    # If still no lenses, use default
    if not selected_lenses:
        selected_lenses = ['wellarchitected']
    
    # Get lens context for selected lenses
    if selected_lenses:
        print(f"\nInitializing lens manager...")
        try:
            from agents.lens_manager import create_lens_manager
            lens_manager = create_lens_manager(aws_region=args.region)
            lens_context = lens_manager.get_lens_context_for_agents(selected_lenses)
            
            print(f"   Analyzing with {len(selected_lenses)} lens(es):")
            for alias in selected_lenses:
                info = lens_context.get("lenses", {}).get(alias, {})
                print(f"     - {info.get('name', alias)}: {info.get('question_count', 0)} questions")
            print(f"   Total questions: {lens_context.get('question_count', 0)}")
        except Exception as e:
            print(f"   Warning: Could not initialize lens manager: {e}")
            print("   Continuing with default Well-Architected Framework only")
            lens_context = None
    
    # Create orchestrator with input processor and lens context
    print("\nInitializing orchestrator...")
    try:
        orchestrator = create_orchestrator(
            input_processor=input_processor,
            aws_region=args.region,
            lens_context=lens_context
        )
        print("   Orchestrator ready")
    except Exception as e:
        print(f"   Error creating orchestrator: {e}")
        sys.exit(1)
    
    # Process input
    print("\nStarting processing...")
    print("   This may take a few minutes...\n")
    
    try:
        if use_file_processing:
            # Use new file processing method for PDFs
            results = orchestrator.process_file(
                file_path=str(input_path),
                session_id=session_id,
                generate_report=not args.no_report,
                create_wa_workload=args.wa_tool,
                client_name=args.client_name,
                environment=args.environment,
                existing_workload_id=args.workload_id
            )
        else:
            # Use existing transcript processing (backward compatibility)
            results = orchestrator.process_transcript(
                transcript=transcript,
                session_id=session_id,
                generate_report=not args.no_report,
                create_wa_workload=args.wa_tool,
                client_name=args.client_name,
                environment=args.environment,
                existing_workload_id=args.workload_id,
                pdf_files=pdf_files if pdf_files else None
            )
        
        # Print results
        print_results(results)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            # Convert datetime objects to strings for JSON serialization
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=json_serializer)
            print(f"\nResults saved to: {output_path}")
        
        # Exit code based on status
        if results.get('status') == 'completed':
            print("\nProcessing completed successfully!")
            sys.exit(0)
        else:
            print(f"\nProcessing completed with status: {results.get('status')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

