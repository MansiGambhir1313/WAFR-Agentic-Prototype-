"""
Input Processor - Central router for processing different input file types.

Supports text, PDF, video, and audio files with unified output format.
Routes to appropriate processors and returns normalized ProcessedInput.
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

TEXT_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".rst", ".text"})
PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v", ".wmv"})
AUDIO_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"})

# Confidence scores
CONFIDENCE_TEXT = 1.0
CONFIDENCE_PDF_DIGITAL = 0.9
CONFIDENCE_PDF_SCANNED = 0.7
CONFIDENCE_PDF_FAILED = 0.1


# =============================================================================
# Enums and Data Classes
# =============================================================================

class InputType(Enum):
    """Supported input file types."""
    TEXT = "text"
    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"


@dataclass
class ProcessedInput:
    """
    Normalized output from any input processor.
    
    Attributes:
        content: Extracted text content
        input_type: Original input type
        source_file: Original file path
        metadata: Processing metadata
        word_count: Total word count
        processing_time: Time to process (seconds)
        confidence: Extraction confidence (0.0-1.0)
        segments: For video/audio: timestamped segments
        tables: For PDF: extracted tables
        errors: Any non-fatal errors encountered
    """
    content: str
    input_type: InputType
    source_file: str
    metadata: dict[str, Any]
    word_count: int
    processing_time: float
    confidence: float
    segments: list[dict[str, Any]] | None = None
    tables: list[dict[str, Any]] | None = None
    errors: list[str] | None = None


# =============================================================================
# Input Processor
# =============================================================================

class InputProcessor:
    """
    Main input processor that routes to appropriate handlers.
    
    Supports text, PDF, video, and audio files with lazy-loaded
    processors for each type.
    """

    def __init__(
        self,
        aws_region: str = "us-east-1",
        use_textract: bool = True,
        ocr_fallback: bool = True,
        use_transcribe: bool = True,
        enable_diarization: bool = True,
        transcribe_language: str = "en-US",
        s3_bucket: str | None = None,
    ):
        """
        Initialize InputProcessor with configuration.
        
        Args:
            aws_region: AWS region for services
            use_textract: Use Amazon Textract for scanned PDFs
            ocr_fallback: Fallback to pytesseract for OCR
            use_transcribe: Use Amazon Transcribe for video/audio
            enable_diarization: Enable speaker identification
            transcribe_language: Language for transcription
            s3_bucket: S3 bucket for Transcribe (required for AWS)
        """
        self.aws_region = aws_region
        self.use_textract = use_textract
        self.ocr_fallback = ocr_fallback
        self.use_transcribe = use_transcribe
        self.enable_diarization = enable_diarization
        self.transcribe_language = transcribe_language
        self.s3_bucket = s3_bucket

        # Initialize processors lazily
        self._pdf_processor: Any = None
        self._video_processor: Any = None

    @property
    def pdf_processor(self) -> Any:
        """
        Lazy initialization of PDF processor.
        
        Returns:
            PDF processor instance
        """
        if self._pdf_processor is None:
            from .pdf_processor import create_pdf_processor

            self._pdf_processor = create_pdf_processor(
                aws_region=self.aws_region,
                use_textract=self.use_textract,
            )
        return self._pdf_processor

    @property
    def video_processor(self) -> Any:
        """
        Lazy initialization of video processor.
        
        Returns:
            Video processor instance
        """
        if self._video_processor is None:
            from .video_processor import create_video_processor

            self._video_processor = create_video_processor(
                aws_region=self.aws_region,
                use_transcribe=self.use_transcribe,
                enable_diarization=self.enable_diarization,
                language=self.transcribe_language,
                s3_bucket=self.s3_bucket,
            )
        return self._video_processor

    def detect_input_type(self, file_path: str) -> InputType:
        """
        Detect the input type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected InputType enum value
        """
        ext = Path(file_path).suffix.lower()

        if ext in TEXT_EXTENSIONS:
            return InputType.TEXT
        if ext in PDF_EXTENSIONS:
            return InputType.PDF
        if ext in VIDEO_EXTENSIONS:
            return InputType.VIDEO
        if ext in AUDIO_EXTENSIONS:
            return InputType.AUDIO

        return InputType.UNKNOWN

    def process(
        self,
        file_path: str,
        input_type: InputType | None = None,
    ) -> ProcessedInput:
        """
        Process any supported input file and return normalized transcript.
        
        Args:
            file_path: Path to input file
            input_type: Optional override for input type detection
            
        Returns:
            ProcessedInput with extracted text and metadata
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input type is unsupported
        """
        start_time = time.time()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        # Detect input type if not provided
        if input_type is None:
            input_type = self.detect_input_type(file_path)

        logger.info("Processing %s file: %s", input_type.value, file_path)

        # Route to appropriate processor
        result = self._route_to_processor(file_path, input_type)

        result.processing_time = time.time() - start_time
        logger.info("Processed %d words in %.2fs", result.word_count, result.processing_time)

        return result

    def _route_to_processor(
        self,
        file_path: str,
        input_type: InputType,
    ) -> ProcessedInput:
        """
        Route file to appropriate processor based on type.
        
        Args:
            file_path: Path to input file
            input_type: Detected or specified input type
            
        Returns:
            ProcessedInput from appropriate processor
            
        Raises:
            ValueError: If input type is unsupported
        """
        if input_type == InputType.TEXT:
            return self._process_text(file_path)

        if input_type == InputType.PDF:
            return self._process_pdf(file_path)

        if input_type == InputType.VIDEO:
            return self.video_processor.process(file_path)

        if input_type == InputType.AUDIO:
            return self.video_processor.process_audio(file_path)

        raise ValueError(f"Unsupported input type: {input_type}")

    def _process_text(self, file_path: str) -> ProcessedInput:
        """
        Process plain text files.
        
        Args:
            file_path: Path to text file
            
        Returns:
            ProcessedInput with extracted content
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return ProcessedInput(
            content=content,
            input_type=InputType.TEXT,
            source_file=file_path,
            metadata={"encoding": "utf-8"},
            word_count=len(content.split()),
            processing_time=0.0,
            confidence=CONFIDENCE_TEXT,
        )

    def _process_pdf(self, file_path: str) -> ProcessedInput:
        """
        Process PDF files.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ProcessedInput with extracted content and metadata
        """
        pdf_result = self.pdf_processor.process_pdf(file_path)

        text_content = pdf_result.get("text", "")
        text_content = self._format_pdf_content(file_path, text_content)

        confidence = self._calculate_pdf_confidence(pdf_result, text_content)
        metadata = self._build_pdf_metadata(pdf_result)
        errors = pdf_result.get("errors", []) or None

        return ProcessedInput(
            content=text_content,
            input_type=InputType.PDF,
            source_file=file_path,
            metadata=metadata,
            word_count=len(text_content.split()),
            processing_time=0.0,  # Will be set by caller
            confidence=confidence,
            tables=pdf_result.get("tables", []),
            errors=errors,
        )

    def _format_pdf_content(self, file_path: str, text_content: str) -> str:
        """
        Format PDF text content with header if needed.
        
        Args:
            file_path: Original file path for header
            text_content: Extracted text content
            
        Returns:
            Formatted text content
        """
        if not text_content:
            return text_content

        # Check if text already has page markers
        if "--- Page" in text_content:
            return text_content

        # Add header to indicate PDF source
        filename = os.path.basename(file_path)
        return f"\n=== PDF DOCUMENT: {filename} ===\n{text_content}"

    def _calculate_pdf_confidence(
        self,
        pdf_result: dict[str, Any],
        text_content: str,
    ) -> float:
        """
        Calculate confidence score based on PDF extraction quality.
        
        Args:
            pdf_result: Raw PDF processor result
            text_content: Extracted text content
            
        Returns:
            Confidence score (0.0-1.0)
        """
        if not text_content:
            return CONFIDENCE_PDF_FAILED

        is_scanned = pdf_result.get("is_scanned", False)
        if is_scanned:
            return CONFIDENCE_PDF_SCANNED

        return CONFIDENCE_PDF_DIGITAL

    def _build_pdf_metadata(self, pdf_result: dict[str, Any]) -> dict[str, Any]:
        """
        Build metadata dictionary from PDF result.
        
        Args:
            pdf_result: Raw PDF processor result
            
        Returns:
            Metadata dictionary
        """
        metadata = pdf_result.get("metadata", {})
        metadata["is_scanned"] = pdf_result.get("is_scanned", False)
        metadata["has_images"] = len(pdf_result.get("images", [])) > 0
        metadata["has_tables"] = len(pdf_result.get("tables", [])) > 0
        return metadata


# =============================================================================
# Factory Function
# =============================================================================

def create_input_processor(**kwargs: Any) -> InputProcessor:
    """
    Factory function for creating input processor.
    
    Args:
        **kwargs: Arguments to pass to InputProcessor constructor
        
    Returns:
        Configured InputProcessor instance
    """
    return InputProcessor(**kwargs)