"""
Video Processor - Extract and transcribe audio from video files
Supports various video formats and transcription services
"""
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed speech"""
    start_time: float       # Start time in seconds
    end_time: float         # End time in seconds
    speaker: Optional[str]  # Speaker label (if diarization enabled)
    text: str              # Transcribed text
    confidence: float       # Transcription confidence


class VideoProcessor:
    """Processor for video/audio file transcription"""
    
    def __init__(
        self,
        aws_region: str = "us-east-1",
        use_transcribe: bool = True,
        enable_diarization: bool = True,
        language: str = "en-US",
        s3_bucket: Optional[str] = None  # For large files
    ):
        self.aws_region = aws_region
        self.use_transcribe = use_transcribe
        self.enable_diarization = enable_diarization
        self.language = language
        self.s3_bucket = s3_bucket
        
        # Initialize clients lazily
        self._transcribe_client = None
        self._s3_client = None
    
    @property
    def transcribe_client(self):
        if self._transcribe_client is None and self.use_transcribe:
            try:
                import boto3
                self._transcribe_client = boto3.client(
                    'transcribe',
                    region_name=self.aws_region
                )
            except Exception as e:
                logger.warning(f"Could not initialize Transcribe client: {e}")
                self.use_transcribe = False
        return self._transcribe_client
    
    @property
    def s3_client(self):
        if self._s3_client is None:
            try:
                import boto3
                self._s3_client = boto3.client('s3', region_name=self.aws_region)
            except Exception as e:
                logger.warning(f"Could not initialize S3 client: {e}")
        return self._s3_client
    
    def process(self, file_path: str) -> 'ProcessedInput':
        """
        Process a video file and extract transcript
        
        Args:
            file_path: Path to video file
            
        Returns:
            ProcessedInput with transcribed text
        """
        from .input_processor import ProcessedInput, InputType
        
        start_time = time.time()
        logger.info(f"Processing video: {file_path}")
        
        # Extract audio from video
        audio_path = None
        try:
            audio_path = self._extract_audio(file_path)
            
            # Transcribe audio
            if self.use_transcribe and self.transcribe_client and self.s3_bucket:
                segments = self._transcribe_with_aws(audio_path)
            else:
                segments = self._transcribe_with_whisper(audio_path)
            
            # Build transcript content
            content = self._build_transcript_content(segments)
            
            # Calculate average confidence
            avg_confidence = (
                sum(s.confidence for s in segments) / len(segments)
                if segments else 0.0
            )
            
            # Get video metadata
            metadata = self._extract_video_metadata(file_path)
            metadata["segment_count"] = len(segments)
            metadata["speakers"] = list(set(s.speaker for s in segments if s.speaker))
            metadata["transcription_method"] = "aws_transcribe" if self.use_transcribe else "whisper"
            
            return ProcessedInput(
                content=content,
                input_type=InputType.VIDEO,
                source_file=file_path,
                metadata=metadata,
                word_count=len(content.split()),
                processing_time=time.time() - start_time,
                confidence=avg_confidence,
                segments=[{
                    "start": s.start_time,
                    "end": s.end_time,
                    "speaker": s.speaker,
                    "text": s.text
                } for s in segments] if segments else None
            )
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}", exc_info=True)
            # Return partial result if possible
            from .input_processor import ProcessedInput, InputType
            return ProcessedInput(
                content="",
                input_type=InputType.VIDEO,
                source_file=file_path,
                metadata={"error": str(e)},
                word_count=0,
                processing_time=time.time() - start_time,
                confidence=0.0,
                errors=[str(e)]
            )
        finally:
            # Cleanup temporary audio file
            if audio_path and audio_path != file_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp audio file: {e}")
    
    def process_audio(self, file_path: str) -> 'ProcessedInput':
        """Process audio file directly (same as video but skip extraction)"""
        from .input_processor import ProcessedInput, InputType
        
        start_time = time.time()
        logger.info(f"Processing audio: {file_path}")
        
        try:
            # Transcribe directly
            if self.use_transcribe and self.transcribe_client and self.s3_bucket:
                segments = self._transcribe_with_aws(file_path)
            else:
                segments = self._transcribe_with_whisper(file_path)
            
            content = self._build_transcript_content(segments)
            avg_confidence = (
                sum(s.confidence for s in segments) / len(segments)
                if segments else 0.0
            )
            
            return ProcessedInput(
                content=content,
                input_type=InputType.AUDIO,
                source_file=file_path,
                metadata={
                    "segment_count": len(segments),
                    "speakers": list(set(s.speaker for s in segments if s.speaker)),
                    "transcription_method": "aws_transcribe" if self.use_transcribe else "whisper"
                },
                word_count=len(content.split()),
                processing_time=time.time() - start_time,
                confidence=avg_confidence,
                segments=[{
                    "start": s.start_time,
                    "end": s.end_time,
                    "speaker": s.speaker,
                    "text": s.text
                } for s in segments] if segments else None
            )
        except Exception as e:
            logger.error(f"Audio processing failed: {e}", exc_info=True)
            from .input_processor import ProcessedInput, InputType
            return ProcessedInput(
                content="",
                input_type=InputType.AUDIO,
                source_file=file_path,
                metadata={"error": str(e)},
                word_count=0,
                processing_time=time.time() - start_time,
                confidence=0.0,
                errors=[str(e)]
            )
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio track from video file using ffmpeg"""
        logger.info("Extracting audio from video")
        
        # Create temporary file for audio
        audio_path = tempfile.mktemp(suffix=".wav")
        
        try:
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vn",                    # No video
                "-acodec", "pcm_s16le",  # PCM format for best compatibility
                "-ar", "16000",           # 16kHz sample rate (optimal for speech)
                "-ac", "1",               # Mono
                "-y",                     # Overwrite output
                audio_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")
            
            if not os.path.exists(audio_path):
                raise RuntimeError("Audio extraction failed - output file not created")
            
            logger.info(f"Audio extracted to: {audio_path}")
            return audio_path
            
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg:\n"
                "  Windows: https://ffmpeg.org/download.html\n"
                "  macOS: brew install ffmpeg\n"
                "  Linux: sudo apt-get install ffmpeg"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out (exceeded 10 minutes)")
    
    def _transcribe_with_aws(self, audio_path: str) -> List[TranscriptSegment]:
        """Transcribe audio using Amazon Transcribe"""
        import uuid
        
        logger.info("Transcribing with Amazon Transcribe")
        
        # Upload to S3 (required for Transcribe)
        if not self.s3_bucket:
            raise ValueError("S3 bucket required for Amazon Transcribe. Set --s3-bucket or use --use-whisper")
        
        s3_key = f"wafr-transcripts/{uuid.uuid4()}/{os.path.basename(audio_path)}"
        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
        
        try:
            # Upload audio to S3
            logger.info(f"Uploading audio to S3: {s3_uri}")
            self.s3_client.upload_file(audio_path, self.s3_bucket, s3_key)
            logger.info(f"Uploaded audio to {s3_uri}")
            
            # Start transcription job
            job_name = f"wafr-{uuid.uuid4()}"
            
            settings = {}
            if self.enable_diarization:
                settings["ShowSpeakerLabels"] = True
                settings["MaxSpeakerLabels"] = 10
            
            logger.info(f"Starting transcription job: {job_name}")
            self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": s3_uri},
                MediaFormat="wav",
                LanguageCode=self.language,
                Settings=settings if settings else None
            )
            
            # Wait for completion
            max_wait = 3600  # 1 hour max
            wait_time = 0
            while wait_time < max_wait:
                response = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                status = response["TranscriptionJob"]["TranscriptionJobStatus"]
                
                if status == "COMPLETED":
                    logger.info("Transcription completed")
                    break
                elif status == "FAILED":
                    failure_reason = response["TranscriptionJob"].get("FailureReason", "Unknown")
                    raise RuntimeError(f"Transcription job failed: {failure_reason}")
                
                time.sleep(5)
                wait_time += 5
                if wait_time % 30 == 0:
                    logger.info(f"Transcription status: {status} (waiting {wait_time}s)")
            
            if wait_time >= max_wait:
                raise RuntimeError("Transcription job timed out")
            
            # Download results
            result_uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            
            import urllib.request
            with urllib.request.urlopen(result_uri) as response:
                result = json.loads(response.read().decode())
            
            # Parse results into segments
            segments = self._parse_transcribe_result(result)
            logger.info(f"Transcribed {len(segments)} segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"AWS Transcribe failed: {e}")
            # Fallback to Whisper if available
            if self._whisper_available():
                logger.info("Falling back to Whisper")
                return self._transcribe_with_whisper(audio_path)
            raise
        finally:
            # Cleanup S3
            try:
                self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
                logger.info("Cleaned up S3 object")
            except Exception as e:
                logger.warning(f"Failed to cleanup S3 object: {e}")
    
    def _transcribe_with_whisper(self, audio_path: str) -> List[TranscriptSegment]:
        """Transcribe audio using OpenAI Whisper (local)"""
        logger.info("Transcribing with Whisper")
        
        if not self._whisper_available():
            raise RuntimeError(
                "Whisper not available. Install with: pip install openai-whisper\n"
                "Or use Amazon Transcribe with --s3-bucket option"
            )
        
        try:
            import whisper
            
            # Load model (base is a good balance of speed/accuracy)
            logger.info("Loading Whisper model...")
            model = whisper.load_model("base")
            
            # Transcribe with word timestamps
            logger.info("Transcribing audio...")
            result = model.transcribe(
                audio_path,
                language=self.language.split("-")[0] if "-" in self.language else self.language,  # "en-US" -> "en"
                word_timestamps=False  # Simplified for now
            )
            
            segments = []
            for segment in result["segments"]:
                segments.append(TranscriptSegment(
                    start_time=segment["start"],
                    end_time=segment["end"],
                    speaker=None,  # Whisper doesn't do diarization
                    text=segment["text"].strip(),
                    confidence=0.9  # Whisper doesn't provide confidence per segment
                ))
            
            logger.info(f"Transcribed {len(segments)} segments with Whisper")
            return segments
            
        except ImportError:
            raise RuntimeError(
                "Whisper not installed. Run: pip install openai-whisper\n"
                "Or use Amazon Transcribe with --s3-bucket option"
            )
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise
    
    def _whisper_available(self) -> bool:
        """Check if Whisper is available"""
        try:
            import whisper
            return True
        except ImportError:
            return False
    
    def _parse_transcribe_result(self, result: Dict) -> List[TranscriptSegment]:
        """Parse Amazon Transcribe result into segments"""
        segments = []
        
        items = result.get("results", {}).get("items", [])
        speaker_labels = result.get("results", {}).get("speaker_labels", {}).get("segments", [])
        
        # Build speaker map from timestamps
        speaker_map = {}
        for seg in speaker_labels:
            speaker_label = seg.get("speaker_label", "SPEAKER_00")
            for item in seg.get("items", []):
                start = float(item.get("start_time", 0))
                end = float(item.get("end_time", 0))
                key = f"{start:.3f}-{end:.3f}"
                speaker_map[key] = speaker_label
        
        # Group words into segments
        current_segment = None
        
        for item in items:
            if item.get("type") == "pronunciation":
                start = float(item.get("start_time", 0))
                end = float(item.get("end_time", 0))
                text = item["alternatives"][0]["content"]
                confidence = float(item["alternatives"][0].get("confidence", 0.9))
                
                # Get speaker
                key = f"{start:.3f}-{end:.3f}"
                speaker = speaker_map.get(key)
                
                # Start new segment on speaker change or long pause
                if (current_segment is None or 
                    speaker != current_segment.speaker or
                    start - current_segment.end_time > 2.0):
                    
                    if current_segment:
                        segments.append(current_segment)
                    
                    current_segment = TranscriptSegment(
                        start_time=start,
                        end_time=end,
                        speaker=speaker,
                        text=text,
                        confidence=confidence
                    )
                else:
                    # Extend current segment
                    current_segment.end_time = end
                    current_segment.text += " " + text
                    current_segment.confidence = (
                        current_segment.confidence + confidence
                    ) / 2
            
            elif item.get("type") == "punctuation":
                if current_segment:
                    current_segment.text += item["alternatives"][0]["content"]
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def _build_transcript_content(self, segments: List[TranscriptSegment]) -> str:
        """Build formatted transcript from segments"""
        lines = []
        current_speaker = None
        
        for segment in segments:
            # Add speaker label if changed
            if segment.speaker and segment.speaker != current_speaker:
                current_speaker = segment.speaker
                lines.append(f"\n[{current_speaker}]:")
            
            # Add timestamp and text
            timestamp = self._format_timestamp(segment.start_time)
            lines.append(f"[{timestamp}] {segment.text}")
        
        return "\n".join(lines)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using ffprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                fmt = data.get("format", {})
                
                return {
                    "duration": float(fmt.get("duration", 0)),
                    "format": fmt.get("format_name"),
                    "size_bytes": int(fmt.get("size", 0)),
                    "bit_rate": int(fmt.get("bit_rate", 0)) if fmt.get("bit_rate") else None
                }
        except FileNotFoundError:
            logger.warning("ffprobe not found - cannot extract video metadata")
        except subprocess.TimeoutExpired:
            logger.warning("ffprobe timed out")
        except Exception as e:
            logger.warning(f"Could not extract video metadata: {e}")
        
        return {}


def create_video_processor(**kwargs) -> VideoProcessor:
    """Factory function for creating video processor"""
    return VideoProcessor(**kwargs)

