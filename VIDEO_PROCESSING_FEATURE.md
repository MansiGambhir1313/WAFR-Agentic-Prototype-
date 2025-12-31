# Video Processing Feature - Implementation Summary

## Overview

The WAFR system now supports video and audio files as input sources, in addition to PDF documents and text transcripts. This feature enables processing of workshop recordings, meeting videos, and audio interviews to generate Well-Architected Framework Review reports.

## What Was Implemented

### 1. Video Processor (`agents/video_processor.py`)

A new processor that handles video and audio file transcription:

**Key Features:**
- Audio extraction from video files using ffmpeg
- Speech-to-text transcription via:
  - Amazon Transcribe (AWS service, high quality)
  - OpenAI Whisper (local, no AWS required)
- Speaker diarization (identify different speakers)
- Timestamped transcript segments
- Video metadata extraction

**Key Methods:**
- `process(file_path)`: Process video files
- `process_audio(file_path)`: Process audio files directly
- `_extract_audio()`: Extract audio track from video
- `_transcribe_with_aws()`: Use Amazon Transcribe
- `_transcribe_with_whisper()`: Use local Whisper model

### 2. Updated Input Processor (`agents/input_processor.py`)

Enhanced to support video and audio processing:

**New Parameters:**
- `use_transcribe`: Enable/disable Amazon Transcribe
- `enable_diarization`: Enable speaker identification
- `transcribe_language`: Language code for transcription
- `s3_bucket`: S3 bucket for Transcribe (required for AWS)

**New Methods:**
- `video_processor` property: Lazy initialization of video processor
- Updated `process()` method to route video/audio files

### 3. Updated CLI (`run_wafr.py`)

Added command-line options for video processing:

**New Options:**
- `--use-whisper`: Use local Whisper instead of Amazon Transcribe
- `--no-diarization`: Disable speaker identification
- `--s3-bucket`: S3 bucket for Transcribe
- `--language`: Language code (default: en-US)

**Updated Help Text:**
- Added video/audio examples
- Updated input file description

### 4. Updated Dependencies (`agents/requirements.txt`)

Added:
- `openai-whisper>=20230314`: For local transcription

### 5. Updated Documentation (`WAFR_WORKFLOW_DOCUMENTATION.md`)

- Added video processing workflow to diagram
- Added video processing section to detailed steps
- Added usage examples for video/audio
- Added requirements and setup instructions

## Usage Examples

### Process Video with Amazon Transcribe

```bash
python run_wafr.py workshop_recording.mp4 \
  --s3-bucket my-wafr-bucket \
  --wa-tool \
  --client-name "Acme Corp" \
  --environment PRODUCTION
```

### Process Video with Local Whisper (No AWS)

```bash
python run_wafr.py meeting.mp4 \
  --use-whisper \
  --wa-tool \
  --client-name "Acme Corp"
```

### Process Audio File

```bash
python run_wafr.py interview.mp3 \
  --s3-bucket my-bucket \
  --language en-US \
  --wa-tool \
  --client-name "Client Name"
```

### Disable Speaker Diarization (Faster)

```bash
python run_wafr.py recording.mp4 \
  --s3-bucket my-bucket \
  --no-diarization
```

## System Requirements

### Required
- **ffmpeg**: For audio extraction from video files
  - Windows: Download from https://ffmpeg.org/download.html
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`

### Optional
- **Whisper**: For local transcription (if not using AWS)
  ```bash
  pip install openai-whisper
  ```

### AWS Requirements (for Amazon Transcribe)
- S3 bucket for file uploads
- IAM permissions for Transcribe and S3
- AWS credentials configured

## Supported Formats

### Video Formats
- MP4 (`.mp4`)
- WebM (`.webm`)
- QuickTime (`.mov`)
- AVI (`.avi`)
- Matroska (`.mkv`)
- MPEG-4 (`.m4v`)
- Windows Media (`.wmv`)

### Audio Formats
- MP3 (`.mp3`)
- WAV (`.wav`)
- M4A (`.m4a`)
- FLAC (`.flac`)
- OGG (`.ogg`)
- AAC (`.aac`)

## Workflow

1. **Input Detection**: System detects video/audio file type
2. **Audio Extraction**: ffmpeg extracts audio track (for video files)
3. **Transcription**: 
   - Amazon Transcribe (if S3 bucket provided)
   - Whisper (if `--use-whisper` flag)
4. **Speaker Diarization**: Identifies different speakers (if enabled)
5. **Transcript Formatting**: Creates timestamped transcript
6. **Pipeline Processing**: Transcript fed into existing WAFR pipeline
7. **Report Generation**: Same as PDF/text processing

## Error Handling

The implementation includes comprehensive error handling:

- **Graceful Fallbacks**: Falls back to Whisper if Transcribe fails
- **Clear Error Messages**: Helpful messages for missing dependencies
- **Partial Results**: Returns partial results if processing fails
- **Cleanup**: Automatically cleans up temporary files

## Key Features

1. **Dual Transcription Options**:
   - Amazon Transcribe: High quality, speaker diarization, requires AWS
   - Whisper: Local, no AWS, good quality, no diarization

2. **Speaker Diarization**:
   - Identifies different speakers in recordings
   - Labels segments with speaker IDs
   - Optional (can be disabled for faster processing)

3. **Timestamped Transcripts**:
   - Each segment includes start/end times
   - Format: `[HH:MM:SS] transcript text`
   - Speaker labels when diarization enabled

4. **Metadata Extraction**:
   - Video duration
   - File format
   - File size
   - Bit rate

## Integration Points

The video processing integrates seamlessly with the existing WAFR pipeline:

- **Input Processor**: Routes video/audio to video processor
- **Orchestrator**: Uses `process_file()` method (already supports all file types)
- **Pipeline**: Transcript from video is processed identically to text/PDF input
- **Report Generation**: Same report format regardless of input type

## Testing Recommendations

1. **Test with different video formats**
2. **Test with/without speaker diarization**
3. **Test with both Transcribe and Whisper**
4. **Test error handling (missing ffmpeg, no S3 bucket, etc.)**
5. **Test with various audio qualities**
6. **Test with different languages**

## Future Enhancements

Potential improvements:
- Video frame extraction for architecture diagrams
- Multiple language support in single video
- Real-time transcription streaming
- Video quality analysis
- Automatic chapter detection
- Noise reduction preprocessing

## Files Modified/Created

### New Files
- `agents/video_processor.py`: Video/audio processing implementation

### Modified Files
- `agents/input_processor.py`: Added video processing support
- `agents/requirements.txt`: Added Whisper dependency
- `run_wafr.py`: Added video processing CLI options
- `WAFR_WORKFLOW_DOCUMENTATION.md`: Updated with video processing info

## Notes

- Video processing is more time-consuming than PDF/text processing
- Large video files may take significant time to process
- Amazon Transcribe has file size limits (check AWS documentation)
- Whisper requires significant disk space for model downloads
- Speaker diarization adds processing time but improves transcript quality

