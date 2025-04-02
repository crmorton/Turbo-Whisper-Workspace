# Vocalis - Advanced Audio Processing

Vocalis is a powerful audio processing package featuring:
- Ultra-fast Whisper V3 Turbo Transcription
- Advanced Speaker Diarization
- Audio Analysis Tools
- Security Monitoring
- FastAPI Integration

## Features

### Audio Processing
- Transcription using Whisper V3 Turbo
- Speaker diarization with pyannote.audio and sherpa-onnx
- Speaker name identification
- Conversation summarization
- Topic extraction

### Security Monitoring
- Detect potential security incidents in audio
- Specialized bar security monitoring
- Threat level assessment
- Incident reporting

### API and UI
- FastAPI integration for all functionality
- Gradio UI for interactive usage
- Command-line interface

## Installation

### Basic Installation

```bash
pip install vocalis
```

### With GPU Support

```bash
pip install vocalis[gpu]
```

### Development Installation

```bash
pip install vocalis[dev]
```

## Usage

### Command Line Interface

Vocalis provides a command-line interface for common tasks:

```bash
# Run the FastAPI server
python -m vocalis api --port 8000

# Run the Gradio UI
python -m vocalis ui

# Run security monitoring on a file
python -m vocalis security --input audio.flac --threat-level 2

# Run bar-specific security monitoring on a directory
python -m vocalis security --input ./examples/bar --bar
```

### API Usage

Start the API server:

```bash
python -m vocalis api
```

Then use the API endpoints:

- `POST /api/transcribe` - Transcribe and diarize audio
- `POST /api/security/analyze` - Analyze audio for security concerns
- `POST /api/analyze` - Analyze audio characteristics
- `GET /api/models` - Get available models

### Python API

```python
from vocalis.core.audio_pipeline import AudioProcessingPipeline

# Initialize pipeline
pipeline = AudioProcessingPipeline()

# Process audio
result = pipeline.process_audio(
    audio_path="audio.flac",
    task="transcribe",
    num_speakers=2
)

# Access results
print(result["text"])
for segment in result["merged_segments"]:
    print(f"{segment['speaker']}: {segment['text']}")
```

## Security Monitoring

```python
from vocalis.security.security_monitor import SecurityMonitor

# Initialize security monitor
monitor = SecurityMonitor(output_dir="security_incidents", min_threat_level=2)

# Process audio file
incident = monitor.process_audio_file("audio.flac")

if incident:
    print(f"Security incident detected: {incident.incident_type}")
    print(f"Threat level: {incident.threat_level}/5")
    print(f"Summary: {incident.summary}")
```

## Credits

This project builds upon several amazing technologies:
- [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio](https://gradio.app/)

## License

Apache License 2.0
