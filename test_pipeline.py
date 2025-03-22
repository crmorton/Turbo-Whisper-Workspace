#!/usr/bin/env python3
"""
Test script for the complete audio processing pipeline
"""

import sys
import os
import logging
import json
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
try:
    from audio_pipeline import AudioProcessingPipeline
    logger.info("Audio pipeline imported successfully")
except ImportError as e:
    logger.error(f"Failed to import audio_pipeline: {e}")
    sys.exit(1)

def simulate_diarization_segments() -> List[Dict[str, Any]]:
    """
    Simulate diarization segments for testing
    """
    return [
        {
            "speaker": "Speaker 0",
            "text": "Hello everyone, my name is Veronica and I'm here to discuss the project timeline.",
            "start": 0.0,
            "end": 5.0
        },
        {
            "speaker": "Speaker 1",
            "text": "Hi Veronica, I'm John. I have some concerns about the budget for this quarter.",
            "start": 5.0,
            "end": 10.0
        },
        {
            "speaker": "Speaker 0",
            "text": "That's a good point, John. We need to review our expenses and make some adjustments.",
            "start": 10.0,
            "end": 15.0
        },
        {
            "speaker": "Speaker 1",
            "text": "I think we should focus on the marketing strategy first, then address the budget concerns.",
            "start": 15.0,
            "end": 20.0
        },
        {
            "speaker": "Speaker 0",
            "text": "Agreed. Let's schedule a meeting with the marketing team next week.",
            "start": 20.0,
            "end": 25.0
        }
    ]

def simulate_transcription() -> Dict[str, Any]:
    """
    Simulate transcription results for testing
    """
    return {
        "text": "Hello everyone, my name is Veronica and I'm here to discuss the project timeline. "
                "Hi Veronica, I'm John. I have some concerns about the budget for this quarter. "
                "That's a good point, John. We need to review our expenses and make some adjustments. "
                "I think we should focus on the marketing strategy first, then address the budget concerns. "
                "Agreed. Let's schedule a meeting with the marketing team next week.",
        "chunks": [
            {"text": "Hello everyone, my name is Veronica and I'm here to discuss the project timeline.", "start": 0.0, "end": 5.0},
            {"text": "Hi Veronica, I'm John. I have some concerns about the budget for this quarter.", "start": 5.0, "end": 10.0},
            {"text": "That's a good point, John. We need to review our expenses and make some adjustments.", "start": 10.0, "end": 15.0},
            {"text": "I think we should focus on the marketing strategy first, then address the budget concerns.", "start": 15.0, "end": 20.0},
            {"text": "Agreed. Let's schedule a meeting with the marketing team next week.", "start": 20.0, "end": 25.0}
        ]
    }

def test_pipeline():
    """Test the complete audio processing pipeline"""
    
    logger.info("Creating audio pipeline instance")
    pipeline = AudioProcessingPipeline()
    
    # Mock the diarize and transcribe methods
    pipeline.diarize = lambda *args, **kwargs: simulate_diarization_segments()
    pipeline.transcribe = lambda *args, **kwargs: simulate_transcription()
    
    # Create a mock audio file path
    mock_audio_path = "/path/to/mock/audio.wav"
    
    logger.info("Processing mock audio file")
    result = pipeline.process_audio(
        audio_path=mock_audio_path,
        task="transcribe",
        segmentation_model="",
        embedding_model="",
        num_speakers=2,
        threshold=0.5
    )
    
    # Print results
    logger.info("\n=== Processing Results ===")
    
    if "error" in result:
        logger.error(f"Error: {result['error']}")
        return
    
    logger.info(f"Processing times: {json.dumps(result.get('processing_times', {}), indent=2)}")
    
    if "speaker_names" in result:
        logger.info(f"Speaker names: {json.dumps(result['speaker_names'], indent=2)}")
    else:
        logger.warning("No speaker names identified")
    
    if "summary" in result:
        logger.info(f"Summary: {result['summary']}")
    else:
        logger.warning("No summary generated")
    
    if "topics" in result:
        logger.info(f"Topics: {json.dumps(result['topics'], indent=2)}")
    else:
        logger.warning("No topics extracted")
    
    # Print the first few segments
    if "segments" in result:
        logger.info(f"Number of segments: {len(result['segments'])}")
        logger.info(f"First 2 segments: {json.dumps(result['segments'][:2], indent=2)}")
    else:
        logger.warning("No segments in result")

if __name__ == "__main__":
    logger.info("Starting pipeline test")
    start_time = time.time()
    test_pipeline()
    end_time = time.time()
    logger.info(f"Test completed in {end_time - start_time:.2f} seconds")
