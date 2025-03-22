#!/usr/bin/env python3
"""
Test script for speaker name identification
"""

import os
import sys
import logging
import traceback
import json
import time

# Configure logging to console with colorful output
logging.basicConfig(
    level=logging.INFO,
    format='\033[1;36m%(asctime)s\033[0m - \033[1;32m%(name)s\033[0m - \033[1;34m%(levelname)s\033[0m - \033[0m%(message)s\033[0m',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_speaker_id")

# Sample conversation data with speaker names
SAMPLE_CONVERSATION = [
    {
        "speaker": "Speaker 0",
        "text": "Hi there! My name is Veronica. I'm calling about the software issue I've been having.",
        "start": 0.0,
        "end": 5.2
    },
    {
        "speaker": "Speaker 1",
        "text": "Hello Veronica, this is Alex from tech support. How can I help you today?",
        "start": 5.5,
        "end": 10.1
    },
    {
        "speaker": "Speaker 0",
        "text": "Thanks Alex. I've been having trouble with the audio processing software. It keeps crashing.",
        "start": 10.5,
        "end": 15.3
    },
    {
        "speaker": "Speaker 1",
        "text": "I understand, Veronica. Let me check what might be causing that issue.",
        "start": 15.8,
        "end": 20.2
    }
]

def test_speaker_identification():
    """Test the speaker name identification functionality"""
    try:
        logger.info("🎤 Testing speaker name identification")
        
        # Import llm_helper
        try:
            logger.info("Importing llm_helper...")
            import llm_helper
            logger.info(f"LLM available: {llm_helper.LLAMA_AVAILABLE}")
            
            # Test identify_speaker_names_llm function
            logger.info("Testing identify_speaker_names_llm function with sample conversation...")
            
            # Debug the segments before passing to the function
            logger.info(f"Sample conversation has {len(SAMPLE_CONVERSATION)} segments")
            for i, segment in enumerate(SAMPLE_CONVERSATION):
                logger.info(f"Segment {i}: {json.dumps(segment, indent=2)}")
            
            # Call the function
            start_time = time.time()
            speaker_names = llm_helper.identify_speaker_names_llm(SAMPLE_CONVERSATION)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Speaker identification completed in {elapsed_time:.2f} seconds")
            logger.info(f"Identified speaker names: {json.dumps(speaker_names, indent=2)}")
            
            # Test the fallback method
            logger.info("Testing identify_speaker_names_fallback function...")
            fallback_names = llm_helper.identify_speaker_names_fallback(SAMPLE_CONVERSATION)
            logger.info(f"Fallback speaker names: {json.dumps(fallback_names, indent=2)}")
            
            # Test with invalid data
            logger.info("Testing with invalid data...")
            invalid_result = llm_helper.identify_speaker_names_llm(None)
            logger.info(f"Result with None: {invalid_result}")
            
            invalid_result = llm_helper.identify_speaker_names_llm([])
            logger.info(f"Result with empty list: {invalid_result}")
            
            invalid_result = llm_helper.identify_speaker_names_llm([{"wrong_key": "value"}])
            logger.info(f"Result with invalid segment: {invalid_result}")
            
        except Exception as e:
            logger.error(f"Error in LLM helper: {e}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())

def test_with_audio_file(audio_file="examples/ChrisAndAlexDiTest.flac"):
    """Test speaker identification with a real audio file"""
    try:
        logger.info(f"🎧 Testing speaker identification with audio file: {audio_file}")
        
        # Check if file exists
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return
        
        # Import audio pipeline
        try:
            logger.info("Importing audio pipeline...")
            from audio_pipeline import AudioProcessingPipeline
            
            # Initialize the pipeline
            pipeline = AudioProcessingPipeline()
            
            # Process the audio file
            logger.info("Starting audio processing...")
            start_time = time.time()
            result = pipeline.process_audio(
                audio_path=audio_file,
                task="transcribe",
                num_speakers=2,  # We know there are 2 speakers in this file
                threshold=0.5
            )
            elapsed_time = time.time() - start_time
            
            logger.info(f"Audio processing completed in {elapsed_time:.2f} seconds")
            
            # Check for errors
            if "error" in result:
                logger.error(f"Error in audio processing: {result['error']}")
                return
                
            # Log processing times
            if "processing_times" in result:
                times = result["processing_times"]
                logger.info("Processing times:")
                for step, time_taken in times.items():
                    logger.info(f"  {step}: {time_taken:.2f} seconds")
            
            # Check if segments were generated
            if "segments" not in result or not result["segments"]:
                logger.error("No segments were generated")
                return
                
            segments = result["segments"]
            logger.info(f"Generated {len(segments)} segments")
            
            # Display a few segments for verification
            logger.info("Sample segments:")
            for i, segment in enumerate(segments[:5]):  # Show first 5 segments
                logger.info(f"Segment {i}: {json.dumps(segment, indent=2)}")
                
            # Check if speaker names were identified
            if "speaker_names" in result and result["speaker_names"]:
                speaker_names = result["speaker_names"]
                logger.info(f"Identified speaker names: {json.dumps(speaker_names, indent=2)}")
                
                # Verify speaker names in segments
                named_segments = 0
                for segment in segments:
                    if "speaker_name" in segment:
                        named_segments += 1
                
                logger.info(f"{named_segments} out of {len(segments)} segments have speaker names")
            else:
                logger.warning("No speaker names were identified in pipeline")
                
                # Try direct speaker identification
                logger.info("Trying direct speaker identification...")
                import llm_helper
                speaker_names = llm_helper.identify_speaker_names_llm(segments)
                logger.info(f"Direct speaker identification result: {json.dumps(speaker_names, indent=2)}")
                
                # Try fallback method
                logger.info("Trying fallback speaker identification...")
                fallback_names = llm_helper.identify_speaker_names_fallback(segments)
                logger.info(f"Fallback speaker names: {json.dumps(fallback_names, indent=2)}")
            
            # Check summary and topics if available
            if "summary" in result and result["summary"]:
                logger.info(f"Summary: {result['summary']}")
                
            if "topics" in result and result["topics"]:
                logger.info(f"Topics: {json.dumps(result['topics'], indent=2)}")
                
        except Exception as e:
            logger.error(f"Error in audio pipeline: {e}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Choose which test to run
    import argparse
    parser = argparse.ArgumentParser(description="Test speaker identification")
    parser.add_argument("--audio", action="store_true", help="Run test with audio file")
    parser.add_argument("--file", type=str, default="examples/ChrisAndAlexDiTest.flac", help="Audio file to test with")
    args = parser.parse_args()
    
    if args.audio:
        test_with_audio_file(args.file)
    else:
        test_speaker_identification()
