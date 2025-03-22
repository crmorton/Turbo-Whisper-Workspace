#!/usr/bin/env python3
"""
Test script for speaker name identification using a real audio file
"""

import os
import sys
import logging
import traceback
import json
import time

# Define some ANSI color codes for pretty output
CYAN = '\033[1;36m'
GREEN = '\033[1;32m'
BLUE = '\033[1;34m'
RESET = '\033[0m'

# Configure logging to console with colorful output
logging.basicConfig(
    level=logging.INFO,
    format='\033[1;36m%(asctime)s\033[0m - \033[1;32m%(name)s\033[0m - \033[1;34m%(levelname)s\033[0m - \033[0m%(message)s\033[0m',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_audio_file")

def test_audio_processing():
    """Test the audio processing pipeline with speaker name identification"""
    try:
        logger.info("🎤 Testing audio processing pipeline with speaker name identification")
        
        # Import audio pipeline
        try:
            logger.info("Importing audio pipeline...")
            from audio_pipeline import AudioProcessingPipeline
            
            # Import llm_helper for direct testing of speaker identification
            import llm_helper
            logger.info(f"LLM available: {llm_helper.LLAMA_AVAILABLE}")
            
            # Audio file path
            audio_file = "examples/ChrisAndAlexDiTest.flac"
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return
                
            logger.info(f"Processing audio file: {audio_file}")
            
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
                logger.warning("No speaker names were identified")
                
                # Try direct speaker identification as a fallback
                logger.info("Trying direct speaker identification...")
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
    test_audio_processing()
