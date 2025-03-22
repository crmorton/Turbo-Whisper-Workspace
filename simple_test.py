#!/usr/bin/env python3
"""
Simple test script for speaker name identification using a real audio file
"""

import os
import sys
import json
import time
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_test")

def main():
    # Audio file path
    audio_file = "examples/ChrisAndAlexDiTest.flac"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return
    
    logger.info(f"Testing with audio file: {audio_file}")
    
    try:
        # Import necessary modules
        from audio_pipeline import AudioProcessingPipeline
        import llm_helper
        
        # Initialize the pipeline
        pipeline = AudioProcessingPipeline()
        
        # Process the audio file
        logger.info("Starting audio processing...")
        result = pipeline.process_audio(
            audio_path=audio_file,
            task="transcribe",
            num_speakers=2,  # We know there are 2 speakers in this file
            threshold=0.5
        )
        
        # Check for errors
        if "error" in result:
            logger.error(f"Error in audio processing: {result['error']}")
            return
        
        # Check if segments were generated
        if "segments" not in result or not result["segments"]:
            logger.error("No segments were generated")
            return
            
        segments = result["segments"]
        logger.info(f"Generated {len(segments)} segments")
        
        # Display a few segments
        for i, segment in enumerate(segments[:3]):  # Show first 3 segments
            logger.info(f"Segment {i}: {json.dumps(segment, indent=2)}")
        
        # Check if speaker names were identified
        if "speaker_names" in result and result["speaker_names"]:
            speaker_names = result["speaker_names"]
            logger.info(f"Identified speaker names: {json.dumps(speaker_names, indent=2)}")
        else:
            logger.warning("No speaker names were identified in pipeline")
            
            # Try direct speaker identification
            logger.info("Trying direct speaker identification...")
            speaker_names = llm_helper.identify_speaker_names_llm(segments)
            logger.info(f"Direct speaker identification result: {json.dumps(speaker_names, indent=2)}")
            
            # Try fallback method
            logger.info("Trying fallback speaker identification...")
            fallback_names = llm_helper.identify_speaker_names_fallback(segments)
            logger.info(f"Fallback speaker names: {json.dumps(fallback_names, indent=2)}")
        
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
