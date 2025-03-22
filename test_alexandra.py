#!/usr/bin/env python3
"""
Test script for Alexandra speaker identification using the provided audio file
"""

import os
import sys
import json
import time
import logging

# Configure basic logging with colorful output
logging.basicConfig(
    level=logging.INFO,
    format='\033[1;36m%(asctime)s\033[0m - \033[1;32m%(name)s\033[0m - \033[1;34m%(levelname)s\033[0m - \033[0m%(message)s\033[0m',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_alexandra")

def main():
    """Test the speaker identification with the Alexandra audio file"""
    try:
        # Audio file path
        audio_file = "examples/ChrisAndAlexDiTest.flac"
        
        # Check if file exists
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return
        
        logger.info(f"🎧 Testing with audio file: {audio_file}")
        
        # First, let's use the existing transcription from output.json
        if os.path.exists("output.json"):
            logger.info("Using existing transcription from output.json")
            with open("output.json", 'r') as f:
                data = json.load(f)
            
            # Create segments with speaker information
            segments = []
            for i, chunk in enumerate(data.get("chunks", [])):
                # Alternate speakers for demonstration
                speaker = f"Speaker {i % 2}"
                
                segments.append({
                    "speaker": speaker,
                    "text": chunk.get("text", ""),
                    "start": chunk.get("timestamp", [0, 0])[0],
                    "end": chunk.get("timestamp", [0, 0])[1]
                })
            
            logger.info(f"Created {len(segments)} segments with dummy speaker assignments")
            
            # Display the segments
            for i, segment in enumerate(segments):
                logger.info(f"Segment {i}: {json.dumps(segment, indent=2)}")
            
            # Now let's try to identify speaker names
            logger.info("🔍 Identifying speaker names using LLM...")
            import llm_helper
            
            # Test the LLM method
            if llm_helper.LLAMA_AVAILABLE:
                start_time = time.time()
                speaker_names = llm_helper.identify_speaker_names_llm(segments)
                elapsed_time = time.time() - start_time
                
                logger.info(f"✨ Speaker identification completed in {elapsed_time:.2f} seconds")
                logger.info(f"Identified speaker names: {json.dumps(speaker_names, indent=2)}")
                
                # Check if Alexandra was identified
                if any(name == "Alexandra" for name in speaker_names.values()):
                    logger.info("🎉 SUCCESS: Alexandra was correctly identified!")
                else:
                    logger.warning("⚠️ Alexandra was not identified by the LLM method")
            else:
                logger.warning("LLM not available, skipping LLM method")
            
            # Try the fallback method
            logger.info("🔄 Trying fallback speaker identification...")
            fallback_names = llm_helper.identify_speaker_names_fallback(segments)
            logger.info(f"Fallback speaker names: {json.dumps(fallback_names, indent=2)}")
            
            # Check if Alexandra was identified by the fallback method
            if any(name == "Alexandra" for name in fallback_names.values()):
                logger.info("🎉 SUCCESS: Alexandra was correctly identified by the fallback method!")
            else:
                logger.warning("⚠️ Alexandra was not identified by the fallback method")
                
            # Apply speaker names to segments
            logger.info("Applying speaker names to segments...")
            for segment in segments:
                if segment["speaker"] in fallback_names:
                    segment["speaker_name"] = fallback_names[segment["speaker"]]
            
            # Display updated segments
            logger.info("Updated segments with speaker names:")
            for i, segment in enumerate(segments):
                logger.info(f"Segment {i}: {json.dumps(segment, indent=2)}")
        
        else:
            logger.error("output.json not found. Please run insanely-fast-whisper on the audio file first.")
    
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
