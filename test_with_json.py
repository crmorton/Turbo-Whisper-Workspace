#!/usr/bin/env python3
"""
Test script for speaker name identification using existing JSON output
"""

import os
import sys
import json
import time
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_with_json")

def main():
    # JSON file path
    json_file = "output.json"
    
    # Check if file exists
    if not os.path.exists(json_file):
        logger.error(f"JSON file not found: {json_file}")
        return
    
    logger.info(f"Testing with JSON file: {json_file}")
    
    try:
        # Load the JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        logger.info("Successfully loaded JSON data")
        
        # Import llm_helper
        import llm_helper
        logger.info(f"LLM available: {llm_helper.LLAMA_AVAILABLE}")
        
        # First, we need to convert the chunks to segments with speaker information
        # Since the JSON doesn't have speaker info, we'll create dummy speakers
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
        logger.info("Identifying speaker names using LLM...")
        start_time = time.time()
        speaker_names = llm_helper.identify_speaker_names_llm(segments)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Speaker identification completed in {elapsed_time:.2f} seconds")
        logger.info(f"Identified speaker names: {json.dumps(speaker_names, indent=2)}")
        
        # Try the fallback method as well
        logger.info("Trying fallback speaker identification...")
        fallback_names = llm_helper.identify_speaker_names_fallback(segments)
        logger.info(f"Fallback speaker names: {json.dumps(fallback_names, indent=2)}")
        
        # Apply speaker names to segments
        if speaker_names:
            logger.info("Applying speaker names to segments...")
            for segment in segments:
                if segment["speaker"] in speaker_names:
                    segment["speaker_name"] = speaker_names[segment["speaker"]]
            
            # Display updated segments
            logger.info("Updated segments with speaker names:")
            for i, segment in enumerate(segments):
                logger.info(f"Segment {i}: {json.dumps(segment, indent=2)}")
        
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
