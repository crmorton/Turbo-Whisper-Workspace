#!/usr/bin/env python3
"""
Minimal test script for the audio processing pipeline
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test segments
test_segments = [
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
    }
]

def main():
    """Main test function"""
    try:
        # Import llm_helper
        import llm_helper
        logger.info("Successfully imported llm_helper")
        
        # Test speaker name identification
        speaker_names = llm_helper.identify_speaker_names_fallback(test_segments)
        logger.info(f"Speaker names: {speaker_names}")
        
        # Apply speaker names to segments
        for segment in test_segments:
            if segment['speaker'] in speaker_names:
                segment['speaker_name'] = speaker_names[segment['speaker']]
        
        logger.info(f"Updated segments: {test_segments}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
