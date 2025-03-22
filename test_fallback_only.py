#!/usr/bin/env python3
"""
Simple test script for the fallback speaker identification method
"""

import json
import logging
import sys

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fallback_test")

# Create sample conversation with Alexandra mentioned
sample_conversation = [
    {"speaker": "Speaker 0", "text": "Hey Alexandra, you gotta talk.", "start": 0.0, "end": 2.0},
    {"speaker": "Speaker 1", "text": "I do. Hey, how are you? How are you doing tonight?", "start": 2.0, "end": 5.0},
    {"speaker": "Speaker 0", "text": "Seven seconds. Could you say 15 seconds?", "start": 5.0, "end": 8.0},
    {"speaker": "Speaker 1", "text": "No? 15 seconds? You want me to say 15 seconds?", "start": 8.0, "end": 12.0},
    {"speaker": "Speaker 0", "text": "That's all I needed. Thank you.", "start": 12.0, "end": 14.0},
    {"speaker": "Speaker 1", "text": "Sure, no problem.", "start": 14.0, "end": 15.0}
]

# Expected correct mapping: Speaker 0 is Chris, Speaker 1 is Alexandra
# Since Speaker 0 is addressing Alexandra in the first segment

if __name__ == "__main__":
    try:
        # Import the llm_helper module
        import llm_helper
        
        # Test the fallback method
        logger.info("🔍 Testing fallback speaker identification...")
        fallback_names = llm_helper.identify_speaker_names_fallback(sample_conversation)
        logger.info(f"Fallback speaker names: {json.dumps(fallback_names, indent=2)}")
        
        # Check if Alexandra was identified correctly
        if "Speaker 1" in fallback_names and fallback_names["Speaker 1"] == "Alexandra":
            logger.info("✅ SUCCESS: Alexandra was correctly identified as Speaker 1!")
        elif "Speaker 0" in fallback_names and fallback_names["Speaker 0"] == "Alexandra":
            logger.error("❌ ERROR: Alexandra was incorrectly identified as Speaker 0 (the person addressing her)")
        elif any(name == "Alexandra" for name in fallback_names.values()):
            logger.warning("⚠️ Alexandra was identified but assigned to an unexpected speaker")
            for speaker, name in fallback_names.items():
                if name == "Alexandra":
                    logger.info(f"Speaker {speaker} was identified as Alexandra")
        else:
            logger.error("❌ ERROR: Alexandra was not identified at all")
            
        # Apply speaker names to segments
        logger.info("\n👥 Applying speaker names to segments...")
        for segment in sample_conversation:
            if segment["speaker"] in fallback_names:
                segment["speaker_name"] = fallback_names[segment["speaker"]]
        
        # Display updated segments
        logger.info("Updated segments with speaker names:")
        for i, segment in enumerate(sample_conversation):
            speaker_name = segment.get("speaker_name", segment["speaker"])
            logger.info(f"Segment {i}: {speaker_name} - \"{segment['text']}\"")
            
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
