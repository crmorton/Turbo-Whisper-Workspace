#!/usr/bin/env python3
"""
Minimal test script for Alexandra speaker identification
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
logger = logging.getLogger("minimal_test")

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

def main():
    """Test the speaker identification with a sample conversation mentioning Alexandra"""
    try:
        logger.info("🎭 Testing speaker identification with sample conversation")
        
        # Display the sample conversation
        for i, segment in enumerate(sample_conversation):
            logger.info(f"Segment {i}: {segment['speaker']} - \"{segment['text']}\"")
        
        # Import the llm_helper module
        import llm_helper
        
        # Test the fallback method first (doesn't require LLM)
        logger.info("\n🔍 Testing fallback speaker identification...")
        fallback_names = llm_helper.identify_speaker_names_fallback(sample_conversation)
        logger.info(f"Fallback speaker names: {json.dumps(fallback_names, indent=2)}")
        
        # Check if Alexandra was identified by the fallback method
        if any(name == "Alexandra" for name in fallback_names.values()):
            logger.info("🎉 SUCCESS: Alexandra was correctly identified by the fallback method!")
            # Show which speaker was identified as Alexandra
            for speaker, name in fallback_names.items():
                if name == "Alexandra":
                    logger.info(f"Speaker {speaker} was identified as Alexandra")
        else:
            logger.warning("⚠️ Alexandra was not identified by the fallback method")
        
        # Test the LLM method if available
        if llm_helper.LLAMA_AVAILABLE:
            logger.info("\n🧠 Testing LLM-based speaker identification...")
            llm_names = llm_helper.identify_speaker_names_llm(sample_conversation)
            logger.info(f"LLM speaker names: {json.dumps(llm_names, indent=2)}")
            
            # Check if Alexandra was identified by the LLM method
            if any(name == "Alexandra" for name in llm_names.values()):
                logger.info("🎉 SUCCESS: Alexandra was correctly identified by the LLM method!")
                # Show which speaker was identified as Alexandra
                for speaker, name in llm_names.items():
                    if name == "Alexandra":
                        logger.info(f"Speaker {speaker} was identified as Alexandra")
            else:
                logger.warning("⚠️ Alexandra was not identified by the LLM method")
        else:
            logger.warning("LLM not available, skipping LLM method")
        
        # Apply speaker names to segments
        logger.info("\n👥 Applying speaker names to segments...")
        names_to_use = llm_helper.LLAMA_AVAILABLE and llm_names or fallback_names
        for segment in sample_conversation:
            if segment["speaker"] in names_to_use:
                segment["speaker_name"] = names_to_use[segment["speaker"]]
        
        # Display updated segments
        logger.info("Updated segments with speaker names:")
        for i, segment in enumerate(sample_conversation):
            speaker_name = segment.get("speaker_name", segment["speaker"])
            logger.info(f"Segment {i}: {speaker_name} - \"{segment['text']}\"")
    
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
