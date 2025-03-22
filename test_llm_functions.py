#!/usr/bin/env python3
"""
Test script for LLM-related functions in the audio processing pipeline
"""

import sys
import os
import logging
import json
import time

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
    import llm_helper
    LLM_AVAILABLE = True
    logger.info("LLM helper imported successfully")
except ImportError as e:
    LLM_AVAILABLE = False
    logger.error(f"Failed to import llm_helper: {e}")

def test_all_llm_functions():
    """Test all LLM-related functions with various test cases"""
    
    if not LLM_AVAILABLE:
        logger.error("LLM helper not available, cannot run test")
        return
    
    # Create test segments
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
    
    # Test speaker name identification
    logger.info("\n=== Testing Speaker Name Identification ===")
    try:
        speaker_names = llm_helper.identify_speaker_names_llm(test_segments)
        if not speaker_names:
            speaker_names = llm_helper.identify_speaker_names_fallback(test_segments)
        logger.info(f"Speaker names: {speaker_names}")
        
        # Apply speaker names to segments
        if speaker_names:
            for segment in test_segments:
                if segment['speaker'] in speaker_names:
                    segment['speaker_name'] = speaker_names[segment['speaker']]
            logger.info("Applied speaker names to segments")
    except Exception as e:
        logger.error(f"Error in speaker name identification: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test summarization
    logger.info("\n=== Testing Summarization ===")
    try:
        summary = llm_helper.summarize_conversation(test_segments)
        logger.info(f"Summary: {summary}")
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test topic extraction
    logger.info("\n=== Testing Topic Extraction ===")
    try:
        topics = llm_helper.extract_topics(test_segments)
        logger.info(f"Topics: {topics}")
    except Exception as e:
        logger.error(f"Error in topic extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test with invalid segments
    logger.info("\n=== Testing with Invalid Segments ===")
    invalid_segments = [
        {"speaker": "Speaker 0", "start": 0.0, "end": 5.0},  # Missing text
        {"text": "Hello", "start": 5.0, "end": 10.0}  # Missing speaker
    ]
    
    try:
        speaker_names = llm_helper.identify_speaker_names_llm(invalid_segments)
        logger.info(f"Speaker names (invalid segments): {speaker_names}")
        
        summary = llm_helper.summarize_conversation(invalid_segments)
        logger.info(f"Summary (invalid segments): {summary}")
        
        topics = llm_helper.extract_topics(invalid_segments)
        logger.info(f"Topics (invalid segments): {topics}")
    except Exception as e:
        logger.error(f"Error with invalid segments: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test with empty segments
    logger.info("\n=== Testing with Empty Segments ===")
    try:
        speaker_names = llm_helper.identify_speaker_names_llm([])
        logger.info(f"Speaker names (empty segments): {speaker_names}")
        
        summary = llm_helper.summarize_conversation([])
        logger.info(f"Summary (empty segments): {summary}")
        
        topics = llm_helper.extract_topics([])
        logger.info(f"Topics (empty segments): {topics}")
    except Exception as e:
        logger.error(f"Error with empty segments: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting LLM functions test")
    start_time = time.time()
    test_all_llm_functions()
    end_time = time.time()
    logger.info(f"Test completed in {end_time - start_time:.2f} seconds")
