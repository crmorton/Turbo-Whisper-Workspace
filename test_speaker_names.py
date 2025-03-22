#!/usr/bin/env python3
"""
Test script for speaker name identification
"""

import sys
import os
import logging
import json

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

def test_speaker_name_identification():
    """Test speaker name identification with various test cases"""
    
    if not LLM_AVAILABLE:
        logger.error("LLM helper not available, cannot run test")
        return
    
    # Test case 1: Basic test with Veronica
    test_segments_1 = [
        {"speaker": "Speaker 0", "text": "Hello, my name is Veronica.", "start": 0.0, "end": 2.0},
        {"speaker": "Speaker 1", "text": "Hi Veronica, I'm John.", "start": 2.0, "end": 4.0}
    ]
    
    # Test case 2: Indirect mention of Veronica
    test_segments_2 = [
        {"speaker": "Speaker 0", "text": "Did you talk to Veronica yesterday?", "start": 0.0, "end": 2.0},
        {"speaker": "Speaker 1", "text": "Yes, she told me about the project.", "start": 2.0, "end": 4.0}
    ]
    
    # Test case 3: No explicit names
    test_segments_3 = [
        {"speaker": "Speaker 0", "text": "How are you doing today?", "start": 0.0, "end": 2.0},
        {"speaker": "Speaker 1", "text": "I'm doing well, thank you for asking.", "start": 2.0, "end": 4.0}
    ]
    
    # Test case 4: Invalid segments
    test_segments_4 = [
        {"speaker": "Speaker 0", "start": 0.0, "end": 2.0},  # Missing text
        {"text": "Hi there", "start": 2.0, "end": 4.0}  # Missing speaker
    ]
    
    # Test case 5: Empty segments
    test_segments_5 = []
    
    # Run tests
    test_cases = [
        ("Test 1: Basic test with Veronica", test_segments_1),
        ("Test 2: Indirect mention of Veronica", test_segments_2),
        ("Test 3: No explicit names", test_segments_3),
        ("Test 4: Invalid segments", test_segments_4),
        ("Test 5: Empty segments", test_segments_5)
    ]
    
    for test_name, segments in test_cases:
        logger.info(f"\n=== {test_name} ===")
        logger.info(f"Input segments: {json.dumps(segments, indent=2)}")
        
        # Test LLM method
        try:
            logger.info("Testing LLM method...")
            llm_names = llm_helper.identify_speaker_names_llm(segments)
            logger.info(f"LLM result: {llm_names}")
        except Exception as e:
            logger.error(f"Error in LLM method: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Test fallback method
        try:
            logger.info("Testing fallback method...")
            fallback_names = llm_helper.identify_speaker_names_fallback(segments)
            logger.info(f"Fallback result: {fallback_names}")
        except Exception as e:
            logger.error(f"Error in fallback method: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting speaker name identification test")
    test_speaker_name_identification()
    logger.info("Test completed")
