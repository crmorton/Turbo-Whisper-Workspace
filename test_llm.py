#!/usr/bin/env python3
"""
Test script for LLM initialization
"""

import os
import sys
import logging
import traceback

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_llm")

def test_llm_init():
    """Test LLM initialization"""
    try:
        logger.info("Testing LLM initialization")
        
        # Import torch and check CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            
            if cuda_available:
                cuda_device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                logger.info(f"CUDA device count: {cuda_device_count}")
                logger.info(f"Current CUDA device: {current_device}")
                logger.info(f"Current CUDA device name: {device_name}")
        except Exception as e:
            logger.error(f"Error checking CUDA: {e}")
        
        # Import llm_helper
        try:
            logger.info("Importing llm_helper...")
            import llm_helper
            logger.info(f"LLM available: {llm_helper.LLAMA_AVAILABLE}")
            logger.info(f"Current model: {llm_helper.CURRENT_MODEL}")
            
            # Try to get LLM
            logger.info("Getting LLM instance...")
            llm = llm_helper.get_llm()
            
            if llm:
                logger.info("LLM initialized successfully!")
                
                # Test a simple completion
                prompt = "Hello, my name is Veronica. What's your name?"
                logger.info(f"Testing completion with prompt: {prompt}")
                
                response = llm.create_completion(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.7
                )
                
                logger.info(f"Response: {response['choices'][0]['text']}")
            else:
                logger.error("LLM initialization failed!")
        except Exception as e:
            logger.error(f"Error in LLM helper: {e}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_llm_init()
