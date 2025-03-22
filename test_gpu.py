#!/usr/bin/env python3
"""
Test script to verify GPU usage with the LLM
"""

import os
import time
import logging
import torch
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_usage():
    """Check GPU usage using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        logger.info(f"GPU Stats: {result.stdout.strip()}")
    except Exception as e:
        logger.error(f"Error checking GPU stats: {e}")

def test_gpu_usage():
    """Test GPU usage with the LLM"""
    # Check CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Check initial GPU usage
    logger.info("Initial GPU usage:")
    check_gpu_usage()
    
    # Import LLM helper after checking GPU
    from llm_helper import get_llm
    
    # Get LLM
    logger.info("Initializing LLM...")
    llm = get_llm()
    if not llm:
        logger.error("Failed to initialize LLM")
        return
    
    logger.info("LLM initialized successfully")
    
    # Check GPU usage after LLM initialization
    logger.info("GPU usage after LLM initialization:")
    check_gpu_usage()
    
    # Test LLM with a simple prompt
    prompt = """
    You are a helpful AI assistant. Please answer the following question:
    What is the capital of France?
    """
    
    logger.info("Sending test prompt to LLM...")
    start_time = time.time()
    response = llm.create_completion(
        prompt=prompt,
        max_tokens=100,
        temperature=0.1,
        stop=["```"]
    )
    elapsed_time = time.time() - start_time
    
    # Check GPU usage after LLM completion
    logger.info(f"LLM completion took {elapsed_time:.2f} seconds")
    logger.info("GPU usage after LLM completion:")
    check_gpu_usage()
    
    # Print response
    response_text = response['choices'][0]['text'].strip()
    logger.info(f"LLM response: {response_text}")
    
    logger.info("GPU test completed successfully")

if __name__ == "__main__":
    test_gpu_usage()
