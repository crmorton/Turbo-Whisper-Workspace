∑#!/usr/bin/env python3
"""
GPU Optimization Script for Turbo-Whisper-Workspace

This script helps diagnose and optimize GPU usage for the LLM.
"""
import os
import sys
import time
import logging
import torch
import subprocess
from pathlib import Path

# Add parent directory to path so we can import from the project
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)∑

def check_gpu_availability():
    """Check if GPU is available and print detailed information"""
    logger.info("=== GPU Availability Check ===")
    
    if torch.cuda.is_available():
        logger.info("✅ CUDA is available")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        
        device_count = torch.cuda.device_count()
        logger.info(f"GPU Count: {device_count}")
        
        for i in range(device_count):
            logger.info(f"=== GPU {i} Details ===")
            logger.info(f"Device Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"Compute Capability: {props.major}.{props.minor}")
            logger.info(f"Multi Processor Count: {props.multi_processor_count}")
            
            # Current memory usage
            logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            logger.info(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        logger.error("❌ CUDA is not available")
        logger.info("Checking system GPU information...")
        
        try:
            # Try to get GPU info using lspci
            result = subprocess.run(["lspci", "-v"], capture_output=True, text=True)
            gpu_info = [line for line in result.stdout.split('\n') if "VGA" in line or "NVIDIA" in line]
            if gpu_info:
                logger.info("GPU Hardware detected:")
                for line in gpu_info:
                    logger.info(f"  {line.strip()}")
            else:
                logger.info("No GPU hardware detected with lspci")
        except Exception as e:
            logger.error(f"Error checking GPU hardware: {e}")
    
    return torch.cuda.is_available()

def test_gpu_performance():
    """Run a simple test to measure GPU performance"""
    if not torch.cuda.is_available():
        logger.error("Cannot test GPU performance - CUDA not available")
        return
    
    logger.info("=== GPU Performance Test ===")
    
    # Test matrix multiplication performance
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        logger.info(f"Testing matrix multiplication with size {size}x{size}")
        
        # Create random matrices
        torch.cuda.empty_cache()  # Clear cache before test
        
        # CPU timing
        start_time = time.time()
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        logger.info(f"CPU Time: {cpu_time:.4f} seconds")
        
        # GPU timing
        start_time = time.time()
        a_gpu = torch.randn(size, size, device='cuda')
        b_gpu = torch.randn(size, size, device='cuda')
        torch.cuda.synchronize()  # Wait for GPU
        start_compute = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()  # Wait for computation to finish
        gpu_compute_time = time.time() - start_compute
        gpu_total_time = time.time() - start_time
        
        logger.info(f"GPU Compute Time: {gpu_compute_time:.4f} seconds")
        logger.info(f"GPU Total Time (including transfers): {gpu_total_time:.4f} seconds")
        logger.info(f"Speedup (compute only): {cpu_time/gpu_compute_time:.2f}x")
        
        # Clean up
        del a_cpu, b_cpu, c_cpu, a_gpu, b_gpu, c_gpu
        torch.cuda.empty_cache()

def check_llama_cpp_gpu_support():
    """Check if llama-cpp-python is built with GPU support"""
    logger.info("=== Checking llama-cpp-python GPU Support ===")
    
    try:
        from llama_cpp import Llama
        
        # Try to create a small model with GPU layers
        logger.info("Attempting to initialize Llama with GPU support...")
        
        # Get model path from environment or use default
        model_dir = os.path.join(Path(__file__).parent.parent, "models")
        model_name = os.environ.get("MODEL_NAME", "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf")
        model_path = os.path.join(model_dir, model_name)
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        logger.info(f"Using model: {model_path}")
        
        try:
            # Try to initialize with GPU support
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=1,  # Just test with 1 layer
                verbose=True
            )
            logger.info("✅ Successfully initialized Llama with GPU support")
            
            # Check if model is actually using GPU
            logger.info("Testing if model is using GPU...")
            start_time = time.time()
            output = llm.create_completion(
                prompt="Hello, world!",
                max_tokens=5,
                temperature=0.7,
            )
            elapsed = time.time() - start_time
            logger.info(f"Completion took {elapsed:.2f} seconds")
            logger.info(f"Output: {output['choices'][0]['text']}")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing Llama with GPU support: {e}")
            return False
    except ImportError:
        logger.error("llama-cpp-python is not installed")
        return False

def suggest_optimizations():
    """Suggest optimizations based on the system configuration"""
    logger.info("=== Optimization Suggestions ===")
    
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        
        # Memory-based suggestions
        total_memory = device.total_memory / 1024**3
        logger.info(f"GPU Memory: {total_memory:.2f} GB")
        
        if total_memory >= 20:  # High-end GPU like RTX 4090
            logger.info("High-memory GPU detected. Suggestions:")
            logger.info("1. Use n_gpu_layers=-1 to offload all layers to GPU")
            logger.info("2. Increase batch size (n_batch=512 or higher)")
            logger.info("3. Use larger context window (n_ctx=4096 or higher)")
            logger.info("4. Enable f16_kv=True for half-precision key/value cache")
        elif total_memory >= 10:  # Mid-range GPU
            logger.info("Mid-range GPU detected. Suggestions:")
            logger.info("1. Use n_gpu_layers=-1 but with lower batch size (n_batch=256)")
            logger.info("2. Use moderate context window (n_ctx=2048)")
            logger.info("3. Enable f16_kv=True for half-precision key/value cache")
        else:  # Lower-end GPU
            logger.info("Lower-memory GPU detected. Suggestions:")
            logger.info("1. Use partial GPU offloading (n_gpu_layers=20-30)")
            logger.info("2. Use smaller batch size (n_batch=128)")
            logger.info("3. Use smaller context window (n_ctx=1024)")
            logger.info("4. Consider using a smaller model or more aggressive quantization")
        
        # Compute capability suggestions
        compute_capability = f"{device.major}.{device.minor}"
        logger.info(f"Compute Capability: {compute_capability}")
        
        if float(compute_capability) >= 7.5:
            logger.info("Modern GPU architecture detected. Enable tensor cores with:")
            logger.info("1. Use f16_kv=True")
            logger.info("2. Consider using a model with Q5_K_M or Q6_K quantization")
        
        # Other optimizations
        logger.info("General optimizations:")
        logger.info("1. Set use_mmap=True and use_mlock=False")
        logger.info("2. Set embedding=False if embeddings are not needed")
        logger.info("3. Reduce last_n_tokens_size if full history is not needed")
        logger.info("4. Limit n_threads to avoid CPU bottlenecks")
    else:
        logger.info("No GPU detected. Consider:")
        logger.info("1. Installing CUDA drivers")
        logger.info("2. Using a CPU-optimized build of llama-cpp-python")
        logger.info("3. Using a smaller model with higher quantization")

def main():
    """Main function"""
    logger.info("=== GPU Optimization Script ===")
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    if gpu_available:
        # Test GPU performance
        test_gpu_performance()
    
    # Check llama-cpp-python GPU support
    llama_gpu_support = check_llama_cpp_gpu_support()
    
    # Suggest optimizations
    suggest_optimizations()
    
    # Summary
    logger.info("=== Summary ===")
    logger.info(f"GPU Available: {'Yes' if gpu_available else 'No'}")
    logger.info(f"llama-cpp-python GPU Support: {'Yes' if llama_gpu_support else 'No'}")
    
    if not gpu_available or not llama_gpu_support:
        logger.info("⚠️ GPU acceleration is not fully functional")
        logger.info("Please check the logs above for details and suggestions")
    else:
        logger.info("✅ GPU acceleration appears to be working")
        logger.info("Check the optimization suggestions for performance improvements")

if __name__ == "__main__":
    main()
