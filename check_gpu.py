import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device count:", torch.cuda.device_count())
    print("GPU device name:", torch.cuda.get_device_name(0))
    
    # Get GPU memory information
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_memory_mb = total_memory / (1024 * 1024)
    print(f"Total GPU memory: {total_memory_mb:.2f} MB")
    
    # Check if TF32 is enabled
    print("TF32 enabled:", torch.backends.cuda.matmul.allow_tf32)
    
    # Test GPU with a simple operation
    print("\nRunning a simple GPU test...")
    a = torch.randn(10000, 10000, device="cuda")
    b = torch.randn(10000, 10000, device="cuda")
    
    # Measure time for matrix multiplication
    import time
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Matrix multiplication time: {end - start:.4f} seconds")
else:
    print("CUDA not available - running on CPU only")