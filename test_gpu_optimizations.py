import torch
import time
import sys
sys.path.append('.')
from app import setup_gpu, get_gpu_memory_info, check_gpu_efficiency

print("===== TESTING GPU OPTIMIZATIONS =====")

# Check initial GPU state
print("\n1. Initial GPU State:")
print("CUDA available:", torch.cuda.is_available())
print("TF32 enabled (before setup):", torch.backends.cuda.matmul.allow_tf32)

# Run our setup_gpu function
print("\n2. Running setup_gpu():")
gpu_available = setup_gpu()
print("GPU setup successful:", gpu_available)

# Check if TF32 is now enabled
print("\n3. GPU Configuration After Setup:")
print("TF32 enabled (after setup):", torch.backends.cuda.matmul.allow_tf32)

# Check GPU memory info
print("\n4. GPU Memory Information:")
memory_info = get_gpu_memory_info()
print(memory_info)

# Check GPU efficiency
print("\n5. GPU Efficiency Check:")
efficiency_info = check_gpu_efficiency()
print(efficiency_info)

# Test performance with and without TF32
print("\n6. Performance Test:")

# Create large matrices for multiplication
size = 5000
a = torch.randn(size, size, device="cuda" if gpu_available else "cpu")
b = torch.randn(size, size, device="cuda" if gpu_available else "cpu")

# Test with TF32 enabled (our setup)
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.synchronize()
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
tf32_time = time.time() - start
print(f"Matrix multiplication with TF32 enabled: {tf32_time:.4f} seconds")

# Test with TF32 disabled
torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.synchronize()
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
no_tf32_time = time.time() - start
print(f"Matrix multiplication with TF32 disabled: {no_tf32_time:.4f} seconds")

# Calculate speedup
speedup = no_tf32_time / tf32_time if tf32_time > 0 else 0
print(f"Speedup with TF32 enabled: {speedup:.2f}x")

print("\n===== GPU OPTIMIZATION TEST COMPLETE =====")