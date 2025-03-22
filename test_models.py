"""
Test script to check if the models can be loaded correctly
"""

import os
import sys
import torch
import numpy as np

def check_gpu():
    """Check if GPU is available and print GPU information"""
    print("\n=== GPU Information ===")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print("CUDA Available: No")
    print("=====================\n")

def test_sherpa_onnx_model(model_path):
    """Test if sherpa-onnx model can be loaded"""
    print(f"\n=== Testing Sherpa-ONNX Model: {model_path} ===")
    
    try:
        # Check if the file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return False
            
        print(f"Model file exists: {model_path}")
        
        # Try to import sherpa-onnx
        try:
            import sherpa_onnx
            print("Successfully imported sherpa_onnx")
        except ImportError as e:
            print(f"Error importing sherpa_onnx: {e}")
            return False
            
        # Check if it's a tar.bz2 file
        if model_path.endswith('.tar.bz2'):
            print("Model is a tar.bz2 file, checking if it can be extracted")
            import tarfile
            try:
                with tarfile.open(model_path, 'r:bz2') as tar:
                    # Just list the contents without extracting
                    file_list = tar.getnames()
                    print(f"Archive contains {len(file_list)} files")
                    for i, file in enumerate(file_list[:5]):  # Show first 5 files
                        print(f"  {i+1}. {file}")
                    if len(file_list) > 5:
                        print(f"  ... and {len(file_list) - 5} more files")
            except Exception as e:
                print(f"Error opening tar.bz2 file: {e}")
                return False
                
        # Try to load the model using sherpa-onnx
        try:
            # For segmentation model
            config = sherpa_onnx.OfflineSpeakerDiarizationConfig()
            config.segmentation.pyannote.model = model_path
            
            # Try to initialize the diarizer
            diarizer = sherpa_onnx.OfflineSpeakerDiarizer(config)
            print("Successfully initialized the diarizer with the model")
            return True
        except Exception as e:
            print(f"Error loading model with sherpa_onnx: {e}")
            print(f"Error type: {type(e).__name__}")
            return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        print("=====================\n")

def test_nemo_onnx_model(model_path):
    """Test if NeMo ONNX model can be loaded"""
    print(f"\n=== Testing NeMo ONNX Model: {model_path} ===")
    
    try:
        # Check if the file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return False
            
        print(f"Model file exists: {model_path}")
        
        # Try to import onnxruntime
        try:
            import onnxruntime as ort
            print("Successfully imported onnxruntime")
            print(f"Available providers: {ort.get_available_providers()}")
            
            # Check if CUDA provider is available
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                print("CUDA provider is available")
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print("CUDA provider is not available, falling back to CPU")
                providers = ['CPUExecutionProvider']
                
            # Try to create an inference session
            try:
                session = ort.InferenceSession(model_path, providers=providers)
                print("Successfully created inference session")
                
                # Get model metadata
                input_name = session.get_inputs()[0].name
                input_shape = session.get_inputs()[0].shape
                print(f"Model input name: {input_name}")
                print(f"Model input shape: {input_shape}")
                
                # Try a dummy inference
                dummy_input = np.random.randn(*input_shape).astype(np.float32)
                outputs = session.run(None, {input_name: dummy_input})
                print(f"Successfully ran inference. Output shapes: {[o.shape for o in outputs]}")
                return True
            except Exception as e:
                print(f"Error creating inference session or running inference: {e}")
                return False
        except ImportError as e:
            print(f"Error importing onnxruntime: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        print("=====================\n")

if __name__ == "__main__":
    # Check GPU
    check_gpu()
    
    # Test the models
    sherpa_model_path = "models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    nemo_model_path = "models/nemo_en_titanet_small.onnx"
    
    sherpa_result = test_sherpa_onnx_model(sherpa_model_path)
    nemo_result = test_nemo_onnx_model(nemo_model_path)
    
    print("\n=== Summary ===")
    print(f"Sherpa-ONNX model test: {'PASSED' if sherpa_result else 'FAILED'}")
    print(f"NeMo ONNX model test: {'PASSED' if nemo_result else 'FAILED'}")
    print("===============\n")