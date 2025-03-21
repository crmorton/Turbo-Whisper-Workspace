#!/usr/bin/env python3
"""
Test script to verify that local models are being used correctly.
"""

import os
import sys
from pprint import pprint
import numpy as np

from model import get_speaker_diarization, get_speaker_embedding_model, get_speaker_segmentation_model

def main():
    print("Testing local model detection...")
    
    # Test speaker segmentation models
    print("\n=== Testing Speaker Segmentation Models ===")
    segmentation_models = [
        "pyannote/segmentation-3.0",
        "Revai/reverb-diarization-v1"
    ]
    
    for model_name in segmentation_models:
        try:
            model_path = get_speaker_segmentation_model(model_name)
            exists = os.path.exists(model_path)
            print(f"Model: {model_name}")
            print(f"  Path: {model_path}")
            print(f"  Exists: {exists}")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    # Test speaker embedding models
    print("\n=== Testing Speaker Embedding Models ===")
    embedding_models = [
        "nemo_en_titanet_small.onnx|38MB",
        "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx|25MB"
    ]
    
    for model_name in embedding_models:
        try:
            model_path = get_speaker_embedding_model(model_name)
            exists = os.path.exists(model_path)
            print(f"Model: {model_name}")
            print(f"  Path: {model_path}")
            print(f"  Exists: {exists}")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    # Test creating a diarization object
    print("\n=== Testing Diarization Object Creation ===")
    try:
        diarizer = get_speaker_diarization(
            segmentation_model="pyannote/segmentation-3.0",
            embedding_model="nemo_en_titanet_small.onnx|38MB",
            num_clusters=2,
            threshold=0.5
        )
        print("Successfully created diarization object")
        print(f"Diarization object: {diarizer}")
    except Exception as e:
        print(f"Error creating diarization object: {e}")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()