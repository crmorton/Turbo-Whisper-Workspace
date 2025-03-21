#!/usr/bin/env python3
"""
Simple script to check if local models exist and can be found.
"""

import os
import sys

def main():
    print("Checking for local models...")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' does not exist")
        return
    
    # List of models we expect to find
    expected_models = [
        "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "sherpa-onnx-reverb-diarization-v1.tar.bz2",
        "nemo_en_titanet_small.onnx",
        "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
        "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
    ]
    
    # Check each model
    found_models = []
    missing_models = []
    
    for model in expected_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            found_models.append(model)
            print(f"✅ Found: {model}")
        else:
            missing_models.append(model)
            print(f"❌ Missing: {model}")
    
    # Summary
    print("\nSummary:")
    print(f"Found {len(found_models)} of {len(expected_models)} models")
    
    if missing_models:
        print("\nMissing models:")
        for model in missing_models:
            print(f"  - {model}")
    else:
        print("\nAll models found successfully!")

if __name__ == "__main__":
    main()