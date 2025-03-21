#!/usr/bin/env python3
"""
Download speaker embedding models to local directory.
This script downloads verified working models for CyberVox from GitHub releases
and stores them locally to avoid dependency on remote repositories.
"""

import os
import sys
import argparse
import requests
import shutil
from tqdm import tqdm
from dotenv import load_dotenv

def download_file(url, local_path):
    """Download a file from a URL to a local path with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with progress bar
        with open(local_path, 'wb') as f, tqdm(
            desc=os.path.basename(local_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"  ❌ Failed to download {url}: {str(e)}")
        return False

def download_models_to_local(models_dir="models", force_download=False):
    """Download speaker embedding models to local directory.
    
    Args:
        models_dir: Directory to save models to
        force_download: Whether to force download even if file exists
        
    Returns:
        Dictionary mapping model names to local file paths
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    # Dictionary to store model paths
    local_models = {}
    
    # List of models to download with their direct URLs from GitHub releases
    models_to_download = [
        # Speaker segmentation models
        {
            "name": "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
        },
        {
            "name": "sherpa-onnx-reverb-diarization-v1.tar.bz2",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-reverb-diarization-v1.tar.bz2"
        },
        
        # Speaker embedding models
        {
            "name": "nemo_en_titanet_small.onnx",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx"
        },
        {
            "name": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
        },
        
        # ASR models (optional - these are large files)
        {
            "name": "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"
        },
        {
            "name": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2",
            "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
        }
    ]
    
    # Download each model
    for model in models_to_download:
        local_path = os.path.join(models_dir, model["name"])
        
        # Skip if file exists and we're not forcing download
        if os.path.exists(local_path) and not force_download:
            print(f"Skipping {model['name']} - already exists")
            local_models[model["name"]] = local_path
            continue
        
        print(f"Downloading {model['name']} from {model['url']}...")
        if download_file(model["url"], local_path):
            local_models[model["name"]] = local_path
            print(f"✅ Successfully downloaded {model['name']}")
        else:
            print(f"⚠️ Could not download {model['name']}")
    
    return local_models

def main():
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download speaker embedding models for CyberVox")
    parser.add_argument("--force", action="store_true", help="Force download even if models already exist")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to store downloaded models")
    args = parser.parse_args()
    
    # Check if models directory exists, create if not
    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir, exist_ok=True)
    
    print(f"📂 Downloading models to {os.path.abspath(args.models_dir)}")
    print(f"🔄 Force download: {args.force}")
    
    # Download models
    models = download_models_to_local(
        models_dir=args.models_dir, 
        force_download=args.force
    )
    
    # Print summary
    print("\n📊 Download Summary:")
    print(f"Downloaded {len(models)} models:")
    for name, path in models.items():
        print(f"  - {name}: {path}")
    
    print("\n🎉 To use these models, simply select them in the CyberVox interface.")
    print("   The application will automatically use the local versions instead of downloading them again.")

if __name__ == "__main__":
    main()
