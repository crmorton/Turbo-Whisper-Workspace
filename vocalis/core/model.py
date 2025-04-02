# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import wave
import os
import tempfile
import time
from functools import lru_cache
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf
import sherpa_onnx
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

# Import our audio utilities
from vocalis.core.audio_utils import read_audio_file

# Alias for backward compatibility
def read_wave(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Read audio file in any format and convert to mono numpy array
    
    Args:
      audio_path:
        Path to an audio file. Can be any format supported by librosa/pydub.
        
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the audio file
    """
    return read_audio_file(audio_path)


@lru_cache(maxsize=30)
def get_file(
    repo_id: str,
    filename: str,
    subfolder: str = ".",
) -> str:
    nn_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return nn_model_filename


def download_models_to_local(models_dir="models", force_download=False, use_auth_token=None):
    """Download speaker embedding models to local directory.
    
    Args:
        models_dir: Directory to save models to
        force_download: Whether to force download even if file exists
        use_auth_token: Optional Hugging Face token for accessing gated models
        
    Returns:
        Dictionary mapping model names to local file paths
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    
    # Dictionary to store model paths
    local_models = {}
    
    # Import shutil here to avoid potential import issues
    import shutil
    
    # List of models to download with their sources and fallbacks
    models_to_download = [
        # NeMo models with fallbacks
        {
            "name": "nemo_en_titanet_small.onnx", 
            "sources": [
                {"repo": "csukuangfj/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_titanet_small.onnx"},
                {"repo": "k2-fsa/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_titanet_small.onnx"},
            ]
        },
        {
            "name": "nemo_en_titanet_large.onnx", 
            "sources": [
                {"repo": "nvidia/speakerverification_en_titanet_large", "file": "model.onnx"},
                {"repo": "csukuangfj/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_titanet_large.onnx"},
                {"repo": "k2-fsa/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_titanet_large.onnx"},
            ]
        },
        {
            "name": "nemo_en_ecapa_tdnn.onnx", 
            "sources": [
                {"repo": "nvidia/speakerverification_en_ecapa_tdnn", "file": "model.onnx"},
                {"repo": "csukuangfj/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_ecapa_tdnn.onnx"},
                {"repo": "k2-fsa/sherpa-onnx-nemo-speaker-models", "file": "nemo_en_ecapa_tdnn.onnx"},
            ]
        },
        
        # 3D-Speaker models with fallbacks
        {
            "name": "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx", 
            "sources": [
                {"repo": "k2-fsa/speaker-embedding-models", "file": "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx"},
                {"repo": "3D-Speaker/3D-Speaker-models", "file": "speech_eres2net_sv_en_voxceleb_16k.onnx"},
            ]
        },
        {
            "name": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx", 
            "sources": [
                {"repo": "k2-fsa/speaker-embedding-models", "file": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"},
                {"repo": "3D-Speaker/3D-Speaker-models", "file": "speech_campplus_sv_en_voxceleb_16k.onnx"},
            ]
        },
        
        # WeSpeaker models with fallbacks
        {
            "name": "wenet_wespeaker_resnet.onnx", 
            "sources": [
                {"repo": "wenet-e2e/wespeaker-models", "file": "resnet/model.onnx"},
                {"repo": "k2-fsa/wespeaker-models", "file": "wenet_wespeaker_resnet.onnx"},
                {"repo": "csukuangfj/wespeaker-models", "file": "wenet_wespeaker_resnet.onnx"},
            ]
        },
        {
            "name": "wenet_wespeaker_ecapa_tdnn.onnx", 
            "sources": [
                {"repo": "wenet-e2e/wespeaker-models", "file": "ecapa-tdnn/model.onnx"},
                {"repo": "k2-fsa/wespeaker-models", "file": "wenet_wespeaker_ecapa_tdnn.onnx"},
                {"repo": "csukuangfj/wespeaker-models", "file": "wenet_wespeaker_ecapa_tdnn.onnx"},
            ]
        },
        {
            "name": "wenet_wespeaker_campplus.onnx", 
            "sources": [
                {"repo": "wenet-e2e/wespeaker-models", "file": "campplus/model.onnx"},
                {"repo": "k2-fsa/wespeaker-models", "file": "wenet_wespeaker_campplus.onnx"},
                {"repo": "csukuangfj/wespeaker-models", "file": "wenet_wespeaker_campplus.onnx"},
            ]
        },
    ]
    
    # Download each model
    for model in models_to_download:
        local_path = os.path.join(models_dir, model["name"])
        
        # Skip if file exists and we're not forcing download
        if os.path.exists(local_path) and not force_download:
            print(f"Skipping {model['name']} - already exists")
            local_models[model["name"]] = local_path
            continue
        
        # Try each source until one works
        downloaded = False
        for source in model["sources"]:
            if downloaded:
                break
                
            print(f"Trying to download {model['name']} from {source['repo']}...")
            try:
                # Download the model
                remote_path = hf_hub_download(
                    repo_id=source["repo"],
                    filename=source["file"],
                    token=use_auth_token,
                )
                
                # Copy to local models directory
                shutil.copy(remote_path, local_path)
                local_models[model["name"]] = local_path
                print(f"✅ Successfully downloaded {model['name']} from {source['repo']}")
                downloaded = True
            except Exception as e:
                print(f"  ❌ Failed from {source['repo']}: {str(e).split('\n')[0]}")
        
        if not downloaded:
            print(f"⚠️ Could not download {model['name']} from any source")
    
    return local_models

def get_speaker_segmentation_model(repo_id) -> str:
    assert repo_id in (
        "pyannote/segmentation-3.0",
        "Revai/reverb-diarization-v1",
    )

    # Check for local models first
    if repo_id == "pyannote/segmentation-3.0":
        local_path = os.path.join("models", "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
        
        # Fall back to remote model
        return get_file(
            repo_id="csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
            filename="model.onnx",
        )
    elif repo_id == "Revai/reverb-diarization-v1":
        local_path = os.path.join("models", "sherpa-onnx-reverb-diarization-v1.tar.bz2")
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
        
        # Fall back to remote model
        return get_file(
            repo_id="csukuangfj/sherpa-onnx-revai-reverb-diarization-v1",
            filename="model.onnx",
        )


def get_speaker_embedding_model(model_name: str) -> str:
    """Get speaker embedding model path.

    Args:
      model_name:
        Format: model_filename|model_size
        Example: nemo_en_titanet_large.onnx|97MB
    """
    parts = model_name.split("|")
    model_filename = parts[0]
    
    # First check if the model exists in the local models directory
    local_model_path = os.path.join("models", model_filename)
    if os.path.exists(local_model_path):
        print(f"Using local model: {local_model_path}")
        return local_model_path
    
    # Check for alternative local models that might work
    if model_filename == "nemo_en_titanet_small.onnx":
        local_path = os.path.join("models", "nemo_en_titanet_small.onnx")
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
    elif "3dspeaker" in model_filename:
        local_path = os.path.join("models", "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
    
    print(f"Local model not found at {local_model_path}, trying to download...")

    # Handle NeMo models
    if model_filename.startswith("nemo_"):
        # Map model filenames to the correct Hugging Face model IDs
        model_mapping = {
            "nemo_en_titanet_large.onnx": "nvidia/speakerverification_en_titanet_large",
            "nemo_en_titanet_small.onnx": "nvidia/speakerverification_en_titanet_small",
            "nemo_en_ecapa_tdnn.onnx": "nvidia/speakerverification_en_ecapa_tdnn",
        }
        
        # Try to find the model in our mapping
        if model_filename in model_mapping:
            try:
                print(f"Loading NeMo model from Hugging Face: {model_mapping[model_filename]}")
                return get_file(
                    repo_id=model_mapping[model_filename],
                    filename="model.onnx",  # NeMo models on HF use model.onnx as the filename
                )
            except Exception as e:
                print(f"Error loading from official NeMo HF repository: {e}")
        
        # Try alternative repositories if not found or if loading failed
        try:
            # Try csukuangfj's repository which has converted NeMo models
            return get_file(
                repo_id="csukuangfj/sherpa-onnx-nemo-speaker-models",
                filename=model_filename,
            )
        except Exception as e2:
            print(f"Error loading from csukuangfj repository: {e2}")
            try:
                # Try k2-fsa repository as another fallback
                return get_file(
                    repo_id="k2-fsa/sherpa-onnx-nemo-speaker-models",
                    filename=model_filename,
                )
            except Exception as e3:
                print(f"Error loading from k2-fsa repository: {e3}")
                # Final fallback
                return get_file(
                    repo_id="csukuangfj/sherpa-onnx-nemo-speaker-models",
                    filename=model_filename,
                )
    # Handle 3D-Speaker models
    elif model_filename.startswith("3dspeaker_"):
        try:
            # Try the official 3D-Speaker repo first
            return get_file(
                repo_id="3D-Speaker/3D-Speaker-models",
                filename=model_filename.replace("3dspeaker_", ""),
            )
        except Exception as e:
            print(f"Error loading from 3D-Speaker repository: {e}")
            # Fallback to the sherpa-onnx repository
            return get_file(
                repo_id="k2-fsa/speaker-embedding-models",
                filename=model_filename,
            )
    # Handle WeSpeaker models with different naming conventions
    elif model_filename.startswith("wespeaker_"):
        # Clean up model name for proper URL encoding
        clean_filename = model_filename.replace("+", "plus")
        
        # Check if it's a standard model or one with LM enhancement
        if "_LM" in model_filename:
            # For LM-enhanced models, try multiple repositories
            try:
                # Try the official WeSpeaker repo first
                model_type = "base"
                if "resnet" in model_filename.lower():
                    model_type = "resnet"
                elif "ecapa" in model_filename.lower():
                    model_type = "ecapa-tdnn"
                elif "camplus" in model_filename.lower() or "cam++" in model_filename.lower():
                    model_type = "campplus"
                elif "gemini" in model_filename.lower():
                    model_type = "gemini"
                
                return get_file(
                    repo_id=f"wenet-e2e/wespeaker-models-v2",
                    filename=f"{model_type}/model.onnx",
                )
            except Exception as e:
                print(f"Error loading from WeSpeaker official repo: {e}")
                # Fallback to k2-fsa repo
                try:
                    return get_file(
                        repo_id="k2-fsa/wespeaker-models",
                        filename=clean_filename,
                    )
                except Exception as e2:
                    print(f"Error loading from k2-fsa repository: {e2}")
                    # Last resort - try csukuangfj repo
                    return get_file(
                        repo_id="csukuangfj/wespeaker-models",
                        filename=clean_filename,
                    )
        else:
            # For standard models, try multiple repositories
            try:
                # Try wenet-e2e repo first
                model_type = "base"
                if "resnet" in model_filename.lower():
                    model_type = "resnet"
                elif "ecapa" in model_filename.lower():
                    model_type = "ecapa-tdnn"
                elif "camplus" in model_filename.lower() or "cam++" in model_filename.lower():
                    model_type = "campplus"
                
                return get_file(
                    repo_id="wenet-e2e/wespeaker-models",
                    filename=f"{model_type}/model.onnx",
                )
            except Exception as e:
                print(f"Error loading from WeSpeaker official repo: {e}")
                # Try k2-fsa repo
                try:
                    return get_file(
                        repo_id="k2-fsa/wespeaker-models",
                        filename=clean_filename,
                    )
                except Exception as e2:
                    print(f"Error loading from k2-fsa repository: {e2}")
                    # Last resort
                    return get_file(
                        repo_id="csukuangfj/wespeaker-models",
                        filename=clean_filename,
                    )
    else:
        # For unknown model types, try to infer the right repository
        print(f"Unknown model type: {model_name}, attempting to infer repository")
        try:
            # Try to infer model type from filename
            if "resnet" in model_filename.lower():
                try:
                    return get_file(
                        repo_id="wenet-e2e/wespeaker-models",
                        filename="resnet/model.onnx",
                    )
                except Exception:
                    pass
            elif "ecapa" in model_filename.lower():
                try:
                    return get_file(
                        repo_id="wenet-e2e/wespeaker-models",
                        filename="ecapa-tdnn/model.onnx",
                    )
                except Exception:
                    pass
            elif "camp" in model_filename.lower():
                try:
                    return get_file(
                        repo_id="wenet-e2e/wespeaker-models",
                        filename="campplus/model.onnx",
                    )
                except Exception:
                    pass
                    
            # If all else fails, use a known working model as final fallback
            print(f"Using default fallback model for: {model_name}")
            return get_file(
                repo_id="csukuangfj/sherpa-onnx-nemo-speaker-models",
                filename="nemo_en_titanet_small.onnx",
            )
        except Exception as e:
            print(f"All attempts to load model failed: {e}")
            print("Using absolute last resort model")
            # Absolute last resort - use a model we know exists
            return get_file(
                repo_id="csukuangfj/sherpa-onnx-nemo-speaker-models",
                filename="nemo_en_titanet_small.onnx",
            )


# Cache for diarization models
_DIARIZATION_CACHE = {}

def get_speaker_diarization(
    segmentation_model: str, embedding_model: str, num_clusters: int, threshold: float, use_gpu: bool = True
):
    # Create a cache key based on the input parameters
    cache_key = f"{segmentation_model}_{embedding_model}_{num_clusters}_{threshold}"
    
    # Check if this configuration is already cached
    if cache_key in _DIARIZATION_CACHE:
        print(f"Using cached diarization model for {cache_key}")
        return _DIARIZATION_CACHE[cache_key]
    
    segmentation = get_speaker_segmentation_model(segmentation_model)
    embedding = get_speaker_embedding_model(embedding_model)
    
    # Check if CUDA is available for ONNX acceleration
    import torch
    # Only use CUDA if it's available AND use_gpu is True
    provider = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
    
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation
            ),
            provider=provider,  # Use CUDA provider if available
            num_threads=4,      # Optimize thread count
        ),
        embedding=sherpa_onnx.OfflineSpeakerEmbeddingModelConfig(
            model=embedding,
            provider=provider,  # Use CUDA provider if available
            num_threads=4,      # Optimize thread count
        ),
        clustering=sherpa_onnx.OfflineSpeakerClusteringConfig(
            num_clusters=num_clusters,
            threshold=threshold,
        ),
    )
    
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    
    # Cache the diarizer for future use
    _DIARIZATION_CACHE[cache_key] = diarizer
    
    return diarizer


# Define model lists for UI
speaker_segmentation_models = [
    "pyannote/segmentation-3.0",
    "Revai/reverb-diarization-v1",
]

embedding2models = {
    "NeMo": [
        "nemo_en_titanet_large.onnx|97MB",
        "nemo_en_titanet_small.onnx|25MB",
        "nemo_en_ecapa_tdnn.onnx|81MB",
    ],
    "3D-Speaker": [
        "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx|25.3MB",
        "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx|14.1MB",
    ],
    "WeSpeaker": [
        "wenet_wespeaker_resnet.onnx|89MB",
        "wenet_wespeaker_ecapa_tdnn.onnx|81MB",
        "wenet_wespeaker_campplus.onnx|14MB",
    ],
}

def get_local_segmentation_models():
    """Get list of locally available segmentation models"""
    local_models = []
    
    # Check for local models
    for model_id in speaker_segmentation_models:
        if model_id == "pyannote/segmentation-3.0":
            local_path = os.path.join("models", "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
            if os.path.exists(local_path):
                local_models.append(model_id)
        elif model_id == "Revai/reverb-diarization-v1":
            local_path = os.path.join("models", "sherpa-onnx-reverb-diarization-v1.tar.bz2")
            if os.path.exists(local_path):
                local_models.append(model_id)
    
    # If no local models found, return the default list
    if not local_models:
        return speaker_segmentation_models
        
    return local_models

def get_local_embedding_models():
    """Get dictionary of locally available embedding models by type"""
    local_models = {}
    
    # Check for local models in each category
    for model_type, models in embedding2models.items():
        local_type_models = []
        for model_name in models:
            model_filename = model_name.split("|")[0]
            local_path = os.path.join("models", model_filename)
            if os.path.exists(local_path):
                local_type_models.append(model_name)
        
        # Only add the type if we found local models
        if local_type_models:
            local_models[model_type] = local_type_models
    
    return local_models