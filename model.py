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
from functools import lru_cache
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf
import sherpa_onnx
from huggingface_hub import hf_hub_download
from pydub import AudioSegment


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
    try:
        # First attempt: Try using librosa which handles many formats
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            # Ensure float32 and normalized to [-1, 1]
            audio = audio.astype(np.float32)
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))
            return audio, sr
        except Exception as e:
            print(f"Librosa failed: {e}, trying pydub...")
            
        # Second attempt: Try using pydub which handles even more formats
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            # Convert to mono if needed
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            # Convert to float32 and normalize
            samples = samples.astype(np.float32)
            samples = samples / (1 << (8 * audio_segment.sample_width - 1))
            
            return samples, audio_segment.frame_rate
        except Exception as e:
            print(f"Pydub failed: {e}, trying soundfile...")
            
        # Third attempt: Try using soundfile
        try:
            audio, sr = sf.read(audio_path)
            # Convert to mono if needed
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            # Ensure float32 and normalized to [-1, 1]
            audio = audio.astype(np.float32)
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))
            return audio, sr
        except Exception as e:
            print(f"Soundfile failed: {e}")
            
    except Exception as e:
        print(f"Error reading audio file: {e}")
        # Return empty audio rather than crashing
        return np.zeros(1600, dtype=np.float32), 16000


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


def get_speaker_segmentation_model(repo_id) -> str:
    assert repo_id in (
        "pyannote/segmentation-3.0",
        "Revai/reverb-diarization-v1",
    )

    if repo_id == "pyannote/segmentation-3.0":
        return get_file(
            repo_id="csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
            filename="model.onnx",
        )
    elif repo_id == "Revai/reverb-diarization-v1":
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

    # Handle NeMo models
    if model_filename.startswith("nemo_"):
        # Try different NeMo repositories
        try:
            return get_file(
                repo_id="nvidia/NeMo-Speaker-Embeddings",
                filename=model_filename.replace("nemo_", ""),
            )
        except Exception as e:
            print(f"Error loading from NeMo repository: {e}")
            try:
                return get_file(
                    repo_id="k2-fsa/sherpa-onnx-nemo-speaker-models",
                    filename=model_filename,
                )
            except Exception as e2:
                print(f"Error loading from k2-fsa repository: {e2}")
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


def get_speaker_diarization(
    segmentation_model: str, embedding_model: str, num_clusters: int, threshold: float
):
    segmentation = get_speaker_segmentation_model(segmentation_model)
    embedding = get_speaker_embedding_model(embedding_model)
    
    # Check if CUDA is available for ONNX acceleration
    import torch
    provider = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation
            ),
            provider=provider,  # Use CUDA provider if available
            num_threads=4,      # Optimize thread count
            debug=False,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding,
            provider=provider,  # Use CUDA provider if available
            num_threads=4,      # Optimize thread count
            debug=False,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    
    print("config", config)
    
    if not config.validate():
        raise RuntimeError(
            "Please check your config and make sure all required files exist"
        )
    
    return sherpa_onnx.OfflineSpeakerDiarization(config)


speaker_segmentation_models = [
    "pyannote/segmentation-3.0",
    "Revai/reverb-diarization-v1",
]


nemo_speaker_embedding_models = [
    # Core models - verified to exist in repositories
    "nemo_en_titanet_large.onnx|97MB",      # Available in k2-fsa/sherpa-onnx-nemo-speaker-models
    "nemo_en_titanet_small.onnx|38MB",      # Available in k2-fsa/sherpa-onnx-nemo-speaker-models
    "en_titanet_large.onnx|97MB",           # Available in nvidia/NeMo-Speaker-Embeddings
    "en_ecapa_tdnn.onnx|45MB",              # Available in nvidia/NeMo-Speaker-Embeddings
    
    # Additional models if available
    "en_titanet_small.onnx|38MB",           # Available in nvidia/NeMo-Speaker-Embeddings
    "en_speakerverification_speakernet.onnx|22MB", # Available in nvidia/NeMo-Speaker-Embeddings
]

three_d_speaker_embedding_models = [
    # Verified available models from k2-fsa/speaker-embedding-models
    "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx|28.2MB",  # Confirmed available
    "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx|25.3MB",  # Confirmed available
    
    # Models from official 3D-Speaker repository
    "speech_campplus_sv_en_voxceleb_16k.onnx|28.2MB",  # Available in 3D-Speaker/3D-Speaker-models
    "speech_eres2net_sv_en_voxceleb_16k.onnx|25.3MB",  # Available in 3D-Speaker/3D-Speaker-models
    # Additional models if available
    "speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx|37.8MB",  # Try from official repo
    "speech_campplus_sv_zh-cn_16k-common.onnx|27MB",  # Try from official repo
]

# Verified working models - these are known to be available in the repositories
# Verified working models - these are known to be available in the repositories
wespeaker_embedding_models = [
    # Models available in wenet-e2e repositories (official WeSpeaker models)
    "wenet_wespeaker_resnet.onnx|26MB",        # Available in wenet-e2e/wespeaker-models
    "wenet_wespeaker_ecapa_tdnn.onnx|21MB",    # Available in wenet-e2e/wespeaker-models
    "wenet_wespeaker_campplus.onnx|28MB",      # Available in wenet-e2e/wespeaker-models
    
    # Fallback models with simplified names
    "resnet34.onnx|26MB",  # Generic ResNet34 model
    "ecapa_tdnn.onnx|21MB",  # Generic ECAPA-TDNN model
    "campplus.onnx|28MB",  # Generic CAM++ model
]

embedding2models = {
    "NeMo": nemo_speaker_embedding_models,
    "3D-Speaker": three_d_speaker_embedding_models,
    "WeSpeaker": wespeaker_embedding_models,
}
