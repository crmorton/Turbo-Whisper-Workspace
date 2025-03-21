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
                
            # Get sample array in float32 and normalize
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            samples = samples / 32768.0  # Normalize to [-1, 1]
            
            return samples, audio_segment.frame_rate
        except Exception as e:
            print(f"Pydub failed: {e}, trying to convert format...")
        
        # Last resort: Convert to wav using pydub and save as temp file
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
                
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                audio_segment.export(temp_path, format="wav")
                
            # Now read it with wave
            with wave.open(temp_path) as f:
                num_samples = f.getnframes()
                samples = f.readframes(num_samples)
                samples_int16 = np.frombuffer(samples, dtype=np.int16)
                samples_float32 = samples_int16.astype(np.float32)
                samples_float32 = samples_float32 / 32768
                sr = f.getframerate()
                
            # Clean up temp file
            os.unlink(temp_path)
            return samples_float32, sr
        except Exception as e:
            raise RuntimeError(f"All methods to read audio file failed. Last error: {e}")
            
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
            repo_id="csukuangfj/sherpa-onnx-reverb-diarization-v1",
            filename="model.onnx",
        )


def get_speaker_embedding_model(model_name) -> str:
    assert (
        model_name
        in three_d_speaker_embedding_models
        + nemo_speaker_embedding_models
        + wespeaker_embedding_models
    )
    model_name = model_name.split("|")[0]

    return get_file(
        repo_id="csukuangfj/speaker-embedding-models",
        filename=model_name,
    )


def get_speaker_diarization(
    segmentation_model: str, embedding_model: str, num_clusters: int, threshold: float
):
    segmentation = get_speaker_segmentation_model(segmentation_model)
    embedding = get_speaker_embedding_model(embedding_model)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation
            ),
            debug=False,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding,
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
    "nemo_en_speakerverification_speakernet.onnx|22MB",
    "nemo_en_titanet_large.onnx|97MB",
    "nemo_en_titanet_small.onnx|38MB",
]

three_d_speaker_embedding_models = [
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx|37.8MB",
    "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx|28.2MB",
    "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx|27MB",
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx|27MB",
    "3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx|37.8MB",
    "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx|111MB",
    "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx|25.3MB",
    "3dspeaker_speech_eres2net_sv_zh-cn_16k-common.onnx|210MB",
    "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx|68.1MB",
]
wespeaker_embedding_models = [
    "wespeaker_en_voxceleb_CAM++.onnx|28MB",
    "wespeaker_en_voxceleb_CAM++_LM.onnx|28MB",
    "wespeaker_en_voxceleb_resnet152_LM.onnx|76MB",
    "wespeaker_en_voxceleb_resnet221_LM.onnx|91MB",
    "wespeaker_en_voxceleb_resnet293_LM.onnx|110MB",
    "wespeaker_en_voxceleb_resnet34.onnx|26MB",
    "wespeaker_en_voxceleb_resnet34_LM.onnx|26MB",
    "wespeaker_zh_cnceleb_resnet34.onnx|26MB",
    "wespeaker_zh_cnceleb_resnet34_LM.onnx|26MB",
]

embedding2models = {
    "3D-Speaker": three_d_speaker_embedding_models,
    "NeMo": nemo_speaker_embedding_models,
    "WeSpeaker": wespeaker_embedding_models,
}
