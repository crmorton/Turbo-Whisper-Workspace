"""
Audio information extraction utilities for CyberVox Audio Workspace
"""

import numpy as np
import librosa
from pydub import AudioSegment

def get_audio_info(audio_path):
    """Get audio file information
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    try:
        # Get basic info using pydub
        audio_segment = AudioSegment.from_file(audio_path)
        duration = len(audio_segment) / 1000  # in seconds
        channels = audio_segment.channels
        sample_width = audio_segment.sample_width
        frame_rate = audio_segment.frame_rate
        bitrate = frame_rate * sample_width * 8 * channels
        
        # Get more detailed info using librosa
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(y**2))
        
        # Calculate zero crossing rate
        zero_crossing_rate = np.sum(np.abs(np.diff(np.signbit(y)))) / len(y)
        
        # Calculate spectral features
        spectral_features = {}
        try:
            # Compute spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
            spectral_features["spectral_centroid"] = float(spectral_centroid)
            
            # Compute spectral bandwidth (width of the spectrum)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
            spectral_features["spectral_bandwidth"] = float(spectral_bandwidth)
            
            # Compute spectral rolloff (frequency below which 85% of the energy is contained)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
            spectral_features["spectral_rolloff"] = float(spectral_rolloff)
            
            # Compute spectral contrast (valleys and peaks in spectrum)
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            spectral_features["spectral_contrast"] = float(spectral_contrast)
        except Exception as e:
            print(f"Error computing spectral features: {e}")
        
        # Return combined info
        info = {
            "duration": duration,
            "channels": channels,
            "sample_width": sample_width,
            "frame_rate": frame_rate,
            "bitrate": bitrate,
            "rms": float(rms),
            "zero_crossing_rate": float(zero_crossing_rate),
            "format": "PCM" if sample_width == 2 else f"PCM-{sample_width*8}",
        }
        
        # Add spectral features
        info.update(spectral_features)
        
        return info
    except Exception as e:
        print(f"Error getting audio info: {e}")
        return {
            "error": str(e)
        }
