"""
Audio Utilities for Vocalis

This module provides utility functions for audio processing, including:
- Audio file loading with robust error handling
- Audio format conversion
- Audio normalization
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from typing import Tuple, Optional

def read_audio_file(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Read audio file in any format and convert to mono numpy array
    
    Args:
        audio_path: Path to an audio file. Can be any format supported by librosa/pydub.
        
    Returns:
        Tuple containing:
        - A 1-D array of dtype np.float32 containing the samples, normalized to [-1, 1]
        - Sample rate of the audio file
    """
    try:
        # First attempt: Try using soundfile which is faster and more reliable for WAV/FLAC
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
            print(f"Soundfile failed: {e}, trying librosa...")
            
        # Second attempt: Try using librosa's recommended approach
        try:
            # Use load instead of deprecated __audioread_load
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            # Ensure float32 and normalized to [-1, 1]
            audio = audio.astype(np.float32)
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))
            return audio, sr
        except Exception as e:
            print(f"Librosa failed: {e}, trying pydub...")
            
        # Third attempt: Try using pydub which handles even more formats
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
            print(f"Pydub failed: {e}")
            
    except Exception as e:
        print(f"Error reading audio file: {e}")
        # Return empty audio rather than crashing
        return np.zeros(1600, dtype=np.float32), 16000

def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        # Try using librosa which is more reliable for duration
        return librosa.get_duration(path=audio_path)
    except Exception:
        try:
            # Try using pydub as fallback
            audio = AudioSegment.from_file(audio_path)
            return audio.duration_seconds
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 0.0

def convert_audio_format(input_path: str, output_path: str, format: str = "wav", 
                        sample_rate: int = 16000, channels: int = 1) -> bool:
    """
    Convert audio file to specified format
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        format: Output format (wav, flac, mp3, etc.)
        sample_rate: Output sample rate
        channels: Output number of channels
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        
        # Convert to mono if needed
        if audio.channels != channels:
            audio = audio.set_channels(channels)
        
        # Resample if needed
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)
        
        # Export to new format
        audio.export(output_path, format=format)
        return True
    except Exception as e:
        print(f"Error converting audio format: {e}")
        return False

def normalize_audio(input_path: str, output_path: str, target_db: float = -20.0) -> bool:
    """
    Normalize audio file to target dB level
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        target_db: Target dB level
        
    Returns:
        True if normalization was successful, False otherwise
    """
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        
        # Normalize
        change_in_db = target_db - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_db)
        
        # Export normalized audio
        normalized_audio.export(output_path, format=os.path.splitext(output_path)[1][1:])
        return True
    except Exception as e:
        print(f"Error normalizing audio: {e}")
        return False