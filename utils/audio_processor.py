"""
Audio processing utilities for CyberVox Audio Workspace
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import tempfile

def process_audio_file(file_path, target_sr=16000, normalize=True):
    """
    Process audio file to prepare for model input
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        normalize: Whether to normalize audio
        
    Returns:
        Processed audio as numpy array and sample rate
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Normalize if requested
        if normalize:
            audio = librosa.util.normalize(audio)
            
        return audio, sr
    except Exception as e:
        raise ValueError(f"Error processing audio file: {str(e)}")

def extract_audio_features(audio, sr, feature_type='mfcc', n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract audio features from audio signal
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        feature_type: Type of feature to extract ('mfcc', 'spectral', 'chroma')
        n_mfcc: Number of MFCCs to extract (if feature_type='mfcc')
        n_fft: FFT window size
        hop_length: Hop length for FFT
        
    Returns:
        Extracted features as numpy array
    """
    try:
        if feature_type == 'mfcc':
            # Extract MFCCs
            features = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length
            )
            return features
            
        elif feature_type == 'spectral':
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, 
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length
            )
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, 
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, 
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Combine features
            features = np.vstack([
                spectral_centroid,
                spectral_bandwidth,
                spectral_rolloff
            ])
            
            return features
            
        elif feature_type == 'chroma':
            # Extract chroma features
            features = librosa.feature.chroma_stft(
                y=audio, 
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length
            )
            return features
            
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
    except Exception as e:
        raise ValueError(f"Error extracting audio features: {str(e)}")

def convert_audio_format(input_path, output_format='wav', output_path=None, target_sr=16000):
    """
    Convert audio file to different format
    
    Args:
        input_path: Path to input audio file
        output_format: Output format (wav, mp3, etc.)
        output_path: Path to save output file (if None, uses temp file)
        target_sr: Target sample rate
        
    Returns:
        Path to converted audio file
    """
    try:
        # Generate output path if not provided
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            filename = os.path.basename(input_path)
            basename = os.path.splitext(filename)[0]
            output_path = os.path.join(temp_dir, f"{basename}.{output_format}")
        
        # Load audio using pydub (supports more formats)
        audio = AudioSegment.from_file(input_path)
        
        # Set sample rate if needed
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        
        # Convert to mono if needed
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Export to output format
        audio.export(output_path, format=output_format)
        
        return output_path
        
    except Exception as e:
        raise ValueError(f"Error converting audio format: {str(e)}")

def split_audio(input_path, segments, output_dir=None):
    """
    Split audio file into segments
    
    Args:
        input_path: Path to input audio file
        segments: List of dicts with start and end times in seconds
        output_dir: Directory to save output files (if None, uses temp dir)
        
    Returns:
        List of paths to segment files
    """
    try:
        # Create output directory if not provided
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        
        # Split audio into segments
        output_paths = []
        for i, segment in enumerate(segments):
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            
            # Extract segment
            segment_audio = audio[start_ms:end_ms]
            
            # Save segment
            output_filename = os.path.join(output_dir, f"segment_{i:03d}.wav")
            segment_audio.export(output_filename, format="wav")
            output_paths.append(output_filename)
        
        return output_paths
        
    except Exception as e:
        raise ValueError(f"Error splitting audio: {str(e)}")

def detect_silence(audio, sr, min_silence_duration=0.5, silence_threshold=-40):
    """
    Detect silent segments in audio
    
    Args:
        audio: Audio signal as numpy array
        sr: Sample rate
        min_silence_duration: Minimum silence duration in seconds
        silence_threshold: Silence threshold in dB
        
    Returns:
        List of dicts with start and end times of silent segments
    """
    try:
        # Convert audio to dB
        db = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        
        # Find frames below threshold
        is_silent = np.mean(db, axis=0) < silence_threshold
        
        # Find silent segments
        silent_segments = []
        in_silence = False
        silence_start = 0
        
        # Convert min_silence_duration to frames
        min_silence_frames = int(min_silence_duration * sr / librosa.get_window_length())
        
        for i, silent in enumerate(is_silent):
            frame_time = librosa.frames_to_time(i, sr=sr)
            
            if silent and not in_silence:
                # Start of silence
                in_silence = True
                silence_start = frame_time
            elif not silent and in_silence:
                # End of silence
                silence_duration = frame_time - silence_start
                if silence_duration >= min_silence_duration:
                    silent_segments.append({
                        'start': silence_start,
                        'end': frame_time,
                        'duration': silence_duration
                    })
                in_silence = False
        
        # Handle case where audio ends in silence
        if in_silence:
            frame_time = librosa.frames_to_time(len(is_silent), sr=sr)
            silence_duration = frame_time - silence_start
            if silence_duration >= min_silence_duration:
                silent_segments.append({
                    'start': silence_start,
                    'end': frame_time,
                    'duration': silence_duration
                })
        
        return silent_segments
        
    except Exception as e:
        raise ValueError(f"Error detecting silence: {str(e)}")
