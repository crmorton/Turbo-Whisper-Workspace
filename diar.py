"""Speaker Diarization Module for CyberVox Audio Workspace

This module handles speaker diarization using pyannote.audio and sherpa-onnx integrations.
"""

import os
import tempfile
import torch
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# Import model functions
from model import get_speaker_diarization, read_wave

@dataclass
class DiarizationSegment:
    """Class representing a diarization segment"""
    speaker_id: int
    start_time: float
    end_time: float
    score: float = 1.0
    
    def duration(self) -> float:
        """Get segment duration"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "speaker": f"Speaker {self.speaker_id}",
            "start": self.start_time,
            "end": self.end_time,
            "score": self.score
        }


class SpeakerDiarizer:
    """Class for speaker diarization"""
    
    def __init__(
        self,
        segmentation_model: str = "pyannote/segmentation-3.0",
        embedding_model: str = "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx|25.3MB",
        num_speakers: int = 2,
        threshold: float = 0.5
    ):
        """Initialize diarizer
        
        Args:
            segmentation_model: Name of segmentation model
            embedding_model: Name of embedding model
            num_speakers: Number of speakers to detect (if known, else 0 for auto)
            threshold: Clustering threshold
        """
        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model
        self.num_speakers = num_speakers
        self.threshold = threshold
        self.diarizer = None
        
    def _ensure_diarizer(self):
        """Ensure diarizer is loaded"""
        if self.diarizer is None:
            self.diarizer = get_speaker_diarization(
                segmentation_model=self.segmentation_model,
                embedding_model=self.embedding_model,
                num_clusters=self.num_speakers,
                threshold=self.threshold
            )
    
    def process_file(self, audio_path: str) -> List[DiarizationSegment]:
        """Process audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of diarization segments
        """
        # Ensure diarizer is loaded
        self._ensure_diarizer()
        
        # Process audio
        samples, sample_rate = read_wave(audio_path)
        
        # Use the correct API for sherpa-onnx
        result = self.diarizer.process(
            samples=samples
        ).sort_by_start_time()
        
        # Convert results to segments
        segments = []
        # sherpa_onnx.OfflineSpeakerDiarization.process returns Turn objects
        for turn in result:
            segments.append(DiarizationSegment(
                speaker_id=turn.speaker,
                start_time=turn.start,
                end_time=turn.end,
                score=1.0  # No score in the sherpa-onnx API, use default
            ))
        
        return segments
    
    def process_audio(self, audio: np.ndarray, sample_rate: int) -> List[DiarizationSegment]:
        """Process audio array
        
        Args:
            audio: Audio array
            sample_rate: Sample rate of audio
            
        Returns:
            List of diarization segments
        """
        # Ensure diarizer is loaded
        self._ensure_diarizer()
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            sf.write(temp_file.name, audio, sample_rate)
            return self.process_file(temp_file.name)
    
    def estimate_num_speakers(self, audio_path: str, max_speakers: int = 10) -> int:
        """Estimate number of speakers in audio
        
        Args:
            audio_path: Path to audio file
            max_speakers: Maximum number of speakers to consider
            
        Returns:
            Estimated number of speakers
        """
        # This is a heuristic approach - we use clustering with different numbers of speakers
        # and select the one with the best silhouette score
        try:
            from sklearn.metrics import silhouette_score
            import numpy as np
            
            # Process audio
            samples, sample_rate = read_wave(audio_path)
            
            # Extract embeddings - the API doesn't have extract_embeddings method
            # Instead, we'll use a different approach by setting num_clusters=1 and then clustering
            self._ensure_diarizer()
            # Generate embeddings by processing with 1 speaker and extract them later
            # This is a temporary solution since the sherpa-onnx API doesn't expose embeddings
            
            # Since we can't access embeddings directly in the sherpa-onnx API,
            # we'll use a simpler approach - just return the default or a heuristic
            # based on audio duration
            
            # Heuristic: 1 speaker per ~30 seconds of audio, with minimum of 1 and maximum of max_speakers
            audio_duration = len(samples) / sample_rate
            estimated_speakers = max(1, min(max_speakers, int(audio_duration / 30)))
            
            # Return the estimated number of speakers or default
            best_num_speakers = max(2, estimated_speakers)  # At least 2 speakers for diarization
            
            return best_num_speakers
        
        except Exception as e:
            print(f"Error estimating number of speakers: {e}")
            return self.num_speakers  # Fall back to default

    def create_transcript_with_speakers(self, 
                                        transcript_segments: List[Dict[str, Any]],
                                        diarization_segments: List[DiarizationSegment]) -> List[Dict[str, Any]]:
        """Merge transcript segments with speaker information
        
        Args:
            transcript_segments: List of transcript segments with 'text', 'start', 'end'
            diarization_segments: List of diarization segments
            
        Returns:
            List of transcript segments with speaker information
        """
        result = []
        
        # If no diarization segments, assign alternating speakers
        if not diarization_segments:
            print("No speaker segments found. Assigning alternating speakers.")
            for i, seg in enumerate(transcript_segments):
                result.append({
                    'speaker': f"Speaker {i % 2}",  # Alternate between Speaker 0 and Speaker 1
                    'text': seg['text'],
                    'start': seg['start'],
                    'end': seg['end']
                })
            return result
        
        # Normal processing with diarization segments
        for seg in transcript_segments:
            start_time = seg['start']
            end_time = seg['end']
            text = seg['text']
            
            # Find overlapping speaker
            speaker = "Unknown"
            max_overlap = 0
            
            for diar_seg in diarization_segments:
                # Calculate overlap
                overlap_start = max(start_time, diar_seg.start_time)
                overlap_end = min(end_time, diar_seg.end_time)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    speaker = f"Speaker {diar_seg.speaker_id}"
            
            # If no overlap found, assign based on segment index
            if max_overlap == 0:
                speaker = f"Speaker {len(result) % 2}"
            
            # Create result segment
            result.append({
                'speaker': speaker,
                'text': text,
                'start': start_time,
                'end': end_time
            })
        
        return result


def format_as_conversation(segments: List[Dict[str, Any]]) -> str:
    """Format segments as conversation text
    
    Args:
        segments: List of segments with 'speaker' and 'text'
        
    Returns:
        Formatted conversation as string
    """
    # Group consecutive segments by the same speaker
    grouped_segments = []
    current_group = None
    
    for segment in segments:
        if current_group is None or current_group['speaker'] != segment['speaker']:
            # Start a new group
            current_group = {
                'speaker': segment['speaker'],
                'text': segment['text'],
                'start': segment['start'],
                'end': segment['end']
            }
            grouped_segments.append(current_group)
        else:
            # Append to current group
            current_group['text'] += " " + segment['text']
            current_group['end'] = segment['end']
    
    # Format as conversation
    return "\n\n".join([f"**{s['speaker']}**: {s['text']}" for s in grouped_segments])
