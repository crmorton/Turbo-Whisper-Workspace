"""
Audio Processing Pipeline for Turbo Whisper
This module provides a flexible pipeline for audio processing tasks including
transcription, diarization, and LLM-based enhancements.
"""

import time
import torch
import librosa
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

# Import from diar.py
from diar import SpeakerDiarizer, format_as_conversation

# Import LLM helper if available
try:
    import llm_helper
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("LLM helper not available. Some features will be disabled.")

# Cache for storing loaded models and GPU setup status
_PIPELINE_CACHE = {
    'transcription_model': None,
    'diarizer': None,
    'gpu_setup_done': False
}

class AudioProcessingPipeline:
    """
    A pipeline for processing audio with transcription, diarization, and LLM enhancements.
    
    This class handles the complete audio processing workflow and manages GPU resources
    efficiently by reusing models and controlling memory usage.
    """
    
    def __init__(self):
        """Initialize the audio processing pipeline."""
        self.gpu_available = self._setup_gpu()
        self.transcription_model = None
        self.diarizer = None
        self.llm_model = None
    
    def _setup_gpu(self) -> bool:
        """Configure GPU settings and optimize memory usage."""
        print("Setting up GPU...")
        # Check if GPU setup has already been done
        if _PIPELINE_CACHE['gpu_setup_done']:
            return True
            
        if not torch.cuda.is_available():
            print("No GPU available - running on CPU")
            return False
            
        try:
            # Get GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            cuda_version = torch.version.cuda
            
            # Print GPU information
            print("🚀 GPU ACCELERATION ENABLED 🚀")
            print(f"GPU Device: {gpu_name}")
            print(f"Number of GPUs: {gpu_count}")
            print(f"Total GPU Memory: {int(total_memory_mb)} MB")
            print(f"CUDA Version: {cuda_version}")
            print(f"PyTorch CUDA: {torch.version.cuda}")
            
            # Check for cuDNN
            if hasattr(torch.backends, 'cudnn'):
                print(f"cuDNN Version: {torch.backends.cudnn.version()}")
                print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
                
                # Enable TF32 for faster processing on Ampere GPUs (RTX 30xx, A100, etc)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Set cuDNN to benchmark mode for optimal performance
                torch.backends.cudnn.benchmark = True
            
            # Set memory allocation strategy
            if hasattr(torch.cuda, 'memory_stats'):
                # Use more aggressive memory allocation for RTX 4090
                torch.cuda.empty_cache()
                # Allow PyTorch to allocate memory as needed
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    # Use up to 90% of GPU memory
                    torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Mark GPU setup as done
            _PIPELINE_CACHE['gpu_setup_done'] = True
            return True
            
        except Exception as e:
            print(f"Error setting up GPU: {e}")
            return False
    
    def _clear_gpu_memory(self):
        """Free up GPU memory after processing."""
        print("Clearing GPU memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
    
    def _get_gpu_memory_info(self) -> Dict[str, str]:
        """Get current GPU memory usage information."""
        print("Getting GPU memory info...")
        if not torch.cuda.is_available():
            return {"error": "GPU not available"}
            
        try:
            # Get memory information
            if hasattr(torch.cuda, 'memory_stats'):
                stats = torch.cuda.memory_stats(0)
                allocated = stats.get('allocated_bytes.all.current', 0) / (1024 * 1024)
                reserved = stats.get('reserved_bytes.all.current', 0) / (1024 * 1024)
                free = stats.get('reserved_bytes.all.current', 0) / (1024 * 1024) - allocated
                
                # Get total memory
                total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                
                # Calculate utilization percentage
                utilization = (allocated / total) * 100 if total > 0 else 0
                
                return {
                    'total_mb': f"{total:.2f} MB",
                    'reserved_mb': f"{reserved:.2f} MB",
                    'allocated_mb': f"{allocated:.2f} MB",
                    'free_mb': f"{total - allocated:.2f} MB",
                    'utilization_percent': f"{utilization:.2f}%"
                }
            else:
                # Fallback for older PyTorch versions
                total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
                allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                free = reserved - allocated
                
                # Calculate utilization percentage
                utilization = (allocated / total) * 100 if total > 0 else 0
                
                return {
                    'total_mb': f"{total:.2f} MB",
                    'reserved_mb': f"{reserved:.2f} MB",
                    'allocated_mb': f"{allocated:.2f} MB",
                    'free_mb': f"{free:.2f} MB",
                    'utilization_percent': f"{utilization:.2f}%"
                }
        except Exception as e:
            return {"error": f"Error getting GPU memory info: {str(e)}"}
    
    def load_transcription_model(self, model_name: str = "openai/whisper-large-v3") -> bool:
        """
        Load the transcription model.
        
        Args:
            model_name: Name of the Whisper model to load
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Check if model is already loaded in cache
        if _PIPELINE_CACHE['transcription_model'] is not None:
            self.transcription_model = _PIPELINE_CACHE['transcription_model']
            print(f"Using cached transcription model: {model_name}")
            return True
            
        try:
            from transformers import pipeline
            
            # Determine device based on GPU availability
            device = "cuda:0" if self.gpu_available else "cpu"
            print(f"Loading transcription model on {device}...")
            
            # Load model with appropriate device
            self.transcription_model = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=device,
                torch_dtype=torch.float16 if self.gpu_available else torch.float32
            )
            
            # Cache the model for future use
            _PIPELINE_CACHE['transcription_model'] = self.transcription_model
            return True
            
        except Exception as e:
            print(f"Error loading transcription model: {e}")
            return False
    
    def load_diarizer(self, segmentation_model: str, embedding_model: str, 
                     num_speakers: int = 2, threshold: float = 0.5) -> bool:
        """
        Load the speaker diarization model.
        
        Args:
            segmentation_model: Path to segmentation model
            embedding_model: Path to embedding model
            num_speakers: Number of speakers to detect (0 for auto)
            threshold: Clustering threshold
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Check if model is already loaded in cache with same parameters
        if _PIPELINE_CACHE['diarizer'] is not None:
            diarizer = _PIPELINE_CACHE['diarizer']
            # Only reuse if parameters match
            if (diarizer.segmentation_model == segmentation_model and 
                diarizer.embedding_model == embedding_model and
                diarizer.threshold == threshold):
                
                # Update num_speakers if needed
                diarizer.num_speakers = num_speakers
                self.diarizer = diarizer
                print("Using cached diarizer")
                return True
        
        try:
            # We already imported SpeakerDiarizer from diar at the top of the file
            # Create SpeakerDiarizer instance
            try:
                # First try with GPU provider
                self.diarizer = SpeakerDiarizer(
                    segmentation_model=segmentation_model,
                    embedding_model=embedding_model,
                    num_speakers=num_speakers,
                    threshold=threshold
                )
            except Exception as gpu_error:
                # If GPU fails, try with CPU provider
                print(f"GPU diarization failed: {gpu_error}")
                print("Falling back to CPU for diarization (this will be slower)")
                
                # Try again with CPU provider explicitly specified
                self.diarizer = SpeakerDiarizer(
                    segmentation_model=segmentation_model,
                    embedding_model=embedding_model,
                    num_speakers=num_speakers,
                    threshold=threshold,
                    use_gpu=False  # Force CPU mode
                )
            
            # Cache the diarizer for future use
            _PIPELINE_CACHE['diarizer'] = self.diarizer
            return True
            
        except Exception as e:
            print(f"Error loading diarizer: {e}")
            return False
    
    def set_llm_model(self, model_name: str) -> bool:
        """
        Set the LLM model for name identification and summarization.
        
        Args:
            model_name: Name of the LLM model to use
            
        Returns:
            bool: True if model set successfully, False otherwise
        """
        if not LLM_AVAILABLE:
            print("LLM helper not available. Cannot set LLM model.")
            return False
            
        try:
            if model_name in llm_helper.AVAILABLE_LLM_MODELS:
                print(f"Setting LLM model to: {model_name}")
                llm_helper.set_current_model(model_name)
                self.llm_model = model_name
                return True
            else:
                print(f"LLM model {model_name} not available")
                return False
                
        except Exception as e:
            print(f"Error setting LLM model: {e}")
            return False
    
    def transcribe(self, audio_path: str, task: str = "transcribe", 
                  return_timestamps: bool = True) -> Dict[str, Any]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            task: Task type ('transcribe' or 'translate')
            return_timestamps: Whether to return timestamps
            
        Returns:
            Dict containing transcription results or error
        """
        # Make sure model is loaded
        if self.transcription_model is None:
            if not self.load_transcription_model():
                return {"error": "Failed to load transcription model"}
        
        try:
            # Log GPU memory before transcription if available
            if self.gpu_available:
                print("GPU memory before transcription:")
                print(self._get_gpu_memory_info())
                
            # Prepare generate kwargs based on task
            generate_kwargs = {"task": task}
                
            # Run transcription with parameters optimized for RTX 4090
            outputs = self.transcription_model(
                audio_path,
                chunk_length_s=60,  # Doubled chunk size for faster processing
                batch_size=512 if self.gpu_available else 32,  # Increased batch size for RTX 4090
                stride_length_s=5,  # Increased stride for faster processing
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps
            )
            
            # Log GPU memory after transcription if available
            if self.gpu_available:
                print("GPU memory after transcription:")
                print(self._get_gpu_memory_info())
                
            return outputs
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {"error": f"Transcription error: {str(e)}"}
    
    def diarize(self, audio_path: str, num_speakers: int = 2) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers to detect (0 for auto)
            
        Returns:
            List of diarization segments or empty list on error
        """
        # Make sure diarizer is loaded
        if self.diarizer is None:
            print("Diarizer not loaded. Cannot perform diarization.")
            return []
            
        try:
            # Update number of speakers if needed
            if self.diarizer.num_speakers != num_speakers:
                self.diarizer.num_speakers = num_speakers
                
            # Estimate number of speakers if set to auto
            if num_speakers == 0:
                print("Estimating number of speakers...")
                estimated_speakers = self.diarizer.estimate_num_speakers(audio_path)
                print(f"Estimated number of speakers: {estimated_speakers}")
                self.diarizer.num_speakers = estimated_speakers
            
            # Log GPU memory before diarization if available
            if self.gpu_available:
                print("GPU memory before diarization:")
                print(self._get_gpu_memory_info())
                
            # Process audio file for diarization
            segments = self.diarizer.process_file(audio_path)
            
            # Log GPU memory after diarization if available
            if self.gpu_available:
                print("GPU memory after diarization:")
                print(self._get_gpu_memory_info())
                
            return segments
            
        except Exception as e:
            print(f"Error during diarization: {e}")
            return []
    
    def identify_speaker_names(self, segments: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Identify speaker names from conversation segments.
        
        Args:
            segments: List of conversation segments
            
        Returns:
            Dictionary mapping speaker IDs to names
        """
        if not LLM_AVAILABLE:
            print("LLM helper not available. Cannot identify speaker names.")
            return {}
        
        # Check if segments is empty or invalid
        if not segments or not isinstance(segments, list):
            print("No valid segments provided for speaker name identification")
            return {}
            
        try:
            # First try the LLM-based approach
            try:
                llm_names = llm_helper.identify_speaker_names_llm(segments)
                if llm_names and len(llm_names) > 0:
                    print(f"LLM identified names: {llm_names}")
                    return llm_names
            except Exception as e:
                print(f"Error using LLM for name identification: {e}")
                # Continue to fallback method
                
            # Fallback to rule-based approach
            print("Using fallback method for name identification")
            try:
                return llm_helper.identify_speaker_names_fallback(segments)
            except Exception as e:
                print(f"Error using fallback name identification: {e}")
                return {}
        except Exception as e:
            print(f"Unexpected error in identify_speaker_names: {e}")
            return {}
    
    def generate_summary(self, segments: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the conversation.
        
        Args:
            segments: List of conversation segments
            
        Returns:
            Summary text or empty string on error
        """
        if not LLM_AVAILABLE:
            print("LLM helper not available. Cannot generate summary.")
            return ""
        
        # Check if segments is empty or invalid
        if not segments or not isinstance(segments, list):
            print("No valid segments provided for summary generation")
            return ""
            
        try:
            # Limit the number of segments to avoid overwhelming the LLM
            limited_segments = segments[:20] if len(segments) > 20 else segments
            summary = llm_helper.summarize_conversation(limited_segments)
            return summary or ""
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""
    
    def extract_topics(self, segments: List[Dict[str, Any]]) -> List[str]:
        """
        Extract main topics from the conversation.
        
        Args:
            segments: List of conversation segments
            
        Returns:
            List of topics or empty list on error
        """
        if not LLM_AVAILABLE:
            print("LLM helper not available. Cannot extract topics.")
            return []
        
        # Check if segments is empty or invalid
        if not segments or not isinstance(segments, list):
            print("No valid segments provided for topic extraction")
            return []
            
        try:
            # Limit the number of segments to avoid overwhelming the LLM
            limited_segments = segments[:20] if len(segments) > 20 else segments
            topics = llm_helper.extract_topics(limited_segments)
            return topics or []
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []
    
    def process_audio(self, audio_path: str, task: str = "transcribe", 
                     segmentation_model: str = "", embedding_model: str = "",
                     num_speakers: int = 2, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Process audio with transcription, diarization, and LLM enhancements.
        
        This is the main pipeline function that orchestrates the complete audio processing workflow.
        
        Args:
            audio_path: Path to audio file
            task: Task type ('transcribe' or 'translate')
            segmentation_model: Path to segmentation model
            embedding_model: Path to embedding model
            num_speakers: Number of speakers to detect (0 for auto)
            threshold: Clustering threshold
            
        Returns:
            Dictionary containing all processing results
        """
        # Initialize result dictionary
        result = {
            "audio_path": audio_path,
            "task": task,
            "num_speakers": num_speakers,
            "threshold": threshold,
            "processing_times": {}
        }
        
        try:
            # Track start time for performance metrics
            start_time = time.time()
            
            # Pre-load audio to avoid multiple file reads
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            result["duration"] = duration
            
            # Step 1: Load models
            if not self.load_transcription_model():
                return {"error": "Failed to load transcription model"}
                
            if not self.load_diarizer(segmentation_model, embedding_model, num_speakers, threshold):
                return {"error": "Failed to load diarizer"}
            
            # Step 2: Transcribe audio
            transcription_start = time.time()
            transcription = self.transcribe(audio_path, task)
            transcription_time = time.time() - transcription_start
            result["processing_times"]["transcription"] = transcription_time
            
            # Check for transcription errors
            if isinstance(transcription, dict) and "error" in transcription:
                return transcription
                
            # Add transcription to result
            result["text"] = transcription.get("text", "")
            result["chunks"] = transcription.get("chunks", [])
            
            # Step 3: Perform diarization
            diarization_start = time.time()
            diarization_segments = self.diarize(audio_path, num_speakers)
            diarization_time = time.time() - diarization_start
            result["processing_times"]["diarization"] = diarization_time
            
            # Convert diarization segments to dictionary format
            segments = []
            for segment in diarization_segments:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": segment["speaker"],
                    "text": segment.get("text", "")
                })
                
            # Add segments to result
            result["segments"] = segments
            
            # Step 4: Merge transcription with diarization
            merge_start = time.time()
            merged_segments = self._merge_transcription_with_diarization(
                transcription, diarization_segments
            )
            merge_time = time.time() - merge_start
            result["processing_times"]["merge"] = merge_time
            
            # Add merged segments to result
            result["merged_segments"] = merged_segments
            
            # Step 5: Add LLM enhancements if available
            if LLM_AVAILABLE:
                try:
                    llm_start = time.time()
                    
                    # Identify speaker names
                    try:
                        speaker_names = self.identify_speaker_names(segments)
                        if speaker_names and any(speaker_names.values()):
                            result["speaker_names"] = speaker_names
                    except Exception as e:
                        print(f"Error identifying speaker names: {e}")
                        result["speaker_names"] = {}
                        
                    # Generate summary
                    try:
                        summary = self.generate_summary(segments)
                        if summary:
                            result["summary"] = summary
                    except Exception as e:
                        print(f"Error generating summary: {e}")
                        result["summary"] = ""
                        
                    # Extract topics
                    try:
                        topics = self.extract_topics(segments)
                        if topics:
                            result["topics"] = topics
                    except Exception as e:
                        print(f"Error extracting topics: {e}")
                        result["topics"] = []
                        
                    llm_time = time.time() - llm_start
                    result["processing_times"]["llm"] = llm_time
                except Exception as e:
                    print(f"Error in LLM processing: {e}")
                    result["processing_times"]["llm"] = 0
            
            # Calculate total processing time
            total_time = time.time() - start_time
            result["processing_times"]["total"] = total_time
            
            # Clean up GPU memory
            self._clear_gpu_memory()
            
            return result
            
        except Exception as e:
            print(f"Error in audio processing pipeline: {e}")
            return {"error": f"Processing error: {str(e)}"}
    
    def _merge_transcription_with_diarization(self, transcription, diarization_segments):
        """
        Merge transcription chunks with diarization segments.
        
        Args:
            transcription: Transcription output from Whisper
            diarization_segments: Speaker diarization segments
            
        Returns:
            List of merged segments with speaker information
        """
        # Extract chunks from transcription
        chunks = transcription.get("chunks", [])
        if not chunks:
            return []
        
        # Create transcript segments from Whisper output
        transcript_segments = []
        for chunk in chunks:
            if "timestamp" not in chunk:
                continue
                
            transcript_segments.append({
                "text": chunk["text"],
                "start": chunk["timestamp"][0],
                "end": chunk["timestamp"][1]
            })
        
        # If no transcript segments were created, return empty list
        if not transcript_segments:
            return []
        
        # Use the diarizer to merge transcript with speaker information
        # This leverages the existing logic in the SpeakerDiarizer class
        merged_segments = self.diarizer.create_transcript_with_speakers(transcript_segments, diarization_segments)
        
        return merged_segments
    
    # The _find_speaker_for_chunk method has been removed as we now use the diarizer's create_transcript_with_speakers method
