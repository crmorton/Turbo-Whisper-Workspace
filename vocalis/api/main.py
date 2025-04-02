"""
FastAPI Main Application for Vocalis

This module provides the FastAPI application for Vocalis, including routes for:
- Audio transcription and diarization
- Security monitoring
- Audio analysis
"""

import os
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import Vocalis components
from vocalis.core.audio_pipeline import AudioProcessingPipeline
from vocalis.security.security_monitor import SecurityMonitor
from vocalis.security.bar_security_monitor import BarSecurityMonitor

# Create FastAPI app
app = FastAPI(
    title="Vocalis API",
    description="API for advanced audio processing, transcription, diarization, and security monitoring",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create global pipeline instance
pipeline = AudioProcessingPipeline()

# Create security monitor instances
security_monitor = SecurityMonitor()
bar_security_monitor = BarSecurityMonitor()

# Define request and response models
class TranscriptionRequest(BaseModel):
    task: str = Field("transcribe", description="Task type: 'transcribe' or 'translate'")
    num_speakers: int = Field(2, description="Number of speakers (0 for auto-detection)")
    segmentation_model: str = Field("pyannote/segmentation-3.0", description="Segmentation model name")
    embedding_model: str = Field("3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx|25.3MB", description="Embedding model name")
    threshold: float = Field(0.5, description="Clustering threshold")

class SecurityRequest(BaseModel):
    min_threat_level: int = Field(2, description="Minimum threat level to report (1-5)")
    bar_specific: bool = Field(False, description="Whether to use bar-specific security monitoring")

class AudioAnalysisRequest(BaseModel):
    include_waveform: bool = Field(True, description="Include waveform visualization")
    include_spectrogram: bool = Field(True, description="Include spectrogram visualization")
    include_pitch: bool = Field(True, description="Include pitch analysis")
    include_chroma: bool = Field(True, description="Include chromagram")

# Helper function to save uploaded file
async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = tmp.name
        return tmp_path
    finally:
        upload_file.file.close()

# Helper function to clean up temporary files
def cleanup_temp_file(file_path: str):
    try:
        os.unlink(file_path)
    except Exception as e:
        print(f"Error cleaning up temporary file: {e}")

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Vocalis API", "version": "0.1.0"}

@app.post("/api/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: TranscriptionRequest = None
):
    """
    Transcribe and diarize an audio file
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Use default request if not provided
    if request is None:
        request = TranscriptionRequest()
    
    # Save uploaded file to temporary location
    temp_file = await save_upload_file_tmp(file)
    
    try:
        # Process audio
        result = pipeline.process_audio(
            audio_path=temp_file,
            task=request.task,
            segmentation_model=request.segmentation_model,
            embedding_model=request.embedding_model,
            num_speakers=request.num_speakers,
            threshold=request.threshold
        )
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        # Clean up on error
        cleanup_temp_file(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/security/analyze")
async def analyze_security(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: SecurityRequest = None
):
    """
    Analyze audio for security concerns
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Use default request if not provided
    if request is None:
        request = SecurityRequest()
    
    # Save uploaded file to temporary location
    temp_file = await save_upload_file_tmp(file)
    
    try:
        # Choose the appropriate security monitor
        monitor = bar_security_monitor if request.bar_specific else security_monitor
        
        # Update threat level
        monitor.min_threat_level = request.min_threat_level
        
        # Process audio
        incident = monitor.process_audio_file(temp_file)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file)
        
        if incident:
            return JSONResponse(content=incident.to_dict())
        else:
            return JSONResponse(content={"message": "No security concerns detected"})
    
    except Exception as e:
        # Clean up on error
        cleanup_temp_file(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: AudioAnalysisRequest = None
):
    """
    Analyze audio characteristics
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Use default request if not provided
    if request is None:
        request = AudioAnalysisRequest()
    
    # Save uploaded file to temporary location
    temp_file = await save_upload_file_tmp(file)
    
    try:
        # Import audio analysis functions
        from vocalis.utils.audio_info import get_audio_info
        from vocalis.utils.visualizer import (
            plot_waveform, plot_spectrogram, plot_pitch_track, plot_chromagram
        )
        
        # Get audio info
        audio_info = get_audio_info(temp_file)
        
        # Initialize result
        result = {
            "audio_info": audio_info,
            "visualizations": {}
        }
        
        # Generate visualizations as requested
        if request.include_waveform:
            result["visualizations"]["waveform"] = plot_waveform(temp_file)
        
        if request.include_spectrogram:
            result["visualizations"]["spectrogram"] = plot_spectrogram(temp_file)
        
        if request.include_pitch:
            result["visualizations"]["pitch"] = plot_pitch_track(temp_file)
        
        if request.include_chroma:
            result["visualizations"]["chroma"] = plot_chromagram(temp_file)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        # Clean up on error
        cleanup_temp_file(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_available_models():
    """
    Get available models for transcription and diarization
    """
    from vocalis.core.model import get_local_segmentation_models, get_local_embedding_models
    
    # Get locally available models
    segmentation_models = get_local_segmentation_models()
    embedding_models = get_local_embedding_models()
    
    return {
        "segmentation_models": segmentation_models,
        "embedding_models": embedding_models
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("vocalis.api.main:app", host="0.0.0.0", port=8000, reload=True)