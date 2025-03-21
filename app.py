"""
# 🎧 CyberVox Audio Workspace ⚡

A powerful audio processing workspace featuring:
- Ultra-fast Whisper V3 Turbo Transcription
- Advanced Speaker Diarization
- Audio Analysis Tools
- Cyberpunk-themed UI
"""

CREDITS = """
## Credits

This project builds upon several amazing technologies:
- [8b-is Team](https://8b.is/?ref=Turbo-Whisper-Workspace)
- [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Claude](https://claude.ai/)
"""

import os
import sys
import json
import time
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from pydub import AudioSegment
import tempfile
import uuid

# Import from local modules
from model import get_speaker_diarization, read_wave, speaker_segmentation_models, embedding2models
from utils.audio_processor import process_audio_file, extract_audio_features
from utils.audio_info import get_audio_info
from utils.visualizer import plot_waveform, plot_spectrogram, plot_pitch_track, plot_chromagram, plot_speaker_diarization
from diar import SpeakerDiarizer, format_as_conversation

# Load environment variables
load_dotenv()

# Set up GPU settings and memory management
def setup_gpu():
    """Configure GPU settings and optimize memory usage"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # Set memory strategy
        torch.cuda.empty_cache()
        return True
    return False

def clear_gpu_memory():
    """Free up GPU memory after processing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# Load Whisper Model with optimizations
def load_whisper_model(device="cuda:0" if torch.cuda.is_available() else "cpu"):
    """Load and optimize the Whisper model"""
    try:
        from transformers import pipeline
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=device,
            model_kwargs={
                "attn_implementation": "eager"  # Disabled flash_attention_2 due to installation issues
            },
        )
        return pipe
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None

# Transcription function with GPU optimization
def transcribe_audio(audio_path, task="transcribe", return_timestamps=False):
    """Transcribe audio using Whisper model"""
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load model
            pipe = load_whisper_model()
            if not pipe:
                return {"error": "Failed to load transcription model"}
            
            # Prepare generate kwargs based on task
            generate_kwargs = {"task": task}
            
            # Run transcription with optimized parameters
            outputs = pipe(
                audio_path,
                chunk_length_s=30,
                batch_size=256 if torch.cuda.is_available() else 32,
                stride_length_s=3,
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps,
            )
            
            # Clear GPU memory
            clear_gpu_memory()
            
            return outputs
    except Exception as e:
        return {"error": f"Transcription error: {str(e)}"}

# Speaker diarization function
def perform_diarization(audio_path, segmentation_model, embedding_model, num_speakers=2, threshold=0.5):
    """Perform speaker diarization on audio file"""
    try:
        # Create diarization model using our improved SpeakerDiarizer class
        diarizer = SpeakerDiarizer(
            segmentation_model=segmentation_model,
            embedding_model=embedding_model,
            num_speakers=num_speakers,
            threshold=threshold
        )
        
        # Estimate number of speakers if set to auto
        if num_speakers == 0:
            gr.Info("Estimating number of speakers...")
            num_speakers = diarizer.estimate_num_speakers(audio_path)
            gr.Info(f"Estimated number of speakers: {num_speakers}")
            diarizer.num_speakers = num_speakers
        
        # Process audio file
        segments = diarizer.process_file(audio_path)
        
        # Convert segments to dictionary format
        speakers = [segment.to_dict() for segment in segments]
        
        return speakers
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Diarization error: {str(e)}"}

# Combined transcription and diarization
def process_audio_with_diarization(audio_path, task, segmentation_model, embedding_model, 
                                  num_speakers=2, threshold=0.5, return_timestamps=True):
    """Process audio with both transcription and speaker diarization"""
    try:
        # Track start time for performance metrics
        start_time = time.time()
        
        # Get transcription with timestamps
        transcription = transcribe_audio(audio_path, task, return_timestamps=True)
        transcription_time = time.time() - start_time
        
        # Create SpeakerDiarizer instance
        diarizer = SpeakerDiarizer(
            segmentation_model=segmentation_model,
            embedding_model=embedding_model,
            num_speakers=num_speakers,
            threshold=threshold
        )
        
        # Estimate number of speakers if set to auto
        if num_speakers == 0:
            gr.Info("Estimating number of speakers...")
            estimated_speakers = diarizer.estimate_num_speakers(audio_path)
            gr.Info(f"Estimated number of speakers: {estimated_speakers}")
            diarizer.num_speakers = estimated_speakers
        
        # Process audio file
        diarization_start = time.time()
        segments = diarizer.process_file(audio_path)
        diarization_time = time.time() - diarization_start
        
        # Get basic audio info for duration
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # If there was an error in transcription
        if isinstance(transcription, dict) and "error" in transcription:
            return transcription
        
        # Convert diarization segments to dictionary format
        speaker_segments = [segment.to_dict() for segment in segments]
        
        # Create transcript segments from Whisper output
        transcript_segments = []
        if "chunks" in transcription:
            for chunk in transcription["chunks"]:
                transcript_segments.append({
                    "text": chunk["text"],
                    "start": chunk["timestamp"][0],
                    "end": chunk["timestamp"][1]
                })
        else:
            # If no timestamps, create a single segment
            transcript_segments.append({
                "text": transcription["text"] if "text" in transcription else str(transcription),
                "start": 0,
                "end": duration
            })
            
        # Merge transcript with speaker information
        result = diarizer.create_transcript_with_speakers(transcript_segments, segments)
        
        # Format as conversation
        conversation = format_as_conversation(result)
        
        # Create speaker diarization plot
        diarization_plot = plot_speaker_diarization(speaker_segments, duration)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        return {
            "text": transcription["text"] if "text" in transcription else "",
            "segments": result,
            "conversation": conversation,
            "diarization_plot": diarization_plot,
            "performance": {
                "transcription_time": f"{transcription_time:.2f}s",
                "diarization_time": f"{diarization_time:.2f}s",
                "total_time": f"{total_time:.2f}s",
                "audio_duration": f"{duration:.2f}s",
                "realtime_factor": f"{total_time/duration:.2f}x"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Processing error: {str(e)}"}

# UI Theme Configuration - Cyberpunk Green
cyberpunk_theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Orbitron"), "ui-sans-serif", "system-ui"],
).set(
    button_primary_background_fill="#00ff9d",
    button_primary_background_fill_hover="#00cc7a",
    button_primary_text_color="black",
    button_primary_border_color="#00ff9d",
    block_label_background_fill="#111111",
    block_label_text_color="#00ff9d",
    block_title_text_color="#00ff9d",
    input_background_fill="#222222",
    slider_color="#00ff9d",
    body_text_color="#cccccc",
    body_background_fill="#111111",
)

# Build Gradio UI
with gr.Blocks(theme=cyberpunk_theme, css="""
    #title {text-align: center; margin-bottom: 10px;}
    #title h1 {
        background: linear-gradient(90deg, #00ff9d 0%, #00ccff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .status-bar {
        border-left: 4px solid #00ff9d;
        padding-left: 10px;
        background-color: rgba(0, 255, 157, 0.1);
    }
    .footer {text-align: center; opacity: 0.7; margin-top: 20px;}
    .tabbed-content {min-height: 400px;}
""") as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div id="title">
                <h1>🎧 CyberVox Audio Workspace ⚡</h1>
                <p>Advanced audio processing with Whisper V3 Turbo and Speaker Diarization</p>
            </div>
            """)
            
    with gr.Tabs(elem_classes="tabbed-content") as tabs:
        # Transcription Tab
        with gr.TabItem("🎤 Transcription & Diarization"):
            with gr.Row():
                with gr.Column(scale=2):
                    audio_input = gr.Audio(
                        label="Audio Input",
                        type="filepath",
                        interactive=True,
                        elem_id="audio-input"
                    )
                    
                    with gr.Row():
                        task = gr.Radio(
                            ["transcribe", "translate"],
                            label="Task",
                            value="transcribe",
                            interactive=True
                        )
                        
                        num_speakers = gr.Slider(
                            minimum=1, 
                            maximum=10, 
                            value=2, 
                            step=1,
                            label="Number of Speakers",
                            interactive=True
                        )
                    
                    with gr.Accordion("Advanced Settings", open=False):
                        segmentation_model = gr.Dropdown(
                            choices=speaker_segmentation_models,
                            value=speaker_segmentation_models[0],
                            label="Segmentation Model"
                        )
                        
                        embedding_model_type = gr.Dropdown(
                            choices=list(embedding2models.keys()),
                            value=list(embedding2models.keys())[0],
                            label="Embedding Model Type"
                        )
                        
                        embedding_model = gr.Dropdown(
                            choices=embedding2models[list(embedding2models.keys())[0]],
                            value=embedding2models[list(embedding2models.keys())[0]][0],
                            label="Embedding Model"
                        )
                        
                        threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Clustering Threshold"
                        )
                    
                    btn_process = gr.Button("Process Audio", variant="primary")
                
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("Conversation"):
                            output_conversation = gr.Markdown(
                                label="Conversation",
                                elem_id="conversation-output",
                                value="Upload an audio file and click 'Process Audio' to start."
                            )
                        
                        with gr.TabItem("Raw Text"):
                            output_raw = gr.Textbox(
                                label="Raw Transcription",
                                interactive=False,
                                elem_id="transcription-output",
                                lines=15
                            )
                        
                        with gr.TabItem("JSON Data"):
                            output_json = gr.JSON(
                                label="Detailed Results"
                            )
            
            with gr.Row():
                status = gr.Markdown(
                    value="*System ready. Upload an audio file to begin.*", 
                    elem_classes="status-bar"
                )
        
        # Audio Analysis Tab
        with gr.TabItem("📊 Audio Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_analysis_input = gr.Audio(
                        label="🎵 Audio Input",
                        type="filepath",
                        interactive=True
                    )
                    
                    btn_analyze = gr.Button("🔍 Analyze Audio", variant="primary")
                    
                    # Info panel
                    audio_info = gr.Markdown(
                        label="Audio Information",
                        elem_id="audio-info-panel"
                    )
                    
                    analysis_status = gr.Markdown(
                        value="*Upload an audio file and click 'Analyze Audio' to begin.*", 
                        elem_classes="status-bar"
                    )
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Waveform"):
                            waveform_plot = gr.Plot(label="📊 Waveform Visualization")
                        
                        with gr.TabItem("Spectrogram"):
                            spectrogram_plot = gr.Plot(label="🔊 Spectrogram Analysis")
                        
                        with gr.TabItem("Pitch Track"):
                            pitch_plot = gr.Plot(label="🎵 Pitch Analysis")
                        
                        with gr.TabItem("Chromagram"):
                            chroma_plot = gr.Plot(label="🎹 Note Distribution")
        
        # Audio Tools Tab
        with gr.TabItem("🔧 Audio Tools"):
            gr.Markdown("Coming soon: Audio enhancement, noise reduction, and more tools!")
    
    # Connect embedding model type to embedding model choices
    def update_embedding_models(model_type):
        return gr.Dropdown(choices=embedding2models[model_type], value=embedding2models[model_type][0])
    
    embedding_model_type.change(
        fn=update_embedding_models,
        inputs=embedding_model_type,
        outputs=embedding_model
    )
    
    # Define the processing function
    def process_audio(audio, task, segmentation_model, embedding_model, num_speakers, threshold):
        if not audio:
            return ("Upload an audio file to begin.", 
                   "*Error: No audio file provided*", 
                   None)
        
        try:
            # Process audio
            yield (
                "Processing audio...", 
                "*Starting transcription and diarization...*",
                None
            )
            
            # Process with transcription and diarization
            result = process_audio_with_diarization(
                audio, 
                task, 
                segmentation_model, 
                embedding_model, 
                num_speakers,
                threshold
            )
            
            # Check for errors
            if isinstance(result, dict) and "error" in result:
                yield (
                    "An error occurred during processing.",
                    f"*Error: {result['error']}*",
                    result
                )
                return
            
            # Handle successful processing
            if "conversation" in result:
                yield (
                    result["text"],
                    f"*Processing completed successfully! Identified {num_speakers} speakers.*",
                    result
                )
                return result["conversation"], f"*Processing completed successfully! Identified {num_speakers} speakers.*", result
            else:
                yield (
                    result["text"],
                    "*Processing completed, but speaker diarization might not be accurate.*",
                    result
                )
                return
            
        except Exception as e:
            yield (
                "An error occurred during processing.",
                f"*Error: {str(e)}*",
                {"error": str(e)}
            )
    
    # Connect the process button
    btn_process.click(
        fn=process_audio,
        inputs=[
            audio_input, 
            task, 
            segmentation_model, 
            embedding_model, 
            num_speakers, 
            threshold
        ],
        outputs=[
            output_raw, 
            status, 
            output_json
        ],
        show_progress=True
    )
    
    # Audio analysis functions
    def analyze_audio(audio_path):
        if not audio_path:
            return None, None, None, None, "*Error: No audio file provided*", None
        
        try:
            # Update status
            yield None, None, None, None, "*Analyzing audio...*", None
            
            gr.Info("Loading and analyzing audio file...")
            
            # Process audio using our utility functions
            audio, sr = process_audio_file(audio_path)
            
            # Get comprehensive audio information
            audio_features = get_audio_info(audio_path)
            
            # Format the information as markdown
            info_text = f"""
            ### 🎛️ Audio Information
            
            - **Duration**: {audio_features.get('duration', 0):.2f} seconds
            - **Sample Rate**: {audio_features.get('frame_rate', sr)} Hz
            - **Channels**: {audio_features.get('channels', 1)}
            - **Format**: {audio_features.get('format', 'Unknown')}
            - **Bit Depth**: {audio_features.get('sample_width', 2) * 8} bits
            - **Bitrate**: {audio_features.get('bitrate', 0) / 1000:.1f} kbps
            
            ### 📊 Audio Analysis
            
            - **RMS Energy**: {audio_features.get('rms', 0):.4f}
            - **Zero Crossing Rate**: {audio_features.get('zero_crossing_rate', 0):.4f}
            """
            
            # Add spectral features if available
            if 'spectral_centroid' in audio_features:
                info_text += f"\n- **Spectral Centroid**: {audio_features['spectral_centroid']:.2f} Hz"
            
            if 'spectral_bandwidth' in audio_features:
                info_text += f"\n- **Spectral Bandwidth**: {audio_features['spectral_bandwidth']:.2f} Hz"
                
            if 'spectral_rolloff' in audio_features:
                info_text += f"\n- **Spectral Rolloff**: {audio_features['spectral_rolloff']:.2f} Hz"
                
            if 'spectral_contrast' in audio_features:
                info_text += f"\n- **Spectral Contrast**: {audio_features['spectral_contrast']:.4f}"
            
            gr.Info("Generating visualizations...")
            
            # Generate visualizations using our enhanced functions
            waveform = plot_waveform(audio, sr, title="📊 Waveform")
            spectrogram = plot_spectrogram(audio, sr, title="🔊 Spectrogram")
            pitch_track = plot_pitch_track(audio, sr, title="🎵 Pitch Track")
            chromagram = plot_chromagram(audio, sr, title="🎹 Chromagram")
            
            # Try to get speaker diarization
            try:
                gr.Info("Analyzing speakers...")
                diarizer = SpeakerDiarizer()
                segments = diarizer.process_file(audio_path)
                speaker_segments = [segment.to_dict() for segment in segments]
                
                # Add speaker info to output
                info_text += "\n\n### 🎤 Speaker Segments\n"
                for segment in speaker_segments:
                    info_text += f"\n- **{segment['speaker']}**: {segment['start']:.2f}s - {segment['end']:.2f}s (Duration: {segment['end']-segment['start']:.2f}s)"
            except Exception as diar_err:
                info_text += f"\n\n### 🎤 Speaker Analysis\n\n*Speaker diarization unavailable: {str(diar_err)}*"
            
            gr.Info("Analysis completed successfully!")
            yield waveform, spectrogram, pitch_track, chromagram, "*Analysis completed successfully!*", info_text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield None, None, None, None, f"*Error during analysis: {str(e)}*", None
    
    # Connect the analyze button
    btn_analyze.click(
        fn=analyze_audio,
        inputs=[audio_analysis_input],
        outputs=[waveform_plot, spectrogram_plot, pitch_plot, chroma_plot, analysis_status, audio_info],
        show_progress=True
    )
    
    # Footer
    gr.Markdown(CREDITS, elem_classes="footer")

# Launch with queue enabled
demo.queue().launch()