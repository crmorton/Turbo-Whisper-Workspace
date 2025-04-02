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
import re
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from pydub import AudioSegment
import tempfile
import uuid
import html
import requests

# Load environment variables
load_dotenv()

# Import from vocalis package
from vocalis.core.model import (
    speaker_segmentation_models, embedding2models,
    get_local_segmentation_models, get_local_embedding_models
)
from vocalis.utils.common_data import COMMON_NAMES
from vocalis.utils.audio_info import get_audio_info
from vocalis.utils.visualizer import plot_waveform, plot_spectrogram, plot_pitch_track, plot_chromagram, plot_speaker_diarization
from vocalis.utils.audio_processor import process_audio_file, extract_audio_features

# Check if LLM helper is available
try:
    print("Attempting to import llm_helper module...")
    from vocalis.llm import llm_helper
    print("Successfully imported llm_helper module")
    LLM_AVAILABLE = True
except ImportError:
    print("LLM helper module not available")
    LLM_AVAILABLE = False

# API Configuration
API_HOST = "localhost"
API_PORT = 8000
API_BASE_URL = f"http://{API_HOST}:{API_PORT}/api"

# Function to start the API server if not already running
def ensure_api_server_running():
    """Start the API server if it's not already running"""
    try:
        # Check if the API server is running
        response = requests.get(f"http://{API_HOST}:{API_PORT}/")
        if response.status_code == 200:
            print("API server is already running")
            return True
    except requests.exceptions.ConnectionError:
        print("API server is not running, starting it...")
        
        # Start the API server in a separate process
        import subprocess
        import threading
        
        def run_api_server():
            subprocess.run([sys.executable, "-m", "vocalis", "api", "--host", API_HOST, "--port", str(API_PORT)])
        
        # Start the API server in a separate thread
        api_thread = threading.Thread(target=run_api_server)
        api_thread.daemon = True  # Make the thread a daemon so it exits when the main program exits
        api_thread.start()
        
        # Wait for the API server to start
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"http://{API_HOST}:{API_PORT}/")
                if response.status_code == 200:
                    print("API server started successfully")
                    return True
            except requests.exceptions.ConnectionError:
                print(f"Waiting for API server to start ({i+1}/{max_retries})...")
                time.sleep(2)
        
        print("Failed to start API server")
        return False

# Ensure the API server is running
ensure_api_server_running()

# Function to process audio with the API
def process_audio_with_api(audio_path, task, segmentation_model, embedding_model, num_speakers=2, threshold=0.5):
    """Process audio with the API"""
    try:
        # Prepare the file for upload
        with open(audio_path, "rb") as f:
            files = {"file": f}
            
            # Prepare the request data
            data = {
                "task": task,
                "num_speakers": num_speakers,
                "segmentation_model": segmentation_model,
                "embedding_model": embedding_model,
                "threshold": threshold
            }
            
            # Make the API request
            response = requests.post(f"{API_BASE_URL}/transcribe", files=files, data=data)
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"API request error: {str(e)}"}

# Function to analyze audio with the API
def analyze_audio_with_api(audio_path):
    """Analyze audio with the API"""
    try:
        # Prepare the file for upload
        with open(audio_path, "rb") as f:
            files = {"file": f}
            
            # Prepare the request data
            data = {
                "include_waveform": True,
                "include_spectrogram": True,
                "include_pitch": True,
                "include_chroma": True
            }
            
            # Make the API request
            response = requests.post(f"{API_BASE_URL}/analyze", files=files, data=data)
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"API request error: {str(e)}"}

# Combined transcription and diarization using the API
def process_audio_with_diarization(audio_path, task, segmentation_model, embedding_model,
                                   num_speakers=2, threshold=0.5, return_timestamps=True):
    """Process audio with both transcription and speaker diarization using the API"""
    try:
        # Process audio using the API
        pipeline_result = process_audio_with_api(
            audio_path=audio_path,
            task=task,
            segmentation_model=segmentation_model,
            embedding_model=embedding_model,
            num_speakers=num_speakers,
            threshold=threshold
        )
        
        # Check for errors in pipeline result
        if "error" in pipeline_result:
            return pipeline_result
            
        # Extract data from pipeline result
        segments = pipeline_result.get("segments", [])
        merged_segments = pipeline_result.get("merged_segments", [])
        duration = pipeline_result.get("duration", 0)
        processing_times = pipeline_result.get("processing_times", {})
        
        # Convert to speaker segments format for plotting
        speaker_segments = []
        for segment in segments:
            speaker_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment["speaker"]
            })
        
        # Format as conversation
        from vocalis.core.diar import format_as_conversation
        conversation = format_as_conversation(merged_segments)
        
        # Create speaker diarization plot
        diarization_plot = plot_speaker_diarization(speaker_segments, duration)
        
        # Prepare enhanced output with additional information
        output_json = {
            "text": pipeline_result.get("text", ""),
            "segments": merged_segments,
            "conversation": conversation,
            "diarization_plot": diarization_plot,
            "performance": {
                "transcription_time": f"{processing_times.get('transcription', 0):.2f}s",
                "diarization_time": f"{processing_times.get('diarization', 0):.2f}s",
                "total_time": f"{processing_times.get('total', 0):.2f}s",
                "audio_duration": f"{duration:.2f}s",
                "realtime_factor": f"{processing_times.get('total', 0)/duration:.2f}x" if duration > 0 else "N/A"
            }
        }
        
        # Add LLM-generated content if available
        if "speaker_names" in pipeline_result:
            output_json["speaker_names"] = pipeline_result["speaker_names"]
            
        if "summary" in pipeline_result:
            output_json["summary"] = pipeline_result["summary"]
            
        if "topics" in pipeline_result:
            output_json["topics"] = pipeline_result["topics"]
        
        # Return the enhanced result as JSON
        return output_json
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Processing error: {str(e)}"}

# UI Theme Configuration - Cyberpunk Green
cyberpunk_theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Quicksand"), "ui-sans-serif", "system-ui"],
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
    
    /* Chat Bubbles CSS */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 15px;
        max-height: 500px;
        height: 500px;
        overflow-y: auto;
        background-color: #1a1a1a;
        border-radius: 8px;
        border: 1px solid #333;
        scroll-behavior: smooth;
    }
    .chat-message {
        display: flex;
        flex-direction: column;
        max-width: 85%;
        transition: all 0.3s ease;
    }
    .chat-message.speaker-0 {
        align-self: flex-start;
    }
    .chat-message.speaker-1 {
        align-self: flex-end;
    }
    .speaker-name {
        font-size: 0.8em;
        margin-bottom: 2px;
        color: #00ff9d;
        font-weight: bold;
    }
    .message-bubble {
        padding: 10px 15px;
        border-radius: 18px;
        position: relative;
        word-break: break-word;
    }
    .speaker-0 .message-bubble {
        background-color: #333;
        border-bottom-left-radius: 4px;
        color: #fff;
    }
    .speaker-1 .message-bubble {
        background-color: #00cc7a;
        border-bottom-right-radius: 4px;
        color: #000;
    }
    .message-time {
        font-size: 0.7em;
        margin-top: 2px;
        color: #999;
        align-self: flex-end;
    }
    .active-message {
        transform: scale(1.02);
    }
    .speaker-0.active-message .message-bubble {
        background-color: #444;
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.3);
    }
    .speaker-1.active-message .message-bubble {
        background-color: #00ff9d;
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
    }
    .chat-controls {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 15px;
    }
    
    /* Conversation Summary and Topics */
    .conversation-summary, .conversation-topics {
        background-color: #222;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        border-left: 4px solid #00ff9d;
    }
    .conversation-summary h3, .conversation-topics h4 {
        color: #00ff9d;
        margin-top: 0;
        margin-bottom: 10px;
    }
    .conversation-summary p {
        color: #ddd;
        line-height: 1.5;
    }
    .conversation-topics ul {
        margin: 0;
        padding-left: 20px;
    }
    .conversation-topics li {
        color: #ddd;
        margin-bottom: 5px;
    }
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
        # Chat Bubbles Tab (now primary and merged with transcription)
        with gr.TabItem("💬 CyberVox Chat"):
            with gr.Row():
                with gr.Column(scale=2, elem_classes="audio-input-column"):
                    audio_input = gr.Audio(
                        label="Audio Input",
                        type="filepath",
                        interactive=True,
                        elem_id="audio-input"
                    )
                    
                    # Add audio playback component
                    audio_playback = gr.Audio(
                        label="Audio Playback",
                        type="filepath",
                        interactive=False,
                        visible=False,
                        elem_id="audio-playback"
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
                        # Get locally available models
                        local_segmentation_models = get_local_segmentation_models()
                        local_embedding_models = get_local_embedding_models()
                        
                        # Use locally available segmentation models
                        segmentation_model = gr.Dropdown(
                            choices=local_segmentation_models,
                            value=local_segmentation_models[0] if local_segmentation_models else speaker_segmentation_models[0],
                            label="Segmentation Model (Local)"
                        )
                        
                        # Use locally available embedding model types
                        local_embedding_types = list(local_embedding_models.keys())
                        embedding_model_type = gr.Dropdown(
                            choices=local_embedding_types,
                            value=local_embedding_types[0] if local_embedding_types else list(embedding2models.keys())[0],
                            label="Embedding Model Type (Local)"
                        )
                        
                        # Use locally available embedding models for the selected type
                        first_type = local_embedding_types[0] if local_embedding_types else list(embedding2models.keys())[0]
                        first_models = local_embedding_models.get(first_type, embedding2models[first_type])
                        embedding_model = gr.Dropdown(
                            choices=first_models,
                            value=first_models[0] if first_models else embedding2models[first_type][0],
                            label="Embedding Model (Local)"
                        )
                        
                        threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Clustering Threshold"
                        )
                        

                    btn_process = gr.Button("Generate Chat", variant="primary")
                
                with gr.Column(scale=3):
                    # Chat container for displaying chat bubbles
                    chat_container = gr.HTML(
                        label="Chat Bubbles",
                        elem_id="chat-bubbles-container",
                        value="<div class='chat-container'>Upload an audio file and click 'Generate Chat' to start.</div>"
                    )
                    
                    with gr.Tabs():
                        with gr.TabItem("Summary"):
                            output_conversation = gr.Markdown(
                                label="Conversation Summary",
                                elem_id="conversation-output",
                                value="Upload an audio file and click 'Generate Chat' to start. A summary will appear here."
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

    
    # Connect embedding model type to embedding model choices
    def update_embedding_models(model_type):
        # Get locally available models
        local_embedding_models = get_local_embedding_models()
        
        # Use locally available models if available, otherwise fall back to all models
        if model_type in local_embedding_models:
            models = local_embedding_models[model_type]
        else:
            models = embedding2models[model_type]
            
        return gr.Dropdown(choices=models, value=models[0] if models else embedding2models[model_type][0])
    
    embedding_model_type.change(
        fn=update_embedding_models,
        inputs=embedding_model_type,
        outputs=embedding_model
    )

    # Function to generate chat bubbles from segments
    def process_chat(audio, task, segmentation_model, embedding_model, num_speakers, threshold):
        """Process audio and generate chat bubbles"""
        # Initialize default values
        chat_html = "<div class='chat-container'>Processing audio...</div>"
        audio_path = None
        summary_markdown = ""
        raw_text = ""
        json_data = None
        status_msg = "*Processing audio...*"
        
        try:
            # Check if audio is provided
            if audio is None:
                return (
                    "<div class='chat-container'>Please upload an audio file.</div>",
                    "Upload an audio file and click 'Generate Chat' to start. A summary will appear here.",
                    "",
                    None,
                    "*Please upload an audio file.*"
                )
                
            # Set audio path
            audio_path = audio
            
            # Process audio file using the API
            result = process_audio_with_diarization(
                audio_path,
                task,
                segmentation_model,
                embedding_model,
                num_speakers,
                threshold
            )
            
            # Generate chat HTML with debug info
            print(f"Result keys: {result.keys()}")
            
            # Check for errors
            if "error" in result:
                return (
                    f"<div class='chat-container'>Error: {result['error']}</div>",
                    f"Error: {result['error']}",
                    "",
                    None,
                    f"*Error: {result['error']}*"
                )
            
            # Get segments and conversation
            segments = result.get("segments", [])
            conversation = result.get("conversation", "")
            
            # Create chat bubbles HTML
            chat_html = "<div class='chat-container'>"
            
            for i, segment in enumerate(segments):
                speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
                text = segment.get("text", "")
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                
                # Determine speaker number (0 or 1) for styling
                speaker_num = 0 if "0" in speaker else 1
                
                # Format time for display
                time_format = lambda t: f"{int(t // 60):02d}:{int(t % 60):02d}"
                
                # Skip empty segments
                if not text.strip():
                    continue
                
                # Make sure HTML is properly escaped
                text_content = html.escape(text)
                
                chat_html += f"""
                <div class='chat-message speaker-{speaker_num}' data-start='{start_time}' data-end='{end_time}'>
                    <div class='speaker-name'>{speaker}</div>
                    <div class='message-bubble'>{text_content}</div>
                    <div class='message-time'>{time_format(start_time)} - {time_format(end_time)}</div>
                </div>
                """
            
            # Get summary and topics if available
            summary = result.get("summary", "")
            topics = result.get("topics", [])
            
            # Create summary for the Summary tab
            if summary:
                summary_markdown += f"## 🤖 AI Summary\n\n{summary}\n\n"
                
                # Also add to chat HTML
                chat_html += f"""
                <div class='conversation-summary'>
                    <h3>🤖 AI Summary</h3>
                    <p>{summary}</p>
                </div>
                """
            
            if topics and len(topics) > 0:
                summary_markdown += f"## 📌 Main Topics\n\n" + "\n".join([f"- {topic}" for topic in topics])
                
                # Also add to chat HTML
                topics_html = "<ul>" + "".join([f"<li>{topic}</li>" for topic in topics]) + "</ul>"
                chat_html += f"""
                <div class='conversation-topics'>
                    <h4>📌 Main Topics</h4>
                    {topics_html}
                </div>
                """
            
            chat_html += "</div>"
            
            # Add a small script to initialize the audio sync on load
            chat_html += """
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                console.log('Chat container loaded - initializing audio sync');
                if (window.setupAudioSync) {
                    window.setupAudioSync();
                } else {
                    console.log('setupAudioSync not available yet');
                }
            });
            
            // Also try immediately if DOM is already loaded
            if (document.readyState === 'complete' || document.readyState === 'interactive') {
                setTimeout(function() {
                    console.log('Trying immediate setup');
                    if (window.setupAudioSync) {
                        window.setupAudioSync();
                    }
                }, 500);
            }
            </script>
            """
            
            # Get raw text
            raw_text = result.get("text", "")
            
            # Get JSON data
            json_data = result
            
            # Format performance metrics
            performance = result.get("performance", {})
            status_msg = f"""
            *Processing complete!*
            - Audio duration: {performance.get('audio_duration', 'N/A')}
            - Processing time: {performance.get('total_time', 'N/A')}
            - Realtime factor: {performance.get('realtime_factor', 'N/A')}
            """
            
            return (
                chat_html,
                summary_markdown,
                raw_text,
                json_data,
                status_msg
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error processing audio: {str(e)}"
            return (
                f"<div class='chat-container'>{error_msg}</div>",
                error_msg,
                "",
                None,
                f"*{error_msg}*"
            )

    # Audio analysis function
    def analyze_audio(audio_path):
        """Analyze audio file and generate visualizations"""
        if not audio_path:
            return (
                None, None, None, None, 
                "*Error: No audio file provided*", 
                "*Upload an audio file to begin analysis.*"
            )
        
        try:
            # Show info message
            gr.Info("Loading and analyzing audio file...")
            
            # Use the API to analyze the audio
            api_result = analyze_audio_with_api(audio_path)
            
            # Check for errors
            if "error" in api_result:
                return (
                    None, None, None, None,
                    f"*Error: {api_result['error']}*",
                    f"*Error: {api_result['error']}*"
                )
            
            # Get audio information
            audio_info_data = api_result.get("audio_info", {})
            
            # Format the information as markdown
            info_text = f"""
            ### 🎛️ Audio Information
            
            - **Duration**: {audio_info_data.get('duration', 0):.2f} seconds
            - **Sample Rate**: {audio_info_data.get('sample_rate', 0)} Hz
            - **Channels**: {audio_info_data.get('channels', 1)}
            - **Format**: {audio_info_data.get('format', 'Unknown')}
            - **Bit Depth**: {audio_info_data.get('bit_depth', 16)} bits
            """
            
            # Get visualizations
            visualizations = api_result.get("visualizations", {})
            
            # Load audio for local visualizations if API doesn't provide them
            if not visualizations:
                audio, sr = process_audio_file(audio_path)
                
                # Generate visualizations locally
                waveform = plot_waveform(audio, sr, title="📊 Waveform")
                spectrogram = plot_spectrogram(audio, sr, title="🔊 Spectrogram")
                pitch = plot_pitch_track(audio, sr, title="🎵 Pitch Track")
                chroma = plot_chromagram(audio, sr, title="🎹 Chromagram")
            else:
                # Use visualizations from API
                waveform = visualizations.get("waveform")
                spectrogram = visualizations.get("spectrogram")
                pitch = visualizations.get("pitch")
                chroma = visualizations.get("chroma")
            
            gr.Info("Analysis completed successfully!")
            return (
                waveform, spectrogram, pitch, chroma,
                "*Analysis completed successfully!*",
                info_text
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error during analysis: {str(e)}"
            return (
                None, None, None, None,
                f"*{error_msg}*",
                f"*{error_msg}*"
            )

    # Connect the process button to the process_chat function
    btn_process.click(
        fn=process_chat,
        inputs=[
            audio_input,
            task,
            segmentation_model,
            embedding_model,
            num_speakers,
            threshold
        ],
        outputs=[
            chat_container,
            output_conversation,
            output_raw,
            output_json,
            status
        ]
    )
    
    # Connect the analyze button to the analyze_audio function
    btn_analyze.click(
        fn=analyze_audio,
        inputs=[audio_analysis_input],
        outputs=[
            waveform_plot,
            spectrogram_plot,
            pitch_plot,
            chroma_plot,
            analysis_status,
            audio_info
        ],
        show_progress=True
    )
    
    # Footer
    gr.Markdown(CREDITS, elem_classes="footer")

# Only launch the app when running as main script
if __name__ == "__main__":
    # Launch with queue enabled
    demo.queue().launch()