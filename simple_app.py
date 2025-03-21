import gradio as gr

def create_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div id="title">
                    <h1>🎧 CyberVox Audio Workspace ⚡</h1>
                    <p>Advanced audio processing with Whisper V3 Turbo and Speaker Diarization</p>
                </div>
                """)
        
        with gr.Tabs() as tabs:
            with gr.TabItem("Transcription"):
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Upload Audio",
                            type="filepath",
                            elem_id="audio-input"
                        )
                        
                        with gr.Row():
                            task = gr.Radio(
                                ["transcribe", "translate"],
                                label="Task",
                                value="transcribe",
                                interactive=True
                            )
                            
                            language = gr.Dropdown(
                                ["auto", "en", "fr", "de", "es", "it", "ja", "zh", "ru", "pt", "ar"],
                                label="Language",
                                value="auto",
                                interactive=True
                            )
                        
                        btn_transcribe = gr.Button("Transcribe", variant="primary")
                    
                    with gr.Column():
                        transcription_output = gr.Textbox(
                            label="Transcription",
                            placeholder="Transcription will appear here...",
                            lines=10,
                            elem_id="transcription-output"
                        )
        
        return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
