import sys
import os

# Get the path to the app.py file
app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')

# Read the app.py file
with open(app_path, 'r') as f:
    lines = f.readlines()

# Find the create_ui function
create_ui_start = -1
for i, line in enumerate(lines):
    if line.strip() == 'def create_ui():':
        create_ui_start = i
        break

if create_ui_start == -1:
    print("Could not find create_ui function")
    sys.exit(1)

# Replace the create_ui function with a simplified version
simplified_create_ui = '''def create_ui():
    # Build Gradio UI
    with gr.Blocks(
        theme=cyberpunk_theme,
        css="""
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
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            background-color: #1a1a1a;
            border-radius: 5px;
        }
        .chat-bubble {
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 15px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .speaker-bubble {
            background-color: #2a2a2a;
            color: #f0f0f0;
            border-left: 3px solid #00ff9d;
            align-self: flex-start;
            margin-right: auto;
        }
        .conversation-summary {
            background-color: #111;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .conversation-topics {
            list-style-type: none;
            padding-left: 10px;
        }
        .conversation-topics li {
            color: #ddd;
            margin-bottom: 5px;
        }
        """
    ) as demo:
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div id="title">
                    <h1>🎧 CyberVox Audio Workspace ⚡</h1>
                    <p>Advanced audio processing with Whisper V3 Turbo and Speaker Diarization</p>
                </div>
                """)
        
        with gr.Tabs(elem_classes="tabbed-content") as tabs:
            with gr.TabItem("Transcription & Diarization"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="Upload Audio",
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
                        
                        btn_transcribe = gr.Button("Transcribe", variant="primary")
                    
                    with gr.Column(scale=2):
                        transcription_output = gr.Textbox(
                            label="Transcription",
                            placeholder="Transcription will appear here...",
                            lines=10,
                            elem_id="transcription-output"
                        )
        
        # Footer
        gr.Markdown(CREDITS, elem_classes="footer")
        
        return demo
'''

# Find the end of the create_ui function
create_ui_end = -1
brace_count = 0
for i in range(create_ui_start, len(lines)):
    if lines[i].strip() == 'def create_ui():':
        brace_count = 0
    elif '{' in lines[i]:
        brace_count += lines[i].count('{')
    elif '}' in lines[i]:
        brace_count -= lines[i].count('}')
    elif 'return demo' in lines[i]:
        create_ui_end = i + 1
        break

if create_ui_end == -1:
    print("Could not find end of create_ui function")
    sys.exit(1)

# Replace the create_ui function
new_lines = lines[:create_ui_start] + [simplified_create_ui] + lines[create_ui_end:]

# Write the modified app.py file
with open(app_path, 'w') as f:
    f.writelines(new_lines)

print("Successfully fixed app.py")
