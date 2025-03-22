#!/usr/bin/env python3
"""
Easy Audio Normalizer

A simple script to normalize audio levels in recordings using pydub.
Makes all voices close to the same level, no matter how loud or soft.

Author: Aye & Hue @ 8b.is (2025)
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import tempfile

# Colorful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print a colorful header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

def install_pydub():
    """Install pydub if not already installed"""
    try:
        import pydub
        print(f"{Colors.GREEN}pydub is already installed!{Colors.ENDC}")
        return True
    except ImportError:
        print(f"{Colors.YELLOW}pydub is not installed. Installing...{Colors.ENDC}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub"])
            print(f"{Colors.GREEN}pydub installed successfully!{Colors.ENDC}")
            return True
        except Exception as e:
            print(f"{Colors.RED}Failed to install pydub: {e}{Colors.ENDC}")
            return False

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"{Colors.GREEN}ffmpeg is installed!{Colors.ENDC}")
        return True
    except FileNotFoundError:
        print(f"{Colors.RED}ffmpeg is not installed. Please install it first.{Colors.ENDC}")
        return False

def normalize_audio(input_file, output_dir=None, target_dBFS=-16):
    """
    Normalize audio using pydub
    
    Parameters:
    - input_file: Path to the input audio file
    - output_dir: Directory to save the normalized file (default: same as input)
    - target_dBFS: Target dBFS level (default: -16)
    
    Returns:
    - Path to the normalized audio file
    """
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize
        
        print(f"{Colors.BLUE}Loading audio file: {input_file}{Colors.ENDC}")
        
        # Load the audio file
        audio = AudioSegment.from_file(input_file)
        
        # Get original levels
        original_dBFS = audio.dBFS
        print(f"{Colors.BLUE}Original audio level: {original_dBFS:.2f} dBFS{Colors.ENDC}")
        print(f"{Colors.BLUE}Target audio level: {target_dBFS:.2f} dBFS{Colors.ENDC}")
        
        # Normalize the audio
        print(f"{Colors.YELLOW}Normalizing audio...{Colors.ENDC}")
        normalized_audio = normalize(audio, target_dBFS=target_dBFS)
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"normalized_{os.path.basename(input_file)}")
        else:
            input_dir = os.path.dirname(input_file)
            input_filename = os.path.basename(input_file)
            output_file = os.path.join(input_dir, f"normalized_{input_filename}")
        
        # Export the normalized audio
        print(f"{Colors.YELLOW}Exporting normalized audio...{Colors.ENDC}")
        normalized_audio.export(output_file, format=os.path.splitext(input_file)[1][1:])
        
        print(f"{Colors.GREEN}Normalized audio saved to: {output_file}{Colors.ENDC}")
        print(f"{Colors.GREEN}New audio level: {normalized_audio.dBFS:.2f} dBFS{Colors.ENDC}")
        
        return output_file
        
    except Exception as e:
        print(f"{Colors.RED}Error normalizing audio: {e}{Colors.ENDC}")
        return None

def process_with_turbo_whisper(audio_file):
    """Process the normalized audio with Turbo Whisper"""
    if not os.path.exists(audio_file):
        print(f"{Colors.RED}Audio file not found: {audio_file}{Colors.ENDC}")
        return False
    
    try:
        # Get the project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        # Run the audio pipeline
        print(f"{Colors.BLUE}Processing with Turbo Whisper...{Colors.ENDC}")
        
        output_dir = os.path.join(project_root, "bar_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.json")
        
        cmd = [
            sys.executable,
            os.path.join(project_root, "app.py"),
            "--audio", audio_file,
            "--output", output_file,
            "--task", "transcribe"
        ]
        
        subprocess.run(cmd, check=True)
        
        print(f"{Colors.GREEN}Processing complete!{Colors.ENDC}")
        print(f"{Colors.GREEN}Output saved to: {output_file}{Colors.ENDC}")
        
        # Now analyze the transcription for security concerns
        print(f"{Colors.BLUE}Analyzing transcription for security concerns...{Colors.ENDC}")
        
        # Check if the output file exists
        if not os.path.exists(output_file):
            print(f"{Colors.RED}Output file not found: {output_file}{Colors.ENDC}")
            return False
        
        # Simple security analysis using grep
        try:
            # Create a security report
            report_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_security_report.txt")
            
            with open(report_file, "w") as f:
                f.write("BAR SECURITY REPORT\n")
                f.write("===================\n\n")
                f.write(f"Audio File: {os.path.basename(audio_file)}\n")
                f.write(f"Date: {subprocess.check_output(['date']).decode().strip()}\n\n")
                
                # Extract the transcript
                transcript = subprocess.check_output(["grep", "-o", '"text": "[^"]*"', output_file]).decode()
                f.write("Transcript:\n")
                for line in transcript.splitlines():
                    text = line.split('"text": "')[1].strip('"')
                    f.write(f"- {text}\n")
                
                f.write("\nSpeakers:\n")
                speakers = subprocess.check_output(["grep", "-o", '"speaker_name": "[^"]*"', output_file]).decode()
                unique_speakers = set()
                for line in speakers.splitlines():
                    speaker = line.split('"speaker_name": "')[1].strip('"')
                    unique_speakers.add(speaker)
                
                for speaker in unique_speakers:
                    f.write(f"- {speaker}\n")
            
            print(f"{Colors.GREEN}Security report saved to: {report_file}{Colors.ENDC}")
            
            # Display the transcript
            print(f"\n{Colors.BLUE}Transcript:{Colors.ENDC}")
            for line in transcript.splitlines():
                text = line.split('"text": "')[1].strip('"')
                print(f"- {text}")
            
            return True
            
        except Exception as e:
            print(f"{Colors.RED}Error analyzing transcription: {e}{Colors.ENDC}")
            return False
        
    except Exception as e:
        print(f"{Colors.RED}Error processing with Turbo Whisper: {e}{Colors.ENDC}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Easy Audio Normalizer")
    parser.add_argument("--input", "-i", required=True, help="Input audio file")
    parser.add_argument("--output-dir", "-o", help="Output directory for normalized files")
    parser.add_argument("--target-level", "-t", type=float, default=-16, 
                        help="Target loudness level in dBFS (default: -16)")
    parser.add_argument("--skip-processing", "-s", action="store_true",
                        help="Skip processing with Turbo Whisper")
    
    args = parser.parse_args()
    
    print_header("EASY AUDIO NORMALIZER")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"{Colors.RED}Input file not found: {args.input}{Colors.ENDC}")
        sys.exit(1)
    
    # Check if ffmpeg is installed
    if not check_ffmpeg():
        sys.exit(1)
    
    # Install pydub if needed
    if not install_pydub():
        sys.exit(1)
    
    # Normalize the audio
    normalized_file = normalize_audio(args.input, args.output_dir, args.target_level)
    
    if normalized_file and not args.skip_processing:
        # Process with Turbo Whisper
        process_with_turbo_whisper(normalized_file)

if __name__ == "__main__":
    main()
