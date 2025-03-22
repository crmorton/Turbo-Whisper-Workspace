#!/usr/bin/env python3
"""
Audio Normalizer for Bar Security Monitor

This script normalizes audio levels in recordings to ensure all voices
are at similar volume levels, regardless of how loud or quiet they are.
It then processes the normalized audio through the bar security monitor.

Author: Aye & Hue @ 8b.is (2025)
"""

import os
import sys
import argparse
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path

# Configure colorful logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Configure logger
logger = logging.getLogger("audio_normalizer")

def print_header(text):
    """Print a colorful header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def normalize_audio(input_file, output_dir=None, target_level=-16):
    """
    Normalize audio using ffmpeg with loudnorm filter
    
    Parameters:
    - input_file: Path to the input audio file
    - output_dir: Directory to save the normalized file (default: same as input)
    - target_level: Target loudness level in LUFS (default: -16)
    
    Returns:
    - Path to the normalized audio file
    """
    if not os.path.exists(input_file):
        logger.error(f"{Colors.RED}Input file not found: {input_file}{Colors.ENDC}")
        return None
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"normalized_{os.path.basename(input_file)}")
    else:
        input_dir = os.path.dirname(input_file)
        input_filename = os.path.basename(input_file)
        output_file = os.path.join(input_dir, f"normalized_{input_filename}")
    
    logger.info(f"{Colors.BLUE}Normalizing audio: {input_file}{Colors.ENDC}")
    logger.info(f"{Colors.BLUE}Target loudness: {target_level} LUFS{Colors.ENDC}")
    
    # First pass: analyze audio
    logger.info(f"{Colors.YELLOW}First pass: Analyzing audio...{Colors.ENDC}")
    analyze_cmd = [
        "ffmpeg", "-i", input_file, 
        "-af", f"loudnorm=I={target_level}:LRA=7:TP=-1.5:print_format=json", 
        "-f", "null", "-"
    ]
    
    try:
        result = subprocess.run(analyze_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"{Colors.RED}Error analyzing audio: {result.stderr}{Colors.ENDC}")
            return None
        
        # Extract the loudnorm stats from the output
        stderr_lines = result.stderr.split('\n')
        json_start = None
        json_end = None
        
        for i, line in enumerate(stderr_lines):
            if '{' in line and '"input_i"' in line:
                json_start = i
            if json_start is not None and '}' in line:
                json_end = i
                break
        
        if json_start is None or json_end is None:
            logger.error(f"{Colors.RED}Could not find loudnorm analysis in ffmpeg output{Colors.ENDC}")
            return None
        
        # Extract the measured values
        measured_i = None
        measured_lra = None
        measured_tp = None
        measured_thresh = None
        
        for line in stderr_lines[json_start:json_end+1]:
            if '"input_i"' in line:
                measured_i = line.split('"input_i" : "')[1].split('"')[0]
            elif '"input_lra"' in line:
                measured_lra = line.split('"input_lra" : "')[1].split('"')[0]
            elif '"input_tp"' in line:
                measured_tp = line.split('"input_tp" : "')[1].split('"')[0]
            elif '"input_thresh"' in line:
                measured_thresh = line.split('"input_thresh" : "')[1].split('"')[0]
        
        if not all([measured_i, measured_lra, measured_tp, measured_thresh]):
            logger.error(f"{Colors.RED}Could not extract all required loudnorm values{Colors.ENDC}")
            return None
        
        logger.info(f"{Colors.GREEN}Analysis complete:{Colors.ENDC}")
        logger.info(f"  - Measured Integrated Loudness: {measured_i} LUFS")
        logger.info(f"  - Measured Loudness Range: {measured_lra} LU")
        logger.info(f"  - Measured True Peak: {measured_tp} dBTP")
        logger.info(f"  - Measured Threshold: {measured_thresh} LUFS")
        
        # Second pass: apply normalization
        logger.info(f"{Colors.YELLOW}Second pass: Applying normalization...{Colors.ENDC}")
        normalize_cmd = [
            "ffmpeg", "-i", input_file,
            "-af", f"loudnorm=I={target_level}:LRA=7:TP=-1.5:measured_I={measured_i}:measured_LRA={measured_lra}:measured_TP={measured_tp}:measured_thresh={measured_thresh}:linear=true:print_format=summary",
            "-ar", "44100",  # Resample to 44.1kHz
            "-y", output_file
        ]
        
        result = subprocess.run(normalize_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"{Colors.RED}Error normalizing audio: {result.stderr}{Colors.ENDC}")
            return None
        
        logger.info(f"{Colors.GREEN}Normalization complete!{Colors.ENDC}")
        logger.info(f"{Colors.GREEN}Normalized file saved to: {output_file}{Colors.ENDC}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"{Colors.RED}Error during normalization: {e}{Colors.ENDC}")
        return None

def process_with_bar_monitor(audio_file):
    """Process the normalized audio with the bar security monitor"""
    if not os.path.exists(audio_file):
        logger.error(f"{Colors.RED}Audio file not found: {audio_file}{Colors.ENDC}")
        return False
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bar_monitor_script = os.path.join(script_dir, "bar_monitor.sh")
    
    if not os.path.exists(bar_monitor_script):
        logger.error(f"{Colors.RED}Bar monitor script not found: {bar_monitor_script}{Colors.ENDC}")
        return False
    
    logger.info(f"{Colors.BLUE}Processing normalized audio with bar security monitor...{Colors.ENDC}")
    
    try:
        # Make sure the script is executable
        os.chmod(bar_monitor_script, 0o755)
        
        # Run the bar monitor script
        cmd = [bar_monitor_script, audio_file]
        subprocess.run(cmd, check=True)
        
        logger.info(f"{Colors.GREEN}Bar security monitoring complete!{Colors.ENDC}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{Colors.RED}Error running bar monitor: {e}{Colors.ENDC}")
        return False
    except Exception as e:
        logger.error(f"{Colors.RED}Unexpected error: {e}{Colors.ENDC}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Audio Normalizer for Bar Security Monitor")
    parser.add_argument("--input", "-i", required=True, help="Input audio file or directory")
    parser.add_argument("--output-dir", "-o", help="Output directory for normalized files")
    parser.add_argument("--target-level", "-t", type=float, default=-16, 
                        help="Target loudness level in LUFS (default: -16)")
    parser.add_argument("--skip-monitor", "-s", action="store_true",
                        help="Skip processing with bar security monitor")
    
    args = parser.parse_args()
    
    print_header("AUDIO NORMALIZER FOR BAR SECURITY")
    
    # Check if ffmpeg is installed
    if not check_ffmpeg():
        logger.error(f"{Colors.RED}Error: ffmpeg is not installed. Please install it first.{Colors.ENDC}")
        sys.exit(1)
    
    # Process input file or directory
    if os.path.isfile(args.input):
        # Process a single file
        normalized_file = normalize_audio(args.input, args.output_dir, args.target_level)
        
        if normalized_file and not args.skip_monitor:
            process_with_bar_monitor(normalized_file)
            
    elif os.path.isdir(args.input):
        # Process all audio files in the directory
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        processed_files = []
        
        for root, _, files in os.walk(args.input):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    input_file = os.path.join(root, file)
                    normalized_file = normalize_audio(input_file, args.output_dir, args.target_level)
                    
                    if normalized_file:
                        processed_files.append(normalized_file)
        
        if processed_files and not args.skip_monitor:
            logger.info(f"{Colors.BLUE}Processing {len(processed_files)} normalized files with bar security monitor...{Colors.ENDC}")
            
            for file in processed_files:
                process_with_bar_monitor(file)
    else:
        logger.error(f"{Colors.RED}Error: Input not found: {args.input}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
