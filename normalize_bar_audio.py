#!/usr/bin/env python3
"""
Bar Audio Normalizer

A simple script to normalize audio levels in bar recordings,
ensuring all voices are at similar volume levels regardless of
how loud or quiet they are.

Author: Aye & Hue @ 8b.is (2025)
"""

import os
import sys
import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Set up colorful logging
class ColoredFormatter(logging.Formatter):
    """Colored formatter for prettier logs"""
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m'    # Reset
    }

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}{log_message}{self.COLORS['RESET']}"

# Configure logger
logger = logging.getLogger("bar_audio_normalizer")
logger.setLevel(logging.INFO)

# Create console handler with colored formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import numpy
        import soundfile
        logger.info("Required Python dependencies are installed!")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Installing required dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "soundfile"])
            logger.info("Dependencies installed successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

def normalize_audio(input_file, output_dir=None, target_level=-16):
    """
    Normalize audio using soundfile and numpy
    
    Parameters:
    - input_file: Path to the input audio file
    - output_dir: Directory to save the normalized file (default: same as input)
    - target_level: Target RMS level in dB (default: -16)
    
    Returns:
    - Path to the normalized audio file
    """
    try:
        import numpy as np
        import soundfile as sf
        
        logger.info(f"Loading audio file: {input_file}")
        
        # Load the audio file
        data, samplerate = sf.read(input_file)
        
        # Calculate current RMS level
        if data.ndim > 1:
            # Convert stereo to mono by averaging channels for RMS calculation
            rms_current = np.sqrt(np.mean(np.mean(data, axis=1)**2))
        else:
            rms_current = np.sqrt(np.mean(data**2))
        
        # Convert to dB
        current_db = 20 * np.log10(rms_current) if rms_current > 0 else -100
        logger.info(f"Current RMS level: {current_db:.2f} dB")
        logger.info(f"Target RMS level: {target_level:.2f} dB")
        
        # Calculate gain needed
        gain_linear = 10**((target_level - current_db) / 20)
        logger.info(f"Applying gain: {gain_linear:.4f}")
        
        # Apply gain
        normalized_data = data * gain_linear
        
        # Prevent clipping
        if np.max(np.abs(normalized_data)) > 0.99:
            logger.warning("Preventing clipping in normalized audio")
            normalized_data = normalized_data / np.max(np.abs(normalized_data)) * 0.99
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"normalized_{os.path.basename(input_file)}")
        else:
            input_dir = os.path.dirname(input_file)
            input_filename = os.path.basename(input_file)
            output_file = os.path.join(input_dir, f"normalized_{input_filename}")
        
        # Save the normalized audio
        logger.info(f"Saving normalized audio to: {output_file}")
        sf.write(output_file, normalized_data, samplerate)
        
        # Calculate new RMS level for verification
        if normalized_data.ndim > 1:
            rms_new = np.sqrt(np.mean(np.mean(normalized_data, axis=1)**2))
        else:
            rms_new = np.sqrt(np.mean(normalized_data**2))
        
        new_db = 20 * np.log10(rms_new) if rms_new > 0 else -100
        logger.info(f"New RMS level: {new_db:.2f} dB")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error normalizing audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_with_turbo_whisper(audio_file):
    """Process the normalized audio with the Turbo-Whisper pipeline"""
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return False
    
    try:
        # Import the audio pipeline
        logger.info("Importing audio processing pipeline...")
        from audio_pipeline import AudioProcessingPipeline
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bar_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the pipeline
        pipeline = AudioProcessingPipeline()
        
        # Process the audio
        logger.info(f"Processing audio with Turbo-Whisper: {audio_file}")
        result = pipeline.process_audio(
            audio_file=audio_file,
            task="transcribe",
            num_speakers=0,  # Auto-detect
            threshold=0.5
        )
        
        if not result or "segments" not in result or not result["segments"]:
            logger.error(f"Failed to process audio file: {audio_file}")
            return False
        
        # Save the result
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Processing complete! Output saved to: {output_file}")
        
        # Analyze for security concerns
        analyze_security_concerns(result, audio_file, output_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing with Turbo-Whisper: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def analyze_security_concerns(result, audio_file, output_dir):
    """Analyze the transcription for security concerns"""
    logger.info("Analyzing transcription for security concerns...")
    
    # Define patterns to look for
    concern_patterns = {
        "weapon": ["gun", "knife", "weapon", "armed", "shoot", "stab"],
        "threat": ["kill", "hurt", "attack", "fight", "beat", "hit", "threat", "threaten"],
        "robbery": ["robbery", "steal", "rob", "wallet", "money", "purse", "take your", "give me your"],
        "aggression": ["fuck", "shit", "bitch", "asshole", "angry", "mad", "pissed", "furious", "shut up", "back off"],
        "drugs": ["drugs", "cocaine", "meth", "pills", "molly", "ecstasy", "weed", "pot", "marijuana", "dealer", "selling"],
        "security": ["cops", "police", "security", "bouncer", "guard"]
    }
    
    # Extract full transcript
    full_transcript = " ".join([s.get("text", "") for s in result["segments"]])
    
    # Find matches for each concern category
    concern_matches = {}
    concerns = 0
    
    for category, terms in concern_patterns.items():
        matches = []
        for term in terms:
            if term.lower() in full_transcript.lower():
                matches.append(term)
        
        if matches:
            concern_matches[category] = matches
            concerns += 1
            logger.warning(f"{category.capitalize()} concerns detected: {', '.join(matches)}")
        else:
            logger.info(f"No {category} concerns detected")
    
    # Generate a concern level
    if concerns == 0:
        concern_level = 0
        logger.info("✅ No security concerns detected")
    elif concerns == 1:
        concern_level = 1
        logger.warning("⚠️ Low security concern level (1/5)")
    elif concerns == 2:
        concern_level = 2
        logger.warning("⚠️⚠️ Moderate security concern level (2/5)")
    elif concerns == 3:
        concern_level = 3
        logger.error("⚠️⚠️⚠️ High security concern level (3/5)")
    elif concerns == 4:
        concern_level = 4
        logger.error("⚠️⚠️⚠️⚠️ Very high security concern level (4/5)")
    else:
        concern_level = 5
        logger.critical("⚠️⚠️⚠️⚠️⚠️ CRITICAL security concern level (5/5)")
    
    # Extract speakers
    speakers = {}
    for segment in result["segments"]:
        speaker_id = segment.get("speaker", "Unknown")
        speaker_name = segment.get("speaker_name", speaker_id)
        
        if speaker_name not in speakers:
            speakers[speaker_name] = 0
        
        speakers[speaker_name] += 1
    
    logger.info("Speakers identified:")
    for speaker, count in speakers.items():
        logger.info(f"  - {speaker}: {count} segments")
    
    # Create a security report
    report_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_security_report.txt")
    
    with open(report_file, "w") as f:
        f.write("BAR SECURITY REPORT\n")
        f.write("===================\n\n")
        f.write(f"Audio File: {os.path.basename(audio_file)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Security Concern Level: {concern_level}/5\n\n")
        
        f.write("Potential Concerns:\n")
        if concern_matches:
            for category, matches in concern_matches.items():
                f.write(f"- {category.capitalize()}: {', '.join(matches)}\n")
        else:
            f.write("- No concerns detected\n")
        
        f.write("\nSpeakers:\n")
        for speaker, count in speakers.items():
            f.write(f"- {speaker}: {count} segments\n")
        
        f.write("\nFull Transcript:\n")
        for i, segment in enumerate(result["segments"]):
            speaker_name = segment.get("speaker_name", segment.get("speaker", "Unknown"))
            text = segment.get("text", "")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            f.write(f"[{start:.2f} - {end:.2f}] {speaker_name}: {text}\n")
    
    logger.info(f"Security report saved to: {report_file}")
    
    # Try to use LLM for summary if available
    try:
        import llm_helper
        if hasattr(llm_helper, "summarize_conversation") and callable(llm_helper.summarize_conversation):
            logger.info("Generating summary with LLM...")
            prompt = "Summarize this conversation, focusing on any potential security concerns or unusual behavior:"
            summary = llm_helper.summarize_conversation(result["segments"], prompt=prompt)
            
            # Add summary to report
            with open(report_file, "a") as f:
                f.write("\nAI-Generated Summary:\n")
                f.write(summary)
            
            logger.info("Summary added to report")
    except Exception as e:
        logger.warning(f"Could not generate summary with LLM: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Bar Audio Normalizer")
    parser.add_argument("--input", "-i", default="examples/bar/BF202417091434ED.flac", 
                        help="Input audio file (default: examples/bar/BF202417091434ED.flac)")
    parser.add_argument("--output-dir", "-o", help="Output directory for normalized files")
    parser.add_argument("--target-level", "-t", type=float, default=-16, 
                        help="Target RMS level in dB (default: -16)")
    parser.add_argument("--skip-processing", "-s", action="store_true",
                        help="Skip processing with Turbo-Whisper")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print(" BAR AUDIO NORMALIZER ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Normalize the audio
    normalized_file = normalize_audio(args.input, args.output_dir, args.target_level)
    
    if normalized_file and not args.skip_processing:
        # Process with Turbo-Whisper
        process_with_turbo_whisper(normalized_file)
    
    logger.info("Bar audio normalization complete!")

if __name__ == "__main__":
    main()
