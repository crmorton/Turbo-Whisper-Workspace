#!/usr/bin/env python3
"""
Bar Audio Analyzer

A simple script to demonstrate analyzing bar audio for security concerns
using the existing Turbo-Whisper audio pipeline.

Author: Aye & Hue @ 8b.is (2025)
"""

import os
import sys
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import our existing audio processing pipeline
from audio_pipeline import AudioProcessingPipeline
import llm_helper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("bar_analyzer")

# Define colorful output helpers
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

def print_section(title):
    """Print a section title"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * len(title)}{Colors.ENDC}")

# Define patterns to look for in bar conversations
CONCERN_PATTERNS = {
    "weapon": [
        r'\b(?:gun|knife|weapon|armed)\b',
        r'\b(?:shoot|stab|cut)\b'
    ],
    "threat": [
        r'\b(?:kill|hurt|attack|fight|beat|hit)\b',
        r'\b(?:threat|threaten|intimidate)\b'
    ],
    "robbery": [
        r'\b(?:robbery|steal|rob|wallet|money|purse)\b',
        r'\b(?:take your|give me your|hand over)\b'
    ],
    "aggression": [
        r'\b(?:fuck|shit|bitch|asshole)\b',
        r'\b(?:angry|mad|pissed|furious)\b',
        r'\b(?:shut up|back off|step back|get away)\b'
    ],
    "drugs": [
        r'\b(?:drugs|cocaine|meth|pills|molly|ecstasy|weed)\b',
        r'\b(?:deal|dealer|selling|buying|score)\b'
    ],
    "security": [
        r'\b(?:cops|police|security|bouncer|guard)\b',
        r'\b(?:call the|get the|bring the)\b'
    ]
}

def find_matches(text, patterns):
    """Find all matches for the given patterns in the text"""
    matches = []
    for pattern in patterns:
        found = re.finditer(pattern, text, re.IGNORECASE)
        for match in found:
            matches.append(match.group(0))
    return matches

def analyze_transcript(segments):
    """Analyze transcript segments for security concerns"""
    # Extract full transcript
    full_transcript = " ".join([s.get("text", "") for s in segments])
    
    print_section("Full Transcript")
    print(full_transcript)
    
    # Find matches for each concern category
    print_section("Security Analysis")
    
    concern_matches = {}
    for category, patterns in CONCERN_PATTERNS.items():
        matches = find_matches(full_transcript, patterns)
        if matches:
            concern_matches[category] = matches
            print(f"{Colors.YELLOW}{category.capitalize()}{Colors.ENDC}: {', '.join(matches)}")
        else:
            print(f"{Colors.GREEN}{category.capitalize()}{Colors.ENDC}: No concerns detected")
    
    # Calculate a simple concern level (1-5)
    concern_level = 1  # Base level
    
    # Increase based on matches
    if concern_matches:
        concern_level += 1
    
    # Check for high-priority concerns
    if "weapon" in concern_matches or "threat" in concern_matches:
        concern_level += 1
    
    # Check for multiple categories
    if len(concern_matches) >= 3:
        concern_level += 1
    
    # Check for many matches
    total_matches = sum(len(matches) for matches in concern_matches.values())
    if total_matches >= 5:
        concern_level += 1
    
    # Print concern level
    level_color = Colors.GREEN if concern_level <= 2 else Colors.YELLOW if concern_level <= 3 else Colors.RED
    print(f"\n{Colors.BOLD}Overall Concern Level:{Colors.ENDC} {level_color}{'⚠️' * concern_level} ({concern_level}/5){Colors.ENDC}")
    
    # Print speaker information
    print_section("Speaker Information")
    speakers = {}
    for segment in segments:
        speaker_id = segment.get("speaker", "Unknown")
        speaker_name = segment.get("speaker_name", speaker_id)
        
        if speaker_name not in speakers:
            speakers[speaker_name] = 0
        
        speakers[speaker_name] += 1
    
    for speaker, count in speakers.items():
        print(f"{Colors.BOLD}{speaker}{Colors.ENDC}: {count} segments")
    
    return {
        "concern_level": concern_level,
        "concern_matches": concern_matches,
        "speakers": list(speakers.keys()),
        "full_transcript": full_transcript
    }

def process_audio_file(audio_file):
    """Process an audio file and analyze for security concerns"""
    print_header(f"ANALYZING BAR AUDIO: {os.path.basename(audio_file)}")
    
    # Create output directory
    output_dir = "bar_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process audio using our existing pipeline
    try:
        pipeline = AudioProcessingPipeline()
        
        print(f"{Colors.BLUE}Processing audio file...{Colors.ENDC}")
        result = pipeline.process_audio(
            audio_file=audio_file,
            task="transcribe",
            num_speakers=0,  # Auto-detect
            threshold=0.5
        )
        
        if not result or "segments" not in result or not result["segments"]:
            print(f"{Colors.RED}Failed to process audio file: {audio_file}{Colors.ENDC}")
            return
        
        # Analyze the transcript
        analysis = analyze_transcript(result["segments"])
        
        # Save the analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/analysis_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump({
                "audio_file": audio_file,
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "segments": result["segments"]
            }, f, indent=2)
        
        print(f"\n{Colors.GREEN}Analysis saved to: {filename}{Colors.ENDC}")
        
        # Generate a summary if possible
        try:
            if hasattr(llm_helper, "summarize_conversation") and callable(llm_helper.summarize_conversation):
                print_section("Conversation Summary")
                prompt = "Summarize this conversation, focusing on any potential security concerns or unusual behavior:"
                summary = llm_helper.summarize_conversation(result["segments"], prompt=prompt)
                print(summary)
        except Exception as e:
            print(f"{Colors.YELLOW}Could not generate summary: {e}{Colors.ENDC}")
        
    except Exception as e:
        print(f"{Colors.RED}Error processing audio: {e}{Colors.ENDC}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bar Audio Analyzer")
    parser.add_argument("--input", "-i", default="examples/ChrisAndAlexDiTest.flac", 
                        help="Input audio file (default: examples/ChrisAndAlexDiTest.flac)")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"{Colors.RED}Error: Input file not found: {args.input}{Colors.ENDC}")
        sys.exit(1)
    
    process_audio_file(args.input)
