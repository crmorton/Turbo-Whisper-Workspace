#!/usr/bin/env python3
"""
Bar Security Monitor

A specialized module for Turbo-Whisper that analyzes bar audio for security concerns.
This lightweight version integrates with the existing audio pipeline and provides
alerts for potentially dangerous situations.

Author: Aye & Hue @ 8b.is (2025)
"""

import os
import re
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import our existing audio processing pipeline
from audio_pipeline import AudioProcessingPipeline
import llm_helper

# Configure logging with colorful output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output"""
    
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[95m', # Purple
        'RESET': '\033[0m'     # Reset
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message

# Set up logger
logger = logging.getLogger("bar_security")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Define security concern patterns
CONCERN_PATTERNS = {
    "weapon": [
        r'\b(?:gun|knife|weapon|armed)\b',
        r'\b(?:shoot|stab|cut)\b',
        r'\b(?:something in my car|something to show you)\b'
    ],
    "threat": [
        r'\b(?:kill|hurt|attack|fight|beat|hit)\b',
        r'\b(?:threat|threaten|intimidate)\b',
        r'\b(?:going to get you|come after you|find you)\b'
    ],
    "robbery": [
        r'\b(?:robbery|steal|rob|wallet|money|purse)\b',
        r'\b(?:take your|give me your|hand over)\b',
        r'\b(?:empty your pockets|clean out)\b'
    ],
    "stalking": [
        r'\b(?:follow|following|stalking|watching)\b',
        r'\b(?:been watching you|know where you live|seen you around)\b',
        r'\b(?:keep running into you|always here)\b'
    ],
    "fear": [
        r'\b(?:scared|afraid|terrified|frightened)\b',
        r'\b(?:help me|help us|save me|call for help)\b',
        r'\b(?:get away from|leave me alone|stop following)\b'
    ],
    "aggression": [
        r'\b(?:fuck|shit|bitch|asshole|motherfucker)\b',
        r'\b(?:angry|mad|pissed|furious)\b',
        r'\b(?:shut up|back off|step back|get away|leave me alone)\b'
    ],
    "drugs": [
        r'\b(?:drugs|cocaine|meth|pills|molly|ecstasy|weed|pot|marijuana|dose|high)\b',
        r'\b(?:deal|dealer|selling|buying|score|hook up|connect)\b',
        r'\b(?:got some stuff|something special|party favors)\b'
    ],
    "security": [
        r'\b(?:cops|police|security|bouncer|guard)\b',
        r'\b(?:call the|get the|bring the)\b',
        r'\b(?:throw out|kick out|escort out)\b'
    ]
}

class BarSecurityMonitor:
    """
    Bar Security Monitor for analyzing audio conversations
    and detecting potential security concerns
    """
    
    def __init__(self, 
                 output_dir: str = "security_alerts",
                 min_concern_level: int = 2):
        """
        Initialize the bar security monitor
        
        Args:
            output_dir: Directory to save security alerts
            min_concern_level: Minimum concern level to report (1-5)
        """
        self.output_dir = output_dir
        self.min_concern_level = min_concern_level
        self.pipeline = AudioProcessingPipeline()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"🍸 Bar Security Monitor initialized")
        logger.info(f"📁 Security alerts will be saved to: {output_dir}")
        logger.info(f"🔍 Minimum concern level: {min_concern_level}/5")
    
    def analyze_audio(self, audio_file: str) -> Dict[str, Any]:
        """
        Analyze an audio file for security concerns
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"🎙️ Processing audio file: {audio_file}")
        
        # Process audio using our existing pipeline
        try:
            result = self.pipeline.process_audio(
                audio_file=audio_file,
                task="transcribe",
                num_speakers=0,  # Auto-detect
                threshold=0.5
            )
            
            if not result or "segments" not in result or not result["segments"]:
                logger.error(f"❌ Failed to process audio file: {audio_file}")
                return {"error": "Failed to process audio file"}
            
            # Analyze the transcript for security concerns
            security_analysis = self._analyze_transcript(result["segments"])
            
            # Add audio file info to the analysis
            security_analysis["audio_file"] = audio_file
            security_analysis["timestamp"] = datetime.now().isoformat()
            
            # Save the analysis if concern level is high enough
            if security_analysis["concern_level"] >= self.min_concern_level:
                self._save_security_alert(security_analysis)
            
            return security_analysis
            
        except Exception as e:
            logger.error(f"❌ Error processing audio: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_transcript(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze transcript segments for security concerns
        
        Args:
            segments: List of transcript segments
            
        Returns:
            Dictionary with analysis results
        """
        # Extract full transcript
        full_transcript = " ".join([s.get("text", "") for s in segments])
        
        # Find matches for each concern category
        concern_matches = {}
        for category, patterns in CONCERN_PATTERNS.items():
            matches = self._find_matches(full_transcript, patterns)
            if matches:
                concern_matches[category] = matches
        
        # Calculate concern level
        concern_level = self._calculate_concern_level(concern_matches)
        
        # Find relevant segments
        all_matches = [match for matches in concern_matches.values() for match in matches]
        relevant_segments = self._find_relevant_segments(segments, all_matches)
        
        # Generate summary using LLM if available
        summary = self._generate_summary(relevant_segments, concern_matches)
        
        # Get speakers involved
        speakers = list(set([s.get("speaker_name", s.get("speaker", "Unknown")) 
                         for s in relevant_segments]))
        
        # Determine primary concern
        primary_concern = self._determine_primary_concern(concern_matches)
        
        return {
            "concern_level": concern_level,
            "primary_concern": primary_concern,
            "concern_matches": concern_matches,
            "relevant_segments": relevant_segments,
            "speakers": speakers,
            "summary": summary,
            "full_transcript": full_transcript
        }
    
    def _find_matches(self, text: str, patterns: List[str]) -> List[str]:
        """Find matches for the given patterns in the text"""
        matches = []
        for pattern in patterns:
            found = re.finditer(pattern, text, re.IGNORECASE)
            for match in found:
                matches.append(match.group(0))
        return matches
    
    def _calculate_concern_level(self, concern_matches: Dict[str, List[str]]) -> int:
        """
        Calculate concern level on a scale of 1-5
        
        Args:
            concern_matches: Dictionary of concern categories and their matches
            
        Returns:
            Concern level (1-5)
        """
        # Base concern level
        concern_level = 1
        
        # Count total matches
        total_matches = sum(len(matches) for matches in concern_matches.values())
        
        # Increase based on number of matches
        if total_matches > 0:
            concern_level += 1
        
        if total_matches > 3:
            concern_level += 1
        
        # Increase based on specific high-priority concerns
        if "weapon" in concern_matches and len(concern_matches["weapon"]) > 0:
            concern_level += 1
        
        if "threat" in concern_matches and len(concern_matches["threat"]) > 0:
            concern_level += 1
        
        # Cap at 5
        return min(5, concern_level)
    
    def _determine_primary_concern(self, concern_matches: Dict[str, List[str]]) -> str:
        """
        Determine the primary concern based on matches
        
        Args:
            concern_matches: Dictionary of concern categories and their matches
            
        Returns:
            Primary concern category
        """
        if not concern_matches:
            return "None"
        
        # Priority order for concerns
        priority_order = ["weapon", "threat", "robbery", "stalking", "fear", 
                         "aggression", "drugs", "security"]
        
        # Check concerns in priority order
        for concern in priority_order:
            if concern in concern_matches and len(concern_matches[concern]) > 0:
                return concern
        
        # If no priority concerns, return the one with the most matches
        return max(concern_matches.items(), key=lambda x: len(x[1]))[0]
    
    def _find_relevant_segments(self, 
                               segments: List[Dict[str, Any]], 
                               matches: List[str]) -> List[Dict[str, Any]]:
        """
        Find segments containing the matches
        
        Args:
            segments: List of transcript segments
            matches: List of matched patterns
            
        Returns:
            List of relevant segments
        """
        if not matches:
            return []
        
        relevant_segments = []
        
        for segment in segments:
            text = segment.get("text", "").lower()
            if any(match.lower() in text for match in matches):
                relevant_segments.append(segment)
                
                # Also include segments immediately before and after for context
                segment_index = segments.index(segment)
                if segment_index > 0:
                    prev_segment = segments[segment_index - 1]
                    if prev_segment not in relevant_segments:
                        relevant_segments.append(prev_segment)
                
                if segment_index < len(segments) - 1:
                    next_segment = segments[segment_index + 1]
                    if next_segment not in relevant_segments:
                        relevant_segments.append(next_segment)
        
        # Sort by start time
        relevant_segments.sort(key=lambda s: s.get("start", 0))
        
        return relevant_segments
    
    def _generate_summary(self, 
                         segments: List[Dict[str, Any]], 
                         concern_matches: Dict[str, List[str]]) -> str:
        """
        Generate a summary of the security concern
        
        Args:
            segments: List of relevant transcript segments
            concern_matches: Dictionary of concern categories and their matches
            
        Returns:
            Summary text
        """
        if not segments:
            return "No security concerns detected."
        
        try:
            # Try to use LLM helper if available
            if hasattr(llm_helper, "summarize_conversation") and callable(llm_helper.summarize_conversation):
                # Create a prompt based on the concerns
                concerns = ", ".join(concern_matches.keys())
                prompt = f"Summarize this bar conversation that contains potential security concerns related to: {concerns}. Focus on safety issues and potential threats."
                
                # Use our existing LLM helper
                summary = llm_helper.summarize_conversation(segments, prompt=prompt)
                return summary
        except Exception as e:
            logger.warning(f"⚠️ Could not generate summary with LLM: {e}")
        
        # Fallback summary generation
        primary_concern = self._determine_primary_concern(concern_matches)
        speakers = list(set([s.get("speaker_name", s.get("speaker", "Unknown")) 
                         for s in segments]))
        
        # Get some example matches
        example_matches = []
        for matches in concern_matches.values():
            example_matches.extend(matches[:2])  # Take up to 2 matches from each category
        
        # Create a simple summary
        summary = f"Potential {primary_concern} situation detected involving {', '.join(speakers)}. "
        summary += f"Concerning language includes: {', '.join(example_matches[:5])}. "
        summary += f"Security staff should monitor this situation."
        
        return summary
    
    def _save_security_alert(self, analysis: Dict[str, Any]) -> str:
        """
        Save security alert to a file
        
        Args:
            analysis: Security analysis results
            
        Returns:
            Path to the saved alert file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/alert_{timestamp}.json"
        
        # Save JSON version
        with open(filename, "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Also save a human-readable text version
        text_filename = f"{self.output_dir}/alert_{timestamp}.txt"
        with open(text_filename, "w") as f:
            f.write(f"🚨 BAR SECURITY ALERT 🚨\n")
            f.write(f"=======================\n\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Audio File: {analysis['audio_file']}\n")
            f.write(f"Concern Level: {'⚠️' * analysis['concern_level']} ({analysis['concern_level']}/5)\n")
            f.write(f"Primary Concern: {analysis['primary_concern']}\n")
            f.write(f"Speakers: {', '.join(analysis['speakers'])}\n\n")
            
            f.write(f"Summary:\n")
            f.write(f"{analysis['summary']}\n\n")
            
            f.write(f"Detected Concerns:\n")
            for category, matches in analysis['concern_matches'].items():
                f.write(f"- {category.capitalize()}: {', '.join(matches)}\n")
            
            f.write(f"\nRelevant Conversation Segments:\n")
            for segment in analysis['relevant_segments']:
                speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
                text = segment.get("text", "")
                time = f"{segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s"
                f.write(f"[{time}] {speaker}: {text}\n")
        
        logger.info(f"📝 Security alert saved to: {text_filename}")
        return text_filename

    def print_colorful_alert(self, analysis: Dict[str, Any]) -> None:
        """
        Print a colorful security alert to the console
        
        Args:
            analysis: Security analysis results
        """
        # ANSI color codes
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        
        # Print header
        print(f"\n{HEADER}{BOLD}{'=' * 80}{ENDC}")
        print(f"{HEADER}{BOLD}{'🚨 BAR SECURITY ALERT 🚨'.center(80)}{ENDC}")
        print(f"{HEADER}{BOLD}{'=' * 80}{ENDC}\n")
        
        # Print basic info
        print(f"{BLUE}{BOLD}Time:{ENDC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{BLUE}{BOLD}Audio File:{ENDC} {analysis['audio_file']}")
        
        # Print concern level with appropriate color
        concern_level = analysis['concern_level']
        level_color = GREEN if concern_level <= 2 else YELLOW if concern_level <= 3 else RED
        print(f"{BLUE}{BOLD}Concern Level:{ENDC} {level_color}{'⚠️' * concern_level} ({concern_level}/5){ENDC}")
        
        # Print primary concern
        print(f"{BLUE}{BOLD}Primary Concern:{ENDC} {YELLOW}{analysis['primary_concern'].capitalize()}{ENDC}")
        
        # Print speakers
        print(f"{BLUE}{BOLD}Speakers:{ENDC} {', '.join(analysis['speakers'])}")
        
        # Print summary
        print(f"\n{BOLD}Summary:{ENDC}")
        print(f"{analysis['summary']}")
        
        # Print detected concerns
        print(f"\n{BOLD}Detected Concerns:{ENDC}")
        for category, matches in analysis['concern_matches'].items():
            print(f"- {YELLOW}{category.capitalize()}{ENDC}: {', '.join(matches)}")
        
        # Print relevant segments
        print(f"\n{BOLD}Relevant Conversation Segments:{ENDC}")
        for segment in analysis['relevant_segments']:
            speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
            text = segment.get("text", "")
            time = f"{segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s"
            print(f"[{time}] {BOLD}{speaker}:{ENDC} {text}")


def process_audio_file(audio_file: str, 
                      output_dir: str = "security_alerts",
                      min_concern_level: int = 2) -> None:
    """
    Process a single audio file and check for security concerns
    
    Args:
        audio_file: Path to the audio file
        output_dir: Directory to save security alerts
        min_concern_level: Minimum concern level to report (1-5)
    """
    monitor = BarSecurityMonitor(output_dir=output_dir, min_concern_level=min_concern_level)
    analysis = monitor.analyze_audio(audio_file)
    
    if "error" in analysis:
        logger.error(f"❌ Error: {analysis['error']}")
        return
    
    monitor.print_colorful_alert(analysis)
    
    if analysis["concern_level"] >= min_concern_level:
        logger.warning(f"⚠️ Security concern detected! Level: {analysis['concern_level']}/5")
    else:
        logger.info(f"✅ No significant security concerns detected. Level: {analysis['concern_level']}/5")


def process_directory(directory: str,
                     output_dir: str = "security_alerts",
                     min_concern_level: int = 2,
                     extensions: List[str] = [".wav", ".mp3", ".flac", ".m4a"]) -> None:
    """
    Process all audio files in a directory
    
    Args:
        directory: Directory containing audio files
        output_dir: Directory to save security alerts
        min_concern_level: Minimum concern level to report (1-5)
        extensions: List of audio file extensions to process
    """
    logger.info(f"🔍 Scanning directory: {directory}")
    logger.info(f"🔎 Looking for files with extensions: {', '.join(extensions)}")
    
    # Find all audio files
    audio_files = []
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            audio_files.append(os.path.join(directory, filename))
    
    if not audio_files:
        logger.error(f"❌ No audio files found in {directory}")
        return
    
    logger.info(f"🎵 Found {len(audio_files)} audio files")
    
    # Process each file
    for audio_file in audio_files:
        logger.info(f"🎧 Processing: {os.path.basename(audio_file)}")
        process_audio_file(audio_file, output_dir, min_concern_level)


def test_with_mock_data(mock_file: str = "examples/mock_bar_conversation.json") -> None:
    """
    Test the bar security monitor with mock data
    
    Args:
        mock_file: Path to mock data file
    """
    logger.info(f"🧪 Testing with mock data: {mock_file}")
    
    try:
        with open(mock_file, 'r') as f:
            mock_data = json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load mock data: {e}")
        return
    
    # Create a monitor
    monitor = BarSecurityMonitor(output_dir="test_security_alerts", min_concern_level=1)
    
    # Create a mock result
    mock_result = {
        "segments": mock_data["segments"],
        "text": mock_data["text"]
    }
    
    # Analyze the mock transcript
    analysis = monitor._analyze_transcript(mock_data["segments"])
    
    # Add mock audio file info
    analysis["audio_file"] = "examples/mock_bar_audio.flac"
    analysis["timestamp"] = datetime.now().isoformat()
    
    # Print the alert
    monitor.print_colorful_alert(analysis)
    
    # Save the alert
    monitor._save_security_alert(analysis)
    
    logger.info(f"✅ Mock data test completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bar Security Monitor")
    parser.add_argument("--input", "-i", help="Input audio file or directory")
    parser.add_argument("--output", "-o", default="security_alerts", help="Output directory for security alerts")
    parser.add_argument("--level", "-l", type=int, default=2, help="Minimum concern level to report (1-5)")
    parser.add_argument("--test", "-t", action="store_true", help="Run test with mock data")
    
    args = parser.parse_args()
    
    if args.test:
        test_with_mock_data()
    elif args.input:
        if os.path.isdir(args.input):
            process_directory(args.input, args.output, args.level)
        elif os.path.isfile(args.input):
            process_audio_file(args.input, args.output, args.level)
        else:
            logger.error(f"❌ Input not found: {args.input}")
    else:
        logger.info("ℹ️ No input specified. Running test with mock data...")
        test_with_mock_data()
