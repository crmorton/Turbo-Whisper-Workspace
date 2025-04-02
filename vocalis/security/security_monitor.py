#!/usr/bin/env python3
"""
Security Monitor Module for Vocalis

This module analyzes audio from security cameras/microphones to detect potentially
concerning situations, threats, or aggressive behavior through speech analysis.

Author: Aye & Hue @ 8b.is
Date: 2025-03-22
"""

import os
import sys
import json
import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Import our existing audio processing pipeline
from vocalis.core.audio_pipeline import AudioProcessingPipeline
from vocalis.llm import llm_helper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("security_monitor")

# Define concerning patterns to watch for
THREAT_PATTERNS = [
    r'\b(?:kill|hurt|attack|fight|gun|knife|weapon|threat|threaten|beat|hit)\b',
    r'\b(?:robbery|steal|rob|take your|give me your)\b',
    r'\b(?:follow|following|stalking|watching)\b',
    r'\b(?:scared|afraid|terrified|help me|help us)\b',
    r'\b(?:cops|police|security|bouncer)\b',
    r"\b(?:don't tell|shut up|be quiet|keep your mouth)\b",
]

AGGRESSION_INDICATORS = [
    r'\b(?:fuck|shit|bitch|asshole|motherfucker)\b',
    r'\b(?:angry|mad|pissed|furious)\b',
    r'\b(?:shut up|back off|step back|get away|leave me alone)\b',
]

DRUG_INDICATORS = [
    r'\b(?:drugs|cocaine|meth|pills|molly|ecstasy|weed|pot|marijuana|dose|high)\b',
    r'\b(?:deal|dealer|selling|buying|score|hook up|connect)\b',
]

class SecurityIncident:
    """Class representing a potential security incident detected in audio"""
    
    def __init__(self, 
                 timestamp: datetime,
                 audio_file: str,
                 transcript: str,
                 threat_level: int,  # 1-5 scale
                 incident_type: str,
                 relevant_segments: List[Dict[str, Any]],
                 speakers_involved: List[str],
                 summary: str):
        self.timestamp = timestamp
        self.audio_file = audio_file
        self.transcript = transcript
        self.threat_level = threat_level
        self.incident_type = incident_type
        self.relevant_segments = relevant_segments
        self.speakers_involved = speakers_involved
        self.summary = summary
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "audio_file": self.audio_file,
            "threat_level": self.threat_level,
            "incident_type": self.incident_type,
            "speakers_involved": self.speakers_involved,
            "summary": self.summary,
            "relevant_segments": self.relevant_segments
        }
    
    def __str__(self) -> str:
        """String representation of the incident"""
        return f"""
🚨 SECURITY INCIDENT DETECTED 🚨
Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
File: {os.path.basename(self.audio_file)}
Threat Level: {'⚠️' * self.threat_level} ({self.threat_level}/5)
Type: {self.incident_type}
Speakers: {', '.join(self.speakers_involved)}

Summary:
{self.summary}

Key Conversation Segments:
{self._format_segments()}
"""
    
    def _format_segments(self) -> str:
        """Format the relevant segments for display"""
        result = ""
        for segment in self.relevant_segments:
            speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
            text = segment.get("text", "")
            time = f"{segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s"
            result += f"[{time}] {speaker}: {text}\n"
        return result


class SecurityMonitor:
    """Main class for monitoring security through audio analysis"""
    
    def __init__(self, 
                 output_dir: str = "security_incidents",
                 min_threat_level: int = 2):
        """
        Initialize the security monitor
        
        Args:
            output_dir: Directory to save incident reports
            min_threat_level: Minimum threat level to report (1-5 scale)
        """
        self.pipeline = AudioProcessingPipeline()
        self.output_dir = output_dir
        self.min_threat_level = min_threat_level
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Security monitor initialized. Reports will be saved to {output_dir}")
        logger.info(f"Minimum threat level for reporting: {min_threat_level}/5")
    
    def process_audio_file(self, audio_file: str) -> Optional[SecurityIncident]:
        """
        Process an audio file and detect potential security incidents
        
        Args:
            audio_file: Path to the audio file to process
            
        Returns:
            SecurityIncident if a concerning situation is detected, None otherwise
        """
        logger.info(f"Processing audio file: {audio_file}")
        
        # Process the audio using our existing pipeline
        result = self.pipeline.process_audio(
            audio_path=audio_file,
            task="transcribe",
            num_speakers=0,  # Auto-detect number of speakers
            threshold=0.5
        )
        
        # Check if processing was successful
        if not result or "segments" not in result or not result["segments"]:
            logger.error(f"Failed to process audio file: {audio_file}")
            return None
        
        # Analyze the transcript for security concerns
        return self._analyze_transcript(audio_file, result)
    
    def _analyze_transcript(self, audio_file: str, result: Dict[str, Any]) -> Optional[SecurityIncident]:
        """
        Analyze the transcript for security concerns
        
        Args:
            audio_file: Path to the audio file
            result: Result dictionary from audio processing
            
        Returns:
            SecurityIncident if a concerning situation is detected, None otherwise
        """
        segments = result["segments"]
        full_transcript = " ".join([s.get("text", "") for s in segments])
        
        # Check for concerning patterns
        threat_matches = self._find_pattern_matches(full_transcript, THREAT_PATTERNS)
        aggression_matches = self._find_pattern_matches(full_transcript, AGGRESSION_INDICATORS)
        drug_matches = self._find_pattern_matches(full_transcript, DRUG_INDICATORS)
        
        # Calculate threat level based on matches
        threat_level = self._calculate_threat_level(threat_matches, aggression_matches, drug_matches)
        
        # If threat level is below threshold, return None
        if threat_level < self.min_threat_level:
            logger.info(f"No significant security concerns detected. Threat level: {threat_level}/5")
            return None
        
        # Determine incident type
        incident_type = self._determine_incident_type(threat_matches, aggression_matches, drug_matches)
        
        # Find relevant segments containing concerning content
        relevant_segments = self._find_relevant_segments(segments, threat_matches + aggression_matches + drug_matches)
        
        # Get speakers involved
        speakers_involved = list(set([s.get("speaker_name", s.get("speaker", "Unknown")) 
                                 for s in relevant_segments]))
        
        # Generate summary using LLM if available
        summary = self._generate_incident_summary(relevant_segments, incident_type)
        
        # Create incident report
        incident = SecurityIncident(
            timestamp=datetime.now(),
            audio_file=audio_file,
            transcript=full_transcript,
            threat_level=threat_level,
            incident_type=incident_type,
            relevant_segments=relevant_segments,
            speakers_involved=speakers_involved,
            summary=summary
        )
        
        # Save incident report
        self._save_incident_report(incident)
        
        logger.warning(f"⚠️ Security incident detected! Threat level: {threat_level}/5")
        return incident
    
    def _find_pattern_matches(self, text: str, patterns: List[str]) -> List[str]:
        """Find all matches for the given patterns in the text"""
        matches = []
        for pattern in patterns:
            found = re.finditer(pattern, text, re.IGNORECASE)
            for match in found:
                matches.append(match.group(0))
        return matches
    
    def _calculate_threat_level(self, 
                               threat_matches: List[str], 
                               aggression_matches: List[str],
                               drug_matches: List[str]) -> int:
        """
        Calculate threat level on a scale of 1-5
        
        Args:
            threat_matches: List of threat pattern matches
            aggression_matches: List of aggression pattern matches
            drug_matches: List of drug-related pattern matches
            
        Returns:
            Threat level (1-5)
        """
        # Base threat level
        threat_level = 1
        
        # Increase based on number and types of matches
        if len(threat_matches) > 0:
            threat_level += min(2, len(threat_matches))
        
        if len(aggression_matches) > 2:
            threat_level += 1
        
        if len(drug_matches) > 0:
            threat_level += 1
        
        # Cap at 5
        return min(5, threat_level)
    
    def _determine_incident_type(self,
                                threat_matches: List[str],
                                aggression_matches: List[str],
                                drug_matches: List[str]) -> str:
        """Determine the type of incident based on pattern matches"""
        # Count matches by category
        threat_count = len(threat_matches)
        aggression_count = len(aggression_matches)
        drug_count = len(drug_matches)
        
        # Determine primary incident type
        if threat_count > aggression_count and threat_count > drug_count:
            if any(word in " ".join(threat_matches).lower() for word in ["gun", "knife", "weapon"]):
                return "Potential Weapon Threat"
            elif any(word in " ".join(threat_matches).lower() for word in ["rob", "steal", "robbery"]):
                return "Potential Robbery"
            else:
                return "Verbal Threat"
        elif drug_count > aggression_count:
            return "Drug-Related Activity"
        elif aggression_count > 0:
            return "Aggressive Behavior"
        else:
            return "Suspicious Activity"
    
    def _find_relevant_segments(self, 
                               segments: List[Dict[str, Any]], 
                               pattern_matches: List[str]) -> List[Dict[str, Any]]:
        """Find segments containing the pattern matches"""
        relevant_segments = []
        
        for segment in segments:
            text = segment.get("text", "").lower()
            if any(match.lower() in text for match in pattern_matches):
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
    
    def _generate_incident_summary(self, 
                                  relevant_segments: List[Dict[str, Any]], 
                                  incident_type: str) -> str:
        """Generate a summary of the incident using LLM if available"""
        try:
            if hasattr(llm_helper, "summarize_conversation") and callable(llm_helper.summarize_conversation):
                # Use our existing LLM helper to summarize
                prompt = f"Summarize this potentially concerning {incident_type} situation in a bar or public venue. Focus on safety concerns:"
                summary = llm_helper.summarize_conversation(relevant_segments, prompt=prompt)
                return summary
        except Exception as e:
            logger.error(f"Error generating summary with LLM: {e}")
        
        # Fallback if LLM is not available or fails
        segment_texts = [f"{s.get('speaker_name', s.get('speaker', 'Person'))}: {s.get('text', '')}" 
                        for s in relevant_segments]
        return f"Potential {incident_type} detected. Relevant conversation:\n" + "\n".join(segment_texts)
    
    def _save_incident_report(self, incident: SecurityIncident) -> None:
        """Save the incident report to a file"""
        timestamp = incident.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/incident_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(incident.to_dict(), f, indent=2)
        
        logger.info(f"Incident report saved to {filename}")
        
        # Also save a human-readable text version
        text_filename = f"{self.output_dir}/incident_{timestamp}.txt"
        with open(text_filename, "w") as f:
            f.write(str(incident))
        
        logger.info(f"Human-readable report saved to {text_filename}")


def monitor_directory(directory: str, 
                     output_dir: str = "security_incidents",
                     min_threat_level: int = 2,
                     extensions: List[str] = [".wav", ".mp3", ".flac", ".m4a"]) -> None:
    """
    Monitor a directory for new audio files and process them
    
    Args:
        directory: Directory to monitor
        output_dir: Directory to save incident reports
        min_threat_level: Minimum threat level to report (1-5 scale)
        extensions: List of audio file extensions to process
    """
    monitor = SecurityMonitor(output_dir=output_dir, min_threat_level=min_threat_level)
    
    logger.info(f"Monitoring directory: {directory}")
    logger.info(f"Looking for files with extensions: {', '.join(extensions)}")
    
    # Process all existing files
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            file_path = os.path.join(directory, filename)
            logger.info(f"Processing file: {filename}")
            incident = monitor.process_audio_file(file_path)
            
            if incident:
                logger.warning(f"⚠️ Security incident detected in {filename}!")
                logger.info(str(incident))
            else:
                logger.info(f"No security concerns detected in {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Monitor for Audio Analysis")
    parser.add_argument("--input", "-i", required=True, help="Input audio file or directory")
    parser.add_argument("--output", "-o", default="security_incidents", help="Output directory for incident reports")
    parser.add_argument("--threat-level", "-t", type=int, default=2, help="Minimum threat level to report (1-5)")
    parser.add_argument("--monitor", "-m", action="store_true", help="Monitor directory for new files")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        if args.monitor:
            # TODO: Implement continuous monitoring with watchdog
            logger.error("Continuous monitoring not yet implemented")
        else:
            monitor_directory(args.input, args.output, args.threat_level)
    elif os.path.isfile(args.input):
        monitor = SecurityMonitor(output_dir=args.output, min_threat_level=args.threat_level)
        incident = monitor.process_audio_file(args.input)
        
        if incident:
            logger.warning("⚠️ Security incident detected!")
            print(str(incident))
        else:
            logger.info("No security concerns detected")
    else:
        logger.error(f"Input not found: {args.input}")