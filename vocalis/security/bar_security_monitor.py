#!/usr/bin/env python3
"""
Bar Security Monitor Module for Vocalis

This module extends the security monitor with specific functionality for monitoring
bar environments, including specialized detection for bar-specific security concerns.

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

# Import our security monitor
from vocalis.security.security_monitor import SecurityMonitor, SecurityIncident, THREAT_PATTERNS, AGGRESSION_INDICATORS, DRUG_INDICATORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("bar_security_monitor")

# Additional bar-specific patterns to watch for
BAR_SPECIFIC_PATTERNS = [
    r'\b(?:bartender|bouncer|manager|security)\b',
    r'\b(?:drink|drinks|beer|shot|shots|whiskey|vodka|rum|tequila)\b',
    r'\b(?:drunk|wasted|hammered|intoxicated|cut off|cut him off|cut her off)\b',
    r'\b(?:tab|bill|check|pay|payment|credit card|cash)\b',
    r'\b(?:bathroom|restroom|toilet|stall)\b',
    r'\b(?:line|queue|waiting|wait)\b',
    r'\b(?:ID|identification|fake ID|card|driver\'s license)\b',
]

INTOXICATION_INDICATORS = [
    r'\b(?:drunk|wasted|hammered|intoxicated|tipsy|buzzed)\b',
    r'\b(?:slurring|stumbling|falling|can\'t stand|wobbling)\b',
    r'\b(?:too much|had enough|cut off|stop serving|no more)\b',
    r'\b(?:throwing up|vomit|sick|nauseous)\b',
]

UNDERAGE_INDICATORS = [
    r'\b(?:fake ID|borrowed ID|using his ID|using her ID)\b',
    r'\b(?:not 21|underage|minor|too young)\b',
    r'\b(?:birthday today|just turned|finally legal)\b',
]

class BarSecurityMonitor(SecurityMonitor):
    """Extended security monitor for bar environments"""
    
    def __init__(self, 
                 output_dir: str = "bar_security",
                 min_threat_level: int = 2):
        """
        Initialize the bar security monitor
        
        Args:
            output_dir: Directory to save incident reports
            min_threat_level: Minimum threat level to report (1-5 scale)
        """
        # Initialize the base security monitor
        super().__init__(output_dir=output_dir, min_threat_level=min_threat_level)
        
        logger.info("Bar security monitor initialized with specialized detection")
    
    def _analyze_transcript(self, audio_file: str, result: Dict[str, Any]) -> Optional[SecurityIncident]:
        """
        Analyze the transcript for bar-specific security concerns
        
        Args:
            audio_file: Path to the audio file
            result: Result dictionary from audio processing
            
        Returns:
            SecurityIncident if a concerning situation is detected, None otherwise
        """
        segments = result["segments"]
        full_transcript = " ".join([s.get("text", "") for s in segments])
        
        # Check for concerning patterns (including base patterns)
        threat_matches = self._find_pattern_matches(full_transcript, THREAT_PATTERNS)
        aggression_matches = self._find_pattern_matches(full_transcript, AGGRESSION_INDICATORS)
        drug_matches = self._find_pattern_matches(full_transcript, DRUG_INDICATORS)
        
        # Check for bar-specific patterns
        bar_matches = self._find_pattern_matches(full_transcript, BAR_SPECIFIC_PATTERNS)
        intoxication_matches = self._find_pattern_matches(full_transcript, INTOXICATION_INDICATORS)
        underage_matches = self._find_pattern_matches(full_transcript, UNDERAGE_INDICATORS)
        
        # Calculate threat level based on matches
        threat_level = self._calculate_bar_threat_level(
            threat_matches, aggression_matches, drug_matches,
            bar_matches, intoxication_matches, underage_matches
        )
        
        # If threat level is below threshold, return None
        if threat_level < self.min_threat_level:
            logger.info(f"No significant security concerns detected. Threat level: {threat_level}/5")
            return None
        
        # Determine incident type
        incident_type = self._determine_bar_incident_type(
            threat_matches, aggression_matches, drug_matches,
            bar_matches, intoxication_matches, underage_matches
        )
        
        # Find relevant segments containing concerning content
        all_matches = (threat_matches + aggression_matches + drug_matches + 
                      bar_matches + intoxication_matches + underage_matches)
        relevant_segments = self._find_relevant_segments(segments, all_matches)
        
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
        
        logger.warning(f"⚠️ Bar security incident detected! Threat level: {threat_level}/5")
        return incident
    
    def _calculate_bar_threat_level(self,
                                   threat_matches: List[str],
                                   aggression_matches: List[str],
                                   drug_matches: List[str],
                                   bar_matches: List[str],
                                   intoxication_matches: List[str],
                                   underage_matches: List[str]) -> int:
        """
        Calculate threat level for bar-specific concerns on a scale of 1-5
        
        Args:
            threat_matches: List of threat pattern matches
            aggression_matches: List of aggression pattern matches
            drug_matches: List of drug-related pattern matches
            bar_matches: List of bar-specific pattern matches
            intoxication_matches: List of intoxication-related pattern matches
            underage_matches: List of underage-related pattern matches
            
        Returns:
            Threat level (1-5)
        """
        # Start with base threat level calculation
        threat_level = self._calculate_threat_level(threat_matches, aggression_matches, drug_matches)
        
        # Add bar-specific threat level adjustments
        if len(intoxication_matches) > 2:
            threat_level += 1
        
        if len(underage_matches) > 0:
            threat_level += 1
        
        # Cap at 5
        return min(5, threat_level)
    
    def _determine_bar_incident_type(self,
                                    threat_matches: List[str],
                                    aggression_matches: List[str],
                                    drug_matches: List[str],
                                    bar_matches: List[str],
                                    intoxication_matches: List[str],
                                    underage_matches: List[str]) -> str:
        """
        Determine the type of bar-specific incident based on pattern matches
        
        Args:
            threat_matches: List of threat pattern matches
            aggression_matches: List of aggression pattern matches
            drug_matches: List of drug-related pattern matches
            bar_matches: List of bar-specific pattern matches
            intoxication_matches: List of intoxication-related pattern matches
            underage_matches: List of underage-related pattern matches
            
        Returns:
            Incident type as string
        """
        # Count matches by category
        threat_count = len(threat_matches)
        aggression_count = len(aggression_matches)
        drug_count = len(drug_matches)
        intoxication_count = len(intoxication_matches)
        underage_count = len(underage_matches)
        
        # Determine primary incident type
        if underage_count > 0 and underage_count >= intoxication_count:
            return "Potential Underage Drinking"
        elif intoxication_count > 2:
            return "Excessive Intoxication"
        elif threat_count > aggression_count and threat_count > drug_count:
            if any(word in " ".join(threat_matches).lower() for word in ["gun", "knife", "weapon"]):
                return "Potential Weapon Threat in Bar"
            elif any(word in " ".join(threat_matches).lower() for word in ["rob", "steal", "robbery"]):
                return "Potential Robbery in Bar"
            else:
                return "Verbal Threat in Bar"
        elif drug_count > aggression_count:
            return "Drug-Related Activity in Bar"
        elif aggression_count > 0:
            return "Aggressive Behavior in Bar"
        else:
            return "Suspicious Bar Activity"


def monitor_bar_directory(directory: str, 
                         output_dir: str = "bar_security",
                         min_threat_level: int = 2,
                         extensions: List[str] = [".wav", ".mp3", ".flac", ".m4a"]) -> None:
    """
    Monitor a directory for new bar audio files and process them
    
    Args:
        directory: Directory to monitor
        output_dir: Directory to save incident reports
        min_threat_level: Minimum threat level to report (1-5 scale)
        extensions: List of audio file extensions to process
    """
    monitor = BarSecurityMonitor(output_dir=output_dir, min_threat_level=min_threat_level)
    
    logger.info(f"Monitoring bar directory: {directory}")
    logger.info(f"Looking for files with extensions: {', '.join(extensions)}")
    
    # Process all existing files
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            file_path = os.path.join(directory, filename)
            logger.info(f"Processing bar audio file: {filename}")
            incident = monitor.process_audio_file(file_path)
            
            if incident:
                logger.warning(f"⚠️ Bar security incident detected in {filename}!")
                logger.info(str(incident))
            else:
                logger.info(f"No bar security concerns detected in {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bar Security Monitor for Audio Analysis")
    parser.add_argument("--input", "-i", required=True, help="Input audio file or directory")
    parser.add_argument("--output", "-o", default="bar_security", help="Output directory for incident reports")
    parser.add_argument("--threat-level", "-t", type=int, default=2, help="Minimum threat level to report (1-5)")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        monitor_bar_directory(args.input, args.output, args.threat_level)
    elif os.path.isfile(args.input):
        monitor = BarSecurityMonitor(output_dir=args.output, min_threat_level=args.threat_level)
        incident = monitor.process_audio_file(args.input)
        
        if incident:
            logger.warning("⚠️ Bar security incident detected!")
            print(str(incident))
        else:
            logger.info("No bar security concerns detected")
    else:
        logger.error(f"Input not found: {args.input}")