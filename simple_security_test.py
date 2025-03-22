#!/usr/bin/env python3
"""
Simple Security Monitor Test

This script demonstrates the basic functionality of the security monitoring system
using a mock conversation in JSON format.

Author: Aye & Hue @ 8b.is (2025)
"""

import os
import json
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("simple_security_test")

# Colorful terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

def find_pattern_matches(text, patterns):
    """Find all matches for the given patterns in the text"""
    import re
    matches = []
    for pattern in patterns:
        found = re.finditer(pattern, text, re.IGNORECASE)
        for match in found:
            matches.append(match.group(0))
    return matches

def calculate_threat_level(threat_matches, aggression_matches, drug_matches):
    """Calculate threat level on a scale of 1-5"""
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

def determine_incident_type(threat_matches, aggression_matches, drug_matches):
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

def find_relevant_segments(segments, pattern_matches):
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

def analyze_mock_conversation():
    """Analyze a mock bar conversation for security concerns"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'ANALYZING MOCK BAR CONVERSATION FOR SECURITY CONCERNS'.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")
    
    # Load mock data
    mock_file = "examples/mock_bar_conversation.json"
    
    try:
        with open(mock_file, 'r') as f:
            mock_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load mock data: {e}")
        return
    
    logger.info(f"Loaded mock data from {mock_file}")
    
    # Extract segments and full transcript
    segments = mock_data["segments"]
    full_transcript = mock_data["text"]
    
    # Check for concerning patterns
    threat_matches = find_pattern_matches(full_transcript, THREAT_PATTERNS)
    aggression_matches = find_pattern_matches(full_transcript, AGGRESSION_INDICATORS)
    drug_matches = find_pattern_matches(full_transcript, DRUG_INDICATORS)
    
    # Print matches
    print(f"{Colors.BLUE}Threat matches:{Colors.ENDC} {threat_matches}")
    print(f"{Colors.BLUE}Aggression matches:{Colors.ENDC} {aggression_matches}")
    print(f"{Colors.BLUE}Drug matches:{Colors.ENDC} {drug_matches}")
    
    # Calculate threat level
    threat_level = calculate_threat_level(threat_matches, aggression_matches, drug_matches)
    print(f"\n{Colors.YELLOW}Calculated threat level: {'⚠️' * threat_level} ({threat_level}/5){Colors.ENDC}")
    
    # Determine incident type
    incident_type = determine_incident_type(threat_matches, aggression_matches, drug_matches)
    print(f"{Colors.YELLOW}Determined incident type: {incident_type}{Colors.ENDC}")
    
    # Find relevant segments
    all_matches = threat_matches + aggression_matches + drug_matches
    relevant_segments = find_relevant_segments(segments, all_matches)
    
    # Print relevant segments
    print(f"\n{Colors.BOLD}Key Conversation Segments:{Colors.ENDC}")
    for segment in relevant_segments:
        speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
        text = segment.get("text", "")
        time = f"{segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s"
        print(f"[{time}] {Colors.BOLD}{speaker}:{Colors.ENDC} {text}")
    
    # Generate a simple summary
    speakers_involved = list(set([s.get("speaker_name", s.get("speaker", "Unknown")) 
                             for s in relevant_segments]))
    
    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"A potential {incident_type.lower()} situation was detected involving {', '.join(speakers_involved)}.")
    print(f"The conversation contains concerning language related to: {', '.join(all_matches[:5])}.")
    print(f"Security staff should monitor this situation closely.")
    
    # Create output directory
    output_dir = "security_incidents"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a simple report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/incident_{timestamp}.txt"
    
    with open(report_file, "w") as f:
        f.write(f"SECURITY INCIDENT REPORT\n")
        f.write(f"=======================\n\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Incident Type: {incident_type}\n")
        f.write(f"Threat Level: {threat_level}/5\n")
        f.write(f"Speakers Involved: {', '.join(speakers_involved)}\n\n")
        f.write(f"Summary:\n")
        f.write(f"A potential {incident_type.lower()} situation was detected involving {', '.join(speakers_involved)}.\n")
        f.write(f"The conversation contains concerning language related to: {', '.join(all_matches[:5])}.\n")
        f.write(f"Security staff should monitor this situation closely.\n\n")
        f.write(f"Key Conversation Segments:\n")
        for segment in relevant_segments:
            speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
            text = segment.get("text", "")
            time = f"{segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s"
            f.write(f"[{time}] {speaker}: {text}\n")
    
    print(f"\n{Colors.GREEN}Report saved to: {report_file}{Colors.ENDC}")

if __name__ == "__main__":
    analyze_mock_conversation()
    print(f"\n{Colors.GREEN}{Colors.BOLD}Analysis complete!{Colors.ENDC}")
    print(f"{Colors.BLUE}You can now use the security monitor with real audio files.{Colors.ENDC}")
    print(f"{Colors.BLUE}Run: python security_monitor.py --input /path/to/audio/file{Colors.ENDC}")
