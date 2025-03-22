#!/usr/bin/env python3
"""
Test script for the Security Monitor

This script tests the security monitor with mock data to demonstrate
its capabilities for detecting potential security incidents in bar audio.

Author: Aye & Hue @ 8b.is
Date: 2025-03-22
"""

import os
import json
import logging
import sys
from security_monitor import SecurityMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_security_monitor")

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

def print_header(text):
    """Print a colorful header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

def test_with_mock_data():
    """Test the security monitor with mock data"""
    print_header("TESTING SECURITY MONITOR WITH MOCK BAR CONVERSATION")
    
    # Create output directory
    output_dir = "test_security_incidents"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize security monitor
    monitor = SecurityMonitor(output_dir=output_dir, min_threat_level=2)
    
    # Load mock data
    mock_file = "examples/mock_bar_conversation.json"
    
    try:
        with open(mock_file, 'r') as f:
            mock_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load mock data: {e}")
        return
    
    logger.info(f"Loaded mock data from {mock_file}")
    
    # Create a mock audio pipeline result
    mock_result = {
        "segments": mock_data["segments"],
        "text": mock_data["text"]
    }
    
    # Mock the audio file path
    mock_audio_file = "examples/mock_bar_audio.flac"
    
    # Analyze the mock transcript
    print(f"{Colors.BLUE}Analyzing mock bar conversation...{Colors.ENDC}")
    incident = monitor._analyze_transcript(mock_audio_file, mock_result)
    
    if incident:
        print(f"\n{Colors.RED}{Colors.BOLD}🚨 SECURITY INCIDENT DETECTED! 🚨{Colors.ENDC}")
        print(f"{Colors.YELLOW}Threat Level: {'⚠️' * incident.threat_level} ({incident.threat_level}/5){Colors.ENDC}")
        print(f"{Colors.YELLOW}Incident Type: {incident.incident_type}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Speakers Involved: {', '.join(incident.speakers_involved)}{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
        print(incident.summary)
        
        print(f"\n{Colors.BOLD}Key Conversation Segments:{Colors.ENDC}")
        for segment in incident.relevant_segments:
            speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
            text = segment.get("text", "")
            time = f"{segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s"
            print(f"[{time}] {Colors.BOLD}{speaker}:{Colors.ENDC} {text}")
        
        print(f"\n{Colors.GREEN}Incident report saved to:{Colors.ENDC}")
        print(f"- {output_dir}/incident_*.json")
        print(f"- {output_dir}/incident_*.txt")
    else:
        print(f"\n{Colors.GREEN}No security concerns detected in the mock conversation.{Colors.ENDC}")

def test_pattern_matching():
    """Test the pattern matching functionality"""
    print_header("TESTING PATTERN MATCHING")
    
    # Initialize security monitor
    monitor = SecurityMonitor(min_threat_level=1)
    
    # Test cases
    test_cases = [
        {
            "name": "Threat Detection",
            "text": "I'm going to hurt you if you don't give me your wallet.",
            "expected_patterns": ["hurt", "give me your"]
        },
        {
            "name": "Weapon Detection",
            "text": "Back off or I'll pull out my knife.",
            "expected_patterns": ["knife", "weapon", "back off"]
        },
        {
            "name": "Drug Detection",
            "text": "Hey man, you looking to score some weed? I can hook you up.",
            "expected_patterns": ["score", "weed", "hook up"]
        },
        {
            "name": "Aggression Detection",
            "text": "Get the fuck away from me! I'm getting really pissed off.",
            "expected_patterns": ["fuck", "pissed", "get away"]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{Colors.BLUE}Testing: {test_case['name']}{Colors.ENDC}")
        print(f"Text: \"{test_case['text']}\"")
        
        # Import patterns from security_monitor module
        from security_monitor import THREAT_PATTERNS, AGGRESSION_INDICATORS, DRUG_INDICATORS
        
        # Test threat patterns
        threat_matches = monitor._find_pattern_matches(test_case['text'], THREAT_PATTERNS)
        print(f"Threat matches: {threat_matches}")
        
        # Test aggression patterns
        aggression_matches = monitor._find_pattern_matches(test_case['text'], AGGRESSION_INDICATORS)
        print(f"Aggression matches: {aggression_matches}")
        
        # Test drug patterns
        drug_matches = monitor._find_pattern_matches(test_case['text'], DRUG_INDICATORS)
        print(f"Drug matches: {drug_matches}")
        
        # Calculate threat level
        threat_level = monitor._calculate_threat_level(threat_matches, aggression_matches, drug_matches)
        print(f"Calculated threat level: {Colors.YELLOW}{'⚠️' * threat_level} ({threat_level}/5){Colors.ENDC}")
        
        # Determine incident type
        incident_type = monitor._determine_incident_type(threat_matches, aggression_matches, drug_matches)
        print(f"Determined incident type: {Colors.YELLOW}{incident_type}{Colors.ENDC}")
        
        # Check if expected patterns were found
        all_matches = threat_matches + aggression_matches + drug_matches
        found_expected = [pattern for pattern in test_case['expected_patterns'] 
                          if any(pattern.lower() in match.lower() for match in all_matches)]
        
        if len(found_expected) == len(test_case['expected_patterns']):
            print(f"{Colors.GREEN}✓ All expected patterns detected!{Colors.ENDC}")
        else:
            print(f"{Colors.RED}✗ Some expected patterns not detected!{Colors.ENDC}")
            print(f"  Expected: {test_case['expected_patterns']}")
            print(f"  Found: {found_expected}")
            print(f"  Missing: {set(test_case['expected_patterns']) - set(found_expected)}")

if __name__ == "__main__":
    # Run pattern matching tests
    test_pattern_matching()
    
    # Run mock data test
    test_with_mock_data()
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}All tests completed!{Colors.ENDC}")
    print(f"{Colors.BLUE}You can now use the security monitor with real audio files.{Colors.ENDC}")
    print(f"{Colors.BLUE}Run: python security_monitor.py --input /path/to/audio/file{Colors.ENDC}")
