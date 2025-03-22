#!/bin/bash
# Run the security monitor on a sample audio file
# Author: Aye & Hue @ 8b.is
# Date: 2025-03-22

# Colorful output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║${YELLOW}                 SECURITY MONITOR LAUNCHER                 ${PURPLE}║${NC}"
echo -e "${PURPLE}║${BLUE}                 Aye & Hue @ 8b.is (2025)                ${PURPLE}║${NC}"
echo -e "${PURPLE}╚════════════════════════════════════════════════════════════╝${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create output directory
SECURITY_DIR="$PROJECT_ROOT/security_incidents"
mkdir -p "$SECURITY_DIR"

# Check if input file or directory is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}No input specified. Looking for audio files in examples directory...${NC}"
    INPUT_DIR="$PROJECT_ROOT/examples"
    
    # Check if examples directory exists
    if [ ! -d "$INPUT_DIR" ]; then
        echo -e "${RED}Error: examples directory not found!${NC}"
        echo -e "${YELLOW}Usage: $0 [audio_file_or_directory]${NC}"
        exit 1
    fi
    
    # Count audio files in examples directory
    AUDIO_FILES=$(find "$INPUT_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.m4a" \) | wc -l)
    
    if [ "$AUDIO_FILES" -eq 0 ]; then
        echo -e "${RED}Error: No audio files found in examples directory!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Found $AUDIO_FILES audio files in examples directory.${NC}"
    INPUT="$INPUT_DIR"
else
    INPUT="$1"
    
    # Check if input exists
    if [ ! -e "$INPUT" ]; then
        echo -e "${RED}Error: Input not found: $INPUT${NC}"
        exit 1
    fi
fi

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Run the security monitor
echo -e "${GREEN}Running security monitor on: $INPUT${NC}"
echo -e "${YELLOW}Reports will be saved to: $SECURITY_DIR${NC}"
echo ""

python "$PROJECT_ROOT/security_monitor.py" --input "$INPUT" --output "$SECURITY_DIR"

# Check exit status
if [ $? -ne 0 ]; then
    echo -e "${RED}Error running security monitor!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Security monitoring complete!${NC}"

# Count incident reports
INCIDENT_COUNT=$(find "$SECURITY_DIR" -type f -name "incident_*.json" | wc -l)

if [ "$INCIDENT_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}Found $INCIDENT_COUNT security incidents.${NC}"
    echo -e "${BLUE}Check $SECURITY_DIR for detailed reports.${NC}"
    
    # List the most recent incidents
    echo ""
    echo -e "${PURPLE}Recent incidents:${NC}"
    ls -lt "$SECURITY_DIR" | grep "incident_" | head -5
else
    echo -e "${GREEN}No security incidents detected.${NC}"
fi

# Deactivate virtual environment if it was activated
if [ -d "$PROJECT_ROOT/.venv" ]; then
    deactivate 2>/dev/null
fi
