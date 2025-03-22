#!/bin/bash
# Bar Audio Security Monitor
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
echo -e "${PURPLE}║${YELLOW}                 BAR SECURITY MONITOR                     ${PURPLE}║${NC}"
echo -e "${PURPLE}║${BLUE}                 Aye & Hue @ 8b.is (2025)                ${PURPLE}║${NC}"
echo -e "${PURPLE}╚════════════════════════════════════════════════════════════╝${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/bar_security"
mkdir -p "$OUTPUT_DIR"

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
    
    # Process each audio file
    find "$INPUT_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.m4a" \) | while read -r file; do
        filename=$(basename "$file")
        echo -e "\n${BLUE}Processing: $filename${NC}"
        
        # Run the audio pipeline
        python "$PROJECT_ROOT/app.py" --audio "$file" --output "$OUTPUT_DIR/${filename%.*}.json" --task transcribe
        
        # Check if processing was successful
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully processed: $filename${NC}"
            
            # Analyze the transcription for security concerns
            echo -e "${YELLOW}Analyzing transcription for security concerns...${NC}"
            
            # Simple security analysis using grep
            CONCERNS=0
            
            # Check for weapon-related terms
            if grep -i -q -E "gun|knife|weapon|armed|shoot|stab" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${RED}⚠️ ALERT: Potential weapon references detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Check for threat-related terms
            if grep -i -q -E "kill|hurt|attack|fight|beat|hit|threat|threaten" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${RED}⚠️ ALERT: Potential threats detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Check for robbery-related terms
            if grep -i -q -E "robbery|steal|rob|wallet|money|purse|take your|give me your" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${RED}⚠️ ALERT: Potential robbery references detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Check for aggression-related terms
            if grep -i -q -E "fuck|shit|bitch|asshole|angry|mad|pissed|furious|shut up|back off" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${YELLOW}⚠️ Aggressive language detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Check for drug-related terms
            if grep -i -q -E "drugs|cocaine|meth|pills|molly|ecstasy|weed|pot|marijuana|dealer|selling" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${YELLOW}⚠️ Drug-related references detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Check for security-related terms
            if grep -i -q -E "cops|police|security|bouncer|guard" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${YELLOW}⚠️ Security-related references detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Generate a concern level
            if [ $CONCERNS -eq 0 ]; then
                echo -e "${GREEN}✅ No security concerns detected.${NC}"
            elif [ $CONCERNS -eq 1 ]; then
                echo -e "${YELLOW}⚠️ Low security concern level (1/5)${NC}"
            elif [ $CONCERNS -eq 2 ]; then
                echo -e "${YELLOW}⚠️⚠️ Moderate security concern level (2/5)${NC}"
            elif [ $CONCERNS -eq 3 ]; then
                echo -e "${RED}⚠️⚠️⚠️ High security concern level (3/5)${NC}"
            elif [ $CONCERNS -eq 4 ]; then
                echo -e "${RED}⚠️⚠️⚠️⚠️ Very high security concern level (4/5)${NC}"
            else
                echo -e "${RED}⚠️⚠️⚠️⚠️⚠️ CRITICAL security concern level (5/5)${NC}"
                echo -e "${RED}IMMEDIATE ATTENTION REQUIRED!${NC}"
            fi
            
            # Extract speakers
            SPEAKERS=$(grep -o '"speaker_name": "[^"]*"' "$OUTPUT_DIR/${filename%.*}.json" | sort | uniq | cut -d'"' -f4)
            
            if [ -n "$SPEAKERS" ]; then
                echo -e "${BLUE}Speakers identified:${NC}"
                echo "$SPEAKERS" | while read -r speaker; do
                    echo -e "  - $speaker"
                done
            else
                echo -e "${YELLOW}No named speakers identified.${NC}"
            fi
            
            # Create a simple security report
            REPORT_FILE="$OUTPUT_DIR/${filename%.*}_security_report.txt"
            {
                echo "BAR SECURITY REPORT"
                echo "==================="
                echo ""
                echo "Audio File: $filename"
                echo "Date: $(date)"
                echo ""
                echo "Security Concern Level: $CONCERNS/5"
                echo ""
                echo "Potential Concerns:"
                
                if grep -i -q -E "gun|knife|weapon|armed|shoot|stab" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Weapon references"
                fi
                
                if grep -i -q -E "kill|hurt|attack|fight|beat|hit|threat|threaten" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Threats"
                fi
                
                if grep -i -q -E "robbery|steal|rob|wallet|money|purse|take your|give me your" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Robbery references"
                fi
                
                if grep -i -q -E "fuck|shit|bitch|asshole|angry|mad|pissed|furious|shut up|back off" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Aggressive language"
                fi
                
                if grep -i -q -E "drugs|cocaine|meth|pills|molly|ecstasy|weed|pot|marijuana|dealer|selling" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Drug-related references"
                fi
                
                if grep -i -q -E "cops|police|security|bouncer|guard" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Security-related references"
                fi
                
                echo ""
                echo "Speakers:"
                if [ -n "$SPEAKERS" ]; then
                    echo "$SPEAKERS" | while read -r speaker; do
                        echo "- $speaker"
                    done
                else
                    echo "- No named speakers identified"
                fi
                
                echo ""
                echo "Relevant Segments:"
                if [ $CONCERNS -gt 0 ]; then
                    # Extract segments with concerning content
                    grep -B 2 -A 2 -i -E "gun|knife|weapon|kill|hurt|attack|fight|robbery|steal|rob|angry|mad|drugs|cocaine|meth|weed|cops|police|security" "$OUTPUT_DIR/${filename%.*}.json" | grep -E '"text"|"speaker_name"|"start"|"end"' | sed 's/^[[:space:]]*//'
                else
                    echo "- No concerning segments found"
                fi
            } > "$REPORT_FILE"
            
            echo -e "${GREEN}Security report saved to: $REPORT_FILE${NC}"
            
        else
            echo -e "${RED}Failed to process: $filename${NC}"
        fi
    done
else
    # Process a single file or directory
    INPUT="$1"
    
    if [ -f "$INPUT" ]; then
        # Process a single file
        filename=$(basename "$INPUT")
        echo -e "\n${BLUE}Processing: $filename${NC}"
        
        # Run the audio pipeline
        python "$PROJECT_ROOT/app.py" --audio "$INPUT" --output "$OUTPUT_DIR/${filename%.*}.json" --task transcribe
        
        # Check if processing was successful
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully processed: $filename${NC}"
            
            # Analyze the transcription for security concerns
            echo -e "${YELLOW}Analyzing transcription for security concerns...${NC}"
            
            # Simple security analysis using grep
            CONCERNS=0
            
            # Check for weapon-related terms
            if grep -i -q -E "gun|knife|weapon|armed|shoot|stab" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${RED}⚠️ ALERT: Potential weapon references detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Check for threat-related terms
            if grep -i -q -E "kill|hurt|attack|fight|beat|hit|threat|threaten" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${RED}⚠️ ALERT: Potential threats detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Check for robbery-related terms
            if grep -i -q -E "robbery|steal|rob|wallet|money|purse|take your|give me your" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${RED}⚠️ ALERT: Potential robbery references detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Check for aggression-related terms
            if grep -i -q -E "fuck|shit|bitch|asshole|angry|mad|pissed|furious|shut up|back off" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${YELLOW}⚠️ Aggressive language detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Check for drug-related terms
            if grep -i -q -E "drugs|cocaine|meth|pills|molly|ecstasy|weed|pot|marijuana|dealer|selling" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${YELLOW}⚠️ Drug-related references detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Check for security-related terms
            if grep -i -q -E "cops|police|security|bouncer|guard" "$OUTPUT_DIR/${filename%.*}.json"; then
                echo -e "${YELLOW}⚠️ Security-related references detected!${NC}"
                CONCERNS=$((CONCERNS + 1))
            fi
            
            # Generate a concern level
            if [ $CONCERNS -eq 0 ]; then
                echo -e "${GREEN}✅ No security concerns detected.${NC}"
            elif [ $CONCERNS -eq 1 ]; then
                echo -e "${YELLOW}⚠️ Low security concern level (1/5)${NC}"
            elif [ $CONCERNS -eq 2 ]; then
                echo -e "${YELLOW}⚠️⚠️ Moderate security concern level (2/5)${NC}"
            elif [ $CONCERNS -eq 3 ]; then
                echo -e "${RED}⚠️⚠️⚠️ High security concern level (3/5)${NC}"
            elif [ $CONCERNS -eq 4 ]; then
                echo -e "${RED}⚠️⚠️⚠️⚠️ Very high security concern level (4/5)${NC}"
            else
                echo -e "${RED}⚠️⚠️⚠️⚠️⚠️ CRITICAL security concern level (5/5)${NC}"
                echo -e "${RED}IMMEDIATE ATTENTION REQUIRED!${NC}"
            fi
            
            # Extract speakers
            SPEAKERS=$(grep -o '"speaker_name": "[^"]*"' "$OUTPUT_DIR/${filename%.*}.json" | sort | uniq | cut -d'"' -f4)
            
            if [ -n "$SPEAKERS" ]; then
                echo -e "${BLUE}Speakers identified:${NC}"
                echo "$SPEAKERS" | while read -r speaker; do
                    echo -e "  - $speaker"
                done
            else
                echo -e "${YELLOW}No named speakers identified.${NC}"
            fi
            
            # Create a simple security report
            REPORT_FILE="$OUTPUT_DIR/${filename%.*}_security_report.txt"
            {
                echo "BAR SECURITY REPORT"
                echo "==================="
                echo ""
                echo "Audio File: $filename"
                echo "Date: $(date)"
                echo ""
                echo "Security Concern Level: $CONCERNS/5"
                echo ""
                echo "Potential Concerns:"
                
                if grep -i -q -E "gun|knife|weapon|armed|shoot|stab" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Weapon references"
                fi
                
                if grep -i -q -E "kill|hurt|attack|fight|beat|hit|threat|threaten" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Threats"
                fi
                
                if grep -i -q -E "robbery|steal|rob|wallet|money|purse|take your|give me your" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Robbery references"
                fi
                
                if grep -i -q -E "fuck|shit|bitch|asshole|angry|mad|pissed|furious|shut up|back off" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Aggressive language"
                fi
                
                if grep -i -q -E "drugs|cocaine|meth|pills|molly|ecstasy|weed|pot|marijuana|dealer|selling" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Drug-related references"
                fi
                
                if grep -i -q -E "cops|police|security|bouncer|guard" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo "- Security-related references"
                fi
                
                echo ""
                echo "Speakers:"
                if [ -n "$SPEAKERS" ]; then
                    echo "$SPEAKERS" | while read -r speaker; do
                        echo "- $speaker"
                    done
                else
                    echo "- No named speakers identified"
                fi
                
                echo ""
                echo "Relevant Segments:"
                if [ $CONCERNS -gt 0 ]; then
                    # Extract segments with concerning content
                    grep -B 2 -A 2 -i -E "gun|knife|weapon|kill|hurt|attack|fight|robbery|steal|rob|angry|mad|drugs|cocaine|meth|weed|cops|police|security" "$OUTPUT_DIR/${filename%.*}.json" | grep -E '"text"|"speaker_name"|"start"|"end"' | sed 's/^[[:space:]]*//'
                else
                    echo "- No concerning segments found"
                fi
            } > "$REPORT_FILE"
            
            echo -e "${GREEN}Security report saved to: $REPORT_FILE${NC}"
            
        else
            echo -e "${RED}Failed to process: $filename${NC}"
        fi
    elif [ -d "$INPUT" ]; then
        # Process all audio files in the directory
        echo -e "${BLUE}Processing all audio files in: $INPUT${NC}"
        
        # Count audio files
        AUDIO_FILES=$(find "$INPUT" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.m4a" \) | wc -l)
        
        if [ "$AUDIO_FILES" -eq 0 ]; then
            echo -e "${RED}Error: No audio files found in $INPUT${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}Found $AUDIO_FILES audio files.${NC}"
        
        # Process each audio file
        find "$INPUT" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.m4a" \) | while read -r file; do
            filename=$(basename "$file")
            echo -e "\n${BLUE}Processing: $filename${NC}"
            
            # Run the audio pipeline
            python "$PROJECT_ROOT/app.py" --audio "$file" --output "$OUTPUT_DIR/${filename%.*}.json" --task transcribe
            
            # Check if processing was successful
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Successfully processed: $filename${NC}"
                
                # Analyze the transcription for security concerns
                echo -e "${YELLOW}Analyzing transcription for security concerns...${NC}"
                
                # Simple security analysis using grep
                CONCERNS=0
                
                # Check for weapon-related terms
                if grep -i -q -E "gun|knife|weapon|armed|shoot|stab" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo -e "${RED}⚠️ ALERT: Potential weapon references detected!${NC}"
                    CONCERNS=$((CONCERNS + 1))
                fi
                
                # Check for threat-related terms
                if grep -i -q -E "kill|hurt|attack|fight|beat|hit|threat|threaten" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo -e "${RED}⚠️ ALERT: Potential threats detected!${NC}"
                    CONCERNS=$((CONCERNS + 1))
                fi
                
                # Check for robbery-related terms
                if grep -i -q -E "robbery|steal|rob|wallet|money|purse|take your|give me your" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo -e "${RED}⚠️ ALERT: Potential robbery references detected!${NC}"
                    CONCERNS=$((CONCERNS + 1))
                fi
                
                # Check for aggression-related terms
                if grep -i -q -E "fuck|shit|bitch|asshole|angry|mad|pissed|furious|shut up|back off" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo -e "${YELLOW}⚠️ Aggressive language detected!${NC}"
                    CONCERNS=$((CONCERNS + 1))
                fi
                
                # Check for drug-related terms
                if grep -i -q -E "drugs|cocaine|meth|pills|molly|ecstasy|weed|pot|marijuana|dealer|selling" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo -e "${YELLOW}⚠️ Drug-related references detected!${NC}"
                    CONCERNS=$((CONCERNS + 1))
                fi
                
                # Check for security-related terms
                if grep -i -q -E "cops|police|security|bouncer|guard" "$OUTPUT_DIR/${filename%.*}.json"; then
                    echo -e "${YELLOW}⚠️ Security-related references detected!${NC}"
                    CONCERNS=$((CONCERNS + 1))
                fi
                
                # Generate a concern level
                if [ $CONCERNS -eq 0 ]; then
                    echo -e "${GREEN}✅ No security concerns detected.${NC}"
                elif [ $CONCERNS -eq 1 ]; then
                    echo -e "${YELLOW}⚠️ Low security concern level (1/5)${NC}"
                elif [ $CONCERNS -eq 2 ]; then
                    echo -e "${YELLOW}⚠️⚠️ Moderate security concern level (2/5)${NC}"
                elif [ $CONCERNS -eq 3 ]; then
                    echo -e "${RED}⚠️⚠️⚠️ High security concern level (3/5)${NC}"
                elif [ $CONCERNS -eq 4 ]; then
                    echo -e "${RED}⚠️⚠️⚠️⚠️ Very high security concern level (4/5)${NC}"
                else
                    echo -e "${RED}⚠️⚠️⚠️⚠️⚠️ CRITICAL security concern level (5/5)${NC}"
                    echo -e "${RED}IMMEDIATE ATTENTION REQUIRED!${NC}"
                fi
                
                # Extract speakers
                SPEAKERS=$(grep -o '"speaker_name": "[^"]*"' "$OUTPUT_DIR/${filename%.*}.json" | sort | uniq | cut -d'"' -f4)
                
                if [ -n "$SPEAKERS" ]; then
                    echo -e "${BLUE}Speakers identified:${NC}"
                    echo "$SPEAKERS" | while read -r speaker; do
                        echo -e "  - $speaker"
                    done
                else
                    echo -e "${YELLOW}No named speakers identified.${NC}"
                fi
                
                # Create a simple security report
                REPORT_FILE="$OUTPUT_DIR/${filename%.*}_security_report.txt"
                {
                    echo "BAR SECURITY REPORT"
                    echo "==================="
                    echo ""
                    echo "Audio File: $filename"
                    echo "Date: $(date)"
                    echo ""
                    echo "Security Concern Level: $CONCERNS/5"
                    echo ""
                    echo "Potential Concerns:"
                    
                    if grep -i -q -E "gun|knife|weapon|armed|shoot|stab" "$OUTPUT_DIR/${filename%.*}.json"; then
                        echo "- Weapon references"
                    fi
                    
                    if grep -i -q -E "kill|hurt|attack|fight|beat|hit|threat|threaten" "$OUTPUT_DIR/${filename%.*}.json"; then
                        echo "- Threats"
                    fi
                    
                    if grep -i -q -E "robbery|steal|rob|wallet|money|purse|take your|give me your" "$OUTPUT_DIR/${filename%.*}.json"; then
                        echo "- Robbery references"
                    fi
                    
                    if grep -i -q -E "fuck|shit|bitch|asshole|angry|mad|pissed|furious|shut up|back off" "$OUTPUT_DIR/${filename%.*}.json"; then
                        echo "- Aggressive language"
                    fi
                    
                    if grep -i -q -E "drugs|cocaine|meth|pills|molly|ecstasy|weed|pot|marijuana|dealer|selling" "$OUTPUT_DIR/${filename%.*}.json"; then
                        echo "- Drug-related references"
                    fi
                    
                    if grep -i -q -E "cops|police|security|bouncer|guard" "$OUTPUT_DIR/${filename%.*}.json"; then
                        echo "- Security-related references"
                    fi
                    
                    echo ""
                    echo "Speakers:"
                    if [ -n "$SPEAKERS" ]; then
                        echo "$SPEAKERS" | while read -r speaker; do
                            echo "- $speaker"
                        done
                    else
                        echo "- No named speakers identified"
                    fi
                    
                    echo ""
                    echo "Relevant Segments:"
                    if [ $CONCERNS -gt 0 ]; then
                        # Extract segments with concerning content
                        grep -B 2 -A 2 -i -E "gun|knife|weapon|kill|hurt|attack|fight|robbery|steal|rob|angry|mad|drugs|cocaine|meth|weed|cops|police|security" "$OUTPUT_DIR/${filename%.*}.json" | grep -E '"text"|"speaker_name"|"start"|"end"' | sed 's/^[[:space:]]*//'
                    else
                        echo "- No concerning segments found"
                    fi
                } > "$REPORT_FILE"
                
                echo -e "${GREEN}Security report saved to: $REPORT_FILE${NC}"
                
            else
                echo -e "${RED}Failed to process: $filename${NC}"
            fi
        done
    else
        echo -e "${RED}Error: Input not found: $INPUT${NC}"
        exit 1
    fi
fi

echo -e "\n${GREEN}Bar security monitoring complete!${NC}"
echo -e "${BLUE}Security reports saved to: $OUTPUT_DIR${NC}"

# Summary of results
TOTAL_FILES=$(find "$OUTPUT_DIR" -name "*_security_report.txt" | wc -l)
HIGH_CONCERN=$(grep -l "Security Concern Level: [3-5]/5" "$OUTPUT_DIR"/*_security_report.txt 2>/dev/null | wc -l)

echo -e "\n${PURPLE}SUMMARY${NC}"
echo -e "${BLUE}Total files processed: $TOTAL_FILES${NC}"
echo -e "${YELLOW}High concern files: $HIGH_CONCERN${NC}"

if [ $HIGH_CONCERN -gt 0 ]; then
    echo -e "\n${RED}⚠️ HIGH CONCERN FILES:${NC}"
    grep -l "Security Concern Level: [3-5]/5" "$OUTPUT_DIR"/*_security_report.txt 2>/dev/null | while read -r file; do
        basename "$file"
    done
fi
