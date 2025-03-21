#!/bin/bash

# Colors for cyberpunk terminal aesthetics
GREEN='\033[38;5;46m'
CYAN='\033[38;5;51m'
MAGENTA='\033[38;5;201m'
YELLOW='\033[38;5;226m'
RED='\033[38;5;196m'
RESET='\033[0m'
BOLD='\033[1m'

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHANGELOG_FILE="${PROJECT_ROOT}/CHANGELOG.md"

# Show banner
echo -e "${GREEN}${BOLD}"
echo "╔═══════════════════════════════════════════════════╗"
echo "║  ${CYAN}█▀█ █ █ █▀▄ █ █▀█   █▀▀ █▀█ █▄█ █▄█ █ ▀█▀${GREEN}  ║"
echo "║  ${CYAN}█▀█ █▄█ █▄▀ █ █▄█   █▄▄ █▄█ ░█░ ░█░ █ ░█░${GREEN}  ║"
echo "║                                               ║"
echo "║  ${MAGENTA}C O M M I T   M A N A G E R   v1.0${GREEN}        ║"
echo "╚═══════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Check if git is initialized
if [ ! -d "${PROJECT_ROOT}/.git" ]; then
    echo -e "${YELLOW}${BOLD}[!] Git repository not initialized. Initializing...${RESET}"
    cd "${PROJECT_ROOT}"
    git init
    echo -e "${GREEN}${BOLD}[✓] Git repository initialized${RESET}"
fi

# Create changelog file if it doesn't exist
if [ ! -f "$CHANGELOG_FILE" ]; then
    echo -e "${YELLOW}${BOLD}[!] Changelog file not found. Creating...${RESET}"
    echo "# Changelog" > "$CHANGELOG_FILE"
    echo "" >> "$CHANGELOG_FILE"
    echo "All notable changes to this project will be documented in this file." >> "$CHANGELOG_FILE"
    echo "" >> "$CHANGELOG_FILE"
    echo -e "${GREEN}${BOLD}[✓] Changelog file created at ${CHANGELOG_FILE}${RESET}"
fi

# Get current date in ISO format
DATE=$(date +"%Y-%m-%d")

# Ask for commit message
echo -e "${CYAN}${BOLD}[?] Enter a commit message:${RESET}"
read -r COMMIT_MESSAGE

# Ask for changelog entry
echo -e "${CYAN}${BOLD}[?] Enter changelog entry (leave empty to skip):${RESET}"
read -r CHANGELOG_ENTRY

# Add changelog entry if provided
if [ -n "$CHANGELOG_ENTRY" ]; then
    # Check if there's already an entry for today's date
    if grep -q "## \[$DATE\]" "$CHANGELOG_FILE"; then
        # Add to existing entry
        sed -i "/## \[$DATE\]/a - $CHANGELOG_ENTRY" "$CHANGELOG_FILE"
    else
        # Create new entry for today
        sed -i "4i ## [$DATE]\n\n- $CHANGELOG_ENTRY\n" "$CHANGELOG_FILE"
    fi
    echo -e "${GREEN}${BOLD}[✓] Changelog updated${RESET}"
fi

# Stage changes
echo -e "${CYAN}${BOLD}[*] Staging changes...${RESET}"
cd "${PROJECT_ROOT}"
git add .

# Commit changes
echo -e "${CYAN}${BOLD}[*] Committing changes...${RESET}"
git commit -m "$COMMIT_MESSAGE"

# Show success message
echo -e "${GREEN}${BOLD}[✓] Changes committed successfully!${RESET}"
echo -e "${YELLOW}${BOLD}[!] Don't forget to push your changes if needed.${RESET}"
