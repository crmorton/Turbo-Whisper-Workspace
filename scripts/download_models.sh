#!/bin/bash
# CyberVox Model Downloader
# Downloads all required speaker embedding models for CyberVox
# and stores them locally for reliable access

# Set colors for pretty output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
echo "  ██████╗██╗   ██╗██████╗ ███████╗██████╗ ██╗   ██╗ ██████╗ ██╗  ██╗"
echo " ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██║   ██║██╔═══██╗╚██╗██╔╝"
echo " ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝██║   ██║██║   ██║ ╚███╔╝ "
echo " ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗╚██╗ ██╔╝██║   ██║ ██╔██╗ "
echo " ╚██████╗   ██║   ██████╔╝███████╗██║  ██║ ╚████╔╝ ╚██████╔╝██╔╝ ██╗"
echo "  ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝"
echo -e "${GREEN}                 Model Downloader v1.0${NC}"
echo -e "${YELLOW}       Ensuring your voice models are always available${NC}\n"

# Set variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$ROOT_DIR/models"
FORCE_DOWNLOAD=false
PYTHON_SCRIPT="$ROOT_DIR/download_models.py"

# Parse arguments
for arg in "$@"; do
  case $arg in
    --force)
      FORCE_DOWNLOAD=true
      shift
      ;;
    --help)
      echo -e "Usage: $0 [OPTIONS]"
      echo -e "Options:"
      echo -e "  --force    Force download even if models already exist"
      echo -e "  --help     Show this help message"
      exit 0
      ;;
  esac
done

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

echo -e "${GREEN}🔍 Checking for existing models in:${NC} $MODELS_DIR"

# Check if we have the Python script
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo -e "${RED}❌ Error: Python download script not found at $PYTHON_SCRIPT${NC}"
  exit 1
fi

# Run the Python script to download models
echo -e "${GREEN}🚀 Starting model download process...${NC}"

if [ "$FORCE_DOWNLOAD" = true ]; then
  echo -e "${YELLOW}⚠️ Force download enabled - will redownload all models${NC}"
  python "$PYTHON_SCRIPT" --force
else
  python "$PYTHON_SCRIPT"
fi

# Check if download was successful
if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Error: Model download failed${NC}"
  exit 1
fi

# List downloaded models
echo -e "\n${GREEN}📋 Downloaded models:${NC}"
find "$MODELS_DIR" -type f | while read -r model; do
  model_name=$(basename "$model")
  model_size=$(du -h "$model" | cut -f1)
  echo -e "${CYAN}▶ ${model_name}${NC} (${model_size})"
done

echo -e "\n${GREEN}✅ All models downloaded successfully!${NC}"
echo -e "${YELLOW}CyberVox will now use these local models instead of downloading them at runtime.${NC}"
echo -e "${CYAN}Run this script periodically to ensure you have the latest models.${NC}\n"
