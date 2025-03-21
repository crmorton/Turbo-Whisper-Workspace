"""
LLM Helper Module for CyberVox

This module provides LLM-powered features for CyberVox:
1. Enhanced speaker name recognition
2. Conversation summarization
3. Topic extraction
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import llama_cpp, but provide fallbacks if not available
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    logger.warning("llama-cpp-python not available. Using fallback methods for name recognition and summarization.")
    LLAMA_AVAILABLE = False

# Path to the model file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Qwen2.5-Dyanka-7B-Preview.IQ4_XS.gguf")
MODEL_REPO = "mradermacher/Qwen2.5-Dyanka-7B-Preview-GGUF"
MODEL_FILENAME = "Qwen2.5-Dyanka-7B-Preview.IQ4_XS.gguf"

# LLM instance (lazy-loaded)
_llm_instance = None

def get_llm() -> Optional[Any]:
    """Get or initialize the LLM instance"""
    global _llm_instance
    
    if not LLAMA_AVAILABLE:
        return None
        
    if _llm_instance is None:
        try:
            # Check if model exists locally
            if os.path.exists(MODEL_PATH):
                logger.info(f"Loading model from local path: {MODEL_PATH}")
                _llm_instance = Llama(model_path=MODEL_PATH)
            else:
                # Create models directory if it doesn't exist
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                
                # Download from Hugging Face
                logger.info(f"Downloading model from {MODEL_REPO}")
                _llm_instance = Llama.from_pretrained(
                    repo_id=MODEL_REPO,
                    filename=MODEL_FILENAME,
                    local_dir=os.path.dirname(MODEL_PATH)
                )
                
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return None
            
    return _llm_instance

def identify_speaker_names_llm(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Use LLM to identify speaker names from conversation segments
    
    Args:
        segments: List of conversation segments with speaker and text
        
    Returns:
        Dictionary mapping speaker IDs to names
    """
    llm = get_llm()
    if not llm:
        logger.warning("LLM not available, using fallback method")
        return {}
    
    # Prepare conversation context
    conversation_text = ""
    for segment in segments[:10]:  # Use first 10 segments to identify speakers
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '').strip()
        conversation_text += f"{speaker}: {text}\n"
    
    # Create prompt for the LLM
    prompt = f"""
You are an AI assistant that analyzes conversations and identifies the real names of speakers.

Here's a conversation with speakers labeled as "Speaker X":
{conversation_text}

Based on this conversation, identify the real names of the speakers. Look for:
1. Self-introductions like "I'm [Name]" or "My name is [Name]"
2. When one speaker addresses another by name
3. Any other context clues that reveal names

Return your answer as a JSON object mapping speaker IDs to names.
Example: {{"Speaker 0": "John", "Speaker 1": "Mary"}}

If you can't identify a name for a speaker, use null instead of making up a name.
"""

    try:
        # Get response from LLM
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,
            stop=["```"]
        )
        
        # Extract and parse the JSON response
        response_text = response['choices'][0]['text'].strip()
        
        # Find JSON block in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                speaker_names = json.loads(json_str)
                return speaker_names
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {json_str}")
        
        return {}
    except Exception as e:
        logger.error(f"Error getting speaker names from LLM: {e}")
        return {}

def summarize_conversation(segments: List[Dict[str, Any]]) -> str:
    """
    Generate a summary of the conversation using the LLM
    
    Args:
        segments: List of conversation segments
        
    Returns:
        Summary text
    """
    llm = get_llm()
    if not llm:
        return "Conversation summary not available (LLM not loaded)"
    
    # Prepare conversation context
    conversation_text = ""
    for segment in segments:
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '').strip()
        conversation_text += f"{speaker}: {text}\n"
    
    # Create prompt for the LLM
    prompt = f"""
You are an AI assistant that summarizes conversations.

Here's a conversation:
{conversation_text}

Please provide a concise summary (3-5 sentences) of this conversation, highlighting:
1. The main topics discussed
2. Any decisions or conclusions reached
3. Any action items or next steps mentioned

Summary:
"""

    try:
        # Get response from LLM
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.3,
            stop=["```"]
        )
        
        # Extract the summary text
        summary = response['choices'][0]['text'].strip()
        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Error generating summary"

def extract_topics(segments: List[Dict[str, Any]]) -> List[str]:
    """
    Extract main topics from the conversation
    
    Args:
        segments: List of conversation segments
        
    Returns:
        List of main topics
    """
    llm = get_llm()
    if not llm:
        return []
    
    # Prepare conversation context
    conversation_text = ""
    for segment in segments:
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '').strip()
        conversation_text += f"{speaker}: {text}\n"
    
    # Create prompt for the LLM
    prompt = f"""
You are an AI assistant that analyzes conversations and extracts the main topics.

Here's a conversation:
{conversation_text}

Please list the 3-5 main topics discussed in this conversation. Return your answer as a JSON array of strings.
Example: ["Project timeline", "Budget concerns", "Marketing strategy"]
"""

    try:
        # Get response from LLM
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,
            stop=["```"]
        )
        
        # Extract and parse the JSON response
        response_text = response['choices'][0]['text'].strip()
        
        # Find JSON block in the response
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                topics = json.loads(json_str)
                return topics
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM response: {json_str}")
        
        return []
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        return []

# Fallback methods that don't require LLM
def identify_speaker_names_fallback(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    """Fallback method for speaker name identification without LLM"""
    from app import COMMON_NAMES
    
    # Dictionary to store detected names for each speaker ID
    detected_names = {}
    name_mentions = {}
    
    # First pass: find potential speaker names in the text
    for segment in segments:
        speaker_id = segment['speaker']
        text = segment['text']
        
        # Extract names from text
        # Look for common name patterns directly
        for name in COMMON_NAMES:
            # Look for the name as a whole word with word boundaries
            import re
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                # Count name mentions
                if name not in name_mentions:
                    name_mentions[name] = 0
                name_mentions[name] += 1
        
        # Look for "I'm [Name]" or "My name is [Name]" patterns
        name_intro_patterns = [
            r"I'?m\s+(\w+)",
            r"[Mm]y name is\s+(\w+)",
            r"[Cc]all me\s+(\w+)",
            r"[Tt]his is\s+(\w+)"  
        ]
        
        for pattern in name_intro_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match in COMMON_NAMES:
                    if speaker_id not in detected_names:
                        detected_names[speaker_id] = match
                    elif detected_names[speaker_id] != match:
                        # If multiple names are detected for the same speaker, keep the most frequent one
                        if match not in name_mentions:
                            name_mentions[match] = 0
                        name_mentions[match] += 3  # Give higher weight to explicit introduction
    
    # Second pass: assign names to speakers based on frequency and context
    for speaker_id in set([segment['speaker'] for segment in segments]):
        if speaker_id not in detected_names:
            # Find the most mentioned name that hasn't been assigned yet
            available_names = [name for name, count in sorted(name_mentions.items(), key=lambda x: x[1], reverse=True) 
                             if name not in detected_names.values()]                
            if available_names:
                detected_names[speaker_id] = available_names[0]
    
    return detected_names
