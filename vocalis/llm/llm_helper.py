"""
LLM Helper Module for Vocalis

This module provides LLM-based functionality for:
- Speaker name identification
- Conversation summarization
- Topic extraction
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union

# Import common data
from vocalis.utils.common_data import COMMON_NAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_helper")

# Global LLM instance
_LLM_INSTANCE = None

def get_llm():
    """Get or initialize the LLM instance"""
    global _LLM_INSTANCE
    
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE
    
    # Try to initialize LLM
    try:
        # Check for environment variables
        model_name = os.environ.get("LLM_MODEL", "Hermes-3-Llama-3.1-8B.Q4_K_M")
        
        # Try to import llama-cpp-python
        try:
            from llama_cpp import Llama
            
            # Check if model file exists
            model_path = os.path.join("models", f"{model_name}")
            if not os.path.exists(model_path):
                # Try alternative paths
                alt_paths = [
                    os.path.join("models", f"{model_name}.gguf"),
                    os.path.join("models", f"{model_name}.bin"),
                    os.path.join("/", "models", f"{model_name}.gguf"),
                    os.path.join("/", "models", f"{model_name}.bin"),
                ]
                
                for path in alt_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                else:
                    logger.error(f"Could not find model file for {model_name}")
                    return None
            
            # Initialize Llama model
            logger.info(f"Initializing LLM with model: {model_path}")
            _LLM_INSTANCE = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window
                n_gpu_layers=-1,  # Auto-detect GPU layers
                n_threads=4,  # CPU threads
                verbose=False
            )
            
            logger.info("LLM initialized successfully")
            return _LLM_INSTANCE
            
        except ImportError:
            logger.warning("llama-cpp-python not available, trying transformers")
            
            # Try to use transformers as fallback
            try:
                from transformers import pipeline
                import torch
                
                # Check if CUDA is available
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                # Initialize model
                logger.info(f"Initializing transformers pipeline with model: {model_name}")
                _LLM_INSTANCE = pipeline(
                    "text-generation",
                    model=model_name,
                    device=device,
                    torch_dtype=torch_dtype
                )
                
                logger.info("LLM initialized successfully with transformers")
                return _LLM_INSTANCE
                
            except (ImportError, Exception) as e:
                logger.error(f"Failed to initialize transformers: {e}")
                return None
    
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return None

def generate_text(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Generate text using the LLM
    
    Args:
        prompt: The prompt to generate from
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation (higher = more random)
        
    Returns:
        Generated text
    """
    llm = get_llm()
    if llm is None:
        logger.error("LLM not initialized")
        return ""
    
    try:
        # Check if we're using llama-cpp-python or transformers
        if hasattr(llm, "generate"):
            # llama-cpp-python
            output = llm.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "<|im_end|>"]
            )
            return output
        elif hasattr(llm, "__call__"):
            # transformers pipeline
            output = llm(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
            
            # Extract generated text
            if isinstance(output, list) and len(output) > 0:
                return output[0].get("generated_text", "").replace(prompt, "")
            
            return ""
        else:
            logger.error("Unknown LLM interface")
            return ""
    
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return ""

def identify_speaker_names_llm(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Identify speaker names from conversation segments using LLM
    
    Args:
        segments: List of conversation segments
        
    Returns:
        Dictionary mapping speaker IDs to names
    """
    # Prepare conversation for LLM
    conversation_text = ""
    speakers = set()
    
    for segment in segments:
        speaker = segment.get("speaker", "")
        text = segment.get("text", "")
        
        if speaker and text:
            conversation_text += f"{speaker}: {text}\n"
            speakers.add(speaker)
    
    # Create prompt for LLM
    prompt = f"""
You are an AI assistant that identifies speaker names in conversation transcripts.
Given the following conversation, identify the real names of the speakers if mentioned.

Conversation:
{conversation_text}

Based on the conversation, identify the real names of the speakers.
Return your answer as a JSON object mapping speaker IDs to names.
If a speaker's name is not mentioned, keep their original speaker ID.

Example output format:
{{
  "Speaker 0": "John",
  "Speaker 1": "Speaker 1"
}}

Your answer (JSON format):
"""
    
    # Generate response from LLM
    response = generate_text(prompt, max_tokens=256, temperature=0.1)
    
    # Extract JSON from response
    try:
        # Look for JSON-like structure in the response
        json_match = re.search(r'({[\s\S]*})', response)
        if json_match:
            json_str = json_match.group(1)
            speaker_names = json.loads(json_str)
            
            # Validate the response
            if isinstance(speaker_names, dict):
                # Filter out any speakers that weren't in the original conversation
                return {k: v for k, v in speaker_names.items() if k in speakers}
    
    except Exception as e:
        logger.error(f"Error parsing LLM response for speaker names: {e}")
    
    # Fallback to rule-based approach
    return identify_speaker_names_fallback(segments)

def identify_speaker_names_fallback(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Identify speaker names from conversation segments using rule-based approach
    
    Args:
        segments: List of conversation segments
        
    Returns:
        Dictionary mapping speaker IDs to names
    """
    detected_names = {}
    name_mentions = {}
    
    # Initialize with speaker IDs
    speakers = set()
    for segment in segments:
        speaker = segment.get("speaker", "")
        if speaker:
            speakers.add(speaker)
            name_mentions[speaker] = {}
    
    # Look for name mentions
    for segment in segments:
        speaker = segment.get("speaker", "")
        text = segment.get("text", "")
        
        if not speaker or not text:
            continue
        
        # Look for common name patterns
        # 1. "I am [Name]" or "My name is [Name]"
        name_patterns = [
            r"(?:I am|I'm|my name is|I'm called|call me|I go by)\s+([A-Z][a-z]+)",
            r"(?:this is|it's|it is)\s+([A-Z][a-z]+)",
            r"([A-Z][a-z]+)(?:\s+here|speaking|talking)"
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(1)
                if name in COMMON_NAMES:
                    if speaker not in name_mentions:
                        name_mentions[speaker] = {}
                    if name not in name_mentions[speaker]:
                        name_mentions[speaker][name] = 0
                    name_mentions[speaker][name] += 3  # Higher weight for self-identification
        
        # 2. Look for names in the text that might be addressing other speakers
        for name in COMMON_NAMES:
            if name in text:
                # Check if the name is at the beginning of the text (likely addressing someone)
                if re.search(f"^{name}[,.]?\\s", text):
                    # This is likely addressing another speaker
                    for other_speaker in speakers:
                        if other_speaker != speaker:
                            if other_speaker not in name_mentions:
                                name_mentions[other_speaker] = {}
                            if name not in name_mentions[other_speaker]:
                                name_mentions[other_speaker][name] = 0
                            name_mentions[other_speaker][name] += 1
    
    # Assign names based on mentions
    for speaker, names in name_mentions.items():
        if names:
            # Get the most mentioned name for this speaker
            best_name = max(names.items(), key=lambda x: x[1])[0]
            detected_names[speaker] = best_name
    
    return detected_names

def summarize_conversation(segments: List[Dict[str, Any]], prompt: Optional[str] = None) -> str:
    """
    Generate a summary of the conversation using LLM
    
    Args:
        segments: List of conversation segments
        prompt: Optional custom prompt
        
    Returns:
        Summary text
    """
    # Prepare conversation for LLM
    conversation_text = ""
    
    for segment in segments:
        speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
        text = segment.get("text", "")
        
        if text:
            conversation_text += f"{speaker}: {text}\n"
    
    # Create prompt for LLM
    if prompt is None:
        prompt = "Summarize the key points of this conversation:"
    
    full_prompt = f"""
{prompt}

Conversation:
{conversation_text}

Summary:
"""
    
    # Generate response from LLM
    summary = generate_text(full_prompt, max_tokens=256, temperature=0.3)
    
    return summary.strip()

def extract_topics(segments: List[Dict[str, Any]]) -> List[str]:
    """
    Extract main topics from the conversation using LLM
    
    Args:
        segments: List of conversation segments
        
    Returns:
        List of topics
    """
    # Prepare conversation for LLM
    conversation_text = ""
    
    for segment in segments:
        speaker = segment.get("speaker_name", segment.get("speaker", "Unknown"))
        text = segment.get("text", "")
        
        if text:
            conversation_text += f"{speaker}: {text}\n"
    
    # Create prompt for LLM
    prompt = f"""
Extract the main topics discussed in this conversation.
Return a list of 3-5 topics, each as a short phrase.

Conversation:
{conversation_text}

Topics:
1.
"""
    
    # Generate response from LLM
    response = generate_text(prompt, max_tokens=128, temperature=0.3)
    
    # Parse topics from response
    topics = []
    for line in response.strip().split("\n"):
        # Look for numbered or bulleted list items
        match = re.match(r'(?:\d+\.|\*|\-)\s*(.*)', line)
        if match:
            topic = match.group(1).strip()
            if topic:
                topics.append(topic)
    
    return topics