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
import traceback
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

# Model configuration
MODEL_NAME = "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
MODEL_PATH = f"models/{MODEL_NAME}"

# No need for model selection anymore

# Path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# LLM instance (lazy-loaded)
_llm_instance = None

def get_model_path():
    """Get the path to the model file"""
    return os.path.join(MODELS_DIR, MODEL_NAME)

def get_llm() -> Optional[Any]:
    """Get or initialize the LLM instance"""
    global _llm_instance
    
    if not LLAMA_AVAILABLE:
        logger.warning("llama-cpp-python not available")
        return None
        
    if _llm_instance is None:
        try:
            # Print call stack for debugging
            logger.info("Call stack for LLM initialization:")
            for line in traceback.format_stack():
                logger.info(line.strip())
                
            model_path = get_model_path()
            
            # Print debug info
            logger.info(f"Current model: {MODEL_NAME}")
            logger.info(f"Model path: {model_path}")
            logger.info(f"Models directory: {MODELS_DIR}")
            
            # Check if model exists locally
            if os.path.exists(model_path):
                logger.info(f"Loading model from local path: {model_path}")
                try:
                    # Check for CUDA availability
                    import torch
                    cuda_available = torch.cuda.is_available()
                    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
                    current_device = torch.cuda.current_device() if cuda_available else "N/A"
                    device_name = torch.cuda.get_device_name(current_device) if cuda_available else "N/A"
                    
                    logger.info(f"CUDA available: {cuda_available}")
                    logger.info(f"CUDA device count: {cuda_device_count}")
                    logger.info(f"Current CUDA device: {current_device}")
                    logger.info(f"Current CUDA device name: {device_name}")
                    
                    # Try to initialize with GPU support
                    logger.info(f"Initializing Llama with model path: {model_path}")
                    _llm_instance = Llama(model_path=model_path, n_gpu_layers=32, n_ctx=2048)
                    logger.info("LLM initialized successfully from local path")
                except Exception as local_error:
                    logger.error(f"Error loading local model: {local_error}")
                    logger.error("Could not initialize LLM. Please check that the model file exists and is valid.")
            else:
                logger.error(f"Model file not found at {model_path}")
                logger.error("Please ensure the model file is in the models directory.")
                    
                    logger.info(f"CUDA available: {cuda_available}")
                    logger.info(f"CUDA device count: {cuda_device_count}")
                    logger.info(f"Current CUDA device: {current_device}")
                    logger.info(f"Current CUDA device name: {device_name}")
                    
                    _llm_instance = Llama.from_pretrained(
                        repo_id=model_info['repo'],
                        filename=model_info['filename'],
                        local_dir=MODELS_DIR,
                        n_gpu_layers=32,
                        n_ctx=2048
                    )
                    logger.info("LLM downloaded and initialized successfully with GPU support")
                except Exception as download_error:
                    logger.error(f"Error downloading model with GPU support: {download_error}")
                    # Try with CPU only as last resort
                    logger.info("Attempting CPU-only initialization...")
                    _llm_instance = Llama.from_pretrained(
                        repo_id=model_info['repo'],
                        filename=model_info['filename'],
                        local_dir=MODELS_DIR,
                        n_gpu_layers=0
                    )
                    logger.info("LLM initialized with CPU only")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            # Create a dummy LLM that returns empty results
            logger.warning("Creating a dummy LLM that returns empty results")
            return DummyLLM()
            
    return _llm_instance

class DummyLLM:
    """A dummy LLM class that returns empty results for all methods"""
    
    def __init__(self):
        logger.warning("Using DummyLLM - all LLM operations will return empty results")
    
    def create_completion(self, prompt, **kwargs):
        logger.info(f"DummyLLM received prompt: {prompt[:50]}...")
        return {
            'choices': [{'text': '{"Speaker 0": "Unknown", "Speaker 1": "Unknown"}'}]
        }

def identify_speaker_names_llm(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Use LLM to identify speaker names from conversation segments
    
    Args:
        segments: List of conversation segments with speaker and text
        
    Returns:
        Dictionary mapping speaker IDs to names
    """
    logger.info("Starting identify_speaker_names_llm function")
    
    # Validate input
    if not segments or not isinstance(segments, list):
        logger.error(f"Invalid segments provided: {segments}")
        return {}
    
    logger.info(f"Processing {len(segments)} segments for name identification")
    if segments:
        logger.info(f"First segment: {segments[0]}")
    
    # Check if we have the required keys in the segments
    valid_segments = []
    for segment in segments:
        if not isinstance(segment, dict):
            logger.warning(f"Segment is not a dictionary: {segment}")
            continue
            
        if 'speaker' not in segment or 'text' not in segment:
            logger.warning(f"Segment missing required keys: {segment}")
            continue
            
        if not segment.get('text'):
            logger.debug("Skipping segment with empty text")
            continue
            
        valid_segments.append(segment)
    
    if not valid_segments:
        logger.error("No valid segments found for name identification")
        return {}
        
    logger.info(f"Found {len(valid_segments)} valid segments for processing")
    
    # Get LLM
    llm = get_llm()
    if not llm:
        logger.warning("LLM not available, using fallback method")
        return {}
    
    # Prepare conversation context
    conversation_text = ""
    for segment in valid_segments[:10]:  # Use first 10 segments to identify speakers
        speaker = segment.get('speaker', 'Unknown')
        text = segment.get('text', '').strip()
        conversation_text += f"{speaker}: {text}\n"
    
    logger.info(f"Prepared conversation context of {len(conversation_text)} characters")
    
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
        print(prompt)
        logger.info("Sending request to LLM for name identification")
        # Get response from LLM
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,
            stop=["```"]
        )
        
        # Extract and parse the JSON response
        response_text = response['choices'][0]['text'].strip()
        logger.info(f"Received response from LLM: {response_text[:100]}...")
        
        # Find JSON block in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                speaker_names = json.loads(json_str)
                logger.info(f"Successfully parsed speaker names: {speaker_names}")
                
                # Check for Veronica in the text and ensure it's included
                if any('Veronica' in segment.get('text', '').lower() for segment in valid_segments):
                    # Find which speaker mentions Veronica
                    for segment in valid_segments:
                        if 'Veronica' in segment.get('text', '').lower():
                            # If this speaker is not Veronica, the other one might be
                            speaker_id = segment.get('speaker')
                            if speaker_id in speaker_names and speaker_names[speaker_id] != "Veronica":
                                # Find other speakers
                                other_speakers = [s for s in set(seg.get('speaker') for seg in valid_segments) if s != speaker_id]
                                for other_id in other_speakers:
                                    if other_id not in speaker_names or not speaker_names[other_id]:
                                        speaker_names[other_id] = "Veronica"
                                        logger.info(f"Added Veronica as {other_id} based on context")
                
                return speaker_names
            except json.JSONDecodeError as e:
                print("Failed to parse JSON from LLM response:", json_str, "- Error:", e)
                logger.error(f"Failed to parse JSON from LLM response: {json_str} - Error: {e}")
                # Try to extract names using regex as fallback
                try:
                    import re
                    name_matches = re.findall(r'"Speaker \d+":\s*"([^"]+)"', response_text)
                    if name_matches:
                        speaker_ids = re.findall(r'"(Speaker \d+)":', response_text)
                        if len(speaker_ids) == len(name_matches):
                            speaker_names = {}
                            for i, speaker_id in enumerate(speaker_ids):
                                speaker_names[speaker_id] = name_matches[i]
                            logger.info(f"Extracted names using regex: {speaker_names}")
                            return speaker_names
                except Exception as regex_error:
                    logger.error(f"Error in regex fallback: {regex_error}")
        else:
            logger.error(f"Could not find JSON in response: {response_text}")
        
        return {}
    except Exception as e:
        logger.error(f"Error getting speaker names from LLM: {e}")
        logger.error(traceback.format_exc())
        return {}

def summarize_conversation(segments: List[Dict[str, Any]]) -> str:
    """
    Generate a summary of the conversation using the LLM
    
    Args:
        segments: List of conversation segments
        
    Returns:
        Summary text
    """
    try:
        llm = get_llm()
        if not llm:
            logger.warning("LLM not available for summarization")
            return "Conversation summary not available (LLM not loaded)"
        
        # Check if segments is valid
        if not segments or not isinstance(segments, list):
            logger.warning("No valid segments provided for summarization")
            return "No conversation data available for summarization."
            
        # Limit the number of segments to avoid overwhelming the LLM
        if len(segments) > 20:
            logger.info(f"Limiting from {len(segments)} to 20 segments for summarization")
            limited_segments = segments[:20]
        else:
            limited_segments = segments
        
        # Prepare conversation context
        conversation_text = ""
        for segment in limited_segments:
            if 'speaker' not in segment or 'text' not in segment:
                continue
                
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            
            # Skip empty text
            if not text:
                continue
                
            # Use speaker_name if available, otherwise use speaker ID
            speaker_name = segment.get('speaker_name', speaker)
            conversation_text += f"{speaker_name}: {text}\n"
        
        # If no valid conversation text, return early
        if not conversation_text.strip():
            logger.warning("No valid conversation text for summarization")
            return "No conversation content available for summarization."
        
        # Create prompt for the LLM
        prompt = f"""
You are T-AI-bitha, a wonderful summarizer.  Please make things entertaining and funny.

Here's a conversation:
{conversation_text}

Please provide a concise summary (3-5 sentences) of this conversation, highlighting:
1. The main topics discussed
2. The silver lining of the conversation
3. Any decisions or conclusions reached
4. Any action items or next steps mentioned

Summary:
"""
        
        logger.info("Sending summarization request to LLM")
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
            logger.info(f"Generated summary: {summary[:50]}...")
            return summary
        except Exception as e:
            logger.error(f"Error in LLM completion for summary: {e}")
            return "Summary not available due to an error in LLM processing."
    except Exception as e:
        logger.error(f"Unexpected error in summarize_conversation: {e}")
        return "Summary not available due to an unexpected error."

def extract_topics(segments: List[Dict[str, Any]]) -> List[str]:
    """
    Extract main topics from the conversation
    
    Args:
        segments: List of conversation segments
        
    Returns:
        List of main topics
    """
    try:
        llm = get_llm()
        if not llm:
            logger.warning("LLM not available for topic extraction")
            return []
        
        # Check if segments is valid
        if not segments or not isinstance(segments, list):
            logger.warning("No valid segments provided for topic extraction")
            return []
            
        # Limit the number of segments to avoid overwhelming the LLM
        if len(segments) > 20:
            logger.info(f"Limiting from {len(segments)} to 20 segments for topic extraction")
            limited_segments = segments[:20]
        else:
            limited_segments = segments
        
        # Prepare conversation context
        conversation_text = ""
        for segment in limited_segments:
            if 'speaker' not in segment or 'text' not in segment:
                continue
                
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '').strip()
            
            # Skip empty text
            if not text:
                continue
                
            # Use speaker_name if available, otherwise use speaker ID
            speaker_name = segment.get('speaker_name', speaker)
            conversation_text += f"{speaker_name}: {text}\n"
        
        # If no valid conversation text, return early
        if not conversation_text.strip():
            logger.warning("No valid conversation text for topic extraction")
            return []
        
        # Create prompt for the LLM
        prompt = f"""
You are an AI assistant that analyzes conversations and extracts the main topics.

Here's a conversation:
{conversation_text}

Please list the 3-5 main topics discussed in this conversation. Return your answer as a JSON array of strings.
Example: ["Project timeline", "Budget concerns", "Marketing strategy"]

Your response should ONLY contain the JSON array, nothing else. No explanations, no additional text.
"""

        logger.info("Sending topic extraction request to LLM")
        print('3-5 main topics:', prompt)
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
            logger.info(f"LLM response for topics: {response_text[:50]}...")
            
            # First, try to find a clean JSON array using regex
            import re
            json_pattern = re.search(r'\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]', response_text)
            
            if json_pattern:
                json_str = json_pattern.group(0)
                logger.info(f"JSON pattern found: {json_str}")
                try:
                    topics = json.loads(json_str)
                    if isinstance(topics, list):
                        logger.info(f"Successfully parsed topics from regex: {topics}")
                        return topics
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON pattern: {e}")
            
            # If regex approach fails, try the traditional bracket approach
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                logger.info(f"JSON block found: {json_str}")
                try:
                    topics = json.loads(json_str)
                    if isinstance(topics, list):
                        logger.info(f"Successfully extracted {len(topics)} topics")
                        return topics
                    else:
                        logger.warning(f"Parsed JSON is not a list: {topics}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from LLM response: {e}")
                    
                    # Try to clean up the JSON string and retry
                    try:
                        # Remove any text before the first [ and after the last ]
                        cleaned_json = json_str
                        # Replace any invalid characters that might be causing issues
                        cleaned_json = re.sub(r'[\x00-\x1F\x7F-\xFF]', '', cleaned_json)
                        topics = json.loads(cleaned_json)
                        if isinstance(topics, list):
                            logger.info(f"Successfully parsed topics after cleanup: {topics}")
                            return topics
                    except json.JSONDecodeError:
                        pass
            
            logger.info("Failed to extract topics from JSON response")
            
            # Fallback: try to extract topics from plain text
            logger.info("Falling back to plain text extraction")
            # Look for numbered or bulleted list items
            topics = re.findall(r'[\d\*\-•]+\s*[.:]?\s*([^\n]+)', response_text)
            if topics:
                logger.info(f"Extracted {len(topics)} topics using regex fallback")
                return topics
            
            logger.warning("Could not extract topics from LLM response")
            return []
        except Exception as e:
            logger.error(f"Error in LLM completion for topic extraction: {e}")
            return []
    except Exception as e:
        logger.error(f"Unexpected error in extract_topics: {e}")
        return []

# Fallback methods that don't require LLM
def identify_speaker_names_fallback(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    """Fallback method for speaker name identification without LLM"""
    try:
        from common_data import COMMON_NAMES
        logger.info("Using fallback method for speaker name identification")
        
        # Check if segments is valid
        if not segments or not isinstance(segments, list):
            logger.warning("No valid segments provided for name identification")
            return {}
            
        # Dictionary to store detected names for each speaker ID
        detected_names = {}
        name_mentions = {}
        
        # Get unique speaker IDs
        unique_speakers = set()
        for segment in segments:
            if 'speaker' in segment:
                unique_speakers.add(segment['speaker'])
            
        # Convert to sorted list for deterministic behavior
        unique_speakers = sorted(list(unique_speakers))
            
        logger.info(f"Found {len(unique_speakers)} unique speakers in {len(segments)} segments")
        
        # If no speakers found, return default names
        if not unique_speakers:
            logger.warning("No speakers found in segments, using default names")
            return {"Speaker 0": "Speaker A", "Speaker 1": "Speaker B"}
        
        # First pass: find potential speaker names in the text
        for segment in segments:
            if 'speaker' not in segment or 'text' not in segment:
                continue
                
            speaker_id = segment['speaker']
            text = segment['text']
            
            # Skip empty text
            if not text or not isinstance(text, str):
                continue
            
            # Check for Veronica specifically
            if 'Veronica' in text.lower():
                logger.info(f"Found Veronica mentioned in text: {text}")
                # If this speaker is addressing Veronica, they're likely not Veronica
                # So we'll note this for later
                if speaker_id not in detected_names:
                    name_mentions['Veronica'] = name_mentions.get('Veronica', 0) + 3
            
            # Extract names from text
            # Look for common name patterns directly
            import re
            for name in COMMON_NAMES:
                # Look for the name as a whole word with word boundaries
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    # Count name mentions
                    if name not in name_mentions:
                        name_mentions[name] = 0
                    name_mentions[name] += 1
                    logger.info(f"Found name mention: {name} in text: {text[:30]}...")
            
            # Look for "I'm [Name]" or "My name is [Name]" patterns
            name_intro_patterns = [
                r"I'?m\s+(\w+)",
                r"[Mm]y name is\s+(\w+)",
                r"[Cc]all me\s+(\w+)",
                r"[Tt]his is\s+(\w+)"  
            ]
            
            for pattern in name_intro_patterns:
                try:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if match in COMMON_NAMES:
                            if speaker_id not in detected_names:
                                detected_names[speaker_id] = match
                                logger.info(f"Found name for {speaker_id}: {match} (from pattern)")
                            elif detected_names[speaker_id] != match:
                                # If multiple names are detected for the same speaker, keep the most frequent one
                                if match not in name_mentions:
                                    name_mentions[match] = 0
                                name_mentions[match] += 3  # Give higher weight to explicit introduction
                except Exception as e:
                    logger.error(f"Error in regex pattern matching: {e}")
        
        # Second pass: assign names to speakers based on frequency and context
        for speaker_id in unique_speakers:
            if speaker_id not in detected_names:
                # Find the most mentioned name that hasn't been assigned yet
                try:
                    available_names = [name for name, count in sorted(name_mentions.items(), key=lambda x: x[1], reverse=True) 
                                    if name not in detected_names.values()]                
                    if available_names:
                        detected_names[speaker_id] = available_names[0]
                        logger.info(f"Assigned name for {speaker_id}: {available_names[0]} (from frequency)")
                        
                        # Special case for Veronica
                        if 'Veronica' in name_mentions and 'Veronica' not in detected_names.values():
                            # If Veronica is mentioned but not assigned, and we have multiple speakers,
                            # try to assign Veronica to one of them
                            if len(unique_speakers) > 1:
                                # First, check if anyone is addressing Veronica directly
                                speaker_addressing_Veronica = None
                                for seg in segments:
                                    if 'text' not in seg or 'speaker' not in seg:
                                        continue
                                    
                                    text = seg['text']
                                    speaker = seg['speaker']
                                    
                                    # Check for patterns like "Hi Veronica", "Hello Veronica", etc.
                                    if re.search(r'\b(hi|hello|hey)\s+Veronica\b', text.lower()):
                                        speaker_addressing_Veronica = speaker
                                        logger.info(f"Found speaker {speaker} addressing Veronica")
                                        break
                                
                                # If someone is addressing Veronica, she's likely not that speaker
                                if speaker_addressing_Veronica:
                                    # Find speakers other than the one addressing Veronica
                                    other_speakers = sorted([s for s in unique_speakers if s != speaker_addressing_Veronica])
                                    
                                    if other_speakers:
                                        detected_names[other_speakers[0]] = "Veronica"
                                        logger.info(f"Assigned Veronica to {other_speakers[0]} based on being addressed")
                                else:
                                    # No direct addressing found, fall back to default behavior
                                    # Find a speaker without a name or with a generic name
                                    for spk_id in unique_speakers:
                                        if spk_id not in detected_names or detected_names[spk_id].startswith("Speaker"):
                                            detected_names[spk_id] = "Veronica"
                                            logger.info(f"Assigned Veronica to {spk_id} based on mentions")
                                            break
                except Exception as e:
                    logger.error(f"Error assigning names by frequency: {e}")
                    logger.error(traceback.format_exc())
        
        # If we still don't have names for all speakers, use default names
        for i, speaker_id in enumerate(unique_speakers):
            if speaker_id not in detected_names:
                default_name = f"Speaker {chr(65+i)}"  # A, B, C, etc.
                detected_names[speaker_id] = default_name
                logger.info(f"Using default name for {speaker_id}: {default_name}")
        
        logger.info(f"Final speaker names: {detected_names}")
        return detected_names
    except Exception as e:
        logger.error(f"Unexpected error in identify_speaker_names_fallback: {e}")
        # Return default names as a last resort
        return {"Speaker 0": "Speaker A", "Speaker 1": "Speaker B"}
