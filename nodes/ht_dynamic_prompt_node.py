"""
File: homage_tools/nodes/ht_dynamic_prompt_node.py
Version: 1.6.0
Description: Dynamic prompt node with enhanced prompt generation capabilities and accurate token limiting
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, List

# Dynamic Prompts imports
from dynamicprompts.enums import SamplingMethod
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.wildcards import WildcardManager

# Configure logging
logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "1.6.0"
DEFAULT_MAX_TOKENS = 75  # Default token limit for prompt (below 77 to be safe)

#------------------------------------------------------------------------------
# Section 3: Helper Functions
#------------------------------------------------------------------------------
def filter_lora_tags(text: str) -> str:
    """
    Remove LORA tags (text within < and >) for token counting purposes.
    
    Args:
        text: Input text with potential LORA tags
        
    Returns:
        str: Text with LORA tags removed
    """
    return re.sub(r'<[^>]+>', '', text)

def find_wildcards_folder() -> Path:
    """
    Find the wildcards folder for Dynamic Prompts.
    
    First looks in the comfy_dynamicprompts folder, then in the custom_nodes folder, 
    then in the ComfyUI base folder.
    
    Returns:
        Path: Path to the wildcards folder
    """
    from folder_paths import base_path, folder_names_and_paths

    wildcard_path = Path(base_path) / "wildcards"

    if wildcard_path.exists():
        return wildcard_path

    extension_path = (
        Path(folder_names_and_paths["custom_nodes"][0][0])
        / "comfyui-dynamicprompts"
    )
    wildcard_path = extension_path / "wildcards"
    wildcard_path.mkdir(parents=True, exist_ok=True)

    return wildcard_path

def clean_whitespace(text: str) -> str:
    """
    Clean excessive whitespace from text.
    
    Args:
        text: Input text to clean
        
    Returns:
        str: Text with normalized whitespace
    """
    # Replace ". ." with ". "
    text = text.replace(". .", ". ")
    # Replace more than one consecutive space with a single space
    text = re.sub(r' {2,}', ' ', text)
    # Remove line feeds
    text = text.replace('\n', ' ')
    # Clean up any spaces created by removing line feeds
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a string without CLIP.
    LORA tags (text within < and >) are ignored for token counting.
    
    Args:
        text: Input text to estimate
        
    Returns:
        int: Estimated token count
    """
    # Remove LORA tags before counting
    text = filter_lora_tags(text)
    
    # Simple approximation: words + punctuation
    if not text:
        return 0
        
    # Count words (splitting on whitespace)
    words = text.split()
    word_count = len(words)
    
    # Count punctuation and special characters that might be separate tokens
    punct_count = sum(1 for char in text if char in ",.!?;:()[]{}\"'`-_=+<>/\\|@#$%^&*")
    
    # Return estimated count with a safety margin
    return word_count + punct_count

def get_clip_token_count(clip, text: str) -> Tuple[int, List[str]]:
    """
    Get exact token count using CLIP tokenizer.
    LORA tags (text within < and >) are ignored for token counting.
    
    Args:
        clip: CLIP model object
        text: Text to tokenize
        
    Returns:
        Tuple[int, List[str]]: Token count and token strings
    """
    if not text:
        return 0, []
    
    # Remove LORA tags before counting
    filtered_text = filter_lora_tags(text)
        
    # Tokenize with CLIP
    tokens = clip.tokenize(filtered_text)
    
    # Get vocabulary for debug purposes
    try:
        vocab = clip.tokenizer.clip_l.inv_vocab
        special_tokens = set(clip.cond_stage_model.clip_l.special_tokens.values())
        tokens_concat = sum(tokens["l"], [])
        tokens_filtered = [t for t in tokens_concat if t[0] not in special_tokens]
        token_strings = [vocab.get(t[0], "<unk>") for t in tokens_filtered]
        return len(tokens_filtered), token_strings
    except (AttributeError, KeyError) as e:
        # Fallback in case of any issue with the tokenizer
        return len(sum(tokens["l"], [])), []

def smart_truncate_prompt(prompt: str, max_tokens: int, token_strings: List[str] = None) -> Tuple[str, str]:
    """
    Smartly truncate a prompt to fit within token limit, respecting word boundaries.
    
    Args:
        prompt: Original prompt
        max_tokens: Maximum tokens allowed
        token_strings: Optional list of actual token strings for detailed truncation
        
    Returns:
        Tuple[str, str]: Truncated prompt and truncation details
    """
    if not token_strings or len(token_strings) <= max_tokens:
        # Fallback to simple word truncation when token details aren't available
        # or when no truncation is needed
        words = prompt.split()
        if len(words) <= max_tokens:
            return prompt, "No truncation needed"
        
        truncated = " ".join(words[:max_tokens])
        details = f"Trimmed {len(words) - max_tokens} words ({len(words) - max_tokens}/{len(words)} = {(len(words) - max_tokens) / len(words):.1%})"
        return truncated, details
    
    # With token information, we can do more precise truncation
    keep_tokens = token_strings[:max_tokens]
    dropped_tokens = token_strings[max_tokens:]
    
    # Try to find word boundaries for cleaner truncation
    if len(keep_tokens) > 0:
        # Join the tokens we're keeping
        kept_text = "".join(keep_tokens)
        # Find the last space in the prompt up to this point
        last_word_end = prompt.rfind(" ", 0, len(kept_text))
        
        if last_word_end > 0:
            # Truncate at the last complete word
            truncated = prompt[:last_word_end]
        else:
            # No good word boundary found, use token-based truncation
            truncated = kept_text
    else:
        # Fallback for empty token list
        truncated = ""
    
    # Create detailed report without truncating content with "..."
    if dropped_tokens:
        # Show full dropped content instead of truncating
        dropped_content = " ".join(dropped_tokens)
        details = (f"Kept {max_tokens}/{len(token_strings)} tokens ({max_tokens/len(token_strings):.1%})\n"
                  f"Dropped content: \"{dropped_content}\"")
    else:
        details = "No truncation needed"
        
    return truncated, details

#------------------------------------------------------------------------------
# Section 4: Node Class Definition
#------------------------------------------------------------------------------
class HTDynamicPromptNode:
    """
    A Dynamic Prompts node with enhanced processing capabilities.
    
    Features:
    - Random sampling from dynamic prompt syntax
    - Support for wildcards from external files
    - Options for whitespace handling
    - Full support for all Dynamic Prompts syntax:
      {variations|like|this}, [number:ranges], __wildcards__, etc.
    - Accurate token limiting using CLIP
    - Excludes LORA tags (<tag>) from token counting
    - Support for float ranges with precision determined by first number
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "generate_prompt"
    RETURN_TYPES = ("STRING", "STRING", "CLIP")
    RETURN_NAMES = ("prompt", "token_info", "clip")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "", 
                    "dynamicPrompts": False,
                    "placeholder": "Enter prompt template with wildcards, variants, etc."
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "display": "number"
                }),
                "autorefresh": (["Yes", "No"], {
                    "default": "No",
                    "tooltip": "Automatically refresh the output on node evaluation"
                }),
                "strip_whitespace": (["Yes", "No"], {
                    "default": "No",
                    "tooltip": "Remove excessive whitespace and linebreaks"
                }),
                "max_tokens": ("INT", {
                    "default": DEFAULT_MAX_TOKENS,
                    "min": 10,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Maximum number of tokens in the output prompt (to avoid token limit errors)"
                })
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "CLIP model for accurate token counting"
                })
            }
        }

    @classmethod
    def IS_CHANGED(cls, text: str, seed: int, autorefresh: str, strip_whitespace: str, max_tokens: int, clip=None) -> Optional[float]:
        """
        Determine if the node should be re-evaluated.
        
        Returns NaN when autorefresh is enabled to force re-evaluation.
        """
        # Force re-evaluation of the node with autorefresh
        if autorefresh == "Yes":
            return float("NaN")
        return None

    #--------------------------------------------------------------------------
    # Section 5: Initialization and Context Management
    #--------------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        """Initialize the node with wildcards manager."""
        super().__init__(*args, **kwargs)
        wildcards_folder = find_wildcards_folder()
        self._wildcard_manager = WildcardManager(path=wildcards_folder)
        self._current_prompt = None
        self._prompts = None

    def _get_context(self) -> SamplingContext:
        """
        Get the random sampling context.
        
        Returns:
            SamplingContext: Context for sampling prompts
        """
        return SamplingContext(
            wildcard_manager=self._wildcard_manager,
            default_sampling_method=SamplingMethod.RANDOM,
        )

    def has_prompt_changed(self, text: str) -> bool:
        """
        Check if the prompt template has changed.
        
        Args:
            text: New prompt template
            
        Returns:
            bool: True if the prompt has changed
        """
        return self._current_prompt != text

    def _get_next_prompt(self, prompts: Iterable[Any], current_prompt: str) -> str:
        """
        Get the next prompt from the prompts generator.
        
        Args:
            prompts: Generator of prompts or sampling results
            current_prompt: Current prompt template
            
        Returns:
            str: Next generated prompt as string
        """
        try:
            result = next(prompts)
            # Handle SamplingResult object or other types
            if hasattr(result, 'prompt'):
                # It's a SamplingResult object with a prompt attribute
                return str(result.prompt)
            elif hasattr(result, '__str__'):
                # Convert any other object to string
                return str(result)
            else:
                # Fallback for unexpected types
                return f"{result}"
        except (StopIteration, RuntimeError) as e:
            print(f"Resetting prompt generator: {str(e)}")
            try:
                self._prompts = self._get_context().sample_prompts(self._current_prompt)
                result = next(self._prompts)
                # Handle the same way as above
                if hasattr(result, 'prompt'):
                    return str(result.prompt)
                else:
                    return str(result)
            except Exception as nested_e:
                logger.exception(f"Failed to generate prompt: {nested_e}")
                return ""

    #--------------------------------------------------------------------------
    # Section 6: Float Range Processing
    #--------------------------------------------------------------------------
    def _process_float_ranges(self, prompt: str, seed: int = None) -> str:
        """
        Process float ranges in the prompt and replace with random values.
        
        Args:
            prompt: Input prompt string
            seed: Random seed for reproducibility
            
        Returns:
            str: Prompt with float ranges replaced by random values
        """
        if seed is not None:
            # Create a local random instance with the given seed
            rand = random.Random(seed)
        else:
            # Use the global random instance
            rand = random
        
        # Pattern for float range: [X.Y:Z.W] where X, Y, Z, W are digits
        pattern = r'\[(\d+\.\d+):(\d+\.\d+)\]'
        
        def replacement(match):
            start = float(match.group(1))
            end = float(match.group(2))
            
            # Determine precision from first number
            precision = len(match.group(1).split('.')[-1])
            
            # Generate random float within range and format with precision
            value = rand.uniform(start, end)
            return f"{value:.{precision}f}"
        
        # Replace all occurrences of the pattern with random values
        result = re.sub(pattern, replacement, prompt)
        return result

    #--------------------------------------------------------------------------
    # Section 7: Main Processing Logic
    #--------------------------------------------------------------------------
    def generate_prompt(
        self, 
        text: str, 
        seed: int, 
        autorefresh: str, 
        strip_whitespace: str,
        max_tokens: int,
        clip=None
    ) -> Tuple[str, str, Any]:
        """
        Generate a dynamic prompt using the sampling context.
        
        Args:
            text: Prompt template with wildcards
            seed: Random seed for reproducible generation
            autorefresh: Whether to refresh on each evaluation
            strip_whitespace: Whether to strip excessive whitespace
            max_tokens: Maximum number of tokens allowed
            clip: Optional CLIP model for accurate tokenization
            
        Returns:
            Tuple[str, str, Any]: Tuple containing the generated prompt, token info, and CLIP model
        """
        print(f"\nHTDynamicPromptNode v{VERSION} - Processing")
        
        try:
            # Apply whitespace stripping if requested
            processed_text = text
            if strip_whitespace == "Yes":
                processed_text = clean_whitespace(text)
            
            # Return empty if input is empty
            if not processed_text.strip():
                return ("", "No prompt provided", clip)

            # Handle seed for reproducibility
            context = self._get_context()
            if seed > 0:
                context.rand.seed(seed)

            # Check if prompt has changed to reset generator
            if self.has_prompt_changed(processed_text) or self._prompts is None:
                self._current_prompt = processed_text
                self._prompts = context.sample_prompts(self._current_prompt)
                print(f"Created new prompt generator for: {processed_text[:30]}...")

            # Get the next prompt from the generator
            if self._prompts is None:
                logger.error("Something went wrong. Prompts is None!")
                return ("Error: Failed to initialize prompt generator", "Error", clip)

            new_prompt = self._get_next_prompt(self._prompts, self._current_prompt)
            
            # Process float ranges with the same seed
            new_prompt = self._process_float_ranges(new_prompt, seed)
            
            # Apply whitespace cleaning to output if requested
            if strip_whitespace == "Yes":
                new_prompt = clean_whitespace(new_prompt)
            
            # Check token count using CLIP if available, otherwise estimate
            token_strings = []
            token_info = ""
            
            if clip:
                # Use CLIP for accurate tokenization (excluding LORA tags)
                token_count, token_strings = get_clip_token_count(clip, new_prompt)
                token_info = f"CLIP tokens: {token_count}"
                
                # Truncate if needed
                if token_count > max_tokens:
                    original_prompt = new_prompt
                    new_prompt, details = smart_truncate_prompt(new_prompt, max_tokens, token_strings)
                    token_info = f"CLIP tokens: {max_tokens}/{token_count}\n{details}"
            else:
                # Use approximation (excluding LORA tags)
                token_count = estimate_token_count(new_prompt)
                token_info = f"Estimated tokens: {token_count}"
                
                # Truncate if needed
                if token_count > max_tokens:
                    original_prompt = new_prompt
                    new_prompt, details = smart_truncate_prompt(new_prompt, max_tokens)
                    token_info = f"Estimated tokens: {max_tokens}/{token_count}\n{details}"
            
            print(f"Generated prompt: {new_prompt}")
            print(f"Token info: {token_info}")
            
            return (str(new_prompt), token_info, clip)
            
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            print(f"Error details: {str(e)}")
            return (f"Error: {str(e)}", "Error generating prompt", clip)