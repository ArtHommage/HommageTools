"""
File: homage_tools/nodes/ht_text_cleanup_node.py

HommageTools Text Cleanup Node
Version: 1.0.0
Description: A node that provides comprehensive text cleanup functionality with
configurable rules and patterns.

Sections:
1. Imports and Type Definitions
2. Node Class Definition
3. Configuration and Constants
4. Cleanup Methods
5. Main Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import re
from typing import Dict, Any, Tuple

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTTextCleanupNode:
    """
    A ComfyUI node that provides comprehensive text cleanup capabilities with
    configurable rules.
    
    Features:
    - Remove multiple spaces
    - Handle linebreaks and carriage returns
    - Clean up punctuation spacing
    - Remove empty lines
    - Configurable cleanup rules
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "cleanup_text"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_text",)

#------------------------------------------------------------------------------
# Section 3: Configuration and Constants
#------------------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define the input types and their default values."""
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "description": "Text to be cleaned up"
                }),
            },
            "optional": {
                "preserve_period_space": ("BOOLEAN", {
                    "default": True,
                    "description": "Keep single space before periods"
                }),
                "preserve_linebreaks": ("BOOLEAN", {
                    "default": False,
                    "description": "Keep single linebreaks"
                }),
                "aggressive_cleanup": ("BOOLEAN", {
                    "default": False,
                    "description": "Apply more aggressive cleanup rules"
                })
            }
        }

#------------------------------------------------------------------------------
# Section 4: Cleanup Methods
#------------------------------------------------------------------------------
    def _normalize_whitespace(self, text: str, preserve_linebreaks: bool) -> str:
        """
        Normalize whitespace in text while respecting preservation rules.
        
        Args:
            text: Input text to normalize
            preserve_linebreaks: Whether to keep single linebreaks
            
        Returns:
            str: Text with normalized whitespace
        """
        # Replace all types of linebreaks with standard \n
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        if preserve_linebreaks:
            # Replace multiple linebreaks with single
            text = re.sub(r'\n{2,}', '\n', text)
        else:
            # Replace all linebreaks with spaces
            text = text.replace('\n', ' ')
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _cleanup_punctuation(
        self,
        text: str,
        preserve_period_space: bool,
        aggressive: bool
    ) -> str:
        """
        Clean up punctuation spacing and formatting.
        
        Args:
            text: Input text to clean
            preserve_period_space: Whether to keep space before periods
            aggressive: Whether to apply more aggressive cleanup
            
        Returns:
            str: Text with cleaned punctuation
        """
        # Basic punctuation cleanup
        text = re.sub(r'\s*,\s*', ', ', text)  # Normalize comma spacing
        text = re.sub(r'\s*;\s*', '; ', text)  # Normalize semicolon spacing
        text = re.sub(r'\s*:\s*', ': ', text)  # Normalize colon spacing
        
        # Handle periods based on preservation setting
        if preserve_period_space:
            text = re.sub(r'\s*\.\s*', ' . ', text)  # Keep space before period
            text = re.sub(r'\s*\.\s+', '. ', text)   # Single space after period
        else:
            text = re.sub(r'\s*\.\s*', '. ', text)   # No space before period
        
        if aggressive:
            # More aggressive cleanup rules
            text = re.sub(r'[.!?]+(?=[.!?])', '', text)  # Remove duplicate ending punctuation
            text = re.sub(r'[\'"]+', '"', text)          # Normalize quotes
            text = re.sub(r'\s*-+\s*', '-', text)        # Clean up dashes
            text = re.sub(r'\s*_+\s*', '_', text)        # Clean up underscores
            
        return text

    def _final_cleanup(self, text: str) -> str:
        """
        Perform final cleanup passes on the text.
        
        Args:
            text: Text to clean
            
        Returns:
            str: Final cleaned text
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Ensure single space between words
        text = ' '.join(word for word in text.split() if word)
        
        # Fix any remaining multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text

#------------------------------------------------------------------------------
# Section 5: Main Processing Logic
#------------------------------------------------------------------------------
    def cleanup_text(
        self,
        input_text: str,
        preserve_period_space: bool = True,
        preserve_linebreaks: bool = False,
        aggressive_cleanup: bool = False
    ) -> Tuple[str]:
        """
        Main processing function to clean up text according to specified rules.
        
        Args:
            input_text: Text to clean up
            preserve_period_space: Whether to keep space before periods
            preserve_linebreaks: Whether to keep single linebreaks
            aggressive_cleanup: Whether to apply more aggressive cleanup
            
        Returns:
            Tuple[str]: Single-element tuple containing cleaned text
        """
        try:
            if not input_text:
                return ("",)
            
            # Step 1: Normalize whitespace
            text = self._normalize_whitespace(input_text, preserve_linebreaks)
            
            # Step 2: Clean up punctuation
            text = self._cleanup_punctuation(
                text,
                preserve_period_space,
                aggressive_cleanup
            )
            
            # Step 3: Final cleanup pass
            text = self._final_cleanup(text)
            
            return (text,)
            
        except Exception as e:
            print(f"Error in HTTextCleanupNode: {str(e)}")
            return (input_text,)  # Return original text on error