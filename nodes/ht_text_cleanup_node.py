"""
File: homage_tools/nodes/ht_text_cleanup_node.py
Version: 1.0.0
Description: Node for text cleanup and formatting
"""

import re
from typing import Dict, Any, Tuple

class HTTextCleanupNode:
    """
    Provides text cleanup with configurable rules.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "cleanup_text"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_text",)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                })
            },
            "optional": {
                "preserve_period_space": ("BOOLEAN", {
                    "default": True
                }),
                "preserve_linebreaks": ("BOOLEAN", {
                    "default": False
                }),
                "aggressive_cleanup": ("BOOLEAN", {
                    "default": False
                })
            }
        }

    def _normalize_whitespace(self, text: str, preserve_linebreaks: bool) -> str:
        """Normalize whitespace in text."""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        if preserve_linebreaks:
            text = re.sub(r'\n{2,}', '\n', text)
        else:
            text = text.replace('\n', ' ')
        return re.sub(r' {2,}', ' ', text).strip()

    def _cleanup_punctuation(self, text: str, preserve_period_space: bool, aggressive: bool) -> str:
        """Clean up punctuation spacing."""
        text = re.sub(r'\s*[,;:]\s*', r'\1 ', text)
        
        if preserve_period_space:
            text = re.sub(r'\s*\.\s*', ' . ', text)
            text = re.sub(r'\s*\.\s+', '. ', text)
        else:
            text = re.sub(r'\s*\.\s*', '. ', text)
        
        if aggressive:
            text = re.sub(r'[.!?]+(?=[.!?])', '', text)
            text = re.sub(r'[\'"]+', '"', text)
            text = re.sub(r'\s*[-_]+\s*', r'\1', text)
            
        return text

    def cleanup_text(
        self,
        input_text: str,
        preserve_period_space: bool = True,
        preserve_linebreaks: bool = False,
        aggressive_cleanup: bool = False
    ) -> Tuple[str]:
        """Clean up text according to specified rules."""
        if not input_text:
            return ("",)
        
        text = self._normalize_whitespace(input_text, preserve_linebreaks)
        text = self._cleanup_punctuation(text, preserve_period_space, aggressive_cleanup)
        text = ' '.join(word for word in text.split() if word)
        
        return (text,)