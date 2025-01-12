"""
File: homage_tools/nodes/ht_regex_node.py
Version: 1.0.0
Description: A simple node for regex pattern matching in ComfyUI
"""

import re
from typing import Tuple, Dict, Any

#------------------------------------------------------------------------------
# Section 1: Node Definition
#------------------------------------------------------------------------------
class HTRegexNode:
    """
    A node that performs regex pattern matching on input text.
    """
    
    CATEGORY = "HommageTools"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("parsed_text",)
    FUNCTION = "parse_text"
    
    #--------------------------------------------------------------------------
    # Section 2: Input Configuration
    #--------------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "regex_pattern": ("STRING", {
                    "default": ".*"
                })
            }
        }
        
    #--------------------------------------------------------------------------
    # Section 3: Text Processing
    #--------------------------------------------------------------------------
    def parse_text(self, input_text: str, regex_pattern: str) -> Tuple[str]:
        """Process text with regex pattern."""
        try:
            # Compile and apply pattern
            pattern = re.compile(regex_pattern)
            matches = pattern.findall(input_text)
            
            # Format result
            if not matches:
                result = input_text
            elif len(matches) == 1:
                result = str(matches[0])
            else:
                result = "\n".join(str(match) for match in matches)
                
            return (result,)
            
        except re.error:
            return (input_text,)