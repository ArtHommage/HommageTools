"""
File: homage_tools/nodes/ht_conversion_node.py
Version: 1.0.0
Description: Simple type conversion node
"""

from typing import Dict, Any, Tuple, Optional

class HTConversionNode:
    """Type conversion node for string, int, and float values."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "convert_value"
    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("string_out", "int_out", "float_out")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "input_value": ("STRING", {
                    "multiline": True,
                    "default": ""
                })
            }
        }

    def convert_value(self, input_value: str) -> Tuple[str, int, float]:
        """Convert input to all types."""
        # String is already handled
        str_val = input_value.strip()
        
        # Try float conversion
        try:
            float_val = float(str_val)
        except (ValueError, TypeError):
            float_val = 0.0

        # Try int conversion
        try:
            int_val = int(float_val)
        except (ValueError, TypeError):
            int_val = 0

        return (str_val, int_val, float_val)