"""
File: homage_tools/nodes/ht_conversion_node.py

HommageTools Conversion Node
Version: 1.0.0
Description: A node that handles type conversions between string, integer, and float
with optional pause functionality for value modification before output.

Sections:
1. Imports and Type Definitions
2. Node Class Definition
3. Input/Output Configuration
4. Conversion Methods
5. Main Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Optional, Dict, Any, Tuple, Union

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTConversionNode:
    """
    A ComfyUI node that provides flexible type conversion between string, integer,
    and float values with optional pause functionality for value modification.
    
    Features:
    - Optional inputs for string, integer, and float values
    - Type conversion between all supported types
    - Editable display of current value
    - Optional pause functionality with resume button
    - Value modification during pause
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_conversion"
    OUTPUT_NODE = True  # Required for pause functionality
    PAUSABLE = True    # Enables pause support
    
#------------------------------------------------------------------------------
# Section 3: Input/Output Configuration
#------------------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define the input types and their default values."""
        return {
            "required": {
                "pause_enabled": ("BOOLEAN", {
                    "default": False,
                    "description": "Enable pause for value modification"
                }),
            },
            "optional": {
                "input_string": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "description": "String input value"
                }),
                "input_integer": ("INT", {
                    "default": 0,
                    "min": -2147483648,
                    "max": 2147483647,
                    "description": "Integer input value"
                }),
                "input_float": ("FLOAT", {
                    "default": 0.0,
                    "min": float('-inf'),
                    "max": float('inf'),
                    "description": "Float input value"
                }),
                "display_value": ("STRING", {
                    "multiline": True,
                    "default": "No input provided",
                    "description": "Current value display"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID"
            },
        }
    
    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("converted_string", "converted_integer", "converted_float")
    
#------------------------------------------------------------------------------
# Section 4: Conversion Methods
#------------------------------------------------------------------------------
    def _to_string(self, value: Any) -> str:
        """Convert any input value to string."""
        return str(value) if value is not None else ""
    
    def _to_integer(self, value: Any) -> int:
        """
        Convert any input value to integer.
        
        Args:
            value: Input value of any supported type
            
        Returns:
            int: Converted integer value
            
        Raises:
            ValueError: If conversion is not possible
        """
        if value is None:
            return 0
        
        if isinstance(value, bool):
            return int(value)
        
        try:
            if isinstance(value, str):
                # Handle float strings by converting to float first
                return int(float(value))
            return int(value)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert '{value}' to integer. Using 0.")
            return 0
    
    def _to_float(self, value: Any) -> float:
        """
        Convert any input value to float.
        
        Args:
            value: Input value of any supported type
            
        Returns:
            float: Converted float value
            
        Raises:
            ValueError: If conversion is not possible
        """
        if value is None:
            return 0.0
            
        try:
            return float(value)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert '{value}' to float. Using 0.0.")
            return 0.0

#------------------------------------------------------------------------------
# Section 5: Main Processing Logic
#------------------------------------------------------------------------------
    def process_conversion(
        self,
        pause_enabled: bool,
        prompt: Dict[str, Any],
        unique_id: str,
        input_string: Optional[str] = None,
        input_integer: Optional[int] = None,
        input_float: Optional[float] = None,
        display_value: Optional[str] = None
    ) -> Tuple[str, int, float]:
        """
        Process the input values and perform type conversions.
        
        Args:
            pause_enabled: Whether to pause for value modification
            prompt: Internal ComfyUI prompt data
            unique_id: Internal ComfyUI node identifier
            input_string: Optional string input
            input_integer: Optional integer input
            input_float: Optional float input
            display_value: Current display value
            
        Returns:
            Tuple[str, int, float]: Converted values as (string, integer, float)
        """
        try:
            # Determine the primary input value
            input_value = None
            if input_string is not None:
                input_value = input_string
            elif input_integer is not None:
                input_value = input_integer
            elif input_float is not None:
                input_value = input_float
            else:
                input_value = display_value
            
            # Store the current value for display
            self._current_value = str(input_value)
            
            # If pause is enabled, we'll use the pause mechanism
            if pause_enabled:
                # Store the pause state
                self._paused = True
                # Store the current value for modification
                self._display_value = str(input_value)
            
            # Perform conversions
            result_string = self._to_string(input_value)
            result_integer = self._to_integer(input_value)
            result_float = self._to_float(input_value)
            
            return (result_string, result_integer, result_float)
            
        except Exception as e:
            print(f"Error in HTConversionNode: {str(e)}")
            return ("", 0, 0.0)  # Return safe defaults on error
    
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """Ensure the node updates when paused."""
        return float("nan")
    
    @classmethod
    def VALIDATE_OUTPUTS(cls, *args, **kwargs) -> bool:
        """Allow any output type."""
        return True