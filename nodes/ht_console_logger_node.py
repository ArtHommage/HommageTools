"""
File: homage_tools/nodes/ht_console_logger_node.py
Version: 1.0.2
Description: Console logging node with input passthrough and timestamp options
"""

import time
from typing import Dict, Any, Tuple, Optional, Union

# Define version
VERSION = "1.0.2"

#------------------------------------------------------------------------------
# Section 1: Type Handling Classes
#------------------------------------------------------------------------------
class AnyType(str):
    """A special class that is always equal in not equal comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False

# Define universal type
any_type = AnyType("*")

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTConsoleLoggerNode:
    """
    Prints custom messages to console with optional timestamp and passthrough.
    """
    
    CATEGORY = "HommageTools/Utility"
    FUNCTION = "log_message"
    RETURN_TYPES = (any_type,)  # Changed from ("*",) to (any_type,)
    RETURN_NAMES = ("passthrough",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "message": ("STRING", {
                    "multiline": True,
                    "default": "Log message"
                }),
                "include_timestamp": ("BOOLEAN", {
                    "default": True
                })
            },
            "optional": {
                "input": (any_type, {})  # Changed from ("*", {}) to (any_type, {})
            }
        }
    
    #--------------------------------------------------------------------------
    # Section 3: Helper Methods
    #--------------------------------------------------------------------------
    def _format_message(self, message: str, include_timestamp: bool) -> str:
        """Format the message with an optional timestamp."""
        prefix = ""
        if include_timestamp:
            timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
            prefix = f"{timestamp} "
        
        return f"{prefix}{message}"
    
    #--------------------------------------------------------------------------
    # Section 4: Main Processing Logic
    #--------------------------------------------------------------------------
    def log_message(
        self, 
        message: str, 
        include_timestamp: bool = True,
        input: Any = None
    ) -> Tuple[Any]:
        """
        Log a message to the console and pass through the input.
        
        Args:
            message: Message to log
            include_timestamp: Whether to include timestamp
            input: Optional input to pass through
            
        Returns:
            Tuple containing the input that was passed through
        """
        # Format and print the message
        formatted_message = self._format_message(message, include_timestamp)
        print(f"[HT_LOGGER] {formatted_message}")
        
        # Return the input as passthrough
        return (input,)