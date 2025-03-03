"""
File: homage_tools/nodes/ht_console_logger_node.py
Version: 1.0.0
Description: Node for printing custom messages to console with input passthrough

Sections:
1. Type Handling Classes
2. Node Definition
3. Processing Logic
4. Console Output Formatting
"""

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
# Section 2: Node Definition
#------------------------------------------------------------------------------
class HTConsoleLoggerNode:
    """
    Prints custom messages to the console with optional input passthrough.
    Useful for debugging, workflow monitoring, and progress tracking.
    """
    
    CATEGORY = "HommageTools/Debug"
    FUNCTION = "log_message"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("passthrough",)
    
    # Version tracking
    VERSION = "1.0.0"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": ("STRING", {
                    "multiline": True,
                    "default": "Debug message",
                    "placeholder": "Enter message to print to console"
                }),
                "include_timestamp": ("BOOLEAN", {
                    "default": True
                })
            },
            "optional": {
                "input": (any_type, {})
            }
        }

#------------------------------------------------------------------------------
# Section 3: Processing Logic
#------------------------------------------------------------------------------
    def log_message(
        self,
        message: str,
        include_timestamp: bool,
        input: any = None
    ) -> tuple:
        """
        Process the logging request and print to console.
        
        Args:
            message: The message to print to console
            include_timestamp: Whether to include timestamp in output
            input_value: Optional input value to pass through
            
        Returns:
            tuple: Tuple containing the input value passed through
        """
        formatted_message = self._format_message(message, include_timestamp, input_value)
        print(formatted_message)
        
        # Pass through the input value
        return (input,)

#------------------------------------------------------------------------------
# Section 4: Console Output Formatting
#------------------------------------------------------------------------------
    def _format_message(
        self,
        message: str,
        include_timestamp: bool,
        input: any
    ) -> str:
        """
        Format the console output message with optional components.
        
        Args:
            message: The base message to format
            include_timestamp: Whether to include timestamp
            input_value: The input value for type information
            
        Returns:
            str: Formatted message for console output
        """
        from datetime import datetime
        
        # Start with header
        result = [f"\n======== HTConsoleLogger v{self.VERSION} ========"]
        
        # Add timestamp if requested
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result.append(f"[{timestamp}]")
        
        # Add message
        result.append(message)
        
        # Add input type information if available
        if input is not None:
            input_type = type(input).__name__
            if hasattr(input, 'shape'):
                input_type += f" shape={input.shape}"
            result.append(f"Input: {input_type}")
        
        # Add footer
        result.append("=" * 40)
        
        return "\n".join(result)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """Ensure node updates on every execution."""
        return float("nan")