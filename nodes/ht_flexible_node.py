"""
File: homage_tools/nodes/ht_flexible_node.py
Version: 1.0.0
Description: A flexible node that can handle any input/output type

Sections:
1. Type Handling Classes
2. Node Definition
3. Processing Logic
4. Error Handling
"""

#------------------------------------------------------------------------------
# Section 1: Type Handling Classes
#------------------------------------------------------------------------------
class AnyType(str):
    """A special class that is always equal in not equal comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False

class FlexibleInputType(dict):
    """Enables flexible/dynamic input types."""
    def __init__(self, type_value):
        self.type = type_value

    def __getitem__(self, key):
        return (self.type,)

    def __contains__(self, key):
        return True

# Define universal type
any_type = AnyType("*")

#------------------------------------------------------------------------------
# Section 2: Node Definition
#------------------------------------------------------------------------------
class HTFlexibleNode:
    """
    A flexible node that can handle any input/output type.
    Automatically adapts to connected nodes' types.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process"
    
    # Dynamic output typing
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ('output',)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input": (any_type, {}),
                "fallback_type": ("STRING", {
                    "default": "STRING",
                    "multiline": False
                })
            }
        }

    #--------------------------------------------------------------------------
    # Section 3: Processing Logic
    #--------------------------------------------------------------------------
    def process(self, fallback_type="STRING", input=None):
        """
        Process any input type and return it with proper typing.
        
        Args:
            fallback_type: Type to use if no input is provided
            input: The input value of any type
            
        Returns:
            Tuple containing the processed value
        """
        if input is not None:
            # Pass through the input value
            return (input,)
        
        # Return empty value of fallback type if no input
        if fallback_type == "STRING":
            return ("",)
        elif fallback_type == "INT":
            return (0,)
        elif fallback_type == "FLOAT":
            return (0.0,)
        else:
            return (None,)

    #--------------------------------------------------------------------------
    # Section 4: Error Handling
    #--------------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Ensure node updates on every execution."""
        return float("nan")