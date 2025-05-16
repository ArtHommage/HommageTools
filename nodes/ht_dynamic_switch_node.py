"""
File: homage_tools/nodes/ht_dynamic_switch_node.py
Version: 1.0.0
Description: Node that outputs the first valid input among multiple optional inputs
"""

#------------------------------------------------------------------------------
# Section 1: Node Class Definition
#------------------------------------------------------------------------------
class HTDynamicSwitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }
    
    # Add an INT return type for the index
    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("value", "used_input")
    FUNCTION = "switch"
    CATEGORY = "HommageTools"
    
    #--------------------------------------------------------------------------
    # Section 2: Main Processing Logic
    #--------------------------------------------------------------------------
    def switch(self, **kwargs):
        # Check each input in numerical order
        for i in range(1, 1000):  # Set a reasonable upper limit
            input_name = f"input{i}"
            if input_name not in kwargs:
                break  # Stop when we run out of inputs
            
            # If this input is not None and not empty, return it
            if kwargs[input_name] is not None:
                # For simple types like strings, check if they're empty
                if isinstance(kwargs[input_name], str) and kwargs[input_name] == "":
                    continue
                    
                # For list-like objects, check if they're empty
                if hasattr(kwargs[input_name], "__len__") and len(kwargs[input_name]) == 0:
                    continue
                    
                # Return both the value and which input was used
                return (kwargs[input_name], i)
        
        # If we get here, all inputs were null or empty
        raise ValueError("No valid input found. All inputs are null or empty.")