"""
File: homage_tools/nodes/ht_dynamic_switch_node.py
Version: 1.2.0
Description: Dynamic switch node that automatically adds new inputs when existing ones are connected
"""

import inspect
from typing import Dict, Any, Tuple, List

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
# Section 2: Dynamic Input Container
#------------------------------------------------------------------------------
class AllInputContainer:
    """Container that dynamically provides inputs with any type."""
    def __contains__(self, item):
        """Always return True for any key check."""
        return True
        
    def __getitem__(self, key):
        """Return any_type configuration for any requested key."""
        return any_type, {"tooltip": "Any input. When connected, another input slot is added."}

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTDynamicSwitchNode:
    """
    Dynamic switch node that automatically adds inputs as existing ones are connected.
    Passes through the first non-null input (by index order) and reports its index.
    Inputs are evaluated in order from top to bottom, with the first non-null value being passed through.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_inputs"
    RETURN_TYPES = (any_type, "INT")
    RETURN_NAMES = ("value", "index")
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input types with dynamic 'any' inputs."""
        # Initial input configuration
        dyn_inputs = {"input_1": (any_type, {"tooltip": "Any input. When connected, another input slot is added."})}
        
        # Check if we're being called during validation
        stack = inspect.stack()
        if len(stack) > 2 and stack[2].function == 'get_input_info':
            # During validation, use the container that provides unlimited inputs
            dyn_inputs = AllInputContainer()
            
        return {
            "required": {},
            "optional": dyn_inputs,
        }
    
    #--------------------------------------------------------------------------
    # Section 4: Input Processing Logic
    #--------------------------------------------------------------------------
    def process_inputs(self, **kwargs):
        """
        Process inputs and return the first non-null input by index order.
        
        Args:
            kwargs: All inputs passed to the node
            
        Returns:
            Tuple[Any, int]: The first non-null input value and its index
        """
        # Default values
        selected_value = None
        selected_index = -1
        
        # Find all connected inputs with values
        input_values = {}
        for key, value in kwargs.items():
            if key.startswith("input_") and value is not None:
                # Extract the index from the input name
                try:
                    idx = int(key.split('_')[1])
                    input_values[idx] = value
                except (ValueError, IndexError):
                    continue
        
        # Sort inputs by index and select the first one with a value
        if input_values:
            sorted_indices = sorted(input_values.keys())
            selected_index = sorted_indices[0]
            selected_value = input_values[selected_index]
        
        return selected_value, selected_index