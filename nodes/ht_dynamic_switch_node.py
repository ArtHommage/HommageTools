"""
File: homage_tools/nodes/ht_dynamic_switch_node.py
Version: 1.1.0
Description: Dynamic switch node with multiple inputs and single active output selection
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Configuration
#------------------------------------------------------------------------------
import torch
from typing import Dict, Any, Tuple, Union, Optional

# Version tracking
VERSION = "1.1.0"

#------------------------------------------------------------------------------
# Section 2: Type Handling Classes
#------------------------------------------------------------------------------
class AnyType(str):
    """A special class that is always equal in not equal comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False

# Define universal type
any_type = AnyType("*")

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTDynamicSwitchNode:
    """
    A dynamic switch node that allows selecting one of multiple inputs to output.
    Automatically creates additional input slots when existing ones are connected.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "doit"
    RETURN_TYPES = (any_type, "STRING", "INT")
    RETURN_NAMES = ("selected_value", "selected_label", "selected_index")
    OUTPUT_TOOLTIPS = (
        "Output is generated only from the input chosen by the 'select' value.", 
        "Slot label of the selected input slot", 
        "Outputs the select value as is"
    )

    #--------------------------------------------------------------------------
    # Section 4: Input/Output Specification
    #--------------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define input types with dynamic input slots."""
        # Dynamic input that creates additional slots when connected
        dyn_inputs = {
            "input1": (any_type, {
                "lazy": True, 
                "tooltip": "Any input. When connected, one more input slot is added."
            })
        }
        
        inputs = {
            "required": {
                "select": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 999999, 
                    "step": 1, 
                    "tooltip": "The input number you want to output among the inputs"
                }),
                "sel_mode": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "select_on_prompt", 
                    "label_off": "select_on_execution", 
                    "forceInput": False
                }),
            },
            "optional": dyn_inputs,
            "hidden": {
                "unique_id": "UNIQUE_ID", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
        return inputs

    #--------------------------------------------------------------------------
    # Section 5: Processing Logic
    #--------------------------------------------------------------------------
    def doit(self, *args, **kwargs) -> Tuple[Any, str, int]:
        """
        Process the dynamic switch selection.
        
        Args:
            **kwargs: Dictionary containing all inputs including dynamic ones
            
        Returns:
            Tuple[Any, str, int]: Selected value, selected label, and selected index
        """
        # Get selection index
        selected_index = int(kwargs['select'])
        input_name = f"input{selected_index}"
        
        # Default label
        selected_label = input_name
        
        # Return the selected input or None if invalid
        if input_name in kwargs:
            return kwargs[input_name], selected_label, selected_index
        else:
            print(f"HTDynamicSwitchNode v{VERSION}: Invalid select index (ignored)")
            return None, "", selected_index