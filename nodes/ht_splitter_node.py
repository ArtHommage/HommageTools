"""
File: homage_tools/nodes/ht_splitter_node.py
Version: 1.0.0
Description: Node for splitting a single input into two conditional outputs

Sections:
1. Imports and Type Definitions
2. Type Handling Classes
3. Node Class Definition
4. Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Dict, Any, Tuple, Optional
import torch

#------------------------------------------------------------------------------
# Section 2: Type Handling Classes
#------------------------------------------------------------------------------
class AnyType(str):
    """A special class that is always equal in not equal comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False

# Define universal type for flexible input handling
any_type = AnyType("*")

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTSplitterNode:
    """
    Routes a single input to two possible outputs based on a boolean condition.
    Supports any input type and maintains proper tensor handling.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "split_input"
    RETURN_TYPES = (any_type, any_type)
    RETURN_NAMES = ("output_true", "output_false")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "input_value": (any_type, {}),
                "condition": ("BOOLEAN", {
                    "default": True
                })
            }
        }

#------------------------------------------------------------------------------
# Section 4: Processing Logic
#------------------------------------------------------------------------------
    def split_input(self, input_value: Any, condition: bool) -> Tuple[Any, Any]:
        """
        Process input and route to appropriate output based on condition.
        
        Args:
            input_value: Input of any type to be routed
            condition: Boolean determining which output receives the value
            
        Returns:
            Tuple[Any, Any]: Tuple containing (true_output, false_output)
        """
        try:
            # For tensor inputs, ensure proper BHWC handling
            if isinstance(input_value, torch.Tensor):
                # If single image tensor, add batch dimension
                if len(input_value.shape) == 3:  # HWC format
                    input_value = input_value.unsqueeze(0)
                    
            # Route input based on condition
            if condition:
                return (input_value, None)
            else:
                return (None, input_value)
                
        except Exception as e:
            print(f"Error in HTSplitterNode: {str(e)}")
            return (None, None)