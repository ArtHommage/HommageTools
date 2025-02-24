"""
File: homage_tools/nodes/ht_status_indicator_node.py
Version: 1.0.0
Description: Node for displaying status indicators based on input values
"""

from typing import Dict, Any, Tuple
import torch
from server import PromptServer

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
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def evaluate_status(value: Any) -> bool:
    """
    Evaluate input value to determine status.
    
    Args:
        value: Input of any type to evaluate
        
    Returns:
        bool: True for positive status, False otherwise
    """
    if value is None:
        return False
    elif isinstance(value, bool):
        return value
    elif isinstance(value, int):
        return value == 1
    elif isinstance(value, float):
        return value > 0
    elif isinstance(value, str):
        return bool(value.strip())
    elif isinstance(value, torch.Tensor):
        # Handle tensor inputs - check if any positive values
        return bool(torch.any(value > 0))
    return False

#------------------------------------------------------------------------------
# Section 3: Node Definition
#------------------------------------------------------------------------------
class HTStatusIndicatorNode:
    """
    Displays a colored status indicator based on input value.
    Green dot for positive values, red for negative.
    """
    
    CATEGORY = "HommageTools/Debug"
    FUNCTION = "process_status"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("passthrough",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input": (any_type, {})
            }
        }

    #--------------------------------------------------------------------------
    # Section 4: Status Processing Logic
    #--------------------------------------------------------------------------
    def process_status(self, input=None) -> Tuple[Any]:
        """
        Process input and update status indicator.
        
        Args:
            input: Any input value to evaluate
            
        Returns:
            Tuple[Any]: Original input value passed through
        """
        # Evaluate status
        status = evaluate_status(input)
        
        # Send status update to UI
        PromptServer.instance.send_sync("status-indicator", {
            "node_id": self.id,  # Node instance will have ID set by ComfyUI
            "status": status
        })
        
        # Pass through the original input
        return (input,)
        
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """Ensure node updates on every execution."""
        return float("nan")