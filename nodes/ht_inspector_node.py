"""
File: homage_tools/nodes/ht_inspector_node.py
Version: 1.0.0
Description: Node for inspecting and reporting input types and values
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
# Section 2: Inspector Node Definition
#------------------------------------------------------------------------------
class HTInspectorNode:
    """
    Diagnostic node that reports information about its input.
    Accepts any input type and reports details about what it received.
    """
    
    CATEGORY = "HommageTools/Debug"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "inspect_input"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input": (any_type, {})
            }
        }

    #--------------------------------------------------------------------------
    # Section 3: Type Inspection Methods
    #--------------------------------------------------------------------------
    def _inspect_tensor(self, tensor: torch.Tensor) -> str:
        """Generate detailed info about a tensor."""
        return f"Tensor[shape={tensor.shape}, dtype={tensor.dtype}]"

    def _inspect_dict(self, d: dict) -> str:
        """Generate detailed info about a dictionary."""
        return f"Dict[keys={list(d.keys())}]"

    def _get_type_info(self, value: Any) -> str:
        """Get detailed type information for any value."""
        if value is None:
            return "None"
        elif isinstance(value, torch.Tensor):
            return self._inspect_tensor(value)
        elif isinstance(value, dict):
            return self._inspect_dict(value)
        elif isinstance(value, (list, tuple)):
            return f"{type(value).__name__}[len={len(value)}]"
        else:
            return f"{type(value).__name__}[value={str(value)}]"

    #--------------------------------------------------------------------------
    # Section 4: Main Processing Logic
    #--------------------------------------------------------------------------
    def inspect_input(self, input=None) -> Tuple[Any]:
        """
        Inspect input and report details while passing through the value.
        
        Args:
            input: Any input value to inspect
            
        Returns:
            Tuple[Any]: Original input value passed through
        """
        # Get detailed type information
        type_info = self._get_type_info(input)
        
        # Send information to UI
        PromptServer.instance.send_sync("notification", {
            "type": "info",
            "message": f"Input Inspection:\n{type_info}",
            "timeout": 5000
        })
        
        print(f"\nHTInspector Debug Info:")
        print(f"Type: {type(input)}")
        print(f"Detailed Info: {type_info}")
        
        # Pass through the original input
        return (input,)