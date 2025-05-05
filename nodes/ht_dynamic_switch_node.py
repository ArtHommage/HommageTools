"""
File: homage_tools/nodes/ht_dynamic_switch_node.py
Version: 1.1.0
Description: Enhanced switch node that triggers once when activated

Sections:
1. Imports and Type Definitions
2. Node Class Definition 
3. State Tracking Logic
4. Execution Control
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Tuple, Dict, Any

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTDynamicSwitchNode:
    """
    Enhanced switch node that triggers only once when activated.
    Maintains state between executions to prevent repeated triggering.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_switch"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("trigger",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define the input types and their default values."""
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": False,
                    "description": "Enable/disable the switch"
                })
            }
        }
    
    #--------------------------------------------------------------------------
    # Section 3: State Tracking Logic
    #--------------------------------------------------------------------------
    def __init__(self):
        """Initialize with default untriggered state."""
        self._triggered = False
    
    def process_switch(self, enabled: bool) -> Tuple[bool]:
        """
        Process switch state with single-trigger logic.
        
        Args:
            enabled: Whether the switch is enabled
            
        Returns:
            Tuple[bool]: Tuple containing trigger state
        """
        # Only trigger once when enabled until disabled again
        if enabled and not self._triggered:
            self._triggered = True
            return (True,)
            
        # Reset trigger state when disabled
        if not enabled:
            self._triggered = False
            
        return (False,)

    #--------------------------------------------------------------------------
    # Section 4: Execution Control
    #--------------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Control when the node updates.
        Returns NaN to ensure the node updates on every execution.
        """
        return float("nan")